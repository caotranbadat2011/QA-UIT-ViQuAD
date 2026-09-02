"""
OOP Architecture for Fine-tuning Transformer Model on Vietnamese Extractive Question Answering.
"""
import argparse
import json
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)


class QADataLoader:
    """Lớp chịu trách nhiệm đọc và chuẩn hóa định dạng dữ liệu (JSON / Parquet)"""
    @staticmethod
    def load_dataset(file_path: str) -> Dataset:
        if str(file_path).endswith(".parquet"):
            df = pd.read_parquet(file_path)

            def format_answers(row):
                if row["is_impossible"] or row["answer_start"] == -1:
                    return {"text": [], "answer_start": []}
                return {
                    "text": [str(row["answer_text"])], 
                    "answer_start": [int(row["answer_start"])]
                }

            df["answers"] = df.apply(format_answers, axis=1)
            df["id"] = df["qa_id"]
            df["question"] = df["question"].astype(str).str.strip()
            
            dataset_df = df[["id", "question", "context", "answers"]]
            return Dataset.from_pandas(dataset_df)

        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        rows = []
        for article in raw["data"]:
            for para in article["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:
                    answers = qa.get("answers", [])
                    rows.append({
                        "id": qa["id"],
                        "question": qa["question"].strip(),
                        "context": context,
                        "answers": {
                            "text": [a["text"] for a in answers],
                            "answer_start": [a["answer_start"] for a in answers],
                        },
                    })
        return Dataset.from_list(rows)


class QATokenizerProcessor:
    """Lớp đóng gói logic Tokenization và căn chỉnh offset (Mapping offsets).

    PhoBERT dùng tokenizer Python chậm (không hỗ trợ return_offsets_mapping),
    nên offset được tính thủ công từ output của tokenizer.
    """
    def __init__(self, tokenizer, max_length: int = 256, doc_stride: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride

    def _tokenize_with_offsets(self, text):
        tokens = self.tokenizer.tokenize(text)
        offsets = []
        pos = 0
        for tok in tokens:
            clean = tok.replace("@@", "")
            if clean == "":
                offsets.append((0, 0))
                continue
            start = text.find(clean, pos)
            if start == -1:
                start = pos
            end = start + len(clean)
            offsets.append((start, end))
            pos = end
        return tokens, offsets

    def prepare_features(self, examples):
        tokenizer = self.tokenizer
        max_length = self.max_length
        doc_stride = self.doc_stride
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id

        input_ids_list, attention_list = [], []
        start_positions, end_positions = [], []

        for i in range(len(examples["question"])):
            question = examples["question"][i]
            context = examples["context"][i]
            answers = examples["answers"][i]

            q_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(question))
            ctx_tokens, ctx_offsets = self._tokenize_with_offsets(context)
            ctx_ids = tokenizer.convert_tokens_to_ids(ctx_tokens)

            # Layout: [CLS] q [SEP] [SEP] ctx [SEP]  -> 4 special tokens
            max_ctx_len = max_length - len(q_ids) - 4
            if max_ctx_len < 1:
                q_ids = q_ids[: max(1, max_length - 5)]
                max_ctx_len = max_length - len(q_ids) - 4

            # Sliding windows over the context tokens
            spans = []
            start = 0
            while start < len(ctx_ids):
                length = min(max_ctx_len, len(ctx_ids) - start)
                spans.append((start, length))
                if start + length == len(ctx_ids):
                    break
                start += min(length, doc_stride)

            answer_start = answers["answer_start"][0] if len(answers["answer_start"]) > 0 else None

            for (s, l) in spans:
                ctx_slice = ctx_ids[s:s + l]
                ctx_off_slice = ctx_offsets[s:s + l]

                input_ids = [cls_id] + q_ids + [sep_id, sep_id] + ctx_slice + [sep_id]
                attention_mask = [1] * len(input_ids)

                # padding
                pad_len = max_length - len(input_ids)
                input_ids += [pad_id] * pad_len
                attention_mask += [0] * pad_len

                # offset_mapping: None for special tokens, char offsets for context tokens
                offset_mapping = [(0, 0)] * (len(q_ids) + 3)
                offset_mapping += [o for o in ctx_off_slice]
                offset_mapping += [(0, 0)]
                offset_mapping += [(0, 0)] * pad_len

                # Determine answer span token positions (if answer is fully inside this window)
                cls_index = 0
                if answer_start is None:
                    start_pos, end_pos = cls_index, cls_index
                else:
                    end_char = answer_start + len(answers["text"][0])
                    context_start = len(q_ids) + 3
                    context_end = context_start + l - 1
                    if l == 0 or not (
                        ctx_off_slice[0][0] <= answer_start and ctx_off_slice[-1][1] >= end_char
                    ):
                        start_pos, end_pos = cls_index, cls_index
                    else:
                        idx = context_start
                        while idx <= context_end and offset_mapping[idx][0] <= answer_start:
                            idx += 1
                        start_pos = idx - 1

                        idx = context_end
                        while idx >= context_start and offset_mapping[idx][1] >= end_char:
                            idx -= 1
                        end_pos = idx + 1

                input_ids_list.append(input_ids)
                attention_list.append(attention_mask)
                start_positions.append(start_pos)
                end_positions.append(end_pos)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_list,
            "start_positions": start_positions,
            "end_positions": end_positions,
        }


class QATrainerPipeline:
    """Lớp quản lý quá trình khởi tạo mô hình, tiền xử lý features và huấn luyện (Trainer)"""
    def __init__(self, args):
        self.args = args
        print(f"Loading Tokenizer & Model: {self.args.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.args.model_name)
        self.processor = QATokenizerProcessor(
            tokenizer=self.tokenizer,
            max_length=self.args.max_length,
            doc_stride=self.args.doc_stride
        )

    def train(self):
        print("Loading Train & Validation Datasets...")
        train_ds = QADataLoader.load_dataset(self.args.train_file)
        val_ds = QADataLoader.load_dataset(self.args.val_file)

        print("Preprocessing & Tokenizing Features...")
        train_features = train_ds.map(
            self.processor.prepare_features,
            batched=True,
            remove_columns=train_ds.column_names,
        )
        val_features = val_ds.map(
            self.processor.prepare_features,
            batched=True,
            remove_columns=val_ds.column_names,
        )

        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.args.learning_rate,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            num_train_epochs=self.args.num_train_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=True,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            gradient_checkpointing=self.args.gradient_checkpointing,
            warmup_ratio=self.args.warmup_ratio,
            seed=self.args.seed,
            save_total_limit=self.args.save_total_limit,
            report_to="none",
            logging_steps=50,
            max_steps=self.args.max_steps,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_features,
            eval_dataset=val_features,
            data_collator=default_data_collator,
            processing_class=self.tokenizer,
        )

        print("Starting Training Process...")
        trainer.train()

        # Save Checkpoint
        trainer.save_model(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)
        print(f"Model successfully saved to {self.args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extractive QA Fine-tuning Pipeline")
    parser.add_argument("--train_file", default="data/processed/train.parquet")
    parser.add_argument("--val_file", default="data/processed/val.parquet")
    parser.add_argument("--model_name", default="vinai/phobert-base")
    parser.add_argument("--output_dir", default="./models/best_checkpoint")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--doc_stride", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = QATrainerPipeline(args)
    pipeline.train()