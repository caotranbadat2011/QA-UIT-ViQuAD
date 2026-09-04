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
    """Lớp đóng gói logic Tokenization và căn chỉnh offset (Mapping offsets)"""
    def __init__(self, tokenizer, max_length: int = 384, doc_stride: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride

    def prepare_features(self, examples):
        tokenized = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_map = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        start_positions, end_positions = [], []
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id) if self.tokenizer.cls_token_id in input_ids else 0
            sample_idx = sample_map[i]
            answers = examples["answers"][sample_idx]

            if len(answers["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            sequence_ids = tokenized.sequence_ids(i)

            idx = 0
            while idx < len(sequence_ids) and sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                idx = context_start
                while idx <= context_end and offsets[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offsets[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        return tokenized


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
            logging_steps=50,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_features,
            eval_dataset=val_features,
            data_collator=default_data_collator,
            tokenizer=self.tokenizer,
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
    parser.add_argument("--model_name", default="xlm-roberta-base")
    parser.add_argument("--output_dir", default="./models/best_checkpoint")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = QATrainerPipeline(args)
    pipeline.train()