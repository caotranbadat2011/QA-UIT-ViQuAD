"""
Evaluate a fine-tuned QA model on ViQuAD test set with SQuAD-style metrics: Exact Match (EM), F1.
Usage:
    python evaluate.py --model_dir models/best_checkpoint --test_file data/processed/test.parquet
"""
import argparse
import collections
import json
import re
import string

import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import Dataset


class QADataLoader:
    """Lớp đọc dữ liệu tương thích với train.py"""
    @staticmethod
    def load_dataset(file_path: str) -> Dataset:
        if str(file_path).endswith(".parquet"):
            df = pd.read_parquet(file_path)

            def format_answers(row):
                if row.get("is_impossible", False) or row.get("answer_start", -1) == -1:
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


def normalize_text(s):
    """Chuẩn hóa văn bản Tiếng Việt cho tính toán EM và F1"""
    s = s.lower()
    # Xóa dấu câu
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    # Xóa khoảng trắng thừa
    return " ".join(s.split())


def compute_exact(gold, pred):
    return int(normalize_text(gold) == normalize_text(pred))


def compute_f1(gold, pred):
    gold_toks = normalize_text(gold).split()
    pred_toks = normalize_text(pred).split()
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def prepare_eval_features(examples, tokenizer, max_length, doc_stride):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = tokenized.pop("overflow_to_sample_mapping")
    example_ids = []
    for i in range(len(tokenized["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])
        sequence_ids = tokenized.sequence_ids(i)
        tokenized["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None
            for k, o in enumerate(tokenized["offset_mapping"][i])
        ]
    tokenized["example_id"] = example_ids
    return tokenized


def postprocess(examples, features, raw_predictions, n_best=20, max_answer_length=64):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {ex["id"]: i for i, ex in enumerate(examples)}
    features_per_example = collections.defaultdict(list)
    for i, feat_id in enumerate(features["example_id"]):
        features_per_example[example_id_to_index[feat_id]].append(i)

    predictions = {}
    for ex_index, example in enumerate(examples):
        context = example["context"]
        best_answer = ""
        best_score = -1e9
        min_null_score = None

        for feat_index in features_per_example[ex_index]:
            start_logits = all_start_logits[feat_index]
            end_logits = all_end_logits[feat_index]
            offset_mapping = features["offset_mapping"][feat_index]

            cls_index = 0
            null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or null_score < min_null_score:
                min_null_score = null_score

            start_idx = np.argsort(start_logits)[-n_best:][::-1]
            end_idx = np.argsort(end_logits)[-n_best:][::-1]
            for s in start_idx:
                for e in end_idx:
                    if s >= len(offset_mapping) or e >= len(offset_mapping):
                        continue
                    if offset_mapping[s] is None or offset_mapping[e] is None:
                        continue
                    if e < s or e - s + 1 > max_answer_length:
                        continue
                    score = start_logits[s] + end_logits[e]
                    if score > best_score:
                        best_score = score
                        best_answer = context[offset_mapping[s][0]: offset_mapping[e][1]]

        if min_null_score is not None and min_null_score > best_score:
            predictions[example["id"]] = ""
        else:
            predictions[example["id"]] = best_answer

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/best_checkpoint")
    parser.add_argument("--test_file", default="data/processed/test.parquet")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    args = parser.parse_args()

    print(f"Loading Tokenizer & Model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"Loading Test Dataset from {args.test_file}...")
    test_ds = QADataLoader.load_dataset(args.test_file)
    examples = list(test_ds)

    print("Preprocessing Test Features...")
    features = test_ds.map(
        lambda ex: prepare_eval_features(ex, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=test_ds.column_names,
    )

    print("Running Inference...")
    all_start_logits, all_end_logits = [], []
    batch_size = 16
    cols = features.remove_columns(["example_id", "offset_mapping"])
    cols.set_format(type="torch")
    for i in range(0, len(cols), batch_size):
        batch = cols[i:i + batch_size]
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
        all_start_logits.append(out.start_logits.cpu().numpy())
        all_end_logits.append(out.end_logits.cpu().numpy())
    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    print("Post-processing Predictions...")
    predictions = postprocess(examples, features, (all_start_logits, all_end_logits))

    em_scores, f1_scores = [], []
    has_ans_em, has_ans_f1 = [], []
    no_ans_correct, no_ans_total = 0, 0

    for ex in examples:
        gold_texts = ex["answers"]["text"]
        pred = predictions[ex["id"]]
        is_impossible = len(gold_texts) == 0

        if is_impossible:
            no_ans_total += 1
            em = compute_exact("", pred)
            f1 = em
            if pred == "":
                no_ans_correct += 1
        else:
            em = max(compute_exact(g, pred) for g in gold_texts)
            f1 = max(compute_f1(g, pred) for g in gold_texts)
            has_ans_em.append(em)
            has_ans_f1.append(f1)

        em_scores.append(em)
        f1_scores.append(f1)

    print("\n===== EVALUATION RESULTS =====")
    print(f"Num examples: {len(examples)}")
    print(f"Overall EM: {100 * np.mean(em_scores):.2f}%")
    print(f"Overall F1: {100 * np.mean(f1_scores):.2f}%")
    if has_ans_em:
        print(f"HasAns EM: {100 * np.mean(has_ans_em):.2f}%")
        print(f"HasAns F1: {100 * np.mean(has_ans_f1):.2f}%")
    if no_ans_total:
        print(f"NoAns Accuracy: {100 * no_ans_correct / no_ans_total:.2f}%")

    with open("predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print("Predictions saved to predictions.json")


if __name__ == "__main__":
    main()