"""
Trich pool ung vien dap an (candidate pool) tu model PhoBERT QA da train.

Model KHONG duoc train lai. Vai tro o day la chay suy dien mot luot va luu lai
K span co diem cao nhat cung logit cua chung, de src/train_reranker.py hoc cach
chon dung span trong so do (ky thuat Extractor -> Reranker).

Usage:
    python src/dump_candidates.py --split test
    python src/dump_candidates.py --split train --batch_size 8
"""
import argparse
import collections
import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from evaluate import QADataLoader, normalize_text, prepare_eval_features


def load_split(split, processed_dir="data/processed"):
    df = pd.read_parquet(os.path.join(processed_dir, f"{split}.parquet"))
    examples = []
    for _, row in df.iterrows():
        impossible = bool(row["is_impossible"]) or int(row["answer_start"]) == -1
        examples.append({
            "id": str(row["qa_id"]),
            "question": str(row["question"]).strip(),
            "context": str(row["context"]),
            "gold": "" if impossible else str(row["answer_text"]),
            "gold_start": -1 if impossible else int(row["answer_start"]),
            "is_impossible": impossible,
        })
    return examples


def window_candidates(start_logits, end_logits, offset_mapping, n_best, max_answer_len):
    """Sinh moi cap (start, end) hop le trong mot cua truot kem diem cua no."""
    out = []
    top_starts = np.argsort(start_logits)[-n_best:][::-1]
    top_ends = np.argsort(end_logits)[-n_best:][::-1]
    for s in top_starts:
        for e in top_ends:
            if e < s or e - s + 1 > max_answer_len:
                continue
            off_s, off_e = offset_mapping[s], offset_mapping[e]
            if off_s is None or off_e is None:
                continue
            cs, ce = off_s[0], off_e[1]
            if ce <= cs:
                continue
            out.append((float(start_logits[s] + end_logits[e]), int(s), int(e), int(cs), int(ce)))
    return out


def dump_split(examples, tokenizer, model, device, args):
    feats = prepare_eval_features(
        {k: [ex[k] for ex in examples] for k in ("id", "question", "context")},
        tokenizer, args.max_length, args.doc_stride,
    )
    per_example = collections.defaultdict(list)
    for i, ex_id in enumerate(feats["example_id"]):
        per_example[ex_id].append(i)

    input_ids = torch.tensor(feats["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(feats["attention_mask"], dtype=torch.long)

    all_start, all_end = [], []
    with torch.no_grad():
        for i in range(0, len(input_ids), args.batch_size):
            batch = {
                "input_ids": input_ids[i:i + args.batch_size].to(device),
                "attention_mask": attention_mask[i:i + args.batch_size].to(device),
            }
            out = model(**batch)
            all_start.append(out.start_logits.float().cpu().numpy())
            all_end.append(out.end_logits.float().cpu().numpy())
            if (i // args.batch_size) % 100 == 0:
                print(f"  inference {i + len(batch['input_ids'])}/{len(input_ids)} windows", flush=True)
    all_start = np.concatenate(all_start, axis=0)
    all_end = np.concatenate(all_end, axis=0)

    records = []
    for ex in examples:
        indices = per_example[ex["id"]]
        pool = []
        null_scores = []
        for w, fi in enumerate(indices):
            null_scores.append(float(all_start[fi][0] + all_end[fi][0]))
            for score, s, e, cs, ce in window_candidates(
                all_start[fi], all_end[fi], feats["offset_mapping"][fi],
                args.n_best, args.max_answer_len,
            ):
                pool.append({"score": score, "s": s, "e": e, "cs": cs, "ce": ce, "win": w})

        pool.sort(key=lambda c: -c["score"])
        pool = pool[: args.top_k]

        context = ex["context"]
        seen = set()
        candidates = []
        for rank, c in enumerate(pool):
            key = (c["cs"], c["ce"])
            if key in seen:
                continue
            seen.add(key)
            text = context[c["cs"]:c["ce"]]
            candidates.append({
                "rk": rank, "win": c["win"], "s": c["s"], "e": c["e"],
                "cs": c["cs"], "ce": c["ce"], "score": round(c["score"], 3),
                "n_tok": c["e"] - c["s"] + 1, "text": text,
            })

        best = candidates[0]["text"] if candidates else ""
        records.append({
            "id": ex["id"],
            "question": ex["question"],
            "n_windows": len(indices),
            "null_scores": [round(v, 3) for v in null_scores],
            "extractor_pick": best,
            "gold": ex["gold"],
            "is_impossible": ex["is_impossible"],
            "gold_hit_in_pool": bool(
                ex["is_impossible"] is False
                and any(normalize_text(c["text"]) == normalize_text(ex["gold"]) for c in candidates)
            ),
            "candidates": candidates,
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--model_dir", default="models/phobert_qa")
    parser.add_argument("--out_dir", default="data/candidates")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--doc_stride", type=int, default=64)
    parser.add_argument("--n_best", type=int, default=20)
    parser.add_argument("--max_answer_len", type=int, default=24)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Chi chay N dau de thu nghiem")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir)
    if device == "cuda":
        model.half()
    model.to(device).eval()

    examples = load_split(args.split)
    if args.limit:
        examples = examples[: args.limit]
    print(f"{args.split}: {len(examples)} examples")

    records = dump_split(examples, tokenizer, model, device, args)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.split}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    answerable = [r for r in records if not r["is_impossible"]]
    recall = np.mean([r["gold_hit_in_pool"] for r in answerable]) if answerable else 0.0
    base_em = np.mean([
        int(normalize_text(r["gold"]) == normalize_text(r["extractor_pick"])) for r in answerable
    ]) if answerable else 0.0
    no_ans_acc = np.mean([r["extractor_pick"] == "" for r in records if r["is_impossible"]]) \
        if any(r["is_impossible"] for r in records) else 0.0

    print(f"\nSaved {out_path}")
    print(f"recall@{args.top_k} (dap an co trong pool): {100 * recall:.2f}%")
    print(f"extractor pick  -> HasAns EM {100 * base_em:.2f}%  NoAns acc {100 * no_ans_acc:.2f}%")
    print("Nho: extractor_pick day chua ap dung nguong null cua evaluate.py, chi la span diem cao nhat.")


if __name__ == "__main__":
    main()
