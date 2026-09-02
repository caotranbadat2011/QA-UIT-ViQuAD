"""
Cham coi cuoi: ap dung reranker da huan luyen len test, so sanh voi hanh vi cu.

Usage:
    python src/evaluate_reranked.py
"""
import argparse
import collections
import json
import os

import joblib
import numpy as np

from evaluate import compute_exact, compute_f1
from reranker_features import build_matrix, load_records
from train_reranker import choose, fmt, metrics, predict_grouped


def categorize(gold, impossible, pred):
    if impossible:
        return "unans_dung" if pred == "" else "unans_bua"
    if pred == "":
        return "ans_tu_choi_oan"
    if compute_exact(gold, pred):
        return "chinh_xac"
    if compute_f1(gold, pred) == 0:
        return "sach_hoan_toan"
    g, p = gold.lower().split(), pred.lower().split()
    if len(p) > len(g):
        return "bien_thua"
    if len(p) < len(g):
        return "bien_thieu"
    return "overlap_khac"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default="data/candidates/test.jsonl")
    parser.add_argument("--model_dir", default="models/reranker")
    parser.add_argument("--out", default="data/results/rerank_report.json")
    parser.add_argument("--tau", type=float, default=None, help="Ghi de tau da chon tren val")
    parser.add_argument("--n_examples", type=int, default=12)
    args = parser.parse_args()

    clf = joblib.load(os.path.join(args.model_dir, "reranker.pkl"))
    with open(os.path.join(args.model_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    prior, tau = meta["prior"], (args.tau if args.tau is not None else meta["tau"])
    print(f"Reranker={meta['model']}  tau={tau:.3f}  (chon tren val)")

    recs = load_records(args.test_file)
    qmap = {r["id"]: r["question"] for r in recs}
    grouped = predict_grouped(clf, recs, prior)

    base_rows, new_rows, dist_before, dist_after = [], [], collections.Counter(), collections.Counter()
    fixed, broken = [], []
    for proba, m in grouped:
        pred = choose(proba, m["texts"], tau)
        old = m["rule_pick"]
        base_rows.append((m["gold"], m["is_impossible"], old))
        new_rows.append((m["gold"], m["is_impossible"], pred))
        dist_before[categorize(m["gold"], m["is_impossible"], old)] += 1
        dist_after[categorize(m["gold"], m["is_impossible"], pred)] += 1
        gain_old = compute_f1(m["gold"], old) if not m["is_impossible"] else float(old == "")
        gain_new = compute_f1(m["gold"], pred) if not m["is_impossible"] else float(pred == "")
        item = {"id": m["id"], "gold": m["gold"], "before": old, "after": pred,
                "confidence": round(float(proba.max()), 3),
                "question": qmap[m["id"]]}
        if gain_new > gain_old + 1e-6:
            fixed.append(item)
        elif gain_new < gain_old - 1e-6:
            broken.append(item)

    mb, ma = metrics(base_rows), metrics(new_rows)
    print(f"\n===== KET QA TEST (n={mb['n']}) =====")
    print(f"  Baseline (evaluate.py) : {fmt(mb)}")
    print(f"  + Reranker             : {fmt(ma)}")
    print("  ------------------------------")
    for k in ("em", "f1", "hasans_em", "hasans_f1", "noans_acc"):
        if k in mb and k in ma:
            print(f"    {k:10s} {mb[k]:6.2f} -> {ma[k]:6.2f}   ({ma[k] - mb[k]:+.2f})")

    print("\nPhan bo loi:")
    for k in sorted(set(dist_before) | set(dist_after), key=lambda x: -dist_before.get(x, 0)):
        print(f"  {k:18s} {dist_before.get(k, 0):5d} -> {dist_after.get(k, 0):5d}")

    print(f"\nCai dc {len(fixed)} cau, lam hong {len(broken)} cau")
    fixed.sort(key=lambda d: -d["confidence"])
    broken.sort(key=lambda d: -d["confidence"])
    print("\nVi du cai dc:")
    for d in fixed[: args.n_examples // 2]:
        print(f"  [{d['confidence']}] Hoi: {d['question'][:60]}")
        print(f"      cu: {d['before'][:70]!r}")
        print(f"      moi: {d['after'][:70]!r}  chuan: {d['gold'][:70]!r}")
    print("\nVi du lam hong:")
    for d in broken[: args.n_examples // 2]:
        print(f"  [{d['confidence']}] Hoi: {d['question'][:60]}")
        print(f"      cu: {d['before'][:70]!r}")
        print(f"      moi: {d['after'][:70]!r}  chuan: {d['gold'][:70]!r}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"baseline": mb, "reranked": ma, "tau": tau,
                   "error_dist_before": dict(dist_before), "error_dist_after": dict(dist_after),
                   "n_fixed": len(fixed), "n_broken": len(broken),
                   "fixed_examples": fixed[:40], "broken_examples": broken[:40]},
                  f, ensure_ascii=False, indent=2)
    print(f"\nLuu bao cao: {args.out}")

    preds = {m["id"]: choose(p, m["texts"], tau) for p, m in grouped}
    with open("predictions_reranked.json", "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    print("Luu predictions_reranked.json")


if __name__ == "__main__":
    main()
