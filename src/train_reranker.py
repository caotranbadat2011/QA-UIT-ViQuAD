"""
Huan luyen bo cham lai ung vien (reranker) tren pool da dump bang dump_candidates.py.

Day la ky thuat Extractor -> Reranker:
    PhoBERT QA (giu nguyen, khong train lai)  ->  pool 40 ung vien
    Reranker (hoc tren dac trung cua tung ung vien) -> chon 1, hoac tu choi

Usage:
    python src/train_reranker.py
"""
import argparse
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from evaluate import compute_exact, compute_f1
from reranker_features import FEATURE_NAMES, build_matrix, fit_prior, load_records


def fmt(m):
    return "  ".join(f"{k}={m[k]:.2f}" for k in ("em", "f1", "hasans_em", "hasans_f1", "noans_acc") if k in m)


def metrics(rows):
    em, f1, ha_em, ha_f1, na_c, na_t = [], [], [], [], 0, 0
    for gold, impossible, pred in rows:
        if impossible:
            e = int(pred == "")
            f = float(e)
            na_t += 1
            na_c += e
        else:
            e = compute_exact(gold, pred)
            f = compute_f1(gold, pred)
            ha_em.append(e)
            ha_f1.append(f)
        em.append(e)
        f1.append(f)
    out = {
        "n": len(rows),
        "em": 100 * float(np.mean(em)),
        "f1": 100 * float(np.mean(f1)),
    }
    if ha_em:
        out["hasans_em"] = 100 * float(np.mean(ha_em))
        out["hasans_f1"] = 100 * float(np.mean(ha_f1))
    if na_t:
        out["noans_acc"] = 100 * na_c / na_t
    return out


def choose(proba, texts, tau):
    """Chon ung vien co diem cao nhat; neu do tin thap hoac chinh la 'khong dap an' thi bo."""
    best = int(np.argmax(proba))
    if texts[best] == "\x00NULL" or proba[best] < tau:
        return ""
    return texts[best]


def build_xy(records, prior, chunk=2000):
    Xs, ys, metas = [], [], []
    for i in range(0, len(records), chunk):
        X, y, meta = [], [], []
        for rec in records[i:i + chunk]:
            xi, yi, mi = build_matrix(rec, prior, with_labels=True)
            X.extend(xi)
            y.extend(yi)
            metas.append(mi)
        Xs.append(np.array(X, dtype=np.float32))
        ys.append(np.array(y, dtype=np.int8))
    return np.vstack(Xs), np.concatenate(ys), metas


def predict_grouped(model, records, prior):
    """Tra ve list (proba_array, meta) theo thu tu tung cau hoi."""
    built = []
    for rec in records:
        xi, _, meta = build_matrix(rec, prior, with_labels=False)
        built.append((np.array(xi, dtype=np.float32), meta))
    batch = np.vstack([x for x, _ in built])
    probs = model.predict_proba(batch)[:, 1]

    grouped = []
    pos = 0
    for xi, meta in built:
        k = xi.shape[0]
        grouped.append((probs[pos:pos + k], meta))
        pos += k
    return grouped


def sweep_tau(grouped, taus):
    best = (-1, None)
    table = []
    for tau in taus:
        rows = []
        for proba, meta in grouped:
            pred = choose(proba, meta["texts"], tau)
            rows.append((meta["gold"], meta["is_impossible"], pred))
        m = metrics(rows)
        table.append((tau, m))
        score = m["f1"]
        if score > best[0]:
            best = (score, tau, m)
    return best, table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data/candidates/train.jsonl")
    parser.add_argument("--val_file", default="data/candidates/val.jsonl")
    parser.add_argument("--out_dir", default="models/reranker")
    parser.add_argument("--model", default="hgb", choices=["hgb", "lr"])
    parser.add_argument("--show_tau_table", action="store_true")
    args = parser.parse_args()

    print("Loading candidate pools...")
    train_recs = load_records(args.train_file)
    val_recs = load_records(args.val_file)
    print(f"  train {len(train_recs)} questions, val {len(val_recs)} questions")

    prior = fit_prior(train_recs)
    print(f"Prior do dai dap an (tu train): {prior}")

    Xtr, ytr, mtr = build_xy(train_recs, prior)
    print(f"  {Xtr.shape[0]} candidate rows, {int(ytr.sum())} duong tinh (match chinh xac)")

    if args.model == "hgb":
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=6,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
            class_weight="balanced", random_state=42,
        )
    else:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)

    print("\nPredicting on val...")
    grouped = predict_grouped(clf, val_recs, prior)

    raw_rows = [(m["gold"], m["is_impossible"], m["extractor_pick"]) for _, m in grouped]
    rule_rows = [(m["gold"], m["is_impossible"], m["rule_pick"]) for _, m in grouped]
    print("  extractor argmax (khong tu choi):      ", fmt(metrics(raw_rows)))
    print("  hanh vi nhu evaluate.py (baseline):     ", fmt(metrics(rule_rows)))

    taus = [round(t, 3) for t in np.arange(0.0, 0.95, 0.025)]
    best, table = sweep_tau(grouped, taus)
    if args.show_tau_table:
        for tau, m in table:
            print(f"  tau={tau:.3f} EM={m['em']:.2f} F1={m['f1']:.2f} "
                  f"NoAns={m.get('noans_acc', 0):.2f} HasAnsEM={m.get('hasans_em', 0):.2f}")

    _, best_tau, best_m = best
    print(f"\nRe-ranker tren val: tau*={best_tau:.3f}  " +
          "  ".join(f"{k}={v:.2f}" for k, v in best_m.items()))

    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(args.out_dir, "reranker.pkl"))
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"prior": prior, "tau": best_tau, "feature_names": FEATURE_NAMES,
                   "model": args.model, "val_metrics": best_m},
                  f, ensure_ascii=False, indent=2)
    print(f"Luu ve {args.out_dir}")

    if hasattr(clf, "feature_importances_"):
        imp = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda t: -t[1])[:12]
        print("\nTop dac trung (gain):")
        for name, v in imp:
            print(f"  {name:24s} {v:.1f}")


if __name__ == "__main__":
    main()
