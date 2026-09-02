"""
Thuc nghiem nhanh: cach hoc nao khiem khac giup reranker chon span tot hon extractor?

Van phat hien duoc: bo phan biet (classifier tren 865k hang) dat AUC 0.93 giua span
dung va span sai, nhung argmax cua no trung gan hoan toan argmax cua model PhoBERT
vi dac trung 'score' du lon de no chi can sao chep quyet dinh cu.

Script nay cache dac trung mot lan roi thu ba cach hoc:
  A) classifier nhu cu (diem chuan bi)
  B) classifier but bo cac dac trung tuyet doi cua score -> buoc hoc so sanh tuong doi
  C) hoc doi ngoai (pairwise) tren hieu so dac trung cua (span dung, span sai) cung cau

Usage:
    python src/experiment_reranker.py            # tan dung cache neu co
    python src/experiment_reranker.py --rebuild  # build lai dac trung
"""
import argparse
import itertools
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from evaluate import compute_exact, compute_f1
from reranker_features import FEATURE_NAMES, build_matrix, fit_prior, load_records

SCORE_COLS = [FEATURE_NAMES.index(n) for n in
              ("score", "score_gap_to_top", "score_z", "min_null_margin",
               "max_null_margin", "span_vs_null", "null_score_raw", "null_top_margin")]


def build_and_cache(records, prior, tag, out_dir):
    path = os.path.join(out_dir, f"feat_{tag}.npz")
    meta_path = os.path.join(out_dir, f"meta_{tag}.jsonl")
    if os.path.exists(path) and os.path.exists(meta_path):
        z = np.load(path)
        metas = [json.loads(l) for l in open(meta_path, encoding="utf-8")]
        return z["X"], z["y"], metas

    X, y, metas, sizes = [], [], [], []
    for rec in records:
        xi, yi, meta = build_matrix(rec, prior, with_labels=True)
        X.append(np.array(xi, dtype=np.float32))
        y.extend(yi)
        sizes.append(len(xi))
        metas.append(meta)
    X = np.vstack(X)
    np.savez(path, X=X, y=np.array(y, dtype=np.int8), sizes=np.array(sizes))
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    return X, np.array(y, dtype=np.int8), metas


def group_slices(sizes):
    idx = np.cumsum([0] + list(sizes))
    return [slice(idx[i], idx[i + 1]) for i in range(len(sizes))]


def eval_selection(X, metas, slices, scores_fn, tag):
    """Chon span tot nhat moi cau (loai bo null) va do EM/F1 tren cau co dap an."""
    em, f1, nfix, nbreak = [], [], 0, 0
    for sl, m in zip(slices, metas):
        if m["is_impossible"]:
            continue
        s = scores_fn(X[sl])
        k = len(m["texts"]) - 1                      # khong tinh ung vien null
        best = int(np.argmax(s[:k]))
        new = m["texts"][best]
        old = m["extractor_pick"]
        em.append(compute_exact(m["gold"], new))
        f1.append(compute_f1(m["gold"], new))
        if compute_exact(m["gold"], new) and not compute_exact(m["gold"], old):
            nfix += 1
        if compute_exact(m["gold"], old) and not compute_exact(m["gold"], new):
            nbreak += 1
    print(f"  [{tag:28s}] n={len(em):5d}  EM {100*np.mean(em):6.2f}  F1 {100*np.mean(f1):6.2f}"
          f"   (+{nfix}/-{nbreak} so voi extractor)")
    return 100 * np.mean(em)


def make_pairwise(X, y, slices, metas, max_pairs=6):
    """Hoc w.tu (f_dung - f_sai) bang logistic hoi quy nhi phan."""
    A, B, T = [], [], []
    for sl, m in zip(slices, metas):
        xs, ys = X[sl], y[sl]
        pos = np.flatnonzero(ys == 1)
        neg = np.flatnonzero(ys == 0)
        if len(pos) == 0 or len(neg) == 0:
            continue
        for p, q in itertools.islice(itertools.product(pos, neg), max_pairs * len(pos)):
            A.append(xs[p])
            B.append(xs[q])
            T.append(1)
    A, B = np.array(A, dtype=np.float32), np.array(B, dtype=np.float32)
    return A - B, np.array(T, dtype=np.int8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--out_dir", default="data/candidates/cache")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.rebuild:
        for f in os.listdir(args.out_dir):
            os.remove(os.path.join(args.out_dir, f))

    train_recs = load_records("data/candidates/train.jsonl")
    val_recs = load_records("data/candidates/val.jsonl")
    prior = fit_prior(train_recs)

    Xt, yt, mt = build_and_cache(train_recs, prior, "train", args.out_dir)
    Xv, yv, mv = build_and_cache(val_recs, prior, "val", args.out_dir)
    sizes_t = np.load(os.path.join(args.out_dir, "feat_train.npz"))["sizes"]
    sizes_v = np.load(os.path.join(args.out_dir, "feat_val.npz"))["sizes"]
    slices_t, slices_v = group_slices(sizes_t), group_slices(sizes_v)

    print(f"train rows {Xt.shape}, val rows {Xv.shape}")

    keep = [i for i in range(len(FEATURE_NAMES)) if i not in SCORE_COLS]
    masks = {"A full features": list(range(len(FEATURE_NAMES))), "B no absolute score": keep}

    for name, cols in masks.items():
        clf = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.08, max_depth=6, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20, class_weight="balanced", random_state=0)
        clf.fit(Xt[:, cols], yt)
        eval_selection(Xv[:, cols], mv, slices_v, lambda s: clf.predict_proba(s)[:, 1], f"hgb {name}")

    for cols, tag in ((list(range(len(FEATURE_NAMES))), "C pairwise full"), (keep, "D pairwise noscore")):
        A, T = make_pairwise(Xt[:, cols], yt, group_slices(sizes_t), mt)
        lr = LogisticRegression(max_iter=400, C=1.0)
        lr.fit(A, T)
        w = lr.coef_[0]
        print(f"     (pairs {A.shape[0]:,})")
        eval_selection(Xv[:, cols], mv, slices_v, lambda s: s @ w, tag)


if __name__ == "__main__":
    main()
