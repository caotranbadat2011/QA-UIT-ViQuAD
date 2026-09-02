"""
Dac trung cua tung ung vien dap an, dung chung cho luc train va luc suy luan.

Moi dac trung chi dung thong tin co san tai thoi diem chay (question, context,
logit cua model QA) - khong dung den dap an chuan, nen khong bi gian lan.
"""
import json
import math
import re
import string

import numpy as np

PUNCT = set(string.punctuation)

# Tu thung lam bien dap bi thua khi model keo dai cum tu
PARTICLES = {
    "của", "và", "tại", "ở", "trong", "ngoài", "là", "các", "những", "một", "để",
    "với", "theo", "vào", "khi", "mà", "thì", "được", "đến", "bởi", "do", "cho",
    "như", "nếu", "rằng", "này", "kia", "đó", "ấy", "thuộc", "hay", "cũng",
}

# Tu de hoi - neu no nam trong dap an thuong la cat sai bien hoac sai cho
QUESTION_WORDS = {
    "bao nhiêu", "khi nào", "năm nào", "ở đâu", "đâu", "ai", "gì", "như thế nào",
    "bao lâu", "tại sao", "như nào", "cách nào", "loại nào", "mấy", "gồm những",
}

WORD_RE = re.compile(r"\w+")


def normalize_text(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in PUNCT)
    return " ".join(s.split())


def words_of(s):
    return WORD_RE.findall(s.lower())


def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def fit_prior(train_records):
    """Uoc luong phan phoi do dai dap an tu train (chi lay ung vien khop chuan)."""
    lens = []
    for r in train_records:
        if r["is_impossible"]:
            continue
        gold = normalize_text(r["gold"])
        for c in r["candidates"]:
            if normalize_text(c["text"]) == gold:
                lens.append(c["n_tok"])
                break
    lens.sort()
    if not lens:
        lens = [10]
    n = len(lens)
    return {
        "median_tok": lens[n // 2],
        "p10_tok": lens[int(0.10 * (n - 1))],
        "p90_tok": lens[int(0.90 * (n - 1))],
        "p95_tok": lens[int(0.95 * (n - 1))],
        "n_observed": n,
    }


FEATURE_NAMES = [
    "score", "rank", "score_gap_to_top", "score_z", "n_cand_log",
    "n_tok", "n_words", "n_chars",
    "log_len_ratio", "abs_log_len_ratio", "too_long", "too_short",
    "starts_particle", "ends_particle", "edge_punct",
    "contains_qword", "q_overlap",
    "n_comma", "has_period", "starts_upper", "has_digit",
    "min_null_margin", "max_null_margin", "span_vs_null",
    "window_idx", "n_windows_log",
    "contained_in_higher", "n_lower_contained", "len_diff_to_container",
    "boundary_variant_exists", "is_null", "null_score_raw", "null_top_margin",
]


def _cand_features(rec, cand, prior, pool_scores, top_score, span_score_of_null):
    text = cand["text"]
    n_tok = cand["n_tok"]
    ws = words_of(text)
    n_words = max(1, len(ws))
    qws = words_of(normalize_text(rec["question"]))
    tw = set(ws)
    qn = normalize_text(text)

    ratio = math.log((n_tok + 1.0) / (prior["median_tok"] + 1.0))
    first = ws[0] if ws else ""
    last = ws[-1] if ws else ""

    contains_q = 0
    for q in QUESTION_WORDS:
        if q in qn:
            contains_q = 1
            break

    q_overlap = len(tw & set(qws)) / n_words

    cs, ce = cand["cs"], cand["ce"]
    contained_in_higher = 0
    n_lower_contained = 0
    len_diff_to_container = 0.0
    boundary_variant = 0
    for other in rec["candidates"]:
        if other is cand:
            continue
        osc, oce = other["cs"], other["ce"]
        if other["score"] > cand["score"] and osc <= cs and oce >= ce and (osc, oce) != (cs, ce):
            contained_in_higher = 1
            ratio_len = (oce - osc) / max(1, (ce - cs))
            len_diff_to_container = max(len_diff_to_container, math.log(ratio_len))
        if other["score"] < cand["score"] and cs <= osc and ce >= oce and (osc, oce) != (cs, ce):
            n_lower_contained += 1
        if (other["cs"] == cs) != (other["ce"] == ce):
            boundary_variant = 1

    std = pool_scores.std() if pool_scores.size > 1 else 1.0
    return [
        cand["score"],
        cand["rk"],
        top_score - cand["score"],
        (cand["score"] - pool_scores.mean()) / (std + 1e-6),
        math.log(max(1, len(rec["candidates"]))),
        n_tok,
        len(ws),
        ce - cs,
        ratio,
        abs(ratio),
        int(n_tok > prior["p95_tok"]),
        int(n_tok <= 1),
        int(first in PARTICLES),
        int(last in PARTICLES),
        int(bool(text) and (text[0] in PUNCT or text[-1] in PUNCT)),
        contains_q,
        q_overlap,
        text.count(","),
        int("." in text or ";" in text),
        int(bool(text) and text[0].isupper()),
        int(bool(re.search(r"\d", text))),
        cand["score"] - span_score_of_null[0],
        cand["score"] - span_score_of_null[1],
        cand["score"] - span_score_of_null[2],
        cand["win"],
        math.log(max(1, rec["n_windows"])),
        contained_in_higher,
        n_lower_contained,
        len_diff_to_container,
        boundary_variant,
        0, 0, 0, 0,
    ]


def _null_features(rec, prior, top_score, min_null, max_null):
    n_real = len(rec["candidates"])
    return [
        min_null, -1, 0.0, 0.0, math.log(max(1, n_real)),
        0, 0, 0, 0.0, 0.0, 0, 0,
        0, 0, 0, 0, 0.0,
        0, 0, 0, 0,
        0.0, 0.0, top_score - min_null,
        -1, math.log(max(1, rec["n_windows"])),
        0, 0, 0.0, 0,
        1, min_null, max_null, top_score - min_null,
    ]


def build_matrix(rec, prior, with_labels):
    """Tra ve (X, y, meta). Null candidate luon duoc them vao pool."""
    cands = rec["candidates"]
    nulls = rec["null_scores"]
    min_null = min(nulls) if nulls else 0.0
    max_null = max(nulls) if nulls else 0.0
    top_score = cands[0]["score"] if cands else min_null
    scores = np.array([c["score"] for c in cands]) if cands else np.array([0.0])
    # span_score_of_null: cac bien the cua null score de candidate so sanh duoc
    span_ref = (min_null, max_null, top_score)

    X = [_cand_features(rec, c, prior, scores, top_score, span_ref) for c in cands]
    X.append(_null_features(rec, prior, top_score, min_null, max_null))

    if with_labels:
        gold = normalize_text(rec["gold"])
        impossible = rec["is_impossible"]
        y = [
            int(not impossible and normalize_text(c["text"]) == gold) for c in cands
        ] + [int(impossible)]
    else:
        y = None

    meta = {
        "id": rec["id"],
        "texts": [c["text"] for c in cands] + ["\x00NULL"],
        "gold": rec["gold"],
        "is_impossible": rec["is_impossible"],
        "extractor_pick": rec["extractor_pick"],
        "rule_pick": "" if (cands and min_null > top_score) else rec["extractor_pick"],
    }
    return X, y, meta
