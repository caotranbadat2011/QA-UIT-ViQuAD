"""
Cham that pipeline hai tang tren mot mau cua test set, de doi chiu voi mot tang.

GPU chi chay MOT LAN cho moi cau: diem cua tung ung vien va diem 'khong co dap an'
khong doi theo nguong tu choi, nen quay nguong tau_chi tinh lai diem so ca mau
trong vai mili giay (khong can model).

Usage (CLI, dung cho bao cao):
    python src/batch_eval.py --n 200 --out batch_eval.json
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from qa_service import DEFAULT_TAU_NULL, decide
try:
    from evaluate import compute_exact, compute_f1
except ImportError:  # goi theo kieu package: python -m src.batch_eval
    from src.evaluate import compute_exact, compute_f1

REFUSED = "(từ chối)"
# Điểm 'không có đáp án' của reranker dồn về gần 1 nên mọi thay đổi quyết định
# nằm trong dải này; quét rộng hơn chỉ vẽ ra một đường nằm ngang.
TAU_GRID = np.round(np.arange(0.88, 1.0001, 0.01), 2)


def load_sample(test_file, n=60, seed=42):
    """Mẫu phân tầng: giữ đúng tỉ lệ câu không có đáp án của test set."""
    df = pd.read_parquet(test_file)
    answerable = df[~df["is_impossible"]]
    unanswerable = df[df["is_impossible"]]
    k_un = min(len(unanswerable), round(n * len(unanswerable) / len(df)))
    k_an = min(len(answerable), max(0, n - k_un))
    k_un = min(len(unanswerable), max(0, n - k_an))
    pick = pd.concat([answerable.sample(k_an, random_state=seed),
                      unanswerable.sample(k_un, random_state=seed)])
    return pick.sample(frac=1, random_state=seed).to_dict("records")


def run(service, rows, progress_callback=None):
    """Phần tốn GPU: trượt cửa sổ, lấy pool 40 ứng viên và chấm lại bằng reranker."""
    records = []
    for i, row in enumerate(rows):
        ranked = service.rank(str(row["question"]), str(row["context"]))
        records.append({
            "id": str(row["qa_id"]),
            "question": str(row["question"]),
            "context": str(row["context"]),
            "gold": "" if row["is_impossible"] else str(row["answer_text"]),
            "is_impossible": bool(row["is_impossible"]),
            "candidates": [{"text": c["text"], "proba": c["proba"]}
                           for c in ranked["candidates"]],
            "null_proba": ranked["null_proba"],
            "n_candidates": ranked["n_candidates"],
            "n_windows": ranked["n_windows"],
            "pointer_answer": str(ranked["pointer_answer"]),
        })
        if progress_callback:
            progress_callback(i + 1, len(rows))
    return records


def two_stage_answer(tau_null=DEFAULT_TAU_NULL):
    """Hàm chọn đáp án của mô hình hai tầng tại một ngưỡng từ chối."""
    def _pick(record):
        d = decide(record, tau_null)
        return "" if d["abstained"] else (d["answer"] or "")
    return _pick


def evaluate(records, answer_of):
    """Điểm số theo đúng chuẩn evaluate.py: câu không đáp án được tính đúng khi
    mô hình bỏ trống, và F1 lấy max trên các đáp án chuẩn."""
    em, f1, has_em, has_f1 = [], [], [], []
    no_correct = no_total = refused = 0
    per_row = []

    for r in records:
        pred = (answer_of(r) or "").strip()
        row_em = compute_exact(r["gold"], pred)
        row_f1 = compute_f1(r["gold"], pred)
        em.append(row_em)
        f1.append(row_f1)
        refused += int(pred == "")
        if r["is_impossible"]:
            no_total += 1
            no_correct += row_em
        else:
            has_em.append(row_em)
            has_f1.append(row_f1)
        per_row.append({"id": r["id"], "pred": pred, "em": int(row_em),
                        "f1": float(row_f1)})

    return {
        "n": len(records),
        "overall_em": 100 * float(np.mean(em)) if em else 0.0,
        "overall_f1": 100 * float(np.mean(f1)) if f1 else 0.0,
        "has_ans_em": 100 * float(np.mean(has_em)) if has_em else 0.0,
        "has_ans_f1": 100 * float(np.mean(has_f1)) if has_f1 else 0.0,
        "no_ans_accuracy": 100 * no_correct / no_total if no_total else 0.0,
        "refusal_rate": 100 * refused / len(records) if records else 0.0,
        "per_row": per_row,
    }


def metrics_table(records, tau_null=DEFAULT_TAU_NULL):
    """Bảng so sánh một tầng / hai tầng + chỉ số thay đổi (delta)."""
    one = evaluate(records, lambda r: r["pointer_answer"])
    two = evaluate(records, two_stage_answer(tau_null))
    keys = [("overall_em", "EM toàn bộ"), ("overall_f1", "F1 toàn bộ"),
            ("has_ans_em", "EM câu có đáp án"), ("has_ans_f1", "F1 câu có đáp án"),
            ("no_ans_accuracy", "Độ chính xác câu bẫy"),
            ("refusal_rate", "Tỉ lệ bỏ trả lời")]
    return [{
        "Chỉ số": label,
        "Một tầng": round(one[k], 2),
        "Hai tầng": round(two[k], 2),
        "Δ": round(two[k] - one[k], 2),
    } for k, label in keys], one, two


def verdict(record, one_row, two_row, tau_null):
    """Nhãn giải thích vì sao câu này khác nhau giữa hai tầng."""
    d = decide(record, tau_null)
    refused = d["abstained"]
    one_refused = not one_row["pred"]
    gold_in_pool = any(compute_exact(record["gold"], c["text"]) == 1
                       for c in record["candidates"])

    if record["is_impossible"]:
        if refused and one_refused:
            return "Cả hai bỏ đúng ✅"
        if refused:
            return "Reranker chặn câu bẫy ✅"
        if one_refused:
            return "Reranker bịa đáp án ❌"
        return "Cả hai bịa đáp án ❌"

    if refused:
        return "Bỏ lỡ, pool có đáp án ❌" if gold_in_pool else "Bỏ lỡ ❌"
    if one_refused:
        if two_row["em"]:
            return "Reranker cứu câu bị bỏ ✅"
        return "Reranker trả lời, vẫn sai ❌"
    if two_row["em"] and not one_row["em"]:
        return "Reranker sửa lỗi ✅"
    if one_row["em"] and not two_row["em"]:
        return "Reranker làm hỏng ❌"
    if two_row["f1"] > one_row["f1"]:
        return "F1 sát hơn ✅"
    if two_row["f1"] < one_row["f1"]:
        return "F1 kém hơn ❌"
    return "Như nhau"


def compare(records, tau_null=DEFAULT_TAU_NULL):
    """Bảng từng câu: đáp án của hai tầng, độ tin cậy và kết luận."""
    one = evaluate(records, lambda r: r["pointer_answer"])
    two = evaluate(records, two_stage_answer(tau_null))
    rows = []
    for r, a, b in zip(records, one["per_row"], two["per_row"]):
        d = decide(r, tau_null)
        rows.append({
            "id": r["id"],
            "Câu hỏi": r["question"],
            "Đáp án đúng": r["gold"] or REFUSED,
            "Một tầng": a["pred"] or REFUSED,
            "Hai tầng": b["pred"] or REFUSED,
            "F1": round(b["f1"], 2),
            "Độ tin cậy": round(d["confidence"], 3),
            "Độ tin 'không có đáp án'": round(r["null_proba"], 3),
            "Ứng viên": r["n_candidates"],
            "Kết luận": verdict(r, a, b, tau_null),
            "context": str(r.get("context", "")),
        })
    return rows


def sweep(records, taus):
    """Đường cong đánh đổi của ngưỡng từ chối — chỉ tính CPU nên vẽ ngay tức thì."""
    out = []
    for tau in taus:
        m = evaluate(records, two_stage_answer(tau))
        out.append({"tau": round(float(tau), 2),
                    "overall_em": m["overall_em"], "overall_f1": m["overall_f1"],
                    "no_ans_accuracy": m["no_ans_accuracy"],
                    "refusal_rate": m["refusal_rate"]})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_file", default="data/processed/test.parquet")
    p.add_argument("--model_dir", default="models/phobert_qa")
    p.add_argument("--reranker_dir", default="models/reranker")
    p.add_argument("--n", type=int, default=200, help="0 = toàn bộ test set")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tau", type=float, default=DEFAULT_TAU_NULL)
    p.add_argument("--out", default=None, help="file JSON chứa bảng so sánh")
    args = p.parse_args()

    from qa_service import QAService
    rows = load_sample(args.test_file, args.n or None, args.seed)
    print(f"Mẫu {len(rows)} câu (seed={args.seed}) từ {args.test_file}")
    service = QAService(model_dir=args.model_dir, reranker_dir=args.reranker_dir)

    t0 = time.perf_counter()
    records = run(service, rows,
                  progress_callback=lambda i, n: print(f"\r  {i}/{n}", end="", flush=True))
    print(f"\r  xong trong {time.perf_counter() - t0:.1f}s "
          f"({(time.perf_counter() - t0) / len(rows) * 1000:.0f} ms/câu)")

    table, one, two = metrics_table(records, args.tau)
    print(f"\nNgưỡng từ chối tau={args.tau}")
    for row in table:
        print(f"  {row['Chỉ số']:<26} một tầng {row['Một tầng']:>7.2f}   "
              f"hai tầng {row['Hai tầng']:>7.2f}   Δ {row['Δ']:+.2f}")

    if args.out:
        payload = {"n": len(records), "seed": args.seed, "tau": args.tau,
                   "records": records,
                   "one_stage": {k: v for k, v in one.items() if k != "per_row"},
                   "two_stage": {k: v for k, v in two.items() if k != "per_row"},
                   "table": table,
                   "sweep": sweep(records, TAU_GRID)}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nĐã ghi {args.out}")


if __name__ == "__main__":
    main()
