"""
Dich vu QA dung chung cho web va cho danh gia.

Mot duong duy nhat cho ca hai: tokenization thu cong (PhobertTokenizer la tokenizer
cham, khong ho tro return_offsets_mapping/sequence_ids), cua so truot, pool ung vien,
roi reranker chon dap an va quyet dinh tu choi tra loi.

Usage:
    from qa_service import QAService
    svc = QAService()
    r = svc.answer("Ai là chủ tịch?", "Ông Nguyễn Văn A là chủ tịch tỉnh.")
    r["answer"], r["confidence"], r["abstained"]
"""
import json
import os

import joblib
import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from dump_candidates import window_candidates
from evaluate import _tokenize_with_offsets, prepare_eval_features
from reranker_features import build_matrix, normalize_text

# Ngưỡng "duoc phep noi khong biet": quan sat tren val, 0.95 la diem F1 cao nhat
# (58.76 so voi 57.62 cua baseline) va con tra loi duoc 77.9% cau co dap an.
DEFAULT_TAU_NULL = 0.95


def decide(ranked, tau_null=DEFAULT_TAU_NULL):
    """Luật ra quyết định duy nhất: im lặng khi ứng viên 'không có đáp án' thắng
    tuyệt đối VÀ đạt ngưỡng. NULL thắng mà dưới ngưỡng thì vẫn trả lời, và trả lời
    bằng phương án thật tốt nhất (không phải NULL)."""
    cands = ranked.get("candidates") or []
    p_null = float(ranked.get("null_proba") or 0.0)
    p_best = float(cands[0]["proba"]) if cands else 0.0
    tau_null = float(tau_null)
    abstained = not cands or (p_null > p_best and p_null >= tau_null)
    return {
        "answer": None if abstained else cands[0]["text"],
        "abstained": abstained,
        "confidence": p_null if abstained else p_best,
        "tau": tau_null,
        "low_confidence": (not abstained) and p_best < 0.5,
    }


class QAService:
    def __init__(self, model_dir="models/phobert_qa", reranker_dir="models/reranker",
                 max_length=256, doc_stride=64, n_best=20, max_answer_len=64, top_k_pool=40,
                 device=None):
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.n_best = n_best
        self.max_answer_len = max_answer_len
        self.top_k_pool = top_k_pool

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_dir, attn_implementation="eager")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            self.model.half()
        self.model.to(self.device).eval()
        self.num_params = sum(p.numel() for p in self.model.parameters())

        self.reranker = None
        self.prior = None
        self.tau = None
        pkl = os.path.join(reranker_dir, "reranker.pkl")
        meta = os.path.join(reranker_dir, "meta.json")
        if os.path.exists(pkl) and os.path.exists(meta):
            self.reranker = joblib.load(pkl)
            with open(meta, "r", encoding="utf-8") as f:
                m = json.load(f)
            self.prior, self.tau = m["prior"], m["tau"]

        # Lan goi dau tien ton ~2s (CUDA chon kernel cho tung hinh + reranker
        # lan dau chay); di qua buoc nay de cau hoi that cua user do duoc ~60ms.
        self.answer("Ông A là ai?", "Ông A là sinh viên trường đại học A.")

    def _feats(self, question, context):
        return prepare_eval_features(
            {"id": ["q"], "question": [question], "context": [context]},
            self.tokenizer, self.max_length, self.doc_stride,
        )

    def _encode(self, question, context):
        feats = self._feats(question, context)
        input_ids = torch.tensor(feats["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(feats["attention_mask"], dtype=torch.long)

        starts, ends = [], []
        with torch.no_grad():
            for i in range(0, len(input_ids), 16):
                out = self.model(
                    input_ids=input_ids[i:i + 16].to(self.device),
                    attention_mask=attention_mask[i:i + 16].to(self.device),
                )
                starts.append(out.start_logits.float().cpu().numpy())
                ends.append(out.end_logits.float().cpu().numpy())
        return feats, np.concatenate(starts), np.concatenate(ends)

    def explain(self, question, context, window=0):
        """Attention cua question len tung token context, tinh tren mot cua so truot."""
        feats = self._feats(question, context)
        window = max(0, min(int(window), len(feats["input_ids"]) - 1))
        ids = torch.tensor([feats["input_ids"][window]], dtype=torch.long)
        amask = torch.tensor([feats["attention_mask"][window]], dtype=torch.long)
        with torch.no_grad():
            out = self.model(
                input_ids=ids.to(self.device),
                attention_mask=amask.to(self.device),
                output_attentions=True,
            )

        layers = out.attentions[-3:]
        mat = np.mean([a[0].float().cpu().numpy().mean(axis=0) for a in layers], axis=0)

        offsets = feats["offset_mapping"][window]
        tokens = self.tokenizer.convert_ids_to_tokens(feats["input_ids"][window])
        ctx_pos = [i for i, o in enumerate(offsets) if o is not None]
        if not ctx_pos:
            return None
        first_ctx = ctx_pos[0]
        q_pos = list(range(1, first_ctx - 2))
        if not q_pos:
            q_pos = [0]

        clean = lambda t: t.replace("@@", "").replace("Ġ", " ")
        sub = mat[np.ix_(q_pos, ctx_pos)]
        importance = sub.sum(axis=0)
        if importance.max() > importance.min():
            importance = (importance - importance.min()) / (importance.max() - importance.min())

        return {
            "z": sub,
            "x": [clean(tokens[i]) for i in ctx_pos],
            "y": [clean(tokens[i]) for i in q_pos],
            "tokens": [clean(tokens[i]) for i in ctx_pos],
            "scores": [float(v) for v in importance],
            "char_spans": [tuple(offsets[i]) for i in ctx_pos],
            "window": window,
            "n_windows": len(feats["input_ids"]),
            "window_char_range": (offsets[ctx_pos[0]][0], offsets[ctx_pos[-1]][1]),
        }

    def build_pool(self, question, context):
        """Tra ve (record cho reranker, thong tin token de truc quan hoa)."""
        feats, start_logits, end_logits = self._encode(question, context)
        offsets = feats["offset_mapping"]

        pool, null_scores = [], []
        for w, (sl, el, off) in enumerate(zip(start_logits, end_logits, offsets)):
            null_scores.append(float(sl[0] + el[0]))
            for score, s, e, cs, ce in window_candidates(
                sl, el, off, self.n_best, self.max_answer_len
            ):
                pool.append({"score": score, "s": s, "e": e, "cs": cs, "ce": ce, "win": w})

        pool.sort(key=lambda c: -c["score"])
        seen, candidates = set(), []
        for c in pool:
            key = (c["cs"], c["ce"])
            if key in seen:
                continue
            seen.add(key)
            if len(candidates) >= self.top_k_pool:
                break
            candidates.append({
                "rk": len(candidates), "win": c["win"], "s": c["s"], "e": c["e"],
                "cs": c["cs"], "ce": c["ce"], "score": round(c["score"], 3),
                "n_tok": c["e"] - c["s"] + 1, "text": context[c["cs"]:c["ce"]],
            })

        rec = {
            "id": "web", "question": question, "n_windows": len(offsets),
            "null_scores": null_scores,
            "extractor_pick": candidates[0]["text"] if candidates else "",
            "gold": "", "is_impossible": False, "candidates": candidates,
        }
        # Vi tri token dau tien/cuoi cua context trong cua so 0 - phuc vu ve attention
        first_ctx = next((i for i, o in enumerate(offsets[0]) if o is not None), 0)
        tokens = self.tokenizer.convert_ids_to_tokens(feats["input_ids"][0])
        return rec, {"tokens": tokens, "offsets": offsets[0],
                     "context_token_start": first_ctx,
                     "question_token_len": first_ctx - 1}

    def rank(self, question, context):
        """Encode + chấm lại mọi ứng viên. Điểm số KHÔNG đổi theo ngưỡng từ chối,
        nên chạy GPU một lần rồi quay ngưỡng ngay tức thì (tab đánh giá hàng loạt)."""
        rec, tok_info = self.build_pool(question, context)
        X, _, meta = build_matrix(rec, self.prior, with_labels=False)
        proba = self.reranker.predict_proba(np.array(X, dtype=np.float32))[:, 1]
        k = len(rec["candidates"])

        candidates = []
        for i in np.argsort(-proba[:k]):
            c = rec["candidates"][i]
            candidates.append({
                "text": c["text"], "proba": float(proba[i]),
                "extractor_score": c["score"], "n_tokens": c["n_tok"],
                "char_start": c["cs"], "char_end": c["ce"], "window": c["win"],
            })

        return {
            "candidates": candidates,
            "null_proba": float(proba[k]),
            "n_candidates": k,
            "n_windows": rec["n_windows"],
            "pointer_pick": rec["extractor_pick"],
            # Mô hình một tầng nguyên bản: bỏ trống khi min_null > best_score
            # (đúng luật của evaluate.py) nên cột so sánh khớp số đã công bố.
            "pointer_answer": meta["rule_pick"],
            "reranker_available": True,
            **tok_info,
        }

    def answer(self, question, context, top_k=5, tau_null=None):
        question = (question or "").strip()
        context = (context or "").strip()
        if not question or not context:
            return {"answer": None, "abstained": False, "confidence": 0.0,
                    "candidates": [], "error": "Thieu cau hoac doan van ban."}

        if self.reranker is None:
            rec, tok_info = self.build_pool(question, context)
            pick = rec["extractor_pick"] or None
            return {"answer": pick, "abstained": pick is None,
                    "confidence": 0.0,
                    "candidates": [
                        {"text": c["text"], "confidence": None,
                         "extractor_score": c["score"], "n_tokens": c["n_tok"],
                         "char_start": c["cs"], "char_end": c["ce"]}
                        for c in rec["candidates"][:top_k]],
                    "reranker_available": False, "warning": "Chua thay models/reranker",
                    **tok_info}

        ranked = self.rank(question, context)
        out = {k: v for k, v in ranked.items() if k != "pointer_answer"}
        out["candidates"] = [
            {k: v for k, v in c.items() if k != "proba"} | {"confidence": c["proba"]}
            for c in ranked["candidates"][:top_k]
        ]
        out["null_margin"] = ranked["null_proba"]
        out["pointer_would_answer"] = normalize_text(ranked["pointer_answer"]) != ""
        out.update(decide(ranked, DEFAULT_TAU_NULL if tau_null is None else tau_null))
        return out
