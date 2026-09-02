"""
Lot REST API mong cho ung dung Vietnamese QA.

API nay khong tai dien bat ky logic nao: no goi thang QAService - cung mot doi
tuong ma web app (app/app.py) dang dung. Vi vay mot cau hoi tra loi khac nhau
giua web va API la khong the xay ra.

Chay (tai cua so thu muc project):
    python src/api.py
hoac:
    python -m uvicorn api:app --app-dir src --port 8000

Thu:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/answer -H "Content-Type: application/json" ^
      -d "{\"question\":\"Ai phat trien PhoBERT?\",\"context\":\"PhoBERT duoc phat trien boi VinAI.\"}"

Model (513MB) duoc nap mot lan luc khoi dong, nen lenh dau tien co the mat ~40 giay.
"""
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from qa_service import DEFAULT_TAU_NULL, QAService  # noqa: E402

# Nhung key mang token/offset chi phuc vu ve attention trong web, khong can cho khong API.
_INTERNAL_KEYS = {"tokens", "offsets", "context_token_start", "question_token_len"}

service = QAService(
    model_dir=str(ROOT / "models" / "phobert_qa"),
    reranker_dir=str(ROOT / "models" / "reranker"),
)

app = FastAPI(
    title="Vietnamese QA API",
    version="1.0",
    description="Extractive QA tren ViQuAD: PhoBERT extractor + reranker ung vien.",
)


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Cau hoi tieng Viet")
    context: str = Field(..., min_length=1, description="Doan van ban chua thong tin")
    top_k: int = Field(5, ge=1, le=40, description="So phuong an tra ve")
    tau_null: Optional[float] = Field(
        DEFAULT_TAU_NULL, ge=0.0, le=1.0,
        description="Model chi tu choi tra loi khi phuong an 'khong dap an' "
                    "dat it nhat nguong nay",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "question": "Ai đã phát triển PhoBERT?",
                "context": "PhoBERT là mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt, "
                           "được phát triển bởi VinAI.",
            }]
        }
    }


def _round_floats(obj, digits=4):
    """Cat du thua float de JSON de doc trong /docs va curl."""
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, digits) for v in obj]
    return obj


@app.get("/health")
def health():
    """Kiem tra nhanh: model da nam trong bo nho chua."""
    return {
        "status": "ok",
        "device": service.device,
        "parameters": service.num_params,
        "reranker_loaded": service.reranker is not None,
        "default_tau_null": DEFAULT_TAU_NULL,
    }


@app.post("/answer")
def answer(req: AnswerRequest):
    """Tra ve dap an, do tin cậy, va danh sach phuong da so sanh kem diem."""
    t0 = time.perf_counter()
    result = service.answer(req.question, req.context, top_k=req.top_k,
                            tau_null=req.tau_null)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])

    result = {k: v for k, v in result.items() if k not in _INTERNAL_KEYS}
    result["latency_ms"] = latency_ms
    return _round_floats(result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
