# 🇻🇳 Vietnamese Question Answering System

> Final Project - Transformer-Based NLP Application  
> Fine-tuned PhoBERT for Extractive Question Answering on ViQuAD Dataset

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30%2B-green)](https://huggingface.co/transformers/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63%2B-red)](https://streamlit.io/)

---

## 🎯 Overview

Hệ thống **Question Answering** sử dụng mô hình **PhoBERT** (Transformer-based) để trích xuất câu trả lời từ đoạn văn tiếng Việt. Dự án được phát triển như Final Project cho môn học về NLP.

### ✨ Features
- 🤖 **Pre-trained PhoBERT:** Fine-tuned trên ViQuAD dataset
- 🎯 **Extractive QA:** Trích xuất câu trả lời trực tiếp từ context, sliding window 256/64
- ❓ **Calibrated abstention:** Tầng reranker quyết định "không có đáp án" và nói được *tin đến đâu*
- 🧪 **Batch evaluation lab:** Chấm thật trên câu hỏi test, so một tầng / hai tầng, quét ngưỡng từ chối tức thì
- 🌐 **Web Interface:** Giao diện web đẹp với Streamlit
- ⚡ **Fast Inference:** Tối ưu với FP16 và GPU support

> ℹ️ **Trọng số model ở trong repo qua Git LFS** (`models/phobert_qa/model.safetensors`, 537MB).
> `git clone` bình thường chỉ lấy về một pointer 134 byte, nên cần:
> `git lfs install && git lfs pull` rồi mới chạy app. Checkpoint trung gian và `optimizer.pt`
> (1.07GB mỗi file) **không** push — repo chỉ chứa model cuối cùng của epoch 3.

---

## 🚀 Quick Start

### 3 bước đơn giản để chạy ứng dụng:

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Test model (optional)
python test_model.py

# 3. Chạy web app
streamlit run app/app.py
```

Hoặc sử dụng script tự động (Windows):
```bash
run_app.bat
```

Truy cập: `http://localhost:8501`

📖 **Chi tiết:** Xem [QUICKSTART.md](QUICKSTART.md)

---

## 📁 Project Structure

```
train_Vit/
├── app/                    # Web application
│   └── app.py             # Streamlit app (2 tab: Hỏi đáp + Đánh giá hàng loạt)
├── src/                   # Source code
│   ├── train.py           # Training pipeline (PhoBERT QA)
│   ├── evaluate.py        # Chấm toàn bộ test set, lưu predictions.json
│   ├── dump_candidates.py # Sinh pool 40 ứng viên/câu cho reranker
│   ├── train_reranker.py  # Train + đo ngưỡng từ chối trên val
│   ├── reranker_features.py # 34 đặc trưng bề mặt của một ứng viên
│   ├── qa_service.py      # Service dùng chung cho web + API
│   ├── batch_eval.py      # Chấm hàng loạt một tầng vs hai tầng, quét τ
│   ├── api.py             # FastAPI mỏng trên QAService
│   └── data_preprocessing.py
├── models/phobert_qa/     # Encoder đã fine-tune (537MB qua Git LFS)
├── models/reranker/       # reranker.pkl + meta.json (có trong git)
├── data/                  # Datasets
├── notebooks/             # Jupyter notebooks
├── requirements.txt       # Dependencies
├── README.md              # This file
├── README_DEPLOYMENT.md   # Deployment guide
├── QUICKSTART.md          # Quick start guide
└── SUBMISSION_GUIDE.md    # Submission instructions
```

---

## 📊 Model Details

### Architecture
- **Base Model:** PhoBERT-base (VinAI Research), `RobertaForQuestionAnswering`
- **Task:** Extractive Question Answering
- **Parameters:** 134,409,218 (~134.4M) — 537MB trọng số FP32
- **Tầng 2:** `HistGradientBoostingClassifier` trên 34 đặc trưng bề mặt của mỗi ứng viên
  (kèm một ứng viên giả `\x00NULL` để học quyết định "không có đáp án")
- **Framework:** Hugging Face Transformers + PyTorch + scikit-learn

### Training Configuration
| Hyperparameter | Value |
|----------------|-------|
| Learning Rate | 3e-5 |
| Batch Size | 8 (effective 16) |
| Epochs | 3 |
| Max Length | 256 tokens (doc_stride 64) |
| Optimizer | AdamW |
| Precision | FP16 |

### Dataset
- **Name:** ViQuAD (Vietnamese Question Answering Dataset)
- **Language:** Vietnamese
- **Splits:** train 22702 câu · val 2872 · test 2882
- **Câu không có đáp án:** 32.3% train · 32.6% val · 32.8% test (944/2882 ở test)
- **Format:** JSON/Parquet

---

## 💻 Usage

### Web Application

```bash
streamlit run app/app.py
```

App có 2 tab:

1. **📝 Hỏi đáp trên đoạn văn của bạn** — dán đoạn văn + câu hỏi, nhận đáp án kèm độ tin cậy
   hiệu chuẩn, danh sách mọi phương án model đã cân nhắc, và thanh *Độ khắt khe* để đổi mức
   dám nói "không có đáp án".
2. **🧪 Đánh giá hàng loạt trên test set** — bấm một nút, model chạy thật trên một mẫu câu hỏi
   của test set rồi tự chấm: bảng một tầng / hai tầng, đường cong đánh đổi của ngưỡng từ chối,
   và từng câu được sửa / bị làm hỏng.

**Ví dụ nhanh (Tab 1):**
```
Context: "Hà Nội là thủ đô của Việt Nam, nằm bên bờ sông Hồng."
Question: "Hà Nội nằm ở đâu?"
Answer: "nằm bên bờ sông Hồng." (độ tin 0.961)
```

### Training (Optional)

```bash
python src/train.py \
    --train_file data/processed/train.parquet \
    --val_file data/processed/val.parquet \
    --model_name vinai/phobert-base \
    --output_dir models/phobert_qa
```

### Evaluation

```bash
# Toàn bộ test set, một tầng -> predictions.json + EM/F1/HasAns/NoAns
python src/evaluate.py \
    --model_dir models/phobert_qa \
    --test_file data/processed/test.parquet

# So sánh một tầng vs hai tầng trên mẫu test, kèm đường cong ngưỡng τ
python src/batch_eval.py --n 200 --out batch_eval_n200.json

# REST API cùng logic với web app
python src/api.py
```

---

## 📈 Results

### 1. Toàn bộ test set — model một tầng (n = 2882)

Đo bằng `python src/evaluate.py --model_dir models/phobert_qa --test_file data/processed/test.parquet`
(1938 câu có đáp án, 944 câu bẫy).

| Chỉ số | Điểm |
|---|---|
| **EM toàn bộ** | **42.40** |
| **F1 toàn bộ** | **57.90** |
| EM câu có đáp án | 47.57 |
| F1 câu có đáp án | 70.62 |
| Độ chính xác câu bẫy | 31.78 |
| F1 khi model chịu trả lời | 79.16 |

**Chẩn đoán lỗi** — 2882 câu chia thành 6 nhóm, tính lại được từ `predictions.json` bằng
`normalize_text / compute_exact / compute_f1` trong `src/evaluate.py`:

| Nhóm lỗi | Câu | % test |
|---|---|---|
| Trả lời đúng (EM) | 922 | 32.0% |
| Bỏ đúng câu bẫy | 300 | 10.4% |
| **Đúng chỗ, cắt sai biên** (375 dài thừa + 233 dài thiếu) | **608** | **21.1%** |
| Bịa đáp án cho câu bẫy | 644 | 22.3% |
| Từ chối oan câu có đáp án | 209 | 7.3% |
| Sai một span khác / sai hoàn toàn | 199 | 6.9% |

Model không thiếu kiến thức: F1 đạt **79.16** trên 1729 câu nó chịu trả lời. Điểm rơi vào hai
chỗ — **biên đáp án** (21.1%) và **không biết im lặng** (22.3%) — và đó chính là hai thứ tầng
reranker nhắm vào.

### 2. Một tầng vs hai tầng — mẫu 200 câu test (seed 42, τ = 0.95)

Đo lại đúng pipeline mà web/API dùng: `python src/batch_eval.py --n 200 --out batch_eval_n200.json`.

| Chỉ số | Một tầng | Hai tầng | Δ |
|---|---|---|---|
| EM toàn bộ | 41.50 | **46.00** | **+4.50** |
| F1 toàn bộ | 56.69 | **58.92** | +2.23 |
| EM câu có đáp án | 44.78 | 41.04 | −3.73 |
| F1 câu có đáp án | 67.45 | 60.33 | −7.12 |
| Độ chính xác câu bẫy | 34.85 | **56.06** | **+21.21** |
| Tỉ lệ bỏ trả lời | 22.0 | 35.5 | +13.50 |

Cột "một tầng" (41.50 / 56.69) khớp số toàn test (42.40 / 57.90) → mẫu 200 câu không lệch.

**Điểm trung thực:** reranker **không** tăng chất lượng chọn span. Trên val, pool 40 ứng viên
đạt recall 92.8% nhưng reranker đặc trưng bề mặt chỉ sửa được **12/821** câu mà extractor xếp
hạng sai — kiến trúc pointer chấm start/end độc lập nên lỗi cắt biên không sửa được bằng hậu
xử lý. Giá trị đo được của tầng hai là **quyền từ chối trả lời có hiệu chuẩn**: bắt câu bẫy
+21 điểm, đổi lấy −7 điểm F1 câu có đáp án.

### 3. Ngưỡng từ chối τ — quét trên val (n = 2872)

| τ | EM | F1 | Câu bẫy | F1 có đáp án |
|---|---|---|---|---|
| 0.90 | 46.41 | 57.89 | 58.27 | 57.70 |
| **0.95** (mặc định) | 45.33 | **58.76** | 50.59 | 62.72 |
| 0.97 | 43.04 | 58.07 | 36.93 | 68.31 |
| 0.99 | 40.46 | 56.33 | 24.33 | 71.83 |
| không từ chối | 33.74 | 50.52 | 0.00 | 74.98 |

### 4. Độ trễ

190–915 ms/câu trên RTX 3050 (FP16), tuỳ đoạn dài hay ngắn; 60 câu ≈ 16–22 giây.

---

## 🛠️ Installation

### Requirements
- Python 3.10+
- 4GB RAM minimum (8GB recommended)
- GPU optional (recommended for training)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Core packages:
- torch >= 2.0.0
- transformers >= 4.30.0
- datasets >= 2.14.0
- streamlit >= 1.63.0
- scikit-learn == 1.7.2 *(reranker được lưu bằng joblib từ bản này — đổi bản có thể lỗi khi nạp)*
- fastapi + uvicorn *(REST API, tuỳ chọn)*

📖 **Chi tiết cài đặt:** Xem [README_DEPLOYMENT.md](README_DEPLOYMENT.md)

---

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Hướng dẫn nhanh 3 bước
- **[README_DEPLOYMENT.md](README_DEPLOYMENT.md)** - Hướng dẫn cài đặt và deploy chi tiết
- **[SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md)** - Hướng dẫn đóng gói và nộp bài
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Tổng kết project
- **[CREATE_REPORT.md](CREATE_REPORT.md)** - Khung báo cáo chi tiết (số liệu đã điền)

Một số tài liệu khác trong repo (`ADVANCED_*.md`, `UPGRADE_*.md`, `DEMO_FEATURES.md`,
`CHECKLIST.md`) là ghi chú quá trình thử nghiệm, **không** mô tả kết quả cuối cùng — số liệu
chính thức nằm ở mục Results của README và trong `batch_eval_n200.json` / `error_analysis.json`.

---

## 🧪 Testing

Verify model trước khi chạy:

```bash
python test_model.py
```

Kiểm tra:
- ✅ Package imports
- ✅ Model loading
- ✅ Inference test

---

## 📝 Report Template

Template báo cáo chi tiết có sẵn trong [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md).

Cấu trúc báo cáo:
1. Giới thiệu task
2. Mô tả dataset
3. Kiến trúc model
4. Quá trình training
5. Kết quả evaluation
6. Web application demo
7. Thách thức và giải pháp
8. Kết luận

---

## 🔗 References

- [PhoBERT GitHub](https://github.com/VinAIresearch/PhoBERT)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [UIT-ViQuAD dataset](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0)
- [ViQuAD: A Vietnamese Dataset for Evaluating Machine Reading Comprehension (COLING 2020)](https://aclanthology.org/2020.coling-main.233/)
- [Streamlit Documentation](https://docs.streamlit.io)
- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)

---

## 👥 Team

**Final Project - Transformer-Based NLP Application**

- **Cao Trần Bá Đạt** - 23127168
- **Trần Danh Thiện** - 23127120

---

## 📄 License

This project is for educational purposes.

PhoBERT model by VinAI Research.

---

## 🙏 Acknowledgments

- VinAI Research for PhoBERT
- Hugging Face for Transformers library
- Streamlit for web framework
- ViQuAD dataset contributors

---

<div align="center">

**Made with ❤️ for Vietnamese NLP**

[Report Bug](../../issues) · [Request Feature](../../issues)

</div>
