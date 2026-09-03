# 🎯 Project Summary - Vietnamese Question Answering System

## ✅ Những gì đã hoàn thành

### 1. Model Training ✓
- **Model:** PhoBERT-base (VinAI) fine-tuned trên ViQuAD dataset
- **Task:** Extractive Question Answering
- **Location:** `models/phobert_qa/`
- **Status:** Đã train xong, có checkpoints và final model

### 2. Evaluation ✓
- **Script:** `src/evaluate.py` (toàn test set) và `src/batch_eval.py` (so một tầng / hai tầng)
- **Metrics:** Exact Match (EM), F1 Score
- **Output:** `predictions.json`
- **Kết quả đo thật trên toàn bộ test set (n = 2882, 944 câu bẫy):**

| Chỉ số | Một tầng |
|---|---|
| EM toàn bộ | 42.40 |
| F1 toàn bộ | 57.90 |
| HasAns EM / F1 | 47.57 / 70.62 |
| NoAns Accuracy | 31.78 |
| F1 khi model chịu trả lời | 79.16 |

- **Sau khi thêm tầng reranker (τ = 0.95, mẫu 200 câu test):** EM 41.50 → **46.00**,
  độ chính xác câu bẫy 34.85 → **56.06**, cái giá là F1 câu có đáp án −7.12.
- **Features:**
  - SQuAD-style evaluation
  - Unanswerable question detection có ngưỡng hiệu chuẩn
  - Sliding window post-processing

### 3. Web Application ✓
- **Framework:** Streamlit
- **File:** `app/app.py`
- **Features:**
  - Giao diện đẹp, responsive
  - Real-time inference
  - Highlight câu trả lời trong context
  - Hiển thị confidence scores
  - Phát hiện câu hỏi không có đáp án
  - Sidebar với thông tin model và hướng dẫn

### 4. Documentation ✓
Đã tạo các files:
- ✅ `README_DEPLOYMENT.md` - Hướng dẫn chi tiết cách cài đặt và chạy
- ✅ `QUICKSTART.md` - Quick start guide (3 bước)
- ✅ `SUBMISSION_GUIDE.md` - Hướng dẫn đóng gói và nộp bài
- ✅ `REPORT.md` template - Template báo cáo chi tiết
- ✅ `requirements.txt` - Dependencies list
- ✅ `.gitignore` - Git ignore rules

### 5. Helper Scripts ✓
- ✅ `test_model.py` - Test script để verify model hoạt động
- ✅ `run_app.bat` - Windows batch script để chạy app tự động
- ✅ `.streamlit/config.toml` - Streamlit configuration

---

## 📁 Cấu trúc Project

```
train_Vit/
│
├── 📱 APPLICATION
│   ├── app/app.py                    # Web application (Streamlit)
│   └── .streamlit/config.toml        # Streamlit config
│
├── 🧠 MODEL
│   └── models/phobert_qa/            # Trained PhoBERT QA model
│       ├── config.json
│       ├── model.safetensors         # Model weights (537MB, đẩy lên git qua Git LFS)
│       ├── tokenizer_config.json
│       ├── special_tokens_map.json
│       ├── vocab.txt
│       ├── bpe.codes
│       └── added_tokens.json
│
├── 🔧 SOURCE CODE
│   ├── src/train.py                  # Training pipeline (OOP)
│   ├── src/evaluate.py               # Evaluation script
│   └── src/data_preprocessing.py     # Data preprocessing
│
├── 📊 DATA
│   ├── data/raw/train_formatted.json # Raw dataset
│   └── data/processed/               # Processed datasets
│       ├── train.parquet
│       ├── val.parquet
│       └── test.parquet
│
├── 📝 DOCUMENTATION
│   ├── README_DEPLOYMENT.md          # Full deployment guide
│   ├── QUICKSTART.md                 # Quick start (3 steps)
│   ├── SUBMISSION_GUIDE.md           # Submission instructions
│   ├── PROJECT_SUMMARY.md            # This file
│   └── requirements.txt              # Python dependencies
│
├── 🧪 TESTING & UTILS
│   ├── test_model.py                 # Model verification script
│   ├── run_app.bat                   # Windows auto-run script
│   └── predictions.json              # Test predictions
│
└── ⚙️ CONFIGURATION
    ├── .gitignore                    # Git ignore rules
    └── notebooks/01_eda_and_flatten.ipynb  # EDA notebook
```

---

## 🚀 Cách chạy (Quick Reference)

### Option 1: Tự động (Windows)
```bash
run_app.bat
```

### Option 2: Thủ công
```bash
# 1. Tạo virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Cài đặt dependencies
pip install -r requirements.txt

# 3. Test model
python test_model.py

# 4. Chạy web app
streamlit run app/app.py
```

### Truy cập ứng dụng
Mở trình duyệt: `http://localhost:8501`

---

## 📊 Thông số Kỹ thuật

### Model Architecture
- **Base:** PhoBERT-base (RoBERTa variant for Vietnamese)
- **Parameters:** 134,409,218 (~134.4M)
- **Hidden size:** 768
- **Attention heads:** 12
- **Layers:** 12
- **Vocabulary:** 64,001 (`vocab_size`; tokenizer thực tế 64,000 token)

### Training Hyperparameters
```yaml
learning_rate: 3e-5
batch_size: 8 (effective 16 with gradient accumulation)
epochs: 3
max_length: 256 tokens
doc_stride: 64 tokens
warmup_ratio: 0.06
weight_decay: 0.01
fp16: true
gradient_checkpointing: true
```

### Dataset
- **Name:** ViQuAD (Vietnamese Question Answering Dataset)
- **Language:** Vietnamese
- **Format:** JSON/Parquet
- **Task:** Extractive QA with unanswerable questions

---

## 🎯 Đáp ứng yêu cầu đề tài

### ✅ Requirement 1: Train/Fine-tune Model (8 points)
- [x] Chọn task: Question Answering
- [x] Chọn model: PhoBERT (Transformer-based)
- [x] Dataset phù hợp: ViQuAD
- [x] Train/finetune model
- [x] Evaluate với metrics phù hợp (EM, F1)
- [x] Mô tả chi tiết trong báo cáo

### ✅ Requirement 2: Web Application (2 points)
- [x] Phát triển web app với Streamlit
- [x] Sử dụng model đã train
- [x] Giao diện thân thiện
- [x] Demo được functionality

### ✅ Submission Requirements
- [x] Source code đầy đủ
- [x] Trained model weights
- [x] Dataset
- [x] Báo cáo chi tiết
- [x] Đóng gói ZIP theo format

---

## 💡 Điểm nổi bật

### Technical Highlights
1. **OOP Architecture:** Code được tổ chức theo OOP với các lớp rõ ràng
2. **Custom Tokenization:** Tự implement offset mapping cho PhoBERT
3. **Sliding Window:** Xử lý context dài hiệu quả
4. **Unanswerable Detection:** Phát hiện câu hỏi không có đáp án
5. **FP16 Training:** Tối ưu memory và tốc độ
6. **Gradient Checkpointing:** Đổi tốc độ lấy memory (train được trên GPU 4GB)

### Application Highlights
1. **User-friendly UI:** Giao diện đẹp, dễ sử dụng
2. **Real-time Inference:** ~40–90 ms với context ngắn, 190–915 ms trên test set (RTX 3050 Laptop 4GB)
3. **Visual Feedback:** Highlight câu trả lời trong context
4. **Confidence Scores:** Hiển thị độ tin cậy của prediction
5. **Sample Data:** Có dữ liệu mẫu để test nhanh

### Documentation Highlights
1. **Comprehensive Guides:** 3 levels (Quick Start → Deployment → Submission)
2. **Report Template:** Template sẵn cho báo cáo
3. **Test Script:** Verify model trước khi deploy
4. **Auto-run Script:** One-click chạy app trên Windows

---

## 🔮 Hướng phát triển tiếp

### Ngắn hạn
- [ ] Deploy lên Streamlit Cloud hoặc Hugging Face Spaces
- [ ] Add authentication/user management
- [ ] Lưu lịch sử queries
- [ ] Export results to CSV/PDF

### Dài hạn
- [ ] Thử PhoBERT-large hoặc ensemble models
- [ ] Multi-turn QA (hỏi đáp hội thoại)
- [ ] Add retrieval component (RAG)
- [ ] Support multiple languages
- [ ] Fine-tune trên domain-specific data

---

## 📞 Hỗ trợ

Nếu gặp vấn đề:
1. Check `README_DEPLOYMENT.md` → Troubleshooting section
2. Run `python test_model.py` để debug
3. Kiểm tra logs trong terminal

---

## 🎓 Kết luận

Project đã hoàn thành đầy đủ các yêu cầu của Final Project:
- ✅ Transformer-based model (PhoBERT)
- ✅ NLP task (Question Answering)
- ✅ Training & Evaluation
- ✅ Web Application
- ✅ Documentation đầy đủ

**Sẵn sàng để submit! 🚀**