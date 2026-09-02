# Vietnamese Question Answering System - Deployment Guide

## 📋 Tổng quan

Đây là ứng dụng web demo cho mô hình **PhoBERT** fine-tuned trên task **Extractive Question Answering** (trích xuất câu trả lời từ đoạn văn). Dự án được phát triển như một phần của Final Project - Transformer-Based NLP Application.

### Kiến trúc mô hình
- **Base Model:** PhoBERT-base (VinAI)
- **Task:** Extractive Question Answering
- **Dataset:** ViQuAD (Vietnamese Question Answering Dataset)
- **Framework:** Hugging Face Transformers + PyTorch
- **Web Interface:** Streamlit

---

## 🚀 Cài đặt và Chạy Ứng dụng

### Yêu cầu hệ thống
- Python 3.8 trở lên
- RAM tối thiểu: 4GB (khuyến nghị 8GB)
- GPU (optional, nhưng khuyến nghị để tăng tốc độ inference)

### Bước 1: Clone hoặc chuẩn bị source code

Đảm bảo bạn có cấu trúc thư mục sau:
```
train_Vit/
├── app/
│   └── app.py              # Web application
├── models/
│   └── phobert_qa/         # Trained model checkpoint
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       └── ...
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── data_preprocessing.py
├── requirements.txt        # Dependencies
└── README_DEPLOYMENT.md    # File này
```

### Bước 2: Tạo virtual environment (khuyến nghị)

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
# Trên Windows:
venv\Scripts\activate
# Trên Linux/Mac:
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

Hoặc cài đặt thủ công:
```bash
pip install torch transformers datasets pandas numpy streamlit sentencepiece
```

### Bước 4: Chạy ứng dụng web

```bash
streamlit run app/app.py
```

Ứng dụng sẽ chạy tại: `http://localhost:8501`

Mở trình duyệt và truy cập địa chỉ trên để sử dụng ứng dụng.

---

## 📊 Sử dụng Ứng dụng

### Giao diện chính

1. **Nhập Context (Đoạn văn):** Dán đoạn văn bản tiếng Việt chứa thông tin cần trích xuất
2. **Nhập Question (Câu hỏi):** Nhập câu hỏi về nội dung đoạn văn
3. **Nhấn "Trả lời":** Hệ thống sẽ xử lý và hiển thị câu trả lời

### Ví dụ minh họa

**Context:**
```
Hà Nội là thủ đô của Việt Nam, nằm bên bờ sông Hồng. Thành phố này có lịch sử hơn 1000 năm và là trung tâm chính trị, kinh tế, văn hóa của cả nước.
```

**Question:**
```
Hà Nội nằm ở đâu?
```

**Kết quả thật** (chạy `python src/qa_service.py` qua Tab 1, đã kiểm chứng):
```
nằm bên bờ sông Hồng.
```
Model trả lời kèm chữ "nằm" và dấu chấm — đây chính là dạng "cắt sai biên" được thống kê
ở mục Results (21.1% test set). Đáp án vàng của ViQuAD cho câu này là "bên bờ sông Hồng".

---

## 🔧 Huấn luyện Mô hình (Optional)

Nếu bạn muốn train lại mô hình:

### Bước 1: Chuẩn bị dữ liệu

```bash
# Dữ liệu đã được tiền xử lý sẵn trong thư mục data/processed/
# Nếu cần xử lý lại từ raw data:
python src/data_preprocessing.py
```

### Bước 2: Train mô hình

```bash
python src/train.py \
    --train_file data/processed/train.parquet \
    --val_file data/processed/val.parquet \
    --model_name vinai/phobert-base \
    --output_dir models/phobert_qa \
    --max_length 256 \
    --num_train_epochs 3 \
    --batch_size 8 \
    --learning_rate 3e-5
```

### Bước 3: Đánh giá mô hình

```bash
python src/evaluate.py \
    --model_dir models/phobert_qa \
    --test_file data/processed/test.parquet
```

Kết quả đánh giá sẽ được lưu vào `predictions.json`.

---

## 📈 Kết quả Đánh giá

Đo thật trên **toàn bộ test set** (n = 2882, 944 câu không có đáp án), model một tầng:

| Metric | Điểm |
|--------|------|
| **Exact Match (EM)** | **42.40** |
| **F1 Score** | **57.90** |
| HasAns EM | 47.57 |
| HasAns F1 | 70.62 |
| NoAns Accuracy | 31.78 |
| F1 riêng khi model chịu trả lời | 79.16 |

Bật thêm tầng reranker với ngưỡng từ chối τ = 0.95, đo trên mẫu 200 câu test (seed 42):
EM 41.50 → **46.00**, độ chính xác câu bẫy 34.85 → **56.06**, cái giá là F1 câu có đáp án
−7.12. Chạy lại: `python src/batch_eval.py --n 200 --out batch_eval_n200.json`.

Chi tiết predictions một tầng được lưu trong file `predictions.json`.

---

## 🏗️ Cấu trúc Code

### `app/app.py`
- Web application sử dụng Streamlit
- Load model và thực hiện inference
- Giao diện người dùng đẹp, dễ sử dụng

### `src/train.py`
- OOP architecture với các lớp:
  - `QADataLoader`: Đọc và chuẩn hóa dữ liệu
  - `QATokenizerProcessor`: Tokenization với offset mapping
  - `QATrainerPipeline`: Quản lý quá trình training
- Fine-tuning PhoBERT trên ViQuAD dataset
- Hỗ trợ unanswerable questions

### `src/evaluate.py`
- Đánh giá mô hình trên test set
- Tính toán metrics: Exact Match (EM), F1 Score
- Xử lý sliding window predictions
- Lưu predictions vào JSON

### `src/data_preprocessing.py`
- Tiền xử lý dữ liệu từ JSON sang Parquet
- Format dữ liệu theo chuẩn SQuAD

---

## ⚙️ Thông số Kỹ thuật

### Hyperparameters
- **Learning rate:** 3e-5
- **Batch size:** 8 (với gradient accumulation steps = 2)
- **Max sequence length:** 256 tokens
- **Doc stride:** 64 tokens
- **Epochs:** 3
- **Optimizer:** AdamW
- **FP16:** Enabled
- **Gradient checkpointing:** Enabled

### Model Architecture
- **Hidden size:** 768
- **Attention heads:** 12
- **Layers:** 12
- **Vocabulary size:** 64,001 (`config.json`); tokenizer báo 64,000 token
- **Position embeddings:** Absolute (max 258)

---

## 🐛 Troubleshooting

### Lỗi: "Không thể tải mô hình"
- Kiểm tra đường dẫn `MODEL_DIR` trong `app/app.py`
- Đảm bảo thư mục `models/phobert_qa/` tồn tại và chứa đầy đủ files

### Lỗi: CUDA out of memory
- Giảm `batch_size` trong `src/train.py`
- Giảm `max_length` nếu cần
- Hoặc chạy trên CPU (chậm hơn)

### Lỗi: Module not found
- Đảm bảo đã cài đặt đầy đủ dependencies: `pip install -r requirements.txt`
- Kiểm tra virtual environment đã được kích hoạt

### Streamlit không chạy
- Kiểm tra port 8501 có bị chiếm không
- Thử chạy với port khác: `streamlit run app/app.py --server.port 8502`

---

## 📝 License & Credits

- **PhoBERT:** VinAI Research
- **Transformers:** Hugging Face
- **Streamlit:** Streamlit Inc.
- **Dataset:** ViQuAD (Vietnamese Question Answering Dataset)

---

## 👥 Liên hệ

Nếu có thắc mắc hoặc góp ý, vui lòng liên hệ nhóm phát triển.

---

**Chúc bạn sử dụng ứng dụng thành công! 🎉**
