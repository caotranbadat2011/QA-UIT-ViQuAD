# 🇻🇳 Vietnamese Question Answering System

> Final Project - Transformer-Based NLP Application  
> Fine-tuned PhoBERT for Extractive Question Answering on ViQuAD Dataset

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30%2B-green)](https://huggingface.co/transformers/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)

---

## 🎯 Overview

Hệ thống **Question Answering** sử dụng mô hình **PhoBERT** (Transformer-based) để trích xuất câu trả lời từ đoạn văn tiếng Việt. Dự án được phát triển như Final Project cho môn học về NLP.

### ✨ Features
- 🤖 **Pre-trained PhoBERT:** Fine-tuned trên ViQuAD dataset
- 🎯 **Extractive QA:** Trích xuất câu trả lời trực tiếp từ context
- ❓ **Unanswerable Detection:** Phát hiện câu hỏi không có đáp án
- 🌐 **Web Interface:** Giao diện web đẹp với Streamlit
- ⚡ **Fast Inference:** Tối ưu với FP16 và GPU support

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
│   └── app.py             # Streamlit app
├── src/                   # Source code
│   ├── train.py           # Training pipeline
│   ├── evaluate.py        # Evaluation script
│   └── data_preprocessing.py
├── models/phobert_qa/     # Trained model
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
- **Base Model:** PhoBERT-base (VinAI Research)
- **Task:** Extractive Question Answering
- **Parameters:** ~135M
- **Framework:** Hugging Face Transformers + PyTorch

### Training Configuration
| Hyperparameter | Value |
|----------------|-------|
| Learning Rate | 3e-5 |
| Batch Size | 8 (effective 16) |
| Epochs | 3 |
| Max Length | 256 tokens |
| Optimizer | AdamW |
| Precision | FP16 |

### Dataset
- **Name:** ViQuAD (Vietnamese Question Answering Dataset)
- **Language:** Vietnamese
- **Samples:** Train/Val/Test splits
- **Format:** JSON/Parquet

---

## 💻 Usage

### Web Application

Sau khi chạy `streamlit run app/app.py`:

1. **Nhập Context:** Dán đoạn văn tiếng Việt
2. **Nhập Question:** Nhập câu hỏi về đoạn văn
3. **Click "Trả lời":** Nhận câu trả lời được highlight

**Example:**
```
Context: "Hà Nội là thủ đô của Việt Nam, nằm bên bờ sông Hồng."
Question: "Hà Nội nằm ở đâu?"
Answer: "bên bờ sông Hồng"
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
python src/evaluate.py \
    --model_dir models/phobert_qa \
    --test_file data/processed/test.parquet
```

---

## 📈 Results

### Evaluation Metrics
- **Exact Match (EM):** See evaluation logs
- **F1 Score:** See evaluation logs
- **HasAns EM/F1:** For answerable questions
- **NoAns Accuracy:** For unanswerable questions

Detailed predictions saved in `predictions.json`.

---

## 🛠️ Installation

### Requirements
- Python 3.8+
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
- streamlit >= 1.28.0
- pandas >= 2.0.0

📖 **Chi tiết cài đặt:** Xem [README_DEPLOYMENT.md](README_DEPLOYMENT.md)

---

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Hướng dẫn nhanh 3 bước
- **[README_DEPLOYMENT.md](README_DEPLOYMENT.md)** - Hướng dẫn cài đặt và deploy chi tiết
- **[SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md)** - Hướng dẫn đóng gói và nộp bài
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Tổng kết project

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
- [ViQuAD Dataset](https://github.com/VinAIresearch/PhoBERT)
- [Streamlit Documentation](https://docs.streamlit.io)
- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)

---

## 👥 Team

**Final Project - Transformer-Based NLP Application**

Student IDs: Replace with your actual student IDs

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
