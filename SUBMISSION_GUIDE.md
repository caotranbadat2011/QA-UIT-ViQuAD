# 📦 Hướng dẫn Đóng gói và Nộp Bài

## ✅ Checklist trước khi nộp

Trước khi đóng gói project, đảm bảo bạn đã hoàn thành các mục sau:

### 1. Model đã được train xong
- [ ] Thư mục `models/phobert_qa/` chứa đầy đủ files:
  - `config.json`
  - `model.safetensors` (hoặc `pytorch_model.bin`)
  - `tokenizer_config.json`
  - `special_tokens_map.json`
  - `vocab.txt`
  - `bpe.codes`
  - `added_tokens.json`

### 2. Đã chạy evaluation
- [ ] File `predictions.json` đã được tạo từ script `evaluate.py`
- [ ] Ghi lại metrics trong báo cáo:
  - Exact Match (EM) score
  - F1 Score
  - HasAns EM/F1 (nếu có)
  - NoAns Accuracy (nếu có)

### 3. Web app hoạt động
- [ ] Chạy thử: `streamlit run app/app.py`
- [ ] Test với ít nhất 3-5 câu hỏi khác nhau
- [ ] Đảm bảo model load thành công và trả lời đúng

### 4. Code sạch sẽ
- [ ] Xóa các file tạm, cache (`__pycache__/`, `.pyc`)
- [ ] Không commit file nhạy cảm (.env, credentials)
- [ ] README rõ ràng, dễ hiểu

---

## 📁 Cấu trúc thư mục nộp bài

```
StudentID1_StudentID2.zip
│
├── README_DEPLOYMENT.md          # Hướng dẫn cài đặt và chạy
├── QUICKSTART.md                 # Quick start guide
├── requirements.txt              # Dependencies
│
├── app/
│   ├── app.py                    # Web application
│   └── __init__.py               # (optional)
│
├── src/
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   └── data_preprocessing.py     # Data preprocessing
│
├── models/
│   └── phobert_qa/               # Trained model (BEST checkpoint)
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       ├── special_tokens_map.json
│       ├── vocab.txt
│       ├── bpe.codes
│       └── added_tokens.json
│
├── data/
│   ├── raw/                      # Raw dataset (nếu nhỏ)
│   │   └── train_formatted.json
│   └── processed/                # Processed datasets
│       ├── train.parquet
│       ├── val.parquet
│       └── test.parquet
│
├── notebooks/
│   └── 01_eda_and_flatten.ipynb  # EDA notebook (nếu có)
│
├── predictions.json              # Test predictions
├── test_model.py                 # Test script
│
└── REPORT.md                     # Báo cáo chi tiết (xem template bên dưới)
```

---

## 📝 Template Báo cáo (REPORT.md)

```markdown
# BÁO CÁO FINAL PROJECT
# Transformer-Based NLP Application

## Thông tin nhóm
- **Họ tên sinh viên 1:** [Tên] - MSSV: [StudentID1]
- **Họ tên sinh viên 2:** [Tên] - MSSV: [StudentID2]
- **Email:** [email@example.com]

---

## 1. Giới thiệu Task

### 1.1 Task lựa chọn
**Question Answering (QA)** - Trích xuất câu trả lời từ đoạn văn

### 1.2 Mô tả
Hệ thống nhận đầu vào là một đoạn văn (context) và một câu hỏi (question), 
sau đó trích xuất câu trả lời trực tiếp từ đoạn văn. Hỗ trợ phát hiện 
các câu hỏi không có đáp án trong context.

### 1.3 Ứng dụng thực tế
- Chatbot hỗ trợ khách hàng
- Hệ thống tra cứu thông tin tự động
- Trợ lý ảo thông minh

---

## 2. Dataset

### 2.1 Mô tả dataset
- **Tên:** ViQuAD (Vietnamese Question Answering Dataset)
- **Ngôn ngữ:** Tiếng Việt
- **Định dạng:** JSON/Parquet
- **Số lượng samples:**
  - Train: [số lượng]
  - Validation: [số lượng]
  - Test: [số lượng]

### 2.2 Cấu trúc dữ liệu
Mỗi sample gồm:
- `id`: Định danh duy nhất
- `question`: Câu hỏi tiếng Việt
- `context`: Đoạn văn chứa thông tin
- `answers`: Danh sách đáp án (text + vị trí bắt đầu)
- `is_impossible`: Flag cho câu hỏi không có đáp án (optional)

### 2.3 Tiền xử lý
- Làm sạch text, chuẩn hóa Unicode
- Tokenization với PhoBERT tokenizer
- Sliding window cho context dài (max_length=256, doc_stride=64)
- Mapping character offsets sang token positions

---

## 3. Mô hình

### 3.1 Kiến trúc lựa chọn
**PhoBERT-base** (VinAI Research)

**Lý do chọn:**
- Pre-trained trên corpus tiếng Việt lớn
- Đạt SOTA trên nhiều benchmarks tiếng Việt
- Phù hợp với task extractive QA
- Community support tốt từ Hugging Face

### 3.2 Thông số mô hình
| Parameter | Value |
|-----------|-------|
| Architecture | RoBERTa-based |
| Hidden size | 768 |
| Attention heads | 12 |
| Layers | 12 |
| Vocabulary | 64,001 tokens |
| Total parameters | ~135M |

### 3.3 Fine-tuning strategy
- Thêm layer QA head (start_logits + end_logits)
- Freeze embedding layer (optional)
- Learning rate: 3e-5 với linear decay
- Batch size: 8 với gradient accumulation (effective batch=16)
- Epochs: 3
- FP16 mixed precision training
- Gradient checkpointing để tiết kiệm memory

---

## 4. Huấn luyện

### 4.1 Environment
- **Framework:** PyTorch + Hugging Face Transformers
- **GPU:** [Loại GPU, VRAM nếu có]
- **Training time:** [X giờ/phút]
- **Max sequence length:** 256 tokens

### 4.2 Hyperparameters
```python
learning_rate = 3e-5
batch_size = 8
gradient_accumulation_steps = 2
num_train_epochs = 3
max_length = 256
doc_stride = 64
warmup_ratio = 0.06
weight_decay = 0.01
```

### 4.3 Loss curves
[Chèn biểu đồ training/validation loss nếu có]

---

## 5. Kết quả Evaluation

### 5.1 Metrics
Sử dụng SQuAD-style metrics:

| Metric | Score |
|--------|-------|
| **Exact Match (EM)** | [XX.XX]% |
| **F1 Score** | [XX.XX]% |
| HasAns EM | [XX.XX]% |
| HasAns F1 | [XX.XX]% |
| NoAns Accuracy | [XX.XX]% |

### 5.2 Phân tích kết quả
- **Điểm mạnh:** [Mô tả]
- **Điểm yếu:** [Mô tả]
- **Error analysis:** Các trường hợp model dự đoán sai phổ biến

### 5.3 So sánh với baseline
[Nếu có] So sánh với các model khác hoặc results từ paper

---

## 6. Web Application

### 6.1 Công nghệ
- **Frontend:** Streamlit
- **Backend:** Python + PyTorch
- **Deployment:** Local/Cloud (tùy chọn)

### 6.2 Tính năng
- Nhập context và question
- Hiển thị câu trả lời với highlight
- Phát hiện câu hỏi không có đáp án
- Hiển thị confidence scores

### 6.3 Demo screenshots
[Chèn ảnh chụp màn hình ứng dụng]

---

## 7. Thách thức và Giải pháp

### 7.1 Thách thức
1. **Tokenization phức tạp:** PhoBERT sử dụng BPE, khó map ngược lại character positions
2. **Context dài:** Vượt quá max_length của model
3. **Unanswerable questions:** Cần phát hiện khi không có đáp án
4. **Memory constraints:** Model lớn cần tối ưu memory

### 7.2 Giải pháp
1. **Custom offset mapping:** Tự implement mapping từ tokens sang characters
2. **Sliding window:** Chia context thành các chunks overlap
3. **Null score threshold:** So sánh score của best answer với no-answer score
4. **Gradient checkpointing + FP16:** Giảm memory usage ~50%

---

## 8. Kết luận

### 8.1 Đạt được
- ✅ Fine-tune thành công PhoBERT trên ViQuAD dataset
- ✅ Đạt [XX.XX]% EM và [XX.XX]% F1 trên test set
- ✅ Xây dựng web app demo hoàn chỉnh
- ✅ Hỗ trợ unanswerable question detection

### 8.2 Hướng phát triển
- Thử nghiệm với PhoBERT-large hoặc multilingual models
- Ensemble nhiều models để cải thiện accuracy
- Deploy lên cloud (Streamlit Cloud, Hugging Face Spaces)
- Add feature multi-turn QA (hỏi đáp hội thoại)

---

## 9. Tài liệu tham khảo

1. PhoBERT: https://github.com/VinAIresearch/PhoBERT
2. ViQuAD Dataset: [Link nếu có]
3. Hugging Face Transformers: https://huggingface.co/docs/transformers
4. SQuAD: https://rajpurkar.github.io/SQuAD-explorer/
5. "Attention Is All You Need" (Vaswani et al., 2017)

---

## Phụ lục: Cách chạy code

Xem file `README_DEPLOYMENT.md` và `QUICKSTART.md`
```

---

## 🗜️ Đóng gói ZIP

### Cách 1: Sử dụng command line (Windows)

```bash
# Tạo file txt chứa student IDs
echo StudentID1 > StudentID1_StudentID2.txt
echo StudentID2 >> StudentID1_StudentID2.txt

# Nén toàn bộ project
powershell Compress-Archive -Path . -DestinationPath StudentID1_StudentID2.zip -Force
```

### Cách 2: Sử dụng 7-Zip/WinRAR
1. Chọn tất cả files/folders cần nén
2. Right-click → Add to archive
3. Đặt tên: `StudentID1_StudentID2.zip`
4. Chọn compression level: Normal

### Cách 3: Nếu file quá lớn (>100MB)

Nén model riêng:
```bash
# Nén model folder
powershell Compress-Archive -Path models -DestinationPath models.zip -Force

# Upload lên Google Drive
# Tạo file StudentID1_StudentID2.txt chứa link Google Drive
echo https://drive.google.com/file/d/YOUR_FILE_ID/view > StudentID1_StudentID2.txt
```

---

## ✅ Kiểm tra cuối cùng trước khi nộp

- [ ] File ZIP có tên đúng format: `StudentID1_StudentID2.zip`
- [ ] Bên trong ZIP có đầy đủ:
  - Source code (app/, src/)
  - Model weights (models/phobert_qa/)
  - Dataset (data/)
  - Báo cáo (REPORT.md)
  - README (README_DEPLOYMENT.md)
- [ ] Test lại bằng cách:
  1. Extract ZIP vào thư mục mới
  2. Cài đặt dependencies: `pip install -r requirements.txt`
  3. Chạy app: `streamlit run app/app.py`
  4. Test với vài câu hỏi

---

## 📤 Nộp bài

### Option 1: Upload trực tiếp
- Upload file ZIP lên hệ thống LMS của trường

### Option 2: Google Drive
1. Upload ZIP lên Google Drive
2. Set sharing permission: "Anyone with the link can view"
3. Copy link
4. Tạo file `.txt` chứa link
5. Nộp file `.txt` đó

---

**Good luck! 🎓**
