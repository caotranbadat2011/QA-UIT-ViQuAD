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
- **Tên:** ViQuAD (Vietnamese Question Answering Dataset) — bản UIT-ViQuAD 2.0 có câu bẫy
- **Ngôn ngữ:** Tiếng Việt
- **Định dạng:** JSON → Parquet (`data/processed/`)
- **Số lượng (đo từ parquet, không phải ghi theo tài liệu):**

| Split | Câu hỏi | Đoạn văn | Câu không có đáp án |
|-------|---------|----------|---------------------|
| Train | 22702 | 3279 | 7336 (32.3%) |
| Validation | 2872 | 411 | 937 (32.6%) |
| Test | 2882 | 411 | 944 (32.8%) |

- **Độ dài (số từ, train):** context trung bình 181, median 162, dài nhất 1537 → bắt buộc
  dùng sliding window vì model chỉ nhận 256 token
- **Câu hỏi:** trung bình 14.6 từ (dài nhất 53)
- **Đáp án:** trung bình 9.9 từ, median 6, dài nhất 122

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
| Vocabulary | 64,001 (`vocab_size`); tokenizer thực tế 64,000 token |
| Total parameters | 134,409,218 (~134.4M) |

### 3.3 Fine-tuning strategy
- Thêm QA head (`start_logits` + `end_logits`, Linear 768→2) — khởi tạo trọng số mới
- **Full fine-tuning:** không đóng băng layer nào (`src/train.py` không có `requires_grad=False`
  hay `freeze_layers`)
- Early-stopping / save theo `eval_loss` trên val set
- Learning rate: 3e-5 với linear decay
- Batch size: 8 với gradient accumulation (effective batch=16)
- Epochs: 3
- FP16 mixed precision training
- Gradient checkpointing để tiết kiệm memory

---

## 4. Huấn luyện

### 4.1 Environment
- **Framework:** PyTorch + Hugging Face Transformers
- **GPU:** NVIDIA GeForce RTX 3050 Laptop, 4GB
- **Training time:** 58 phút 58 giây cho 3 epochs = 6621 steps (29.94 samples/giây,
  1.87 steps/giây) — nguồn `train_log.txt`
- **Validation eval:** 31.3–31.5 giây cho 2872 câu (141.3 samples/giây)
- **Max sequence length:** 256 tokens, doc_stride 64
- **Model đã dùng để báo cáo:** weights cuối epoch 3 tại `models/phobert_qa/`
  (eval_loss 1.2667). Early-stopping theo eval_loss lại chọn epoch 2 (`checkpoint-4414`,
  eval_loss 1.2203); chưa đo test set riêng cho weights epoch 2.

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
Số liệu thật từ `train_log.txt` (train loss là giá trị log gần nhất tại epoch đó):

| Epoch | train_loss | eval_loss |
|-------|------------|-----------|
| 0.02 (start) | 5.444 | – |
| 1.00 | 1.342 | 1.3073 |
| 1.99 | 0.963 | **1.2203** ← thấp nhất, checkpoint-4414 |
| 2.99 | 0.762 | 1.2667 (epoch 3.0) |
| Trung bình cả run | 1.2231 | – |

Train loss giảm liên tục nhưng eval_loss đáy ở epoch 2 rồi **tăng** ở epoch 3 → dấu hiệu
overfit nhẹ; weights được dùng để báo cáo là epoch 3.

---

## 5. Kết quả Evaluation

### 5.1 Metrics
Sử dụng SQuAD-style metrics:

| Metric | Score |
|--------|-------|
| **Exact Match (EM)** | 42.40% |
| **F1 Score** | 57.90% |
| HasAns EM | 47.57% |
| HasAns F1 | 70.62% |
| NoAns Accuracy | 31.78% |

Toàn bộ test set, n = 2882 câu (1938 có đáp án / 944 câu bẫy), model một tầng.

Bật tầng reranker (τ = 0.95) trên mẫu 200 câu test cùng pipeline với web/API, seed 42:

| Metric | Một tầng | Hai tầng | Δ |
|--------|----------|----------|---|
| EM toàn bộ | 41.50 | **46.00** | +4.50 |
| F1 toàn bộ | 56.69 | **58.92** | +2.23 |
| Độ chính xác câu bẫy | 34.85 | **56.06** | +21.21 |
| F1 câu có đáp án | 67.45 | 60.33 | −7.12 |

### 5.2 Phân tích kết quả
- **Điểm mạnh:** F1 đạt **79.16** trên 1729 câu model chịu trả lời — nó tìm đúng vùng chứa
  thông tin; chỉ khi nào ép trả lời câu bẫy mới bịa.
- **Điểm yếu:** 644/944 câu bẫy (**68.2%**) bị bịa đáp án; biên đáp án cắt chưa chuẩn.
- **Error analysis** — 2882 câu chia thành 6 nhóm (tính từ `predictions.json` bằng
  `normalize_text / compute_exact / compute_f1` trong `src/evaluate.py`):

| Nhóm lỗi | Câu | % test |
|---|---|---|
| Trả lời đúng (EM) | 922 | 32.0% |
| Bỏ đúng câu bẫy | 300 | 10.4% |
| Đúng chỗ, cắt sai biên (375 dài thừa + 233 dài thiếu) | 608 | **21.1%** |
| Bịa đáp án cho câu bẫy | 644 | **22.3%** |
| Từ chối oan câu có đáp án | 209 | 7.3% |
| Sai span khác / sai hoàn toàn | 199 | 6.9% |

  Kết luận: hai lỗi lớn nhất là **biên đáp án** và **không biết im lặng**, không phải thiếu
  kiến thức — đó là lý do tầng reranker tập trung vào quyền từ chối trả lời.

### 5.3 So sánh với baseline
Bài này **không train thêm backbone nào khác** (mBERT/XLM-R/PhoBERT-large) nên không có số
đối chiếu ngoài, và cũng không trích số từ paper khác làm "baseline tự chạy". Toàn bộ so sánh
đều *trong cùng một run trên cùng dữ liệu của nhóm*:
- một tầng vs hai tầng trên cùng mẫu test (bảng 5.1),
- đường cong ngưỡng từ chối τ đo trên val (README, mục Results),
- cột "một tầng" của mẫu 200 câu (41.50/56.69) khớp số toàn test (42.40/57.90) → pipeline
  web/API không làm lệch kết quả.

---

## 6. Web Application

### 6.1 Công nghệ
- **Frontend:** Streamlit
- **Backend:** Python + PyTorch
- **Deployment:** Local/Cloud (tùy chọn)

### 6.2 Tính năng
**Tab 1 — Hỏi đáp trên đoạn văn của bạn**
- Nhập context + question, trả lời trích xuất kèm highlight vị trí trong đoạn
- Độ tin cậy hiệu chuẩn và danh sách mọi phương án đã cân nhắc (kèm điểm của từng phương án)
- Quyền "từ chối trả lời" với ngưỡng τ điều chỉnh được bằng thanh *Độ khắt khe*

**Tab 2 — Đánh giá hàng loạt trên test set**
- Chạy thật trên mẫu câu hỏi test (seed cố định 42) bằng đúng pipeline Tab 1
- Bảng một tầng / hai tầng + Δ, đường cong đánh đổi của ngưỡng từ chối
- Kết luận từng câu: câu nào reranker sửa được, câu nào nó làm hỏng, câu nào cả hai cùng sai
- Xuất CSV toàn bộ kết quả

Kèm theo: `src/api.py` — REST API dùng chung `QAService` với web app.

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
4. **Gradient checkpointing + FP16:** đủ để train trên GPU laptop 4GB (RTX 3050)

---

## 8. Kết luận

### 8.1 Đạt được
- ✅ Fine-tune thành công PhoBERT-base (134.4M tham số) trên ViQuAD, 3 epochs
- ✅ Đạt **42.40% EM / 57.90% F1** trên toàn bộ test set (n = 2882)
- ✅ Xây dựng web app demo hoàn chỉnh, có tab tự chấm điểm trên test set thật
- ✅ Abstention có hiệu chuẩn: độ chính xác câu bẫy 34.85 → **56.06** trên mẫu 200 câu
      (τ = 0.95), đổi lấy −7.12 F1 câu có đáp án — đo được, không phải cảm tính
- ✅ Chỉ ra bằng số lý do kiến trúc pointer (start/end độc lập) không sửa được lỗi biên:
      pool 40 ứng viên recall 92.8% nhưng reranker đặc trưng bề mặt chỉ cứu 12/821 câu

### 8.2 Hướng phát triển
- **Đổi kiến trúc đầu ra, không đổi backbone:** thay pointer bằng span-based exhaustive scorer
  (chấm điểm từng cặp (start, end) thay vì cộng điểm start + end độc lập). Đây là hướng duy
  nhất còn lại sau khi đã loại trừ bằng thực nghiệm: reranking đặc trưng bề mặt, trim từ thừa
  hai đầu, và chỉnh ngưỡng đều không sửa được lỗi biên (mục 5.2).
- Huấn luyện lại trên chính pool 40 ứng viên đã có (`src/dump_candidates.py`) — chi phí
  ~13 phút/epoch trên RTX 3050, chỉ đáng làm nếu pilot với encoder đóng cho thấy span scorer
  thắng pointer.
- Deploy cloud (Streamlit Cloud / Hugging Face Spaces) — cần nguồn trọng số 537MB riêng vì
  git không chứa chúng.

---

## 9. Tài liệu tham khảo

1. PhoBERT: https://github.com/VinAIresearch/PhoBERT
2. UIT-ViQuAD 2.0: https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0
3. ViQuAD: A Vietnamese Dataset for Evaluating Machine Reading Comprehension, COLING 2020:
   https://aclanthology.org/2020.coling-main.233/
4. Hugging Face Transformers: https://huggingface.co/docs/transformers
5. SQuAD: https://rajpurkar.github.io/SQuAD-explorer/
6. "Attention Is All You Need" (Vaswani et al., 2017)

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
