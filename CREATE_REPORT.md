# 📝 Hướng dẫn tạo REPORT.md

Đây là guide nhanh để tạo báo cáo final project. Sử dụng template từ `SUBMISSION_GUIDE.md` và điền thông tin của bạn vào.

---

## ⚡ Quick Method (5 phút)

### Step 1: Copy Template

```bash
# Tạo file REPORT.md mới
touch REPORT.md

# Hoặc copy từ template nếu có
cp SUBMISSION_GUIDE.md REPORT_TEMPLATE.md
```

### Step 2: Điền thông tin cơ bản

Mở `REPORT.md` và điền các thông tin sau:

```markdown
# BÁO CÁO FINAL PROJECT
# Transformer-Based NLP Application - Question Answering

## Thông tin nhóm
- **Họ tên sinh viên 1:** [ĐIỀN TÊN] - MSSV: [ĐIỀN MSSV]
- **Họ tên sinh viên 2:** [ĐIỀN TÊN] - MSSV: [ĐIỀN MSSV]
- **Email:** [ĐIỀN EMAIL]
```

### Step 3: Chạy Evaluation để lấy metrics

```bash
python src/evaluate.py --model_dir models/phobert_qa --test_file data/processed/test.parquet
```

Kết quả đã đo (toàn bộ test set, n = 2882):

```markdown
### 5.1 Metrics

| Metric | Score |
|--------|-------|
| **Exact Match (EM)** | 42.40% |
| **F1 Score** | 57.90% |
```

### Step 4: Chụp ảnh màn hình Web App

1. Chạy app: `streamlit run app/app.py`
2. Chụp 2-3 screenshots:
   - Màn hình chính
   - Ví dụ trả lời đúng
   - Ví dụ phát hiện không có đáp án

Chèn vào report:
```markdown
### 6.3 Demo screenshots

![Main Interface](screenshots/main.png)
![Answer Example](screenshots/answer.png)
```

### Step 5: Done! ✅

---

## 📋 Full Template (Copy-Paste Ready)

Dưới đây là template đầy đủ. Copy toàn bộ và paste vào `REPORT.md`, sau đó thay thế các phần trong ngoặc `[...]`.

```markdown
# BÁO CÁO FINAL PROJECT
# Transformer-Based NLP Application

## Task: Vietnamese Question Answering với PhoBERT

---

## 1. Giới thiệu

### 1.1 Thành viên nhóm
- **Sinh viên 1:** [Họ tên] - MSSV: [xxxxxxx] - Email: [email@example.com]
- **Sinh viên 2:** [Họ tên] - MSSV: [xxxxxxx] - Email: [email@example.com]

### 1.2 Task lựa chọn
**Question Answering (QA)** - Hệ thống trả lời câu hỏi dựa trên đoạn văn tiếng Việt.

### 1.3 Mục tiêu
Xây dựng hệ thống có thể:
- Nhận đầu vào là một đoạn văn (context) và câu hỏi (question)
- Trích xuất câu trả lời trực tiếp từ đoạn văn
- Phát hiện khi câu hỏi không có đáp án trong đoạn văn

### 1.4 Ứng dụng thực tế
- Chatbot hỗ trợ khách hàng tự động
- Hệ thống tra cứu thông tin thông minh
- Trợ lý ảo cho doanh nghiệp
- Hệ thống FAQ tự động

---

## 2. Dataset

### 2.1 Mô tả
- **Tên dataset:** ViQuAD (Vietnamese Question Answering Dataset), bản UIT-ViQuAD 2.0 có câu bẫy
- **Ngôn ngữ:** Tiếng Việt
- **Nguồn:** VinAI Research / Cộng đồng NLP tiếng Việt

### 2.2 Thống kê (đo từ `data/processed/*.parquet`)
| Split | Câu hỏi | Đoạn văn | Câu không có đáp án |
|-------|---------|----------|---------------------|
| Train | 22702 | 3279 | 7336 (32.3%) |
| Validation | 2872 | 411 | 937 (32.6%) |
| Test | 2882 | 411 | 944 (32.8%) |

Độ dài tính theo từ (train): context TB 181 · median 162 · dài nhất 1537; câu hỏi TB 14.6 ·
dài nhất 53; đáp án TB 9.9 · median 6 · dài nhất 122. Context vượt xa cửa sổ 256 token của
model nên bắt buộc dùng sliding window.

### 2.3 Cấu trúc dữ liệu
Mỗi sample gồm:
```json
{
  "id": "unique_id",
  "question": "Câu hỏi tiếng Việt?",
  "context": "Đoạn văn chứa thông tin...",
  "answers": {
    "text": ["câu trả lời"],
    "answer_start": [vị trí bắt đầu]
  }
}
```

### 2.4 Tiền xử lý
Các bước đã thực hiện:
1. Làm sạch text, chuẩn hóa Unicode
2. Convert JSON sang Parquet format
3. Xử lý missing values
4. Format answers theo chuẩn SQuAD
5. Split train/val/test

Code tiền xử lý: `src/data_preprocessing.py`

---

## 3. Mô hình

### 3.1 Kiến trúc lựa chọn
**PhoBERT-base** (VinAI Research)

**Lý do chọn:**
1. Pre-trained trên corpus tiếng Việt lớn (tin tức + Wikipedia của VinAI)
2. Đạt SOTA trên nhiều benchmarks tiếng Việt
3. Dựa trên RoBERTa architecture - proven effective for QA
4. Community support tốt, dễ sử dụng với Hugging Face
5. Phù hợp với tài nguyên compute available

### 3.2 Thông số kỹ thuật
Lấy từ `models/phobert_qa/config.json` và `training_args.bin` (không phải số của bài báo gốc):

| Parameter | Value |
|-----------|-------|
| Architecture | RoBERTa-base variant (`RobertaForQuestionAnswering`) |
| Hidden size | 768 |
| Attention heads | 12 |
| Transformer layers | 12 |
| Intermediate size | 3072 |
| Vocabulary size | 64,001 (`vocab_size` trong config.json; tokenizer thực tế 64,000) |
| Max position embeddings | 258 |
| Total parameters | 134,409,218 (~134.4M, gồm 2 linear head start/end) |
| Trọng số FP32 | 537,660,792 bytes (537MB) |
| Seed | 42 |

### 3.3 Fine-tuning Strategy

**QA Head:**
- Thêm 2 linear layers cho start_logits và end_logits
- Output: vị trí bắt đầu và kết thúc của answer span

**Training configuration:**
```python
learning_rate = 3e-5          # Standard for fine-tuning
batch_size = 8                # Limited by GPU memory
gradient_accumulation = 2     # Effective batch size = 16
epochs = 3                    # Prevent overfitting
max_length = 256              # Balance accuracy and speed
doc_stride = 64               # Overlap for sliding window
warmup_ratio = 0.06           # Linear warmup
weight_decay = 0.01           # Regularization
fp16 = True                   # Mixed precision training
gradient_checkpointing = True # Save memory
```

**Optimizer:** AdamW với linear learning rate decay

**Loss function:** CrossEntropyLoss cho start và end positions

---

## 4. Quá trình Huấn luyện

### 4.1 Environment
- **Framework:** PyTorch + Hugging Face Transformers 4.57.1
- **GPU:** NVIDIA GeForce RTX 3050 Laptop, 4GB VRAM
- **Training time:** 3537.8 giây = **58 phút 58 giây** cho 6621 steps / 3 epochs
  (29.94 samples/giây, 1.872 steps/giây) — nguồn `train_log.txt`
- **Validation eval:** 30.8–31.5 giây cho 2872 câu (141–145 samples/giây)
- **Storage:** trọng số FP32 537MB; bản FP16 khi load vào GPU chiếm ~269MB
- **Suy luận web app:** 190–915 ms/câu (RTX 3050, FP16), tuỳ độ dài đoạn văn

### 4.2 Command chạy training
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

### 4.3 Training Logs
Số thật từ `train_log.txt` (train loss = giá trị log gần nhất của epoch):

| Epoch | train_loss | eval_loss |
|-------|------------|-----------|
| 0.02 | 5.444 | – |
| 1.00 | 1.342 | 1.3073 |
| 1.99 | 0.963 | **1.2203** ← thấp nhất, `checkpoint-4414` |
| 2.99 | 0.762 | 1.2667 (epoch 3.0) |
| Trung bình cả run | 1.2231 | – |

- Training loss giảm đều qua 3 epochs (5.44 → 0.76).
- **Eval_loss chạm đáy ở epoch 2 rồi TĂNG ở epoch 3** → overfit nhẹ. Handler của Trainer
  đã chọn `checkpoint-4414` (epoch 2) làm best, nhưng model được load để báo cáo là
  `models/phobert_qa/` (epoch 3). Đây là một hạn chế đã biết: chưa đo lại test set với
  weights epoch 2.
- Vẽ biểu đồ: đọc `train_log.txt` rồi plot `loss` và `eval_loss` theo `epoch`
  (đoạn code ở mục "Biểu đồ" phía dưới).

---

## 5. Kết quả Evaluation

### 5.1 Metrics sử dụng
Sử dụng SQuAD-style evaluation metrics:

1. **Exact Match (EM):** % predictions khớp chính xác với gold answer
2. **F1 Score:** Harmonic mean của precision và recall ở token level
3. **HasAns EM/F1:** Metrics riêng cho câu hỏi có đáp án
4. **NoAns Accuracy:** Accuracy phát hiện câu hỏi không có đáp án

### 5.2 Results

**Command chạy evaluation:**
```bash
python src/evaluate.py \
    --model_dir models/phobert_qa \
    --test_file data/processed/test.parquet
```

**Kết quả — toàn bộ test set, n = 2882 (model một tầng):**

| Metric | Score |
|--------|-------|
| **Overall Exact Match (EM)** | 42.40% |
| **Overall F1 Score** | 57.90% |
| HasAns EM | 47.57% |
| HasAns F1 | 70.62% |
| NoAns Accuracy | 31.78% |
| F1 riêng trên 1729 câu model chịu trả lời | 79.16% |

**Kết quả — bật tầng reranker, τ = 0.95, mẫu 200 câu test (seed 42):**

| Metric | Một tầng | Hai tầng | Δ |
|--------|----------|----------|---|
| EM toàn bộ | 41.50 | **46.00** | +4.50 |
| F1 toàn bộ | 56.69 | **58.92** | +2.23 |
| Độ chính xác câu bẫy | 34.85 | **56.06** | +21.21 |
| F1 câu có đáp án | 67.45 | 60.33 | −7.12 |
| Tỉ lệ bỏ trả lời | 22.0% | 35.5% | +13.50 |

Chạy lại: `python src/batch_eval.py --n 200 --out batch_eval_n200.json`.

### 5.3 Phân tích kết quả

**Điểm mạnh:**
- F1 79.16 khi model chịu trả lời → model định vị đúng vùng chứa thông tin trong đoạn văn.
- Sliding window 256/64 phủ được những đoạn văn dài tới 1537 từ bằng nhiều cửa sổ overlap.
- Pool 40 ứng viên của reranker chứa đáp án đúng trong **92.8%** câu có đáp án (đo trên val).

**Điểm yếu:**
- Cắt biên đáp án chưa chuẩn.
- Nói khi đáng ra phải im lặng: phần lớn câu bẫy vẫn bị gán đáp án.
- Reranker không cải thiện được việc *chọn* span — chỉ cải thiện việc *từ chối*.

**Error Analysis** — 6 nhóm, tính từ `predictions.json` bằng `normalize_text / compute_exact /
compute_f1` trong `src/evaluate.py`:

| Nhóm lỗi | Câu | % test |
|---|---|---|
| Trả lời đúng (EM) | 922 | 32.0% |
| Bỏ đúng câu bẫy | 300 | 10.4% |
| **Đúng chỗ, cắt sai biên** (375 dài thừa + 233 dài thiếu) | **608** | **21.1%** |
| **Bịa đáp án cho câu bẫy** | **644** | **22.3%** |
| Từ chối oan câu có đáp án | 209 | 7.3% |
| Sai span khác / sai hoàn toàn | 199 | 6.9% |

Kết luận quan trọng nhất: model **không thiếu kiến thức**, nó mất điểm ở hai việc cụ thể là
*biên đáp án* và *biết khi nào nên im lặng*. Đây là lý do tầng thứ hai được xây để chấm
"độ tin không có đáp án" thay vì cố đoán giỏi hơn.

### 5.4 Predictions Sample
File `predictions.json` (id → đáp án, chuỗi rỗng = model từ chối) chứa toàn bộ 2882
predictions; **509 câu (17.7%) model bỏ trống** — 300 trong số đó là câu bẫy trả lời đúng,
209 là câu có đáp án bị bỏ lỡ.

```json
{
  "uit_000013": "thiếu tướng Quân đội Nhân dân Việt Nam, phó giám đốc Viện Khoa học và Công nghệ Quân sự",
  "uit_000014": "Trung Quốc, Liên Xô",
  "uit_000015": "nửa quên nửa nhớ"
}
```

---

## 6. Web Application

### 6.1 Công nghệ sử dụng
- **Frontend & Backend:** Streamlit (Python)
- **Model Inference:** PyTorch + Transformers
- **Deployment:** Local (có thể deploy lên cloud)

### 6.2 Tính năng
1. **Tab "Hỏi đáp trên đoạn văn của bạn":**
   - Text area cho context, text input cho question
   - Highlight đáp án trong context + kỹ thuật chi tiết (số cửa sổ, số ứng viên) mở rộng được
   - Độ tin cậy hiệu chuẩn và danh sách top phương án đã cân nhắc, mỗi phương án kèm điểm
   - Thanh "Độ khắt khe" đổi ngưỡng từ chối τ (mặc định 0.95) ngay trên giao diện

2. **Inference:**
   - 190–915 ms/câu trên RTX 3050 (FP16), phụ thuộc độ dài đoạn
   - Sliding window 256/64 cho context dài (tới 1537 từ trong test set)
   - Hai tầng: encoder PhoBERT sinh pool ứng viên → reranker chọn và quyết định im lặng

3. **Tab "Đánh giá hàng loạt trên test set":**
   - Chạy thật trên mẫu câu hỏi test (seed 42) bằng đúng pipeline Tab 1
   - Bảng một tầng vs hai tầng + đường cong ngưỡng τ + kết luận từng câu, xuất được CSV

### 6.3 Demo Screenshots

*[Chèn ảnh chụp màn hình ở đây]*

**Hình 1:** Giao diện chính
![Main Interface](screenshots/main_interface.png)

**Hình 2:** Ví dụ trả lời đúng
![Answer Example](screenshots/answer_example.png)

**Hình 3:** Phát hiện không có đáp án
![No Answer](screenshots/no_answer.png)

### 6.4 Cách chạy
```bash
streamlit run app/app.py
```

Truy cập: `http://localhost:8501`

Chi tiết: Xem `README_DEPLOYMENT.md`

---

## 7. Thách thức và Giải pháp

### 7.1 Thách thức kỹ thuật

**1. Tokenization và Offset Mapping**
- **Vấn đề:** PhoBERT dùng BPE tokenizer, khó map tokens ngược lại character positions
- **Giải pháp:** Tự implement custom offset mapping logic trong `QATokenizerProcessor`

**2. Context vượt quá max length**
- **Vấn đề:** Nhiều đoạn văn dài > 256 tokens
- **Giải pháp:** Sliding window với doc_stride=64, aggregate predictions từ多个 windows

**3. Unanswerable Questions**
- **Vấn đề:** Cần phát hiện khi không có đáp án trong context
- **Giải pháp:** So sánh null score (CLS token) với best answer score, sử dụng threshold

**4. Memory Constraints**
- **Vấn đề:** Model lớn (537MB weights FP32), GPU memory limited
- **Giải pháp:** 
  - FP16 mixed precision training
  - Gradient checkpointing
  - Gradient accumulation steps

### 7.2 Thách thức về dữ liệu

**1. Data Quality**
- **Vấn đề:** Dataset có noise, inconsistent formatting
- **Giải pháp:** Data cleaning pipeline, validation checks

**2. Class Imbalance**
- **Vấn đề:** Ít samples cho unanswerable questions
- **Giải pháp:** Careful evaluation với separate metrics

### 7.3 Bài học rút ra
- Importance của data quality over quantity
- Need for thorough error analysis
- Trade-off giữa model complexity và inference speed
- Value của good documentation và code organization

---

## 8. Kết luận

### 8.1 Những gì đã đạt được
✅ **Hoàn thành yêu cầu đề bài:**
- Fine-tune Transformer-based model (PhoBERT-base, 134.4M tham số) trên ViQuAD
- **EM 42.40 / F1 57.90** trên toàn bộ test set 2882 câu, kèm HasAns và NoAns tách riêng
- Xây dựng web application hoàn chỉnh, có tab tự chấm điểm trên test set thật

✅ **Technical achievements:**
- OOP training pipeline, custom offset mapping cho BPE tiếng Việt
- Tầng reranker + abstention có hiệu chuẩn: độ chính xác câu bẫy 34.85 → **56.06**
  trên mẫu 200 câu (τ = 0.95)
- đường cong đánh đổi EM/F1/câu bẫy theo ngưỡng τ đo được trên val, hiển thị trong app
- Training với FP16 + gradient checkpointing trong 59 phút trên laptop GPU 4GB

✅ **Bằng chứng phủ định (cũng là kết quả):**
- Pool 40 ứng viên recall 92.8% nhưng reranker đặc trưng bề mặt chỉ sửa được **12/821** câu
  xếp hạng sai → kết luận có số liệu: lỗi biên không sửa được bằng hậu xử lý, vì kiến trúc
  pointer chấm start và end độc lập rồi cộng lại

### 8.2 Hạn chế
- **Reranker đổi F1 câu có đáp án lấy câu bẫy:** −7.12 điểm HasAns F1, tỉ lệ bỏ trả lời
  tăng 22% → 35.5%. Hai cột phải đọc cùng nhau, không được trích một mình số EM +4.5.
- Model báo cáo là weights epoch 3 (eval_loss 1.2667) trong khi early-stopping chọn epoch 2
  (1.2203); chưa đo test set riêng cho weights epoch 2.
- Chưa train thêm backbone nào khác (PhoBERT-large, XLM-R) nên không có so sánh ngoài.
- Evaluation chỉ trên 1 dataset; web app chưa có authentication.

### 8.3 Hướng phát triển tương lai

**Short-term:**
1. Deploy lên Streamlit Cloud hoặc Hugging Face Spaces
2. Add user feedback mechanism
3. Lưu query history và analytics
4. Improve UI/UX với more examples

**Long-term:**
1. Experiment với PhoBERT-large hoặc ensemble methods
2. Multi-turn QA (conversational context)
3. Retrieval-Augmented Generation (RAG) cho open-domain QA
4. Multi-language support
5. Domain-specific fine-tuning (y tế, pháp luật, giáo dục)
6. Real-time performance monitoring và A/B testing

### 8.4 Lời kết
Project đã giúp chúng em hiểu sâu hơn về:
- Transformer architecture và cách fine-tune pre-trained models
- Challenges của Vietnamese NLP
- End-to-end ML pipeline từ data → training → evaluation → deployment
- Importance của documentation và user experience

Cảm ơn thầy/cô đã hướng dẫn và tạo điều kiện để chúng em thực hiện project này!

---

## 9. Tài liệu tham khảo

1. **PhoBERT:**
   - GitHub: https://github.com/VinAIresearch/PhoBERT
   - Paper: "PhoBERT: A Vietnamese language model" (2020)

2. **Hugging Face Transformers:**
   - Documentation: https://huggingface.co/docs/transformers
   - Model Hub: https://huggingface.co/vinai/phobert-base

3. **ViQuAD Dataset:**
   - [Link nếu có]

4. **SQuAD:**
   - Rajpurkar et al., "Know What You Don't Know: Unanswerable Questions for SQuAD" (2018)
   - Website: https://rajpurkar.github.io/SQuAD-explorer/

5. **Transformer Architecture:**
   - Vaswani et al., "Attention Is All You Need" (2017)
   - Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/

6. **Streamlit:**
   - Documentation: https://docs.streamlit.io
   - Gallery: https://streamlit.io/gallery

7. **PyTorch:**
   - Documentation: https://pytorch.org/docs/

---

## Phụ lục

### A. Installation Guide
Xem file `README_DEPLOYMENT.md` và `QUICKSTART.md`

### B. Code Structure
Xem file `PROJECT_SUMMARY.md`

### C. Submission Checklist
Xem file `CHECKLIST.md`

### D. Contact Information
- Student 1: [Name] - [Email]
- Student 2: [Name] - [Email]

---

*Báo cáo được tạo ngày: [DD/MM/YYYY]*
*Version: 1.0*
```

---

## 🎨 Tips để Report đẹp hơn

### 1. Thêm biểu đồ
Nếu có training logs, vẽ loss curves:
```python
import matplotlib.pyplot as plt
import pandas as pd

# Đọc logs
logs = pd.read_csv('train_logs.csv')

# Vẽ
plt.figure(figsize=(10, 6))
plt.plot(logs['epoch'], logs['train_loss'], label='Train Loss')
plt.plot(logs['epoch'], logs['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('loss_curves.png', dpi=300)
plt.show()
```

### 2. Chèn bảng so sánh
Chỉ so những gì nhóm tự chạy trên cùng test set — không copy số từ paper khác:

| Cấu hình | EM | F1 | Tham số |
|----------|-----|-----|---------|
| PhoBERT-base một tầng (toàn test, n=2882) | 42.40 | 57.90 | 134.4M |
| + reranker, τ=0.95 (mẫu 200 câu, seed 42) | 46.00 | 58.92 | 134.4M + GBM |
| Chính mẫu 200 câu, một tầng (để so công bằng) | 41.50 | 56.69 | 134.4M |

Ghi chú cho báo cáo: mBERT/XLM-R/PhoBERT-large **không** được train trong project này, nên
đừng đưa vào bảng — người chấm có thể yêu cầu chạy lại số liệu bất kỳ dòng nào ở trên.

### 3. Screenshots chất lượng cao
- Chụp ở độ phân giải cao
- Crop gọn gàng
- Thêm caption rõ ràng
- Lưu vào thư mục `screenshots/`

### 4. Formatting
- Sử dụng consistent heading levels
- Bold/italic cho emphasis
- Code blocks với syntax highlighting
- Tables cho structured data

---

## ✅ Checklist trước khi finalize Report

- [ ] Đã điền đầy đủ thông tin nhóm
- [ ] Có metrics từ evaluation
- [ ] Có ít nhất 2-3 screenshots
- [ ] References đầy đủ
- [ ] Spell check
- [ ] Formatting nhất quán
- [ ] File không quá lớn (< 10MB)
- [ ] Export sang PDF (optional nhưng recommended)

---

## 📤 Export sang PDF (Optional)

### Option 1: Pandoc
```bash
pandoc REPORT.md -o REPORT.pdf --pdf-engine=xelatex
```

### Option 2: VS Code Extension
- Install "Markdown PDF" extension
- Right-click → Export to PDF

### Option 3: Online Converter
- https://markdowntopdf.com/
- Upload .md file → Download PDF

---

**Good luck với báo cáo! 🎓**
