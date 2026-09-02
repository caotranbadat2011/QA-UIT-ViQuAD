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

Copy kết quả vào report:

```markdown
### 5.1 Metrics

| Metric | Score |
|--------|-------|
| **Exact Match (EM)** | [COPY TỪ OUTPUT]% |
| **F1 Score** | [COPY TỪ OUTPUT]% |
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
- **Tên dataset:** ViQuAD (Vietnamese Question Answering Dataset)
- **Ngôn ngữ:** Tiếng Việt
- **Nguồn:** VinAI Research / Cộng đồng NLP tiếng Việt
- **License:** [Nếu biết]

### 2.2 Thống kê
| Split | Số lượng samples |
|-------|------------------|
| Train | [Xem data/processed/train.parquet] |
| Validation | [Xem data/processed/val.parquet] |
| Test | [Xem data/processed/test.parquet] |

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
1. Pre-trained trên corpus tiếng Việt lớn (20GB+)
2. Đạt SOTA trên nhiều benchmarks tiếng Việt
3. Dựa trên RoBERTa architecture - proven effective for QA
4. Community support tốt, dễ sử dụng với Hugging Face
5. Phù hợp với tài nguyên compute available

### 3.2 Thông số kỹ thuật
| Parameter | Value |
|-----------|-------|
| Architecture | RoBERTa-base variant |
| Hidden size | 768 dimensions |
| Attention heads | 12 |
| Transformer layers | 12 |
| Vocabulary size | 64,001 tokens |
| Total parameters | ~135 million |
| Max position embeddings | 258 |

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
- **Framework:** PyTorch 2.x + Hugging Face Transformers 4.x
- **GPU:** [Điền loại GPU nếu có, ví dụ: NVIDIA GTX 1650 4GB]
- **CPU:** [Nếu train trên CPU]
- **RAM:** [Amount used]
- **Training time:** [X hours/minutes]
- **Storage:** Model size ~540MB

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
[Chèn biểu đồ loss curves nếu có từ train_log.txt]

Hoặc mô tả:
- Training loss giảm đều qua các epochs
- Validation loss ổn định, không overfitting
- Best checkpoint saved at epoch [X]

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

**Kết quả:**

| Metric | Score |
|--------|-------|
| **Overall Exact Match (EM)** | [XX.XX]% |
| **Overall F1 Score** | [XX.XX]% |
| HasAns EM | [XX.XX]% |
| HasAns F1 | [XX.XX]% |
| NoAns Accuracy | [XX.XX]% |

*Lưu ý: Thay thế [XX.XX] bằng số thực tế từ output của evaluate.py*

### 5.3 Phân tích kết quả

**Điểm mạnh:**
- Model học được pattern trích xuất thông tin cơ bản
- Xử lý tốt các câu hỏi đơn giản, trực tiếp
- Phát hiện tương đối tốt câu hỏi không có đáp án

**Điểm yếu:**
- Khó khăn với câu hỏi đòi hỏi suy luận phức tạp
- Nhạy cảm với wording của câu hỏi
- Context dài có thể làm giảm accuracy

**Error Analysis:**
Các trường hợp sai phổ biến:
1. Answer span quá dài/ngắn so với expected
2. Không xử lý tốt negation (không, chưa, chẳng)
3. Confusion khi có multiple entities cùng loại

### 5.4 Predictions Sample
File `predictions.json` chứa toàn bộ predictions trên test set.

Ví dụ:
```json
{
  "qa_001": "Hà Nội",
  "qa_002": "",
  "qa_003": "sông Hồng"
}
```

---

## 6. Web Application

### 6.1 Công nghệ sử dụng
- **Frontend & Backend:** Streamlit (Python)
- **Model Inference:** PyTorch + Transformers
- **Deployment:** Local (có thể deploy lên cloud)

### 6.2 Tính năng
1. **Input Interface:**
   - Text area cho context (đoạn văn)
   - Text input cho question (câu hỏi)
   - Responsive design

2. **Inference:**
   - Real-time prediction (< 1 second)
   - Sliding window cho context dài
   - Unanswerable question detection

3. **Output Display:**
   - Highlight answer trong context
   - Hiển thị confidence scores
   - Technical details expandable

4. **User Experience:**
   - Clean, intuitive interface
   - Helpful tooltips và examples
   - Loading indicators

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
- **Vấn đề:** Model lớn (~540MB), GPU memory limited
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
- Fine-tune Transformer-based model (PhoBERT) cho NLP task
- Đạt kết quả khả quan trên ViQuAD dataset
- Xây dựng web application hoàn chỉnh

✅ **Technical achievements:**
- Implement OOP architecture cho training pipeline
- Custom tokenization và offset mapping
- Unanswerable question detection
- Optimized training với FP16 và gradient checkpointing

✅ **Documentation:**
- Comprehensive guides (Quick Start → Deployment → Submission)
- Clean, well-commented code
- Test scripts và verification tools

### 8.2 Hạn chế
- Chưa thử nghiệm với larger models (PhoBERT-large)
- Evaluation chỉ trên 1 dataset
- Web app chưa có authentication hay user management
- Chưa optimize cho production deployment

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
So sánh với baseline hoặc các models khác:

| Model | EM | F1 | Parameters |
|-------|-----|-----|------------|
| PhoBERT-base (ours) | XX.XX% | XX.XX% | 135M |
| mBERT | XX.XX% | XX.XX% | 178M |
| XLM-R | XX.XX% | XX.XX% | 279M |

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
