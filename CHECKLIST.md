# ✅ Final Checklist - Trước khi Submit

Sử dụng checklist này để đảm bảo project đã sẵn sàng 100% trước khi nộp.

---

## 📦 Files & Folders

### Core Files
- [x] `app/app.py` - Web application
- [x] `src/train.py` - Training script
- [x] `src/evaluate.py` - Evaluation script
- [x] `src/data_preprocessing.py` - Data preprocessing
- [x] `requirements.txt` - Dependencies list
- [x] `.gitignore` - Git ignore rules

### Model Files
- [ ] `models/phobert_qa/config.json` ✓
- [ ] `models/phobert_qa/model.safetensors` ✓
- [ ] `models/phobert_qa/tokenizer_config.json` ✓
- [ ] `models/phobert_qa/special_tokens_map.json` ✓
- [ ] `models/phobert_qa/vocab.txt` ✓
- [ ] `models/phobert_qa/bpe.codes` ✓
- [ ] `models/phobert_qa/added_tokens.json` ✓

**Lưu ý:** Xóa các checkpoint trung gian nếu không cần thiết để giảm kích thước:
```bash
# Kiểm tra
ls models/phobert_qa/checkpoint-*/

# Xóa nếu muốn (optional)
rm -rf models/phobert_qa/checkpoint-*
```

### Dataset
- [ ] `data/raw/train_formatted.json` (nếu nhỏ)
- [ ] `data/processed/train.parquet`
- [ ] `data/processed/val.parquet`
- [ ] `data/processed/test.parquet`

**Lưu ý:** Nếu dataset quá lớn (>100MB), cân nhắc upload riêng lên Google Drive và chỉ include link trong submission.

### Documentation
- [x] `README.md` - Main README
- [x] `QUICKSTART.md` - Quick start guide
- [x] `README_DEPLOYMENT.md` - Deployment guide
- [x] `SUBMISSION_GUIDE.md` - Submission instructions
- [x] `PROJECT_SUMMARY.md` - Project summary
- [ ] `REPORT.md` - **CẦN TẠO** (sử dụng template từ SUBMISSION_GUIDE.md)

### Helper Scripts
- [x] `test_model.py` - Test script
- [x] `run_app.bat` - Windows auto-run
- [x] `.streamlit/config.toml` - Streamlit config

### Outputs
- [ ] `predictions.json` - Từ evaluation script
- [ ] Training logs (optional)

---

## 🧪 Testing

### Test 1: Model Loading
```bash
python test_model.py
```
- [ ] Passes all 3 tests (Imports, Model Loading, Inference)

### Test 2: Web App
```bash
streamlit run app/app.py
```
- [ ] App starts without errors
- [ ] Model loads successfully (< 30 seconds)
- [ ] Can answer sample questions
- [ ] UI displays correctly

### Test 3: Sample Questions
Test với ít nhất 3 câu hỏi:

**Question 1:**
```
Context: "PhoBERT là mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt, được phát triển bởi VinAI."
Question: "Ai đã phát triển PhoBERT?"
Expected: "VinAI"
```
- [ ] Trả lời đúng

**Question 2:**
```
Context: "Hà Nội là thủ đô của Việt Nam."
Question: "TP.HCM nằm ở đâu?"
Expected: No answer / Không thể trả lời
```
- [ ] Phát hiện đúng là không có đáp án

**Question 3:** (Tự chọn)
```
Context: [Your context]
Question: [Your question]
```
- [ ] Trả lời hợp lý

### Test 4: Clean Install
Thử trên môi trường mới:
```bash
# Tạo venv mới
python -m venv test_env
test_env\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```
- [ ] Cài đặt thành công
- [ ] App chạy được

---

## 📊 Evaluation Results

Đảm bảo đã có kết quả evaluation:

```bash
python src/evaluate.py --model_dir models/phobert_qa --test_file data/processed/test.parquet
```

- [ ] Có file `predictions.json`
- [ ] Ghi lại metrics trong báo cáo:
  - EM Score: _______%
  - F1 Score: _______%
  - HasAns EM: _______%
  - HasAns F1: _______%
  - NoAns Accuracy: _______%

---

## 📝 Report

File `REPORT.md` cần có:

### Required Sections
- [ ] Thông tin nhóm (tên, MSSV, email)
- [ ] Giới thiệu task (Question Answering)
- [ ] Mô tả dataset (ViQuAD)
- [ ] Kiến trúc model (PhoBERT)
- [ ] Hyperparameters training
- [ ] Kết quả evaluation với bảng metrics
- [ ] Mô tả web application
- [ ] Thách thức và giải pháp
- [ ] Kết luận
- [ ] Tài liệu tham khảo

### Optional but Recommended
- [ ] Loss curves biểu đồ
- [ ] Screenshots của web app
- [ ] Error analysis
- [ ] So sánh với baseline

📖 Sử dụng template từ `SUBMISSION_GUIDE.md` để tạo report nhanh.

---

## 🗜️ Packaging

### Step 1: Clean up
```bash
# Xóa files không cần thiết
rm -rf __pycache__/
rm -rf .pytest_cache/
rm -f *.log train_log.txt train_err.txt
rm -rf models/*/checkpoint-*/  # Giữ lại best model only
```

### Step 2: Create student ID file
```bash
# Tạo file chứa student IDs
echo StudentID1 > StudentID1_StudentID2.txt
echo StudentID2 >> StudentID1_StudentID2.txt
```

### Step 3: Compress
```bash
# Option A: PowerShell (Windows)
powershell Compress-Archive -Path * -DestinationPath StudentID1_StudentID2.zip -Force

# Option B: 7-Zip/WinRAR GUI
# Select all → Right-click → Add to archive → StudentID1_StudentID2.zip
```

### Step 4: Verify ZIP
- [ ] Extract ZIP vào thư mục mới
- [ ] Chạy `pip install -r requirements.txt`
- [ ] Chạy `streamlit run app/app.py`
- [ ] Test với vài câu hỏi
- [ ] Everything works? ✓

---

## 📤 Submission

### If uploading directly:
- [ ] File name correct: `StudentID1_StudentID2.zip`
- [ ] File size < limit (check LMS)
- [ ] Upload to LMS before deadline

### If using Google Drive:
- [ ] Upload ZIP to Google Drive
- [ ] Set sharing: "Anyone with the link can view"
- [ ] Copy link
- [ ] Create text file: `StudentID1_StudentID2.txt` containing the link
- [ ] Submit the text file to LMS
- [ ] Test the link from incognito mode

---

## ⏰ Timeline Suggestion

**T-2 days (2 ngày trước deadline):**
- [ ] Hoàn thành REPORT.md
- [ ] Run final evaluation
- [ ] Test toàn bộ application

**T-1 day (1 ngày trước deadline):**
- [ ] Clean up project
- [ ] Create ZIP file
- [ ] Verify ZIP bằng cách extract và test lại

**T-0 (Ngày deadline):**
- [ ] Submit sớm trước 12h trưa
- [ ] Lưu confirmation email/receipt
- [ ] Inform team members đã submit xong

---

## 🚨 Common Mistakes to Avoid

- [ ] ❌ Quên include model weights
- [ ] ❌ File ZIP quá lớn (>500MB)
- [ ] ❌ Sai tên file ZIP (phải có StudentIDs)
- [ ] ❌ Quên file predictions.json
- [ ] ❌ Report thiếu metrics
- [ ] ❌ Code không chạy được sau khi extract
- [ ] ❌ Quên set Google Drive permissions
- [ ] ❌ Submit trễ deadline

---

## ✅ Final Verification

Trước khi nhấn nút Submit, tự hỏi:

1. **Code có chạy được không?**
   - [ ] Đã test clean install
   
2. **Model có load được không?**
   - [ ] Đã test với test_model.py
   
3. **Web app có hoạt động không?**
   - [ ] Đã thử trả lời vài câu hỏi
   
4. **Report có đầy đủ không?**
   - [ ] Có metrics
   - [ ] Có thông tin nhóm
   - [ ] Có mô tả đầy đủ
   
5. **File name có đúng không?**
   - [ ] StudentID1_StudentID2.zip

6. **Đã backup chưa?**
   - [ ] Copy ra USB/cloud

---

## 🎉 Ready to Submit!

Nếu tất cả checkboxes đều được tick:

**YOU'RE READY! Go ahead and submit! 🚀**

Good luck! 🍀

---

*Last updated: 2026-09-01*
