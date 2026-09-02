# 🚀 Quick Guide: Testing New Features

> ⚠️ **Trạng thái thực tế:** `app/app.py` hiện chỉ có **2 tab** — "📝 Hỏi đáp trên đoạn văn
> của bạn" và "🧪 Đánh giá hàng loạt trên test set". Các tính năng mà guide này hướng dẫn
> test **đã bị bỏ khỏi app**; các mục Inference Speed / Memory Usage bên dưới là
> **ước lượng chưa đo**. Số đo thật nằm trong `README.md` (mục Results).

Hướng dẫn nhanh để cài đặt và test các tính năng nâng cấp.

---

## 📦 Bước 1: Cài đặt Dependencies mới

```bash
# Kích hoạt virtual environment (nếu có)
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Cài đặt packages mới
pip install plotly matplotlib seaborn scikit-learn tqdm

# Hoặc dùng requirements.txt
pip install -r requirements.txt
```

---

## 🧪 Bước 2: Test từng Feature

### Test 1: Attention Visualization

```bash
# Chạy web app
streamlit run app/app.py

# Trong trình duyệt:
1. Click tab "🔍 Explainability"
2. Nhập context: "PhoBERT là mô hình do VinAI phát triển."
3. Nhập question: "Ai phát triển PhoBERT?"
4. Click "Giải thích"
5. ✅ Xem: Token importance bar chart + Attention heatmap
```

**Expected:** 
- Bar chart hiển thị importance scores cho tokens
- Heatmap (nếu thành công) showing question-context attention

**Troubleshooting:**
- Nếu heatmap không hiện: Check console log, có thể do tokenization issues
- Fallback: Chỉ show bar chart vẫn OK

---

### Test 2: Error Analysis Dashboard

```bash
# Bước 1: Đảm bảo có predictions.json
python src/evaluate.py \
    --model_dir models/phobert_qa \
    --test_file data/processed/test.parquet

# Bước 2: Chạy advanced analysis
python src/metrics_advanced.py \
    --predictions predictions.json \
    --test_file data/processed/test.parquet

# Bước 3: Xem trong web app
streamlit run app/app.py
# -> Tab "📊 Analytics"

# ✅ Xem: Confusion matrix, error categories, bad cases
```

**Expected:**
- File `error_analysis.json` được tạo ra
- Web app hiển thị confusion matrix heatmap
- Bar chart của error distribution
- Table với accuracy per category
- Top 10 bad cases expandable

**Troubleshooting:**
- Nếu lỗi import: `pip install pandas scikit-learn`
- Nếu không load được test data: Check path `data/processed/test.parquet`

---

### Test 3: Multi-Turn Conversation

```bash
# Chạy web app
streamlit run app/app.py

# Trong tab "💬 Conversation":
1. Nhập context bất kỳ
2. Hỏi: "Câu hỏi 1?"
3. ✅ Xem answer
4. Hỏi tiếp: "Câu hỏi 2 về cùng context?"
5. ✅ Context tự động reuse
6. Chat history hiển thị như messaging app
7. Click "Xóa lịch sử" để reset
```

**Expected:**
- Messages hiển thị trong chat bubbles
- Context từ câu trước được giữ nguyên (có thể edit)
- Nút clear history hoạt động

---

## ⚡ Quick Demo Script

Kịch bản demo 5 phút cho grading:

### Minute 1: Introduction
```
"Project này không chỉ train model mà còn có 4 tính năng độc đáo..."
```

### Minute 2: Basic QA + Explainability
```
1. Mở tab "Basic QA"
2. Context: "Hà Nội là thủ đô Việt Nam, dân số 8.4 triệu."
3. Question: "Hà Nội có bao nhiêu dân?"
4. Answer: "8.4 triệu" ✅
5. Switch to "Explainability" tab
6. Show same question -> Click "Giải thích"
7. Point to heatmap: "Model tập trung vào '8.4 triệu' và 'dân số'"
```

### Minute 3: Multi-Turn Conversation
```
1. Switch to "Conversation" tab
2. Context: Info về Hà Nội
3. Q1: "Thủ đô Việt Nam là gì?" -> "Hà Nội"
4. Q2: "Dân số của nó?" -> "8.4 triệu" (nó = Hà Nội)
5. Q3: "Thành phố này nổi tiếng gì?" -> Answer
6. Show chat history scrollable
```

### Minute 4: Analytics Dashboard
```
1. Switch to "Analytics" tab
2. Show metrics: EM, F1 scores
3. Point to confusion matrix
4. Show error categories chart
5. Expand 1-2 bad cases
6. "Đây là professional-level evaluation"
```

### Minute 5: Conclusion
```
"Tóm lại, project có:"
- ✅ Basic QA working perfectly
- ✅ Explainability (unique!)
- ✅ Multi-turn conversation (unique!)
- ✅ Advanced analytics (unique!)
- ✅ Clean code & documentation

"Các tính năng này ít projects nào có!"
```

---

## 🎯 Checklist trước khi Demo

- [ ] Đã cài đặt plotly, matplotlib, seaborn
- [ ] Đã chạy `evaluate.py` -> có `predictions.json`
- [ ] Đã chạy `metrics_advanced.py` -> có `error_analysis.json`
- [ ] Đã test cả 4 tabs trong web app
- [ ] Chuẩn bị sẵn 3-5 câu hỏi demo hay
- [ ] Kiểm tra app chạy mượt, không crash
- [ ] Screenshot key features (backup nếu live demo fail)

---

## 🐛 Common Issues & Fixes

### Issue 1: Plotly không hiển thị
```bash
# Fix: Update streamlit và plotly
pip install --upgrade streamlit plotly

# Restart app
```

### Issue 2: Import error trong explainability
```bash
# Check file exists
ls src/explainability.py

# Check imports work
python -c "from src.explainability import AttentionVisualizer"
```

### Issue 3: Error analysis chạy lâu
```bash
# Reduce dataset size for demo
# Edit src/metrics_advanced.py, line ~200:
bad_cases[:10]  # Thay vì [:20]
```

### Issue 4: Memory error với attention extraction
```bash
# Model quá lớn cho RAM
# Fix: Use CPU only, limit layers
# Already handled in code with try-except
```

---

## 📊 Expected Performance

### Inference Speed:
- Basic QA: < 1 second per query
- Explainability: 2-3 seconds (attention extraction overhead)
- Conversation: < 1 second (same as basic)

### Memory Usage:
- Base model: ~1GB RAM
- With explainability: ~1.5GB RAM
- Streamlit app: ~500MB overhead

### Accuracy:
- Khớp với kết quả `src/evaluate.py` trên toàn test set (n=2882):
  - EM: 42.40%
  - F1: 57.90%
  - HasAns EM/F1: 47.57% / 70.62%, NoAns Accuracy: 31.78%

---

## 🎉 Success Criteria

Demo thành công nếu:

✅ App chạy không crash trong 5 phút  
✅ Show được ít nhất 3 unique features  
✅ Graders hiểu được điểm khác biệt  
✅ Trả lời được câu hỏi technical về implementation  
✅ Code clean, well-documented  

---

**Good luck với demo! Bạn đã có những features rất ấn tượng! 🚀**
