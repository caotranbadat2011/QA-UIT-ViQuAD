# 🚀 Quick Start Guide

## Chạy ứng dụng trong 3 bước đơn giản

### Cách 1: Sử dụng script tự động (Khuyến nghị)

```bash
# Trên Windows - Double click hoặc chạy:
run_app.bat
```

Script sẽ tự động:
- ✅ Tạo virtual environment nếu chưa có
- ✅ Cài đặt tất cả dependencies
- ✅ Khởi động Streamlit app

### Cách 2: Chạy thủ công

```bash
# Bước 1: Tạo và kích hoạt virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Bước 2: Cài đặt dependencies
pip install -r requirements.txt

# Bước 3: Chạy ứng dụng
streamlit run app/app.py
```

---

## 📱 Truy cập ứng dụng

Sau khi chạy, mở trình duyệt và truy cập:
```
http://localhost:8501
```

---

## 💡 Sử dụng thử

1. **Copy đoạn văn sau vào ô Context:**
   ```
   PhoBERT là mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt, được phát triển bởi VinAI Research. 
   Mô hình này dựa trên kiến trúc RoBERTa và đạt hiệu suất state-of-the-art trên nhiều tác vụ NLP 
   tiếng Việt như phân loại văn bản, nhận diện thực thể, và trả lời câu hỏi.
   ```

2. **Nhập câu hỏi:**
   ```
   Ai đã phát triển PhoBERT?
   ```

3. **Nhấn "Trả lời"** → Kết quả: **"VinAI Research"**

---

## 📊 Thông tin nhanh về Model

| Thông số | Giá trị |
|----------|---------|
| Base Model | PhoBERT-base (VinAI) |
| Task | Extractive Question Answering |
| Dataset | ViQuAD |
| Parameters | ~135M |
| Max Sequence Length | 256 tokens |
| Supported Languages | Vietnamese |

---

## ❓ FAQ

**Q: Ứng dụng chạy chậm?**  
A: Lần đầu load model sẽ mất 5-10 giây. Các lần sau sẽ nhanh hơn nhờ cache.

**Q: Có cần GPU không?**  
A: Không bắt buộc. CPU vẫn chạy được nhưng chậm hơn. GPU khuyến nghị cho training.

**Q: Model được train trên dataset nào?**  
A: ViQuAD - Vietnamese Question Answering Dataset, tương tự SQuAD nhưng cho tiếng Việt.

**Q: Làm sao biết câu hỏi không có đáp án?**  
A: Model có cơ chế phát hiện unanswerable questions. Nếu không tìm thấy đáp án phù hợp, hệ thống sẽ hiển thị cảnh báo.

---

## 🔗 Links hữu ích

- [PhoBERT GitHub](https://github.com/VinAIresearch/PhoBERT)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Streamlit Documentation](https://docs.streamlit.io)

---

**Enjoy! 🎉**
