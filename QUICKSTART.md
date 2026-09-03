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

3. **Nhấn "Trả lời"** → Kết quả: **"VinAI Research"** (đã chạy thật, độ tin 0.985)

---

## 📊 Thông tin nhanh về Model

| Thông số | Giá trị |
|----------|---------|
| Base Model | PhoBERT-base (VinAI) |
| Task | Extractive Question Answering |
| Dataset | ViQuAD |
| Parameters | 134.4M encoder (134,409,218) |
| Max Sequence Length | 256 tokens, stride 64 |
| Supported Languages | Vietnamese |
| EM / F1 toàn test set | 42.40 / 57.90 |
| F1 câu có đáp án | 70.62 |
| Độ chính xác câu bẫy | 31.78 (một tầng) · 56.06 (hai tầng, τ=0.95) |
| Độ trễ | ~40–90 ms với context ngắn; 190–915 ms trên test set (RTX 3050 Laptop 4GB) |

Số liệu đầy đủ: xem mục Results trong [README.md](README.md).

---

## ❓ FAQ

**Q: Ứng dụng chạy chậm?**  
A: Lần đầu load model (weights 537MB + reranker) mất khoảng 17–21 giây trên máy này. Sau khi
app đã chạy thì mỗi câu chỉ mất ~40–230 ms; model được giữ trong RAM, không load lại.

**Q: Có cần GPU không?**  
A: Không bắt buộc. CPU vẫn chạy được nhưng chậm hơn. GPU khuyến nghị cho training.

**Q: Model được train trên dataset nào?**  
A: ViQuAD - Vietnamese Question Answering Dataset, tương tự SQuAD nhưng cho tiếng Việt.

**Q: Làm sao biết câu hỏi không có đáp án?**  
A: Tầng reranker cho mỗi phương án một điểm "không có đáp án". Hệ thống chỉ im lặng khi
phương án đó thắng *và* đạt ngưỡng τ = 0.95, nên bao giờ cũng kèm độ tin cậy để bạn tự kiểm
tra. Kéo thanh *Độ khắt khe* trong Tab 1 để đổi mức dám nói "không có đáp án".

**Q: App báo không load được model?**  
A: Trọng số 537MB nằm trong repo nhưng qua **Git LFS**. Nếu `models/phobert_qa/model.safetensors`
chỉ có 134 byte và mở ra thấy dòng `version https://git-lfs...`, tức là bạn chưa tải object LFS:
```bash
git lfs install
git lfs pull
```
Checkpoint trung gian (`checkpoint-4414`, `checkpoint-6621`) cố ý không có trong repo — bản push
chỉ chứa model cuối của epoch 3 cùng tokenizer.

---

## 🔗 Links hữu ích

- [PhoBERT GitHub](https://github.com/VinAIresearch/PhoBERT)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Streamlit Documentation](https://docs.streamlit.io)

---