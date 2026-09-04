"""
Streamlit web app for the Vietnamese Question Answering demo.

Run:
    streamlit run app.py
"""
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Đã cập nhật đúng đường dẫn lưu mô hình của dự án
MODEL_DIR = "models/best_checkpoint"  
NO_ANSWER_THRESHOLD = 0.0  # Tùy chỉnh ngưỡng loại bỏ câu hỏi không đáp án (-1.0 đến -2.0 nếu muốn khắt khe hơn)

st.set_page_config(page_title="Vietnamese QA Demo", page_icon="🔎")


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


def answer_question(question, context, tokenizer, model, max_length=384):
    inputs = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = inputs.pop("offset_mapping")[0]
    sequence_ids = inputs.sequence_ids(0)

    # Đưa inputs vào thiết bị phù hợp (CPU/GPU)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Đã sửa lỗi: ép về cpu() trước khi chuyển sang numpy()
    start_logits = outputs.start_logits[0].detach().cpu().numpy()
    end_logits = outputs.end_logits[0].detach().cpu().numpy()

    null_score = start_logits[0] + end_logits[0]

    n_best = 20
    max_answer_length = 64
    start_idx = start_logits.argsort()[-n_best:][::-1]
    end_idx = end_logits.argsort()[-n_best:][::-1]

    best_score, best_answer = -1e9, ""
    for s in start_idx:
        for e in end_idx:
            # Chỉ xét các token nằm trong context (sequence_ids == 1)
            if sequence_ids[s] != 1 or sequence_ids[e] != 1:
                continue
            if e < s or e - s + 1 > max_answer_length:
                continue
            score = start_logits[s] + end_logits[e]
            if score > best_score:
                best_score = score
                start_char, end_char = offset_mapping[s][0], offset_mapping[e][1]
                best_answer = context[start_char:end_char]

    if null_score > best_score + NO_ANSWER_THRESHOLD:
        return None, best_score, null_score
    return best_answer, best_score, null_score

st.title("🔎 Vietnamese Question Answering")
st.caption("Extractive QA fine-tuned on ViQuAD (with unanswerable question detection)")

# Load mô hình & tokenizer
try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Không thể tải mô hình từ thư mục '{MODEL_DIR}'. Vui lòng kiểm tra lại đường dẫn!")
    st.stop()

context = st.text_area(
    "Đoạn văn (context)",
    height=200,
    placeholder="Dán đoạn văn bản tiếng Việt vào đây...",
)
question = st.text_input("Câu hỏi (question)", placeholder="Nhập câu hỏi về đoạn văn trên...")

if st.button("Trả lời", type="primary") and context and question:
    with st.spinner("Đang suy nghĩ..."):
        answer, ans_score, null_score = answer_question(question, context, tokenizer, model)
    if answer:
        st.success(f"**Trả lời:** {answer}")
    else:
        st.warning("Mô hình cho rằng câu hỏi này **không thể trả lời** dựa trên đoạn văn.")
    with st.expander("Chi tiết điểm số"):
        st.write(f"Answer span score: {ans_score:.3f}")
        st.write(f"No-answer score: {null_score:.3f}")