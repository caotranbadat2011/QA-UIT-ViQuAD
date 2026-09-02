"""
Streamlit web app for the Vietnamese Question Answering demo.

Run:
    streamlit run app/app.py
"""
import streamlit as st
import os
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Add src to path for the shared QA service
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from qa_service import QAService, DEFAULT_TAU_NULL
from batch_eval import TAU_GRID, compare, load_sample, metrics_table, sweep
from batch_eval import run as run_batch

# Đường dẫn đến model đã train
MODEL_DIR = str(ROOT / "models" / "phobert_qa")
RERANKER_DIR = str(ROOT / "models" / "reranker")
TEST_FILE = str(ROOT / "data" / "processed" / "test.parquet")
PREDICTIONS_FILE = ROOT / "predictions.json"
LAB_SEED = 42

st.set_page_config(
    page_title="Vietnamese QA System - Advanced", 
    page_icon="🔎",
    layout="wide"
)


@st.cache_resource
def get_service():
    """Loader model một lần; toàn bộ suy luận đi qua cùng một đường với evaluate.py."""
    return QAService(model_dir=MODEL_DIR, reranker_dir=RERANKER_DIR)


def mark_answer(context, answer):
    """In đậm đúng vị trí đáp án trong context."""
    if not answer or answer not in context:
        return context
    i = context.find(answer)
    return context[:i] + "**[" + answer + "]**" + context[i + len(answer):]


def load_sample_data():
    """Load dữ liệu mẫu từ file predictions.json nếu có"""
    if PREDICTIONS_FILE.exists():
        try:
            with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception:
            return {}
    return {}


# ==================== GIAO DIỆN CHÍNH ====================

st.title("🔎 Vietnamese Question Answering System - Advanced")
st.caption("Extractive QA trên ViQuAD · PhoBERT + bộ chấm lại ứng viên (reranker)")
st.markdown("---")

try:
    with st.spinner("⏳ Đang nạp model 513MB + reranker vào bộ nhớ (60–90 giây, chỉ một lần)..."):
        service = get_service()
except Exception as exc:
    st.error(f"Không tải được model từ `{MODEL_DIR}`: {exc}")
    st.stop()

# Sidebar - trạng thái model + hướng dẫn nhanh
with st.sidebar:
    if os.path.exists(MODEL_DIR):
        st.success(f"✅ Model đã load: `{MODEL_DIR}`")
    else:
        st.error(f"❌ Không tìm thấy model tại: `{MODEL_DIR}`")

    st.divider()

    st.header("📊 Hướng dẫn sử dụng")
    st.markdown("""
    **Tab Hỏi đáp:** dán đoạn văn + câu hỏi của bạn, xem đáp án, độ tin cậy và
    mọi phương án model đã cân nhắc. Kéo thanh *Độ khắt khe* để đổi mức dám nói
    'không có đáp án'.

    **Tab Đánh giá hàng loạt:** bấm một nút, model chạy thật trên một mẫu câu hỏi
    của test set rồi tự chấm. Thấy ngay model **sửa được bao nhiêu câu, làm hỏng
    bao nhiêu câu**, và đường cong đánh đổi của ngưỡng từ chối.
    """)

# ==================== HỒ SƠ MÔ HÌNH ====================
with st.expander("📇 Hồ sơ mô hình — kiến trúc hai tầng và số đo trên test set",
                 expanded=False):
    col_arch1, col_arch2 = st.columns(2)
    with col_arch1:
        st.markdown(
            "**1 · Extractor**\n\n"
            "PhoBERT-base fine-tune trên ViQuAD, dùng sliding window (256 token, "
            "stride 64), xuất top-40 phương án span."
        )
    with col_arch2:
        st.markdown(
            "**2 · Reranker**\n\n"
            "Gradient Boosting chấm lại 34 đặc trưng bề mặt của từng phương án "
            "(độ dài, giới từ hai đầu, trùng với câu hỏi, so với điểm CLS) và "
            "quyết định **từ chối trả lời** khi không có phương án đáng tin."
        )

    st.divider()

    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    col_i1.metric("EM toàn bộ", "42.40")
    col_i2.metric("F1 toàn bộ", "57.90")
    col_i3.metric("F1 câu có đáp án", "70.62")
    col_i4.metric("Tham số encoder", f"{service.num_params / 1e6:.0f}M",
                  help="Toàn bộ trọng số load vào bộ nhớ mỗi lần chạy app.")
    st.caption(
        "Đo trên test set ViQuAD (n=2882, 32.8% câu không có đáp án) · fine-tune "
        "3 epochs, learning rate 3e-5 · độ tin cậy hiệu chuẩn và quyền 'từ chối trả "
        "lời' là hai thứ giao diện này thêm vào so với model một tầng."
    )
    st.info(
        "Reranker không tăng F1 nhiều; giá trị thật của nó là nói được *model tin "
        "đến đâu* và *khi nào nên im lặng*."
    )

# Create tabs
tab_qa, tab_lab = st.tabs([
    "📝 Hỏi đáp trên đoạn văn của bạn",
    "🧪 Đánh giá hàng loạt trên test set",
])

# ==================== TAB 1: BASIC QA ====================
with tab_qa:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Nhập liệu")
        
        # Context input với height lớn hơn
        context_basic = st.text_area(
            "Đoạn văn (Context)",
            height=250,
            key="basic_context",
            placeholder="Dán đoạn văn bản tiếng Việt vào đây...\n\nVí dụ: Hà Nội là thủ đô của Việt Nam, nằm bên bờ sông Hồng. Thành phố này có lịch sử hơn 1000 năm và là trung tâm chính trị, kinh tế, văn hóa của cả nước.",
        )
        
        # Question input
        question_basic = st.text_input(
            "Câu hỏi (Question)", 
            key="basic_question",
            placeholder="Nhập câu hỏi về đoạn văn trên... Ví dụ: Hà Nội nằm ở đâu?",
        )
        
        # Nút trả lời
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            answer_button_basic = st.button("🔍 Trả lời", type="primary", use_container_width=True, key="btn_basic")
        with col_btn2:
            tau_null = st.slider(
                "Độ khắt khe khi nói 'không có đáp án'",
                min_value=0.90, max_value=0.99, value=DEFAULT_TAU_NULL, step=0.01,
                help="Model chỉ từ chối khi phương án 'không có đáp án' thắng VÀ đạt ít nhất "
                     "ngưỡng này. Hạ thấp = dám trả lời hơn nhưng dễ bịa đáp án cho câu hỏi ma.",
                key="slider_tau",
            )
    
    with col2:
        st.subheader("💡 Gợi ý")
        
        # Load sample predictions nếu có
        sample_data = load_sample_data()
        
        if sample_data:
            st.info("Có sẵn dữ liệu mẫu từ quá trình evaluation")
            
            # Hiển thị một vài ví dụ ngẫu nhiên
            if len(sample_data) > 0:
                sample_keys = list(sample_data.keys())[:3]
                for idx, key in enumerate(sample_keys):
                    with st.expander(f"Ví dụ {idx+1}"):
                        st.write(f"**ID:** {key}")
                        st.write(f"**Câu trả lời dự đoán:** {sample_data[key] if sample_data[key] else '(Không có đáp án)'}")
        else:
            st.markdown("""
            **Một số câu hỏi mẫu để thử:**
            
            📌 Context: *"PhoBERT là mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt, được phát triển bởi VinAI. Mô hình này đạt hiệu suất cao trên nhiều tác vụ NLP tiếng Việt."*
            
            ❓ Question: *"Ai đã phát triển PhoBERT?"*
            
            ➡️ Expected: *"VinAI"*
            
            ---
            
            📌 Context: *"Transformer là kiến trúc mạng neural sử dụng cơ chế self-attention, được giới thiệu trong bài báo 'Attention Is All You Need' năm 2017."*
            
            ❓ Question: *"Transformer sử dụng cơ chế gì?"*
            
            ➡️ Expected: *"self-attention"*
            """)
    
    # Xử lý khi nhấn nút Trả lời
    if answer_button_basic:
        if not context_basic:
            st.warning("⚠️ Vui lòng nhập đoạn văn bản (context)!")
        elif not question_basic:
            st.warning("⚠️ Vui lòng nhập câu hỏi!")
        else:
            with st.spinner("🤖 Mô hình đang suy nghĩ..."):
                t0 = time.perf_counter()
                result = service.answer(question_basic, context_basic, top_k=6,
                                        tau_null=tau_null)
                latency_ms = (time.perf_counter() - t0) * 1000

            st.markdown("---")
            st.subheader("📋 Kết quả")

            if result.get("error"):
                st.warning(result["error"])
            else:
                if result["abstained"]:
                    st.error("🚫 Model **từ chối trả lời**: phương án 'không có đáp án' "
                             "thuyết phục hơn mọi đoạn trích.")
                    if result["candidates"]:
                        top = result["candidates"][0]
                        st.info(
                            f"**Nếu vẫn buộc model trả lời**, phương án nó chọn là "
                            f"`{top['text'][:120]}` (độ tin cậy {top['confidence']:.2f}). "
                            f"Hạ thanh trượt 'Độ khắt khe' xuống để model dám trả lời hơn."
                        )
                else:
                    st.success(f"✅ **Câu trả lời:** {result['answer']}")
                    if result.get("low_confidence"):
                        st.warning("⚠️ Model trả lời nhưng **không chắc chắn** — "
                                   "đọc kỹ bảng phương án bên dưới trước khi tin.")

                c1, c2, c3, c4 = st.columns(4)
                conf = result["confidence"]
                c1.metric("Độ tin cậy rằng KHÔNG có đáp án" if result["abstained"]
                          else "Độ tin cậy vào đáp án",
                          f"{conf:.3f}",
                          delta=None if not result.get("tau") else f"ngưỡng {result['tau']:.2f}")
                c2.metric("Ứng viên đã cân nhắc", result.get("n_candidates", 0))
                c3.metric("Cửa sổ token", result.get("n_windows", 1))
                c4.metric("Độ trễ", f"{latency_ms:.0f} ms")
                st.caption(
                    f"Encoder {service.num_params/1e6:.0f}M tham số · chạy trên "
                    f"`{service.device}` · cửa sổ trượt 256 token, stride 64"
                )

                if result["answer"] and result["answer"] in context_basic:
                    with st.expander("🔍 Xem vị trí câu trả lời trong context", expanded=True):
                        st.markdown(mark_answer(context_basic, result["answer"]))

                if result["candidates"]:
                    st.markdown("**Các phương án model đã so sánh:**")
                    st.dataframe(
                        [
                            {
                                "#": i + 1,
                                "Độ tin cậy": round(c["confidence"], 3)
                                          if c.get("confidence") is not None else None,
                                "Phương án": c["text"][:90],
                                "Số token": c["n_tokens"],
                            }
                            for i, c in enumerate(result["candidates"])
                        ],
                        hide_index=True, use_container_width=True,
                    )

                with st.expander("📊 So sánh với mô hình một tầng (không reranker)"):
                    st.write(f"Model gốc chọn: **`{result.get('pointer_pick') or '(trống)'}`**")
                    st.caption(
                        "Model gốc cộng điểm token bắt đầu và token kết thúc rời nhau nên hay cắt "
                        "thừa/thiếu biên. Reranker chấm lại cả cụm từ nên có thể chọn phương án "
                        "khác hoặc từ chối trả lời."
                    )
                    st.markdown("**Đánh đổi của ngưỡng từ chối** (đo thật trên val, n=2872):")
                    st.dataframe(
                        [
                            {"Ngưỡng": "0.90", "Trả lời câu có đáp án": "70.4%",
                             "EM": 46.41, "F1": 57.89, "NoAns Accuracy": "58.3%"},
                            {"Ngưỡng": "0.95 (mặc định)", "Trả lời câu có đáp án": "77.9%",
                             "EM": 45.33, "F1": 58.76, "NoAns Accuracy": "50.6%"},
                            {"Ngưỡng": "0.97", "Trả lời câu có đáp án": "87.2%",
                             "EM": 43.04, "F1": 58.07, "NoAns Accuracy": "36.9%"},
                            {"Ngưỡng": "0.99", "Trả lời câu có đáp án": "93.0%",
                             "EM": 40.46, "F1": 56.33, "NoAns Accuracy": "24.3%"},
                            {"Ngưỡng": "không bao giờ từ chối", "Trả lời câu có đáp án": "100%",
                             "EM": 33.74, "F1": 50.52, "NoAns Accuracy": "0.0%"},
                        ],
                        hide_index=True, use_container_width=True,
                    )
                    if result.get("warning"):
                        st.warning(result["warning"])

# ==================== TAB 2: ĐÁNH GIÁ HÀNG LOẠT ====================
with tab_lab:
    st.header("🧪 Đánh giá hàng loạt trên câu hỏi test thật")
    st.markdown(
        "Thay vì chỉ thử vài câu tự nhập, tab này chạy **chính câu hỏi trong test set ViQuAD** "
        "qua cả hai tầng rồi tự chấm điểm bằng đúng thước dùng trong báo cáo (EM/F1)."
    )
    st.caption(
        "GPU chỉ chạy một lần cho mỗi câu: điểm của các phương án không đổi theo ngưỡng từ chối, "
        "nên kéo thanh ngưỡng bên dưới tính lại cả bảng trong vài mili giây."
    )

    if "lab_records" not in st.session_state:
        st.session_state.lab_records = []
        st.session_state.lab_secs = 0.0

    col_n, col_btn = st.columns([3, 1])
    with col_n:
        n_lab = st.slider(
            "Số câu hỏi trong mẫu (giữ nguyên tỉ lệ 32.8% câu bẫy không có đáp án)",
            min_value=20, max_value=300, value=60, step=20, key="lab_n",
            help="Đo trên máy này: khoảng 0.4–0.9 giây/câu. 60 câu ≈ 30–60 giây, "
                 "300 câu ≈ 4 phút. Kết quả chỉ phải tính lại khi đổi số câu.",
        )
    with col_btn:
        st.write("")
        st.write("")
        run_clicked = st.button("▶️ Chạy đánh giá", type="primary",
                                use_container_width=True, key="lab_run")

    if run_clicked:
        rows = load_sample(TEST_FILE, n_lab, seed=LAB_SEED)
        bar = st.progress(0.0, text="Đang chuẩn bị...")
        t0 = time.perf_counter()
        try:
            st.session_state.lab_records = run_batch(
                service, rows,
                progress_callback=lambda i, n: bar.progress(
                    i / n, text=f"Đang chấm {i}/{n} câu hỏi..."))
            st.session_state.lab_secs = time.perf_counter() - t0
        except Exception as exc:
            st.error(f"Chạy giữa chừng thì lỗi: {exc}")
        bar.empty()

    records = st.session_state.lab_records
    if not records:
        st.info("Chưa có kết quả. Chọn số câu hỏi rồi bấm **▶️ Chạy đánh giá**.")
    else:
        secs = st.session_state.lab_secs
        st.success(
            f"✅ Đã chạy thật {len(records)} câu trong {secs:.1f} giây "
            f"({secs / len(records) * 1000:.0f} ms/câu). Mẫu lấy theo seed cố định "
            f"{LAB_SEED} nên chạy lại vẫn ra đúng số cũ."
        )

        tau_lab = st.slider(
            "Ngưỡng từ chối τ — model chỉ dám nói 'không có đáp án' khi chắc hơn mức này",
            min_value=0.88, max_value=1.00, value=DEFAULT_TAU_NULL, step=0.005,
            key="lab_tau",
            help="τ thấp = model bỏ trả lời nhiều hơn (ít bịa hơn nhưng lỡ nhiều câu làm được); "
                 "τ = 1.00 = không bao giờ bỏ. Điểm của phương án 'không có đáp án' đa số rất "
                 "gần 1 nên mọi thay đổi đều nằm trong dải 0.90–1.00 này.",
        )

        table, one, two = metrics_table(records, tau_lab)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("EM toàn bộ", f"{two['overall_em']:.2f}",
                  f"{two['overall_em'] - one['overall_em']:+.2f} so với 1 tầng")
        c2.metric("F1 toàn bộ", f"{two['overall_f1']:.2f}",
                  f"{two['overall_f1'] - one['overall_f1']:+.2f}")
        c3.metric("F1 câu có đáp án", f"{two['has_ans_f1']:.2f}",
                  f"{two['has_ans_f1'] - one['has_ans_f1']:+.2f}")
        c4.metric("Bắt được câu bẫy", f"{two['no_ans_accuracy']:.1f}%",
                  f"{two['no_ans_accuracy'] - one['no_ans_accuracy']:+.1f} điểm")
        c5.metric("Bỏ trả lời", f"{two['refusal_rate']:.1f}%",
                  f"{two['refusal_rate'] - one['refusal_rate']:+.1f} điểm")

        st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
        st.caption(
            "Cột 'Một tầng' = PhoBERT không có reranker, chạy trên cùng một lần encode nên "
            "hai cột so sánh được trực tiếp. Mẫu nhỏ hơn test set đầy đủ nên số chỉ xấp xỉ."
        )

        st.divider()
        st.subheader("Đường cong đánh đổi của ngưỡng từ chối")
        curve = sweep(records, TAU_GRID)
        fig = go.Figure()
        for key, label in [("overall_f1", "F1 toàn bộ"),
                           ("no_ans_accuracy", "Bắt được câu bẫy"),
                           ("refusal_rate", "Bỏ trả lời")]:
            fig.add_trace(go.Scatter(
                x=[p["tau"] for p in curve], y=[p[key] for p in curve],
                name=label, mode="lines+markers"))
        fig.add_vline(x=tau_lab, line_dash="dot", line_color="gray",
                      annotation_text="đang xem")
        fig.update_layout(height=380, margin=dict(t=40),
                          xaxis_title="Ngưỡng từ chối τ", yaxis_title="%",
                          legend=dict(orientation="h", y=1.18, x=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Đọc trên đúng mẫu câu hỏi vừa chạy: mỗi ngưỡng là một đánh đổi giữa F1, số câu bẫy "
            "bắt được và số câu bị bỏ. Ngưỡng mặc định 0.95 được chọn theo số đo đầy đủ trên "
            "val (bảng ở tab Hỏi đáp), không phải chỉnh cho đẹp biểu đồ này."
        )

        st.divider()
        st.subheader("Từng câu một")
        detail = compare(records, tau_lab)
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "context"}
                           for r in detail])
        counts = df["Kết luận"].value_counts()
        st.markdown(" · ".join(f"**{k}** {v}" for k, v in counts.items()))
        chosen = st.multiselect("Lọc theo kết luận", counts.index.tolist(),
                                default=counts.index.tolist(), key="lab_filter")
        view = df[df["Kết luận"].isin(chosen)]
        st.dataframe(view, hide_index=True, use_container_width=True, height=340)

        if not view.empty:
            with st.expander("🔎 Soi chi tiết một câu", expanded=True):
                by_id = {r["id"]: r for r in detail}
                rec_by_id = {r["id"]: r for r in records}
                pick_id = st.selectbox(
                    "Câu hỏi", view["id"].tolist(), key="lab_pick",
                    format_func=lambda i: by_id[i]["Câu hỏi"][:90])
                rec, row = rec_by_id[pick_id], by_id[pick_id]

                st.markdown(f"**Câu hỏi:** {rec['question']}")
                st.markdown(
                    "**Đoạn văn** (đáp án đúng in đậm): "
                    + (mark_answer(rec["context"], rec["gold"]) if rec["gold"]
                       else rec["context"] + " — *câu này không có đáp án trong đoạn*"))
                p_null = row["Độ tin 'không có đáp án'"]
                st.write(
                    f"Đáp án đúng: **{row['Đáp án đúng']}** · một tầng: "
                    f"**{row['Một tầng']}** · hai tầng: **{row['Hai tầng']}** "
                    f"(độ tin cậy {row['Độ tin cậy']:.3f}, phương án 'không có đáp án' "
                    f"{p_null:.3f})"
                )
                st.dataframe(
                    [{"#": str(i + 1),
                      "Độ tin cậy": round(c["proba"], 3),
                      "Phương án": c["text"][:90],
                      "Số token": len(c["text"].split())}
                     for i, c in enumerate(rec["candidates"][:8])]
                    + [{"#": "NULL", "Độ tin cậy": round(rec["null_proba"], 3),
                        "Phương án": "(không có đáp án)", "Số token": 0}],
                    hide_index=True, use_container_width=True)

        st.download_button(
            "⬇️ Tải bảng kết quả (CSV)",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name="batch_eval_sample.csv", mime="text/csv")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>
    <b>Vietnamese Question Answering System - Advanced Edition</b> | 
    Built with PhoBERT + Streamlit + Plotly | 
    Final Project - Transformer-Based NLP Application
    </small>
</div>
""", unsafe_allow_html=True)
