from pathlib import Path
from html import escape

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard import prediction_frames
from src.dataset import read_comments_csv
from src.inference import AspectPredictor
from src.schema import ASPECT_NAMES, STATE_COLORS, STATE_NAMES

st.set_page_config(page_title="AspectLens AI", page_icon="🔎", layout="wide", initial_sidebar_state="collapsed")
st.markdown(f"<style>{(Path(__file__).parent / 'styles.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
st.markdown("""
<div class="masthead">
  <div class="masthead-copy">
    <div class="masthead-kicker">Vietnamese review intelligence · UIT‑ViSFD</div>
    <div class="masthead-title">ASPECTLENS AI</div>
  </div>
  <div class="masthead-status"><strong>System ready</strong>10 khía cạnh · 3 sắc thái<br>SLP + TF‑IDF</div>
</div>
<div class="command-bar">
  <span class="brand-pill">AspectLens</span>
  <span class="command-links"><span>Phân tích</span><span>Dashboard</span><span>Mô hình</span></span>
  <span class="command-chip">Review console · VI</span>
</div>
<div class="subnav-strip"><span>Bình luận</span><span>Khía cạnh</span><span>Sắc thái</span><span>Thống kê phiên hiện tại</span></div>
<div class="hero-plate">
  <div class="section-kicker">01 / Review analysis system</div>
  <h1>KHÁCH HÀNG<br>ĐANG NÓI GÌ?<span class="hero-arrow">›</span></h1>
  <p>Phân tích bình luận theo từng khía cạnh, nhận diện sắc thái và tổng hợp những điểm khách hàng phàn nàn nhiều nhất.</p>
</div>
<div class="chrome-note">DỮ LIỆU PHIÊN HIỆN TẠI · Thống kê chỉ tính trên bình luận bạn phân tích hoặc nạp vào, không đại diện toàn thị trường.</div>
""", unsafe_allow_html=True)

artifact_path = Path(__file__).parent / "results.zip"

@st.cache_resource(show_spinner="Đang nạp mô hình ABSA…")
def load_predictor(path: str, modified: int):
    return AspectPredictor(path)

try:
    predictor = load_predictor(str(artifact_path), artifact_path.stat().st_mtime_ns)
except Exception as exc:
    st.error("Không nạp được checkpoint. Kiểm tra file results.zip trong repository và chạy lại ứng dụng.")
    st.exception(exc)
    st.stop()

tab_single, tab_dashboard, tab_about = st.tabs(["Phân tích bình luận", "Dashboard", "Giới thiệu & chất lượng mô hình"])

with tab_single:
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.subheader("Phân tích một bình luận")
        comment = st.text_area("Nội dung bình luận", height=155, max_chars=4000, placeholder="Ví dụ: Pin dùng lâu, camera chụp đẹp nhưng máy chạy ứng dụng hơi nóng…")
        analyze = st.button("Phân tích", type="primary", use_container_width=True, key="single-analyze")
        if analyze:
            if not comment.strip():
                st.warning("Nhập bình luận trước khi phân tích.")
            else:
                prediction = predictor.predict_many([comment.strip()])[0]
                st.session_state["last_single"] = prediction
                st.session_state["predictions"] = st.session_state.get("predictions", []) + [prediction]
    with right:
        st.subheader("Kết quả theo khía cạnh")
        result = st.session_state.get("last_single")
        if result:
            if not result["aspects"]:
                st.info("Mô hình không phát hiện khía cạnh được đề cập rõ trong bình luận này.")
            for aspect in result["aspects"]:
                state = aspect["sentiment"]
                st.markdown(
                    f"<div class='result-card {state.lower()}'><span class='signal-dot'></span>"
                    f"<strong>{escape(aspect['aspect_vi'])}</strong>"
                    f"<span class='sentiment'>{escape(STATE_NAMES[state])}</span>"
                    f"<span class='confidence'>{aspect['confidence']:.0%}</span></div>",
                    unsafe_allow_html=True,
                )
            st.caption("Độ tin cậy hiển thị là softmax, chưa được hiệu chuẩn.")
        else:
            st.info("Kết quả phân tích sẽ xuất hiện tại đây.")

    st.divider()
    st.subheader("Phân tích nhiều bình luận")
    st.caption("Dán mỗi bình luận trên một dòng hoặc tải CSV có cột comment/review/text. Tối đa 5.000 dòng mỗi lần.")
    text_batch = st.text_area("Danh sách bình luận", height=130, placeholder="Pin tốt, dùng cả ngày vẫn còn pin.\nGiá hơi cao so với cấu hình.\nCamera chụp đêm chưa đẹp.", key="batch-text")
    uploaded = st.file_uploader("Hoặc chọn file CSV", type=["csv"], key="batch-csv")
    run_batch = st.button("Phân tích batch", type="primary", key="batch-analyze")
    if run_batch:
        try:
            if uploaded is not None:
                batch_frame, comment_col = read_comments_csv(uploaded)
                comments = batch_frame[comment_col].astype(str).tolist()
            else:
                comments = [line.strip() for line in text_batch.splitlines() if line.strip()]
                batch_frame = None
            if not comments:
                st.warning("Dán bình luận hoặc tải CSV trước.")
            elif len(comments) > 5000:
                st.error("Giới hạn 5.000 bình luận mỗi lần.")
            else:
                st.session_state["batch_predictions"] = predictor.predict_many(comments)
                if batch_frame is None:
                    st.session_state.pop("batch_frame", None)
                else:
                    st.session_state["batch_frame"] = batch_frame
        except Exception as exc:
            st.error(f"Không đọc được dữ liệu: {exc}")

    batch = st.session_state.get("batch_predictions")
    if batch:
        detail, _ = prediction_frames(batch)
        st.success(f"Đã phân tích {len(batch):,} bình luận · {len(detail):,} aspect được phát hiện")
        if not detail.empty:
            view = detail[["comment", "aspect_vi", "sentiment_vi", "confidence"]].rename(columns={"comment":"Bình luận", "aspect_vi":"Khía cạnh", "sentiment_vi":"Sắc thái", "confidence":"Độ tin cậy"})
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.download_button("Tải kết quả CSV", view.to_csv(index=False).encode("utf-8-sig"), file_name="aspectlens_analysis.csv", mime="text/csv")
        else:
            st.info("Mô hình chưa nhận diện khía cạnh rõ trong batch này.")

with tab_dashboard:
    st.subheader("Bức tranh sắc thái khách hàng")
    batch = st.session_state.get("batch_predictions")
    if not batch:
        st.info("Hãy phân tích batch ở tab ‘Phân tích bình luận’ để tạo dashboard cho tập bình luận của bạn.")
    else:
        detail, summary = prediction_frames(batch)
        if detail.empty:
            st.info("Chưa có khía cạnh nào được dự đoán trong batch này.")
        else:
            mentions = int(len(detail))
            negatives = int((detail.sentiment == "Negative").sum())
            unique_complaints = int(detail.loc[detail.sentiment == "Negative", "comment_id"].nunique())
            worst = summary.sort_values(["Tiêu cực", "Tỷ lệ tiêu cực (%)"], ascending=False).iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Bình luận", f"{len(batch):,}")
            c2.metric("Lượt nhắc khía cạnh", f"{mentions:,}")
            c3.metric("Lượt tiêu cực", f"{negatives:,}", help="Một bình luận có thể có nhiều aspect tiêu cực.")
            c4.metric("Phàn nàn nhiều nhất", worst["Khía cạnh"])
            st.caption(f"{unique_complaints:,} bình luận có ít nhất một aspect tiêu cực. Thống kê chỉ phản ánh batch hiện tại.")

            rank = summary.sort_values("Tiêu cực", ascending=True)
            fig = px.bar(rank, x="Tiêu cực", y="Khía cạnh", orientation="h", text="Tiêu cực", color="Tỷ lệ tiêu cực (%)", color_continuous_scale=["#c0d5e6", "#8ba1d4", "#e60012"], title="Khía cạnh có nhiều lượt phàn nàn nhất")
            fig.update_layout(height=440, margin=dict(l=10, r=10, t=60, b=10), coloraxis_colorbar_title="Tỷ lệ tiêu cực (%)", xaxis_title="Số lượt dự đoán tiêu cực", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Phân bố sắc thái theo khía cạnh**")
            polarity = detail.groupby(["aspect_vi", "sentiment"]).size().reset_index(name="Số lượt")
            polarity["Sắc thái"] = polarity["sentiment"].map(STATE_NAMES)
            color_map = {STATE_NAMES[k]: v for k, v in STATE_COLORS.items()}
            chart = px.bar(polarity, x="aspect_vi", y="Số lượt", color="Sắc thái", barmode="stack", category_orders={"aspect_vi": [ASPECT_NAMES[a] for a in ASPECT_NAMES]}, color_discrete_map=color_map)
            chart.update_layout(height=390, xaxis_title="", yaxis_title="Số lượt khía cạnh", legend_title="")
            st.plotly_chart(chart, use_container_width=True)

            st.markdown("**Bảng xếp hạng**")
            st.dataframe(summary, use_container_width=True, hide_index=True)
            negative = detail[detail.sentiment == "Negative"]
            if not negative.empty:
                st.markdown("**Một số bình luận tiêu cực theo khía cạnh**")
                selected = st.selectbox("Chọn khía cạnh", list(negative.aspect_vi.drop_duplicates()), key="complaint-aspect")
                samples = negative[negative.aspect_vi == selected][["comment", "confidence"]].drop_duplicates("comment").head(10)
                st.dataframe(samples.rename(columns={"comment":"Bình luận", "confidence":"Độ tin cậy"}), use_container_width=True, hide_index=True)

with tab_about:
    st.subheader("Mô hình và phạm vi sử dụng")
    st.caption("Mô hình học từ đánh giá smartphone. Nhãn ‘Không đề cập’ được ẩn khỏi kết quả khía cạnh.")
    st.markdown("""
- **Bài toán:** dự đoán `Absent / Negative / Neutral / Positive` riêng cho từng aspect.
- **Mô hình:** character TF-IDF (n-gram 2–5) + một tầng tuyến tính SLP bằng PyTorch, không có tầng ẩn.
- **Dữ liệu:** UIT‑ViSFD, bình luận đánh giá smartphone tiếng Việt; dùng split gốc.
- **Kết quả test trong notebook:** Pair F1 **70,07%**, Macro Pair F1 **44,84%**, Exact Match **24,91%**. Macro F1 thấp hơn do nhiều aspect hiếm; nên xem dự đoán là gợi ý cần rà soát.
- **Giới hạn aspect:** UIT‑ViSFD không có nhãn nhiệt độ riêng. “Máy nóng” có thể bị gắn với hiệu năng/đánh giá chung hoặc bị bỏ sót; không thể coi đây là phân loại nhiệt độ chuyên biệt.
- **Dashboard:** phản ánh batch người dùng nạp trong phiên hiện tại, không đại diện toàn thị trường nếu chưa cung cấp đủ dữ liệu.
    """)
    st.info("UIT‑ViSFD được phát hành cho mục đích nghiên cứu. Kiểm tra điều khoản dataset trước khi dùng thương mại hoặc phân phối dữ liệu nguồn.")

st.markdown('<div class="footer-bar">ASPECTLENS AI · VIETNAMESE ASPECT BASED SENTIMENT ANALYSIS · UIT‑ViSFD</div>', unsafe_allow_html=True)
