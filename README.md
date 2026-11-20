# AspectLens AI — Vietnamese Aspect-Based Sentiment Analysis

Web app phân tích bình luận đánh giá smartphone tiếng Việt theo từng khía cạnh: pin, camera, màn hình, hiệu năng, giá cả và các khía cạnh khác. App trả sắc thái tích cực / trung lập / tiêu cực và dashboard xếp hạng các khía cạnh bị phàn nàn trong batch người dùng cung cấp.

**Web demo:** đang triển khai trên Streamlit Community Cloud.

## Chạy ứng dụng

Python 3.11 hoặc 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Ứng dụng cần `results.zip` ở thư mục gốc. File này chứa model SLP PyTorch và vectorizer TF-IDF đã train; không cần train lại để chạy demo. Import dùng pickle của scikit-learn, chỉ nạp artifact tin cậy do notebook project xuất ra.

## Tính năng

- Phân tích một bình luận và hiển thị khía cạnh, sentiment và điểm softmax.
- Phân tích tối đa 5.000 bình luận bằng cách dán từng dòng hoặc tải CSV có cột `comment`, `review`, `text` hoặc `content`.
- Dashboard tổng hợp lượt nhắc khía cạnh, số lượt tiêu cực, tỷ lệ tiêu cực và ví dụ bình luận phàn nàn.
- Tải kết quả phân tích batch ra CSV.

Dashboard chỉ tổng hợp batch người dùng đang phân tích trong phiên hiện tại. Nó không tự phản ánh toàn bộ 11.122 bình luận của dataset hoặc dữ liệu khách hàng thời gian thực.

## Mô hình và dữ liệu

- Dataset: [UIT‑ViSFD](https://huggingface.co/datasets/visolex/ViSFD), 11.122 đánh giá smartphone tiếng Việt và 10 khía cạnh.
- Notebook Kaggle dùng split Train/Dev/Test gốc; TF-IDF chỉ fit trên Train; model là SLP một tầng tuyến tính.
- Kết quả test trong notebook: Pair F1 70,07%, Macro Pair F1 44,84%, Exact Match 24,91%.
- UIT‑ViSFD không có khía cạnh nhiệt độ riêng. “Máy nóng” có thể được gắn với hiệu năng/đánh giá chung hoặc bị bỏ sót.
- Dataset được phát hành cho mục đích nghiên cứu; kiểm tra điều khoản trước khi dùng thương mại.

## Cấu trúc

```text
app.py                 # Streamlit UI và dashboard
src/schema.py          # Aspect và sentiment labels
src/model.py           # SLP PyTorch
src/inference.py       # Nạp artifact và dự đoán
src/dashboard.py       # Tổng hợp số liệu batch
src/dataset.py         # Đọc CSV bình luận
results.zip            # Model, vectorizer, history từ Kaggle
ABSA_UIT_ViSFD_Kaggle_10LR_vs_MLP.ipynb  # Notebook thí nghiệm
```
