# AspectLens AI

**Phân tích cảm xúc theo khía cạnh (Aspect-Based Sentiment Analysis) cho bình luận smartphone tiếng Việt.** AspectLens giúp nhìn nhanh khách hàng đang khen/chê điều gì — như pin, camera, hiệu năng hay giá cả — và tổng hợp các phàn nàn trong tập bình luận bạn cung cấp.

🌐 **Web demo:** [aspectlens-ai.streamlit.app](https://aspectlens-ai.streamlit.app/)

## Mục lục

- [Bài toán](#bài-toán)
- [Tính năng](#tính-năng)
- [Các khía cạnh](#các-khía-cạnh)
- [Dữ liệu và mô hình](#dữ-liệu-và-mô-hình)
- [Kết quả đánh giá](#kết-quả-đánh-giá)
- [Chạy ứng dụng trên máy](#chạy-ứng-dụng-trên-máy)
- [Cấu trúc project](#cấu-trúc-project)
- [Triển khai web](#triển-khai-web)
- [Giới hạn](#giới-hạn)
- [Nguồn và điều khoản](#nguồn-và-điều-khoản)

## Bài toán

Một bình luận thường nhắc đến nhiều khía cạnh với cảm xúc khác nhau. Ví dụ:

> “Pin tốt, camera đẹp nhưng máy rất nóng.”

Mục tiêu là dự đoán một trạng thái riêng cho từng khía cạnh được nhắc tới: **tích cực**, **trung lập** hoặc **tiêu cực**. Các khía cạnh được model xác định là “không đề cập” sẽ không hiện trong kết quả.

## Tính năng

- Phân tích một bình luận và xem khía cạnh, sắc thái cùng điểm softmax của dự đoán.
- Phân tích nhiều bình luận bằng cách dán mỗi bình luận trên một dòng hoặc tải file CSV.
- Tạo dashboard cho batch hiện tại: lượt nhắc, lượt/tỷ lệ tiêu cực, phân bố sắc thái, bảng xếp hạng và ví dụ phàn nàn.
- Tải kết quả phân tích batch dưới dạng CSV.
- Xử lý tối đa **5.000 bình luận mỗi batch** trong giao diện.

### Định dạng CSV

File CSV cần có một cột nội dung bình luận. App nhận diện các tên cột phổ biến như comment, review, text hoặc content (không phân biệt chữ hoa/thường). Mỗi hàng được xem là một bình luận.

## Các khía cạnh

Model được huấn luyện với 10 nhãn aspect của UIT‑ViSFD:

| Nhãn | Tên hiển thị |
|---|---|
| SCREEN | Màn hình |
| CAMERA | Camera |
| FEATURES | Tính năng |
| BATTERY | Pin |
| PERFORMANCE | Hiệu năng |
| STORAGE | Bộ nhớ |
| DESIGN | Thiết kế |
| PRICE | Giá cả |
| GENERAL | Đánh giá chung |
| SER&ACC | Dịch vụ & phụ kiện |

## Dữ liệu và mô hình

- **Dataset:** [UIT‑ViSFD trên Hugging Face](https://huggingface.co/datasets/visolex/ViSFD), gồm 11.122 bình luận smartphone tiếng Việt và 10 khía cạnh.
- **Split dùng trong notebook:** Train 7.786, Dev 1.112, Test 2.224. Có 10 bản trùng chính xác được loại khỏi Train trước khi fit, còn 7.776 mẫu train.
- **Đặc trưng:** TF-IDF ký tự, n-gram 2–5, tối đa 30.000 đặc trưng; vectorizer chỉ được fit trên Train.
- **Classifier:** một tầng tuyến tính (Single-Layer Perceptron / SLP) bằng PyTorch, dự đoán 4 trạng thái cho mỗi khía cạnh: Absent, Negative, Neutral, Positive.
- **Inference:** chạy trên CPU; checkpoint và vectorizer đã train được đóng gói trong results.zip, nên chạy web demo không cần huấn luyện lại.

## Kết quả đánh giá

Các số liệu dưới đây được ghi trong notebook trên Test split:

| Metric | Kết quả |
|---|---:|
| Pair F1 | 70,07% |
| Macro Pair F1 | 44,84% |
| Exact Match | 24,91% |
| State accuracy | 85,95% |

Macro Pair F1 thấp hơn Pair F1 cho thấy hiệu quả chưa đồng đều giữa các khía cạnh; các nhãn hiếm vẫn khó dự đoán. Hãy xem output như gợi ý cần kiểm tra, đặc biệt khi dùng kết quả để ra quyết định sản phẩm.

## Chạy ứng dụng trên máy

Yêu cầu Python 3.11 hoặc 3.12. Từ thư mục gốc project:

~~~bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
~~~

Mở URL local mà Streamlit hiển thị (thường là http://localhost:8501). Cần giữ results.zip cạnh app.py; archive chứa absa_best_model.pt và absa_tfidf.pkl.

## Cấu trúc project

~~~text
.
├── app.py                                  # Giao diện Streamlit, phân tích và dashboard
├── requirements.txt                        # Thư viện runtime
├── results.zip                             # Checkpoint PyTorch, TF-IDF và lịch sử train
├── notebooks/
│   └── ABSA_UIT_ViSFD_Kaggle_10LR_vs_MLP.ipynb # Notebook huấn luyện/đánh giá trên Kaggle
├── .streamlit/config.toml                  # Cấu hình giao diện Streamlit
├── data/README.md                          # Hướng dẫn đặt dữ liệu local (không commit dataset)
├── docs/IMPLEMENTATION.md                  # Chi tiết triển khai, split và artifact
├── artifacts/.gitkeep                      # Nơi lưu artifact sinh khi chạy local
├── scripts/__init__.py                     # Tiện ích project
├── tests/.gitkeep                           # Thư mục dành cho kiểm thử
└── src/
    ├── __init__.py
    ├── schema.py                           # Tên aspect và sentiment
    ├── model.py                            # Kiến trúc SLP khớp checkpoint
    ├── inference.py                        # Nạp artifact và dự đoán
    ├── dataset.py                          # Đọc bình luận từ CSV
    └── dashboard.py                        # Chuyển dự đoán thành bảng thống kê
~~~

## Triển khai web

Demo được host trên **Streamlit Community Cloud**, lấy source từ repository GitHub này, nhánh main, entry point app.py. Các commit mới trên nhánh deploy sẽ được Streamlit dùng để cập nhật ứng dụng.

Gói miễn phí có thể đưa app vào trạng thái ngủ sau thời gian không hoạt động. Khi mở lại, nhấn nút đánh thức app và chờ dịch vụ khởi động. Xem [tài liệu quản lý app của Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).

## Giới hạn

- Dashboard chỉ thống kê những bình luận được dán/tải lên và phân tích trong batch hiện tại; **không tự động đại diện cho toàn bộ 11.122 mẫu hay thị trường**, cũng không thu thập đánh giá trực tiếp từ cửa hàng.
- Dataset không có aspect **nhiệt độ** riêng. Ví dụ “máy nóng” có thể bị gán sang hiệu năng/đánh giá chung hoặc không được phát hiện; model không phải bộ phân loại chuyên biệt cho nhiệt độ.
- Softmax score là điểm của model, chưa được hiệu chuẩn thành xác suất tin cậy.
- Dữ liệu đánh giá smartphone ngoài đời có thể khác dataset huấn luyện; model có thể phân loại sai, nhất là câu mỉa mai, viết tắt, lỗi chính tả hoặc khía cạnh ít mẫu.
- App không lưu dashboard thành cơ sở dữ liệu lâu dài. Nếu bình luận chứa thông tin cá nhân hoặc nhạy cảm, hãy loại bỏ trước khi tải lên dịch vụ web.

## Nguồn và điều khoản

UIT‑ViSFD được cung cấp tại [Hugging Face](https://huggingface.co/datasets/visolex/ViSFD). Hãy đọc điều khoản và giấy phép hiện hành của dataset/model trước khi phân phối lại hoặc sử dụng thương mại. Notebook và các metric trong README phục vụ mục đích mô tả thí nghiệm; chúng không bảo đảm hiệu năng trên dữ liệu mới.
