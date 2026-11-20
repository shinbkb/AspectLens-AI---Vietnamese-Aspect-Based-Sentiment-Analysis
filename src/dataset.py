"""Input helpers for comment text and user-provided CSV batches."""
import pandas as pd

def read_comments_csv(file):
    frame = pd.read_csv(file)
    if frame.empty:
        raise ValueError("File CSV không có dòng dữ liệu.")
    candidates = [c for c in frame.columns if str(c).strip().lower() in {"comment", "review", "text", "content", "binh_luan", "bình luận"}]
    if not candidates:
        raise ValueError("Không nhận diện được cột bình luận. Đặt tên cột là comment, review hoặc text.")
    comment_col = candidates[0]
    valid = frame[comment_col].fillna("").astype(str).str.strip().ne("")
    if not valid.any():
        raise ValueError("Không tìm thấy bình luận nào trong file.")
    return frame.loc[valid].reset_index(drop=True), comment_col
