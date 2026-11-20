"""Aggregate aspect sentiment over the currently loaded comment batch."""
import pandas as pd
from .schema import ASPECT_NAMES, STATE_NAMES

def prediction_frames(predictions):
    rows = []
    for comment_id, item in enumerate(predictions):
        for aspect in item["aspects"]:
            rows.append({"comment_id": comment_id, "comment": item["comment"],
                         "aspect": aspect["aspect"], "aspect_vi": aspect["aspect_vi"],
                         "sentiment": aspect["sentiment"], "sentiment_vi": STATE_NAMES[aspect["sentiment"]],
                         "confidence": aspect["confidence"]})
    cols = ["comment_id", "comment", "aspect", "aspect_vi", "sentiment", "sentiment_vi", "confidence"]
    detail = pd.DataFrame(rows, columns=cols)
    if detail.empty:
        return detail, pd.DataFrame(columns=["Mã khía cạnh", "Khía cạnh", "Số lượt đề cập", "Tiêu cực", "Tỷ lệ tiêu cực (%)", "Tích cực", "Trung lập"])
    grouped = detail.groupby(["aspect", "aspect_vi", "sentiment"]).size().unstack(fill_value=0)
    for state in ["Negative", "Neutral", "Positive"]:
        if state not in grouped: grouped[state] = 0
    counts = grouped[["Negative", "Neutral", "Positive"]].sum(axis=1).clip(lower=1)
    summary = pd.DataFrame({"Mã khía cạnh": grouped.index.get_level_values("aspect").to_numpy(),
        "Khía cạnh": grouped.index.get_level_values("aspect_vi").to_numpy(),
        "Số lượt đề cập": counts.to_numpy(), "Tiêu cực": grouped["Negative"].to_numpy(),
        "Tỷ lệ tiêu cực (%)": (100 * grouped["Negative"] / counts).round(1).to_numpy(),
        "Tích cực": grouped["Positive"].to_numpy(), "Trung lập": grouped["Neutral"].to_numpy()})
    rank = {aspect: i for i, aspect in enumerate(ASPECT_NAMES)}
    summary["_order"] = summary["Mã khía cạnh"].map(rank)
    return detail, summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)
