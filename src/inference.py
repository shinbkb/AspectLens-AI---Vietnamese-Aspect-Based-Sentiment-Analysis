"""Load trusted Kaggle export and run CPU inference."""
import io
import pickle
import zipfile
from pathlib import Path

import numpy as np
import torch
from .model import AspectSentimentSLP
from .schema import ASPECT_NAMES

class AspectPredictor:
    def __init__(self, artifact_zip):
        artifact_zip = Path(artifact_zip)
        if not artifact_zip.is_file():
            raise FileNotFoundError(f"Thiếu artifact {artifact_zip}. Cần results.zip từ Kaggle.")
        # This ZIP is the user's own Kaggle export; only load trusted pickle artifacts.
        with zipfile.ZipFile(artifact_zip) as archive:
            self.checkpoint = torch.load(io.BytesIO(archive.read("absa_best_model.pt")), map_location="cpu", weights_only=True)
            self.vectorizer = pickle.loads(archive.read("absa_tfidf.pkl"))
        self.aspects = self.checkpoint["aspects"]
        self.states = self.checkpoint["states"]
        if set(self.aspects) != set(ASPECT_NAMES):
            raise ValueError("Aspect trong checkpoint không khớp UIT-ViSFD.")
        if self.states != ["Absent", "Negative", "Neutral", "Positive"]:
            raise ValueError("Các nhãn sentiment trong checkpoint không khớp.")
        input_dim = int(self.checkpoint["features"])
        if len(self.vectorizer.vocabulary_) != input_dim:
            raise ValueError("Vectorizer và model không cùng số chiều.")
        self.model = AspectSentimentSLP(input_dim, len(self.aspects), len(self.states))
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def predict_many(self, comments, batch_size=32):
        results = []
        for start in range(0, len(comments), batch_size):
            chunk = comments[start:start + batch_size]
            dense = self.vectorizer.transform(chunk).toarray().astype(np.float32, copy=False)
            probabilities = self.model(torch.from_numpy(dense)).softmax(dim=-1).cpu().numpy()
            predicted = probabilities.argmax(axis=-1)
            for row, comment in enumerate(chunk):
                aspects = []
                for col, aspect in enumerate(self.aspects):
                    state_id = int(predicted[row, col])
                    state = self.states[state_id]
                    if state == "Absent":
                        continue
                    aspects.append({"aspect": aspect, "aspect_vi": ASPECT_NAMES[aspect],
                                    "sentiment": state, "confidence": float(probabilities[row, col, state_id])})
                results.append({"comment": comment, "aspects": aspects})
        return results
