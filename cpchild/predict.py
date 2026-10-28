"""Offline inference for saved local models; SVM scores are not probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from .classical import decision_scores
from .features import (extract_hand, feature_matrix, image_tensor,
                       pooled_forward, stem_forward)


class Predictor:
    def __init__(self, model_path, threads=6, device="auto"):
        self.model_path = Path(model_path).resolve()
        self.output = self.model_path.parent.parent
        self.network = None
        self.device = "cpu"
        if self.model_path.suffix == ".pt":
            import torch
            from torchvision.models import resnet18
            torch.set_num_threads(threads)
            self.checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
            self.method, self.threshold = "M4_cnn_finetune", self.checkpoint["threshold"]
            self.size = self.checkpoint["image_size"]
            self.network = resnet18(weights=None)
            self.network.fc = torch.nn.Linear(512, 2)
            self.network.load_state_dict(self.checkpoint["state_dict"])
            self.network.eval()
            self.score_type = "softmax_positive_class"
        else:
            self.bundle = joblib.load(self.model_path)
            self.method, self.threshold = self.bundle["method"], self.bundle["threshold"]
            self.size, self.score_type = self.bundle["image_size"], self.bundle["score_type"]
            if self.method in ("M3_cnn_svm", "M5_fusion_svm"):
                import torch
                from torchvision.models import resnet18
                torch.set_num_threads(threads)
                self.network = resnet18(weights=None)
                self.network.load_state_dict(torch.load(self.output / "models" / "frozen_resnet18.pt", map_location="cpu", weights_only=True))
                self.network.eval()
        if self.network is not None:
            import torch
            self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
            self.network.to(self.device)

    def scores(self, images):
        converted = [image.convert("RGB") for image in images]
        if self.method == "M4_cnn_finetune":
            import torch
            with torch.inference_mode():
                batch = torch.stack([image_tensor(image, self.size) for image in converted]).to(self.device)
                stem = stem_forward(self.network, batch).to(torch.float16).to(torch.float32)
                logits = self.network.fc(pooled_forward(self.network, stem))
                return logits.softmax(dim=1)[:, 1].cpu().numpy().astype(float)
        hand = np.stack([extract_hand(image) for image in converted])
        cnn = None
        if self.network is not None:
            import torch
            with torch.inference_mode():
                batch = torch.stack([image_tensor(image, self.size) for image in converted]).to(self.device)
                cnn = pooled_forward(self.network, stem_forward(self.network, batch)).cpu().numpy()
        matrix = feature_matrix(hand, cnn, self.method)
        return decision_scores(self.bundle["estimator"], matrix, self.method)

    def predict(self, paths):
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        scores = self.scores(images)
        return [dict(path=str(Path(path).resolve()), model=self.method, score=float(score),
                     score_type=self.score_type, threshold=float(self.threshold),
                     prediction=int(score >= self.threshold),
                     label="Polyp" if score >= self.threshold else "Non-Polyp") for path, score in zip(paths, scores)]


def main():
    parser = argparse.ArgumentParser(description="Predict CP-CHILD labels with a saved model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = Predictor(args.model).predict(args.images)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
