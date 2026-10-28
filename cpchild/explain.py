"""Occlusion sensitivity of the selected model's own final score."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .predict import Predictor
from .fonts import image_font


def occlusion_gallery(cfg, dataset, method, destination):
    output = Path(cfg["output_dir"])
    suffix = ".pt" if method == "M4_cnn_finetune" else ".joblib"
    predictor = Predictor(output / "models" / f"{dataset}__{method}{suffix}")
    predictions = pd.read_csv(output / "predictions" / f"{dataset}__{method}__test.csv")
    errors = predictions[predictions.target != predictions.prediction]
    selected = pd.concat([errors[errors.target == 1].sort_values("score").head(2),
                          errors[errors.target == 0].sort_values("score", ascending=False).head(2)])
    if selected.empty:
        selected = predictions.groupby("target", sort=True).head(1)
    font = image_font(15)
    title_font = image_font(20, bold=True)
    canvas = Image.new("RGB", (600, 95 + len(selected) * 310), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 12), f"{dataset} | {method} | 模型自身评分遮挡分析", font=title_font, fill="#17324d")
    draw.text((15, 45), "原图                       对原预测类别的正向支持（遮挡后评分下降）", font=font, fill="#222222")
    records = []
    for ordinal, (_, row) in enumerate(selected.iterrows()):
        with Image.open(Path(cfg["dataset_root"]) / row.path) as source:
            image = source.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        baseline = float(predictor.scores([image])[0])
        sign = 1 if baseline >= predictor.threshold else -1
        fill = tuple(int(value) for value in np.asarray(image).mean(axis=(0, 1)))
        masked = []
        for grid_row in range(4):
            for grid_column in range(4):
                variant = image.copy()
                ImageDraw.Draw(variant).rectangle((grid_column * 64, grid_row * 64,
                    (grid_column + 1) * 64 - 1, (grid_row + 1) * 64 - 1), fill=fill)
                masked.append(variant)
        scores = predictor.scores(masked)
        changes = sign * (baseline - scores)
        heat = np.maximum(changes.reshape(4, 4), 0)
        heat = heat / max(float(heat.max()), 1e-12)
        heat_image = Image.fromarray(np.uint8(255 * heat)).resize((256, 256), Image.Resampling.BILINEAR)
        colors = matplotlib.colormaps["inferno"](np.asarray(heat_image) / 255.0)[:, :, :3]
        overlay = Image.blend(image, Image.fromarray(np.uint8(colors * 255)), .45)
        y = 85 + ordinal * 310
        canvas.paste(image, (15, y))
        canvas.paste(overlay, (315, y))
        caption = f"真实：{'息肉' if row.target else '非息肉'} | 预测：{'息肉' if sign==1 else '非息肉'} | {Path(row.path).name}"
        draw.text((15, y + 259), caption, font=font, fill="#222222")
        draw.text((15, y + 282), "4×4 平均色遮挡；仅为敏感性分析，不能作为病灶定位证据。", font=font, fill="#555555")
        for index, score in enumerate(scores):
            records.append(dict(path=row.path, target=int(row.target), original_prediction=int(sign==1),
                score_type=predictor.score_type, original_score=baseline, grid_row=index // 4,
                grid_column=index % 4, occluded_score=float(score),
                contribution_to_original_prediction=float(changes[index])))
    canvas.save(destination)
    pd.DataFrame(records).to_csv(destination.with_suffix(".csv"), index=False, encoding="utf-8-sig")
