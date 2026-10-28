"""Validation-only selection and persisted classical/hybrid estimators."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import save_json
from .features import HAND_DIM, feature_matrix
from .metrics import binary_metrics, select_threshold


def make_estimator(method, params, cfg):
    if method == "M0_majority":
        return DummyClassifier(strategy="most_frequent", random_state=cfg["seed"])
    if method == "M2_hand_rf":
        return RandomForestClassifier(n_estimators=cfg["rf_estimators"], class_weight="balanced_subsample",
                                      random_state=cfg["seed"], n_jobs=cfg["torch_threads"], **params)
    if method == "M5_fusion_svm":
        scaler = ColumnTransformer(
            [("hand", StandardScaler(), slice(0, HAND_DIM)),
             ("cnn", StandardScaler(), slice(HAND_DIM, HAND_DIM + 512))],
            transformer_weights={"hand": 1 / np.sqrt(HAND_DIM), "cnn": 1 / np.sqrt(512)})
    else:
        scaler = StandardScaler()
    return Pipeline([("scaler", scaler), ("classifier", SVC(cache_size=384, **params))])


def parameter_candidates(method, cfg):
    if method == "M0_majority":
        return [{}]
    if method == "M2_hand_rf":
        return [dict(min_samples_leaf=leaf) for leaf in cfg["rf_min_samples_leaf"]]
    return [dict(C=value, kernel=kernel, gamma="scale", class_weight="balanced")
            for kernel in ("linear", "rbf") for value in cfg["svm_c"]] + [
                dict(C=1.0, kernel="rbf", gamma="scale", class_weight=None)]


def decision_scores(estimator, features, method):
    if method == "M0_majority":
        return np.zeros(len(features), dtype=float)
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(features), dtype=float)
    position = list(estimator.classes_).index(1)
    return np.asarray(estimator.predict_proba(features)[:, position], dtype=float)


def selection_path(output, dataset, method):
    return Path(output) / "selection" / f"{dataset}__{method}.json"


def prediction_frame(frame, ids, scores, threshold):
    result = frame.iloc[ids][["image_index", "path", "dataset", "label", "target", "phase", "unique_test", "similarity_group"]].copy()
    result["score"] = scores
    result["threshold"] = float(threshold)
    result["prediction"] = (np.asarray(scores) >= threshold).astype(int)
    return result


def fit_classical(cfg, frame, protocol, methods=None, log=print):
    output = Path(cfg["output_dir"])
    hand = np.load(output / "features" / "hand_features.npy", mmap_mode="r")
    cnn_path = output / "features" / "cnn_features.npy"
    cnn = np.load(cnn_path, mmap_mode="r") if cnn_path.exists() else None
    methods = methods or [m for m in cfg["models"] if m != "M4_cnn_finetune"]
    (output / "selection").mkdir(exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)
    for dataset in sorted(frame.dataset.unique()):
        train_ids = np.flatnonzero((frame.dataset == dataset) & (frame.phase == "train"))
        val_ids = np.flatnonzero((frame.dataset == dataset) & (frame.phase == "validation"))
        train_y, val_y = frame.target.to_numpy()[train_ids], frame.target.to_numpy()[val_ids]
        for method in methods:
            meta_path = selection_path(output, dataset, method)
            model_path = output / "models" / f"{dataset}__{method}.joblib"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta["fingerprint"] != protocol["fingerprint"] or not model_path.exists():
                    raise ValueError(f"Invalid resumed model: {dataset} {method}")
                log(f"Reuse fitted {dataset} {method}")
                continue
            if method in ("M3_cnn_svm", "M5_fusion_svm") and cnn is None:
                raise ValueError("CNN feature extraction must complete before hybrid training")
            matrix = feature_matrix(hand, cnn, method)
            train_x, val_x = matrix[train_ids], matrix[val_ids]
            trials, best = [], None
            start = time.monotonic()
            for params in parameter_candidates(method, cfg):
                estimator = make_estimator(method, params, cfg)
                estimator.fit(train_x, train_y)
                scores = decision_scores(estimator, val_x, method)
                threshold = 1.0 if method == "M0_majority" else select_threshold(val_y, scores)
                metrics = binary_metrics(val_y, scores, threshold)
                trials.append(dict(params=json.dumps(params, sort_keys=True), **metrics))
                log(f"Validation {dataset} {method} {params}: AP={metrics['average_precision']:.4f}, F1={metrics['f1']:.4f}")
                rank = (metrics["average_precision"], metrics["f1"])
                if best is None or rank > best[0]:
                    best = (rank, estimator, params, threshold, metrics, scores)
            _, estimator, params, threshold, metrics, scores = best
            bundle = dict(estimator=estimator, method=method, dataset=dataset, threshold=threshold,
                          fingerprint=protocol["fingerprint"], feature_dimension=int(matrix.shape[1]),
                          image_size=cfg["image_size"], labels={0: "Non-Polyp", 1: "Polyp"},
                          score_type="decision_function" if hasattr(estimator, "decision_function") else "positive_probability")
            joblib.dump(bundle, model_path, compress=3)
            pd.DataFrame(trials).to_csv(meta_path.with_suffix(".trials.csv"), index=False, encoding="utf-8-sig")
            prediction_frame(frame, val_ids, scores, threshold).to_csv(
                output / "predictions" / f"{dataset}__{method}__validation.csv", index=False, encoding="utf-8-sig")
            save_json(meta_path, dict(fingerprint=protocol["fingerprint"], dataset=dataset, method=method,
                selected_parameters=params, threshold=threshold, validation=metrics, train_n=len(train_ids),
                train_positives=int(train_y.sum()), selection="AP, then positive F1, on fixed validation split only",
                model_path=model_path.relative_to(output).as_posix(), training_seconds=time.monotonic() - start))
            log(f"Saved {dataset} {method}; test data not scored")
