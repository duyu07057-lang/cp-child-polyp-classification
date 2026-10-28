"""Lock every model/threshold before any test scoring, then evaluate once."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .classical import decision_scores, prediction_frame, selection_path
from .data import save_json
from .features import configure_torch, feature_matrix
from .metrics import binary_metrics


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lock_selection(cfg, frame, protocol):
    output = Path(cfg["output_dir"])
    selections, artifacts = {}, {}
    for dataset in sorted(frame.dataset.unique()):
        for method in cfg["models"]:
            path = selection_path(output, dataset, method)
            if not path.exists():
                raise ValueError(f"Test remains locked: missing validation selection for {dataset} {method}")
            meta = json.loads(path.read_text(encoding="utf-8"))
            if meta["fingerprint"] != protocol["fingerprint"]:
                raise ValueError("Test remains locked: inconsistent model protocol")
            key = f"{dataset}__{method}"
            selections[key] = meta
            artifacts[key] = dict(selection_sha256=file_hash(path), model_sha256=file_hash(output / meta["model_path"]),
                                  threshold=meta["threshold"], validation_average_precision=meta["validation"]["average_precision"])
    choices = [meta for meta in selections.values() if meta["dataset"] == "CP-CHILD-A" and
               meta["method"].startswith("M") and meta["method"] != "M0_majority"]
    winner = max(choices, key=lambda item: (item["validation"]["average_precision"], item["validation"]["f1"]))
    backbone_hash = file_hash(output / "models" / "frozen_resnet18.pt")
    lock_path = output / "selection_lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if (lock["fingerprint"] != protocol["fingerprint"] or lock["artifacts"] != artifacts or
                lock["feature_backbone_sha256"] != backbone_hash):
            raise ValueError("A model/threshold changed after selection locking; use a new experiment directory")
    else:
        sources = sorted((Path(cfg["project_root"]) / "cpchild").glob("*.py"))
        lock = dict(fingerprint=protocol["fingerprint"], created_at=datetime.now(timezone.utc).isoformat(),
                    test_used_for_selection=False, artifacts=artifacts,
                    feature_backbone_sha256=backbone_hash,
                    source_hashes={path.name: file_hash(path) for path in sources},
                    cross_device_source="CP-CHILD-A", cross_device_target="CP-CHILD-B",
                    cross_device_selected_model=winner["method"],
                    cross_device_selection_rule="Highest A-validation AP among M1-M5; tie break validation F1; no B training")
        save_json(lock_path, lock)
    return lock, selections


def evaluate(cfg, frame, protocol, log=print):
    output = Path(cfg["output_dir"])
    lock, selections = lock_selection(cfg, frame, protocol)
    log("All model artifacts and validation thresholds locked; test scoring is now enabled")
    hand = np.load(output / "features" / "hand_features.npy", mmap_mode="r")
    cnn = np.load(output / "features" / "cnn_features.npy", mmap_mode="r")
    activation = None
    rows = []

    def score(meta, ids):
        nonlocal activation
        if meta["method"] == "M4_cnn_finetune":
            from torchvision.models import resnet18
            from .deep import cached_scores
            torch = configure_torch(cfg)
            checkpoint = torch.load(output / meta["model_path"], map_location="cpu", weights_only=True)
            network = resnet18(weights=None)
            network.fc = torch.nn.Linear(512, 2)
            network.load_state_dict(checkpoint["state_dict"])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            network.to(device)
            if activation is None:
                activation = np.load(output / "features" / "original_stem.npy", mmap_mode="r")
            return cached_scores(network, activation, ids, cfg, device)
        bundle = joblib.load(output / meta["model_path"])
        matrix = feature_matrix(hand, cnn, meta["method"])
        return decision_scores(bundle["estimator"], matrix[ids], meta["method"])

    def process(meta, target_dataset, experiment):
        method, source_dataset = meta["method"], meta["dataset"]
        ids = np.flatnonzero((frame.dataset == target_dataset) & (frame.phase == "test"))
        tag = f"{source_dataset}__{method}__test" if experiment == "within_device" else f"cross_A_to_B__{method}__test"
        path = output / "predictions" / f"{tag}.csv"
        if path.exists():
            predictions = pd.read_csv(path)
            if predictions.path.tolist() != frame.iloc[ids].path.tolist():
                raise ValueError("Saved test predictions do not match this manifest")
            scores = predictions.score.to_numpy()
            if not np.allclose(predictions.threshold, meta["threshold"], rtol=0, atol=1e-12):
                raise ValueError("Test threshold changed after locking")
        else:
            scores = score(meta, ids)
            predictions = prediction_frame(frame, ids, scores, meta["threshold"])
            predictions.to_csv(path, index=False, encoding="utf-8-sig")
        for policy in ("official", "unique_pixels"):
            mask = np.ones(len(ids), dtype=bool) if policy == "official" else frame.iloc[ids].unique_test.to_numpy(dtype=bool)
            metrics = binary_metrics(frame.iloc[ids].target.to_numpy()[mask], scores[mask], meta["threshold"])
            rows.append(dict(experiment=experiment, train_dataset=source_dataset, test_dataset=target_dataset,
                             model=method, test_policy=policy, **metrics))
        errors = predictions[predictions.target != predictions.prediction].copy()
        errors["error_type"] = np.where(errors.target == 1, "false_negative", "false_positive")
        errors.to_csv(output / "predictions" / f"{tag}__errors.csv", index=False, encoding="utf-8-sig")
        metrics = rows[-2]
        log(f"TEST {source_dataset}->{target_dataset} {method}: AP={metrics['average_precision']:.4f}, Recall={metrics['recall']:.4f}, Specificity={metrics['specificity']:.4f}, F1={metrics['f1']:.4f}")

    for meta in selections.values():
        process(meta, meta["dataset"], "within_device")
    chosen = selections[f"CP-CHILD-A__{lock['cross_device_selected_model']}"]
    process(chosen, "CP-CHILD-B", "cross_device")
    result = pd.DataFrame(rows)
    result.to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
    save_json(output / "metrics.json", result.to_dict("records"))
    save_json(output / "evaluation_completed.json", dict(fingerprint=protocol["fingerprint"],
        completed_at=datetime.now(timezone.utc).isoformat(), rows=len(result),
        selection_lock_sha256=file_hash(output / "selection_lock.json"), metrics_sha256=file_hash(output / "metrics.csv")))
    return result
