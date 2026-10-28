"""Immutable data manifests and conservative similarity-group holdouts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    project = path.parent.parent
    for name in ("dataset_root", "audit_dir", "output_dir"):
        cfg[name] = str((project / cfg[name]).resolve())
    cfg["config_path"] = str(path)
    cfg["project_root"] = str(project)
    return cfg


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class UnionFind:
    def __init__(self, count):
        self.parents = list(range(count))

    def find(self, i):
        while self.parents[i] != i:
            self.parents[i] = self.parents[self.parents[i]]
            i = self.parents[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parents[max(a, b)] = min(a, b)


def assert_partition_integrity(frame):
    usable = frame[frame.phase.isin(["train", "validation", "test"])]
    if not usable.path.is_unique:
        raise ValueError("A path appears more than once in the manifest")
    for group_column in ("pixel_sha256", "similarity_group"):
        sets = {phase: set(usable.loc[usable.phase == phase, group_column])
                for phase in ("train", "validation", "test")}
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if sets[a] & sets[b]:
                raise ValueError(f"{group_column} leakage across {a}/{b}")
    if set(usable.target.unique()) != {0, 1}:
        raise ValueError("Expected binary labels Non-Polyp=0 and Polyp=1")
    expected = (usable.label == "Polyp").astype(int)
    if not np.array_equal(expected, usable.target):
        raise ValueError("Folder labels and numeric targets disagree")


def prepare(cfg, log=print, verify_sources=False):
    audit, output = Path(cfg["audit_dir"]), Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists() and (output / "split_manifest.csv").exists() and not verify_sources:
        previous = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        if previous["config"] != cfg:
            raise ValueError("Configuration changed; choose a new output_dir")
        frame = read_manifest(cfg)
        records = frame[["path", "file_sha256", "pixel_sha256", "target", "phase", "similarity_group", "unique_test"]].to_dict("records")
        if fingerprint({"config": cfg, "manifest": records}) != previous["fingerprint"]:
            raise ValueError("The persisted split manifest changed after the protocol was fixed")
        log("Reuse verified source inventory and immutable split; --verify-source forces a full source recheck")
        return frame, previous
    original = pd.read_csv(audit / "image_manifest.csv", keep_default_na=False)
    original = original.sort_values("path").reset_index(drop=True)
    if (original.error != "").any():
        raise ValueError("Audit contains image decoding failures")
    original["image_index"] = np.arange(len(original))
    original["target"] = original.target.astype(int)
    recommendations = pd.read_csv(audit / "proposed_exact_duplicate_handling.csv", keep_default_na=False)
    if set(original.path) != set(recommendations.path) or not recommendations.path.is_unique:
        raise ValueError("Audit and duplicate-handling manifests disagree")
    original = original.merge(recommendations[["path", "recommendation"]], on="path", validate="one_to_one")
    root = Path(cfg["dataset_root"])

    def verify(row):
        path = (root / row.path).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Image path leaves dataset root: {row.path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row.file_sha256:
            raise ValueError(f"Data changed since audit: {row.path}; rerun the audit")

    log(f"Verifying source hashes for {len(original)} images")
    with ThreadPoolExecutor(max_workers=4) as pool:
        for completed, _ in enumerate(pool.map(verify, original.itertuples(index=False)), 1):
            if completed % 1000 == 0 or completed == len(original):
                log(f"Source hash verification {completed}/{len(original)}")

    union = UnionFind(len(original))
    for ids in original.groupby("pixel_sha256").indices.values():
        for idx in ids[1:]:
            union.union(int(ids[0]), int(idx))
    paths = dict(zip(original.path, original.image_index))
    candidates = pd.read_csv(audit / "near_duplicate_candidates.csv", keep_default_na=False)
    for pair in candidates.itertuples(index=False):
        union.union(paths[pair.path_1], paths[pair.path_2])
    original["similarity_group"] = [f"similarity_{union.find(i):05d}" for i in range(len(original))]
    original["phase"] = "excluded"
    original.loc[original.split == "Test", "phase"] = "test"
    original["unique_test"] = (original.split == "Test") & (original.recommendation == "keep")
    for dataset in sorted(original.dataset.unique()):
        pool = original[(original.dataset == dataset) & (original.split == "Train") &
                        (original.recommendation == "keep")]
        splitter = StratifiedGroupKFold(n_splits=cfg["validation_folds"], shuffle=True, random_state=cfg["seed"])
        folds = list(splitter.split(pool.path, pool.target, groups=pool.similarity_group))
        train, val = folds[cfg["validation_fold"]]
        original.loc[pool.index[train], "phase"] = "train"
        original.loc[pool.index[val], "phase"] = "validation"
    assert_partition_integrity(original)
    if len(original) != 9500 or sum(original.phase == "test") != 1400:
        raise ValueError("Unexpected dataset size; this protocol targets the audited CP-CHILD copy")
    if sum(original.phase.isin(["train", "validation"])) != 8002:
        raise ValueError("Expected 8,002 deduplicated development images")
    if int(original.unique_test.sum()) != 1376:
        raise ValueError("Expected 1,376 unique test images")
    records = original[["path", "file_sha256", "pixel_sha256", "target", "phase", "similarity_group", "unique_test"]].to_dict("records")
    run_fingerprint = fingerprint({"config": cfg, "manifest": records})
    protocol = dict(fingerprint=run_fingerprint, config=cfg, image_count=len(original),
                    label_mapping={"Non-Polyp": 0, "Polyp": 1}, patient_ids_available=False,
                    near_duplicate_policy="All 74 candidate pairs conservatively grouped; no deletions based on perceptual hashes",
                    test_policy="Official 1400-image test and 1376-unique-image sensitivity analysis; all model choices locked before test scoring",
                    counts=original.groupby(["dataset", "phase", "label"]).size().rename("count").reset_index().to_dict("records"))
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous["fingerprint"] != run_fingerprint:
            raise ValueError("An existing output directory has a different protocol. Choose a new output_dir.")
    else:
        save_json(protocol_path, protocol)
    original.to_csv(output / "split_manifest.csv", index=False, encoding="utf-8-sig")
    log(f"Prepared immutable split: {dict(Counter(original.phase))}")
    return original, protocol


def read_manifest(cfg):
    output = Path(cfg["output_dir"])
    frame = pd.read_csv(output / "split_manifest.csv", keep_default_na=False)
    frame["unique_test"] = frame.unique_test.astype(str).str.lower().eq("true")
    assert_partition_integrity(frame)
    return frame
