from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cpchild.data import assert_partition_integrity
from cpchild.features import HAND_DIM, extract_hand, hand_feature_names
from cpchild.metrics import binary_metrics, select_threshold


def small_manifest():
    return pd.DataFrame(dict(path=["a", "b", "c", "d"], phase=["train", "train", "validation", "test"],
                             pixel_sha256=["a", "b", "c", "d"], similarity_group=["a", "b", "c", "d"],
                             target=[0, 1, 1, 0], label=["Non-Polyp", "Polyp", "Polyp", "Non-Polyp"]))


def test_partition_rejects_same_pixels_across_validation():
    frame = small_manifest()
    frame.loc[2, "pixel_sha256"] = "a"
    with pytest.raises(ValueError, match="pixel_sha256 leakage"):
        assert_partition_integrity(frame)


def test_partition_rejects_near_duplicate_group_leakage():
    frame = small_manifest()
    frame.loc[3, "similarity_group"] = "b"
    with pytest.raises(ValueError, match="similarity_group leakage"):
        assert_partition_integrity(frame)


def test_partition_rejects_inverted_labels():
    frame = small_manifest()
    frame.loc[1, "target"] = 0
    with pytest.raises(ValueError, match="labels and numeric targets"):
        assert_partition_integrity(frame)


def test_clean_partition_passes():
    assert_partition_integrity(small_manifest())


def test_threshold_ties_and_ge_convention():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([-.7, .1, .1, .9])
    threshold = select_threshold(labels, scores)
    assert threshold == .1
    metrics = binary_metrics(labels, scores, threshold)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 0, 2)
    assert metrics["f1"] == pytest.approx(.8)


def test_majority_accuracy_does_not_hide_zero_recall():
    metrics = binary_metrics(np.array([0, 0, 0, 1]), np.zeros(4), 1.0)
    assert metrics["accuracy"] == .75
    assert metrics["recall"] == 0.0
    assert metrics["specificity"] == 1.0
    assert metrics["balanced_accuracy"] == .5


def test_threshold_rejects_nonfinite_or_one_class():
    with pytest.raises(ValueError):
        select_threshold([0, 1], [0., np.nan])
    with pytest.raises(ValueError):
        select_threshold([0, 0], [0., 1.])


def test_hand_features_are_finite_on_uniform_and_textured_images():
    flat = Image.new("RGB", (256, 256), (160, 80, 70))
    pattern = np.random.default_rng(42).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    first, second = extract_hand(flat), extract_hand(Image.fromarray(pattern))
    assert first.shape == second.shape == (HAND_DIM,)
    assert len(set(hand_feature_names())) == HAND_DIM
    assert np.isfinite(first).all() and np.isfinite(second).all()
    assert not np.allclose(first, second)
