import json
from pathlib import Path

import pandas as pd
import pytest

from cpchild.data import save_json
from cpchild.evaluate import lock_selection


def create_selections(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "frozen_resnet18.pt").write_bytes(b"fixed-pretrained-weights")
    cfg = dict(output_dir=str(tmp_path), project_root=str(tmp_path), models=["M1_hand_svm"])
    frame = pd.DataFrame({"dataset": ["CP-CHILD-A", "CP-CHILD-B"]})
    for dataset in frame.dataset:
        artifact = f"models/{dataset}.joblib"
        (tmp_path / artifact).write_bytes(b"fixed-estimator")
        save_json(tmp_path / "selection" / f"{dataset}__M1_hand_svm.json", dict(
            fingerprint="fixed-data", dataset=dataset, method="M1_hand_svm", model_path=artifact,
            threshold=.3, validation={"average_precision": .8, "f1": .7}))
    return cfg, frame, {"fingerprint": "fixed-data"}


def test_cannot_unlock_test_before_all_validation_choices_exist(tmp_path):
    cfg, frame, protocol = create_selections(tmp_path)
    (tmp_path / "selection" / "CP-CHILD-B__M1_hand_svm.json").unlink()
    with pytest.raises(ValueError, match="Test remains locked"):
        lock_selection(cfg, frame, protocol)
    assert not (tmp_path / "selection_lock.json").exists()


def test_model_or_threshold_change_after_lock_is_rejected(tmp_path):
    cfg, frame, protocol = create_selections(tmp_path)
    lock, _ = lock_selection(cfg, frame, protocol)
    assert lock["test_used_for_selection"] is False
    assert lock["cross_device_selected_model"] == "M1_hand_svm"
    lock_selection(cfg, frame, protocol)  # Idempotent resume.
    (tmp_path / "models" / "CP-CHILD-A.joblib").write_bytes(b"retuned-after-test")
    with pytest.raises(ValueError, match="changed after selection locking"):
        lock_selection(cfg, frame, protocol)
