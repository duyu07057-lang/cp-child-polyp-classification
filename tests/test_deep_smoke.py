"""End-to-end training regression for Windows int32 labels and checkpoint reload."""

import json

import numpy as np
import pandas as pd
import pytest


def test_two_stage_cnn_training_updates_tail_and_reloads(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from torchvision.models import resnet18
    from cpchild.deep import cached_scores, fit_deep

    # Keep this small regression independent of the machine's GPU and external downloads.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.set_num_threads(1)
    torch.manual_seed(5)
    frozen = resnet18(weights=None)
    (tmp_path / "models").mkdir()
    (tmp_path / "features").mkdir()
    torch.save(frozen.state_dict(), tmp_path / "models" / "frozen_resnet18.pt")
    rng = np.random.default_rng(5)
    values = np.abs(rng.normal(size=(8, 256, 14, 14))).astype(np.float16)
    np.save(tmp_path / "features" / "original_stem.npy", values)
    np.save(tmp_path / "features" / "augmented_train_stem.npy", values[:6] * .95)
    np.save(tmp_path / "features" / "augmented_train_indices.npy", np.arange(6))
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
    frame = pd.DataFrame(dict(image_index=np.arange(8), path=[f"{i}.jpg" for i in range(8)],
        dataset=["CP-CHILD-A"] * 8, label=np.where(labels, "Polyp", "Non-Polyp"), target=labels,
        phase=["train"] * 6 + ["validation"] * 2, unique_test=[False] * 8,
        similarity_group=[str(i) for i in range(8)]))
    cfg = dict(output_dir=str(tmp_path), project_root=str(tmp_path), seed=42, torch_threads=1,
        image_size=224, batch_size=2, cnn_backbone_lr=1e-4, cnn_head_lr=1e-3,
        cnn_weight_decay=1e-4, cnn_max_epochs=2, cnn_warmup_epochs=1, cnn_patience=3)
    fit_deep(cfg, frame, {"fingerprint": "smoke-int32-labels"}, log=lambda message: None)
    resume = torch.load(tmp_path / "models" / "CP-CHILD-A__M4_cnn_finetune.resume.pt", weights_only=False)
    assert resume["next_epoch"] == 2
    assert not torch.equal(resume["model"]["layer4.0.conv1.weight"], frozen.state_dict()["layer4.0.conv1.weight"])
    best = torch.load(tmp_path / "models" / "CP-CHILD-A__M4_cnn_finetune.pt", weights_only=True)
    reloaded = resnet18(weights=None)
    reloaded.fc = torch.nn.Linear(512, 2)
    reloaded.load_state_dict(best["state_dict"])
    scores = cached_scores(reloaded, values, np.array([6, 7]), cfg, torch.device("cpu"))
    saved = pd.read_csv(tmp_path / "predictions" / "CP-CHILD-A__M4_cnn_finetune__validation.csv")
    assert np.allclose(scores, saved.score, atol=1e-7)
    meta = json.loads((tmp_path / "selection" / "CP-CHILD-A__M4_cnn_finetune.json").read_text())
    assert meta["selected_parameters"]["epochs_run"] == 2
