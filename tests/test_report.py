"""Report verification selects devices, releases models and rejects mismatches."""

import json
import sys
import weakref
from types import SimpleNamespace

import pandas as pd
import pytest

from cpchild import predict
from cpchild.report import verify_saved_inference


@pytest.mark.parametrize("score_error", [0.0, 0.1])
def test_report_checks_select_gpu_and_release_models(tmp_path, monkeypatch, score_error):
    methods = ("M1_hand_svm", "M3_cnn_svm", "M4_cnn_finetune", "M5_fusion_svm")
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    for dataset in ("CP-CHILD-A", "CP-CHILD-B"):
        for method in methods:
            pd.DataFrame(dict(path=["negative.jpg", "positive.jpg"], target=[0, 1],
                              score=[0.1, 0.9], prediction=[0, 1])).to_csv(
                prediction_dir / f"{dataset}__{method}__test.csv", index=False)

    instances = []

    class SequentialPredictor:
        def __init__(self, model_path, threads=6, device="auto"):
            self.device = "cuda" if device == "auto" else device
            assert self.device == "cuda", "Report must allow the predictor to select the GPU"
            assert all(reference() is None for reference in instances), "Previous model is still live"
            instances.append(weakref.ref(self))

        def predict(self, paths):
            return [dict(score=score + score_error, prediction=label)
                    for score, label in ((0.1, 0), (0.9, 1))]

    monkeypatch.setattr(predict, "Predictor", SequentialPredictor)
    cfg = dict(dataset_root=str(tmp_path))
    result_path = tmp_path / "inference_roundtrip_checks.csv"
    if score_error:
        with pytest.raises(ValueError, match="Saved-model inference mismatch"):
            verify_saved_inference(cfg, None, tmp_path, log=lambda message: None)
        assert not result_path.exists()
    else:
        verify_saved_inference(cfg, None, tmp_path, log=lambda message: None)
        checks = pd.read_csv(result_path)
        assert len(checks) == 16
        assert checks.label_matches.all()
        assert checks.absolute_error.max() == 0.0
        assert len(instances) == 8
        assert all(reference() is None for reference in instances)


@pytest.mark.parametrize("cuda_available", [True, False])
def test_report_startup_records_available_device(tmp_path, monkeypatch, cuda_available):
    import run_experiment as runner
    from cpchild import report

    cfg = dict(output_dir=str(tmp_path))
    (tmp_path / "protocol.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--stage", "report"])
    monkeypatch.setattr(runner, "load_config", lambda path: cfg)
    monkeypatch.setattr(runner, "read_manifest", lambda config: pd.DataFrame())
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda package: "test-version")
    cuda = SimpleNamespace(is_available=lambda: cuda_available,
                           get_device_name=lambda index: "Test GPU",
                           get_device_properties=lambda index: SimpleNamespace(total_memory=6 * 1024**3))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="12.1")))
    calls = []
    monkeypatch.setattr(report, "build_report", lambda *args: calls.append(args))

    runner.main()

    assert len(calls) == 1
    environment = json.loads((tmp_path / "environment_report.json").read_text(encoding="utf-8"))
    assert environment["torch_runtime"] == dict(cuda_compiled="12.1", cuda_available=cuda_available,
        device="Test GPU" if cuda_available else "cpu",
        total_gpu_memory_bytes=6 * 1024**3 if cuda_available else None)
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "complete"
