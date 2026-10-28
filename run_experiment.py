"""Command-line entry point; all substantive choices come from a locked JSON protocol."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from cpchild.data import load_config, prepare, read_manifest, save_json


def log(message):
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="CP-CHILD leakage-controlled classification experiment")
    parser.add_argument("--config", default="configs/cp_child.json")
    parser.add_argument("--verify-source", action="store_true", help="Rehash all source images even when resuming an unchanged manifest")
    parser.add_argument("--stage", default="all", choices=["all", "baseline", "prepare", "hand", "traditional", "cnn-features", "hybrid", "deep", "evaluate", "report"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in ("numpy", "pandas", "Pillow", "scipy", "scikit-learn", "scikit-image", "matplotlib", "joblib", "torch", "torchvision"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    environment = dict(python=sys.version, executable=sys.executable,
                        platform=platform.platform(), cpu_count=os.cpu_count(), versions=versions)
    if versions["torch"]:
        import torch
        environment["torch_runtime"] = dict(cuda_compiled=torch.version.cuda, cuda_available=torch.cuda.is_available(),
            device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            total_gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None)
    save_json(output / "environment.json", environment)
    save_json(output / f"environment_{args.stage}.json", environment)
    if versions["torch"]:
        locked = ["# Observed experiment versions; GPU torch/torchvision wheels require the official matching CUDA wheel index."]
        locked += [f"{package}=={version}" for package, version in versions.items() if version]
        (output / "requirements-lock.txt").write_text("\n".join(locked) + "\n", encoding="utf-8")
    status = dict(pid=os.getpid(), requested_stage=args.stage, state="running", started_at=datetime.now(timezone.utc).isoformat(), completed_stages=[])
    status_path = output / "status.json"
    started = time.monotonic()
    try:
        stages = {
            "all": ["prepare", "hand", "traditional", "cnn-features", "hybrid", "deep", "evaluate", "report"],
            "baseline": ["prepare", "hand", "traditional"],
        }.get(args.stage, [args.stage])
        for stage in stages:
            status.update(stage=stage, updated_at=datetime.now(timezone.utc).isoformat())
            save_json(status_path, status)
            log(f"=== STAGE: {stage} ===")
            if stage == "prepare":
                frame, protocol = prepare(cfg, log, verify_sources=args.verify_source)
            else:
                frame = read_manifest(cfg)
                protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
                if stage == "hand":
                    from cpchild.features import extract_hand_cache
                    extract_hand_cache(cfg, frame, protocol, log)
                elif stage == "traditional":
                    from cpchild.classical import fit_classical
                    methods = [m for m in cfg["models"] if m not in ("M3_cnn_svm", "M4_cnn_finetune", "M5_fusion_svm")]
                    fit_classical(cfg, frame, protocol, methods, log)
                elif stage == "cnn-features":
                    from cpchild.features import extract_cnn_cache
                    extract_cnn_cache(cfg, frame, protocol, log)
                elif stage == "hybrid":
                    from cpchild.classical import fit_classical
                    fit_classical(cfg, frame, protocol, ["M3_cnn_svm", "M5_fusion_svm"], log)
                elif stage == "deep":
                    from cpchild.deep import fit_deep
                    fit_deep(cfg, frame, protocol, log)
                elif stage == "evaluate":
                    from cpchild.evaluate import evaluate
                    evaluate(cfg, frame, protocol, log)
                elif stage == "report":
                    from cpchild.report import build_report
                    build_report(cfg, frame, protocol, log)
            status["completed_stages"].append(stage)
        status.update(state="complete", elapsed_seconds=time.monotonic() - started,
                        finished_at=datetime.now(timezone.utc).isoformat())
        save_json(status_path, status)
        log(f"=== COMPLETE: {args.stage} ({status['elapsed_seconds']:.1f}s) ===")
    except Exception as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic() - started)
        save_json(status_path, status)
        log(f"!!! STAGE FAILED: {status.get('stage')}: {status['error']}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
