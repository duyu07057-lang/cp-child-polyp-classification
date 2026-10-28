"""Fine-tune ResNet18 layer4/classifier using cached frozen layer3 outputs."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .classical import prediction_frame, selection_path
from .data import save_json
from .features import configure_torch, pooled_forward, pretrained_model
from .metrics import binary_metrics, select_threshold


def cached_scores(model, activation, ids, cfg, device):
    import torch

    model.eval()
    pieces = []
    with torch.inference_mode():
        for offset in range(0, len(ids), cfg["batch_size"]):
            batch = torch.from_numpy(np.array(activation[ids[offset:offset + cfg["batch_size"]]], dtype=np.float32)).to(device)
            logits = model.fc(pooled_forward(model, batch))
            pieces.append(logits.softmax(dim=1)[:, 1].cpu().numpy())
    return np.concatenate(pieces).astype(float)


def fit_deep(cfg, frame, protocol, log=print):
    torch = configure_torch(cfg)
    output = Path(cfg["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original = np.load(output / "features" / "original_stem.npy", mmap_mode="r")
    augmented = np.load(output / "features" / "augmented_train_stem.npy", mmap_mode="r")
    aug_ids = np.load(output / "features" / "augmented_train_indices.npy")
    lookup = np.full(len(frame), -1, dtype=int)
    lookup[aug_ids] = np.arange(len(aug_ids))
    # NumPy's platform int is 32-bit on some Windows builds; CUDA CE requires int64.
    labels = frame.target.to_numpy(dtype=np.int64)
    (output / "selection").mkdir(exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)
    for dataset_offset, dataset in enumerate(sorted(frame.dataset.unique())):
        method = "M4_cnn_finetune"
        meta_path = selection_path(output, dataset, method)
        model_path = output / "models" / f"{dataset}__{method}.pt"
        resume_path = output / "models" / f"{dataset}__{method}.resume.pt"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta["fingerprint"] != protocol["fingerprint"] or not model_path.exists():
                raise ValueError("Fine-tuned model metadata does not match the protocol")
            log(f"Reuse fitted {dataset} {method}")
            continue
        model = pretrained_model(cfg, log)
        torch.manual_seed(cfg["seed"] + dataset_offset)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.fc = torch.nn.Linear(512, 2)
        model = model.to(device)
        optimizer = torch.optim.AdamW([
            {"params": model.layer4.parameters(), "lr": cfg["cnn_backbone_lr"]},
            {"params": model.fc.parameters(), "lr": cfg["cnn_head_lr"]}], weight_decay=cfg["cnn_weight_decay"])
        train_ids = np.flatnonzero((frame.dataset == dataset) & (frame.phase == "train"))
        val_ids = np.flatnonzero((frame.dataset == dataset) & (frame.phase == "validation"))
        counts = np.bincount(labels[train_ids], minlength=2)
        weight = torch.tensor(len(train_ids) / (2 * counts), dtype=torch.float32, device=device)
        loss_function = torch.nn.CrossEntropyLoss(weight=weight)
        rng = np.random.default_rng(cfg["seed"] + dataset_offset)
        history, best_ap, stale, next_epoch = [], -1.0, 0, 0
        start = time.monotonic()
        if resume_path.exists():
            state = torch.load(resume_path, map_location=device, weights_only=False)
            if state["fingerprint"] != protocol["fingerprint"]:
                raise ValueError("Deep resume checkpoint belongs to another protocol")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            history, best_ap, stale, next_epoch = state["history"], state["best_ap"], state["stale"], state["next_epoch"]
            rng.bit_generator.state = state["rng_state"]
            torch.set_rng_state(state["torch_rng"].cpu())
            log(f"Resume {dataset} CNN at epoch {next_epoch + 1}")
        for epoch in range(next_epoch, cfg["cnn_max_epochs"]):
            if stale >= cfg["cnn_patience"]:
                break
            model.eval()  # Frozen running BN statistics; affine parameters in layer4 can still train.
            model.fc.train()
            tune_tail = epoch >= cfg["cnn_warmup_epochs"]
            for parameter in model.layer4.parameters():
                parameter.requires_grad_(tune_tail)
            total_loss = 0.0
            shuffled = rng.permutation(train_ids)
            epoch_start = time.monotonic()
            last_log = epoch_start
            for offset in range(0, len(shuffled), cfg["batch_size"]):
                ids = shuffled[offset:offset + cfg["batch_size"]]
                values = np.array(original[ids], dtype=np.float32)
                use_aug = rng.random(len(ids)) < .5
                if np.any(lookup[ids] < 0):
                    raise ValueError("A non-training image entered augmented-tail training")
                values[use_aug] = np.asarray(augmented[lookup[ids[use_aug]]], dtype=np.float32)
                batch = torch.from_numpy(values).to(device)
                target = torch.from_numpy(labels[ids]).to(device=device, dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                logits = model.fc(pooled_forward(model, batch))
                loss = loss_function(logits, target)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite CNN training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                optimizer.step()
                total_loss += float(loss.item()) * len(ids)
                if time.monotonic() - last_log >= 25:
                    log(f"CNN {dataset} epoch {epoch+1}: {offset+len(ids)}/{len(shuffled)} training images")
                    last_log = time.monotonic()
            scores = cached_scores(model, original, val_ids, cfg, device)
            threshold = select_threshold(labels[val_ids], scores)
            metrics = binary_metrics(labels[val_ids], scores, threshold)
            record = dict(epoch=epoch + 1, tail_trainable=tune_tail, loss=total_loss / len(train_ids),
                          seconds=time.monotonic() - epoch_start, **metrics)
            history.append(record)
            if metrics["average_precision"] > best_ap + 1e-9:
                best_ap, stale = metrics["average_precision"], 0
                torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                fingerprint=protocol["fingerprint"], best_epoch=epoch + 1,
                                stem_cache_dtype="float16", threshold=threshold, dataset=dataset,
                                validation=metrics, image_size=cfg["image_size"]), model_path)
            else:
                stale += 1
            save_json(output / "selection" / f"{dataset}__cnn_history.json", history)
            torch.save(dict(fingerprint=protocol["fingerprint"], model=model.state_dict(), optimizer=optimizer.state_dict(),
                            history=history, best_ap=best_ap, stale=stale, next_epoch=epoch + 1,
                            rng_state=rng.bit_generator.state, torch_rng=torch.get_rng_state()), resume_path)
            log(f"CNN {dataset} epoch {epoch+1}/{cfg['cnn_max_epochs']}: validation AP={metrics['average_precision']:.4f}, F1={metrics['f1']:.4f}, loss={record['loss']:.4f}, {record['seconds']:.1f}s")
        best = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(best["state_dict"])
        model.to(device)
        scores = cached_scores(model, original, val_ids, cfg, device)
        threshold = best["threshold"]
        metrics = binary_metrics(labels[val_ids], scores, threshold)
        prediction_frame(frame, val_ids, scores, threshold).to_csv(
            output / "predictions" / f"{dataset}__{method}__validation.csv", index=False, encoding="utf-8-sig")
        save_json(meta_path, dict(fingerprint=protocol["fingerprint"], dataset=dataset, method=method,
            threshold=threshold, validation=metrics, train_n=len(train_ids), train_positives=int(counts[1]),
            model_path=model_path.relative_to(output).as_posix(), training_seconds=time.monotonic() - start,
            selected_parameters=dict(best_epoch=best["best_epoch"], epochs_run=len(history),
                frozen_stem="conv1 through layer3", trainable="layer4 and fc, after head warmup",
                batchnorm="frozen running statistics", optimizer="AdamW", class_weights=weight.cpu().tolist(),
                cached_training_augmentation="original plus one fixed augmented view, sampled 50/50 per epoch",
                numerical_precision="float16 frozen stem cache, float32 trainable tail")))
        log(f"Saved {dataset} CNN best epoch {best['best_epoch']}; test data not scored")
