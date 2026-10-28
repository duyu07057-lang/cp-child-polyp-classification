"""Image-only hand-crafted features and fixed ImageNet ResNet18 features."""

from __future__ import annotations

import json
import hashlib
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from .data import save_json


COLOR_DIM = 38
HAND_DIM = 140
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def hand_feature_names():
    names = [f"hsv_{channel}_{i}" for channel, bins in (("h", 16), ("s", 8), ("v", 8)) for i in range(bins)]
    names += [f"lab_{stat}_{channel}" for stat in ("mean", "std") for channel in ("l", "a", "b")]
    names += [f"lbp_r{radius}_{i}" for radius, points in ((1, 8), (2, 16), (3, 24)) for i in range(points + 2)]
    names += [f"glcm_{prop}_d{distance}_a{angle}" for prop in
              ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
              for distance in (1, 3) for angle in range(4)]
    return names


def extract_hand(image):
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            image = source.convert("RGB")
    image = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    hsv, lab = rgb2hsv(rgb), rgb2lab(rgb)
    values = []
    for channel, bins in enumerate((16, 8, 8)):
        histogram = np.histogram(hsv[:, :, channel], bins=bins, range=(0, 1))[0].astype(float)
        values.extend(histogram / histogram.sum())
    values.extend(lab.mean(axis=(0, 1)))
    values.extend(lab.std(axis=(0, 1)))
    gray = np.rint(np.clip(rgb2gray(rgb), 0, 1) * 255).astype(np.uint8)
    for radius, points in ((1, 8), (2, 16), (3, 24)):
        lbp = local_binary_pattern(gray, points, radius, method="uniform")
        histogram = np.bincount(lbp.astype(int).ravel(), minlength=points + 2)
        values.extend(histogram / histogram.sum())
    glcm = graycomatrix(gray // 8, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        values.extend(graycoprops(glcm, prop).ravel())
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (HAND_DIM,) or not np.isfinite(result).all():
        raise ValueError("Invalid hand-crafted feature vector")
    return result


def extract_hand_cache(cfg, frame, protocol, log=print):
    directory = Path(cfg["output_dir"]) / "features"
    directory.mkdir(parents=True, exist_ok=True)
    path, meta_path = directory / "hand_features.npy", directory / "hand_features.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["fingerprint"] != protocol["fingerprint"]:
            raise ValueError("Hand-feature cache belongs to another protocol")
        features = np.load(path, mmap_mode="r")
        if features.shape != (len(frame), HAND_DIM) or not np.isfinite(features).all():
            raise ValueError("Hand-feature cache has an invalid shape or values")
        return features
    root = Path(cfg["dataset_root"])
    features = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(frame), HAND_DIM))
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=cfg["feature_workers"]) as pool:
        for i, vector in enumerate(pool.map(extract_hand, (root / p for p in frame.path)), 1):
            features[i - 1] = vector
            if i % 500 == 0 or i == len(frame):
                log(f"Hand features {i}/{len(frame)} ({time.monotonic() - start:.1f}s)")
    features.flush()
    save_json(meta_path, dict(fingerprint=protocol["fingerprint"], names=hand_feature_names(),
                              color_dimension=COLOR_DIM, texture_dimension=HAND_DIM - COLOR_DIM,
                              shape=list(features.shape), seconds=time.monotonic() - start))
    return features


def configure_torch(cfg):
    import torch

    torch.set_num_threads(cfg["torch_threads"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch


def image_tensor(image, size=224):
    import torch

    image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = torch.from_numpy(array.copy())
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    return (tensor - mean) / std


def load_tensor(path, size, augmented=False, index=0, seed=42):
    with Image.open(path) as source:
        image = source.convert("RGB")
    if augmented:
        rng = np.random.default_rng(seed + int(index) * 31)
        if rng.random() < .5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        else:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        # No crops; preserve corner polyps. These deterministic views use training images only.
        fill = tuple(np.asarray(image).mean(axis=(0, 1)).astype(int))
        image = image.rotate(float(rng.uniform(-8, 8)), resample=Image.Resampling.BILINEAR, fillcolor=fill)
        image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(.92, 1.08)))
    return image_tensor(image, size)


def stem_forward(model, batch):
    batch = model.maxpool(model.relu(model.bn1(model.conv1(batch))))
    return model.layer3(model.layer2(model.layer1(batch)))


def pooled_forward(model, stem):
    return model.avgpool(model.layer4(stem)).flatten(1)


def pretrained_model(cfg, log=print):
    torch = configure_torch(cfg)
    from torchvision.models import ResNet18_Weights, resnet18

    output = Path(cfg["output_dir"])
    torch.hub.set_dir(str(Path(cfg["project_root"]) / ".cache" / "torch"))
    checkpoint = output / "models" / "frozen_resnet18.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        model = resnet18(weights=None)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    else:
        reference = Path(cfg["project_root"]) / "experiments" / "cp_child_20261028"
        reference_weights = reference / "models" / "frozen_resnet18.pt"
        reference_lock = reference / "selection_lock.json"
        if reference_weights.exists() and reference_lock.exists():
            expected = json.loads(reference_lock.read_text(encoding="utf-8"))["feature_backbone_sha256"]
            actual = hashlib.sha256(reference_weights.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError("The bundled pretrained weights do not match the recorded SHA-256")
            shutil.copyfile(reference_weights, checkpoint)
            model = resnet18(weights=None)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            log("Copied and verified the original ImageNet ResNet18 weights")
        else:
            log("Loading official ImageNet ResNet18 weights (download once, then cached locally)")
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            torch.save(model.state_dict(), checkpoint)
    return model.eval()


def _activation_pass(cfg, frame, protocol, model, indices, augmented, log):
    torch = configure_torch(cfg)
    directory = Path(cfg["output_dir"]) / "features"
    prefix = "augmented_train" if augmented else "original"
    meta_path, data_path = directory / f"{prefix}_stem.json", directory / f"{prefix}_stem.npy"
    n = len(indices)
    completed = 0
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["fingerprint"] != protocol["fingerprint"] or meta["indices"] != list(map(int, indices)):
            raise ValueError("Activation cache metadata does not match the fixed split")
        completed = meta["completed"]
    data = np.lib.format.open_memmap(data_path, mode="r+" if completed else "w+", dtype=np.float16,
                                    shape=(n, 256, 14, 14))
    pooled_path = directory / "cnn_features.npy"
    pooled = None
    if not augmented:
        pooled = np.lib.format.open_memmap(pooled_path, mode="r+" if completed else "w+", dtype=np.float32,
                                          shape=(len(frame), 512))
    if completed == n:
        return data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    root = Path(cfg["dataset_root"])
    start = time.monotonic()
    last_log = 0.0
    with ThreadPoolExecutor(max_workers=cfg["feature_workers"]) as pool, torch.inference_mode():
        for offset in range(completed, n, cfg["batch_size"]):
            ids = np.asarray(indices[offset:offset + cfg["batch_size"]], dtype=int)
            tensors = list(pool.map(lambda idx: load_tensor(root / frame.iloc[int(idx)].path,
                                                          cfg["image_size"], augmented, int(idx), cfg["seed"]), ids))
            batch = torch.stack(tensors).to(device)
            activation = stem_forward(model, batch)
            data[offset:offset + len(ids)] = activation.cpu().numpy().astype(np.float16)
            if pooled is not None:
                pooled[ids] = pooled_forward(model, activation).cpu().numpy()
            done = offset + len(ids)
            elapsed = time.monotonic() - start
            if elapsed - last_log >= 15 or done == n:
                data.flush()
                if pooled is not None:
                    pooled.flush()
                save_json(meta_path, dict(fingerprint=protocol["fingerprint"], indices=list(map(int, indices)),
                                          completed=done, shape=list(data.shape), augmented=augmented,
                                          augmentation="fixed training-only flip, +/-8deg, brightness +/-8%; no cropping" if augmented else None))
                log(f"CNN {prefix}: {done}/{n}, {((done-completed)/max(elapsed,1e-9)):.1f} images/s, device={device}")
                last_log = elapsed
    data.flush()
    return data


def extract_cnn_cache(cfg, frame, protocol, log=print):
    directory = Path(cfg["output_dir"]) / "features"
    directory.mkdir(parents=True, exist_ok=True)
    model = pretrained_model(cfg, log)
    _activation_pass(cfg, frame, protocol, model, np.arange(len(frame)), False, log)
    train_ids = frame.loc[frame.phase == "train", "image_index"].to_numpy(dtype=int)
    np.save(directory / "augmented_train_indices.npy", train_ids)
    _activation_pass(cfg, frame, protocol, model, train_ids, True, log)
    features = np.load(directory / "cnn_features.npy", mmap_mode="r")
    if features.shape != (len(frame), 512) or not np.isfinite(features).all():
        raise ValueError("Invalid CNN feature cache")
    save_json(directory / "cnn_features.json", dict(fingerprint=protocol["fingerprint"], shape=list(features.shape),
        weights="torchvision ResNet18 IMAGENET1K_V1", input_size=[224, 224],
        resize="bilinear full frame, no crop", feature="512-d global average pooling before classifier",
        stem_cache="float16, layer3 output; float32 tail training; pretrained stem remains frozen"))
    return features


def feature_matrix(hand, cnn, method):
    if method in ("M1_hand_svm", "M2_hand_rf"):
        return np.asarray(hand)
    if method == "A1_color_svm":
        return np.asarray(hand[:, :COLOR_DIM])
    if method == "A2_texture_svm":
        return np.asarray(hand[:, COLOR_DIM:])
    if method == "M3_cnn_svm":
        return np.asarray(cnn)
    if method == "M5_fusion_svm":
        return np.concatenate([hand, cnn], axis=1)
    if method == "M0_majority":
        return np.zeros((len(hand), 1), dtype=np.float32)
    raise ValueError(f"Unknown feature method: {method}")
