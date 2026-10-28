"""Read-only CP-CHILD image audit. Requires numpy and Pillow; no model training.

Run: python analysis_cp_child/audit_dataset.py
Only generated audit files are written, to this script's directory by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from datetime import date
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def figure_font(size, bold=False):
    """Prefer CJK fonts while remaining runnable on non-Windows systems."""
    paths = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in paths:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def hash_bits(bits):
    return int.from_bytes(np.packbits(bits.reshape(-1)).tobytes(), "big")


# Orthonormal DCT-II: compute only the 8 low-frequency rows used by pHash.
_x = np.arange(32)[None, :]
_k = np.arange(8)[:, None]
DCT = np.cos(np.pi * (2 * _x + 1) * _k / 64) * np.sqrt(2 / 32)
DCT[0] /= np.sqrt(2)


def inspect_image(path, root):
    parts = path.relative_to(root).parts
    row = dict(path=path.relative_to(root).as_posix(), dataset=parts[0],
               split=parts[1], label=parts[2], target=int(parts[2] == "Polyp"),
               bytes=path.stat().st_size, error="")
    try:
        raw = path.read_bytes()
        row["file_sha256"] = hashlib.sha256(raw).hexdigest()
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            row.update(width=im.width, height=im.height, mode=im.mode,
                       exif_tag_count=len(im.getexif()))
            rgb = im.convert("RGB")
        row["pixel_sha256"] = hashlib.sha256(
            f"{rgb.width}x{rgb.height}:RGB:".encode() + rgb.tobytes()).hexdigest()
        gray = rgb.convert("L")
        arr = np.asarray(gray)
        rgb_array = np.asarray(rgb)
        row.update(mean_gray=round(float(arr.mean()), 4),
                   std_gray=round(float(arr.std()), 4),
                   dark_fraction=round(float((arr < 10).mean()), 6),
                   bright_fraction=round(float((arr > 245).mean()), 6),
                   mean_red=round(float(rgb_array[:, :, 0].mean()), 4),
                   mean_green=round(float(rgb_array[:, :, 1].mean()), 4),
                   mean_blue=round(float(rgb_array[:, :, 2].mean()), 4))
        small = np.asarray(gray.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)
        lap = (-4 * small[1:-1, 1:-1] + small[:-2, 1:-1] + small[2:, 1:-1]
               + small[1:-1, :-2] + small[1:-1, 2:])
        row["laplacian_variance_128"] = round(float(lap.var()), 4)
        dh = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
        row["dhash"] = f"{hash_bits(dh[:, 1:] > dh[:, :-1]):016x}"
        ph = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
        coeff = DCT @ ph @ DCT.T
        bits = coeff > np.median(coeff.ravel()[1:])
        row["phash"] = f"{hash_bits(bits):016x}"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def duplicate_stats(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    duplicates = [g for g in groups.values() if len(g) > 1]
    crossing = [g for g in duplicates if len({r["split"] for r in g}) > 1]
    per_dataset = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        within = [[r for r in g if r["dataset"] == dataset] for g in duplicates]
        leak = [g for g in within if len({r["split"] for r in g}) > 1]
        per_dataset[dataset] = dict(
            groups=len(leak),
            train_images=sum(r["split"] == "Train" for g in leak for r in g),
            test_images=sum(r["split"] == "Test" for g in leak for r in g))
    return dict(unique_images=len(groups), duplicate_groups=len(duplicates),
                images_in_duplicate_groups=sum(map(len, duplicates)),
                redundant_copies=sum(len(g) - 1 for g in duplicates),
                cross_split_groups=len(crossing),
                cross_split_test_images=sum(r["split"] == "Test" for g in crossing for r in g),
                cross_dataset_groups=sum(len({r["dataset"] for r in g}) > 1 for g in duplicates),
                conflicting_label_groups=sum(len({r["label"] for r in g}) > 1 for g in duplicates),
                within_dataset_cross_split=per_dataset), duplicates


def near_duplicate_candidates(rows):
    # With <=6 bit differences, at least one of seven disjoint bands is equal.
    # This retrieves all pairs under the pHash threshold without an all-pairs matrix.
    buckets = defaultdict(list)
    sizes = [10, 9, 9, 9, 9, 9, 9]
    offsets = np.cumsum([0] + sizes[:-1]).tolist()
    hashes = [int(r["phash"], 16) for r in rows]
    dhashes = [int(r["dhash"], 16) for r in rows]
    pairs = []
    for i, row in enumerate(rows):
        ph = hashes[i]
        keys = [(band, (ph >> offset) & ((1 << size) - 1))
                for band, (size, offset) in enumerate(zip(sizes, offsets))]
        candidates = set(j for key in keys for j in buckets[key])
        for j in candidates:
            other = rows[j]
            if row["pixel_sha256"] == other["pixel_sha256"]:
                continue
            pd = (ph ^ hashes[j]).bit_count()
            if pd > 6:
                continue
            dd = (dhashes[i] ^ dhashes[j]).bit_count()
            if dd > 8:
                continue
            pairs.append(dict(path_1=other["path"], path_2=row["path"],
                              dataset_1=other["dataset"], dataset_2=row["dataset"],
                              split_1=other["split"], split_2=row["split"],
                              label_1=other["label"], label_2=row["label"],
                              phash_distance=pd, dhash_distance=dd,
                              cross_split=other["split"] != row["split"],
                              cross_dataset=other["dataset"] != row["dataset"],
                              conflicting_label=other["label"] != row["label"]))
        for key in keys:
            buckets[key].append(i)
    return sorted(pairs, key=lambda p: (not p["cross_split"], p["phash_distance"], p["dhash_distance"]))


def contact_sheet(root, rows, output):
    font = figure_font(17)
    title_font = figure_font(24, bold=True)
    canvas = Image.new("RGB", (1100, 70 + 8 * 218), "#f5f5f5")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), "CP-CHILD 样本概览｜每组随机 4 张（种子 20261025）", font=title_font, fill="#222222")
    rng = np.random.default_rng(20261025)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["split"], row["label"])].append(row)
    for i, (key, members) in enumerate(sorted(groups.items())):
        y = 70 + i * 218
        draw.text((16, y + 45), key[0], font=font, fill="#222222")
        draw.text((16, y + 74), key[1], font=font, fill="#222222")
        draw.text((16, y + 103), "息肉" if key[2] == "Polyp" else "非息肉", font=font, fill="#222222")
        for c, idx in enumerate(rng.choice(len(members), 4, replace=False)):
            row = members[int(idx)]
            with Image.open(root / row["path"]) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (192, 192))
            x = 210 + c * 218
            canvas.paste(thumb, (x, y))
            draw.text((x, y + 192), Path(row["path"]).name, font=font, fill="#444444")
    canvas.save(output)


def pair_sheet(root, pairs, output, exact=False):
    if not pairs:
        return
    font = figure_font(15)
    selected = pairs[:12]
    canvas = Image.new("RGB", (1260, 310 * ((len(selected) + 2) // 3)), "#f5f5f5")
    draw = ImageDraw.Draw(canvas)
    for i, pair in enumerate(selected):
        x, y = (i % 3) * 420, (i // 3) * 310
        for side in range(2):
            with Image.open(root / pair[f"path_{side + 1}"]) as im:
                canvas.paste(im.convert("RGB").resize((196, 196)), (x + side * 210, y))
            path = Path(pair[f"path_{side + 1}"])
            text = f"{pair[f'dataset_{side + 1}']} {pair[f'split_{side + 1}']}\n{pair[f'label_{side + 1}']}\n{path.name}"
            draw.multiline_text((x + side * 210, y + 198), text, font=font, fill="#222222")
        caption = (f"跨训练/测试重复 {i+1} | 解码 RGB 像素完全相同" if exact else
                   f"候选 {i+1} | pHash={pair['phash_distance']} dHash={pair['dhash_distance']}")
        draw.text((x + 4, y + 265), caption, font=font, fill="#333333")
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent / "6.Colonoscopy_CP_CHILD")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.root.rglob("*.jpg"))
    print(f"Scanning {len(paths)} images", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(inspect_image, p, args.root) for p in paths]
        for i, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if i % 1000 == 0 or i == len(paths):
                print(f"Decoded {i}/{len(paths)}", flush=True)
    rows.sort(key=lambda r: r["path"])
    write_csv(args.output / "image_manifest.csv", rows)
    valid = [r for r in rows if not r["error"]]
    summary = dict(audit_date=date.today().isoformat(), data_root=str(args.root.resolve()),
                   images=len(rows), valid_images=len(valid), errors=[r for r in rows if r["error"]],
                   total_image_bytes=sum(r["bytes"] for r in rows),
                   resolution_counts=dict(Counter(f"{r['width']}x{r['height']}" for r in valid)),
                   mode_counts=dict(Counter(r["mode"] for r in valid)),
                   images_with_exif=sum(r["exif_tag_count"] > 0 for r in valid),
                   patient_ids_available=False, groups=[])
    groups = defaultdict(list)
    for row in valid:
        groups[(row["dataset"], row["split"], row["label"])].append(row)
    for key, members in sorted(groups.items()):
        stats = dict(dataset=key[0], split=key[1], label=key[2], count=len(members))
        for field in ["mean_gray", "dark_fraction", "bright_fraction", "laplacian_variance_128"]:
            stats[field] = dict(zip(["min", "p05", "median", "p95", "max"],
                                   np.quantile([r[field] for r in members], [0, .05, .5, .95, 1]).round(5).tolist()))
        summary["groups"].append(stats)
    duplicate_rows = []
    for field in ["file_sha256", "pixel_sha256"]:
        stats, duplicates = duplicate_stats(valid, field)
        summary[field] = stats
        for idx, members in enumerate(duplicates, 1):
            for row in members:
                duplicate_rows.append(dict(hash_type=field, group_id=idx, path=row["path"],
                                           dataset=row["dataset"], split=row["split"], label=row["label"],
                                           hash=row[field]))
    cross_split_pairs = []
    for members in duplicates:  # decoded pixel groups from the final iteration
        train = [r for r in members if r["split"] == "Train"]
        test = [r for r in members if r["split"] == "Test"]
        for tr in train:
            for te in test:
                cross_split_pairs.append(dict(path_1=tr["path"], path_2=te["path"],
                    dataset_1=tr["dataset"], dataset_2=te["dataset"], split_1=tr["split"],
                    split_2=te["split"], label_1=tr["label"], label_2=te["label"],
                    pixel_sha256=tr["pixel_sha256"]))
    write_csv(args.output / "train_test_pixel_overlap.csv", cross_split_pairs,
              ["path_1", "path_2", "dataset_1", "dataset_2", "split_1", "split_2", "label_1", "label_2", "pixel_sha256"])
    summary["pixel_sha256"]["cross_split_test_images_by_class"] = dict(Counter(
        p["dataset_2"] + "/" + p["label_2"] for p in cross_split_pairs))
    # Suggestions only: retain all original paths, and explicitly flag proposed exclusions.
    pixel_groups = defaultdict(list)
    for row in valid:
        pixel_groups[row["pixel_sha256"]].append(row)
    recommendations = []
    for members in pixel_groups.values():
        test = sorted([r for r in members if r["split"] == "Test"], key=lambda r: r["path"])
        train = sorted([r for r in members if r["split"] == "Train"], key=lambda r: r["path"])
        for row in members:
            reason = "keep"
            if row["split"] == "Train" and test:
                reason = "exclude_train_pixel_match_in_test"
            elif row["split"] == "Train" and row is not train[0]:
                reason = "exclude_repeated_train_pixels"
            elif row["split"] == "Test" and row is not test[0]:
                reason = "optional_exclude_repeated_test_pixels_for_unique_image_metrics"
            recommendations.append(dict(path=row["path"], dataset=row["dataset"], split=row["split"],
                                        label=row["label"], pixel_group=row["pixel_sha256"],
                                        recommendation=reason))
    recommendations.sort(key=lambda r: r["path"])
    write_csv(args.output / "proposed_exact_duplicate_handling.csv", recommendations)
    summary["proposed_exact_handling_counts"] = dict(Counter(r["recommendation"] for r in recommendations))
    summary["proposed_unique_retained_counts"] = dict(sorted(Counter(
        r["dataset"] + "/" + r["split"] + "/" + r["label"]
        for r in recommendations if r["recommendation"] == "keep").items()))
    write_csv(args.output / "exact_duplicate_groups.csv", duplicate_rows,
              ["hash_type", "group_id", "path", "dataset", "split", "label", "hash"])
    print("Searching perceptual-hash candidate pairs", flush=True)
    pairs = near_duplicate_candidates(valid)
    summary["near_duplicates"] = dict(
        method="64-bit DCT pHash <=6 and 64-bit dHash <=8, excluding decoded-pixel-identical pairs; candidates only, requires visual review",
        candidate_pairs=len(pairs), cross_split_candidate_pairs=sum(p["cross_split"] for p in pairs),
        cross_dataset_candidate_pairs=sum(p["cross_dataset"] for p in pairs),
        conflicting_label_candidate_pairs=sum(p["conflicting_label"] for p in pairs),
        cross_split_by_dataset={dataset: sum(p["cross_split"] and p["dataset_1"] == p["dataset_2"] == dataset for p in pairs)
                                for dataset in sorted({r["dataset"] for r in valid})})
    write_csv(args.output / "near_duplicate_candidates.csv", pairs,
              ["path_1", "path_2", "dataset_1", "dataset_2", "split_1", "split_2", "label_1", "label_2",
               "phash_distance", "dhash_distance", "cross_split", "cross_dataset", "conflicting_label"])
    contact_sheet(args.root, valid, args.output / "sample_contact_sheet.jpg")
    pair_sheet(args.root, pairs, args.output / "near_duplicate_review.jpg")
    pair_sheet(args.root, cross_split_pairs, args.output / "train_test_overlap_review.jpg", exact=True)
    (args.output / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: val for key, val in summary.items() if key != "groups"}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
