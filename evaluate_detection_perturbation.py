#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate detection robustness under controlled RGB/IR perturbations.

Uses existing RGB baseline and input-level RGB-IR fusion checkpoints; no training.
All metrics use RGB validation labels and the same first N sorted RGB validation
pairs. Perturbations are applied to image pixels before Ultralytics letterboxing.
This is a synthetic stress test, not a substitute for natural-condition testing.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.obb import OBBValidator

VARIANT = "original"
LIMIT = 300
IMAGE_COUNT = 0


def paired_path(rgb_path: str | Path, target: str = "ir") -> Path:
    p = Path(rgb_path)
    parts = list(p.parts)
    try:
        i = parts.index("rgb")
    except ValueError as exc:
        raise RuntimeError(f"Expected path containing 'rgb': {p}") from exc
    parts[i] = target
    return Path(*parts)


def perturb(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "original":
        return image
    if mode == "dark":
        return np.clip(image.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
    if mode == "blur":
        return cv2.GaussianBlur(image, (9, 9), 2.0)
    if mode == "noise":
        # Stable per-image noise is generated in load_image using the image index.
        raise RuntimeError("Noise perturbation requires a per-image seed")
    raise ValueError(f"Unknown perturbation: {mode}")


def apply_perturbation(image: np.ndarray, mode: str, seed: int) -> np.ndarray:
    # RGB baseline variants are named rgb_dark/rgb_blur/rgb_noise, while this
    # function receives only the modality-specific operation for fusion inputs.
    if mode.startswith("rgb_") or mode.startswith("ir_"):
        mode = mode.split("_", 1)[1]
    if mode != "noise":
        return perturb(image, mode)
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.float32) + rng.normal(0.0, 25.0, image.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


class LimitedDatasetMixin:
    """Limit the sorted image list before labels are loaded, keeping RGB labels."""

    def get_img_files(self, img_path):
        files = super().get_img_files(img_path)
        return files[:LIMIT] if LIMIT > 0 else files


class RGBPerturbDataset(LimitedDatasetMixin, YOLODataset):
    def load_image(self, i, rect_mode=True):
        image, (h0, w0), (h, w) = super().load_image(i, rect_mode)
        image = apply_perturbation(image, VARIANT, seed=20261029 + i)
        return image, (h0, w0), (h, w)


class PairedPerturbDataset(LimitedDatasetMixin, YOLODataset):
    def load_image(self, i, rect_mode=True):
        rgb, (h0, w0), (h, w) = super().load_image(i, rect_mode)
        ir_path = paired_path(self.im_files[i], "ir")
        ir = cv2.imread(str(ir_path), cv2.IMREAD_COLOR)
        if ir is None:
            raise FileNotFoundError(f"Cannot read paired IR image: {ir_path}")
        if ir.shape[:2] != rgb.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Variants are named rgb_* or ir_*; the other modality stays untouched.
        if VARIANT.startswith("rgb_"):
            rgb = apply_perturbation(rgb, VARIANT[4:], seed=20261029 + i)
        elif VARIANT.startswith("ir_"):
            ir = apply_perturbation(ir, VARIANT[3:], seed=20261029 + i)
        return np.concatenate((rgb, ir), axis=2), (h0, w0), (h, w)


class PerturbValidator(OBBValidator):
    dataset_cls = RGBPerturbDataset

    def build_dataset(self, img_path, mode="val", batch=None):
        return self.dataset_cls(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=self.args.rect,
            cache=False,
            single_cls=self.args.single_cls,
            stride=32,
            pad=0.5,
            prefix=f"{mode}: ",
            data=self.data,
            task="obb",
        )


class FusionPerturbValidator(PerturbValidator):
    dataset_cls = PairedPerturbDataset


def extract_metrics(result, model_name, variant):
    box = result.box
    return {
        "model": model_name,
        "variant": variant,
        "images": IMAGE_COUNT,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }


def main():
    global VARIANT, LIMIT, IMAGE_COUNT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=r"D:\\dronevehicle_feasibility\\yolo_obb_data")
    parser.add_argument("--rgb-weights", required=True)
    parser.add_argument("--fusion-weights", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=300, help="First N sorted validation pairs; 0 means all")
    parser.add_argument("--output-dir", default="runs/detection_perturbation")
    args = parser.parse_args()
    LIMIT = args.limit
    data_dir = Path(args.data_dir).resolve()
    yaml_path = data_dir / "rgb.yaml"
    rgb_val = data_dir / "images" / "rgb" / "val"
    ir_val = data_dir / "images" / "ir" / "val"
    if not yaml_path.is_file() or not rgb_val.is_dir() or not ir_val.is_dir():
        raise FileNotFoundError(f"Expected RGB YAML and paired validation folders under {data_dir}")
    rgb_paths = sorted(p for p in rgb_val.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
    if LIMIT > 0:
        rgb_paths = rgb_paths[:LIMIT]
    if not rgb_paths:
        raise RuntimeError(f"No RGB validation images found in {rgb_val}")
    IMAGE_COUNT = len(rgb_paths)
    missing = [p.name for p in rgb_paths if not paired_path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing IR pairs ({len(missing)}), first few: {missing[:10]}")
    print(f"Using {len(rgb_paths)} fixed validation pairs. Metrics use RGB labels.")

    # Checkpoint class references saved during fusion training.
    import __main__
    import train_yolo_obb_fusion
    __main__.ReliabilityOBBModel = train_yolo_obb_fusion.ReliabilityOBBModel
    __main__.ReliabilityGate = train_yolo_obb_fusion.ReliabilityGate

    rgb_model = YOLO(args.rgb_weights, task="obb")
    fusion_model = YOLO(args.fusion_weights, task="obb")
    common = dict(
        data=str(yaml_path), imgsz=args.imgsz, batch=args.batch, device=args.device,
        split="val", plots=False, save_json=False, workers=0, rect=False,
        fraction=1.0, project=str(Path(args.output_dir).resolve()), exist_ok=True,
        verbose=False,
    )

    # Shared RGB perturbations allow direct baseline-vs-fusion comparisons.
    tests = [
        ("original", "original"),
        ("rgb_dark", "rgb_dark"),
        ("rgb_blur", "rgb_blur"),
        ("rgb_noise", "rgb_noise"),
        ("ir_dark", "ir_dark"),
        ("ir_blur", "ir_blur"),
        ("ir_noise", "ir_noise"),
    ]
    rows = []
    for variant, _ in tests:
        VARIANT = variant
        print(f"\\n=== Variant: {variant} ===")
        # RGB baseline has no IR input, so IR-only variants are not applicable.
        if not variant.startswith("ir_"):
            result = rgb_model.val(**common, name=f"rgb_{variant}", validator=PerturbValidator)
            rows.append(extract_metrics(result, "rgb_baseline", variant))
            print(json.dumps(rows[-1], ensure_ascii=False))
        result = fusion_model.val(**common, name=f"fusion_{variant}", validator=FusionPerturbValidator)
        rows.append(extract_metrics(result, "rgb_ir_fusion", variant))
        print(json.dumps(rows[-1], ensure_ascii=False))

    # Calculate drops relative to each model's own original-input result.
    baselines = {r["model"]: r for r in rows if r["variant"] == "original"}
    for row in rows:
        base = baselines[row["model"]]
        row["delta_map50"] = row["map50"] - base["map50"]
        row["delta_map50_95"] = row["map50_95"] - base["map50_95"]
        row["delta_recall"] = row["recall"] - base["recall"]

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "detection_perturbation_summary.json"
    csv_path = out / "detection_perturbation_summary.csv"
    json_path.write_text(json.dumps({
        "rgb_label_yaml": str(yaml_path),
        "images_requested": args.limit,
        "images_evaluated": len(rgb_paths),
        "image_selection": "first N sorted RGB validation filenames; same pairs in every variant" if args.limit > 0 else "all sorted RGB validation filenames; same pairs in every variant",
        "selected_filenames": [p.name for p in rgb_paths],
        "note": "Synthetic robustness stress test. RGB baseline cannot receive IR-only perturbations. Delta metrics are relative to each model's own original variant.",
        "results": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\\nSaved: {json_path}\\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
