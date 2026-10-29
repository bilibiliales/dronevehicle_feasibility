#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fairly evaluate RGB, IR, and reliability-gate OBB checkpoints on the SAME RGB labels.

Run from the repository root after training. No training is performed.
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


def paired_path(rgb_path: str | Path, target: str) -> Path:
    p = Path(rgb_path)
    parts = list(p.parts)
    try:
        i = parts.index("rgb")
    except ValueError as exc:
        raise RuntimeError(f"Expected RGB image path containing 'rgb': {p}") from exc
    parts[i] = target
    return Path(*parts)


class RGBLabelIRDataset(YOLODataset):
    """Load IR pixels via the RGB image index; labels remain inferred from RGB paths."""
    def load_image(self, i, rect_mode=True):
        rgb, (h0, w0), (h, w) = super().load_image(i, rect_mode)
        ir_path = paired_path(self.im_files[i], "ir")
        ir = cv2.imread(str(ir_path), cv2.IMREAD_COLOR)
        if ir is None:
            raise FileNotFoundError(f"Cannot read paired IR image: {ir_path}")
        if ir.shape[:2] != rgb.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        return ir, (h0, w0), (h, w)


class PairedRGBIRDataset(YOLODataset):
    """Load aligned RGB+IR pixels (six channels); labels remain RGB labels."""
    def load_image(self, i, rect_mode=True):
        rgb, (h0, w0), (h, w) = super().load_image(i, rect_mode)
        ir_path = paired_path(self.im_files[i], "ir")
        ir = cv2.imread(str(ir_path), cv2.IMREAD_COLOR)
        if ir is None:
            raise FileNotFoundError(f"Cannot read paired IR image: {ir_path}")
        if ir.shape[:2] != rgb.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        return np.concatenate((rgb, ir), axis=2), (h0, w0), (h, w)


class _RGBLabelValidator(OBBValidator):
    dataset_kind = "rgb"

    def build_dataset(self, img_path, mode="val", batch=None):
        return self.dataset_kind and self._make_dataset(img_path, mode, batch)

    def _make_dataset(self, img_path, mode, batch):
        # Mirror the standard OBB validator's YOLODataset arguments.
        return self.dataset_cls(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=self.args.rect,
            cache=self.args.cache,
            single_cls=self.args.single_cls,
            stride=max(int(self.stride), 32),
            pad=0.5,
            prefix=f"{mode}: ",
            data=self.data,
            task="obb",
        )


class IROnRGBLabelsValidator(_RGBLabelValidator):
    dataset_kind = "ir"
    dataset_cls = RGBLabelIRDataset


class PairedFusionValidator(_RGBLabelValidator):
    dataset_kind = "rgb+ir"
    dataset_cls = PairedRGBIRDataset


def metrics_dict(results, label):
    box = results.box
    return {
        "model": label,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class_map50": [float(x) for x in box.ap50],
        "per_class_map50_95": [float(x) for x in box.ap],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=r"D:\\dronevehicle_feasibility\\yolo_obb_data")
    p.add_argument("--rgb-weights", required=True)
    p.add_argument("--ir-weights", required=True)
    p.add_argument("--fusion-weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="runs/fair_validation")
    p.add_argument("--name", default="rgb_labels_all_models")
    args = p.parse_args()

    data_dir = Path(args.data_dir).resolve()
    yaml_path = data_dir / "rgb.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing RGB-label dataset YAML: {yaml_path}")
    rgb_val = data_dir / "images" / "rgb" / "val"
    ir_val = data_dir / "images" / "ir" / "val"
    if not rgb_val.is_dir() or not ir_val.is_dir():
        raise FileNotFoundError(f"Expected validation folders: {rgb_val} and {ir_val}")

    # Import custom class definitions before loading the fusion checkpoint.
    import train_yolo_obb_fusion  # noqa: F401

    common = dict(data=str(yaml_path), imgsz=args.imgsz, batch=args.batch,
                  device=args.device, split="val", plots=False, save_json=False,
                  project=str(Path(args.project).resolve()), exist_ok=True)

    runs = [
        ("rgb", args.rgb_weights, None),
        ("ir_on_rgb_labels", args.ir_weights, IROnRGBLabelsValidator),
        ("reliability_gate_fusion", args.fusion_weights, PairedFusionValidator),
    ]
    rows = []
    for name, weights, validator_cls in runs:
        print(f"\\n=== Fair validation: {name} ===")
        model = YOLO(weights, task="obb")
        kwargs = dict(common, name=name)
        if validator_cls is not None:
            kwargs["validator"] = validator_cls
        result = model.val(**kwargs)
        row = metrics_dict(result, name)
        rows.append(row)
        print(json.dumps(row, indent=2, ensure_ascii=False))

    out_dir = Path(args.project).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.name}_metrics.json"
    csv_path = out_dir / f"{args.name}_metrics.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\\nSaved fair metrics to:\\n  {json_path}\\n  {csv_path}")


if __name__ == "__main__":
    main()
