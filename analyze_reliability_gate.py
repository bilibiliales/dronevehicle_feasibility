#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze learned RGB/IR gate weights on paired validation images.

This is a diagnostic, not a detection metric. It feeds resized paired images
through the trained gate and reports modality weights globally and by simple
brightness bins. It does not use labels or modify any checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def paired_path(rgb_path: Path, target: str) -> Path:
    parts = list(rgb_path.parts)
    try:
        i = parts.index("rgb")
    except ValueError as exc:
        raise RuntimeError(f"Expected path containing 'rgb': {rgb_path}") from exc
    parts[i] = target
    return Path(*parts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fusion-weights", required=True)
    p.add_argument("--data-dir", default=r"D:\dronevehicle_feasibility\yolo_obb_data")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--limit", type=int, default=0, help="0 means all validation pairs")
    p.add_argument("--output-dir", default="runs/gate_diagnostics")
    args = p.parse_args()

    device = torch.device(args.device if args.device == "cpu" or not args.device.isdigit()
                          else f"cuda:{args.device}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    # Checkpoints were serialized with custom classes under __main__.
    import __main__
    import train_yolo_obb_fusion
    __main__.ReliabilityOBBModel = train_yolo_obb_fusion.ReliabilityOBBModel
    __main__.ReliabilityGate = train_yolo_obb_fusion.ReliabilityGate
    from ultralytics import YOLO

    model = YOLO(args.fusion_weights, task="obb")
    gate = model.model.reliability_gate.to(device).eval()

    data_dir = Path(args.data_dir).resolve()
    rgb_dir = data_dir / "images" / "rgb" / "val"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"Missing validation directory: {rgb_dir}")
    rgb_paths = sorted(x for x in rgb_dir.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
    if args.limit > 0:
        rgb_paths = rgb_paths[:args.limit]
    if not rgb_paths:
        raise RuntimeError(f"No validation images found in {rgb_dir}")

    # Brightness is mean grayscale intensity on the original paired images.
    # Gate weights are measured on resized, [0,1]-normalized BGR tensors.
    bins = {
        "dark_rgb_<64": [],
        "mid_rgb_64_128": [],
        "bright_rgb_>=128": [],
        "ir_dim_minus_rgb>=20": [],
        "rgb_dim_minus_ir>=20": [],
        "brightness_similar_absdiff<20": [],
    }
    all_weights = []
    per_image = []
    missing = []
    for idx, rgb_path in enumerate(rgb_paths, 1):
        ir_path = paired_path(rgb_path, "ir")
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        ir = cv2.imread(str(ir_path), cv2.IMREAD_COLOR)
        if rgb is None or ir is None:
            missing.append({"rgb": str(rgb_path), "ir": str(ir_path)})
            continue
        if ir.shape[:2] != rgb.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        rgb_gray = float(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).mean())
        ir_gray = float(cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY).mean())
        rgb_in = cv2.resize(rgb, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
        ir_in = cv2.resize(ir, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
        pair = np.concatenate((rgb_in, ir_in), axis=2)
        tensor = torch.from_numpy(pair).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
        with torch.inference_mode():
            _, weights = gate(tensor)
        # Average spatial weights for RGB and IR.
        rgb_w = float(weights[0, 0].mean().item())
        ir_w = float(weights[0, 1].mean().item())
        row = {
            "image": rgb_path.name,
            "rgb_gray_mean": rgb_gray,
            "ir_gray_mean": ir_gray,
            "rgb_weight_mean": rgb_w,
            "ir_weight_mean": ir_w,
            "weight_rgb_minus_ir": rgb_w - ir_w,
        }
        per_image.append(row)
        all_weights.append([rgb_w, ir_w])
        if rgb_gray < 64:
            bins["dark_rgb_<64"].append(row)
        elif rgb_gray < 128:
            bins["mid_rgb_64_128"].append(row)
        else:
            bins["bright_rgb_>=128"].append(row)
        diff = ir_gray - rgb_gray
        if diff >= 20:
            bins["ir_dim_minus_rgb>=20"].append(row)
        elif diff <= -20:
            bins["rgb_dim_minus_ir>=20"].append(row)
        else:
            bins["brightness_similar_absdiff<20"].append(row)
        if idx % 200 == 0:
            print(f"Processed {idx}/{len(rgb_paths)}")

    arr = np.asarray(all_weights, dtype=np.float64)
    def summarize(rows):
        if not rows:
            return {"n": 0}
        return {
            "n": len(rows),
            "rgb_weight_mean": float(np.mean([r["rgb_weight_mean"] for r in rows])),
            "ir_weight_mean": float(np.mean([r["ir_weight_mean"] for r in rows])),
            "rgb_weight_std": float(np.std([r["rgb_weight_mean"] for r in rows])),
            "ir_weight_std": float(np.std([r["ir_weight_mean"] for r in rows])),
            "mean_rgb_gray": float(np.mean([r["rgb_gray_mean"] for r in rows])),
            "mean_ir_gray": float(np.mean([r["ir_gray_mean"] for r in rows])),
        }

    report = {
        "checkpoint": str(Path(args.fusion_weights).resolve()),
        "validation_pairs_found": len(rgb_paths),
        "processed_pairs": len(per_image),
        "missing_pairs": missing,
        "diagnostic_note": "Gate weights are mean spatial softmax weights on directly resized normalized images; brightness bins are descriptive, not proof of causal reliability.",
        "overall": summarize(per_image),
        "by_rgb_brightness": {k: summarize(v) for k, v in bins.items() if k in {"dark_rgb_<64", "mid_rgb_64_128", "bright_rgb_>=128"}},
        "by_relative_brightness": {k: summarize(v) for k, v in bins.items() if k not in {"dark_rgb_<64", "mid_rgb_64_128", "bright_rgb_>=128"}},
        "gate_rgb_weight_range": [float(arr[:, 0].min()), float(arr[:, 0].max())] if len(arr) else None,
        "gate_ir_weight_range": [float(arr[:, 1].min()), float(arr[:, 1].max())] if len(arr) else None,
    }

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate_weight_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = out_dir / "gate_weight_per_image.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report: {out_dir / 'gate_weight_summary.json'}")
    print(f"Saved per-image CSV: {csv_path}")


if __name__ == "__main__":
    main()
