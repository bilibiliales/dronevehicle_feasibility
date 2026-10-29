#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Controlled sensitivity test for the learned RGB/IR input reliability gate.

This test measures gate weights only (not detection accuracy). For each paired
validation image, it compares the original gate output with variants where one
modality is darkened, blurred, or mildly noised. The test is diagnostic: synthetic
perturbations are not a substitute for natural-condition evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def paired_path(rgb_path: Path) -> Path:
    parts = list(rgb_path.parts)
    try:
        i = parts.index("rgb")
    except ValueError as exc:
        raise RuntimeError(f"Expected path containing 'rgb': {rgb_path}") from exc
    parts[i] = "ir"
    return Path(*parts)


def degrade(image: np.ndarray, mode: str) -> np.ndarray:
    x = image.astype(np.float32)
    if mode == "dark":
        x = x * 0.35
    elif mode == "blur":
        x = cv2.GaussianBlur(image, ( nine := 9, nine), 2.0).astype(np.float32)
    elif mode == "noise":
        rng = np.random.default_rng(20261029)
        x = x + rng.normal(0.0, 25.0, x.shape).astype(np.float32)
    return np.clip(x, 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fusion-weights", required=True)
    p.add_argument("--data-dir", default=r"D:\dronevehicle_feasibility\yolo_obb_data")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--limit", type=int, default=300, help="First N sorted validation pairs; 0 means all")
    p.add_argument("--output-dir", default="runs/gate_perturbation")
    args = p.parse_args()

    device = torch.device(args.device if args.device == "cpu" or not args.device.isdigit()
                          else f"cuda:{args.device}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is unavailable")

    import __main__
    import train_yolo_obb_fusion
    __main__.ReliabilityOBBModel = train_yolo_obb_fusion.ReliabilityOBBModel
    __main__.ReliabilityGate = train_yolo_obb_fusion.ReliabilityGate
    from ultralytics import YOLO

    yolo = YOLO(args.fusion_weights, task="obb")
    gate = yolo.model.reliability_gate.to(device).eval()
    data_dir = Path(args.data_dir).resolve()
    rgb_dir = data_dir / "images" / "rgb" / "val"
    paths = sorted(x for x in rgb_dir.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
    if args.limit > 0:
        paths = paths[:args.limit]
    if not paths:
        raise RuntimeError(f"No images found: {rgb_dir}")

    variants = [
        ("original", None, None),
        ("rgb_dark", "rgb", "dark"),
        ("ir_dark", "ir", "dark"),
        ("rgb_blur", "rgb", "blur"),
        ("ir_blur", "ir", "blur"),
        ("rgb_noise", "rgb", "noise"),
        ("ir_noise", "ir", "noise"),
    ]
    records = []
    missing = []
    for n, rp in enumerate(paths, 1):
        ip = paired_path(rp)
        rgb = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        ir = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if rgb is None or ir is None:
            missing.append(rp.name)
            continue
        if ir.shape[:2] != rgb.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        for name, target, mode in variants:
            r, i = rgb.copy(), ir.copy()
            if target == "rgb":
                r = degrade(r, mode)
            elif target == "ir":
                i = degrade(i, mode)
            r = cv2.resize(r, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            i = cv2.resize(i, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            pair = np.concatenate((r, i), axis=2)
            x = torch.from_numpy(pair).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
            with torch.inference_mode():
                _, w = gate(x)
            rw = float(w[0, 0].mean().item())
            iw = float(w[0, 1].mean().item())
            records.append({
                "image": rp.name,
                "variant": name,
                "rgb_weight": rw,
                "ir_weight": iw,
                "rgb_minus_ir": rw - iw,
            })
        if n % 50 == 0:
            print(f"Processed {n}/{len(paths)} pairs")

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "gate_perturbation_per_image.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "variant", "rgb_weight", "ir_weight", "rgb_minus_ir"])
        writer.writeheader()
        writer.writerows(records)

    summary = {}
    for name, _, _ in variants:
        vals = [r for r in records if r["variant"] == name]
        summary[name] = {
            "n": len(vals),
            "rgb_weight_mean": float(np.mean([r["rgb_weight"] for r in vals])),
            "ir_weight_mean": float(np.mean([r["ir_weight"] for r in vals])),
            "rgb_minus_ir_mean": float(np.mean([r["rgb_minus_ir"] for r in vals])),
        }
    base = {r["image"]: r for r in records if r["variant"] == "original"}
    deltas = {}
    for name, _, _ in variants[1:]:
        paired = [(r, base[r["image"]]) for r in records if r["variant"] == name and r["image"] in base]
        deltas[name] = {
            "n": len(paired),
            "mean_change_rgb_weight_vs_original": float(np.mean([a["rgb_weight"] - b["rgb_weight"] for a, b in paired])),
            "mean_change_ir_weight_vs_original": float(np.mean([a["ir_weight"] - b["ir_weight"] for a, b in paired])),
        }
    report = {
        "checkpoint": str(Path(args.fusion_weights).resolve()),
        "images_requested": len(paths),
        "images_with_missing_pair": missing,
        "processed_images": len(base),
        "note": "Synthetic sensitivity test of gate weights only; does not measure detection mAP and cannot prove that the gate improves predictions.",
        "mean_weights_by_variant": summary,
        "paired_weight_change_vs_original": deltas,
    }
    json_path = out / "gate_perturbation_summary.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
