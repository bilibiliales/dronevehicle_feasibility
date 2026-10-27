#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cache YOLO-OBB RGB/IR validation predictions and compare decision-level fusion.

Validation only. Main metric treats the five vehicle subclasses as one "vehicle"
class to reduce the impact of cross-modality class-label disagreement.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


def image_index(folder: Path) -> dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            if p.stem in files:
                raise RuntimeError(f"Duplicate image stem in {folder}: {p.stem}")
            files[p.stem] = p
    return files


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(4, 2)
    b = np.asarray(b, dtype=np.float32).reshape(4, 2)
    area_a = abs(float(cv2.contourArea(a)))
    area_b = abs(float(cv2.contourArea(b)))
    if area_a <= 1e-8 or area_b <= 1e-8:
        return 0.0
    try:
        inter, _ = cv2.intersectConvexConvex(a, b)
    except cv2.error:
        return 0.0
    union = area_a + area_b - float(inter)
    return float(inter) / union if union > 1e-8 else 0.0


def read_labels(path: Path, width: int, height: int) -> list[dict]:
    """YOLO OBB row: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized)."""
    out = []
    if not path.exists():
        raise FileNotFoundError(f"Missing validation label: {path}")
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 9:
            raise ValueError(f"{path}:{line_no}: expected 9 values, got {len(parts)}")
        vals = [float(x) for x in parts]
        poly = np.array(vals[1:], dtype=np.float32).reshape(4, 2)
        poly[:, 0] *= width
        poly[:, 1] *= height
        if abs(float(cv2.contourArea(poly))) > 1e-6:
            out.append({"poly": poly.tolist(), "cls": 0})
    return out


def mean_luma(path: Path) -> float:
    with Image.open(path) as im:
        return float(np.asarray(im.convert("L"), dtype=np.float32).mean())


def infer(model_path: Path, image_paths: dict[str, Path], imgsz: int,
          device: str, conf: float, cache_path: Path) -> dict:
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if cache.get("model") == str(model_path.resolve()) and cache.get("imgsz") == imgsz and cache.get("conf") == conf:
            print(f"[CACHE] {cache_path}")
            return cache["predictions"]
    print(f"[INFER] {model_path}")
    model = YOLO(str(model_path))
    out = {}
    for i, (key, path) in enumerate(sorted(image_paths.items()), 1):
        result = model.predict(
            source=str(path), imgsz=imgsz, device=device, conf=conf,
            verbose=False, save=False, augment=False
        )[0]
        h, w = result.orig_shape
        rows = []
        if result.obb is not None and len(result.obb):
            polys = result.obb.xyxyxyxy.cpu().numpy()
            scores = result.obb.conf.cpu().numpy()
            classes = result.obb.cls.cpu().numpy().astype(int)
            for poly, score, cls in zip(polys, scores, classes):
                if abs(float(cv2.contourArea(poly.astype(np.float32)))) > 1e-6:
                    rows.append({
                        "poly": poly.astype(float).tolist(),
                        "score": float(score),
                        "cls": 0,  # collapse all five vehicle categories
                        "source_cls": int(cls),
                    })
        out[key] = {"width": int(w), "height": int(h), "predictions": rows}
        if i % 100 == 0 or i == len(image_paths):
            print(f"  {i}/{len(image_paths)}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "model": str(model_path.resolve()), "imgsz": imgsz, "conf": conf, "predictions": out
    }), encoding="utf-8")
    return out


def greedy_nms(rows: list[dict], iou_threshold: float) -> list[dict]:
    ordered = sorted(rows, key=lambda r: r["score"], reverse=True)
    kept = []
    for row in ordered:
        if all(polygon_iou(np.array(row["poly"]), np.array(k["poly"])) < iou_threshold
               for k in kept):
            kept.append(row)
    return kept


def ap_for_class(gt_by_key: dict[str, list], pred_by_key: dict[str, list],
                 iou_threshold: float) -> tuple[float, float, float, float]:
    n_gt = sum(len(v) for v in gt_by_key.values())
    if n_gt == 0:
        return 0.0, 0.0, 0.0, 0.0
    ranked = []
    for key, rows in pred_by_key.items():
        for row in rows:
            ranked.append((float(row["score"]), key, row["poly"]))
    ranked.sort(key=lambda x: x[0], reverse=True)
    matched = {k: set() for k in gt_by_key}
    tp, fp = [], []
    for _, key, poly in ranked:
        gts = gt_by_key.get(key, [])
        best_iou, best_idx = 0.0, -1
        for idx, gt in enumerate(gts):
            if idx in matched.setdefault(key, set()):
                continue
            iou = polygon_iou(np.asarray(poly), np.asarray(gt["poly"]))
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched[key].add(best_idx)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    if not ranked:
        return 0.0, 0.0, 0.0, 0.0
    tp_c = np.cumsum(tp, dtype=float)
    fp_c = np.cumsum(fp, dtype=float)
    recall = tp_c / n_gt
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-12)
    # 101-point interpolated AP
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        ap += (float(precision[recall >= r].max()) if np.any(recall >= r) else 0.0) / 101
    f1_curve = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.argmax(f1_curve))
    best_p = float(precision[best_idx])
    best_r = float(recall[best_idx])
    best_f1 = float(f1_curve[best_idx])
    return ap, best_p, best_r, best_f1


def evaluate(name: str, gt_by_key: dict[str, list], pred_by_key: dict[str, list],
             ious: list[float], subset_keys: set[str] | None = None) -> dict:
    gt = gt_by_key if subset_keys is None else {k: v for k, v in gt_by_key.items() if k in subset_keys}
    pred = pred_by_key if subset_keys is None else {k: v for k, v in pred_by_key.items() if k in subset_keys}
    metrics = {}
    for threshold in ious:
        ap, p, r, f1 = ap_for_class(gt, pred, threshold)
        metrics[f"AP{threshold:.2f}"] = round(ap, 5)
        if abs(threshold - 0.5) < 1e-6:
            metrics.update(P_at_best_F1=round(p, 5), R_at_best_F1=round(r, 5), max_F1=round(f1, 5))
    return {"method": name, "images": len(gt), "ground_truth_boxes": sum(map(len, gt.values())), **metrics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=r"D:\dronevehicle_feasibility\yolo_obb_data")
    ap.add_argument("--rgb-weights", required=True)
    ap.add_argument("--ir-weights", required=True)
    ap.add_argument("--out-dir", default=r"D:\dronevehicle_feasibility\fusion_results")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001, help="Low inference threshold for AP evaluation")
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--eval-ious", default="0.5,0.75")
    ap.add_argument("--dark-threshold", type=float, default=60.0,
                    help="RGB mean grayscale threshold (0-255) for dark-subset analysis")
    ap.add_argument("--rgb-floor", type=float, default=0.25,
                    help="Minimum RGB confidence multiplier for brightness-guided fusion")
    ap.add_argument("--force-infer", action="store_true", help="Ignore cached predictions")
    args = ap.parse_args()

    data = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    rgb_dir = data / "images" / "rgb" / "val"
    ir_dir = data / "images" / "ir" / "val"
    rgb_labels = data / "labels" / "rgb" / "val"
    ir_labels = data / "labels" / "ir" / "val"
    for p in [rgb_dir, ir_dir, rgb_labels, ir_labels]:
        if not p.is_dir():
            raise SystemExit(f"Required path missing: {p}")

    rgb_images, ir_images = image_index(rgb_dir), image_index(ir_dir)
    common = sorted(set(rgb_images) & set(ir_images))
    if not common:
        raise SystemExit("No paired RGB/IR validation image stems found.")
    if set(rgb_images) != set(ir_images):
        print(f"[WARN] RGB-only={len(set(rgb_images)-set(ir_images))}, IR-only={len(set(ir_images)-set(rgb_images))}")
    rgb_images = {k: rgb_images[k] for k in common}
    ir_images = {k: ir_images[k] for k in common}

    rgb_cache = out_dir / "cache_rgb.json"
    ir_cache = out_dir / "cache_ir.json"
    if args.force_infer:
        for p in (rgb_cache, ir_cache):
            if p.exists():
                p.unlink()
    rgb_pred = infer(Path(args.rgb_weights), rgb_images, args.imgsz, args.device, args.conf, rgb_cache)
    ir_pred = infer(Path(args.ir_weights), ir_images, args.imgsz, args.device, args.conf, ir_cache)

    gt_rgb, gt_ir, brightness = {}, {}, {}
    for key in common:
        w, h = rgb_pred[key]["width"], rgb_pred[key]["height"]
        gt_rgb[key] = read_labels(rgb_labels / f"{key}.txt", w, h)
        gt_ir[key] = read_labels(ir_labels / f"{key}.txt", w, h)
        brightness[key] = mean_luma(rgb_images[key])
    # The primary evaluation uses RGB annotations consistently; IR labels are summarized for transparency.
    gt = gt_rgb
    dark_keys = {k for k in common if brightness[k] < args.dark_threshold}
    light_keys = set(common) - dark_keys
    print(f"[DATA] paired={len(common)} RGB_GT={sum(map(len, gt_rgb.values()))} "
          f"IR_GT={sum(map(len, gt_ir.values()))}; dark={len(dark_keys)} light={len(light_keys)}")
    if sum(map(len, gt_rgb.values())) != sum(map(len, gt_ir.values())):
        print("[WARN] RGB/IR annotation counts differ. Main metric uses RGB validation labels only; "
              "interpret modality-specific AP cautiously and report this limitation.")

    methods = {}
    methods["RGB"] = {k: rgb_pred[k]["predictions"] for k in common}
    methods["IR"] = {k: ir_pred[k]["predictions"] for k in common}
    merged, guided = {}, {}
    for key in common:
        r = [dict(x) for x in rgb_pred[key]["predictions"]]
        i = [dict(x) for x in ir_pred[key]["predictions"]]
        merged[key] = greedy_nms(r + i, args.nms_iou)
        luma = brightness[key]
        # Linear confidence multiplier: dark RGB is downweighted, bright RGB approaches full weight.
        weight = args.rgb_floor + (1.0 - args.rgb_floor) * min(1.0, max(0.0, luma / 128.0))
        rg = [dict(x, score=float(x["score"]) * weight) for x in r]
        guided[key] = greedy_nms(rg + i, args.nms_iou)
    methods["MergeNMS"] = merged
    methods["BrightnessGuided"] = guided

    ious = [float(x.strip()) for x in args.eval_ious.split(",") if x.strip()]
    rows = []
    for name, pred in methods.items():
        rows.append(evaluate(name, gt, pred, ious))
        rows.append(evaluate(name + "_dark", gt, pred, ious, dark_keys))
        rows.append(evaluate(name + "_light", gt, pred, ious, light_keys))

    report = {
        "data_dir": str(data), "rgb_weights": str(Path(args.rgb_weights).resolve()),
        "ir_weights": str(Path(args.ir_weights).resolve()), "imgsz": args.imgsz,
        "conf": args.conf, "nms_iou": args.nms_iou, "eval_ious": ious,
        "primary_label_reference": "RGB validation labels; five classes collapsed into one vehicle class",
        "paired_images": len(common), "rgb_gt_instances": sum(map(len, gt_rgb.values())),
        "ir_gt_instances": sum(map(len, gt_ir.values())),
        "rgb_ir_gt_count_difference": sum(map(len, gt_ir.values())) - sum(map(len, gt_rgb.values())),
        "dark_threshold": args.dark_threshold, "rgb_floor": args.rgb_floor,
        "dark_images": len(dark_keys), "light_images": len(light_keys),
        "brightness_mean": round(float(np.mean(list(brightness.values()))), 3),
        "results": rows,
        "notes": [
            "Validation split only; do not use this report as a test-set result.",
            "Main metrics use RGB labels consistently because RGB/IR annotation counts differ.",
            "Five predicted classes are collapsed into one vehicle class; rotated polygon IoU is used.",
            "Brightness is a proxy for low RGB visibility, not a general fog/blur quality estimator.",
            "MergeNMS and BrightnessGuided are post-hoc decision-level fusion baselines, not learned feature fusion."
        ]
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "fusion_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = out_dir / "fusion_report.csv"
    import csv
    keys = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print("\n=== Validation results (single vehicle class; RGB labels) ===")
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    print(f"\nSaved: {report_path}\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
