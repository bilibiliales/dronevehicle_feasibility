
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DroneVehicle RGB-IR feasibility checker (no training, no inference).

Purpose:
  1. Inspect train/val/test split pairing for RGB images, IR images,
     RGB XML labels and IR XML labels.
  2. Parse Pascal-VOC-like XML containing oriented 4-point polygons.
  3. Compare modality-specific annotations and quantify label disagreement.
  4. NEW - Matching-algorithm SENSITIVITY analysis: run bbox IoU vs polygon
     IoU across several IoU thresholds (e.g. 0.3/0.5/0.7) and report how much
     the "unmatched rate" moves. If it swings a lot, the disagreement number
     is NOT a reliable measure of true annotation gaps - it is an artifact of
     the matcher. See --sensitivity.
  5. NEW - Visualization: render a few representative samples side-by-side
     (RGB + IR images with their own annotation boxes), focused on images with
     many IR-only `car` boxes, to separate real object gaps from registration
     shift, class-spelling differences and matching failures. See --visualize.

Modes (mutually exclusive, default = audit):
  audit        single matcher (bbox IoU, or polygon IoU with --polygon-iou),
               single threshold. Produces pair_annotation_audit.csv/.json.
  sensitivity  cross product of metrics x thresholds. Produces
               sensitivity_matrix.csv/.json with an automatic reliability verdict.
  visualize    pick representative samples and save side-by-side PNG figures.

Expected directory layout (DroneVehicle, as distributed):
  dataset/
    train/
      trainimg/     # RGB images (.jpg)
      trainimgr/    # IR  images (.jpg)
      trainlabel/   # RGB XML labels (.xml)
      trainlabelr/  # IR  XML labels (.xml)
    val/
      valimg/   valimgr/   vallabel/   vallabelr/
    test/
      testimg/  testimgr/  testlabel/  testlabelr/

Image and annotation files use matching numeric basenames (e.g. ``00001.jpg``
and ``00001.xml``). Within each split the non-``r`` folders hold RGB data and
the folders ending in ``r`` hold IR data.

These folder names are NOT recognized by automatic RGB/IR folder discovery, so
pass the directories explicitly for each split with:
  --<split>-rgb-dir, --<split>-ir-dir,
  --<split>-rgb-label-dir (also accepts --<split>-rgb-labels-dir),
  --<split>-ir-label-dir  (also accepts --<split>-ir-labels-dir)
where <split> is train, val or test. A split root can be supplied with
--data-root; each split is found under train/val/test.

Research hygiene reminder: train is for analysis, val may confirm a phenomenon,
but test annotations are for integrity auditing only - do NOT tune the method on
test repeatedly.
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RGB_TOKENS = {"rgb", "visible", "vis", "color", "colour", "rgb_images", "visible_images"}
IR_TOKENS = {"ir", "infrared", "thermal", "tir", "ir_images", "infrared_images"}
RGB_LABEL_TOKENS = {"rgb_label", "rgb_labels", "visible_label", "visible_labels",
                    "annotations_rgb", "ann_rgb", "labels_rgb", "rgb_annotations"}
IR_LABEL_TOKENS = {"ir_label", "ir_labels", "infrared_label", "infrared_labels",
                   "annotations_ir", "ann_ir", "labels_ir", "ir_annotations"}
LABEL_TOKENS = {"label", "labels", "annotation", "annotations", "ann", "xml", "groundtruth", "gt"}


@dataclass
class ObjectAnnotation:
    name: str
    polygon: List[Tuple[float, float]]
    bbox: Tuple[float, float, float, float]
    difficult: int = 0


@dataclass
class ImageAnnotation:
    xml_path: str
    filename: str
    width: int
    height: int
    objects: List[ObjectAnnotation]
    parse_errors: List[str]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def image_stem_key(path: Path) -> str:
    """Normalize common RGB/IR suffixes, e.g. DJI_0001_R -> dji0001."""
    stem = path.stem.lower()
    stem = re.sub(r"([_-])(r|rgb|ir|i|t|thermal|visible)$", "", stem)
    return norm(stem)


def xml_stem_key(path: Path) -> str:
    return image_stem_key(path)


def find_dirs(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_dir()]


def classify_dir(p: Path) -> str:
    n = norm(p.name)
    if n in {norm(x) for x in RGB_LABEL_TOKENS}:
        return "rgb_labels"
    if n in {norm(x) for x in IR_LABEL_TOKENS}:
        return "ir_labels"
    parent = norm(p.parent.name)
    if n in {norm(x) for x in LABEL_TOKENS}:
        if parent in {norm(x) for x in RGB_TOKENS}:
            return "rgb_labels"
        if parent in {norm(x) for x in IR_TOKENS}:
            return "ir_labels"
    if n in {norm(x) for x in RGB_TOKENS}:
        return "rgb"
    if n in {norm(x) for x in IR_TOKENS}:
        return "ir"
    return ""


def discover_dirs(split_root: Path) -> Dict[str, Path]:
    dirs = find_dirs(split_root)
    found: Dict[str, List[Path]] = defaultdict(list)
    for p in dirs:
        role = classify_dir(p)
        if role:
            found[role].append(p)
    result = {}
    for role, paths in found.items():
        paths.sort(key=lambda p: (len(p.parts), len(str(p))))
        result[role] = paths[0]
    return result


def parse_xml(xml_path: Path) -> ImageAnnotation:
    errors: List[str] = []
    try:
        root = ET.parse(xml_path).getroot()
    except Exception as e:
        return ImageAnnotation(str(xml_path), xml_path.stem, 0, 0, [], [f"XML parse failed: {e}"])
    filename = (root.findtext("filename") or xml_path.stem).strip()
    size = root.find("size")
    width = int(float(size.findtext("width", "0"))) if size is not None else 0
    height = int(float(size.findtext("height", "0"))) if size is not None else 0
    objects: List[ObjectAnnotation] = []
    for idx, obj in enumerate(root.findall("object")):
        name = (obj.findtext("name") or "unknown").strip().lower()
        difficult = int(float(obj.findtext("difficult", "0") or 0))
        poly = obj.find("polygon")
        points: List[Tuple[float, float]] = []
        if poly is not None:
            for i in range(1, 9):
                x = poly.findtext(f"x{i}")
                y = poly.findtext(f"y{i}")
                if x is not None and y is not None:
                    points.append((float(x), float(y)))
            if len(points) < 3:
                for pt in poly.findall("point"):
                    x, y = pt.findtext("x"), pt.findtext("y")
                    if x is not None and y is not None:
                        points.append((float(x), float(y)))
        if len(points) < 3:
            errors.append(f"object[{idx}] has no valid polygon")
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        objects.append(ObjectAnnotation(name, points, bbox, difficult))
    return ImageAnnotation(str(xml_path), filename, width, height, objects, errors)


def bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return inter / union if union > 0 else 0.0


_POLY_OK = None


def polygon_iou(poly_a, poly_b) -> Optional[float]:
    global _POLY_OK
    try:
        from shapely.geometry import Polygon
        a, b = Polygon(poly_a), Polygon(poly_b)
        if not a.is_valid:
            a = a.buffer(0)
        if not b.is_valid:
            b = b.buffer(0)
        union = a.union(b).area
        _POLY_OK = True
        return a.intersection(b).area / union if union > 0 else 0.0
    except ImportError:
        _POLY_OK = False
        return None
    except Exception:
        return 0.0


def pair_iou(a: ObjectAnnotation, b: ObjectAnnotation, metric: str) -> float:
    """Return the IoU between two objects under the requested metric.

    metric in {"bbox", "polygon"}; polygon falls back to bbox when Shapely is
    unavailable so the sensitivity sweep never crashes."""
    if metric == "polygon":
        iou = polygon_iou(a.polygon, b.polygon)
        if iou is not None:
            return iou
    return bbox_iou(a.bbox, b.bbox)


def greedy_match(rgb_objs: List[ObjectAnnotation], ir_objs: List[ObjectAnnotation],
                 iou_threshold: float, metric: str = "bbox"):
    candidates = []
    for i, a in enumerate(rgb_objs):
        for j, b in enumerate(ir_objs):
            if a.name != b.name:
                continue
            candidates.append((pair_iou(a, b, metric), i, j))
    candidates.sort(reverse=True)
    used_a, used_b, matches = set(), set(), []
    for iou, i, j in candidates:
        if iou < iou_threshold or i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((i, j, iou))
    unmatched_rgb = [i for i in range(len(rgb_objs)) if i not in used_a]
    unmatched_ir = [j for j in range(len(ir_objs)) if j not in used_b]
    return matches, unmatched_rgb, unmatched_ir


def legacy_metric_flag(use_polygon: bool) -> str:
    return "polygon" if use_polygon else "bbox"


def list_files(folder: Optional[Path], suffixes) -> List[Path]:
    if folder is None or not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in suffixes])


def make_index(paths: List[Path], key_fn):
    out = {}
    duplicates = defaultdict(list)
    for p in paths:
        k = key_fn(p)
        if k in out:
            duplicates[k].append(str(p))
        else:
            out[k] = p
    return out, duplicates


def resolve_role_paths(split: str, root: Path, args) -> Dict[str, Optional[Path]]:
    discovered = discover_dirs(root) if root.exists() else {}
    role_paths: Dict[str, Optional[Path]] = {}
    for role in ("rgb", "ir", "rgb_labels", "ir_labels"):
        override = getattr(args, f"{split}_{role}_dir", None)
        role_paths[role] = Path(override) if override else discovered.get(role)
    return role_paths


def split_has_explicit_dirs(split: str, args) -> bool:
    return any(getattr(args, f"{split}_{role}_dir", None)
               for role in ("rgb", "ir", "rgb_labels", "ir_labels"))


def load_split_objects(split_root: Path, args, split: str):
    """Parse every key once and cache the RGB/IR object lists + image paths.

    Both audit and sensitivity reuse this so XML parsing / pairing is done a
    single time regardless of how many metrics x thresholds are evaluated."""
    role_paths = resolve_role_paths(split, split_root, args)
    idx_rgb, dup_rgb = make_index(list_files(role_paths["rgb"], IMAGE_EXTS), image_stem_key)
    idx_ir, dup_ir = make_index(list_files(role_paths["ir"], IMAGE_EXTS), image_stem_key)
    lab_rgb, dup_lr = make_index(list_files(role_paths["rgb_labels"], {".xml"}), xml_stem_key)
    lab_ir, dup_li = make_index(list_files(role_paths["ir_labels"], {".xml"}), xml_stem_key)
    all_keys = sorted(set(idx_rgb) | set(idx_ir) | set(lab_rgb) | set(lab_ir))
    limit = getattr(args, "limit", 0)
    if limit and limit > 0:
        all_keys = all_keys[:limit]

    cache = []
    for k in all_keys:
        rp, ip = idx_rgb.get(k), idx_ir.get(k)
        rxml, ixml = lab_rgb.get(k), lab_ir.get(k)
        ra = parse_xml(rxml) if rxml else None
        ia = parse_xml(ixml) if ixml else None
        cache.append({
            "key": k, "rp": rp, "ip": ip, "rxml": rxml, "ixml": ixml,
            "ro": ra.objects if ra else [], "io": ia.objects if ia else [],
            "r_err": ra.parse_errors if ra else [], "i_err": ia.parse_errors if ia else [],
            "has_rgb_image": bool(rp), "has_ir_image": bool(ip),
            "has_rgb_label": bool(rxml), "has_ir_label": bool(ixml),
        })
    dup = {"rgb_images": dict(dup_rgb), "ir_images": dict(dup_ir),
           "rgb_labels": dict(dup_lr), "ir_labels": dict(dup_li)}
    return cache, dup, role_paths


def audit_one_split(split: str, split_root: Path, args) -> dict:
    metric = legacy_metric_flag(args.polygon_iou)
    cache, dup, role_paths = load_split_objects(split_root, args, split)
    rows = []
    totals = Counter()
    class_disagreements = Counter()
    examples = []
    for c in cache:
        ro, io = c["ro"], c["io"]
        matches, ur, ui = greedy_match(ro, io, args.iou_threshold, metric)
        row = {
            "split": split, "key": c["key"],
            "rgb_image": str(c["rp"]) if c["rp"] else "",
            "ir_image": str(c["ip"]) if c["ip"] else "",
            "rgb_xml": str(c["rxml"]) if c["rxml"] else "",
            "ir_xml": str(c["ixml"]) if c["ixml"] else "",
            "has_rgb_image": c["has_rgb_image"], "has_ir_image": c["has_ir_image"],
            "has_rgb_label": c["has_rgb_label"], "has_ir_label": c["has_ir_label"],
            "rgb_objects": len(ro), "ir_objects": len(io),
            "matched_same_class": len(matches),
            "unmatched_rgb": len(ur), "unmatched_ir": len(ui),
            "rgb_classes": json.dumps(dict(Counter(o.name for o in ro)), ensure_ascii=False),
            "ir_classes": json.dumps(dict(Counter(o.name for o in io)), ensure_ascii=False),
            "rgb_parse_errors": "; ".join(c["r_err"]), "ir_parse_errors": "; ".join(c["i_err"]),
            "mean_matched_iou": round(sum(m[2] for m in matches) / len(matches), 4) if matches else None,
        }
        rows.append(row)
        totals["keys"] += 1
        for present, label in [(c["rp"] is not None, "rgb_images"), (c["ip"] is not None, "ir_images"),
                               (c["rxml"] is not None, "rgb_xml"), (c["ixml"] is not None, "ir_xml")]:
            if present:
                totals[label] += 1
        totals["rgb_objects"] += len(ro)
        totals["ir_objects"] += len(io)
        totals["matched_same_class"] += len(matches)
        totals["unmatched_rgb"] += len(ur)
        totals["unmatched_ir"] += len(ui)
        for i in ur:
            class_disagreements["rgb_only:" + ro[i].name] += 1
        for i in ui:
            class_disagreements["ir_only:" + io[i].name] += 1
        if (ur or ui) and len(examples) < 30:
            examples.append({
                "key": c["key"], "rgb_xml": str(c["rxml"]) if c["rxml"] else "",
                "ir_xml": str(c["ixml"]) if c["ixml"] else "",
                "rgb_only": [{"class": ro[i].name, "bbox": ro[i].bbox} for i in ur[:10]],
                "ir_only": [{"class": io[i].name, "bbox": io[i].bbox} for i in ui[:10]],
            })
    summary = {
        "split": split, "root": str(split_root),
        "matcher": {"metric": metric, "iou_threshold": args.iou_threshold},
        "directories": {k: str(v) if v else None for k, v in role_paths.items()},
        "counts": dict(totals),
        "unmatched_by_class": dict(class_disagreements),
        "duplicate_keys": dup,
        "examples_of_unmatched": examples,
        "rows": rows,
    }
    return summary


def sensitivity_one_split(split: str, split_root: Path, args) -> dict:
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()] or ["bbox"]
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()] or [0.5]
    cache, dup, role_paths = load_split_objects(split_root, args, split)

    # Pre-count objects once (independent of matcher).
    total_rgb_objs = sum(len(c["ro"]) for c in cache)
    total_ir_objs = sum(len(c["io"]) for c in cache)

    cells = []
    # Store per-key unmatched_ir class breakdown for the (bbox, mid-threshold) cell
    # so the visualizer can reuse the same selection policy.
    for metric in metrics:
        for thr in thresholds:
            matched = un_rgb = un_ir = 0
            cls_unmatched = Counter()
            for c in cache:
                m, ur, ui = greedy_match(c["ro"], c["io"], thr, metric)
                matched += len(m)
                un_rgb += len(ur)
                un_ir += len(ui)
                for i in ur:
                    cls_unmatched["rgb_only:" + c["ro"][i].name] += 1
                for i in ui:
                    cls_unmatched["ir_only:" + c["io"][i].name] += 1
            denom = total_rgb_objs + total_ir_objs
            cell = {
                "split": split, "metric": metric, "iou_threshold": thr,
                "rgb_objects": total_rgb_objs, "ir_objects": total_ir_objs,
                "matched_same_class": matched,
                "unmatched_rgb": un_rgb, "unmatched_ir": un_ir,
                "unmatched_total": un_rgb + un_ir,
                "unmatched_rate": round((un_rgb + un_ir) / denom, 4) if denom else None,
                "unmatched_ir_by_class": dict(cls_unmatched),
            }
            cells.append(cell)

    # Reliability verdict: how much does the unmatched_rate swing across the grid?
    rates = [c["unmatched_rate"] for c in cells if c["unmatched_rate"] is not None]
    verdict = {}
    if rates:
        lo, hi = min(rates), max(rates)
        spread = hi - lo
        rel = spread / lo if lo > 1e-9 else float("inf")
        # Also check the metric effect at the reference threshold.
        ref_thr = args.iou_threshold if args.iou_threshold in thresholds else thresholds[len(thresholds) // 2]
        rate_by_metric = {c["metric"]: c["unmatched_rate"] for c in cells if c["iou_threshold"] == ref_thr}
        bbox_only = [c["unmatched_rate"] for c in cells if c["metric"] == "bbox" and c["unmatched_rate"] is not None]
        poly_only = [c["unmatched_rate"] for c in cells if c["metric"] == "polygon" and c["unmatched_rate"] is not None]
        threshold_driven = (max(bbox_only) - min(bbox_only)) if len(bbox_only) > 1 else 0.0
        metric_driven = abs((max(poly_only) - min(poly_only))) if len(poly_only) > 1 else 0.0
        verdict = {
            "reference_threshold": ref_thr,
            "unmatched_rate_min": round(lo, 4), "unmatched_rate_max": round(hi, 4),
            "unmatched_rate_spread": round(spread, 4),
            "relative_spread": round(rel, 4) if rel != float("inf") else "inf",
            "threshold_driven_spread": round(threshold_driven, 4),
            "metric_driven_spread": round(metric_driven, 4),
        }
        if spread >= 0.10 or rel >= 0.5:
            verdict["conclusion"] = ("UNRELIABLE: unmatched rate swings strongly with matcher/threshold; "
                                     "treat label-disagreement as a matcher artifact, not proof of real gaps.")
        else:
            verdict["conclusion"] = ("STABLE-enough: unmatched rate is fairly insensitive here; still verify "
                                     "visually before trusting it as an annotation-gap measure.")
    return {
        "split": split, "root": str(split_root),
        "directories": {k: str(v) if v else None for k, v in role_paths.items()},
        "metrics": metrics, "thresholds": thresholds,
        "polygon_iou_available": _POLY_OK,
        "cells": cells,
        "verdict": verdict,
    }


def visualize_one_split(split: str, split_root: Path, args) -> dict:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as e:
        print(f"[WARN] Visualization needs matplotlib + Pillow ({e}); skipping.", file=sys.stderr)
        return {"split": split, "skipped": True, "reason": f"missing dep: {e}"}

    thr = args.iou_threshold
    metric = legacy_metric_flag(args.polygon_iou)
    focus = (args.focus_class or "car").strip().lower()
    cache, dup, role_paths = load_split_objects(split_root, args, split)

    scored = []
    for c in cache:
        m, ur, ui = greedy_match(c["ro"], c["io"], thr, metric)
        ir_only_focus = sum(1 for i in ui if c["io"][i].name == focus)
        if ir_only_focus <= 0:
            continue
        if not (c["rp"] and c["ip"]):
            continue
        scored.append((ir_only_focus, len(ur) + len(ui), c))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    chosen = scored[:args.visualize_top]

    out_viz = args.out_dir / "visual" / split
    out_viz.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rank, (ir_only_focus, total_un, c) in enumerate(chosen, 1):
        try:
            img_rgb = Image.open(c["rp"]).convert("RGB")
            img_ir = Image.open(c["ip"]).convert("RGB")
        except Exception as e:
            print(f"[WARN] cannot open images for {c['key']}: {e}", file=sys.stderr)
            continue
        m, ur, ui = greedy_match(c["ro"], c["io"], thr, metric)
        matched_rgb = {i for i, j, _ in m}
        matched_ir = {j for i, j, _ in m}
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        for ax, img, objs, matched, title, only_tag in (
            (axes[0], img_rgb, c["ro"], matched_rgb, f"RGB  ({c['key']})", "rgb_only"),
            (axes[1], img_ir, c["io"], matched_ir, f"IR  ({c['key']})", "ir_only"),
        ):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")
            for idx, o in enumerate(objs):
                x1, y1, x2, y2 = o.bbox
                if idx in matched:
                    color, ls, lw = "lime", "-", 1.5
                elif o.name == focus and only_tag == "ir_only":
                    color, ls, lw = "red", "-", 2.5
                elif only_tag == "ir_only":
                    color, ls, lw = "orange", "--", 1.5
                else:
                    color, ls, lw = "cyan", "--", 1.5
                ax.add_patch(plt.Rectangle((x1, y1), max(0, x2 - x1), max(0, y2 - y1),
                                            fill=False, edgecolor=color, linestyle=ls, linewidth=lw))
                ax.text(x1, max(0, y1 - 2), o.name, color=color, fontsize=7)
        handles = [plt.Line2D([0], [0], color="lime", lw=2, label="matched (same class)"),
                   plt.Line2D([0], [0], color="red", lw=2, label=f"IR-only {focus} (focus)"),
                   plt.Line2D([0], [0], color="orange", lw=2, ls="--", label="IR-only (other)"),
                   plt.Line2D([0], [0], color="cyan", lw=2, ls="--", label="RGB-only")]
        fig.legend(handles=handles, loc="lower center", ncol=4)
        fig.suptitle(f"[{split}] key={c['key']}  IR-only-{focus}={ir_only_focus}  total_unmatched={total_un}")
        png = out_viz / f"{rank:03d}_{c['key']}.png"
        fig.savefig(png, dpi=110, bbox_inches="tight")
        plt.close(fig)
        manifest.append({"rank": rank, "split": split, "key": c["key"],
                         "ir_only_focus": ir_only_focus, "total_unmatched": total_un,
                         "figure": str(png)})
    return {"split": split, "matcher": {"metric": metric, "iou_threshold": thr},
            "focus_class": focus, "figures_written": len(manifest), "manifest": manifest}


def main():
    ap = argparse.ArgumentParser(description="Inspect DroneVehicle RGB/IR pairs and modality annotation differences.")
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Root containing train/, val/, test/ subdirectories.")
    ap.add_argument("--out-dir", type=Path, default=Path("feasibility_report"))
    ap.add_argument("--iou-threshold", type=float, default=0.5,
                    help="Reference matching IoU threshold (default 0.5).")
    ap.add_argument("--polygon-iou", action="store_true",
                    help="Audit mode: use exact polygon IoU if Shapely is installed.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only audit the first N keys per split (0 = all). Use for big train set / quick looks.")

    # Mode selectors.
    ap.add_argument("--sensitivity", action="store_true",
                    help="Run matching-algorithm sensitivity analysis (metrics x thresholds).")
    ap.add_argument("--metrics", type=str, default="bbox,polygon",
                    help="Comma list for --sensitivity (default 'bbox,polygon').")
    ap.add_argument("--thresholds", type=str, default="0.3,0.5,0.7",
                    help="Comma list of IoU thresholds for --sensitivity (default '0.3,0.5,0.7').")
    ap.add_argument("--visualize", action="store_true",
                    help="Render representative side-by-side RGB/IR figures with annotations.")
    ap.add_argument("--visualize-top", type=int, default=8,
                    help="How many sample figures per split to render (default 8).")
    ap.add_argument("--focus-class", type=str, default="car",
                    help="Class to prioritize in visualization (default 'car').")
    ap.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to process, e.g. train,val."
    )

    # Per-split directory overrides; label dirs accept both singular and plural flag names.
    _role_flag = {"rgb": "rgb", "ir": "ir", "rgb_labels": "rgb-label", "ir_labels": "ir-label"}
    for split in ("train", "val", "test"):
        for role in ("rgb", "ir", "rgb_labels", "ir_labels"):
            flags = [f"--{split}-{_role_flag[role]}-dir"]
            if role.endswith("_labels"):
                flags.append(f"--{split}-{_role_flag[role]}s-dir")
            ap.add_argument(*flags, dest=f"{split}_{role}_dir", type=Path, default=None,
                            help=f"Explicit {split} {role} directory, overrides discovery.")

    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def split_to_process(split):
        return (args.data_root / split).exists() or split_has_explicit_dirs(split, args)

    mode = "sensitivity" if args.sensitivity else ("visualize" if args.visualize else "audit")
    full = {"data_root": str(args.data_root), "mode": mode,
            "iou_threshold": args.iou_threshold, "polygon_iou_requested": args.polygon_iou,
            "splits": {}}
    all_rows = []
    cells_all = []
    viz_all = []
    selected_splits = [
        s.strip().lower()
        for s in args.splits.split(",")
        if s.strip()
    ]
    invalid_splits = set(selected_splits) - {"train", "val", "test"}
    if invalid_splits:
        raise SystemExit(f"Invalid splits: {sorted(invalid_splits)}")

    for split in selected_splits:
        if not split_to_process(split):
            print(f"[WARN] Missing split directory and no explicit dirs: {split}", file=sys.stderr)
            continue
        split_root = args.data_root / split
        if mode == "sensitivity":
            rep = sensitivity_one_split(split, split_root, args)
            full["splits"][split] = rep
            cells_all.extend(rep["cells"])
            print(f"\n[{split}] SENSITIVITY ({', '.join(rep['metrics'])} x {rep['thresholds']})")
            print(json.dumps({"verdict": rep["verdict"]}, ensure_ascii=False, indent=2))
        elif mode == "visualize":
            rep = visualize_one_split(split, split_root, args)
            full["splits"][split] = rep
            viz_all.extend(rep.get("manifest", []))
            print(f"\n[{split}] VISUALIZE wrote {rep.get('figures_written', 0)} figures")
        else:
            rep = audit_one_split(split, split_root, args)
            full["splits"][split] = {k: v for k, v in rep.items() if k != "rows"}
            all_rows.extend(rep["rows"])
            print(f"\n[{split}]")
            print(json.dumps(full["splits"][split], ensure_ascii=False, indent=2))

    if not full["splits"]:
        raise SystemExit("No train/val/test split found under --data-root (and no explicit dirs given).")

    if mode == "audit":
        csv_path = args.out_dir / "pair_annotation_audit.csv"
        if all_rows:
            with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_rows)
            print(f"\nSaved: {csv_path}")
        json_path = args.out_dir / "pair_annotation_audit.json"
        json_path.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {json_path}")
        print("\nInterpretation caution: unmatched annotations may be caused by imperfect matching, "
              "registration differences, annotation policy, or missing labels; do not assume they are errors. "
              "Run --sensitivity next.")
    elif mode == "sensitivity":
        csv_path = args.out_dir / "sensitivity_matrix.csv"
        if cells_all:
            fieldnames = ["split", "metric", "iou_threshold", "rgb_objects", "ir_objects",
                          "matched_same_class", "unmatched_rgb", "unmatched_ir",
                          "unmatched_total", "unmatched_rate"]
            with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(cells_all)
            print(f"\nSaved: {csv_path}")
        json_path = args.out_dir / "sensitivity_matrix.json"
        json_path.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {json_path}")
        print("\nVerdict rule: if unmatched_rate_spread >= 0.10 or relative_spread >= 0.5, the disagreement "
              "count is a matcher artifact, not proof of real cross-modal gaps.")
    else:
        json_path = args.out_dir / "visual_manifest.json"
        json_path.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {json_path}")
        print(f"Figures under: {args.out_dir / 'visual'}")
        print("\nHow to read: green=matched same-class; red=IR-only focus class; "
              "orange=IR-only other; cyan=RGB-only. A red box sitting on top of an already-covered "
              "green area usually means registration shift or duplicate label, NOT a real miss.")


if __name__ == "__main__":
    main()