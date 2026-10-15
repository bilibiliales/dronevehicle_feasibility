
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DroneVehicle RGB-IR feasibility checker (no training, no inference).

Purpose:
  1. Inspect train/val/test split pairing for RGB images, IR images,
     RGB XML labels and IR XML labels.
  2. Parse Pascal-VOC-like XML containing oriented 4-point polygons.
  3. Compare modality-specific annotations by polygon IoU (axis-aligned
     bbox IoU by default; optional Shapely polygon IoU).
  4. Export CSV/JSON summaries to inform whether label-disagreement-aware
     or conflict-aware fusion is worth implementing.

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
  --<split>-rgb-dir, --<split>-ir-dir, --<split>-rgb-label-dir, --<split>-ir-label-dir
where <split> is train, val or test. The two label flags also accept the plural
spelling --<split>-rgb-labels-dir / --<split>-ir-labels-dir (both forms work).
A split root can be supplied with --data-root; each split is found under
train/val/test, but if you pass the explicit --<split>-*-dir flags for a split it
is processed even when the matching train/val/test sub-directory is absent.
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
from dataclasses import dataclass, asdict
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
    # labels/rgb or annotations/ir
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
    if not split_root.exists():
        return {}
    dirs = find_dirs(split_root)
    found: Dict[str, List[Path]] = defaultdict(list)
    for p in dirs:
        role = classify_dir(p)
        if role:
            # Prefer shallow matching directories to avoid nested duplicates.
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
            # The example uses x1/y1 ... x4/y4.
            for i in range(1, 9):
                x = poly.findtext(f"x{i}")
                y = poly.findtext(f"y{i}")
                if x is not None and y is not None:
                    points.append((float(x), float(y)))
            if len(points) < 3:
                # Also accept point/point-like alternative encodings.
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

def bbox_iou(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2,bx2)-max(ax1,bx1))
    ih = max(0.0, min(ay2,by2)-max(ay1,by1))
    inter = iw * ih
    aa = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    ba = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = aa + ba - inter
    return inter / union if union > 0 else 0.0

def polygon_iou(poly_a, poly_b) -> Optional[float]:
    try:
        from shapely.geometry import Polygon
        a, b = Polygon(poly_a), Polygon(poly_b)
        if not a.is_valid: a = a.buffer(0)
        if not b.is_valid: b = b.buffer(0)
        union = a.union(b).area
        return a.intersection(b).area / union if union > 0 else 0.0
    except ImportError:
        return None
    except Exception:
        return 0.0

def greedy_match(rgb_objs: List[ObjectAnnotation], ir_objs: List[ObjectAnnotation],
                 iou_threshold: float, use_polygon_iou: bool):
    candidates = []
    for i, a in enumerate(rgb_objs):
        for j, b in enumerate(ir_objs):
            if a.name != b.name:
                continue
            iou = polygon_iou(a.polygon, b.polygon) if use_polygon_iou else None
            if iou is None:
                iou = bbox_iou(a.bbox, b.bbox)
            candidates.append((iou, i, j))
    candidates.sort(reverse=True)
    used_a, used_b, matches = set(), set(), []
    for iou, i, j in candidates:
        if iou < iou_threshold or i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j)
        matches.append((i, j, iou))
    unmatched_rgb = [i for i in range(len(rgb_objs)) if i not in used_a]
    unmatched_ir = [j for j in range(len(ir_objs)) if j not in used_b]
    return matches, unmatched_rgb, unmatched_ir

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

def inspect_split(split: str, root: Path, args) -> dict:
    discovered = discover_dirs(root)
    role_paths = {}
    for role in ("rgb", "ir", "rgb_labels", "ir_labels"):
        override = getattr(args, f"{split}_{role}_dir", None)
        if override:
            role_paths[role] = Path(override)
        else:
            role_paths[role] = discovered.get(role)
    imgs_rgb = list_files(role_paths["rgb"], IMAGE_EXTS)
    imgs_ir = list_files(role_paths["ir"], IMAGE_EXTS)
    xml_rgb = list_files(role_paths["rgb_labels"], {".xml"})
    xml_ir = list_files(role_paths["ir_labels"], {".xml"})
    idx_rgb, dup_rgb = make_index(imgs_rgb, image_stem_key)
    idx_ir, dup_ir = make_index(imgs_ir, image_stem_key)
    lab_rgb, dup_lr = make_index(xml_rgb, xml_stem_key)
    lab_ir, dup_li = make_index(xml_ir, xml_stem_key)
    all_keys = sorted(set(idx_rgb) | set(idx_ir) | set(lab_rgb) | set(lab_ir))
    rows = []
    totals = Counter()
    class_disagreements = Counter()
    examples = []
    for k in all_keys:
        rp, ip = idx_rgb.get(k), idx_ir.get(k)
        rxml, ixml = lab_rgb.get(k), lab_ir.get(k)
        ra = parse_xml(rxml) if rxml else None
        ia = parse_xml(ixml) if ixml else None
        ro = ra.objects if ra else []
        io = ia.objects if ia else []
        matches, ur, ui = greedy_match(ro, io, args.iou_threshold, args.polygon_iou)
        row = {
            "split": split, "key": k,
            "rgb_image": str(rp) if rp else "",
            "ir_image": str(ip) if ip else "",
            "rgb_xml": str(rxml) if rxml else "",
            "ir_xml": str(ixml) if ixml else "",
            "has_rgb_image": bool(rp), "has_ir_image": bool(ip),
            "has_rgb_label": bool(rxml), "has_ir_label": bool(ixml),
            "rgb_objects": len(ro), "ir_objects": len(io),
            "matched_same_class": len(matches),
            "unmatched_rgb": len(ur), "unmatched_ir": len(ui),
            "rgb_classes": json.dumps(dict(Counter(o.name for o in ro)), ensure_ascii=False),
            "ir_classes": json.dumps(dict(Counter(o.name for o in io)), ensure_ascii=False),
            "rgb_parse_errors": "; ".join(ra.parse_errors) if ra else "",
            "ir_parse_errors": "; ".join(ia.parse_errors) if ia else "",
            "mean_matched_iou": round(sum(m[2] for m in matches)/len(matches), 4) if matches else None,
        }
        rows.append(row)
        totals["keys"] += 1
        for present, label in [(rp is not None,"rgb_images"),(ip is not None,"ir_images"),
                               (rxml is not None,"rgb_xml"),(ixml is not None,"ir_xml")]:
            if present: totals[label] += 1
        totals["rgb_objects"] += len(ro); totals["ir_objects"] += len(io)
        totals["matched_same_class"] += len(matches)
        totals["unmatched_rgb"] += len(ur); totals["unmatched_ir"] += len(ui)
        for i in ur: class_disagreements["rgb_only:"+ro[i].name] += 1
        for i in ui: class_disagreements["ir_only:"+io[i].name] += 1
        if (ur or ui) and len(examples) < 30:
            examples.append({
                "key": k, "rgb_xml": str(rxml) if rxml else "", "ir_xml": str(ixml) if ixml else "",
                "rgb_only": [{"class": ro[i].name, "bbox": ro[i].bbox} for i in ur[:10]],
                "ir_only": [{"class": io[i].name, "bbox": io[i].bbox} for i in ui[:10]],
            })
    summary = {
        "split": split, "root": str(root),
        "directories": {k: str(v) if v else None for k,v in role_paths.items()},
        "counts": dict(totals),
        "unmatched_by_class": dict(class_disagreements),
        "duplicate_keys": {"rgb_images": {k:v for k,v in dup_rgb.items()},
                           "ir_images": {k:v for k,v in dup_ir.items()},
                           "rgb_labels": {k:v for k,v in dup_lr.items()},
                           "ir_labels": {k:v for k,v in dup_li.items()}},
        "examples_of_unmatched": examples,
        "rows": rows,
    }
    return summary

def main():
    ap = argparse.ArgumentParser(description="Inspect DroneVehicle RGB/IR pairs and modality annotation differences.")
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Root containing train/, val/, test/ subdirectories.")
    ap.add_argument("--out-dir", type=Path, default=Path("feasibility_report"))
    ap.add_argument("--iou-threshold", type=float, default=0.5,
                    help="Same-class matching IoU threshold (default 0.5).")
    ap.add_argument("--polygon-iou", action="store_true",
                    help="Use exact polygon IoU if Shapely is installed; otherwise falls back to bbox IoU.")
    # Each role may be passed with one or more flag spellings. The label roles
    # accept both the singular (-label-dir, matching the README) and the plural
    # (-labels-dir, the original spelling) so either command line works.
    _role_flags = {
        "rgb":        ["--{s}-rgb-dir"],
        "ir":         ["--{s}-ir-dir"],
        "rgb_labels": ["--{s}-rgb-label-dir", "--{s}-rgb-labels-dir"],
        "ir_labels":  ["--{s}-ir-label-dir", "--{s}-ir-labels-dir"],
    }
    for split in ("train", "val", "test"):
        for role in ("rgb", "ir", "rgb_labels", "ir_labels"):
            flags = [f.format(s=split) for f in _role_flags[role]]
            ap.add_argument(*flags, dest=f"{split}_{role}_dir",
                            type=Path, default=None,
                            help=f"Explicit {split} {role} directory, overrides discovery.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    full = {"data_root": str(args.data_root), "iou_threshold": args.iou_threshold,
            "polygon_iou_requested": args.polygon_iou, "splits": {}}
    all_rows = []
    for split in ("train", "val", "test"):
        split_root = args.data_root / split
        explicit = any(getattr(args, f"{split}_{role}_dir", None)
                       for role in ("rgb", "ir", "rgb_labels", "ir_labels"))
        if not split_root.exists() and not explicit:
            print(f"[WARN] Missing split directory: {split_root}", file=sys.stderr)
            continue
        report = inspect_split(split, split_root, args)
        full["splits"][split] = {k:v for k,v in report.items() if k != "rows"}
        all_rows.extend(report["rows"])
        print(f"\n[{split}]")
        print(json.dumps(full["splits"][split], ensure_ascii=False, indent=2))
    if not full["splits"]:
        raise SystemExit("No train/val/test split found under --data-root.")
    csv_path = args.out_dir / "pair_annotation_audit.csv"
    if all_rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader(); writer.writerows(all_rows)
    json_path = args.out_dir / "pair_annotation_audit.json"
    json_path.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print("\nInterpretation caution: unmatched annotations may be caused by imperfect matching,")
    print("registration differences, annotation policy, or missing labels; do not assume they are errors.")

if __name__ == "__main__":
    main()