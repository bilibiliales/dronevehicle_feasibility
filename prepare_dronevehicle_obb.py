#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare DroneVehicle RGB/IR XML quadrilaterals as Ultralytics YOLO-OBB data.
Defaults match D:\\dronevehicle_feasibility\\dataset.
Images are hard-linked when possible (no duplicate image storage); copy fallback.
Fail-fast on missing pairs, duplicate normalized keys, invalid polygons, or split leakage.
"""
from __future__ import annotations
import argparse, json, os, re, shutil, sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from PIL import Image
import math
import yaml  # ultralytics 依赖里已带，无需额外安装

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = {
    "train": ("train/trainimg", "train/trainimgr", "train/trainlabel", "train/trainlabelr"),
    "val":   ("val/valimg", "val/valimgr", "val/vallabel", "val/vallabelr"),
    "test":  ("test/testimg", "test/testimgr", "test/testlabel", "test/testlabelr"),
}

def key_of(p: Path) -> str:
    # Dataset's RGB/IR counterparts normally share numeric stems.
    s = p.stem.lower()
    s = re.sub(r"([_-])(r|rgb|ir|i|t|thermal|visible)$", "", s)
    return re.sub(r"[^a-z0-9]+", "", s)

def index_files(folder: Path, suffixes):
    if not folder.is_dir():
        raise FileNotFoundError(f"Required directory not found: {folder}")
    out, dup = {}, []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in suffixes:
            k = key_of(p)
            if k in out:
                dup.append((k, str(out[k]), str(p)))
            else:
                out[k] = p
    if dup:
        raise RuntimeError(f"Duplicate normalized keys in {folder}: {dup[:5]}")
    return out

def parse_xml(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    objects = []

    name_normalize = {
        "feright_car": "freight_car",
        "feright car": "freight_car",
        "feright": "freight_car",
        "truvk": "truck",
        "*": "freight_car",
    }

    for i, obj in enumerate(root.findall("object")):
        name = (obj.findtext("name") or "").strip().lower()

        if not name:
            raise ValueError(
                f"{xml_path}: object[{i}] has no class name"
            )

        # 统一类别名称，确保 RGB、IR 和各数据集划分使用相同类别
        name = name_normalize.get(name, name)

        difficult = int(float(obj.findtext("difficult", "0") or 0))
        pts = None

        poly = obj.find("polygon")
        if poly is not None:
            pts = []

            for j in range(1, 5):
                x_elem = poly.find(f"x{j}")
                y_elem = poly.find(f"y{j}")

                if x_elem is None or y_elem is None:
                    print(
                        f"[WARN] {xml_path}: object[{i}] "
                        f"polygon missing x{j}/y{j}, "
                        "fallback to bndbox"
                    )
                    pts = None
                    break

                x_text = (x_elem.text or "").strip()
                y_text = (y_elem.text or "").strip()

                if not x_text or not y_text:
                    print(
                        f"[WARN] {xml_path}: object[{i}] "
                        f"polygon x{j}/y{j} empty, "
                        "fallback to bndbox"
                    )
                    pts = None
                    break

                pts.append((float(x_text), float(y_text)))

            if pts and len(set(pts)) < 3:
                print(
                    f"[WARN] {xml_path}: object[{i}] "
                    "degenerate polygon, fallback to bndbox"
                )
                pts = None

        if pts is None and obj.find("bndbox") is not None:
            bnd = obj.find("bndbox")

            xmin = float(bnd.findtext("xmin", "0"))
            ymin = float(bnd.findtext("ymin", "0"))
            xmax = float(bnd.findtext("xmax", "0"))
            ymax = float(bnd.findtext("ymax", "0"))

            pts = [
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
            ]

        if pts is None:
            raise ValueError(
                f"{xml_path}: object[{i}] has no usable polygon "
                "or bndbox. Exclude this paired sample "
                "before dataset conversion."
            )

        objects.append({
            "class": name,
            "points": pts,
            "difficult": difficult,
        })

    return objects

def _valid_polygon(poly: ET.Element | None) -> bool:
    """Return True only when polygon coordinates form a non-degenerate quadrilateral."""
    if poly is None:
        return False

    try:
        pts = []
        for j in range(1, 5):
            x_elem = poly.find(f"x{j}")
            y_elem = poly.find(f"y{j}")
            if x_elem is None or y_elem is None:
                return False

            x_text = (x_elem.text or "").strip()
            y_text = (y_elem.text or "").strip()
            if not x_text or not y_text:
                return False

            x, y = float(x_text), float(y_text)
            if not (os.path.exists and __import__("math").isfinite(x)
                    and __import__("math").isfinite(y)):
                return False
            pts.append((x, y))

        if len(set(pts)) < 3:
            return False

        # Shoelace formula: reject zero-area polygons, including collinear points.
        area2 = abs(sum(
            pts[i][0] * pts[(i + 1) % 4][1]
            - pts[(i + 1) % 4][0] * pts[i][1]
            for i in range(4)
        ))
        return area2 > 1e-6

    except (ValueError, TypeError, OverflowError):
        return False


def _valid_bndbox(obj: ET.Element) -> bool:
    """Return True only when the bounding box has finite, positive dimensions."""
    bnd = obj.find("bndbox")
    if bnd is None:
        return False

    try:
        values = [
            float(bnd.findtext(name, ""))
            for name in ("xmin", "ymin", "xmax", "ymax")
        ]
        import math
        if not all(math.isfinite(v) for v in values):
            return False

        xmin, ymin, xmax, ymax = values
        return xmax > xmin and ymax > ymin

    except (ValueError, TypeError, OverflowError):
        return False


def load_exclude_set(root: Path) -> set[str]:
    """
    Load existing exclusions and scan every split for unusable annotations.
    Any bad XML excludes its entire paired RGB/IR sample.
    """
    exclude_file = root / "exclude_keys.txt"

    # Preserve existing exclusions.
    exclude_keys: set[str] = set()
    if exclude_file.exists():
        exclude_keys = {
            line.strip()
            for line in exclude_file.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        print(f"[INFO] Loaded {len(exclude_keys)} existing exclusion keys")

    discovered: set[str] = set()

    for split, dirs in SPLITS.items():
        for rel_dir in (dirs[2], dirs[3]):
            lab_dir = root / rel_dir
            if not lab_dir.is_dir():
                continue

            for xml_path in sorted(lab_dir.glob("*.xml")):
                key = key_of(xml_path)
                exclude_key = f"{split}:{key}"

                try:
                    tree = ET.parse(xml_path)
                    xml_root = tree.getroot()

                    for i, obj in enumerate(xml_root.findall("object")):
                        # A valid polygon is preferred. A valid bndbox is
                        # an acceptable fallback when the polygon is invalid.
                        if (
                            not _valid_polygon(obj.find("polygon"))
                            and not _valid_bndbox(obj)
                        ):
                            discovered.add(exclude_key)
                            print(
                                f"[WARN] Excluding paired sample {exclude_key}: "
                                f"{xml_path.name} object[{i}] has no usable "
                                "polygon or bndbox"
                            )
                            break

                except (ET.ParseError, OSError, ValueError) as exc:
                    discovered.add(exclude_key)
                    print(
                        f"[WARN] Excluding paired sample {exclude_key}: "
                        f"cannot parse {xml_path}: {exc}"
                    )

    new_keys = discovered - exclude_keys
    exclude_keys.update(discovered)

    if exclude_keys:
        exclude_file.write_text(
            "\n".join(sorted(exclude_keys)) + "\n",
            encoding="utf-8"
        )

    print(
        f"[INFO] Exclusion list: {len(exclude_keys)} total keys, "
        f"{len(new_keys)} newly discovered: {exclude_file}"
    )
    return exclude_keys

def safe_link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=r"D:\dronevehicle_feasibility\dataset")
    ap.add_argument("--out", default=r"D:\dronevehicle_feasibility\yolo_obb_data")
    ap.add_argument("--allow-missing-modality", action="store_true",
                    help="Skip samples missing an image/XML in either modality. Default is fail-fast.")
    ap.add_argument("--include-difficult", action="store_true",
                    help="Include difficult=1 objects. Default excludes them.")
    ap.add_argument("--force", action="store_true", help="Delete and recreate output directory.")
    ap.add_argument("--exclude-list", default=None,
                    help="Path to exclude_keys.txt; default auto-detects dataset/exclude_keys.txt")
    args = ap.parse_args()
    root, out = Path(args.data_root).resolve(), Path(args.out).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if out.exists():
        if not args.force:
            raise FileExistsError(f"{out} already exists. Use --force only if safe to overwrite.")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    exclude_keys = load_exclude_set(root) if args.exclude_list is None else \
        {line.strip() for line in Path(args.exclude_list).read_text(encoding="utf-8").splitlines() if line.strip()}

    indexes = {}
    class_names = {
        "car",
        "truck",
        "bus",
        "van",
        "freight_car",
    }
    audit = {"data_root": str(root), "output": str(out), "splits": {}, "warnings": [],
             "excluded_keys_count": len(exclude_keys)}
    audit["coordinate_repairs"] = []   # 变更 1：坐标修复记录

    for split, dirs in SPLITS.items():
        rgb_dir, ir_dir, rgb_lab_dir, ir_lab_dir = [root / d for d in dirs]
        ix = {
            "rgb": index_files(rgb_dir, EXTS),
            "ir": index_files(ir_dir, EXTS),
            "rgb_xml": index_files(rgb_lab_dir, {".xml"}),
            "ir_xml": index_files(ir_lab_dir, {".xml"}),
        }
        key_sets = {k: set(v) for k, v in ix.items()}
        common = set.intersection(*key_sets.values())
        union = set.union(*key_sets.values())
        missing = {k: sorted(union - v) for k, v in key_sets.items()}
        if any(missing.values()) and not args.allow_missing_modality:
            sample = {k: v[:5] for k, v in missing.items() if v}
            raise RuntimeError(f"{split}: image/XML key mismatch. Examples: {sample}. "
                               "Inspect naming/paths; only use --allow-missing-modality if intentional.")
        if not common:
            raise RuntimeError(f"{split}: no common keys among RGB/IR images and XML labels")

        # 变更 2：按 "split:key" 排除，避免跨 split 同名文件误排除
        before = len(common)
        common = {k for k in common if f"{split}:{k}" not in exclude_keys}
        excluded = before - len(common)
        if excluded:
            print(f"[INFO] {split}: excluded {excluded} paired samples (RGB+IR)")
            audit["warnings"].append(f"{split}: {excluded} samples excluded")

        parsed = {}
        for modality in ("rgb", "ir"):
            lab_key = modality + "_xml"
            for k in sorted(common):
                objs = parse_xml(ix[lab_key][k])
                parsed[(modality, k)] = objs
                names = sorted(class_names)
                if not names:
                    raise RuntimeError("No usable classes found in XML annotations.")

                class_to_id = {n: i for i, n in enumerate(names)}
        indexes[split] = (ix, sorted(common), parsed)
        audit["splits"][split] = {
            "counts": {k: len(v) for k, v in ix.items()},
            "common_keys": len(common), "union_keys": len(union),
            "excluded": excluded,
            "missing_key_examples": {k: v[:10] for k, v in missing.items() if v},
            "objects": {m: sum(len(v) for (mm, _), v in parsed.items() if mm == m)
                        for m in ("rgb", "ir")},
        }

    names = sorted(class_names)
    if not names:
        raise RuntimeError("No usable classes found in XML annotations.")
    class_to_id = {n: i for i, n in enumerate(names)}
    audit["classes"] = class_to_id
    audit["label_format"] = "Ultralytics OBB: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized)"

    for split, (ix, keys, parsed) in indexes.items():
        for modality in ("rgb", "ir"):
            image_key = modality
            label_key = modality + "_xml"
            for k in keys:
                src = ix[image_key][k]
                dst = out / "images" / modality / split / (k + src.suffix.lower())
                safe_link_or_copy(src, dst)
                try:
                    with Image.open(src) as im:
                        w, h = im.size
                except Exception as e:
                    raise RuntimeError(f"Cannot read image {src}: {e}") from e
                lines = []
                for obj in parsed[(modality, k)]:
                    if obj["difficult"] and not args.include_difficult:
                        continue
                    vals = []

                    # 变更 1：轻微越界允许修复；严重越界仍然报错，避免掩盖错误标注。
                    # 例如 840x712 图片，x=845 只越界 5px，可以裁剪到边界。
                    tol_x = max(5.0, w * 0.02)
                    tol_y = max(5.0, h * 0.02)
                    clipped_points = []
                    for x, y in obj["points"]:
                        if (x < -tol_x or x > w + tol_x
                                or y < -tol_y or y > h + tol_y):
                            raise ValueError(
                                f"Severely out-of-image point in {ix[label_key][k]}: "
                                f"{(x, y)} vs {w}x{h}"
                            )
                        cx = min(float(w), max(0.0, x))
                        cy = min(float(h), max(0.0, y))
                        if cx != x or cy != y:
                            audit["coordinate_repairs"].append({
                                "split": split,
                                "modality": modality,
                                "key": k,
                                "xml": str(ix[label_key][k]),
                                "original": [x, y],
                                "clipped": [cx, cy],
                            })
                        clipped_points.append((cx, cy))

                    # 裁剪后检查是否仍然是有效四边形，避免顶点重合形成退化 OBB。
                    if len(set(clipped_points)) < 3:
                        raise ValueError(
                            f"Polygon degenerated after clipping: "
                            f"{ix[label_key][k]} object={obj}"
                        )

                    for x, y in clipped_points:
                        vals.extend([x / w, y / h])

                    lines.append(str(class_to_id[obj["class"]]) + " " + " ".join(f"{v:.6f}" for v in vals))
                label_dst = out / "labels" / modality / split / (k + ".txt")
                label_dst.parent.mkdir(parents=True, exist_ok=True)
                label_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    for modality in ("rgb", "ir"):
        data_cfg = {
            "path": out.as_posix(),
            "train": f"images/{modality}/train",
            "val": f"images/{modality}/val",
            "test": f"images/{modality}/test",
            "names": {i: n for i, n in enumerate(names)},
        }
        yaml_text = yaml.safe_dump(data_cfg, sort_keys=False, allow_unicode=True)
        (out / f"{modality}.yaml").write_text(yaml_text, encoding="utf-8")
    (out / "class_mapping.json").write_text(json.dumps(class_to_id, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"\nPrepared data at: {out}")
    print(f"RGB YAML: {out / 'rgb.yaml'}")
    print(f"IR YAML:  {out / 'ir.yaml'}")
    print(f"Coordinate repairs recorded: {len(audit['coordinate_repairs'])}")
    print("Next: run the smoke test before full training.")

if __name__ == "__main__":
    main()
