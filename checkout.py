#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scan DroneVehicle XML labels and report objects missing <polygon>.
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\dronevehicle_feasibility\dataset")

LABEL_DIRS = [
    ROOT / "train" / "trainlabel",
    ROOT / "train" / "trainlabelr",
    ROOT / "val" / "vallabel",
    ROOT / "val" / "vallabelr",
    ROOT / "test" / "testlabel",
    ROOT / "test" / "testlabelr",
]

def scan_xml(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    total = 0
    with_polygon = 0
    with_bndbox = 0
    neither = 0
    for obj in root.findall("object"):
        total += 1
        has_poly = obj.find("polygon") is not None
        has_bnd = obj.find("bndbox") is not None
        if has_poly:
            with_polygon += 1
        if has_bnd:
            with_bndbox += 1
        if not has_poly and not has_bnd:
            neither += 1
    return total, with_polygon, with_bndbox, neither

def main():
    stats = {
        "total_xml": 0,
        "total_objects": 0,
        "objects_with_polygon": 0,
        "objects_with_bndbox_only": 0,
        "objects_neither": 0,
        "files_with_bndbox_only_objects": [],
        "files_with_neither_objects": [],
        "empty_xml": [],
    }

    for d in LABEL_DIRS:
        if not d.is_dir():
            print(f"[WARN] label dir not found: {d}")
            continue

        for xml_path in sorted(d.glob("*.xml")):
            stats["total_xml"] += 1
            total, poly, bnd, nei = scan_xml(xml_path)
            stats["total_objects"] += total
            stats["objects_with_polygon"] += poly
            stats["objects_with_bndbox_only"] += bnd
            stats["objects_neither"] += nei

            if total == 0:
                stats["empty_xml"].append(str(xml_path))

            if bnd > 0 and poly == 0:
                stats["files_with_bndbox_only_objects"].append(str(xml_path))
            elif bnd > 0 and poly > 0:
                stats["files_with_bndbox_only_objects"].append(str(xml_path))

            if nei > 0:
                stats["files_with_neither_objects"].append(str(xml_path))

    print("=== DroneVehicle XML annotation scan ===")
    print(f"Total XML files: {stats['total_xml']}")
    print(f"Total objects: {stats['total_objects']}")
    print(f"Objects with <polygon>: {stats['objects_with_polygon']}")
    print(f"Objects with only <bndbox>: {stats['objects_with_bndbox_only']}")
    print(f"Objects with neither: {stats['objects_neither']}")

    if stats["objects_with_bndbox_only"]:
        ratio = stats["objects_with_bndbox_only"] / stats["total_objects"] * 100
        print(f"\nOnly-bndbox objects ratio: {ratio:.2f}%")

    if stats["files_with_bndbox_only_objects"]:
        print("\nXML files containing <bndbox>-only objects:")
        for p in stats["files_with_bndbox_only_objects"][:50]:
            print(" ", p)
        if len(stats["files_with_bndbox_only_objects"]) > 50:
            print(f"  ... and {len(stats['files_with_bndbox_only_objects']) - 50} more")

    if stats["empty_xml"]:
        print("\nEmpty XML files (no objects):")
        for p in stats["empty_xml"][:20]:
            print(" ", p)

    if stats["files_with_neither_objects"]:
        print("\nXML files with objects missing both polygon and bndbox:")
        for p in stats["files_with_neither_objects"][:20]:
            print(" ", p)

    # 生成删除建议：只包含 bndbox、没有 polygon 的 XML
    delete_candidates = stats["files_with_bndbox_only_objects"]
    if delete_candidates:
        del_list = ROOT / "bndbox_only_files.txt"
        del_list.write_text("\n".join(delete_candidates) + "\n", encoding="utf-8")
        print(f"\nDelete candidates written to: {del_list}")
        print(f"Count: {len(delete_candidates)}")
    else:
        print("\nNo bndbox-only XML files found.")

if __name__ == "__main__":
    main()