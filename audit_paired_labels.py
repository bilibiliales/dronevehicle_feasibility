
import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np


def read_labels(path):
    result = []
    if not path.exists():
        return result

    for line_no, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 9:
            raise ValueError(
                f"{path}:{line_no}: expected 9 fields, got {len(fields)}"
            )

        cls = int(float(fields[0]))
        points = np.asarray(
            [float(x) for x in fields[1:]], dtype=float
        ).reshape(4, 2)
        result.append((cls, points))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=r"D:\dronevehicle_feasibility\yolo_obb_data",
    )
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument(
        "--out",
        default=r"D:\dronevehicle_feasibility\log\paired_label_audit.json",
    )
    parser.add_argument(
        "--coord-tol",
        type=float,
        default=0.002,
        help="Tolerance in normalized coordinates",
    )
    args = parser.parse_args()

    root = Path(args.data_dir)
    rgb_dir = root / "labels" / "rgb" / args.split
    ir_dir = root / "labels" / "ir" / args.split

    if not rgb_dir.is_dir() or not ir_dir.is_dir():
        raise FileNotFoundError(f"Missing label directory: {rgb_dir} or {ir_dir}")

    rgb = {p.stem: p for p in rgb_dir.glob("*.txt")}
    ir = {p.stem: p for p in ir_dir.glob("*.txt")}
    keys = sorted(set(rgb) | set(ir))

    report = {
        "split": args.split,
        "rgb_files": len(rgb),
        "ir_files": len(ir),
        "rgb_only_keys": sorted(set(rgb) - set(ir)),
        "ir_only_keys": sorted(set(ir) - set(rgb)),
        "count_mismatches": [],
        "class_count_mismatches": [],
        "geometry_mismatches": [],
        "rgb_class_counts": Counter(),
        "ir_class_counts": Counter(),
        "rgb_instances": 0,
        "ir_instances": 0,
        "examples": [],
    }

    for key in keys:
        r = read_labels(rgb[key]) if key in rgb else []
        t = read_labels(ir[key]) if key in ir else []

        report["rgb_instances"] += len(r)
        report["ir_instances"] += len(t)
        report["rgb_class_counts"].update(x[0] for x in r)
        report["ir_class_counts"].update(x[0] for x in t)

        rc = Counter(x[0] for x in r)
        tc = Counter(x[0] for x in t)
        mismatches = []

        if len(r) != len(t):
            report["count_mismatches"].append({
                "key": key, "rgb": len(r), "ir": len(t)
            })
            mismatches.append("count")

        if rc != tc:
            report["class_count_mismatches"].append({
                "key": key, "rgb": dict(rc), "ir": dict(tc)
            })
            mismatches.append("class_count")

        # Greedy one-to-one matching by same class and coordinate distance.
        used = set()
        matches = 0

        for cls, poly in r:
            candidates = []
            for j, (other_cls, other_poly) in enumerate(t):
                if j in used or other_cls != cls:
                    continue
                distance = float(np.mean(np.abs(poly - other_poly)))
                candidates.append((distance, j))

            if candidates:
                distance, j = min(candidates)
                if distance <= args.coord_tol:
                    used.add(j)
                    matches += 1

        if matches != len(r) or matches != len(t):
            report["geometry_mismatches"].append({
                "key": key,
                "rgb_instances": len(r),
                "ir_instances": len(t),
                "matched_same_class_geometry": matches,
            })
            mismatches.append("geometry")

        if mismatches and len(report["examples"]) < 30:
            report["examples"].append({
                "key": key,
                "issues": mismatches,
                "rgb_count": len(r),
                "ir_count": len(t),
                "rgb_classes": dict(rc),
                "ir_classes": dict(tc),
                "geometry_matches": matches,
            })

    report["rgb_class_counts"] = dict(report["rgb_class_counts"])
    report["ir_class_counts"] = dict(report["ir_class_counts"])

    report["summary"] = {
        "union_keys": len(keys),
        "count_mismatch_images": len(report["count_mismatches"]),
        "class_count_mismatch_images": len(report["class_count_mismatches"]),
        "geometry_mismatch_images": len(report["geometry_mismatches"]),
        "instance_delta_ir_minus_rgb":
            report["ir_instances"] - report["rgb_instances"],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("RGB class counts:", report["rgb_class_counts"])
    print("IR class counts:", report["ir_class_counts"])
    print("Mismatch examples:")
    print(json.dumps(report["examples"][:10], ensure_ascii=False, indent=2))
    print("Full report:", out)


if __name__ == "__main__":
    main()
