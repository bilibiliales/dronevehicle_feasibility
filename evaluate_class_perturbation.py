
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A2: Per-class AP analysis for RGB / RGB-IR perturbation experiments.

Reuses evaluate_detection_perturbation.py:
- Same RGB labels
- Same paired RGB/IR images
- Same perturbation implementation
- Same Ultralytics validators
- No training

Supports resume from partial CSV.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASS_NAMES = {
    0: "bus",
    1: "car",
    2: "freight_car",
    3: "truck",
    4: "van",
}

TESTS = [
    ("rgb_baseline", "original", "rgb"),
    ("rgb_ir_fusion", "original", "fusion"),

    ("rgb_baseline", "rgb_dark", "rgb"),
    ("rgb_ir_fusion", "rgb_dark", "fusion"),

    ("rgb_baseline", "rgb_blur", "rgb"),
    ("rgb_ir_fusion", "rgb_blur", "fusion"),

    ("rgb_baseline", "rgb_noise", "rgb"),
    ("rgb_ir_fusion", "rgb_noise", "fusion"),

    ("rgb_ir_fusion", "ir_dark", "fusion"),
    ("rgb_ir_fusion", "ir_blur", "fusion"),
    ("rgb_ir_fusion", "ir_noise", "fusion"),
]


def extract_class_metrics(result, model_name, variant, image_count):
    """
    Extract per-class AP50 and AP50-95 from Ultralytics DetectionMetrics.

    Ultralytics 8.4.x:
      result.box.ap_class_index: (nc,)
      result.box.ap50:           (nc,)
      result.box.ap:             (nc,)
    """

    names = {
        0: "bus",
        1: "car",
        2: "freight_car",
        3: "truck",
        4: "van",
    }

    box = result.box

    class_indices = np.asarray(box.ap_class_index).reshape(-1)
    ap50_values = np.asarray(box.ap50).reshape(-1)
    ap_values = np.asarray(box.ap).reshape(-1)

    if not (
        len(class_indices) == len(ap50_values) == len(ap_values)
    ):
        raise RuntimeError(
            "Per-class metric length mismatch: "
            f"class_indices={class_indices.shape}, "
            f"ap50={ap50_values.shape}, "
            f"ap={ap_values.shape}"
        )

    rows = []

    # Only classes represented in the validation results are returned
    # by Ultralytics. Missing classes are recorded as NaN.
    metrics = {
        int(class_id): (float(ap50), float(ap))
        for class_id, ap50, ap in zip(
            class_indices, ap50_values, ap_values
        )
    }

    for class_id, class_name in names.items():
        ap50, ap = metrics.get(class_id, (float("nan"), float("nan")))

        rows.append({
            "model": model_name,
            "variant": variant,
            "class_id": class_id,
            "class_name": class_name,
            "images": image_count,
            "ap50": ap50,
            "ap50_95": ap,
        })

    return rows

def calculate_deltas(rows):

    baselines = {}

    for row in rows:
        if row["variant"] == "original":
            key = (row["model"], row["class_id"])
            baselines[key] = row

    for row in rows:

        baseline = baselines.get(
            (row["model"], row["class_id"])
        )

        if baseline is None:
            row["delta_ap50"] = float("nan")
            row["delta_ap50_95"] = float("nan")
            continue

        for metric in ("ap50", "ap50_95"):

            current = float(row[metric])
            original = float(baseline[metric])

            delta_key = "delta_" + metric

            if np.isfinite(current) and np.isfinite(original):
                row[delta_key] = current - original
            else:
                row[delta_key] = float("nan")

    return rows


def save_results(rows, output_dir, metadata):

    output_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "model",
        "variant",
        "class_id",
        "class_name",
        "images",
        "ap50",
        "ap50_95",
        "delta_ap50",
        "delta_ap50_95",
    ]

    csv_path = output_dir / "class_perturbation_summary.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "class_perturbation_summary.json"

    payload = dict(metadata)
    payload["results"] = rows

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    return csv_path, json_path


def draw_heatmap(rows, model_name, metric, output_dir):

    if model_name == "rgb_baseline":
        variants = [
            "original",
            "rgb_dark",
            "rgb_blur",
            "rgb_noise",
        ]
    else:
        variants = [
            "original",
            "rgb_dark",
            "rgb_blur",
            "rgb_noise",
            "ir_dark",
            "ir_blur",
            "ir_noise",
        ]

    matrix = np.full(
        (len(CLASS_NAMES), len(variants)),
        np.nan,
        dtype=float,
    )

    for row in rows:

        if row["model"] != model_name:
            continue

        if row["variant"] not in variants:
            continue

        value = row.get(metric)

        if value is not None and np.isfinite(float(value)):
            matrix[
                row["class_id"],
                variants.index(row["variant"])
            ] = float(value)

    fig, ax = plt.subplots(
        figsize=(max(8, len(variants) * 1.5), 5)
    )

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdYlGn",
        vmin=-0.6,
        vmax=0.1,
    )

    ax.set_xticks(
        range(len(variants)),
        variants,
        rotation=25,
        ha="right",
    )

    ax.set_yticks(
        range(len(CLASS_NAMES)),
        [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))],
    )

    ax.set_title(f"{model_name}: {metric} change vs original")
    ax.set_xlabel("Input condition")
    ax.set_ylabel("Object class")

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = matrix[r, c]

            if np.isfinite(value):
                ax.text(
                    c, r,
                    f"{value:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    fig.colorbar(im, ax=ax, label="AP change")
    fig.tight_layout()

    filename = f"{model_name}_{metric}_heatmap.png"

    fig.savefig(
        output_dir / filename,
        dpi=220,
    )

    plt.close(fig)


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-dir",
        default=r"D:\dronevehicle_feasibility\yolo_obb_data",
    )

    parser.add_argument("--rgb-weights", required=True)
    parser.add_argument("--fusion-weights", required=True)

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="100 for smoke test; 0 for all validation images",
    )

    parser.add_argument(
        "--output-dir",
        default="runs/class_perturbation",
    )

    parser.add_argument(
        "--only",
        choices=["all", "rgb", "fusion"],
        default="all",
    )

    args = parser.parse_args()

    import evaluate_detection_perturbation as evaluator

    from ultralytics import YOLO

    data_dir = Path(args.data_dir).resolve()
    yaml_path = data_dir / "rgb.yaml"

    rgb_val_dir = data_dir / "images" / "rgb" / "val"

    if not yaml_path.is_file():
        raise FileNotFoundError(yaml_path)

    rgb_paths = sorted(
        p for p in rgb_val_dir.iterdir()
        if p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
        }
    )

    if args.limit > 0:
        rgb_paths = rgb_paths[:args.limit]

    image_count = len(rgb_paths)

    if image_count == 0:
        raise RuntimeError("No validation images found.")

    # Validate all paired images before expensive inference.
    missing = []

    for rgb_path in rgb_paths:
        ir_path = evaluator.paired_path(rgb_path)

        if not ir_path.is_file():
            missing.append(rgb_path.name)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} missing IR pairs: {missing[:10]}"
        )

    # Reuse the original validator's global settings.
    evaluator.LIMIT = args.limit
    evaluator.IMAGE_COUNT = image_count

    # Restore custom model classes required by the fusion checkpoint.
    import __main__
    import train_yolo_obb_fusion

    __main__.ReliabilityOBBModel = (
        train_yolo_obb_fusion.ReliabilityOBBModel
    )

    __main__.ReliabilityGate = (
        train_yolo_obb_fusion.ReliabilityGate
    )

    rgb_model = None
    fusion_model = None

    if args.only in ("all", "rgb"):
        rgb_model = YOLO(args.rgb_weights, task="obb")

    if args.only in ("all", "fusion"):
        fusion_model = YOLO(args.fusion_weights, task="obb")

    common = {
        "data": str(yaml_path),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "split": "val",
        "plots": False,
        "save_json": False,
        "workers": 0,
        "rect": False,
        "fraction": 1.0,
        "project": str(Path(args.output_dir).resolve()),
        "exist_ok": True,
        "verbose": False,
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "class_perturbation_summary.csv"

    # Resume from previous completed runs.
    rows = []

    completed = set()

    if csv_path.is_file():

        with csv_path.open(
            "r", newline="", encoding="utf-8-sig"
        ) as f:

            reader = csv.DictReader(f)
            old_rows = list(reader)

        for row in old_rows:

            for key in (
                "class_id",
                "images",
                "ap50",
                "ap50_95",
                "delta_ap50",
                "delta_ap50_95",
            ):
                try:
                    row[key] = int(row[key]) if key in (
                        "class_id", "images"
                    ) else float(row[key])
                except (ValueError, TypeError):
                    row[key] = float("nan")

            rows.append(row)

        # Only consider a condition complete if all five class rows exist.
        for model_name, variant, _ in TESTS:

            subset = [
                r for r in rows
                if r["model"] == model_name
                and r["variant"] == variant
            ]

            if (
                len(subset) == 5
                and all(np.isfinite(float(r["ap50"])) for r in subset)
            ):
                completed.add((model_name, variant))

        print(f"Resume: {len(completed)} completed conditions found.")

    metadata = {
        "images_evaluated": image_count,
        "image_selection": (
            "all sorted RGB validation images"
            if args.limit == 0
            else f"first {args.limit} sorted RGB validation images"
        ),
        "class_names": CLASS_NAMES,
        "label_source": "RGB labels",
        "perturbations": {
            "dark": "pixel * 0.35",
            "blur": "GaussianBlur (9,9), sigma=2.0",
            "noise": "Gaussian noise std=25, seed=20261029+image_index",
        },
        "note": (
            "Deltas are computed within each model and class "
            "relative to its own original input condition."
        ),
    }

    for model_name, variant, model_kind in TESTS:

        if args.only == "rgb" and model_kind != "rgb":
            continue

        if args.only == "fusion" and model_kind != "fusion":
            continue

        if (model_name, variant) in completed:
            print(f"Skip: {model_name}/{variant}")
            continue

        evaluator.VARIANT = variant

        print(
            f"\nEvaluating {model_name}/{variant} "
            f"on {image_count} images..."
        )

        if model_kind == "rgb":

            result = rgb_model.val(
                **common,
                name=f"class_rgb_{variant}",
                validator=evaluator.PerturbValidator,
            )

        else:

            result = fusion_model.val(
                **common,
                name=f"class_fusion_{variant}",
                validator=evaluator.FusionPerturbValidator,
            )

        new_rows = extract_class_metrics(
            result,
            model_name,
            variant,
            image_count,
        )

        # Replace rows for this exact condition.
        rows = [
            r for r in rows
            if not (
                r["model"] == model_name
                and r["variant"] == variant
            )
        ]

        rows.extend(new_rows)

        rows = calculate_deltas(rows)

        # Save immediately after every validation condition.
        save_results(rows, output_dir, metadata)

        for row in new_rows:

            print(
                f"  {row['class_name']:12s} "
                f"AP50={row['ap50']:.4f} "
                f"AP50-95={row['ap50_95']:.4f}"
            )

        print("Saved partial results.")

    rows = calculate_deltas(rows)

    save_results(rows, output_dir, metadata)

    # Generate heatmaps only for requested model types.
    if args.only in ("all", "rgb"):

        draw_heatmap(
            rows,
            "rgb_baseline",
            "delta_ap50",
            output_dir,
        )

    if args.only in ("all", "fusion"):

        draw_heatmap(
            rows,
            "rgb_ir_fusion",
            "delta_ap50",
            output_dir,
        )

    print("\nA2 finished.")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
