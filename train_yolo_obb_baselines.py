#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train standard Ultralytics YOLO OBB single-modality baselines (RGB and/or IR)."""
import argparse, json, platform, sys
from pathlib import Path
from datetime import datetime

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=r"D:\dronevehicle_feasibility\yolo_obb_data")
    p.add_argument("--model", default="yolo11n-obb.pt",
                   help="Official Ultralytics OBB checkpoint, e.g. yolo11n-obb.pt")
    p.add_argument("--modality", choices=["rgb", "ir", "both"], default="both")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="0", help="CUDA device index; use cpu to force CPU")
    p.add_argument("--seed", type=int, default=20261025)
    p.add_argument("--project", default="runs/dronevehicle_baselines")
    p.add_argument("--name", default="baseline")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--smoke", action="store_true",
                   help="Run a tiny pipeline check: 1 epoch, train/val limited to 32 images")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Missing dependencies. Install with: pip install ultralytics pillow") from e

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}. Run prepare_dronevehicle_obb.py first.")
    modalities = ["rgb", "ir"] if args.modality == "both" else [args.modality]
    for m in modalities:
        if not (data_dir / f"{m}.yaml").is_file():
            raise SystemExit(f"Missing {data_dir / (m+'.yaml')}")

    info = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "args": vars(args),
    }
    print(json.dumps(info, indent=2, ensure_ascii=False))
    run_root = Path(args.project)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "environment.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    for modality in modalities:
        model = YOLO(args.model)
        epochs = 1 if args.smoke else args.epochs
        kwargs = dict(
            data=str(data_dir / f"{modality}.yaml"),
            task="obb",
            epochs=epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=0 if platform.system() == "Windows" else args.workers,
            device=args.device,
            seed=args.seed,
            deterministic=True,
            pretrained=True,
            amp=True,
            cache=False,
            patience=0 if args.smoke else args.patience,
            project=str(run_root),
            name=f"{args.name}_{modality}" + ("_smoke" if args.smoke else ""),
            exist_ok=False,
            plots=True,
            save=True,
            val=True,
            verbose=True,
        )
        if args.smoke:
            kwargs["fraction"] = 0.01
        if args.resume:
            kwargs["resume"] = True
        print(f"\n=== Training {modality.upper()} baseline ===")
        results = model.train(**kwargs)
        run_dir = Path(results.save_dir)
        # Explicit metadata alongside Ultralytics' args/results.
        metadata = {
            "modality": modality, "model": args.model,
            "data_yaml": str(data_dir / f"{modality}.yaml"),
            "run_dir": str(run_dir), "args": kwargs,
        }
        (run_dir / "baseline_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"Finished {modality}. Run artifacts: {run_dir}")
        print("Best checkpoint:", run_dir / "weights" / "best.pt")
        print("Metrics/logs are in the run directory (results.csv, results.png, args.yaml).")

if __name__ == "__main__":
    main()
