#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Native Ultralytics OBB training with paired RGB/IR images and a learnable
reliability gate at the input.

Important: this is INPUT-LEVEL reliability fusion (6-channel paired input ->
2-way pixel-wise softmax gate -> 3-channel image -> native YOLO11-OBB).
It is not a middle-feature fusion implementation.

Expected data layout created by prepare_dronevehicle_obb.py:
  DATA/images/rgb/{train,val,test}/<stem>.<ext>
  DATA/images/ir/{train,val,test}/<same stem>.<ext>
  DATA/labels/rgb/{train,val,test}/<stem>.txt
RGB labels are used as the single supervision source.

Tested status: source implementation only; run --smoke before any long run.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import ultralytics
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils.torch_utils import fuse_conv_and_bn


class PairedYOLODataset(YOLODataset):
    """Use RGB image paths/labels, but load the matching IR image as channels 4-6."""

    def load_image(self, i, rect_mode=True):
        rgb, (h0, w0), (h, w) = super().load_image(i, rect_mode)
        rgb_path = Path(self.im_files[i])
        # Prepared dataset stores the modality in the path component.
        parts = list(rgb_path.parts)
        try:
            modality_index = parts.index("rgb")
        except ValueError as exc:
            raise RuntimeError(
                f"Expected RGB path containing a 'rgb' component, got: {rgb_path}"
            ) from exc
        parts[modality_index] = "ir"
        ir_path = Path(*parts)
        if not ir_path.is_file():
            raise FileNotFoundError(f"Missing paired IR image for {rgb_path}: {ir_path}")

        ir = cv2.imread(str(ir_path), cv2.IMREAD_COLOR)
        if ir is None:
            raise RuntimeError(f"OpenCV could not read IR image: {ir_path}")
        if ir.shape[:2] != (h0, w0):
            # Pair registration is assumed; resizing only resolves different stored sizes.
            ir = cv2.resize(ir, (w0, h0), interpolation=cv2.INTER_LINEAR)
        # Ultralytics reads BGR; both modalities use the same convention.
        if ir.shape[:2] != rgb.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        paired = np.concatenate((rgb, ir), axis=2)
        return paired, (h0, w0), (h, w)


class ReliabilityGate(nn.Module):
    """Predict per-pixel RGB/IR reliability weights that sum to one."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1, bias=True),
        )
        # Start close to equal fusion, avoiding a strong random modality preference.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.net(x).softmax(dim=1)
        rgb, ir = x[:, :3], x[:, 3:6]
        fused = weights[:, 0:1] * rgb + weights[:, 1:2] * ir
        return fused, weights


class ReliabilityOBBModel(OBBModel):
    """OBBModel-compatible wrapper; native OBB loss and prediction heads are retained."""

    def __init__(self, cfg="yolo11n-obb.yaml", ch=3, nc=None, verbose=True):
        # Parent model remains a standard 3-channel YOLO OBB model.
        super().__init__(cfg=cfg, ch=3, nc=nc, verbose=verbose)
        self.reliability_gate = ReliabilityGate()

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            # Base OBBModel uses dict input to calculate its native criterion/loss.
            return self.loss(x, *args, **kwargs)
        if not torch.is_tensor(x) or x.ndim != 4:
            return super().forward(x, *args, **kwargs)
        if x.shape[1] == 6:
            x, _weights = self.reliability_gate(x)
        elif x.shape[1] != 3:
            raise ValueError(f"Expected 6-channel paired input or 3-channel image, got {tuple(x.shape)}")
        return super().forward(x, *args, **kwargs)


class PairedOBBTrainer(OBBTrainer):
    """Keep Ultralytics' native training loop, loss, optimizer, AMP and validation."""

    def build_dataset(self, img_path, mode="train", batch=None):
        stride = max(int(self.model.stride.max()), 32) if self.model is not None else 32
        return PairedYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=mode == "val",
            cache=False,
            single_cls=self.args.single_cls,
            stride=stride,
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            data=self.data,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = ReliabilityOBBModel(
            cfg=cfg or self.args.model,
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model


# Import after class definition so this file remains easy to inspect.
try:
    from ultralytics.utils import RANK
except Exception:
    RANK = -1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=r"D:\dronevehicle_feasibility\yolo_obb_data")
    parser.add_argument("--model", default="yolo11n-obb.yaml",
                        help="Ultralytics OBB model YAML; pretrained weights are optional")
    parser.add_argument("--pretrained", default="yolo11n-obb.pt",
                        help="Native pretrained OBB checkpoint; use '' to disable")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--project", default="runs/dronevehicle_fusion")
    parser.add_argument("--name", default="reliability_gate")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--smoke", action="store_true",
                        help="1 epoch, small image size and fraction for pipeline debugging")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_yaml = data_dir / "rgb.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Missing dataset YAML: {data_yaml}")
    for split in ("train", "val"):
        rgb_dir = data_dir / "images" / "rgb" / split
        ir_dir = data_dir / "images" / "ir" / split
        if not rgb_dir.is_dir() or not ir_dir.is_dir():
            raise FileNotFoundError(f"Expected paired folders: {rgb_dir} and {ir_dir}")

    if args.smoke:
        args.epochs = 1
        args.imgsz = min(args.imgsz, 320)
        args.batch = 1

    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "args": vars(args),
        "fusion": "input-level learned pixel-wise reliability gate",
        "supervision": "RGB labels",
    }
    print(json.dumps(info, indent=2, ensure_ascii=False, default=str))

    model_yaml = args.model
    # OBBTrainer resolves cfg and pretrained weights using native Ultralytics arguments.
    overrides = {
        "model": model_yaml,
        "data": str(data_yaml),
        "task": "obb",
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": 0 if platform.system() == "Windows" else args.workers,
        "device": args.device,
        "seed": args.seed,
        "deterministic": True,
        "pretrained": args.pretrained if args.pretrained else False,
        "amp": True,
        "cache": False,
        "patience": 0 if args.smoke else args.patience,
        "project": str(Path(args.project).resolve()),
        "name": args.name + ("_smoke" if args.smoke else ""),
        "exist_ok": False,
        "plots": True,
        "save": True,
        "val": True,
        "verbose": True,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "fraction": 0.01 if args.smoke else 1.0,
    }
    trainer = PairedOBBTrainer(overrides=overrides)
    trainer.train()
    run_dir = Path(trainer.save_dir)
    (run_dir / "fusion_metadata.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"Finished. Run directory: {run_dir}")
    print(f"Best checkpoint: {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
