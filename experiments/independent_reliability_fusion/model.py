"""Independent dual-stream RGB/IR reliability fusion model.

This file intentionally does not import anything from the repository's drone/ tree.
It builds two Ultralytics OBB graphs and combines backbone features with learned reliability
weights, then runs the RGB model's neck/head over the fused feature maps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from ultralytics.nn.tasks import OBBModel


class ReliabilityGate(nn.Module):
    """Spatially adaptive, two-modality reliability gate."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.score = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1, bias=True),
        )

    def forward(self, rgb: Tensor, ir: Tensor) -> tuple[Tensor, Tensor]:
        if rgb.shape != ir.shape:
            raise ValueError(f"paired feature shapes differ: RGB={rgb.shape}, IR={ir.shape}")
        weights = self.score(torch.cat((rgb, ir), dim=1)).softmax(dim=1)
        rgb_weight, ir_weight = weights[:, 0:1], weights[:, 1:2]
        return rgb_weight * rgb + ir_weight * ir, weights


class IndependentReliabilityOBB(nn.Module):
    """Dual YOLO11-OBB streams with reliability-weighted backbone fusion.

    Fusion locations default to layers 4, 6, and 10 in the standard YOLO11-OBB graph.
    The fused features are used by the RGB neck/head; the IR neck/head is not duplicated.
    """

    def __init__(
        self,
        model_cfg: str | Path,
        nc: int = 5,
        fusion_layers: tuple[int, ...] = (4, 6, 10),
    ) -> None:
        super().__init__()
        self.rgb = OBBModel(str(model_cfg), ch=3, nc=nc, verbose=False)
        self.ir = OBBModel(str(model_cfg), ch=3, nc=nc, verbose=False)
        self.fusion_layers = tuple(fusion_layers)
        self.gates = nn.ModuleDict()
        self._validate_graph()
        for idx in self.fusion_layers:
            channels = self._infer_channels(idx)
            self.gates[str(idx)] = ReliabilityGate(channels)

    def _validate_graph(self) -> None:
        if len(self.rgb.model) != len(self.ir.model):
            raise ValueError("RGB and IR model graphs differ in length")
        if not self.fusion_layers or max(self.fusion_layers) >= len(self.rgb.model):
            raise ValueError(f"invalid fusion layers {self.fusion_layers}")

    @torch.no_grad()
    def _infer_channels(self, idx: int) -> int:
        parameter = next(self.rgb.parameters())
        x: Any = torch.zeros(1, 3, 320, 320, device=parameter.device, dtype=parameter.dtype)
        saved: list[Any] = []
        for layer in self.rgb.model:
            if layer.f != -1:
                x_in = saved[layer.f] if isinstance(layer.f, int) else [x if j == -1 else saved[j] for j in layer.f]
            else:
                x_in = x
            x = layer(x_in)
            saved.append(x if layer.i in self.rgb.save else None)
            if layer.i == idx:
                if not isinstance(x, Tensor) or x.ndim != 4:
                    raise TypeError(f"layer {idx} did not produce a 4D feature map")
                return int(x.shape[1])
        raise RuntimeError(f"could not infer channels for layer {idx}")

    @staticmethod
    def _run_backbone(model: OBBModel, image: Tensor, stop: int) -> dict[int, Tensor]:
        x: Any = image
        saved: list[Any] = []
        features: dict[int, Tensor] = {}
        for layer in model.model:
            if layer.i > stop:
                break
            if layer.f != -1:
                x_in = saved[layer.f] if isinstance(layer.f, int) else [x if j == -1 else saved[j] for j in layer.f]
            else:
                x_in = x
            x = layer(x_in)
            saved.append(x if layer.i in model.save else None)
            if isinstance(x, Tensor) and x.ndim == 4:
                features[layer.i] = x
        return features

    def forward(self, rgb_image: Tensor, ir_image: Tensor) -> Any:
        if rgb_image.shape != ir_image.shape:
            raise ValueError(f"RGB/IR input shapes differ: {rgb_image.shape} vs {ir_image.shape}")
        stop = max(self.fusion_layers)
        rgb_features = self._run_backbone(self.rgb, rgb_image, stop)
        ir_features = self._run_backbone(self.ir, ir_image, stop)

        x: Any = rgb_image
        saved: list[Any] = []
        for layer in self.rgb.model:
            if layer.i > stop:
                break
            if layer.f != -1:
                x_in = saved[layer.f] if isinstance(layer.f, int) else [x if j == -1 else saved[j] for j in layer.f]
            else:
                x_in = x
            x = layer(x_in)
            if layer.i in self.fusion_layers:
                x, _ = self.gates[str(layer.i)](rgb_features[layer.i], ir_features[layer.i])
            saved.append(x if layer.i in self.rgb.save else None)

        for layer in self.rgb.model:
            if layer.i <= stop:
                continue
            if layer.f != -1:
                x_in = saved[layer.f] if isinstance(layer.f, int) else [x if j == -1 else saved[j] for j in layer.f]
            else:
                x_in = x
            x = layer(x_in)
            saved.append(x if layer.i in self.rgb.save else None)
        return x

    def load_pretrained_branches(self, rgb_checkpoint: str | Path, ir_checkpoint: str | Path) -> None:
        """Load trusted local Ultralytics checkpoints into their corresponding streams."""
        self._load_one(self.rgb, rgb_checkpoint, "RGB")
        self._load_one(self.ir, ir_checkpoint, "IR")

    @staticmethod
    def _load_one(model: OBBModel, checkpoint_path: str | Path, label: str) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {path}")
        # Ultralytics checkpoints contain serialized model objects. Only load trusted local files.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        source = (ckpt.get("ema") or ckpt.get("model")) if isinstance(ckpt, dict) else ckpt
        if hasattr(source, "state_dict"):
            source_state = source.float().state_dict()
        elif isinstance(source, dict):
            source_state = source
        else:
            raise TypeError(f"unsupported checkpoint format for {label}: {type(source)}")
        target_state = model.state_dict()
        compatible = {
            key: value for key, value in source_state.items()
            if key in target_state and isinstance(value, Tensor) and target_state[key].shape == value.shape
        }
        if not compatible:
            raise RuntimeError(f"no compatible weights found in {label} checkpoint: {path}")
        incompatible = model.load_state_dict(compatible, strict=False)
        print(f"[{label}] loaded {len(compatible)} tensors; missing={len(incompatible.missing_keys)}")
