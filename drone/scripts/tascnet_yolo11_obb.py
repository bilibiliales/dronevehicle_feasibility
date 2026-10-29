import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def _sanitize_omp_num_threads():
    val = os.environ.get("OMP_NUM_THREADS")
    if val is None:
        return
    try:
        iv = int(str(val).strip())
        if iv <= 0:
            raise ValueError
    except Exception:
        os.environ["OMP_NUM_THREADS"] = "1"
        print(f"[WARN] Invalid OMP_NUM_THREADS={val!r}, reset to 1")


_sanitize_omp_num_threads()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from ultralytics.nn.tasks import OBBModel
from ultralytics.utils.loss import v8OBBLoss
from ultralytics.utils.metrics import batch_probiou

try:
    from ultralytics.utils import nms, ops
    non_max_suppression = nms.non_max_suppression
except Exception:
    from ultralytics.utils import ops
    from ultralytics.utils.nms import non_max_suppression


def resolve_existing_path(path_like, desc="path", required=True):
    if path_like in (None, ""):
        if required:
            raise ValueError(f"Missing required {desc}.")
        return None

    p = Path(path_like).expanduser().resolve()
    if not p.exists():
        if required:
            raise FileNotFoundError(f"{desc} does not exist: {p}")
        print(f"[WARN] Optional {desc} does not exist, ignore it: {p}")
        return None
    return str(p)



def load_yaml_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"YAML config must parse to a dict, got: {type(cfg)}")
    return cfg



def first_present(cfg, *keys, default=None):
    for key in keys:
        if key in cfg and cfg[key] not in (None, ""):
            return cfg[key]
    return default



def resolve_runtime_paths(args, cfg):
    model_cfg = resolve_existing_path(
        args.model_cfg or first_present(cfg, "model_cfg", "model", "model_yaml"),
        desc="model cfg",
        required=True,
    )
    rgb_pretrained = resolve_existing_path(
        args.rgb_pretrained or first_present(cfg, "rgb_pretrained", "rgb_weights", "pretrained", "weights"),
        desc="RGB pretrained checkpoint",
        required=False,
    )
    ir_pretrained = resolve_existing_path(
        args.ir_pretrained or first_present(cfg, "ir_pretrained", "ir_weights"),
        desc="IR pretrained checkpoint",
        required=False,
    )

    copy_rgb_to_ir_cfg = first_present(cfg, "copy_rgb_to_ir", default=True)
    copy_rgb_to_ir = copy_rgb_to_ir_cfg if args.copy_rgb_to_ir is None else args.copy_rgb_to_ir
    copy_rgb_to_ir = bool(copy_rgb_to_ir)

    return {
        "model_cfg": model_cfg,
        "rgb_pretrained": rgb_pretrained,
        "ir_pretrained": ir_pretrained,
        "copy_rgb_to_ir": copy_rgb_to_ir,
    }



def append_metrics_csv(csv_path, row_dict):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)



def save_train_config(save_dir, train_cfg):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(train_cfg, f, indent=2, ensure_ascii=False)


# =========================
# 1. Dataset
# =========================
class RGBIRDataset(Dataset):

    def __init__(self, cfg_path, split="train", imgsz=640, strict_label_pair=True):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg
        self.root = Path(cfg.get("path", ".")).expanduser()
        self.imgsz = int(imgsz)
        self.split = split
        self.strict_label_pair = strict_label_pair
        self._warned_label_mismatch = False

        rgb_dir = self._resolve_cfg_dir(cfg, f"{split}_rgb")
        ir_dir = self._resolve_cfg_dir(cfg, f"{split}_ir")
        rgb_label_dir = self._resolve_cfg_dir(
            cfg,
            f"{split}_rgb_labels",
            fallback_keys=[f"{split}_labels", f"{split}_label", f"{split}_label_rgb"],
        )
        ir_label_dir = self._resolve_cfg_dir(
            cfg,
            f"{split}_ir_labels",
            fallback_keys=[f"{split}_labelsr", f"{split}_labels_r", f"{split}_label_ir"],
            required=False,
        )

        rgb_files = {p.stem: p for p in rgb_dir.iterdir() if p.is_file()}
        ir_files = {p.stem: p for p in ir_dir.iterdir() if p.is_file()}
        rgb_label_files = {p.stem: p for p in rgb_label_dir.glob("*.txt")}
        ir_label_files = {p.stem: p for p in ir_label_dir.glob("*.txt")} if ir_label_dir is not None else {}

        stems = set(rgb_files.keys()) & set(ir_files.keys()) & set(rgb_label_files.keys())
        if self.strict_label_pair and ir_label_dir is not None:
            stems &= set(ir_label_files.keys())

        self.samples = []
        for s in sorted(stems):
            self.samples.append(
                (
                    rgb_files[s],
                    ir_files[s],
                    rgb_label_files[s],
                    ir_label_files.get(s),
                    s,
                )
            )

        self.nc = int(cfg["nc"])
        self.names = cfg.get("names", [f"class_{i}" for i in range(self.nc)])
        print(f"[{split}] paired samples: {len(self.samples)}")
        print(f"[{split}] rgb dir      : {rgb_dir}")
        print(f"[{split}] ir dir       : {ir_dir}")
        print(f"[{split}] rgb labels   : {rgb_label_dir}")
        print(f"[{split}] ir labels    : {ir_label_dir if ir_label_dir is not None else 'None'}")

    @staticmethod
    def _resolve_cfg_dir(cfg, primary_key, fallback_keys=None, required=True):
        fallback_keys = fallback_keys or []
        all_keys = [primary_key, *fallback_keys]
        val = None
        used_key = None
        for key in all_keys:
            if key in cfg and cfg[key] not in (None, ""):
                val = cfg[key]
                used_key = key
                break

        if val is None:
            if required:
                raise KeyError(f"Missing dataset config key: one of {all_keys}")
            return None

        p = Path(str(val)).expanduser()
        if not p.is_absolute():
            root = Path(cfg.get("path", ".")).expanduser()
            p = root / p
        p = p.resolve()
        if required and not p.exists():
            raise FileNotFoundError(f"Config key '{used_key}' points to a non-existing directory: {p}")
        if not required and not p.exists():
            print(f"[WARN] Optional directory for '{used_key}' does not exist, ignore it: {p}")
            return None
        return p

    def __len__(self):
        return len(self.samples)

    def _read_img(self, path):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"cannot read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        return img

    def _read_label_obb8(self, path):
        if path is None or not Path(path).exists():
            return torch.zeros((0, 9), dtype=torch.float32)

        targets = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 9:
                    continue
                vals = list(map(float, parts))
                cls_id = vals[0]
                pts = np.array(vals[1:], dtype=np.float32).reshape(4, 2)
                pts = np.clip(pts, 0.0, 1.0)
                targets.append([cls_id, *pts.reshape(-1).tolist()])

        if len(targets) == 0:
            return torch.zeros((0, 9), dtype=torch.float32)
        return torch.tensor(targets, dtype=torch.float32)

    @staticmethod
    def _labels_close(a, b, atol=1e-6):
        if a.shape != b.shape:
            return False
        if a.numel() == 0 and b.numel() == 0:
            return True
        return torch.allclose(a, b, atol=atol, rtol=0.0)

    def _select_targets(self, rgb_labels, ir_labels, stem):
        if ir_labels is None:
            return rgb_labels

        if rgb_labels.numel() == 0 and ir_labels.numel() > 0:
            return ir_labels
        if ir_labels.numel() == 0 and rgb_labels.numel() > 0:
            return rgb_labels

        if not self._labels_close(rgb_labels, ir_labels) and not self._warned_label_mismatch:
            print(
                f"[WARN] Label mismatch detected in split='{self.split}', sample='{stem}'. "
                f"Training will use RGB labels as supervision."
            )
            self._warned_label_mismatch = True

        return rgb_labels

    @staticmethod
    def _obb8_to_xywhr(labels_obb8):
        if labels_obb8.numel() == 0:
            return torch.zeros((0, 6), dtype=torch.float32)

        cls = labels_obb8[:, 0:1]
        corners = labels_obb8[:, 1:9].contiguous()

        rboxes = ops.xyxyxyxy2xywhr(corners)
        rboxes = rboxes.to(dtype=torch.float32)

        rboxes[:, 0:4] = rboxes[:, 0:4].clamp(min=0.0, max=1.0)
        valid = (rboxes[:, 2] > 1e-6) & (rboxes[:, 3] > 1e-6)
        if not valid.all():
            cls = cls[valid]
            rboxes = rboxes[valid]

        return torch.cat([cls, rboxes], dim=1)

    def __getitem__(self, idx):
        rgb_path, ir_path, rgb_label_path, ir_label_path, stem = self.samples[idx]
        rgb = self._read_img(rgb_path)
        ir = self._read_img(ir_path)

        rgb_targets_obb8 = self._read_label_obb8(rgb_label_path)
        ir_targets_obb8 = self._read_label_obb8(ir_label_path) if ir_label_path is not None else None
        targets_obb8 = self._select_targets(rgb_targets_obb8, ir_targets_obb8, stem)
        targets_xywhr = self._obb8_to_xywhr(targets_obb8)

        return {
            "rgb": rgb,
            "ir": ir,
            "targets": targets_xywhr,
            "stem": stem,
        }

def collate_fn(batch):
    rgb = torch.stack([b["rgb"] for b in batch], dim=0)
    ir = torch.stack([b["ir"] for b in batch], dim=0)

    all_batch_idx = []
    all_cls = []
    all_bboxes = []

    for i, b in enumerate(batch):
        t = b["targets"]
        if len(t) > 0:
            n = t.shape[0]
            all_batch_idx.append(torch.full((n, 1), i, dtype=torch.float32))
            all_cls.append(t[:, 0:1])
            all_bboxes.append(t[:, 1:6])

    if len(all_cls) > 0:
        batch_idx = torch.cat(all_batch_idx, dim=0)
        cls = torch.cat(all_cls, dim=0)
        bboxes = torch.cat(all_bboxes, dim=0)
    else:
        batch_idx = torch.zeros((0, 1), dtype=torch.float32)
        cls = torch.zeros((0, 1), dtype=torch.float32)
        bboxes = torch.zeros((0, 5), dtype=torch.float32)

    return {
        "img": rgb,
        "rgb": rgb,
        "ir": ir,
        "batch_idx": batch_idx,
        "cls": cls,
        "bboxes": bboxes,
        "stems": [b["stem"] for b in batch],
    }



# =========================
# 2. Fusion / refinement / reliability modules
# =========================
class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualRefineBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, channels, 3),
            ConvBNAct(channels, channels, 3, act=False),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class SimpleAddFusion(nn.Module):
    def forward(self, rgb_feat, ir_feat):
        return 0.5 * (rgb_feat + ir_feat)


class TaRFusion(nn.Module):
    """Target-aware Reliability Fusion."""
    def __init__(self, c_rgb, c_ir, c_out=None, min_hidden=16):
        super().__init__()
        c_out = int(c_out or max(c_rgb, c_ir))
        hidden = max(c_out // 4, min_hidden)
        self.rgb_proj = ConvBNAct(c_rgb, c_out, 1)
        self.ir_proj = ConvBNAct(c_ir, c_out, 1)
        self.target_est = nn.Sequential(
            ConvBNAct(2 * c_out, hidden, 1),
            nn.Conv2d(hidden, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.reliability = nn.Sequential(
            ConvBNAct(2 * c_out, hidden, 1),
            nn.Conv2d(hidden, 2, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.cross_modal = nn.Sequential(
            ConvBNAct(2 * c_out, c_out, 1),
            ConvBNAct(c_out, c_out, 3),
        )

    def forward(self, rgb_feat, ir_feat):
        rgb = self.rgb_proj(rgb_feat)
        ir = self.ir_proj(ir_feat)
        joint = torch.cat([rgb, ir], dim=1)
        target_map = self.target_est(joint)

        rgb_focus = rgb * target_map
        ir_focus = ir * target_map

        rel_logits = self.reliability(torch.cat([rgb_focus, ir_focus], dim=1))
        rel_weights = torch.softmax(rel_logits, dim=1)
        w_rgb = rel_weights[:, 0:1]
        w_ir = rel_weights[:, 1:2]

        cross = self.cross_modal(torch.cat([rgb_focus, ir_focus], dim=1))
        fused = w_rgb * rgb_focus + w_ir * ir_focus + cross
        aux = {
            "target_map": target_map,
            "w_rgb": w_rgb,
            "w_ir": w_ir,
            "rgb_proj": rgb,
            "ir_proj": ir,
            "rgb_focus": rgb_focus,
            "ir_focus": ir_focus,
            "cross": cross,
        }
        return fused, aux


class S2RHHead(nn.Module):
    """Sparse-guided Super-resolution Refinement Head.

    Compatible implementation:
    - keeps the original 3-scale OBB head;
    - injects local high-resolution refinement into the highest-resolution detect feature.
    """
    def __init__(self, f2_ch, n3_ch, n4_ch, n5_ch, refine_ch=None, topk_ratio=0.02):
        super().__init__()
        refine_ch = int(refine_ch or max(n3_ch // 2, 32))
        self.topk_ratio = float(topk_ratio)

        self.f2_proj = ConvBNAct(f2_ch, refine_ch, 1)
        self.n4_proj = ConvBNAct(n4_ch, refine_ch, 1)
        self.n5_proj = ConvBNAct(n5_ch, refine_ch, 1)

        self.heat_head = nn.Sequential(
            ConvBNAct(2 * refine_ch, refine_ch, 3),
            nn.Conv2d(refine_ch, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.local_refine = nn.Sequential(
            ResidualRefineBlock(refine_ch),
            ResidualRefineBlock(refine_ch),
        )
        self.n3_fuse = ConvBNAct(n3_ch + refine_ch, n3_ch, 3)

    def _build_sparse_mask(self, heatmap):
        b, _, h, w = heatmap.shape
        flat = heatmap.flatten(2)
        k = max(1, int(round(h * w * self.topk_ratio)))
        topv = flat.topk(k, dim=-1).values
        thresh = topv[:, :, -1].view(b, 1, 1, 1)
        return (heatmap >= thresh).float()

    def forward(self, f2, n3, n4, n5):
        f2p = self.f2_proj(f2)
        n4p = self.n4_proj(n4)
        n5p = self.n5_proj(n5)
        n5_up = F.interpolate(n5p, size=n4p.shape[-2:], mode="bilinear", align_corners=False)

        candidate_logits = self.heat_head(torch.cat([n5_up, n4p], dim=1))
        candidate_heatmap = torch.sigmoid(candidate_logits)
        sparse_mask = self._build_sparse_mask(candidate_heatmap)

        sparse_mask_f2 = F.interpolate(sparse_mask, size=f2p.shape[-2:], mode="nearest")
        candidate_heat_f2 = F.interpolate(candidate_heatmap, size=f2p.shape[-2:], mode="bilinear", align_corners=False)
        f2_sparse = f2p * sparse_mask_f2 * candidate_heat_f2

        h2 = self.local_refine(f2_sparse) + f2_sparse
        h2_to_n3 = F.interpolate(h2, size=n3.shape[-2:], mode="bilinear", align_corners=False)
        n3_enhanced = self.n3_fuse(torch.cat([n3, h2_to_n3], dim=1))

        aux = {
            "candidate_heatmap": candidate_heatmap,
            "candidate_logits": candidate_logits,
            "sparse_mask": sparse_mask,
            "f2_sparse": f2_sparse,
            "h2": h2,
            "h2_to_n3": h2_to_n3,
        }
        return n3_enhanced, aux


class SARMHead(nn.Module):
    """Scene-Adaptive Reliability Modulation."""
    def __init__(self, sem_ch, hidden=128, state_dim=64, nms_gamma=0.15):
        super().__init__()
        in_dim = sem_ch * 3 + 6
        self.nms_gamma = float(nms_gamma)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, state_dim),
            nn.SiLU(inplace=True),
        )
        self.quality_head = nn.Linear(state_dim, 1)
        self.conf_gain_head = nn.Linear(state_dim, 1)
        self.nms_head = nn.Linear(state_dim, 1)

    @staticmethod
    def _global_stats(img):
        mean = img.mean(dim=(1, 2, 3))
        std = img.std(dim=(1, 2, 3), unbiased=False)
        vmin = img.amin(dim=(1, 2, 3))
        vmax = img.amax(dim=(1, 2, 3))
        return mean, std, vmax - vmin

    def forward(self, rgb_img, ir_img, rgb_sem, ir_sem, fused_sem, base_iou_thres=0.7):
        rgb_pool = F.adaptive_avg_pool2d(rgb_sem, 1).flatten(1)
        ir_pool = F.adaptive_avg_pool2d(ir_sem, 1).flatten(1)
        fused_pool = F.adaptive_avg_pool2d(fused_sem, 1).flatten(1)

        rgb_mean, rgb_std, rgb_range = self._global_stats(rgb_img)
        ir_mean, ir_std, ir_range = self._global_stats(ir_img)
        stats = torch.stack([rgb_mean, rgb_std, rgb_range, ir_mean, ir_std, ir_range], dim=1)

        scene_input = torch.cat([rgb_pool, ir_pool, fused_pool, stats], dim=1)
        scene_state = self.encoder(scene_input)

        quality_score = torch.sigmoid(self.quality_head(scene_state))
        conf_gain = torch.sigmoid(self.conf_gain_head(scene_state))
        nms_bias = torch.tanh(self.nms_head(scene_state))
        adaptive_iou_thres = torch.clamp(base_iou_thres + self.nms_gamma * nms_bias.squeeze(1), 0.35, 0.85)

        agreement_target = 0.5 * (1.0 + F.cosine_similarity(rgb_pool, ir_pool, dim=1, eps=1e-6))
        semantic_consistency = (1.0 - F.cosine_similarity(rgb_pool, ir_pool, dim=1, eps=1e-6)).mean()

        return {
            "scene_state": scene_state,
            "quality_score": quality_score,
            "conf_gain": conf_gain,
            "adaptive_iou_thres": adaptive_iou_thres,
            "agreement_target": agreement_target.unsqueeze(1),
            "semantic_consistency": semantic_consistency,
        }


# =========================
# 3. Dual-branch YOLO11-OBB baseline / TASCNet
# =========================
class DualYOLO11MiddleFusionOBB(nn.Module):

    variant_name = "baseline_obb"

    def __init__(self, model_cfg=None, nc=5, verbose=True):
        super().__init__()
        model_cfg = resolve_existing_path(model_cfg, desc="model cfg", required=True)
        print(f"[INFO] Resolved model cfg: {model_cfg}")

        self.rgb_model = OBBModel(cfg=model_cfg, ch=3, nc=nc, verbose=verbose)
        self.rgb_layers = self.rgb_model.model

        self.backbone_end_idx = self._infer_backbone_end_idx(self.rgb_model)
        self.detect_idx = self._find_detect_idx(self.rgb_layers)
        if self.detect_idx <= self.backbone_end_idx:
            raise RuntimeError(
                f"Invalid graph layout: detect_idx={self.detect_idx}, backbone_end_idx={self.backbone_end_idx}"
            )

        ir_full_model = OBBModel(cfg=model_cfg, ch=3, nc=nc, verbose=False)
        self.ir_backbone = nn.ModuleList(list(ir_full_model.model[: self.backbone_end_idx + 1]))
        del ir_full_model

        self.fusion_indices = self._infer_fusion_indices(self.rgb_layers, self.backbone_end_idx)
        self.fusions = nn.ModuleDict({str(idx): SimpleAddFusion() for idx in self.fusion_indices})

        self.detect = self.rgb_layers[self.detect_idx]
        self.model = self.rgb_model.model
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, angle=1.0, task="obb")
        self.criterion = None
        self.nc = nc
        self.stride = self.rgb_model.stride
        self._last_aux = {}
        self._last_aux_losses = {}

        print(f"[INFO] Backbone ends at layer {self.backbone_end_idx}; detect head at layer {self.detect_idx}")
        print(f"[INFO] Auto-discovered fusion nodes: {self.fusion_indices}")

    @staticmethod
    def _infer_backbone_end_idx(model):
        model_yaml = getattr(model, "yaml", None)
        if isinstance(model_yaml, dict) and isinstance(model_yaml.get("backbone"), list) and len(model_yaml["backbone"]) > 0:
            return len(model_yaml["backbone"]) - 1
        raise RuntimeError("Could not infer backbone boundary from model.yaml['backbone'].")

    @staticmethod
    def _find_detect_idx(layers):
        detect_candidates = [
            i for i, m in enumerate(layers)
            if m.__class__.__name__.lower() in {"obb", "detect"}
        ]
        if not detect_candidates:
            raise RuntimeError("Could not find Detect/OBB head in model graph.")
        return detect_candidates[-1]

    @staticmethod
    def _as_list(value):
        return list(value) if isinstance(value, (list, tuple)) else [value]

    def _infer_fusion_indices(self, layers, backbone_end_idx):
        fusion_indices = {backbone_end_idx}
        for idx in range(backbone_end_idx + 1, len(layers)):
            refs = self._as_list(getattr(layers[idx], "f", -1))
            for ref in refs:
                if ref == -1:
                    if idx == backbone_end_idx + 1:
                        fusion_indices.add(backbone_end_idx)
                    continue
                if 0 <= ref <= backbone_end_idx:
                    fusion_indices.add(ref)
        return sorted(fusion_indices)

    @staticmethod
    def _resolve_layer_input(module, outputs, input_tensor=None):
        f = getattr(module, "f", -1)
        if f == -1:
            if outputs:
                return outputs[-1]
            if input_tensor is None:
                raise RuntimeError("Missing input tensor for the first layer.")
            return input_tensor
        if isinstance(f, int):
            return outputs[f]
        return [outputs[-1] if j == -1 else outputs[j] for j in f]

    def _forward_graph(self, layers, input_tensor, start_idx=0, end_idx=None, outputs=None):
        if end_idx is None:
            end_idx = len(layers) - 1
        if outputs is None:
            outputs = []

        for idx in range(start_idx, end_idx + 1):
            layer = layers[idx]
            x_in = self._resolve_layer_input(layer, outputs, input_tensor=input_tensor)
            x_out = layer(x_in)
            if len(outputs) == idx:
                outputs.append(x_out)
            elif len(outputs) > idx:
                outputs[idx] = x_out
            else:
                raise RuntimeError(f"Graph outputs out of sync at layer {idx}.")
        return outputs[end_idx], outputs

    def init_criterion(self):
        return v8OBBLoss(self)

    def compute_loss(self, batch, preds=None):
        if self.criterion is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["rgb"], batch["ir"])
        batch_for_loss = dict(batch)
        batch_for_loss["img"] = batch["rgb"]
        return self.criterion(preds, batch_for_loss)

    def _forward_backbones(self, rgb, ir):
        _, rgb_outputs = self._forward_graph(self.rgb_layers, rgb, end_idx=self.backbone_end_idx)
        _, ir_outputs = self._forward_graph(self.ir_backbone, ir, end_idx=self.backbone_end_idx)

        fused_outputs = list(rgb_outputs)
        for idx in self.fusion_indices:
            fused_outputs[idx] = self.fusions[str(idx)](rgb_outputs[idx], ir_outputs[idx])
        return fused_outputs

    def forward(self, rgb, ir):
        self._last_aux = {}
        fused_outputs = self._forward_backbones(rgb, ir)
        _, outputs = self._forward_graph(
            self.rgb_layers,
            input_tensor=rgb,
            start_idx=self.backbone_end_idx + 1,
            end_idx=self.detect_idx,
            outputs=fused_outputs,
        )
        return outputs[self.detect_idx]

    def load_branch_pretrained(self, rgb_ckpt=None, ir_ckpt=None, copy_rgb_to_ir=True):
        if rgb_ckpt is not None:
            self._load_matching_weights(self.rgb_model, rgb_ckpt, branch_name="RGB")
            if ir_ckpt is None and copy_rgb_to_ir:
                self._load_matching_weights(self.ir_backbone, rgb_ckpt, branch_name="IR(copy RGB)")
        if ir_ckpt is not None:
            self._load_matching_weights(self.ir_backbone, ir_ckpt, branch_name="IR")

    @staticmethod
    def _extract_state_dict(ckpt_obj):
        if isinstance(ckpt_obj, dict):
            if "model_state_dict" in ckpt_obj:
                return ckpt_obj["model_state_dict"]
            if "state_dict" in ckpt_obj:
                return ckpt_obj["state_dict"]
            if "model" in ckpt_obj and hasattr(ckpt_obj["model"], "state_dict"):
                return ckpt_obj["model"].float().state_dict()
            return ckpt_obj
        if hasattr(ckpt_obj, "state_dict"):
            return ckpt_obj.state_dict()
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt_obj)}")

    @staticmethod
    def _canonical_state_key(key):
        for prefix in ("module.", "model.", "rgb_model.", "ir_model.", "ir_backbone."):
            if key.startswith(prefix):
                return key[len(prefix):]
        return key

    def _load_matching_weights(self, module, ckpt_path, branch_name="branch"):
        ckpt_path = Path(ckpt_path).expanduser().resolve()
        if not ckpt_path.exists():
            print(f"[WARN] {branch_name} pretrained checkpoint not found: {ckpt_path}")
            return
        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )
        src_state = self._extract_state_dict(ckpt)
        src_lookup = {self._canonical_state_key(k): v for k, v in src_state.items()}

        dst_state = module.state_dict()
        matched = 0
        for dst_key, dst_value in dst_state.items():
            src_value = src_lookup.get(self._canonical_state_key(dst_key))
            if src_value is not None and dst_value.shape == src_value.shape:
                dst_state[dst_key] = src_value
                matched += 1
        module.load_state_dict(dst_state, strict=False)
        print(f"[INFO] Loaded {matched} matched params into {branch_name} from {ckpt_path}")


class ReliabilityOnlyFusionOBB(DualYOLO11MiddleFusionOBB):
    """Controlled ablation: reliability-weighted feature fusion only.

    Uses the same dual YOLO11-OBB branches and detection neck/head as the
    simple-add baseline. It intentionally excludes S2RH, SARM, and their
    auxiliary losses so the fusion operator is the main experimental variable.
    """
    variant_name = "reliability_only_obb"

    def __init__(self, model_cfg=None, nc=5, verbose=True):
        super().__init__(model_cfg=model_cfg, nc=nc, verbose=verbose)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            _, rgb_outputs = self._forward_graph(
                self.rgb_layers, dummy, end_idx=self.backbone_end_idx
            )
        if was_training:
            self.train()

        self.fusions = nn.ModuleDict({
            str(idx): TaRFusion(
                int(rgb_outputs[idx].shape[1]),
                int(rgb_outputs[idx].shape[1]),
                c_out=int(rgb_outputs[idx].shape[1]),
            )
            for idx in self.fusion_indices
        })

    def _forward_backbones(self, rgb, ir):
        _, rgb_outputs = self._forward_graph(
            self.rgb_layers, rgb, end_idx=self.backbone_end_idx
        )
        _, ir_outputs = self._forward_graph(
            self.ir_backbone, ir, end_idx=self.backbone_end_idx
        )
        fused_outputs = list(rgb_outputs)
        fusion_aux = {}
        for idx in self.fusion_indices:
            fused, aux = self.fusions[str(idx)](rgb_outputs[idx], ir_outputs[idx])
            fused_outputs[idx] = fused
            fusion_aux[idx] = aux
        self._last_aux = {
            "rgb_outputs": rgb_outputs,
            "ir_outputs": ir_outputs,
            "fusion_aux": fusion_aux,
        }
        return fused_outputs


class TASCNetOBB(DualYOLO11MiddleFusionOBB):
    variant_name = "tascnet_obb"

    def __init__(
        self,
        model_cfg=None,
        nc=5,
        verbose=True,
        s2rh_topk_ratio=0.02,
        small_area_thr=0.01,
        small_alpha=2.0,
        small_tau=0.0025,
        lambda_cal=0.2,
        lambda_qua=0.2,
        lambda_small=0.3,
        sarm_nms_gamma=0.15,
        semantic_consistency_weight=0.25,
    ):
        super().__init__(model_cfg=model_cfg, nc=nc, verbose=verbose)

        self.feature_layout = self._infer_feature_layout()
        self.primary_fusion_indices = self.feature_layout["backbone_scales"]
        self.detect_feature_indices = self.feature_layout["detect_scales"]
        self.f2_idx = self.feature_layout["f2_idx"]

        fusion_modules = {}
        for idx in self.fusion_indices:
            channels = self.feature_layout["channels"][idx]
            if idx in self.primary_fusion_indices:
                fusion_modules[str(idx)] = TaRFusion(channels, channels, c_out=channels)
            else:
                fusion_modules[str(idx)] = SimpleAddFusion()
        self.fusions = nn.ModuleDict(fusion_modules)

        n3_idx, n4_idx, n5_idx = self.detect_feature_indices
        ch_f2 = self.feature_layout["channels"][self.f2_idx]
        ch_n3 = self.feature_layout["channels"][n3_idx]
        ch_n4 = self.feature_layout["channels"][n4_idx]
        ch_n5 = self.feature_layout["channels"][n5_idx]
        p5_idx = self.primary_fusion_indices[-1]
        ch_p5 = self.feature_layout["channels"][p5_idx]

        self.s2rh = S2RHHead(
            f2_ch=ch_f2,
            n3_ch=ch_n3,
            n4_ch=ch_n4,
            n5_ch=ch_n5,
            topk_ratio=s2rh_topk_ratio,
        )
        self.sarm = SARMHead(sem_ch=ch_p5, nms_gamma=sarm_nms_gamma)

        self.small_area_thr = float(small_area_thr)
        self.small_alpha = float(small_alpha)
        self.small_tau = float(small_tau)
        self.lambda_cal = float(lambda_cal)
        self.lambda_qua = float(lambda_qua)
        self.lambda_small = float(lambda_small)
        self.semantic_consistency_weight = float(semantic_consistency_weight)
        self.base_iou_thres = 0.7

        print(f"[INFO] TASCNet backbone scales (P3/P4/P5): {self.primary_fusion_indices}")
        print(f"[INFO] TASCNet shallow feature F2 index: {self.f2_idx}")
        print(f"[INFO] TASCNet detect scales (N3/N4/N5): {self.detect_feature_indices}")

    def _infer_feature_layout(self, input_size=128):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            _, backbone_outputs = self._forward_graph(self.rgb_layers, dummy, end_idx=self.backbone_end_idx)
            _, all_outputs = self._forward_graph(self.rgb_layers, dummy, end_idx=self.detect_idx)
        if was_training:
            self.train()

        def area_of(idx, outputs):
            x = outputs[idx]
            return int(x.shape[-2] * x.shape[-1])

        backbone_candidates = []
        seen_hw = set()
        for idx in sorted(self.fusion_indices, key=lambda j: (-area_of(j, all_outputs), j)):
            hw = tuple(all_outputs[idx].shape[-2:])
            if hw not in seen_hw:
                backbone_candidates.append(idx)
                seen_hw.add(hw)
        if len(backbone_candidates) < 3:
            raise RuntimeError(f"Need at least 3 multi-scale backbone features, got {backbone_candidates}")
        backbone_scales = backbone_candidates[:3]
        backbone_scales = sorted(backbone_scales, key=lambda j: area_of(j, all_outputs), reverse=True)

        p3_idx = backbone_scales[0]
        p3_area = area_of(p3_idx, backbone_outputs)
        f2_candidates = [
            idx for idx in range(p3_idx)
            if torch.is_tensor(backbone_outputs[idx]) and backbone_outputs[idx].ndim == 4 and area_of(idx, backbone_outputs) > p3_area
        ]
        if f2_candidates:
            f2_idx = min(f2_candidates, key=lambda j: area_of(j, backbone_outputs))
        else:
            all_spatial = [idx for idx in range(p3_idx) if torch.is_tensor(backbone_outputs[idx]) and backbone_outputs[idx].ndim == 4]
            if not all_spatial:
                raise RuntimeError("Could not infer a shallow high-resolution feature F2.")
            f2_idx = max(all_spatial, key=lambda j: area_of(j, backbone_outputs))

        detect_refs = [idx for idx in self._as_list(getattr(self.detect, "f", [])) if isinstance(idx, int) and idx >= 0]
        if len(detect_refs) < 3:
            raise RuntimeError(f"Detect layer must reference at least 3 input scales, got {detect_refs}")
        detect_scales = sorted(detect_refs, key=lambda j: area_of(j, all_outputs), reverse=True)

        interesting = set(backbone_scales + detect_scales + [f2_idx])
        channels = {idx: int(all_outputs[idx].shape[1]) for idx in interesting}
        return {
            "backbone_scales": backbone_scales,
            "detect_scales": detect_scales,
            "f2_idx": f2_idx,
            "channels": channels,
        }

    def _forward_backbones(self, rgb, ir):
        _, rgb_outputs = self._forward_graph(self.rgb_layers, rgb, end_idx=self.backbone_end_idx)
        _, ir_outputs = self._forward_graph(self.ir_backbone, ir, end_idx=self.backbone_end_idx)

        fused_outputs = list(rgb_outputs)
        fusion_aux = {}
        for idx in self.fusion_indices:
            module = self.fusions[str(idx)]
            if isinstance(module, TaRFusion):
                fused_outputs[idx], fusion_aux[idx] = module(rgb_outputs[idx], ir_outputs[idx])
            else:
                fused_outputs[idx] = module(rgb_outputs[idx], ir_outputs[idx])
        return fused_outputs, {
            "rgb_outputs": rgb_outputs,
            "ir_outputs": ir_outputs,
            "fusion_aux": fusion_aux,
        }

    def forward(self, rgb, ir):
        fused_outputs, backbone_aux = self._forward_backbones(rgb, ir)

        _, outputs = self._forward_graph(
            self.rgb_layers,
            input_tensor=rgb,
            start_idx=self.backbone_end_idx + 1,
            end_idx=self.detect_idx - 1,
            outputs=fused_outputs,
        )

        n3_idx, n4_idx, n5_idx = self.detect_feature_indices
        f2 = 0.5 * (backbone_aux["rgb_outputs"][self.f2_idx] + backbone_aux["ir_outputs"][self.f2_idx])

        n3 = outputs[n3_idx]
        n4 = outputs[n4_idx]
        n5 = outputs[n5_idx]
        n3_enhanced, s2rh_aux = self.s2rh(f2, n3, n4, n5)
        outputs[n3_idx] = n3_enhanced

        detect_layer = self.rgb_layers[self.detect_idx]
        detect_in = self._resolve_layer_input(detect_layer, outputs, input_tensor=rgb)
        preds = detect_layer(detect_in)

        p5_idx = self.primary_fusion_indices[-1]
        tarf_p5 = backbone_aux["fusion_aux"].get(p5_idx, {})
        rgb_sem = tarf_p5.get("rgb_focus", backbone_aux["rgb_outputs"][p5_idx])
        ir_sem = tarf_p5.get("ir_focus", backbone_aux["ir_outputs"][p5_idx])
        fused_sem = fused_outputs[p5_idx]
        sarm_aux = self.sarm(
            rgb_img=rgb,
            ir_img=ir,
            rgb_sem=rgb_sem,
            ir_sem=ir_sem,
            fused_sem=fused_sem,
            base_iou_thres=self.base_iou_thres,
        )

        self._last_aux = {
            "rgb_outputs": backbone_aux["rgb_outputs"],
            "ir_outputs": backbone_aux["ir_outputs"],
            "fusion_aux": backbone_aux["fusion_aux"],
            "fused_outputs": fused_outputs,
            "neck_outputs": outputs,
            "s2rh": s2rh_aux,
            "sarm": sarm_aux,
        }
        return preds

    def compute_auxiliary_losses(self, batch, det_loss_scalar):
        device = batch["rgb"].device
        aux = getattr(self, "_last_aux", {}) or {}

        candidate_heatmap = aux.get("s2rh", {}).get("candidate_heatmap")
        if candidate_heatmap is None:
            zero = torch.zeros((), device=device)
            self._last_aux_losses = {
                "l_cal": 0.0,
                "l_qua": 0.0,
                "l_small": 0.0,
                "quality_mean": 0.0,
                "adaptive_iou": self.base_iou_thres,
            }
            return {"l_cal": zero, "l_qua": zero, "l_small": zero}

        small_target = build_small_object_target_map(
            batch=batch,
            spatial_size=candidate_heatmap.shape[-2:],
            area_thr=self.small_area_thr,
            device=device,
        )
        small_weight = compute_small_object_weight(
            batch=batch,
            area_thr=self.small_area_thr,
            alpha=self.small_alpha,
            tau=self.small_tau,
            device=device,
        )
        l_small = F.binary_cross_entropy(candidate_heatmap, small_target) * small_weight

        sarm_aux = aux["sarm"]
        quality_pred = sarm_aux["quality_score"]
        agreement_target = sarm_aux["agreement_target"]
        l_qua = F.mse_loss(quality_pred, agreement_target.detach()) + self.semantic_consistency_weight * sarm_aux["semantic_consistency"]

        det_quality_target = torch.exp(-det_loss_scalar.detach()).clamp(0.0, 1.0)
        det_quality_target = det_quality_target.expand_as(sarm_aux["conf_gain"])
        l_cal = F.mse_loss(sarm_aux["conf_gain"], det_quality_target)

        self._last_aux_losses = {
            "l_cal": float(l_cal.detach().cpu()),
            "l_qua": float(l_qua.detach().cpu()),
            "l_small": float(l_small.detach().cpu()),
            "quality_mean": float(quality_pred.mean().detach().cpu()),
            "adaptive_iou": float(sarm_aux["adaptive_iou_thres"].mean().detach().cpu()),
        }
        return {"l_cal": l_cal, "l_qua": l_qua, "l_small": l_small}

    def compute_loss(self, batch, preds=None):
        if self.criterion is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["rgb"], batch["ir"])

        batch_for_loss = dict(batch)
        batch_for_loss["img"] = batch["rgb"]
        det_loss_out = self.criterion(preds, batch_for_loss)
        det_loss, det_items = normalize_loss_output(det_loss_out, batch["rgb"].device)

        aux_losses = self.compute_auxiliary_losses(batch, det_loss)
        total_loss = det_loss + self.lambda_cal * aux_losses["l_cal"] + self.lambda_qua * aux_losses["l_qua"] + self.lambda_small * aux_losses["l_small"]
        return total_loss, det_items


# =========================
# 4. Train / Val Utils
# =========================
def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out



def normalize_loss_output(loss_out, device):
    if isinstance(loss_out, (tuple, list)) and len(loss_out) == 2:
        loss, loss_items = loss_out
    else:
        loss, loss_items = loss_out, None

    if not torch.is_tensor(loss):
        raise TypeError(f"Unexpected loss type: {type(loss)}")

    if loss.ndim > 0:
        if loss.numel() == 1:
            loss = loss.reshape(())
        else:
            if loss_items is None:
                loss_items = loss.detach()
            loss = loss.sum()

    if loss_items is None:
        loss_items = torch.tensor([loss.item(), 0.0, 0.0, 0.0], device=device)
    elif not torch.is_tensor(loss_items):
        loss_items = torch.tensor([float(loss), 0.0, 0.0, 0.0], device=device)
    else:
        loss_items = loss_items.detach().flatten()
        if loss_items.numel() == 0:
            loss_items = torch.tensor([loss.item(), 0.0, 0.0, 0.0], device=device)
        elif loss_items.numel() == 1:
            loss_items = torch.tensor([loss_items[0].item(), 0.0, 0.0, 0.0], device=device)
        elif loss_items.numel() == 2:
            loss_items = torch.tensor([loss_items[0].item(), loss_items[1].item(), 0.0, 0.0], device=device)
        elif loss_items.numel() == 3:
            loss_items = torch.tensor([loss_items[0].item(), loss_items[1].item(), loss_items[2].item(), 0.0], device=device)
        else:
            loss_items = loss_items[:4]

    return loss, loss_items



def unpack_model_outputs(preds):
    if isinstance(preds, (tuple, list)) and len(preds) == 2 and torch.is_tensor(preds[0]):
        return preds[0], preds
    if torch.is_tensor(preds):
        return preds, preds
    if isinstance(preds, dict):
        return None, preds
    raise TypeError(f"Unsupported prediction container type: {type(preds)}")



def build_small_object_target_map(batch, spatial_size, area_thr=0.01, device=None):
    h, w = spatial_size
    device = device or batch["rgb"].device
    bs = int(batch["rgb"].shape[0])
    target = torch.zeros((bs, 1, h, w), dtype=torch.float32, device=device)

    if batch["bboxes"].numel() == 0:
        return target

    batch_idx = batch["batch_idx"].view(-1).long()
    boxes = batch["bboxes"]
    for bi in range(bs):
        mask = batch_idx == bi
        if not mask.any():
            continue
        boxes_i = boxes[mask]
        area_i = boxes_i[:, 2] * boxes_i[:, 3]
        small_mask = area_i <= area_thr
        if not small_mask.any():
            continue
        boxes_small = boxes_i[small_mask]
        for box in boxes_small:
            cx = float(box[0].clamp(0.0, 1.0) * (w - 1))
            cy = float(box[1].clamp(0.0, 1.0) * (h - 1))
            bw = max(1.0, float(box[2].clamp(0.0, 1.0) * w))
            bh = max(1.0, float(box[3].clamp(0.0, 1.0) * h))
            rx = max(1, int(round(bw / 2.0)))
            ry = max(1, int(round(bh / 2.0)))

            x0 = max(0, int(round(cx)) - rx)
            x1 = min(w, int(round(cx)) + rx + 1)
            y0 = max(0, int(round(cy)) - ry)
            y1 = min(h, int(round(cy)) + ry + 1)
            target[bi, 0, y0:y1, x0:x1] = 1.0

    return target


def compute_small_object_weight(batch, area_thr=0.01, alpha=2.0, tau=0.0025, device=None):
    device = device or batch["rgb"].device
    if batch["bboxes"].numel() == 0:
        return torch.tensor(1.0, device=device)

    areas = batch["bboxes"][:, 2] * batch["bboxes"][:, 3]
    small_mask = areas <= area_thr
    if not small_mask.any():
        return torch.tensor(1.0, device=device)

    small_areas = areas[small_mask]
    weights = 1.0 + alpha * torch.exp(-small_areas / max(tau, 1e-6))
    return weights.mean()


def get_model_aux_losses(model):
    aux = getattr(model, "_last_aux_losses", None)
    if not isinstance(aux, dict):
        return {}
    return aux


def scale_xywhrn_to_xywhr(x, w=640, h=640):
    y = x.clone()
    y[:, 0] *= w
    y[:, 1] *= h
    y[:, 2] *= w
    y[:, 3] *= h
    return y



def obb_postprocess(
    preds,
    conf_thres=0.001,
    iou_thres=0.7,
    nc=5,
    max_det=300,
    agnostic=False,
    per_image_iou_thres=None,
    quality_scores=None,
):
    inference_preds, _ = unpack_model_outputs(preds)
    if inference_preds is None:
        raise TypeError("OBB postprocess expects an inference tensor or a (inference, raw_preds) tuple.")

    if inference_preds.ndim < 3:
        raise ValueError(f"Expected batched inference predictions, got shape: {tuple(inference_preds.shape)}")

    bs = int(inference_preds.shape[0])
    if per_image_iou_thres is None:
        per_image_iou_thres = [float(iou_thres)] * bs
    elif torch.is_tensor(per_image_iou_thres):
        per_image_iou_thres = per_image_iou_thres.detach().flatten().tolist()
    else:
        per_image_iou_thres = [float(x) for x in per_image_iou_thres]

    if quality_scores is None:
        quality_scores = [1.0] * bs
    elif torch.is_tensor(quality_scores):
        quality_scores = quality_scores.detach().flatten().tolist()
    else:
        quality_scores = [float(x) for x in quality_scores]

    processed = []
    for bi in range(bs):
        outputs = non_max_suppression(
            inference_preds[bi : bi + 1],
            conf_thres,
            float(per_image_iou_thres[min(bi, len(per_image_iou_thres) - 1)]),
            nc=nc,
            multi_label=True,
            agnostic=agnostic,
            max_det=max_det,
            rotated=True,
        )
        x = outputs[0] if outputs else None
        if x is None or x.numel() == 0:
            dev = inference_preds.device if torch.is_tensor(inference_preds) else "cpu"
            processed.append(
                {
                    "bboxes": torch.zeros((0, 5), device=dev),
                    "conf": torch.zeros((0,), device=dev),
                    "cls": torch.zeros((0,), device=dev),
                }
            )
            continue

        q = float(quality_scores[min(bi, len(quality_scores) - 1)])
        x = x.clone()
        x[:, 4] = x[:, 4] * q
        angle = x[:, 6:7] if x.shape[1] > 6 else torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        bboxes = torch.cat([x[:, :4], angle], dim=1)
        processed.append({
            "bboxes": bboxes,
            "conf": x[:, 4],
            "cls": x[:, 5],
        })
    return processed



def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return np.trapz(np.interp(x, mrec, mpre), x)



def ap_per_class_custom(tp, conf, pred_cls, target_cls, eps=1e-16):
    if len(tp) == 0:
        return np.zeros(0), np.zeros(0), np.zeros((0, 10)), np.array([], dtype=int)
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes = np.unique(target_cls.astype(int))
    nc = unique_classes.shape[0]
    ap = np.zeros((nc, tp.shape[1]))
    precision = np.zeros(nc)
    recall = np.zeros(nc)
    for ci, c in enumerate(unique_classes):
        mask = pred_cls == c
        n_l = (target_cls == c).sum()
        n_p = mask.sum()
        if n_l == 0 or n_p == 0:
            continue
        fpc = (1 - tp[mask]).cumsum(0)
        tpc = tp[mask].cumsum(0)
        recall_curve = tpc / (n_l + eps)
        precision_curve = tpc / (tpc + fpc + eps)
        recall[ci] = recall_curve[-1, 0]
        precision[ci] = precision_curve[-1, 0]
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall_curve[:, j], precision_curve[:, j])
    return precision, recall, ap, unique_classes



def process_batch_obb(preds, batch_gt, iouv):
    pred_boxes = preds["bboxes"]
    pred_cls = preds["cls"]
    gt_boxes = batch_gt["bboxes"]
    gt_cls = batch_gt["cls"]

    correct = torch.zeros((pred_boxes.shape[0], iouv.numel()), dtype=torch.bool, device=iouv.device)
    if gt_boxes.shape[0] == 0 or pred_boxes.shape[0] == 0:
        return correct

    iou = batch_probiou(gt_boxes, pred_boxes)
    correct_class = gt_cls[:, None] == pred_cls[None]

    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), dim=1).cpu().numpy()
            if matches.shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct



def train_one_epoch(model, loader, optimizer, device, epoch, accumulation_steps=1, print_freq=20):
    model.train()
    running_loss = 0.0
    running_items = torch.zeros(4, device=device)
    running_aux = {"l_cal": 0.0, "l_qua": 0.0, "l_small": 0.0, "quality_mean": 0.0, "adaptive_iou": 0.0}
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        preds = model(batch["rgb"], batch["ir"])
        loss_out = model.compute_loss(batch, preds)
        loss, loss_items = normalize_loss_output(loss_out, device)
        (loss / accumulation_steps).backward()

        do_step = ((i + 1) % accumulation_steps == 0) or ((i + 1) == len(loader))
        if do_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        running_items += loss_items

        aux_losses = get_model_aux_losses(model)
        for key in running_aux:
            running_aux[key] += float(aux_losses.get(key, 0.0))

        if (i + 1) % print_freq == 0 or (i + 1) == len(loader):
            avg_loss = running_loss / (i + 1)
            avg_items = running_items / (i + 1)
            avg_aux = {k: v / (i + 1) for k, v in running_aux.items()}
            print(
                f"Epoch [{epoch}] Iter [{i + 1}/{len(loader)}] loss={avg_loss:.4f} "
                f"(box={avg_items[0].item():.4f}, cls={avg_items[1].item():.4f}, "
                f"dfl={avg_items[2].item():.4f}, angle={avg_items[3].item():.4f}, "
                f"cal={avg_aux['l_cal']:.4f}, qua={avg_aux['l_qua']:.4f}, small={avg_aux['l_small']:.4f})"
            )

    avg_aux = {k: v / max(len(loader), 1) for k, v in running_aux.items()}
    return running_loss / max(len(loader), 1), (running_items / max(len(loader), 1)).detach().cpu(), avg_aux


@torch.no_grad()
def validate_one_epoch(model, loader, device, conf_thres=0.001, iou_thres=0.7, max_det=300):
    model.eval()
    running_loss = 0.0
    running_items = torch.zeros(4, device=device)
    running_aux = {"l_cal": 0.0, "l_qua": 0.0, "l_small": 0.0, "quality_mean": 0.0, "adaptive_iou": 0.0}
    iouv = torch.linspace(0.50, 0.95, 10, device=device)
    stats = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        preds = model(batch["rgb"], batch["ir"])
        inference_preds, loss_preds = unpack_model_outputs(preds)

        loss_out = model.compute_loss(batch, loss_preds)
        loss, loss_items = normalize_loss_output(loss_out, device)
        running_loss += loss.item()
        running_items += loss_items

        aux_losses = get_model_aux_losses(model)
        for key in running_aux:
            running_aux[key] += float(aux_losses.get(key, 0.0))

        model_aux = getattr(model, "_last_aux", {}) or {}
        sarm_aux = model_aux.get("sarm", {})
        adaptive_iou = sarm_aux.get("adaptive_iou_thres", None)
        quality_scores = sarm_aux.get("quality_score", None)

        preds_nms = obb_postprocess(
            inference_preds,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            nc=model.nc,
            max_det=max_det,
            per_image_iou_thres=adaptive_iou,
            quality_scores=quality_scores,
        )
        bs = batch["rgb"].shape[0]
        h, w = batch["rgb"].shape[2], batch["rgb"].shape[3]

        for si in range(bs):
            pred = preds_nms[si]
            idx_mask = batch["batch_idx"][:, 0] == si
            labels_i = torch.cat([batch["cls"][idx_mask], batch["bboxes"][idx_mask]], dim=1)

            if labels_i.shape[0]:
                gt_bboxes = scale_xywhrn_to_xywhr(labels_i[:, 1:6], w=w, h=h)
                gt = {"cls": labels_i[:, 0], "bboxes": gt_bboxes}
            else:
                gt = {
                    "cls": torch.zeros((0,), device=device),
                    "bboxes": torch.zeros((0, 5), device=device),
                }

            if pred is None or pred["cls"].numel() == 0:
                stats.append(
                    (
                        torch.zeros((0, iouv.numel()), dtype=torch.bool),
                        torch.tensor([]),
                        torch.tensor([]),
                        gt["cls"].cpu(),
                    )
                )
                continue

            correct = process_batch_obb(pred, gt, iouv)
            stats.append((correct.cpu(), pred["conf"].cpu(), pred["cls"].cpu(), gt["cls"].cpu()))

    metrics = {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0}
    if len(stats):
        stats = [torch.cat(x, 0).numpy() if len(x) and isinstance(x[0], torch.Tensor) else np.array([]) for x in zip(*stats)]
        if len(stats) >= 4 and stats[0].shape[0] > 0:
            tp, confs, pred_cls, target_cls = stats
            p, r, ap, _ = ap_per_class_custom(tp, confs, pred_cls, target_cls)
            ap50 = ap[:, 0] if len(ap) else np.array([0.0])
            ap5095 = ap.mean(1) if len(ap) else np.array([0.0])
            metrics["precision"] = float(p.mean() if len(p) else 0.0)
            metrics["recall"] = float(r.mean() if len(r) else 0.0)
            metrics["map50"] = float(ap50.mean() if len(ap50) else 0.0)
            metrics["map50_95"] = float(ap5095.mean() if len(ap5095) else 0.0)

    avg_aux = {k: v / max(len(loader), 1) for k, v in running_aux.items()}
    return running_loss / max(len(loader), 1), (running_items / max(len(loader), 1)).detach().cpu(), metrics, avg_aux



def save_checkpoint(model, optimizer, scheduler, epoch, best_val_map50, save_path, extra_cfg=None, metrics=None):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_map50": best_val_map50,
        "metrics": metrics or {},
        "extra_cfg": extra_cfg or {},
    }
    torch.save(ckpt, save_path)



def build_model(variant, model_cfg, nc, verbose=True, model_kwargs=None):
    variant = (variant or "baseline_obb").lower()
    model_kwargs = model_kwargs or {}

    baseline_aliases = {"baseline", "baseline_obb", "yolo11_baseline", "yolo11_obb"}
    tascnet_aliases = {"tascnet", "tascnet_obb", "tasc", "tasc-net"}
    reliability_aliases = {"reliability_only", "reliability_only_obb", "tar_only"}

    if variant in tascnet_aliases:
        return TASCNetOBB(model_cfg=model_cfg, nc=nc, verbose=verbose, **model_kwargs)
    if variant in reliability_aliases:
        if model_kwargs:
            print("[WARN] Reliability-only variant ignores TASCNet-specific kwargs.")
        return ReliabilityOnlyFusionOBB(model_cfg=model_cfg, nc=nc, verbose=verbose)
    if variant in baseline_aliases:
        if model_kwargs:
            print("[WARN] Baseline variant ignores TASCNet-specific kwargs.")
        return DualYOLO11MiddleFusionOBB(model_cfg=model_cfg, nc=nc, verbose=verbose)

    print(f"[WARN] Unknown variant '{variant}'. Fallback to TASCNet OBB.")
    return TASCNetOBB(model_cfg=model_cfg, nc=nc, verbose=verbose, **model_kwargs)



def has_split_config(cfg_path, split="test"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required_keys = [f"{split}_rgb", f"{split}_ir"]
    has_primary = all(k in cfg and cfg[k] not in (None, "") for k in required_keys)
    has_rgb_label = any(k in cfg and cfg[k] not in (None, "") for k in [f"{split}_rgb_labels", f"{split}_labels", f"{split}_label"])
    return has_primary and has_rgb_label

def run_training(args):
    device_arg = str(args.device).strip().lower()

    if device_arg.isdigit():
        device = torch.device(
            f"cuda:{device_arg}" if torch.cuda.is_available() else "cpu"
        )
    elif device_arg == "cuda":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)

    cfg_path = args.data
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = load_yaml_config(cfg_path)
    runtime_paths = resolve_runtime_paths(args, data_cfg)
    model_cfg = runtime_paths["model_cfg"]
    rgb_pretrained = runtime_paths["rgb_pretrained"]
    ir_pretrained = runtime_paths["ir_pretrained"]
    copy_rgb_to_ir = runtime_paths["copy_rgb_to_ir"]

    train_set = RGBIRDataset(cfg_path, split="train", imgsz=args.imgsz, strict_label_pair=args.strict_label_pair)
    val_set = RGBIRDataset(cfg_path, split="val", imgsz=args.imgsz, strict_label_pair=args.strict_label_pair)

    test_set = None
    if has_split_config(cfg_path, split="test"):
        test_set = RGBIRDataset(cfg_path, split="test", imgsz=args.imgsz, strict_label_pair=args.strict_label_pair)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(args.workers // 2, 1),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(args.workers // 2, 1),
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
        )

    model_kwargs = {
        "s2rh_topk_ratio": args.s2rh_topk_ratio,
        "small_area_thr": args.small_area_thr,
        "small_alpha": args.small_alpha,
        "small_tau": args.small_tau,
        "lambda_cal": args.lambda_cal,
        "lambda_qua": args.lambda_qua,
        "lambda_small": args.lambda_small,
        "sarm_nms_gamma": args.sarm_nms_gamma,
        "semantic_consistency_weight": args.semantic_consistency_weight,
    }
    model = build_model(args.variant, model_cfg=model_cfg, nc=train_set.nc, verbose=True, model_kwargs=model_kwargs).to(device)
    print(f"[INFO] RGB pretrained: {rgb_pretrained if rgb_pretrained else 'None'}")
    if ir_pretrained:
        print(f"[INFO] IR pretrained: {ir_pretrained}")
    else:
        print(f"[INFO] IR pretrained: None{' (copy RGB)' if copy_rgb_to_ir and rgb_pretrained else ''}")
    model.load_branch_pretrained(
        rgb_ckpt=rgb_pretrained,
        ir_ckpt=ir_pretrained,
        copy_rgb_to_ir=copy_rgb_to_ir,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    best_val_map50 = -1.0
    best_epoch = -1

    print(f"\n========== Start Training ({args.variant}, YOLO11-OBB) ==========")
    print(f"Device: {device}")
    print(
        f"Epochs: {args.epochs}, imgsz: {args.imgsz}, batch: {args.batch_size}, accum: {args.accumulation_steps}, "
        f"eff_batch: {args.batch_size * args.accumulation_steps}, lr: {args.lr}"
    )
    print(f"Model cfg: {model_cfg}")
    print(f"Data cfg: {cfg_path}")
    print(f"Save dir: {save_dir}")

    train_cfg = {
        "cfg_path": cfg_path,
        "model_cfg": model_cfg,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "workers": args.workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "save_dir": str(save_dir),
        "variant": args.variant,
        "best_metric": "val_map50",
        "task": "obb",
        "rgb_pretrained": rgb_pretrained,
        "ir_pretrained": ir_pretrained,
        "copy_rgb_to_ir": copy_rgb_to_ir,
        "strict_label_pair": args.strict_label_pair,
        "has_test_split": test_set is not None,
        "s2rh_topk_ratio": args.s2rh_topk_ratio,
        "small_area_thr": args.small_area_thr,
        "small_alpha": args.small_alpha,
        "small_tau": args.small_tau,
        "lambda_cal": args.lambda_cal,
        "lambda_qua": args.lambda_qua,
        "lambda_small": args.lambda_small,
        "sarm_nms_gamma": args.sarm_nms_gamma,
        "semantic_consistency_weight": args.semantic_consistency_weight,
    }
    save_train_config(save_dir, train_cfg)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_items, train_aux = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            accumulation_steps=args.accumulation_steps,
            print_freq=args.print_freq,
        )
        val_loss, val_items, val_metrics, val_aux = validate_one_epoch(
            model,
            val_loader,
            device,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            max_det=args.max_det,
        )
        scheduler.step()

        current_metrics = {
            "train_loss": float(train_loss),
            "train_box_loss": float(train_items[0]),
            "train_cls_loss": float(train_items[1]),
            "train_dfl_loss": float(train_items[2]),
            "train_angle_loss": float(train_items[3]),
            "val_loss": float(val_loss),
            "val_box_loss": float(val_items[0]),
            "val_cls_loss": float(val_items[1]),
            "val_dfl_loss": float(val_items[2]),
            "val_angle_loss": float(val_items[3]),
            "train_l_cal": float(train_aux["l_cal"]),
            "train_l_qua": float(train_aux["l_qua"]),
            "train_l_small": float(train_aux["l_small"]),
            "train_quality_mean": float(train_aux["quality_mean"]),
            "train_adaptive_iou": float(train_aux["adaptive_iou"]),
            "val_l_cal": float(val_aux["l_cal"]),
            "val_l_qua": float(val_aux["l_qua"]),
            "val_l_small": float(val_aux["l_small"]),
            "val_quality_mean": float(val_aux["quality_mean"]),
            "val_adaptive_iou": float(val_aux["adaptive_iou"]),
            "precision": float(val_metrics["precision"]),
            "recall": float(val_metrics["recall"]),
            "map50": float(val_metrics["map50"]),
            "map50_95": float(val_metrics["map50_95"]),
        }

        log_row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **current_metrics,
        }
        append_metrics_csv(save_dir / "train_log.csv", log_row)

        print(
            f"\n[Epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} "
            f"(box={train_items[0]:.4f}, cls={train_items[1]:.4f}, dfl={train_items[2]:.4f}, angle={train_items[3]:.4f}, "
            f"cal={train_aux['l_cal']:.4f}, qua={train_aux['l_qua']:.4f}, small={train_aux['l_small']:.4f}) | "
            f"val_loss={val_loss:.4f} "
            f"(box={val_items[0]:.4f}, cls={val_items[1]:.4f}, dfl={val_items[2]:.4f}, angle={val_items[3]:.4f}, "
            f"cal={val_aux['l_cal']:.4f}, qua={val_aux['l_qua']:.4f}, small={val_aux['l_small']:.4f}) | "
            f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} "
            f"mAP50={val_metrics['map50']:.4f} mAP50-95={val_metrics['map50_95']:.4f} | "
            f"q={val_aux['quality_mean']:.4f} nms_iou={val_aux['adaptive_iou']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        save_checkpoint(model, optimizer, scheduler, epoch, best_val_map50, save_dir / "last.pt", extra_cfg=train_cfg, metrics=current_metrics)

        if val_metrics["map50"] > best_val_map50:
            best_val_map50 = val_metrics["map50"]
            best_epoch = epoch
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_map50, save_dir / "best.pt", extra_cfg=train_cfg, metrics=current_metrics)
            print(f"[INFO] Best model updated at epoch {epoch}, val mAP50={val_metrics['map50']:.4f}")

    print("\n========== Training Finished ==========")
    print(f"Best epoch: {best_epoch}, best val mAP50: {best_val_map50:.4f}")

    if test_loader is not None:
        best_ckpt_path = save_dir / "best.pt"
        if best_ckpt_path.exists():
            ckpt = torch.load(best_ckpt_path, map_location=device)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Loaded best checkpoint for test evaluation: {best_ckpt_path}")

        test_loss, test_items, test_metrics, test_aux = validate_one_epoch(
            model,
            test_loader,
            device,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            max_det=args.max_det,
        )

        test_summary = {
            "test_loss": float(test_loss),
            "test_box_loss": float(test_items[0]),
            "test_cls_loss": float(test_items[1]),
            "test_dfl_loss": float(test_items[2]),
            "test_angle_loss": float(test_items[3]),
            "test_l_cal": float(test_aux["l_cal"]),
            "test_l_qua": float(test_aux["l_qua"]),
            "test_l_small": float(test_aux["l_small"]),
            "test_quality_mean": float(test_aux["quality_mean"]),
            "test_adaptive_iou": float(test_aux["adaptive_iou"]),
            "test_precision": float(test_metrics["precision"]),
            "test_recall": float(test_metrics["recall"]),
            "test_map50": float(test_metrics["map50"]),
            "test_map50_95": float(test_metrics["map50_95"]),
        }
        with open(save_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_summary, f, indent=2, ensure_ascii=False)

        print(
            f"\n[Test] loss={test_loss:.4f} "
            f"(box={test_items[0]:.4f}, cls={test_items[1]:.4f}, dfl={test_items[2]:.4f}, angle={test_items[3]:.4f}, "
            f"cal={test_aux['l_cal']:.4f}, qua={test_aux['l_qua']:.4f}, small={test_aux['l_small']:.4f}) | "
            f"P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f} "
            f"mAP50={test_metrics['map50']:.4f} mAP50-95={test_metrics['map50_95']:.4f} | "
            f"q={test_aux['quality_mean']:.4f} nms_iou={test_aux['adaptive_iou']:.4f}"
        )
        print(f"[INFO] Test metrics saved to: {save_dir / 'test_metrics.json'}")

def parse_args():
    parser = argparse.ArgumentParser(description="Dual-branch RGB-IR YOLO11-OBB baseline / TASCNet training script.")
    parser.add_argument("--data", type=str, default="./configs/a18_rgbir_obb_train_aug.yaml", help="RGB-IR OBB dataset yaml path.")
    parser.add_argument(
        "--model-cfg",
        type=str,
        default=None,
        help="Ultralytics YOLO11-OBB model yaml path. If omitted, try reading model_cfg/model/model_yaml from --data.",
    )
    parser.add_argument(
        "--rgb-pretrained",
        type=str,
        default=None,
        help="RGB branch pretrained checkpoint path. If omitted, try reading rgb_pretrained/rgb_weights/pretrained/weights from --data.",
    )
    parser.add_argument(
        "--ir-pretrained",
        type=str,
        default=None,
        help="IR branch pretrained checkpoint path. If omitted, try reading ir_pretrained/ir_weights from --data.",
    )
    parser.add_argument(
        "--copy-rgb-to-ir",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to initialize the IR backbone from RGB weights when --ir-pretrained is not provided. If omitted, try reading copy_rgb_to_ir from --data (default: True).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="tascnet_obb",
        help="Model variant: baseline_obb or tascnet_obb.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument("--s2rh-topk-ratio", type=float, default=0.02, help="Top-K sparse selection ratio for S2RH.")
    parser.add_argument("--small-area-thr", type=float, default=0.01, help="Normalized area threshold for small targets.")
    parser.add_argument("--small-alpha", type=float, default=2.0, help="Alpha in small-target reweighting.")
    parser.add_argument("--small-tau", type=float, default=0.0025, help="Tau in small-target reweighting.")
    parser.add_argument("--lambda-cal", type=float, default=0.2, help="Weight for SARM calibration loss.")
    parser.add_argument("--lambda-qua", type=float, default=0.2, help="Weight for SARM quality / semantic consistency loss.")
    parser.add_argument("--lambda-small", type=float, default=0.3, help="Weight for S2RH small-target loss.")
    parser.add_argument("--sarm-nms-gamma", type=float, default=0.15, help="Scene-adaptive NMS threshold modulation amplitude.")
    parser.add_argument("--semantic-consistency-weight", type=float, default=0.25, help="Weight of weather-invariant semantic consistency inside L_qua.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--save-dir", type=str, default="./runs/a18_rgbir_yolo11_obb_baseline")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--strict-label-pair", action="store_true", help="Require both RGB/IR label files to exist for each sample.")
    return parser.parse_args()



if __name__ == "__main__":
    run_training(parse_args())
