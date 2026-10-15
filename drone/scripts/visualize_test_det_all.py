import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


IMG_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def load_module(py_path):
    py_path = str(Path(py_path).resolve())
    spec = importlib.util.spec_from_file_location('train_mod', py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_names(data_yaml):
    with open(data_yaml, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    nc = int(cfg['nc'])
    names = cfg.get('names', [f'class_{i}' for i in range(nc)])
    return names, nc


def pick_existing_path(candidates, label):
    for p in candidates:
        if p.exists():
            return p
    joined = '\n'.join(str(p) for p in candidates)
    raise FileNotFoundError(f'No valid {label} found. Tried:\n{joined}')


def resolve_optional_path(path_str, label):
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f'{label} not found: {p}')
    return p


def collect_paired_images(rgb_dir, ir_dir):
    rgb_dir = Path(rgb_dir)
    ir_dir = Path(ir_dir)

    if not rgb_dir.exists():
        raise FileNotFoundError(f'RGB dir not found: {rgb_dir}')
    if not ir_dir.exists():
        raise FileNotFoundError(f'IR dir not found: {ir_dir}')

    rgb_map = {
        p.stem: p for p in sorted(rgb_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMG_SUFFIXES
    }
    ir_map = {
        p.stem: p for p in sorted(ir_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMG_SUFFIXES
    }

    common_stems = sorted(set(rgb_map.keys()) & set(ir_map.keys()))
    if not common_stems:
        raise RuntimeError(
            f'No paired RGB-IR images found between:\nRGB: {rgb_dir}\nIR: {ir_dir}'
        )

    return [(rgb_map[s], ir_map[s]) for s in common_stems]


def xywhr_to_4pts_float(cx, cy, w, h, angle):
    rect = (
        (float(cx), float(cy)),
        (float(w), float(h)),
        float(angle * 180.0 / np.pi)
    )
    pts = cv2.boxPoints(rect)
    return pts.astype(np.float32)


def get_class_color(cls_name, cls_id):
    color_map = {
        'car': (0, 255, 0),
        'truck': (255, 0, 0),
        'bus': (0, 165, 255),
        'van': (255, 0, 255),
        'freight_car': (0, 255, 255),
        'freight car': (0, 255, 255),
    }

    if cls_name in color_map:
        return color_map[cls_name]

    fallback_palette = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 165, 255),
        (255, 0, 255),
        (0, 255, 255),
        (128, 255, 0),
        (255, 128, 0),
    ]
    return fallback_palette[cls_id % len(fallback_palette)]


def draw_obb_predictions_on_original(img_bgr, pred, names, orig_h, orig_w, input_size):
    if img_bgr is None or img_bgr.size == 0:
        raise RuntimeError('img_bgr is empty before drawing')

    out = img_bgr.copy()

    if pred is None:
        return out
    if 'cls' not in pred or pred['cls'] is None or pred['cls'].numel() == 0:
        return out

    cls_ids = pred['cls'].detach().cpu().numpy().astype(int).tolist()
    confs = pred['conf'].detach().cpu().numpy().tolist()
    boxes = pred['bboxes'].detach().cpu().numpy()

    scale_x = float(orig_w) / float(input_size)
    scale_y = float(orig_h) / float(input_size)

    for box, cls_id, conf in zip(boxes, cls_ids, confs):
        cx, cy, bw, bh, ang = box.tolist()

        pts = xywhr_to_4pts_float(cx, cy, bw, bh, ang)
        pts[:, 0] *= scale_x
        pts[:, 1] *= scale_y

        pts[:, 0] = np.clip(pts[:, 0], 0, max(orig_w - 1, 0))
        pts[:, 1] = np.clip(pts[:, 1], 0, max(orig_h - 1, 0))
        pts_int = np.round(pts).astype(np.int32)

        cls_name = names[cls_id] if 0 <= cls_id < len(names) else f'class_{cls_id}'
        color = get_class_color(cls_name, cls_id)

        cv2.polylines(out, [pts_int], isClosed=True, color=color, thickness=2)

        label = f'{cls_name} {conf:.2f}'
        x = int(np.min(pts_int[:, 0]))
        y = int(np.min(pts_int[:, 1]))

        cv2.putText(
            out,
            label,
            (max(0, x), max(19, y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if 'model_state_dict' in ckpt_obj:
            return ckpt_obj['model_state_dict']
        if 'state_dict' in ckpt_obj:
            return ckpt_obj['state_dict']
        if 'model' in ckpt_obj and hasattr(ckpt_obj['model'], 'state_dict'):
            return ckpt_obj['model'].float().state_dict()
        return ckpt_obj
    if hasattr(ckpt_obj, 'state_dict'):
        return ckpt_obj.state_dict()
    raise TypeError(f'Unsupported checkpoint type: {type(ckpt_obj)}')


def preprocess_bgr_crop(crop_bgr, imgsz, force_gray=False):
    if force_gray:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        orig_h, orig_w = gray.shape[:2]
        vis_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        img = cv2.resize(gray, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.stack([img, img, img], axis=-1)
    else:
        orig_h, orig_w = crop_bgr.shape[:2]
        vis_bgr = crop_bgr.copy()

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0

    img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    return img_tensor, vis_bgr, (orig_h, orig_w)


@torch.no_grad()
def infer_single_crop(
    model,
    mod,
    rgb_crop_bgr,
    ir_crop_bgr,
    imgsz,
    device,
    names,
    nc,
    conf_thres=0.25,
    iou_thres=0.7,
    max_det=300,
    ir_gray=False,
):
    rgb, rgb_vis, (rgb_h, rgb_w) = preprocess_bgr_crop(
        rgb_crop_bgr, imgsz, force_gray=False
    )
    ir, _, _ = preprocess_bgr_crop(
        ir_crop_bgr, imgsz, force_gray=ir_gray
    )

    preds = model(
        rgb.unsqueeze(0).to(device),
        ir.unsqueeze(0).to(device)
    )

    pred_list = mod.obb_postprocess(
        preds,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        nc=nc,
        max_det=max_det,
    )
    pred = pred_list[0] if pred_list else None

    drawn = draw_obb_predictions_on_original(
        rgb_vis,
        pred,
        names,
        orig_h=rgb_h,
        orig_w=rgb_w,
        input_size=imgsz,
    )

    num_boxes = 0
    if pred is not None and 'cls' in pred and pred['cls'] is not None:
        num_boxes = int(pred['cls'].numel())

    return drawn, num_boxes


def split_image_grid_2x2(img_bgr, overlap_x=0, overlap_y=0):
    h, w = img_bgr.shape[:2]
    if h < 2 or w < 2:
        return [{
            'name': 'full',
            'x0': 0, 'x1': w,
            'y0': 0, 'y1': h,
            'img': img_bgr.copy(),
        }]

    mid_x = w // 2
    mid_y = h // 2

    overlap_x = max(0, int(overlap_x))
    overlap_y = max(0, int(overlap_y))

    x_ranges = [
        (0, min(w, mid_x + overlap_x)),
        (max(0, mid_x - overlap_x), w),
    ]
    y_ranges = [
        (0, min(h, mid_y + overlap_y)),
        (max(0, mid_y - overlap_y), h),
    ]

    names = ['tl', 'tr', 'bl', 'br']
    coords = [
        (x_ranges[0][0], x_ranges[0][1], y_ranges[0][0], y_ranges[0][1]),
        (x_ranges[1][0], x_ranges[1][1], y_ranges[0][0], y_ranges[0][1]),
        (x_ranges[0][0], x_ranges[0][1], y_ranges[1][0], y_ranges[1][1]),
        (x_ranges[1][0], x_ranges[1][1], y_ranges[1][0], y_ranges[1][1]),
    ]

    crops = []
    for name, (x0, x1, y0, y1) in zip(names, coords):
        crops.append({
            'name': name,
            'x0': x0, 'x1': x1,
            'y0': y0, 'y1': y1,
            'img': img_bgr[y0:y1, x0:x1].copy(),
        })
    return crops


def stitch_grid_crops(crop_results, orig_h, orig_w):
    canvas = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

    # 固定顺序覆盖：tl -> tr -> bl -> br
    # 重叠区域以后贴的块为准
    for item in crop_results:
        drawn = item['drawn']
        x0, x1 = item['x0'], item['x1']
        y0, y1 = item['y0'], item['y1']

        expected_h = y1 - y0
        expected_w = x1 - x0
        if drawn.shape[0] != expected_h or drawn.shape[1] != expected_w:
            raise RuntimeError(
                f"Crop size mismatch for {item['name']}: "
                f"drawn={drawn.shape[:2]}, expected={(expected_h, expected_w)}"
            )

        canvas[y0:y1, x0:x1] = drawn

    return canvas


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb-dir', type=str, default='/root/A18/MVV')
    parser.add_argument('--ir-dir', type=str, default=None, help='可选；不填时默认复用 rgb-dir')
    parser.add_argument('--ir-gray', action='store_true', help='IR 按灰度读入并扩成 3 通道')
    parser.add_argument('--split-name', type=str, default='test')
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.7)
    parser.add_argument('--max-det', type=int, default=300)
    parser.add_argument('--save-dir', type=str, default='/root/A18/MVV_pred')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0, help='0 means process all images')

    parser.add_argument('--ckpt', type=str, default='/root/A18/runs/exp_tascnet/best.pt')
    parser.add_argument('--train-script', type=str, default=None)
    parser.add_argument('--data-yaml', type=str, default=None)
    parser.add_argument('--model-cfg', type=str, default=None)
    parser.add_argument('--variant', type=str, default='tascnet_obb')

    parser.add_argument('--split-mode', type=str, default='grid4', choices=['none', 'grid4'])
    parser.add_argument('--split-overlap-x', type=int, default=120, help='横向切块重叠像素')
    parser.add_argument('--split-overlap-y', type=int, default=120, help='纵向切块重叠像素')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    if args.ir_dir is None or str(args.ir_dir).strip() == '':
        args.ir_dir = args.rgb_dir
        print(f'[INFO] --ir-dir not provided, reuse rgb-dir as IR dir: {args.ir_dir}')

    if not args.ir_gray and Path(args.rgb_dir).resolve() == Path(args.ir_dir).resolve():
        print('[WARN] rgb-dir 和 ir-dir 相同，但未开启 --ir-gray。')
        print('[WARN] 这会把同一张 RGB 图直接送进两个分支；通常建议加上 --ir-gray。')

    if args.train_script is not None:
        train_script = resolve_optional_path(args.train_script, 'train script')
    else:
        train_script = pick_existing_path(
            [
                project_root / 'scripts' / 'tascnet_yolo11_obb.py',
                project_root / 'scripts' / 'tascnet yolo11 obb.py',
            ],
            'train script'
        )

    if args.data_yaml is not None:
        data_yaml = resolve_optional_path(args.data_yaml, 'data yaml')
    else:
        data_yaml = pick_existing_path(
            [
                project_root / 'configs' / 'a18_rgbir_obb_train.yaml',
                project_root / 'configs' / 'a18 rgbir obb train.yaml',
            ],
            'data yaml'
        )

    if args.model_cfg is not None:
        model_cfg = resolve_optional_path(args.model_cfg, 'model cfg')
    else:
        model_cfg = project_root / 'ultralytics' / 'ultralytics' / 'cfg' / 'models' / '11' / 'yolo11-obb.yaml'

    ckpt_path = resolve_optional_path(args.ckpt, 'checkpoint')

    for p in [train_script, data_yaml, model_cfg]:
        if not Path(p).exists():
            raise FileNotFoundError(f'Required file not found: {p}')

    mod = load_module(train_script)
    names, nc = load_names(data_yaml)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'[INFO] train_script   : {train_script}')
    print(f'[INFO] data_yaml      : {data_yaml}')
    print(f'[INFO] model_cfg      : {model_cfg}')
    print(f'[INFO] checkpoint     : {ckpt_path}')
    print(f'[INFO] rgb_dir        : {args.rgb_dir}')
    print(f'[INFO] ir_dir         : {args.ir_dir}')
    print(f'[INFO] ir_gray        : {args.ir_gray}')
    print(f'[INFO] variant        : {args.variant}')
    print(f'[INFO] split_mode     : {args.split_mode}')
    print(f'[INFO] split_overlap_x: {args.split_overlap_x}')
    print(f'[INFO] split_overlap_y: {args.split_overlap_y}')

    pairs = collect_paired_images(args.rgb_dir, args.ir_dir)
    total_images = len(pairs)

    start = max(0, int(args.start))
    end = total_images if int(args.limit) <= 0 else min(total_images, start + int(args.limit))
    pairs = pairs[start:end]

    if not pairs:
        raise RuntimeError(
            f'No image pairs selected. total={total_images}, start={start}, limit={args.limit}'
        )

    model = mod.build_model(
        args.variant,
        model_cfg=str(model_cfg),
        nc=nc,
        verbose=True,
        model_kwargs={},
    ).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f'[INFO] Loaded checkpoint: {ckpt_path}')
    if missing:
        print(f'[WARN] Missing keys: {len(missing)}')
    if unexpected:
        print(f'[WARN] Unexpected keys: {len(unexpected)}')

    model.eval()

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    if args.split_mode == 'grid4':
        print('[INFO] Output visualization uses 2x2 grid inference and stitched RGB result')
    else:
        print('[INFO] Output visualization uses whole-image inference and saves RGB result only')

    print(f'[INFO] Processing {len(pairs)} RGB-IR pairs from index range [{start}, {end})')

    for local_idx, (rgb_path, ir_path) in enumerate(pairs):
        global_idx = start + local_idx
        stem = rgb_path.stem

        rgb_full = cv2.imread(str(rgb_path))
        if rgb_full is None:
            raise FileNotFoundError(f'cannot read rgb image: {rgb_path}')

        ir_full = cv2.imread(str(ir_path))
        if ir_full is None:
            raise FileNotFoundError(f'cannot read ir image: {ir_path}')

        orig_h, orig_w = rgb_full.shape[:2]

        if args.split_mode == 'none':
            pred_rgb, num_boxes = infer_single_crop(
                model=model,
                mod=mod,
                rgb_crop_bgr=rgb_full,
                ir_crop_bgr=ir_full,
                imgsz=args.imgsz,
                device=device,
                names=names,
                nc=nc,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                max_det=args.max_det,
                ir_gray=args.ir_gray,
            )
        else:
            rgb_crops = split_image_grid_2x2(
                rgb_full,
                overlap_x=args.split_overlap_x,
                overlap_y=args.split_overlap_y,
            )
            ir_crops = split_image_grid_2x2(
                ir_full,
                overlap_x=args.split_overlap_x,
                overlap_y=args.split_overlap_y,
            )

            crop_results = []
            total_boxes = 0

            for rgb_crop, ir_crop in zip(rgb_crops, ir_crops):
                drawn, num_boxes_crop = infer_single_crop(
                    model=model,
                    mod=mod,
                    rgb_crop_bgr=rgb_crop['img'],
                    ir_crop_bgr=ir_crop['img'],
                    imgsz=args.imgsz,
                    device=device,
                    names=names,
                    nc=nc,
                    conf_thres=args.conf_thres,
                    iou_thres=args.iou_thres,
                    max_det=args.max_det,
                    ir_gray=args.ir_gray,
                )

                crop_results.append({
                    'name': rgb_crop['name'],
                    'x0': rgb_crop['x0'],
                    'x1': rgb_crop['x1'],
                    'y0': rgb_crop['y0'],
                    'y1': rgb_crop['y1'],
                    'drawn': drawn,
                })
                total_boxes += num_boxes_crop

            pred_rgb = stitch_grid_crops(crop_results, orig_h=orig_h, orig_w=orig_w)
            num_boxes = total_boxes

        save_path_rgb = save_root / f'{args.split_name}_{global_idx:05d}_{stem}_pred_rgb.png'
        cv2.imwrite(str(save_path_rgb), pred_rgb)

        if (local_idx + 1) % 20 == 0 or (local_idx + 1) == len(pairs):
            print(
                f'[INFO] Done {local_idx + 1}/{len(pairs)}: '
                f'stem={stem}, out_size=({pred_rgb.shape[0]},{pred_rgb.shape[1]}), boxes={num_boxes}'
            )

    print('\n========== Done ==========')
    print(f'Processed image pairs: {len(pairs)} / {total_images}')
    print(f'Output root: {save_root}')
    print('Saved files:')
    print('  *_pred_rgb.png  (grid-stitched RGB result only)')


if __name__ == '__main__':
    main()