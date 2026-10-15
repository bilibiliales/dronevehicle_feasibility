from pathlib import Path
import math
import xml.etree.ElementTree as ET
import cv2

"""
Convert DroneVehicle RGB data to YOLO-OBB format.

Output label format (one object per line):
    class_id x1 y1 x2 y2 x3 y3 x4 y4
where x/y are normalized to [0, 1].

This keeps the original rotated box information from the XML polygon
instead of collapsing it into an axis-aligned rectangle.
"""

RAW_ROOT = Path("./raw_data/DroneVehicle")
OUT_ROOT = Path("./datasets/a18_rgb_obb")

SPLITS = {
    "train": {
        "img_dir": RAW_ROOT / "trainimg",
        "label_dir": RAW_ROOT / "trainlabel",
    },
    "val": {
        "img_dir": RAW_ROOT / "valimg",
        "label_dir": RAW_ROOT / "vallabel",
    },
    "test": {
        "img_dir": RAW_ROOT / "testimg",
        "label_dir": RAW_ROOT / "testlabel",
    },
}

IMG_EXTS = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG", "*.bmp", "*.BMP")

CLASS_MAP = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "van": 3,
    "freight car": 4,
    "feright car": 4,  # dataset typo
}


def ensure_dirs():
    for split in SPLITS:
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def collect_images(folder: Path):
    files = []
    for pattern in IMG_EXTS:
        files.extend(folder.glob(pattern))
    return sorted(files)


def clip_point(x: float, y: float, img_w: int, img_h: int):
    x = min(max(x, 0.0), float(img_w - 1))
    y = min(max(y, 0.0), float(img_h - 1))
    return x, y


def polygon_area(points):
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def order_points_cyclic(points):
    """Return 4 points in a stable cyclic order and rotate so point 0 is top-left-ish."""
    cx = sum(x for x, _ in points) / len(points)
    cy = sum(y for _, y in points) / len(points)

    pts = sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    # Rotate so the first point is the one closest to the top-left corner.
    start_idx = min(range(len(pts)), key=lambda i: (pts[i][1], pts[i][0]))
    pts = pts[start_idx:] + pts[:start_idx]
    return pts


def parse_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is None or not name_node.text:
            continue

        name = name_node.text.strip().lower()
        if name not in CLASS_MAP:
            continue

        polygon = obj.find("polygon")
        if polygon is None:
            continue

        try:
            points = []
            for i in range(1, 5):
                x = float(polygon.find(f"x{i}").text)
                y = float(polygon.find(f"y{i}").text)
                points.append((x, y))
        except (AttributeError, TypeError, ValueError):
            continue

        class_id = CLASS_MAP[name]
        objects.append((points, class_id))

    return img_w, img_h, objects


def polygon_to_yolo_obb(points, class_id: int, img_w: int, img_h: int):
    points = [clip_point(x, y, img_w, img_h) for x, y in points]
    points = order_points_cyclic(points)

    if polygon_area(points) <= 1.0:
        return None

    norm = []
    for x, y in points:
        norm.append(x / img_w)
        norm.append(y / img_h)

    if any(v < 0.0 or v > 1.0 for v in norm):
        return None

    coords = " ".join(f"{v:.6f}" for v in norm)
    return f"{class_id} {coords}"


def process_split(split: str, img_dir: Path, label_dir: Path):
    if not img_dir.exists():
        print(f"[WARN] 图像目录不存在，跳过 {split}: {img_dir}")
        return
    if not label_dir.exists():
        print(f"[WARN] 标注目录不存在，跳过 {split}: {label_dir}")
        return

    img_files = collect_images(img_dir)
    print(f"[{split}] 找到图像 {len(img_files)} 张")

    count = 0
    for img_path in img_files:
        stem = img_path.stem
        xml_path = label_dir / f"{stem}.xml"

        if not xml_path.exists():
            print(f"[WARN] 缺失标注: {xml_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 无法读取图像: {img_path}")
            continue

        img_w, img_h, objects = parse_xml(xml_path)
        real_h, real_w = img.shape[:2]
        if (img_w, img_h) != (real_w, real_h):
            print(
                f"[WARN] XML尺寸与图像尺寸不一致: {img_path.name} xml=({img_w},{img_h}) img=({real_w},{real_h})，以XML尺寸归一化"
            )

        obb_lines = []
        for points, class_id in objects:
            line = polygon_to_yolo_obb(points, class_id, img_w, img_h)
            if line is not None:
                obb_lines.append(line)

        if not obb_lines:
            continue

        out_img_path = OUT_ROOT / "images" / split / img_path.name
        out_label_path = OUT_ROOT / "labels" / split / f"{stem}.txt"

        cv2.imwrite(str(out_img_path), img)
        with open(out_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(obb_lines))

        count += 1

    print(f"[{split}] 成功处理 {count} 张")


def main():
    ensure_dirs()
    for split, cfg in SPLITS.items():
        process_split(split, cfg["img_dir"], cfg["label_dir"])

    print("\n[OK] RGB OBB dataset generated successfully.")
    print("Structure:")
    print("datasets/a18_rgb_obb/images/{train,val,test}")
    print("datasets/a18_rgb_obb/labels/{train,val,test}")


if __name__ == "__main__":
    main()
