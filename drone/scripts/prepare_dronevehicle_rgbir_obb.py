from pathlib import Path
import shutil

"""
Build a paired RGB-IR DroneVehicle dataset with OBB labels.

Inputs:
    datasets/a18_rgb_obb
    datasets/a18_ir_obb

Outputs:
    datasets/a18_rgbir_obb/rgb/{train,val,test}
    datasets/a18_rgbir_obb/ir/{train,val,test}
    datasets/a18_rgbir_obb/labels/{train,val,test}      # RGB labels for backward compatibility
    datasets/a18_rgbir_obb/labels_rgb/{train,val,test}  # explicit RGB labels
    datasets/a18_rgbir_obb/labels_ir/{train,val,test}   # explicit IR labels

This keeps the original paired-dataset workflow while also preserving both
modalities' label files for later inspection or modality-specific training.
"""

RGB_ROOT = Path("./datasets/a18_rgb_obb")
IR_ROOT = Path("./datasets/a18_ir_obb")
OUT_ROOT = Path("./datasets/a18_rgbir_obb")

SPLITS = ["train", "val", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def ensure_dirs():
    for split in SPLITS:
        (OUT_ROOT / "rgb" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "ir" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels_rgb" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels_ir" / split).mkdir(parents=True, exist_ok=True)


def collect_images(folder: Path):
    if not folder.exists():
        return {}
    files = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files[p.stem] = p
    return files


def collect_labels(folder: Path):
    if not folder.exists():
        return {}
    return {p.stem: p for p in folder.glob("*.txt")}


def reset_split(split: str):
    for sub in ["rgb", "ir", "labels", "labels_rgb", "labels_ir"]:
        d = OUT_ROOT / sub / split
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()


def build_split(split: str):
    rgb_img_dir = RGB_ROOT / "images" / split
    ir_img_dir = IR_ROOT / "images" / split
    rgb_label_dir = RGB_ROOT / "labels" / split
    ir_label_dir = IR_ROOT / "labels" / split

    rgb_imgs = collect_images(rgb_img_dir)
    ir_imgs = collect_images(ir_img_dir)
    rgb_labels = collect_labels(rgb_label_dir)
    ir_labels = collect_labels(ir_label_dir)

    paired_stems = sorted(
        set(rgb_imgs.keys())
        & set(ir_imgs.keys())
        & set(rgb_labels.keys())
        & set(ir_labels.keys())
    )

    print(f"\n[{split}]")
    print(f"RGB images: {len(rgb_imgs)}")
    print(f"IR  images: {len(ir_imgs)}")
    print(f"RGB labels: {len(rgb_labels)}")
    print(f"IR  labels: {len(ir_labels)}")
    print(f"Final paired samples: {len(paired_stems)}")

    missing_in_ir = sorted(list(set(rgb_imgs.keys()) - set(ir_imgs.keys())))[:10]
    missing_in_rgb = sorted(list(set(ir_imgs.keys()) - set(rgb_imgs.keys())))[:10]

    if missing_in_ir:
        print(f"[INFO] RGB-only examples: {missing_in_ir}")
    if missing_in_rgb:
        print(f"[INFO] IR-only examples: {missing_in_rgb}")

    reset_split(split)

    copied = 0
    for stem in paired_stems:
        rgb_src = rgb_imgs[stem]
        ir_src = ir_imgs[stem]
        rgb_label_src = rgb_labels[stem]
        ir_label_src = ir_labels[stem]

        rgb_dst = OUT_ROOT / "rgb" / split / f"{stem}{rgb_src.suffix.lower()}"
        ir_dst = OUT_ROOT / "ir" / split / f"{stem}{ir_src.suffix.lower()}"
        compat_label_dst = OUT_ROOT / "labels" / split / f"{stem}.txt"
        rgb_label_dst = OUT_ROOT / "labels_rgb" / split / f"{stem}.txt"
        ir_label_dst = OUT_ROOT / "labels_ir" / split / f"{stem}.txt"

        shutil.copy2(rgb_src, rgb_dst)
        shutil.copy2(ir_src, ir_dst)
        shutil.copy2(rgb_label_src, compat_label_dst)  # keep old behavior
        shutil.copy2(rgb_label_src, rgb_label_dst)
        shutil.copy2(ir_label_src, ir_label_dst)
        copied += 1

    print(f"[{split}] copied paired samples: {copied}")


def main():
    ensure_dirs()
    for split in SPLITS:
        build_split(split)

    print("\n[OK] a18_rgbir_obb paired dataset generated successfully.")
    print("Structure:")
    print("datasets/a18_rgbir_obb/rgb/{train,val,test}")
    print("datasets/a18_rgbir_obb/ir/{train,val,test}")
    print("datasets/a18_rgbir_obb/labels/{train,val,test}      # RGB labels for compatibility")
    print("datasets/a18_rgbir_obb/labels_rgb/{train,val,test}")
    print("datasets/a18_rgbir_obb/labels_ir/{train,val,test}")


if __name__ == "__main__":
    main()
