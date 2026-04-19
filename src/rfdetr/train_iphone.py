"""
train_iphone.py
Train RF-DETR-B on the Road_poles_iPhone dataset.

Original layout:
    Road_poles_iPhone/
        images/Train/train/*.jpg
        images/Validation/val/*.jpg
        images/Test/*.jpg          (no labels)
        labels/Train/train/*.txt
        labels/Validation/val/*.txt  (or Validation/*.txt)

This script:
    1. Converts iPhone YOLO labels to COCO JSON for RF-DETR
       → Road_poles_iPhone/train/_annotations.coco.json
       → Road_poles_iPhone/valid/_annotations.coco.json
       → Road_poles_iPhone/test/_annotations.coco.json
    2. Starts RF-DETR-B training

Usage (with your venv activated):
    python train_iphone.py
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional
from PIL import Image
import torch
from rfdetr import RFDETRBase

# Paths (this repo: data/Poles2025)
_REPO_ROOT = Path(__file__).resolve().parents[2]
IPHONE_DIR = str(_REPO_ROOT / "data" / "Poles2025" / "Road_poles_iPhone")
DATASET_DIR = IPHONE_DIR  # RF-DETR dataset_dir; creates train/ valid/ test/ below
OUTPUT_DIR = str(_REPO_ROOT / "runs" / "rfdetr" / "iphone")

# Training hyperparameters
BATCH_SIZE       = 2
GRAD_ACCUM_STEPS = 8
NUM_EPOCHS       = 100
LEARNING_RATE    = 1e-4
NUM_CLASSES      = 1        # single class: pole
RESOLUTION       = 672      # must be a multiple of 56

# Per-split image/label subdirs under IPHONE_DIR
# Tuple: (split_name, img_subdir, lbl_subdir)
#   split_name — RF-DETR expects train / valid / test
#   img_subdir — relative to IPHONE_DIR/images/
#   lbl_subdir — relative to IPHONE_DIR/labels/ (None = no labels)
SPLIT_MAP = [
    ("train", "Train/train", "Train/train"),
    ("valid", "Validation/val", "Validation/val"),
    ("test",  "Test",         None),           # test: no labels
]

CATEGORIES = [{"id": 1, "name": "pole", "supercategory": "object"}]


# YOLO → COCO conversion

def convert_split(split_name: str, img_subdir: str, lbl_subdir: Optional[str]):
    """
    Convert one split from YOLO to COCO JSON under DATASET_DIR/<split_name>/.

    RF-DETR expects:
        dataset_dir/
            train/
                images/
                _annotations.coco.json
            valid/
                images/
                _annotations.coco.json
            test/
                images/
                _annotations.coco.json
    """
    src_img_dir = os.path.join(IPHONE_DIR, "images", img_subdir)
    src_lbl_dir = os.path.join(IPHONE_DIR, "labels", lbl_subdir) if lbl_subdir else None

    dst_split_dir = os.path.join(DATASET_DIR, split_name)
    dst_img_dir   = os.path.join(dst_split_dir, "images")
    out_json_path = os.path.join(dst_split_dir, "_annotations.coco.json")

    os.makedirs(dst_img_dir, exist_ok=True)

    img_files = sorted([
        f for f in os.listdir(src_img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    images      = []
    annotations = []
    ann_id      = 1

    for img_id, img_file in enumerate(img_files, start=1):
        src_img_path = os.path.join(src_img_dir, img_file)
        dst_img_path = os.path.join(dst_img_dir, img_file)

        if not os.path.exists(dst_img_path):
            shutil.copy2(src_img_path, dst_img_path)

        with Image.open(src_img_path) as img:
            w, h = img.size

        images.append({
            "id": img_id,
            "file_name": "images/" + img_file,
            "width": w,
            "height": h,
        })

        if src_lbl_dir is None:
            continue  # test: no labels

        stem     = os.path.splitext(img_file)[0]
        lbl_path = os.path.join(src_lbl_dir, stem + ".txt")

        if not os.path.exists(lbl_path):
            continue  # negative sample

        with open(lbl_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            abs_w  = bw * w
            abs_h  = bh * h
            x_min  = (cx - bw / 2) * w
            y_min  = (cy - bh / 2) * h

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id + 1,
                "bbox": [round(x_min, 4), round(y_min, 4), round(abs_w, 4), round(abs_h, 4)],
                "area": round(abs_w * abs_h, 4),
                "iscrowd": 0,
            })
            ann_id += 1

    coco_json = {
        "info": {"description": f"Road_poles_iPhone {split_name} set (converted from YOLO)"},
        "licenses": [],
        "categories": CATEGORIES,
        "images": images,
        "annotations": annotations,
    }

    with open(out_json_path, "w") as f:
        json.dump(coco_json, f, indent=2)

    print(f"[{split_name:5s}] {len(images):4d} images, {len(annotations):4d} annotations → {out_json_path}")


def prepare_dataset():
    """Prepare iPhone dataset in RF-DETR layout."""
    print("=" * 60)
    print("Step 1: Convert iPhone YOLO → COCO JSON")
    print("=" * 60)
    for split_name, img_subdir, lbl_subdir in SPLIT_MAP:
        convert_split(split_name, img_subdir, lbl_subdir)
    print()


# Training callback

def on_epoch_end(metrics: dict):
    """Print summary metrics at the end of each epoch."""
    epoch      = metrics.get("epoch", "?")
    train_loss = metrics.get("train_loss", None)

    coco_bbox     = metrics.get("test_coco_eval_bbox", None)
    ema_coco_bbox = metrics.get("ema_test_coco_eval_bbox", None)

    map_5095 = coco_bbox[0]     if coco_bbox     else None
    map_50   = coco_bbox[1]     if coco_bbox     else None
    ema_5095 = ema_coco_bbox[0] if ema_coco_bbox else None
    ema_50   = ema_coco_bbox[1] if ema_coco_bbox else None

    print("\n" + "=" * 70)
    line = f"  Epoch {epoch:>3}"
    if train_loss is not None:
        line += f"  |  train_loss={train_loss:.4f}"
    if map_5095 is not None:
        line += f"  |  mAP@0.5:0.95={map_5095:.4f}"
    if map_50 is not None:
        line += f"  |  mAP@0.5={map_50:.4f}"
    if ema_5095 is not None:
        line += f"  |  EMA mAP@0.5:0.95={ema_5095:.4f}"
    if ema_50 is not None:
        line += f"  |  EMA mAP@0.5={ema_50:.4f}"
    print(line)
    print("=" * 70 + "\n", flush=True)


# Entry point

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: No GPU detected. CPU training will be very slow.")

    need_convert = any(
        not os.path.exists(os.path.join(DATASET_DIR, s, "_annotations.coco.json"))
        for s, _, _ in SPLIT_MAP
    )
    if need_convert:
        prepare_dataset()
    else:
        print("COCO JSON already present; skipping conversion.\n")

    for split_name, _, _ in SPLIT_MAP:
        ann_path = os.path.join(DATASET_DIR, split_name, "_annotations.coco.json")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Missing {ann_path}; dataset preparation failed.")

    print("=" * 60)
    print("Step 2: Train RF-DETR-B on Road_poles_iPhone")
    print("=" * 60)
    print(f"Dataset:    {DATASET_DIR}")
    print(f"Output:     {OUTPUT_DIR}")
    print(f"Epochs:     {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}  (grad accum {GRAD_ACCUM_STEPS} steps → effective {BATCH_SIZE * GRAD_ACCUM_STEPS})")
    print(f"LR:         {LEARNING_RATE}\n")

    model = RFDETRBase(num_classes=NUM_CLASSES, resolution=RESOLUTION)
    model.callbacks["on_fit_epoch_end"].append(on_epoch_end)

    model.train(
        dataset_dir=DATASET_DIR,
        dataset_file="roboflow",
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        lr=LEARNING_RATE,
        output_dir=OUTPUT_DIR,
        num_workers=2,
        use_ema=True,
        early_stopping=True,
        early_stopping_patience=15,
        early_stopping_use_ema=True,
        tensorboard=True,
    )

    print("\nTraining finished.")
    print(f"Weights and logs: {OUTPUT_DIR}")
    print(f"TensorBoard: tensorboard --logdir {OUTPUT_DIR}")
