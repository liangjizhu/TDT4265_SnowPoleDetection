"""
yolo_to_coco.py
Convert roadpoles_v1 YOLO-format labels to COCO JSON.
RF-DETR expects `_annotations.coco.json` under each split directory.

Usage:
    python yolo_to_coco.py

Outputs (under DATASET_DIR):
    train/_annotations.coco.json
    valid/_annotations.coco.json
    test/_annotations.coco.json
"""

import os
import json
from pathlib import Path
from PIL import Image

# Paths (this repo: dataset at data/Poles2025/roadpoles_v1)
_REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = str(_REPO_ROOT / "data" / "Poles2025" / "roadpoles_v1")

CATEGORIES = [{"id": 1, "name": "pole", "supercategory": "object"}]

SPLITS = ["train", "valid", "test"]


def convert_split(split: str):
    img_dir = os.path.join(DATASET_DIR, split, "images")
    lbl_dir = os.path.join(DATASET_DIR, split, "labels")
    out_path = os.path.join(DATASET_DIR, split, "_annotations.coco.json")

    images = []
    annotations = []
    ann_id = 1

    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    for img_id, img_file in enumerate(img_files, start=1):
        img_path = os.path.join(img_dir, img_file)

        # Image dimensions
        with Image.open(img_path) as img:
            w, h = img.size

        images.append({
            "id": img_id,
            "file_name": "images/" + img_file,  # rfdetr root=split/; images under split/images/
            "width": w,
            "height": h,
        })

        # Matching label file
        stem = os.path.splitext(img_file)[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt")

        if not os.path.exists(lbl_path):
            continue  # No labels (negative sample)

        with open(lbl_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # YOLO normalized → COCO absolute [x_min, y_min, width, height]
            abs_w  = bw * w
            abs_h  = bh * h
            x_min  = (cx - bw / 2) * w
            y_min  = (cy - bh / 2) * h

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id + 1,  # COCO category_id starts at 1
                "bbox": [round(x_min, 4), round(y_min, 4), round(abs_w, 4), round(abs_h, 4)],
                "area": round(abs_w * abs_h, 4),
                "iscrowd": 0,
            })
            ann_id += 1

    coco_json = {
        "info": {"description": f"roadpoles_v1 {split} set (converted from YOLO)"},
        "licenses": [],
        "categories": CATEGORIES,
        "images": images,
        "annotations": annotations,
    }

    with open(out_path, "w") as f:
        json.dump(coco_json, f, indent=2)

    print(f"[{split:5s}] {len(images):4d} images, {len(annotations):4d} annotations → {out_path}")


def main():
    print("Converting YOLO → COCO JSON...\n")
    for split in SPLITS:
        convert_split(split)
    print("\nDone. Run train_v1.py to start training.")


if __name__ == "__main__":
    main()
