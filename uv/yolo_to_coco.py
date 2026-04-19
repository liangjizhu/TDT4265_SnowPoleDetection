"""
yolo_to_coco.py
将 roadpoles_v1 的 YOLO 格式标注转换为 COCO JSON 格式。
RF-DETR 需要每个子集目录下存在 _annotations.coco.json。

用法：
    python yolo_to_coco.py

输出：
    roadpoles_v1/train/_annotations.coco.json
    roadpoles_v1/valid/_annotations.coco.json
    roadpoles_v1/test/_annotations.coco.json
"""

import os
import json
from PIL import Image

# ── 配置 ───────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "roadpoles_v1")

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

        # 获取图片尺寸
        with Image.open(img_path) as img:
            w, h = img.size

        images.append({
            "id": img_id,
            "file_name": "images/" + img_file,   # rfdetr root=train/, 图片在 train/images/
            "width": w,
            "height": h,
        })

        # 读取对应标注文件
        stem = os.path.splitext(img_file)[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt")

        if not os.path.exists(lbl_path):
            continue  # 无标注（负样本），跳过

        with open(lbl_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # YOLO 归一化 → COCO 绝对坐标 [x_min, y_min, width, height]
            abs_w  = bw * w
            abs_h  = bh * h
            x_min  = (cx - bw / 2) * w
            y_min  = (cy - bh / 2) * h

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id + 1,  # COCO category_id 从 1 开始
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
    print("\n完成！可以运行 train.py 开始训练。")


if __name__ == "__main__":
    main()
