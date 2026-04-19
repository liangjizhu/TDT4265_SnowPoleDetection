"""
rf_detr_workflow.py
RF-DETR Snow Pole Detection - Complete Workflow
TDT4265 Project

This file provides a complete end-to-end workflow for training and evaluating
RF-DETR on the snow pole detection dataset.

Steps:
    1. Convert YOLO annotations to COCO format  → yolo_to_coco.py
    2. (Optional) Merge iPhone dataset           → merge_dataset.py
    3. Train RF-DETR-B model                     → train.py
    4. Run inference on test set                 → predict_v1test.py

Usage:
    python rf_detr_workflow.py [--step STEP]

    --step all       Run all steps (default)
    --step convert   Only convert YOLO → COCO
    --step train     Only train
    --step predict   Only predict on test set
    --step eval      Only evaluate predictions
"""

import os
import sys
import argparse
import json
import zipfile
from pathlib import Path

# Paths: this repo — src/rfdetr → repo root is parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATASET = _REPO_ROOT / "data" / "Poles2025" / "roadpoles_v1"
DATASET_DIR = str(_DATASET)
OUTPUT_DIR = str(_REPO_ROOT / "runs" / "rfdetr" / "workflow")
WEIGHTS_PATH = str(Path(OUTPUT_DIR) / "checkpoint_best_ema.pth")
TEST_IMG_DIR = str(_DATASET / "test" / "images")
PRED_DIR = str(_REPO_ROOT / "runs" / "rfdetr" / "predictions_v1test")
ZIP_PATH = str(_REPO_ROOT / "submissions" / "submission_rfdetr_workflow.zip")

# Training hyperparameters
BATCH_SIZE          = 2
GRAD_ACCUM_STEPS    = 8
NUM_EPOCHS          = 100
LEARNING_RATE       = 1e-4
NUM_CLASSES         = 1        # only 'pole'
RESOLUTION          = 560
CONFIDENCE_THRESHOLD = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Convert YOLO → COCO JSON
# ─────────────────────────────────────────────────────────────────────────────
def step_convert():
    """Convert YOLO format annotations to COCO JSON for RF-DETR."""
    from PIL import Image

    CATEGORIES = [{"id": 1, "name": "pole", "supercategory": "object"}]
    SPLITS = ["train", "valid", "test"]

    def convert_split(split: str):
        img_dir  = os.path.join(DATASET_DIR, split, "images")
        lbl_dir  = os.path.join(DATASET_DIR, split, "labels")
        out_path = os.path.join(DATASET_DIR, split, "_annotations.coco.json")

        images, annotations = [], []
        ann_id = 1

        img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        for img_id, img_file in enumerate(img_files, start=1):
            img_path = os.path.join(img_dir, img_file)
            with Image.open(img_path) as img:
                w, h = img.size

            images.append({
                "id": img_id,
                "file_name": "images/" + img_file,
                "width": w,
                "height": h,
            })

            stem     = os.path.splitext(img_file)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.exists(lbl_path):
                continue

            with open(lbl_path) as f:
                lines = [l.strip() for l in f if l.strip()]

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, bw, bh = (
                    int(parts[0]), float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4])
                )
                abs_w = bw * w
                abs_h = bh * h
                x_min = (cx - bw / 2) * w
                y_min = (cy - bh / 2) * h

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id + 1,
                    "bbox": [round(x_min, 4), round(y_min, 4),
                             round(abs_w, 4), round(abs_h, 4)],
                    "area": round(abs_w * abs_h, 4),
                    "iscrowd": 0,
                })
                ann_id += 1

        coco_json = {
            "info": {"description": f"roadpoles_v1 {split} set (YOLO→COCO)"},
            "licenses": [],
            "categories": CATEGORIES,
            "images": images,
            "annotations": annotations,
        }
        with open(out_path, "w") as f:
            json.dump(coco_json, f, indent=2)
        print(f"[{split:5s}] {len(images):4d} images, "
              f"{len(annotations):4d} annotations → {out_path}")

    print("=" * 60)
    print("Step 1: Converting YOLO → COCO JSON")
    print("=" * 60)
    for split in SPLITS:
        convert_split(split)
    print("Conversion complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Train RF-DETR-B
# ─────────────────────────────────────────────────────────────────────────────
def step_train():
    """Train RF-DETR-B on roadpoles_v1 dataset."""
    import torch
    from rfdetr import RFDETRBase

    print("=" * 60)
    print("Step 2: Training RF-DETR-B")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: No GPU detected. Training on CPU will be very slow.")

    # Verify COCO JSON files exist
    for split in ["train", "valid", "test"]:
        ann_path = os.path.join(DATASET_DIR, split, "_annotations.coco.json")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(
                f"Missing {ann_path}\nPlease run: python src/rfdetr/rf_detr_workflow.py --step convert"
            )

    print(f"\nDataset:    {DATASET_DIR}")
    print(f"Output:     {OUTPUT_DIR}")
    print(f"Epochs:     {NUM_EPOCHS}")
    print(f"Batch:      {BATCH_SIZE} × grad_accum {GRAD_ACCUM_STEPS} = effective {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"LR:         {LEARNING_RATE}")
    print(f"Resolution: {RESOLUTION}\n")

    def on_epoch_end(metrics: dict):
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
        early_stopping_patience=10,
        early_stopping_use_ema=True,
        tensorboard=True,
    )

    print("\nTraining complete!")
    print(f"Weights saved to: {OUTPUT_DIR}")
    print(f"TensorBoard: tensorboard --logdir {OUTPUT_DIR}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Predict on test set
# ─────────────────────────────────────────────────────────────────────────────
def step_predict():
    """Run inference on roadpoles_v1/test and produce YOLO-format .txt files."""
    from PIL import Image
    from rfdetr import RFDETRBase

    print("=" * 60)
    print("Step 3: Predicting on test set")
    print("=" * 60)

    os.makedirs(PRED_DIR, exist_ok=True)

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Weights not found: {WEIGHTS_PATH}\n"
            "Please run training first: python src/rfdetr/rf_detr_workflow.py --step train"
        )

    print(f"Loading weights: {WEIGHTS_PATH}")
    model = RFDETRBase(num_classes=NUM_CLASSES, pretrain_weights=WEIGHTS_PATH)

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = sorted([
        p for p in Path(TEST_IMG_DIR).iterdir()
        if p.suffix.lower() in img_extensions
    ])
    print(f"Found {len(img_paths)} test images in roadpoles_v1/test/images\n")

    for img_path in img_paths:
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        detections = model.predict(str(img_path), threshold=CONFIDENCE_THRESHOLD)

        txt_name = img_path.stem + ".txt"
        txt_path = os.path.join(PRED_DIR, txt_name)

        lines = []
        if detections and len(detections) > 0:
            for i in range(len(detections.xyxy)):
                x1, y1, x2, y2 = detections.xyxy[i]
                conf   = float(detections.confidence[i])
                cls_id = int(detections.class_id[i]) - 1  # 1-indexed → 0-indexed

                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                w  = (x2 - x1) / img_w
                h  = (y2 - y1) / img_h

                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}")

        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

        print(f"  {img_path.name}: {len(lines)} detections → {txt_name}")

    # Pack into zip
    print(f"\nPacking predictions to: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for txt_file in sorted(Path(PRED_DIR).glob("*.txt")):
            zf.write(txt_file, txt_file.name)

    zip_size = os.path.getsize(ZIP_PATH) / 1024
    print(f"Done! ZIP size: {zip_size:.1f} KB  →  {ZIP_PATH}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Evaluate predictions (mAP on test set)
# ─────────────────────────────────────────────────────────────────────────────
def step_eval():
    """Evaluate YOLO-format predictions against ground-truth labels."""
    print("=" * 60)
    print("Step 4: Evaluating predictions")
    print("=" * 60)

    gt_lbl_dir = os.path.join(DATASET_DIR, "test", "labels")
    pred_dir   = PRED_DIR

    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(
            f"Predictions directory not found: {pred_dir}\n"
            "Please run: python src/rfdetr/rf_detr_workflow.py --step predict"
        )

    iou_threshold = 0.5
    tp_total = fp_total = fn_total = 0

    pred_files = sorted(Path(pred_dir).glob("*.txt"))
    print(f"Evaluating {len(pred_files)} prediction files at IoU={iou_threshold}\n")

    def compute_iou(boxA, boxB):
        """Compute IoU between two [cx,cy,w,h] normalized boxes."""
        ax1 = boxA[0] - boxA[2] / 2; ay1 = boxA[1] - boxA[3] / 2
        ax2 = boxA[0] + boxA[2] / 2; ay2 = boxA[1] + boxA[3] / 2
        bx1 = boxB[0] - boxB[2] / 2; by1 = boxB[1] - boxB[3] / 2
        bx2 = boxB[0] + boxB[2] / 2; by2 = boxB[1] + boxB[3] / 2
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        areaA = (ax2 - ax1) * (ay2 - ay1)
        areaB = (bx2 - bx1) * (by2 - by1)
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0.0

    for pred_file in pred_files:
        gt_file = os.path.join(gt_lbl_dir, pred_file.name)

        # Load predictions (class cx cy w h conf)
        preds = []
        with open(pred_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    preds.append([float(x) for x in parts[1:5]])

        # Load ground truth (class cx cy w h)
        gts = []
        if os.path.exists(gt_file):
            with open(gt_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        gts.append([float(x) for x in parts[1:5]])

        matched_gt = set()
        tp = fp = 0
        for pred in preds:
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gts):
                if j in matched_gt:
                    continue
                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_j)
            else:
                fp += 1
        fn = len(gts) - len(matched_gt)

        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall    = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    print(f"Results at IoU={iou_threshold}:")
    print(f"  TP={tp_total}  FP={fp_total}  FN={fn_total}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="RF-DETR Snow Pole Detection - Complete Workflow"
    )
    parser.add_argument(
        "--step",
        choices=["all", "convert", "train", "predict", "eval"],
        default="all",
        help="Which step to run (default: all)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  RF-DETR Snow Pole Detection Workflow")
    print("  TDT4265 - Computer Vision")
    print("=" * 60 + "\n")

    if args.step in ("all", "convert"):
        step_convert()
    if args.step in ("all", "train"):
        step_train()
    if args.step in ("all", "predict"):
        step_predict()
    if args.step in ("all", "eval"):
        step_eval()

    print("All requested steps completed successfully!")


if __name__ == "__main__":
    main()
