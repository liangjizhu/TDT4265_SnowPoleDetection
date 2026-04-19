"""
train_v1.py
Train RF-DETR-B on roadpoles_v1 for pole detection (optionally after merging iPhone data).
Target: mAP@0.5:0.95 >= 0.70

Usage (with your venv activated):
    python train_v1.py

Prerequisites:
    1. (Optional) merge_dataset.py — merge ~30% iPhone images into roadpoles_v1/train
    2. yolo_to_coco.py — convert YOLO labels to COCO JSON
    3. train_v1.py — start training
"""

import os
from pathlib import Path
import torch
from rfdetr import RFDETRBase

# Paths (this repo)
_REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = str(_REPO_ROOT / "data" / "Poles2025" / "roadpoles_v1")
OUTPUT_DIR = str(_REPO_ROOT / "runs" / "rfdetr" / "v1_only")

# Training hyperparameters
# RTX 3050 4GB: batch_size=2, grad_accum 8 → effective batch 16
BATCH_SIZE       = 2
GRAD_ACCUM_STEPS = 8
NUM_EPOCHS       = 100
LEARNING_RATE    = 1e-4
NUM_CLASSES      = 1        # single class: pole
RESOLUTION       = 560      # e.g. 560→672 (must be multiple of patch_size×num_windows=56)


def on_epoch_end(metrics: dict):
    """Print summary metrics at the end of each epoch."""
    epoch      = metrics.get("epoch", "?")
    train_loss = metrics.get("train_loss", None)

    # mAP lives in test_coco_eval_bbox: [0]=mAP@0.5:0.95, [1]=mAP@0.5, [2]=mAP@0.75
    coco_bbox     = metrics.get("test_coco_eval_bbox", None)
    ema_coco_bbox = metrics.get("ema_test_coco_eval_bbox", None)

    map_5095 = coco_bbox[0]     if coco_bbox     else None
    map_50   = coco_bbox[1]     if coco_bbox     else None
    ema_5095 = ema_coco_bbox[0] if ema_coco_bbox else None
    ema_50   = ema_coco_bbox[1] if ema_coco_bbox else None

    print("\n" + "="*70)
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
    print("="*70 + "\n", flush=True)


# Windows multiprocessing: keep training entry under if __name__ == "__main__"
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Environment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: No GPU detected. CPU training will be very slow.")

    # Ensure COCO JSON exists
    for split in ["train", "valid", "test"]:
        ann_path = os.path.join(DATASET_DIR, split, "_annotations.coco.json")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(
                f"Missing {ann_path}\nRun first: python src/rfdetr/yolo_to_coco.py"
            )

    print(f"\nDataset:    {DATASET_DIR}")
    print(f"Output:     {OUTPUT_DIR}")
    print(f"Epochs:     {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}  (grad accum {GRAD_ACCUM_STEPS} steps → effective {BATCH_SIZE * GRAD_ACCUM_STEPS})")
    print(f"LR:         {LEARNING_RATE}\n")

    # Model + epoch-end callback
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

    print("\nTraining finished.")
    print(f"Weights and logs: {OUTPUT_DIR}")
    print(f"TensorBoard: tensorboard --logdir {OUTPUT_DIR}")
