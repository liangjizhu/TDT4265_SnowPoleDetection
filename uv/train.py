"""
train.py
使用 RF-DETR-B 在 roadpoles_v1（含合并的 iPhone 数据）上训练路杆检测模型。
目标：mAP@0.5:0.95 >= 0.70

用法（在激活 .venv 后）：
    python train.py

前置步骤：
    1. 运行 merge_dataset.py  → 将 iPhone 30% 数据合并进训练集
    2. 运行 yolo_to_coco.py   → 将 YOLO 标注转换为 COCO JSON 格式
    3. 运行 train.py          → 开始训练
"""

import os
import torch
from rfdetr import RFDETRBase

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "roadpoles_v1")
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

# ── 训练超参数 ─────────────────────────────────────────────────────────────────
# RTX 3050 4GB 显存：batch_size=2，梯度累积 8 步 = 等效 batch 16
BATCH_SIZE       = 2
GRAD_ACCUM_STEPS = 8
NUM_EPOCHS       = 100
LEARNING_RATE    = 1e-4
NUM_CLASSES      = 1        # 只有 pole
RESOLUTION       = 560      # 提高分辨率：560→672（必须是 patch_size×num_windows=56 的倍数）


def on_epoch_end(metrics: dict):
    """每个 epoch 结束时打印汇总指标"""
    epoch      = metrics.get("epoch", "?")
    train_loss = metrics.get("train_loss", None)

    # mAP 存储在 test_coco_eval_bbox 列表中
    # [0]=mAP@0.5:0.95, [1]=mAP@0.5, [2]=mAP@0.75
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


# ── Windows 多进程必须放在 if __name__ == '__main__' 块中 ──────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 环境检查
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("警告：未检测到 GPU，将使用 CPU 训练（速度极慢）")

    # 检查 COCO JSON 是否存在
    for split in ["train", "valid", "test"]:
        ann_path = os.path.join(DATASET_DIR, split, "_annotations.coco.json")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(
                f"缺少 {ann_path}\n请先运行: python yolo_to_coco.py"
            )

    print(f"\n数据集路径: {DATASET_DIR}")
    print(f"输出路径:   {OUTPUT_DIR}")
    print(f"训练轮数:   {NUM_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}  (梯度累积 {GRAD_ACCUM_STEPS} 步，等效 {BATCH_SIZE * GRAD_ACCUM_STEPS})")
    print(f"学习率:     {LEARNING_RATE}\n")

    # 初始化模型，注册 epoch 结束回调
    model = RFDETRBase(num_classes=NUM_CLASSES, resolution=RESOLUTION)
    model.callbacks["on_fit_epoch_end"].append(on_epoch_end)

    # 开始训练
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

    print("\n训练完成！")
    print(f"模型权重和日志保存在: {OUTPUT_DIR}")
    print("查看训练曲线：tensorboard --logdir runs")
