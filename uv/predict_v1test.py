"""
predict_v1test.py
使用最佳 EMA 权重对 roadpoles_v1/test 测试集进行推理，
输出 YOLO 格式的 .txt 预测文件，并打包成 submission_v1test.zip。

格式：每行 <class> <cx> <cy> <w> <h> <confidence>
（坐标均为相对图像宽高的归一化值）

用法（在激活 .venv 后）：
    python predict_v1test.py
"""

import os
import zipfile
from pathlib import Path
from PIL import Image
import torch
from rfdetr import RFDETRBase

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_IMG_DIR   = os.path.join(BASE_DIR, "roadpoles_v1", "test", "images")
WEIGHTS_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "checkpoint_best_ema.pth")
OUTPUT_TXT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions_v1test")
ZIP_PATH       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission_v1test4.zip")

# ── 推理参数 ──────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.01  # 置信度阈值
NUM_CLASSES = 1


def main():
    os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)

    # 加载模型
    print(f"加载权重: {WEIGHTS_PATH}")
    model = RFDETRBase(num_classes=NUM_CLASSES, pretrain_weights=WEIGHTS_PATH)

    # 获取所有测试图片
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = sorted([
        p for p in Path(TEST_IMG_DIR).iterdir()
        if p.suffix.lower() in img_extensions
    ])
    print(f"找到 {len(img_paths)} 张测试图片（来自 roadpoles_v1/test/images）")

    for img_path in img_paths:
        # 读取图片尺寸
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        # 推理
        detections = model.predict(str(img_path), threshold=CONFIDENCE_THRESHOLD)

        # 写入 YOLO 格式 txt
        txt_name = img_path.stem + ".txt"
        txt_path = os.path.join(OUTPUT_TXT_DIR, txt_name)

        lines = []
        if detections and len(detections) > 0:
            # detections 是 supervision.Detections 对象
            # xyxy: [x1, y1, x2, y2], confidence, class_id
            for i in range(len(detections.xyxy)):
                x1, y1, x2, y2 = detections.xyxy[i]
                conf = float(detections.confidence[i])
                cls_id = int(detections.class_id[i]) - 1  # RF-DETR class_id 从1开始，转为从0开始

                # 转换为归一化的 cx, cy, w, h
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                w  = (x2 - x1) / img_w
                h  = (y2 - y1) / img_h

                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}")

        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

        print(f"  {img_path.name}: {len(lines)} 个检测框 → {txt_name}")

    # 打包成 zip
    print(f"\n打包预测结果到: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for txt_file in sorted(Path(OUTPUT_TXT_DIR).glob("*.txt")):
            zf.write(txt_file, txt_file.name)

    zip_size = os.path.getsize(ZIP_PATH) / 1024
    print(f"完成！ZIP 文件大小: {zip_size:.1f} KB")
    print(f"ZIP 路径: {ZIP_PATH}")


if __name__ == "__main__":
    main()
