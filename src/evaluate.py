"""
Evaluation script — compute Precision, Recall, mAP@50, mAP@50:95
on the validation (or test) split.

Usage:
    python src/evaluate.py --model runs/train/snow_poles/weights/best.pt \
                           --config configs/poles2025.yaml --split val
"""

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model weights (.pt)")
    parser.add_argument("--config", type=str, default="configs/road_poles_iphone.yaml",
                        help="Path to dataset YAML config")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="Which split to evaluate on")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold for evaluation")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="IoU threshold for NMS")
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)
    metrics = model.val(
        data=args.config,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        plots=True,
        save_json=True,
    )

    print("\n===== Evaluation Results =====")
    print(f"  Precision:      {metrics.box.mp:.4f}")
    print(f"  Recall:         {metrics.box.mr:.4f}")
    print(f"  mAP@50:         {metrics.box.map50:.4f}")
    print(f"  mAP@50:95:      {metrics.box.map:.4f}")
    print("==============================\n")

    return metrics


if __name__ == "__main__":
    main()
