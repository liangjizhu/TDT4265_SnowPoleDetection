"""
Inference / prediction script — run a trained model on images or video.
Saves annotated outputs and YOLO-format label files for leaderboard submission.

Usage:
    python src/predict.py --model runs/train/snow_poles/weights/best.pt \
                          --source data/Poles2025/images/test \
                          --save-txt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained YOLO model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model weights")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to images/video/directory for inference")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs/predict")
    parser.add_argument("--name", type=str, default="snow_poles")
    parser.add_argument("--save-txt", action="store_true",
                        help="Save YOLO-format label txt files (for leaderboard)")
    parser.add_argument("--save-conf", action="store_true",
                        help="Include confidence scores in saved txt")
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
    )

    output_dir = Path(args.project) / args.name
    print(f"\nPredictions saved to: {output_dir}")
    if args.save_txt:
        print(f"Label files saved to: {output_dir / 'labels'}")

    return results


if __name__ == "__main__":
    main()
