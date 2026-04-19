"""
Training script for Snow Pole Detection using Ultralytics YOLO.

Usage:
    python src/train.py --config configs/poles2025.yaml --model yolov8n.pt --epochs 100
    python src/train.py --config configs/poles2025.yaml --model yolo11n.pt --epochs 100
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO for snow pole detection")
    parser.add_argument("--config", type=str, default="configs/road_poles_iphone.yaml",
                        help="Path to dataset YAML config")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Pretrained model name or path (e.g. yolov8n.pt, yolov8s.pt, yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device(s), e.g. '0' or '0,1' or 'cpu'")
    parser.add_argument("--project", type=str, default="runs/train",
                        help="Project directory for saving results")
    parser.add_argument("--name", type=str, default="snow_poles",
                        help="Experiment name")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Enable data augmentation")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--profile", type=str, default="default",
                        choices=["default", "anti-overfit", "finetune"],
                        help="Training profile: 'default', 'anti-overfit', or 'finetune' for transfer learning")
    parser.add_argument("--freeze", type=int, default=0,
                        help="Freeze first N layers of backbone (e.g. 10)")
    parser.add_argument("--lr0", type=float, default=None,
                        help="Override initial learning rate")
    parser.add_argument("--box", type=float, default=None,
                        help="Override box loss weight (default 7.5)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.resume:
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)

        if args.profile == "anti-overfit":
            train_kwargs = dict(
                dropout=0.15,
                weight_decay=0.001,
                label_smoothing=0.1,
                lr0=0.001,
                cos_lr=True,
                warmup_epochs=5,
                freeze=args.freeze if args.freeze > 0 else None,
                hsv_h=0.02,
                hsv_s=0.8,
                hsv_v=0.5,
                degrees=10.0,
                translate=0.2,
                scale=0.7,
                flipud=0.1,
                fliplr=0.5,
                mosaic=1.0,
                mixup=0.3,
                erasing=0.2,
                close_mosaic=20,
            )
        elif args.profile == "finetune":
            train_kwargs = dict(
                lr0=0.0005,
                lrf=0.01,
                cos_lr=True,
                warmup_epochs=3,
                freeze=args.freeze if args.freeze > 0 else None,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=5.0,
                translate=0.15,
                scale=0.5,
                flipud=0.0,
                fliplr=0.5,
                mosaic=1.0,
                mixup=0.15,
                close_mosaic=15,
            )
        else:
            train_kwargs = dict(
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=5.0,
                translate=0.1,
                scale=0.5,
                flipud=0.0,
                fliplr=0.5,
                mosaic=1.0,
                mixup=0.1,
                freeze=args.freeze if args.freeze > 0 else None,
            )

        if args.lr0 is not None:
            train_kwargs["lr0"] = args.lr0
        if args.box is not None:
            train_kwargs["box"] = args.box

        results = model.train(
            data=args.config,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            patience=args.patience,
            save=True,
            save_period=10,
            plots=True,
            verbose=True,
            **train_kwargs,
        )

    print("\nTraining complete!")
    print(f"Best model saved at: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    return results


if __name__ == "__main__":
    main()
