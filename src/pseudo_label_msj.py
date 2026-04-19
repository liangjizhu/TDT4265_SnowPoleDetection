"""
Pseudo-label RoadPoles-MSJ (unlabeled) with a trained teacher for semi-supervised training.

Typical workflow:
  1. Train or pick your best roadpoles_v1 (+ optional iPhone fine-tune) checkpoint as teacher.
  2. Run this script with a conservative --conf (e.g. 0.35–0.5) to reduce noisy boxes.
  3. Link into the YOLO tree: python src/link_msj_pseudo_into_dataset.py
     (uses data/msj_pseudo_manifest.csv + data/pseudo_msj_labels -> data/combined_full/...).
  4. Merge iPhone + v1 train pairs into the same images/train and labels/train (if not already).
  5. Put roadpoles_v1 val images/labels under images/val and labels/val; train with configs/combined_full.yaml.

Ultralytics expects parallel images/ and labels/ trees; label lines are class xc yc w h (normalized).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description="Pseudo-label MSJ images for YOLO training")
    p.add_argument(
        "--teacher",
        type=str,
        required=True,
        help="Teacher weights (e.g. runs/detect/runs/train/snow_poles_v1/weights/best.pt)",
    )
    p.add_argument(
        "--source",
        type=str,
        default="data/Poles2025/RoadPoles-MSJ",
        help="Root folder of MSJ images (searched recursively if --recursive)",
    )
    p.add_argument(
        "--labels-out",
        type=str,
        default="data/pseudo_msj_labels",
        help="Directory to write YOLO label .txt files (created if missing)",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Min confidence to keep a box (higher = fewer but cleaner pseudo-labels)",
    )
    p.add_argument("--iou", type=float, default=0.7, help="Teacher NMS IoU")
    p.add_argument("--device", type=str, default="0")
    p.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search images under subfolders (default: on)",
    )
    p.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only immediate files under --source",
    )
    p.add_argument(
        "--max-width",
        type=float,
        default=None,
        help="Drop boxes with normalized width above this (optional, e.g. 0.05 for poles)",
    )
    p.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional CSV: label_stem,src_image_path (use to symlink/copy images to stem.ext for YOLO)",
    )
    return p.parse_args()


def label_stem_for_image(image_path: Path, root: Path) -> str:
    """Unique stem (no extension) for nested folders — training image must use this same stem."""
    try:
        rel = image_path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = Path(image_path.name)
    return rel.with_suffix("").as_posix().replace("/", "__").replace("\\", "__")


def collect_images(root: Path, recursive: bool) -> list[Path]:
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")
    out: list[Path] = []
    it = root.rglob("*") if recursive else root.iterdir()
    for p in it:
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            out.append(p)
    return sorted(out)


def main():
    args = parse_args()
    root = Path(args.source)
    labels_dir = Path(args.labels_out)
    labels_dir.mkdir(parents=True, exist_ok=True)

    recursive = args.recursive and not args.no_recursive
    images = collect_images(root, recursive)
    if not images:
        raise SystemExit(f"No images under {root} (recursive={recursive})")

    model = YOLO(args.teacher)
    n_kept = 0
    n_empty = 0
    n_dropped_wide = 0
    manifest_lines: list[str] = []

    # Stream one image at a time so we control output naming for nested paths.
    for i, img_path in enumerate(images):
        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
            save=False,
        )
        r = results[0]
        h, w = r.orig_shape
        lines: list[str] = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            if args.max_width is not None and bw > args.max_width:
                n_dropped_wide += 1
                continue
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            cls = int(box.cls[0])
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        stem = label_stem_for_image(img_path, root)
        out_path = labels_dir / f"{stem}.txt"
        out_path.write_text("".join(lines))
        if args.manifest is not None:
            manifest_lines.append(f"{stem},{img_path.resolve().as_posix()}\n")
        if lines:
            n_kept += 1
        else:
            n_empty += 1

        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(images)} ...")

    if args.manifest is not None:
        mp = Path(args.manifest)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text("label_stem,src_image\n" + "".join(manifest_lines))
        print(f"Wrote manifest: {mp}")

    print(
        f"Wrote {len(images)} label files to {labels_dir}\n"
        f"  images with >=1 box (after filters): {n_kept}\n"
        f"  empty label files: {n_empty}\n"
        f"  boxes dropped by --max-width: {n_dropped_wide}"
    )
    print(
        "\nYOLO pairs labels by stem: for each MSJ source image, copy or symlink it to "
        "images/train/<label_stem>.<ext> matching the .txt basename (use --manifest to automate)."
    )


if __name__ == "__main__":
    main()
