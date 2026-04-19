"""
Wire pseudo_label_msj.py outputs into a YOLO dataset (images/train + labels/train).

Expects:
  - Manifest CSV from pseudo_label_msj.py --manifest (header: label_stem,src_image)
  - Matching .txt files in --labels-dir

Creates symlinks (default) or copies so stems match for training with configs/combined_full.yaml.

You must still add iPhone + roadpoles_v1 train images/labels into the same folders (or use
your existing combined layout and only add MSJ here).
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Link MSJ pseudo labels into a YOLO dataset tree")
    p.add_argument("--manifest", type=str, default="data/msj_pseudo_manifest.csv")
    p.add_argument(
        "--labels-dir",
        type=str,
        default="data/pseudo_msj_labels",
        help="Directory with <label_stem>.txt from pseudo_label_msj.py",
    )
    p.add_argument(
        "--dataset-root",
        type=str,
        default="data/combined_full",
        help="Dataset root containing images/train and labels/train",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinking (useful if you move the repo)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    manifest = Path(args.manifest)
    labels_dir = Path(args.labels_dir)
    root = Path(args.dataset_root)
    img_train = root / "images" / "train"
    lbl_train = root / "labels" / "train"
    img_train.mkdir(parents=True, exist_ok=True)
    lbl_train.mkdir(parents=True, exist_ok=True)

    if not manifest.is_file():
        raise SystemExit(f"Missing manifest: {manifest}")
    n_img = 0
    n_lbl = 0
    with manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = row["label_stem"].strip()
            src = Path(row["src_image"].strip())
            txt = labels_dir / f"{stem}.txt"
            if not src.is_file():
                print(f"skip (missing image): {src}")
                continue
            if not txt.is_file():
                print(f"skip (missing label): {txt}")
                continue
            ext = src.suffix
            dst_img = img_train / f"{stem}{ext}"
            dst_lbl = lbl_train / f"{stem}.txt"
            if args.copy:
                shutil.copy2(src, dst_img)
                shutil.copy2(txt, dst_lbl)
            else:
                for dst in (dst_img, dst_lbl):
                    if dst.is_symlink() or dst.exists():
                        dst.unlink()
                dst_img.symlink_to(src.resolve(), target_is_directory=False)
                dst_lbl.symlink_to(txt.resolve(), target_is_directory=False)
            n_img += 1
            n_lbl += 1

    print(
        f"Linked {n_img} image + label pairs under {root}\n"
        f"  images -> {img_train}\n"
        f"  labels -> {lbl_train}"
    )
    if n_img == 0:
        print("No pairs linked. Check manifest paths and --labels-dir.")


if __name__ == "__main__":
    main()
