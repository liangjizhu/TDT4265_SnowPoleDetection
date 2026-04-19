"""
Compare v1-val metrics from two Ultralytics runs: iPhone+v1 only vs +MSJ pseudo.

Usage (after both trainings finished):
  python src/summarize_combined_ablation.py

Override CSV paths:
  python src/summarize_combined_ablation.py \\
    --no-msj runs/detect/runs/train/ablation_iv1_no_msj/results.csv \\
    --with-msj runs/detect/runs/train/combined_full_msj952/results.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def best_by_map5095(csv_path: Path) -> dict[str, str]:
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")
    key = "metrics/mAP50-95(B)"
    return max(rows, key=lambda r: float(r[key]))


def parse_args():
    p = argparse.ArgumentParser(description="Summarize combined-dataset ablation from results.csv")
    root = Path(__file__).resolve().parents[1]
    p.add_argument(
        "--no-msj",
        type=Path,
        default=root / "runs/detect/runs/train/ablation_iv1_no_msj/results.csv",
        help="Train run: configs/combined.yaml (1264 train)",
    )
    p.add_argument(
        "--with-msj",
        type=Path,
        default=root / "runs/detect/runs/train/combined_full_msj952/results.csv",
        help="Train run: configs/combined_full.yaml (2216 train)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    sections = []
    for label, path in (
        ("iPhone + v1 (no MSJ)", args.no_msj),
        ("iPhone + v1 + MSJ pseudo", args.with_msj),
    ):
        try:
            b = best_by_map5095(path)
            sections.append(
                (label, path, b["epoch"], b["metrics/mAP50(B)"], b["metrics/mAP50-95(B)"], b["metrics/precision(B)"], b["metrics/recall(B)"])
            )
        except FileNotFoundError:
            sections.append((label, path, None, None, None, None, None))

    print("Combined-dataset ablation (best epoch by val mAP50-95 on roadpoles_v1 val)\n")
    print(f"{'Setting':<28} {'epoch':>6} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8}")
    print("-" * 78)
    for label, path, ep, m50, m95, pr, re in sections:
        if ep is None:
            print(f"{label:<28}  (missing {path})")
        else:
            print(
                f"{label:<28} {int(float(ep)):>6} {float(m50):>8.4f} {float(m95):>10.4f} {float(pr):>8.4f} {float(re):>8.4f}"
            )
    print("\nTrain sizes: no-MSJ = 1264 (configs/combined.yaml); +MSJ = 2216 (configs/combined_full.yaml).")


if __name__ == "__main__":
    main()
