"""
3-model (or 4-model) ensemble on roadpoles_v1 test + optional width/height box tuning.

Caches raw boxes once, then sweeps geometry / fusion without re-inference.

Fusion (--fuse): wbf (default), nmw (non-maximum weighted), nms (standard NMS on
concatenated experts), soft_nms. Use --shrink-w 1.0 --scale-h 1.0 to disable width/height tuning.

Examples:
  python src/ensemble_wbf_v1.py --mode single --shrink-w 0.91 --out-dir submissions/wbf91
  python src/ensemble_wbf_v1.py --mode single --fuse nms --shrink-w 1.0 --wbf-iou 0.5 --out-dir submissions/nms_plain
  python src/ensemble_wbf_v1.py --mode sweep --submissions-dir submissions
  python src/ensemble_wbf_v1.py --mode multiscale-sweep --submissions-dir submissions
  python src/ensemble_wbf_v1.py --mode multiscale-sweep --ms-shrink 0.904 0.907 0.910 --ms-skip 0.13 --ms-wiou 0.50
  python src/ensemble_wbf_v1.py --mode multiscale-sweep --fuse nmw --ms-shrink 0.904 --ms-skip 0.14 --ms-wiou 0.52
  python src/ensemble_wbf_v1.py --mode multiscale-sweep --ms-sc-preset d75 flat --ms-shrink 0.902 0.904 0.906 --ms-skip 0.14 --ms-wiou 0.52
  python src/ensemble_wbf_v1.py --mode multiscale-wbf12 --submissions-dir submissions
"""

from __future__ import annotations

import argparse
import glob
import os
import zipfile
from pathlib import Path

import numpy as np
from ensemble_boxes import (
    non_maximum_weighted,
    nms,
    soft_nms,
    weighted_boxes_fusion,
)
from PIL import Image
from ultralytics import YOLO


DEFAULT_MODELS = [
    "runs/detect/runs/train/snow_poles_v1/weights/best.pt",
    "runs/detect/runs/train/yolov8s_finetune_v1/weights/best.pt",
    "runs/detect/runs/train/yolov8n_v1_640_200ep/weights/best.pt",
]
DEFAULT_WEIGHTS = [1.0, 1.0, 0.8]

DEFAULT_MODELS_WBF12 = DEFAULT_MODELS + [
    "runs/detect/runs/train/yolo11n_v1_640/weights/best.pt",
]
DEFAULT_WEIGHTS_WBF12 = [1.0, 1.0, 0.8, 0.65]

# Per-inference-size multipliers (576, 640, 704); fused with per-model WBF weights.
EXPERT_SCALE_PRESETS = {
    "d75": [0.75, 1.0, 0.75],
    "flat": [1.0, 1.0, 1.0],
    "em640": [0.65, 1.15, 0.65],
    "em704": [0.55, 0.9, 1.2],
}

# Zip filename fragment: v1_{abbr}{n_experts}_ms / v1_{abbr}3_sw...
FUSE_ZIP_ABBR = {"wbf": "wbf", "nmw": "nmw", "nms": "nms", "soft_nms": "snm"}


def parse_args():
    p = argparse.ArgumentParser(description="Ensemble fusion + box tuning for v1 test")
    p.add_argument(
        "--mode",
        choices=["single", "sweep", "multiscale-sweep", "multiscale-wbf12"],
        default="single",
    )
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--weights", nargs="+", type=float, default=DEFAULT_WEIGHTS)
    p.add_argument("--source", type=str, default="data/Poles2025/roadpoles_v1/test/images")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument(
        "--fuse",
        choices=["wbf", "nmw", "nms", "soft_nms"],
        default="wbf",
        help="Box fusion: WBF, NMW (ensemble_boxes), or NMS / Soft-NMS on concatenated experts",
    )
    p.add_argument(
        "--wbf-iou",
        type=float,
        default=0.5,
        help="Fusion IoU: WBF/NMW iou_thr, or NMS/Soft-NMS IoU threshold",
    )
    p.add_argument(
        "--skip-box",
        type=float,
        default=0.1,
        help="WBF/NMW skip_box_thr only (ignored for nms / soft_nms)",
    )
    p.add_argument(
        "--soft-sigma",
        type=float,
        default=0.5,
        help="soft_nms Gaussian sigma (ensemble_boxes)",
    )
    p.add_argument(
        "--soft-score-thresh",
        type=float,
        default=0.001,
        help="soft_nms score cutoff after decay",
    )
    p.add_argument(
        "--soft-method",
        type=int,
        choices=[1, 2],
        default=2,
        help="soft_nms: 1=linear, 2=Gaussian",
    )
    p.add_argument("--shrink-w", type=float, default=0.91)
    p.add_argument("--scale-h", type=float, default=1.0)
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--min-score", type=float, default=0.05)
    p.add_argument("--submissions-dir", type=str, default="submissions")
    p.add_argument(
        "--imgszs",
        nargs="+",
        type=int,
        default=[576, 640, 704],
        help="For multiscale-sweep: inference sizes per model (same order for each model)",
    )
    p.add_argument(
        "--scale-weights",
        nargs="+",
        type=float,
        default=None,
        help="Weights per --imgszs (default: emphasize center size, len must match imgszs)",
    )
    p.add_argument(
        "--ms-shrink",
        nargs="+",
        type=float,
        default=None,
        help="multiscale-sweep: width shrink α values (default: grid below 0.910 .. 0.912, sk/wi from --ms-skip/--ms-wiou)",
    )
    p.add_argument(
        "--ms-skip",
        nargs="+",
        type=float,
        default=[0.13],
        help="multiscale: skip_box_thr for wbf/nmw only (ignored for nms/soft_nms)",
    )
    p.add_argument(
        "--ms-wiou",
        nargs="+",
        type=float,
        default=[0.50],
        help="multiscale: fusion IoU grid (all --fuse modes)",
    )
    p.add_argument(
        "--ms-sc-preset",
        nargs="+",
        default=["d75"],
        choices=list(EXPERT_SCALE_PRESETS.keys()),
        help="multiscale-*: expert scale-weight presets (576/640/704); no extra inference vs other presets",
    )
    return p.parse_args()


def cache_predictions(models, test_images, imgsz, conf, iou):
    """List of per-image: list of models, each list of (x1,y1,x2,y2,conf,cls) in pixel coords."""
    loaded = [YOLO(m) for m in models]
    cache = []
    for img_path in test_images:
        img = Image.open(img_path)
        w, h = img.size
        basename = Path(img_path).stem
        per_model = []
        for model in loaded:
            results = model.predict(img_path, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
            preds = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    preds.append(
                        (float(x1), float(y1), float(x2), float(y2), float(box.conf[0]), int(box.cls[0]))
                    )
            per_model.append(preds)
        cache.append({"basename": basename, "w": w, "h": h, "per_model": per_model})
    return cache


def cache_predictions_multiscale(models, test_images, imgszs, conf, iou):
    """Each image: experts = flat list len(models)*len(imgszs), each item list of xyxy preds."""
    loaded = [YOLO(m) for m in models]
    cache = []
    for img_path in test_images:
        img = Image.open(img_path)
        w, h = img.size
        basename = Path(img_path).stem
        experts = []
        for model in loaded:
            for sz in imgszs:
                results = model.predict(img_path, imgsz=sz, conf=conf, iou=iou, verbose=False)
                preds = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        preds.append(
                            (float(x1), float(y1), float(x2), float(y2), float(box.conf[0]), int(box.cls[0]))
                        )
                experts.append(preds)
        cache.append({"basename": basename, "w": w, "h": h, "experts": experts})
    return cache


def build_expert_lists(entry, shrink_w: float, scale_h: float):
    """Normalized xyxy lists per expert (same layout as ensemble_boxes WBF/NMS)."""
    preds_lists = entry.get("experts") or entry["per_model"]
    w, h = entry["w"], entry["h"]
    all_boxes, all_scores, all_labels = [], [], []

    for preds in preds_lists:
        boxes_norm, scores, labels = [], [], []
        for x1, y1, x2, y2, conf, cls in preds:
            x1n, x2n = x1 / w, x2 / w
            y1n, y2n = y1 / h, y2 / h
            cx = (x1n + x2n) / 2
            bw = (x2n - x1n) * shrink_w
            cy = (y1n + y2n) / 2
            bh = (y2n - y1n) * scale_h
            boxes_norm.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            scores.append(conf)
            labels.append(cls)
        all_boxes.append(boxes_norm if boxes_norm else [[0, 0, 0, 0]])
        all_scores.append(scores if scores else [0])
        all_labels.append(labels if labels else [0])

    return all_boxes, all_scores, all_labels


def run_fusion_on_cache(
    entry,
    weights,
    shrink_w: float,
    scale_h: float,
    fuse: str,
    fusion_iou: float,
    skip_box: float,
    min_score: float,
    soft_sigma: float = 0.5,
    soft_score_thresh: float = 0.001,
    soft_method: int = 2,
):
    all_boxes, all_scores, all_labels = build_expert_lists(entry, shrink_w, scale_h)
    wt = list(weights)

    if fuse == "wbf":
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            all_boxes,
            all_scores,
            all_labels,
            weights=wt,
            iou_thr=fusion_iou,
            skip_box_thr=skip_box,
        )
    elif fuse == "nmw":
        fused_boxes, fused_scores, fused_labels = non_maximum_weighted(
            all_boxes,
            all_scores,
            all_labels,
            weights=wt,
            iou_thr=fusion_iou,
            skip_box_thr=skip_box,
        )
    elif fuse == "nms":
        fused_boxes, fused_scores, fused_labels = nms(
            all_boxes,
            all_scores,
            all_labels,
            iou_thr=fusion_iou,
            weights=wt,
        )
    elif fuse == "soft_nms":
        fused_boxes, fused_scores, fused_labels = soft_nms(
            all_boxes,
            all_scores,
            all_labels,
            method=soft_method,
            iou_thr=fusion_iou,
            sigma=soft_sigma,
            thresh=soft_score_thresh,
            weights=wt,
        )
    else:
        raise ValueError(f"Unknown fuse={fuse!r}")

    lines = []
    for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
        if score < min_score:
            continue
        x1, y1, x2, y2 = box
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        bw = x2 - x1
        bh = y2 - y1
        lines.append(f"{int(label)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {score:.6f}\n")
    return lines


def write_zip(label_dir: Path, zip_path: Path):
    label_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(label_dir.glob("*.txt")):
            zf.write(p, arcname=p.name)


def main():
    args = parse_args()
    test_images = sorted(glob.glob(os.path.join(args.source, "*.PNG")))
    if not test_images:
        raise SystemExit(f"No .PNG in {args.source}")

    if args.mode in ("multiscale-sweep", "multiscale-wbf12"):
        if args.mode == "multiscale-wbf12":
            if len(args.models) < 4:
                models = list(DEFAULT_MODELS_WBF12)
                weights = list(DEFAULT_WEIGHTS_WBF12)
            else:
                models = list(args.models)
                weights = list(args.weights)
        else:
            models = list(args.models)
            weights = list(args.weights)

        if len(weights) != len(models):
            raise SystemExit("--weights must match --models length")

        imgszs = list(args.imgszs)
        n_exp = len(models) * len(imgszs)
        fuse_abbr = FUSE_ZIP_ABBR[args.fuse]
        zip_prefix = f"v1_{fuse_abbr}{n_exp}_ms"

        # (zip_suffix, scale_w per imgsz). Explicit --scale-weights disables preset loop.
        preset_runs: list[tuple[str, list[float]]] = []
        if args.scale_weights is not None:
            sw = list(args.scale_weights)
            if len(sw) != len(imgszs):
                raise SystemExit("--scale-weights length must match --imgszs")
            preset_runs.append(("", sw))
        elif len(imgszs) != 3:
            sw = [1.0] * len(imgszs)
            preset_runs.append(("", sw))
        else:
            for pname in args.ms_sc_preset:
                sw = list(EXPERT_SCALE_PRESETS[pname])
                if len(args.ms_sc_preset) == 1 and pname == "d75":
                    suff = ""
                else:
                    suff = f"_sc{pname}"
                preset_runs.append((suff, sw))

        print(
            f"Multiscale cache: {len(test_images)} images x {len(models)} models x {len(imgszs)} sizes = {n_exp} experts ..."
        )
        cache = cache_predictions_multiscale(
            models, test_images, imgszs, args.conf, args.iou
        )
        print("Cache done.")

        sub_root = Path(args.submissions_dir)
        sub_root.mkdir(parents=True, exist_ok=True)

        shrinks = args.ms_shrink or [
            0.898,
            0.900,
            0.902,
            0.904,
            0.905,
            0.906,
            0.908,
            0.909,
            0.910,
            0.912,
        ]
        sweeps_ms = [
            (sw, skip, wiou)
            for sw in shrinks
            for skip in args.ms_skip
            for wiou in args.ms_wiou
        ]

        for sc_suffix, scale_w in preset_runs:
            expert_weights = []
            for mw in weights:
                for sw in scale_w:
                    expert_weights.append(float(mw) * float(sw))
            if len(expert_weights) != n_exp:
                raise SystemExit("Internal: expert_weights length mismatch")

            for shrink_w, skip_box, wbf_iou in sweeps_ms:
                name = (
                    f"{zip_prefix}_sw{int(round(shrink_w * 1000))}"
                    f"_sk{int(round(skip_box * 100))}"
                    f"_wi{int(round(wbf_iou * 100))}{sc_suffix}"
                )
                out_dir = sub_root / f"_tmp_{name}"
                out_dir.mkdir(parents=True, exist_ok=True)
                for entry in cache:
                    lines = run_fusion_on_cache(
                        entry,
                        expert_weights,
                        shrink_w,
                        1.0,
                        args.fuse,
                        wbf_iou,
                        skip_box,
                        args.min_score,
                        soft_sigma=args.soft_sigma,
                        soft_score_thresh=args.soft_score_thresh,
                        soft_method=args.soft_method,
                    )
                    (out_dir / f"{entry['basename']}.txt").write_text("".join(lines))
                zip_path = sub_root / f"{name}.zip"
                write_zip(out_dir, zip_path)
                for p in out_dir.glob("*.txt"):
                    p.unlink()
                out_dir.rmdir()
                print(f"  {zip_path.name}")

        print("Multiscale sweep zips under", sub_root)
        return

    if len(args.weights) != len(args.models):
        raise SystemExit("--weights must match --models length")

    print(f"Caching {len(test_images)} images x {len(args.models)} models ...")
    cache = cache_predictions(args.models, test_images, args.imgsz, args.conf, args.iou)
    print("Cache done.")

    sub_root = Path(args.submissions_dir)
    sub_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        if not args.out_dir:
            raise SystemExit("--out-dir required for mode=single")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for entry in cache:
            lines = run_fusion_on_cache(
                entry,
                args.weights,
                args.shrink_w,
                args.scale_h,
                args.fuse,
                args.wbf_iou,
                args.skip_box,
                args.min_score,
                soft_sigma=args.soft_sigma,
                soft_score_thresh=args.soft_score_thresh,
                soft_method=args.soft_method,
            )
            (out_dir / f"{entry['basename']}.txt").write_text("".join(lines))
        print(f"Wrote {len(cache)} files to {out_dir}")
        return

    # sweep: fine shrink around 0.91, WBF iou, slight height scale
    sweeps = []

    for sw in [0.905, 0.907, 0.909, 0.910, 0.911, 0.912, 0.914, 0.915]:
        sweeps.append(("shrink", {"shrink_w": sw, "wbf_iou": 0.5, "scale_h": 1.0}))

    for wiou in [0.42, 0.45, 0.48, 0.52, 0.55]:
        sweeps.append(("wiou", {"shrink_w": 0.91, "wbf_iou": wiou, "scale_h": 1.0}))

    for sh in [1.005, 1.01, 1.015, 1.02]:
        sweeps.append(("height", {"shrink_w": 0.91, "wbf_iou": 0.5, "scale_h": sh}))

    for skip in [0.07, 0.12]:
        sweeps.append(("skip", {"shrink_w": 0.91, "wbf_iou": 0.5, "scale_h": 1.0, "skip_box": skip}))

    for tag, kw in sweeps:
        shrink_w = kw.get("shrink_w", 0.91)
        wbf_iou = kw.get("wbf_iou", 0.5)
        scale_h = kw.get("scale_h", 1.0)
        skip_box = kw.get("skip_box", 0.1)

        ab = FUSE_ZIP_ABBR[args.fuse]
        nm = len(args.models)
        if tag == "shrink":
            name = f"v1_{ab}{nm}_sw{int(round(shrink_w * 1000))}"
        elif tag == "wiou":
            name = f"v1_{ab}{nm}_s91_wiou{int(round(wbf_iou * 100))}"
        elif tag == "height":
            name = f"v1_{ab}{nm}_s91_h{int(round(scale_h * 1000))}"
        else:
            name = f"v1_{ab}{nm}_s91_skip{int(round(skip_box * 100))}"

        out_dir = sub_root / f"_tmp_{name}"
        if out_dir.exists():
            for p in out_dir.glob("*.txt"):
                p.unlink()
        else:
            out_dir.mkdir(parents=True, exist_ok=True)

        for entry in cache:
            lines = run_fusion_on_cache(
                entry,
                args.weights,
                shrink_w,
                scale_h,
                args.fuse,
                wbf_iou,
                skip_box,
                args.min_score,
                soft_sigma=args.soft_sigma,
                soft_score_thresh=args.soft_score_thresh,
                soft_method=args.soft_method,
            )
            (out_dir / f"{entry['basename']}.txt").write_text("".join(lines))

        zip_path = sub_root / f"{name}.zip"
        write_zip(out_dir, zip_path)
        for p in out_dir.glob("*.txt"):
            p.unlink()
        out_dir.rmdir()
        print(f"  {zip_path.name}")

    print("Sweep zips written under", sub_root)


if __name__ == "__main__":
    main()
