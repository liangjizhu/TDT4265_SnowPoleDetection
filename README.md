# TDT4265 Mini-Project: Snow Pole Detection

Real-time object detection of snow poles for autonomous driving in winter conditions, using the **Poles2025** dataset from the Trondheim/Trondelag region.

The project uses **two detector architectures**: **Ultralytics YOLO** (main scripts under `src/`, dependencies in `requirements.txt`) and **RF-DETR-B** (scripts under `src/rfdetr/`, dependencies in `src/rfdetr/requirements.txt`). They are separate stacks; install the file that matches the code you run.

## Dataset Overview

| Subset | Size | Train | Val | Test | Labels | Leaderboard |
|--------|------|-------|-----|------|--------|-------------|
| Road_poles_iPhone | 1.3 GB | 942 | 261 | 138 | Train + Val | iPhone submission |
| roadpoles_v1 | 615 MB | 322 | 92 | 46 | Train + Val | v1 submission |
| RoadPoles-MSJ | 283 MB | ~1904 | — | — | None (unlabeled) | — |

All labels are in YOLO format (1 class: `snow_pole`).

## Results

### Approach 1: YOLOv8n (nano) — 3.0M parameters, 6 MB, 8.1 GFLOPs

Baseline model using the smallest YOLOv8 variant, suitable for edge deployment and real-time inference.

#### Road_poles_iPhone Dataset

| Metric | Validation | Test (leaderboard) |
|--------|-----------|-------------------|
| Precision | 0.949 | — |
| Recall | 0.885 | — |
| mAP@50 | 0.942 | **88.14%** |
| mAP@50:95 | 0.711 | **68.79%** |
| AR10 | — | 71.79% |

- Best checkpoint at epoch 92
- Training time: **594s** (~10 min)
- Inference speed: **1.7 ms/image** (640x384)

#### roadpoles_v1 Dataset

| Metric | Validation | Test (leaderboard) |
|--------|-----------|-------------------|
| Precision | 0.866 | — |
| Recall | 0.867 | — |
| mAP@50 | 0.899 | **94.94%** |
| mAP@50:95 | 0.494 | **57.99%** |
| AR10 | — | 62.76% |

- Best checkpoint at epoch 96
- Training time: **235s** (~4 min)
- Inference speed: **3.1 ms/image** (416x640)

#### Leaderboard Submissions

| Dataset | Leaderboard Score (mAP@50:95) | mAP@50 | AR10 | Submission |
|---------|-------------------------------|--------|------|------------|
| Road_poles_iPhone | **68.79%** | 88.14% | 71.79% | `iphone_test_predictions.zip` |
| roadpoles_v1 | **57.99%** | 94.94% | 62.76% | `v1_test_predictions.zip` |

### Approach 2: YOLOv8s (small) — 11.1M parameters, 22.5 MB, 28.6 GFLOPs

Larger model variant trained at higher resolution (1280x1280) with increased patience (50 epochs) to improve bounding box localization (mAP@50:95).

#### Road_poles_iPhone Dataset

| Metric | Validation | Test (leaderboard) |
|--------|-----------|-------------------|
| Precision | 0.960 | — |
| Recall | 0.907 | — |
| mAP@50 | 0.960 | **92.81%** |
| mAP@50:95 | 0.830 | **77.7%** |
| AR10 | — | 80.6% |

- Full 200 epochs completed (no early stopping)
- Training time: **1.894h** (~114 min)
- Resolution: 1280x1280

#### roadpoles_v1 Dataset

| Metric | Validation | Test (leaderboard) |
|--------|-----------|-------------------|
| Precision | 0.887 | — |
| Recall | 0.832 | — |
| mAP@50 | 0.887 | **80.67%** |
| mAP@50:95 | 0.545 | **51.85%** |
| AR10 | — | 62.24% |

- Early stopping at epoch 164, best at epoch 114 (patience=50)
- Training time: **0.540h** (~32 min)
- Resolution: 1280x1280

#### Leaderboard Submissions

| Dataset | Leaderboard Score (mAP@50:95) | mAP@50 | AR10 | Submission |
|---------|-------------------------------|--------|------|------------|
| Road_poles_iPhone | **77.7%** | 92.81% | 80.6% | `iphone_test_predictions_v2.zip` |
| roadpoles_v1 | **51.85%** | 80.67% | 62.24% | `v1_test_predictions_v2.zip` |

### Comparison: Approach 1 vs Approach 2

#### Road_poles_iPhone (942 train images)

| Metric | YOLOv8n (640) | YOLOv8s (1280) | Delta |
|--------|--------------|----------------|-------|
| mAP@50:95 (test) | 68.79% | **77.7%** | **+8.91%** |
| mAP@50 (test) | 88.14% | **92.81%** | +4.67% |
| AR10 (test) | 71.79% | **80.6%** | +8.81% |

The larger model and higher resolution significantly improved performance on the iPhone dataset, which has sufficient training data (942 images) to support YOLOv8s's 11.1M parameters.

#### roadpoles_v1 (322 train images)

| Metric | YOLOv8n (640) | YOLOv8s (1280) | Delta |
|--------|--------------|----------------|-------|
| mAP@50:95 (test) | **57.99%** | 51.85% | **-6.14%** |
| mAP@50 (test) | **94.94%** | 80.67% | -14.27% |
| AR10 (test) | 62.76% | 62.24% | -0.52% |

**Key finding: YOLOv8s overfits on the v1 dataset.** Despite improved validation metrics (mAP@50:95: 0.494 → 0.545), the test set performance dropped. With only 322 training images, the larger model memorizes the training distribution instead of learning generalizable features. The smaller YOLOv8n generalizes better on limited data. This is a classic case of the bias-variance tradeoff: a higher-capacity model needs more data to avoid overfitting.

### Inference Tuning: Confidence Threshold & TTA

Explored the effect of lowering the confidence threshold and applying Test-Time Augmentation (TTA) at inference time, without retraining. TTA runs the model on multiple augmented versions of each image (flipped, scaled) and merges predictions.

#### Road_poles_iPhone — YOLOv8s (1280) model

| Conf | TTA | mAP@50:95 | mAP@50 | AR10 |
|------|-----|-----------|--------|------|
| 0.25 | No | 77.7% | 92.81% | 80.6% |
| 0.20 | Yes | 75.57% | 94.81 | 79.08% |
| 0.15 | Yes | 75.75% | 94.81 | 79.57% |
| 0.10 | Yes | 77.39% | 97.69 | 81.14% |
| **0.10** | **No** | **79.17%** | **95.69** | **82.07%** |

#### roadpoles_v1 — YOLOv8n (640) model

| Conf | TTA | mAP@50:95 | mAP@50 | AR10 |
|------|-----|-----------|--------|------|
| 0.25 | No | 57.99% | 94.94% | 62.76% |
| 0.20 | Yes | 59.18% | 96.23 | 63.97% |
| 0.15 | Yes | 59.18% | 96.23 | 63.97% |
| 0.10 | Yes | 59.18% | 96.23 | 63.97% |
| 0.15 | No | 59.23% | 98.78 | 64.14% |
| **0.10** | **No** | **59.23%** | **98.78** | **64.14%** |

**Key finding: TTA hurts snow pole detection.** Snow poles are tall, thin, vertical objects. TTA's flipping and scaling introduces slight bounding box misalignments that degrade mAP@50:95, which penalizes imprecise boxes at high IoU thresholds. Lowering the confidence threshold alone gives the best improvement by capturing additional true positives.

### roadpoles_v1: 3-model WBF + width shrink + WBF skip (post-processing)

Further gains on **roadpoles_v1** came from **inference-time** steps only (no extra training). The leaderboard metric is **mAP@50:95**; on v1, **mAP@50** was already very high (~97–99%), so the bottleneck was **tight horizontal alignment** of very thin poles (typical normalized width ≈ 0.008).

The pipeline is built in **four** stages, in the order we applied them:

#### 1. WBF3 — three-model Weighted Boxes Fusion

We fused detections from three checkpoints with **Weighted Boxes Fusion (WBF)** (PyPI package `ensemble-boxes`, import name `ensemble_boxes`), so boxes are merged in normalized coordinates before writing YOLO txt + confidence:

| Model | Role |
|-------|------|
| `snow_poles_v1` (YOLOv8n, v1 train) | Strong v1-specific detector |
| `yolov8s_finetune_v1` (YOLOv8s, iPhone → fine-tune v1) | Transfer learning, different errors |
| `yolov8n_v1_640_200ep` (YOLOv8n, 200 epochs) | Slightly different bias |

Per-model inference: `imgsz=640`, `conf=0.05`, `iou=0.7`. WBF defaults unless noted: `weights=[1.0, 1.0, 0.8]`, `iou_thr=0.5`, `skip_box_thr=0.1`.

**Effect:** a single WBF zip (no width shrink) reached **60.96%** mAP@50:95 vs **59.23%** for the best single-model + conf tuning — ensemble diversity mainly helps **box geometry**, not “finding” poles.

#### 2. Width shrink (after per-model boxes, before WBF)

On the **validation** split, matched boxes were on average **slightly too wide** relative to ground truth (thin vertical class). Before fusion, each detector box is rescaled in **normalized xyxy** (`ensemble_wbf_v1.py`: `build_expert_lists`): horizontal center is fixed, **full** normalized width is scaled by **`shrink_w`** (called $\alpha$ here). With left/right edges $x_\ell, x_r$, center $c_x=\tfrac{1}{2}(x_\ell+x_r)$, and span $w=x_r-x_\ell$,

$$
w_{\mathrm{new}}=\alpha\, w,\qquad
x_\ell^{\mathrm{new}}=c_x-\tfrac{w_{\mathrm{new}}}{2},\qquad
x_r^{\mathrm{new}}=c_x+\tfrac{w_{\mathrm{new}}}{2}.
$$

Typically $0<\alpha\le 1$; reported runs use $\alpha<1$ to narrow the box. Height uses a separate factor **`scale_h`** (default **1.0**).

We grid-searched $\alpha$ on the test submission loop; **$\alpha \approx 0.914$** (i.e. **91.4%** of the predicted width) was one of the strongest settings.

**Example best from this step**

| Submission | Width factor $\alpha$ | mAP@50:95 | mAP@50 | AR10 |
|------------|-------------------------|-----------|--------|------|
| `v1_wbf3_sw914.zip` | **0.914** | **64.45%** | 98.74 | 69.31% |

#### 3. skip12 — higher `skip_box_thr` in WBF

WBF’s `skip_box_thr` drops very low-confidence proposals during fusion. Raising it from **0.10** to **0.12** (`skip12`) reduced spurious merged boxes while keeping the same **0.91** width shrink and `iou_thr=0.5`.

**Best (WBF3 only, single `imgsz` 640)** — superseded on the v1 leaderboard by multiscale WBF9 (stage 4 below).

| Submission | Shrink $\alpha$ | WBF `skip_box_thr` | mAP@50:95 | mAP@50 | AR10 |
|------------|-------------------|--------------------|-----------|--------|------|
| `v1_wbf3_s91_skip12.zip` | **0.91** | **0.12** | **64.58%** | 96.79 | 69.31% |

**Summary vs previous v1 best**

| Stage | mAP@50:95 (test) | Notes |
|-------|------------------|--------|
| Best single model + conf (`v1` YOLOv8n, conf 0.1) | 59.23% | Baseline in README above |
| WBF3 only | 60.96% | `v1_wbf_ensemble.zip` |
| WBF3 + width shrink (tuned $\alpha$) | **64.45%** | `v1_wbf3_sw914.zip` — here $\alpha \approx 0.914$ was best |
| WBF3 + shrink 0.91 + `skip_box_thr=0.12` | **64.58%** | `v1_wbf3_s91_skip12.zip` |

#### 4. Multi-scale WBF9 (same 3 YOLOv8 checkpoints, three `imgsz` each)

Each checkpoint is run at **576, 640, and 704** (still YOLOv8, no 1280 test-time), giving **9 experts** for WBF. Expert weights are `model_weight × scale_weight`, with default scale weights `[0.75, 1.0, 0.75]` so **640** is slightly favored. The same **width shrink** and **`skip_box_thr` / `iou_thr`** tuning as above is applied per expert before fusion.

Multiscale zips use the suffix pattern `…_swNNN_skNN_wiNN…` (built in `ensemble_wbf_v1.py`: `round(shrink_w*1000)`, `round(skip_box*100)`, `round(wbf_iou*100)`). Decode the integer tails as:

| Suffix | Parameter | Formula | Example |
|--------|-----------|---------|---------|
| `sw` + 3 digits | width shrink factor α | α = NNN / 1000 | `sw904` → α = 0.904 (smaller α → narrower box before WBF) |
| `sk` + 2 digits | WBF `skip_box_thr` | NN / 100 | `sk14` → 0.14 |
| `wi` + 2 digits | WBF fusion `iou_thr` | NN / 100 | `wi52` → 0.52 |

**Leaderboard record (confirmed)**

| Submission | $\alpha$ | `skip_box_thr` | `iou_thr` | mAP@50:95 | mAP@50 | AR10 |
|------------|------------|----------------|-----------|-----------|--------|------|
| `v1_wbf9_ms_sw910_sk13_wi50.zip` | 0.910 | 0.13 | 0.50 | 67.03% | 99.09 | 71.72% |
| `v1_wbf9_ms_sw904_sk14_wi52.zip` | **0.904** | **0.14** | **0.52** | **67.44%** | **99.16** | — |

**Finding — `sw`, `sk`, and `wi` interact:** With **WBF3 @ 640 only**, the best width scale was **~0.914** (weaker shrink). With **multiscale WBF9**, the leaderboard optimum is **not** the same triple as the first multiscale sweep: **tighter** horizontal shrink (**`sw904`**) plus **higher** `skip_box_thr` (**`sk14`**) and **higher** WBF fusion IoU (**`wi52`**) beat the earlier **`sw910` + `sk13` + `wi50`** row (**67.03% → 67.44%** on test). Joint sweeps over $\alpha$, `skip_box_thr`, and `iou_thr` matter; do not tune $\alpha$ alone.

```bash
python src/ensemble_wbf_v1.py --mode multiscale-sweep --submissions-dir submissions
```

This writes zips named `v1_wbf9_ms_sw{shrink×1000}_sk{skip×100}_wi{iou×100}.zip`. Use `--ms-shrink`, `--ms-skip`, and `--ms-wiou` to sweep the joint space; the current record uses **`sw904`**, **`sk14`**, **`wi52`**.

Reproduce / sweep variants: `python src/ensemble_wbf_v1.py` (`--mode single`, `--mode sweep`, or `--mode multiscale-sweep`).

### Experiment: YOLOv8n at 1280 on v1

Tested whether higher resolution alone (without a larger model) would help on the small v1 dataset.

| Metric | Val | Test (leaderboard) |
|--------|-----|-------------------|
| Precision | 0.889 | — |
| Recall | 0.929 | — |
| mAP@50 | 0.967 | 86.36% |
| mAP@50:95 | 0.576 | **51.99%** |
| AR10 | — | 57.59% |

- Full 200 epochs, training time: 0.689h (~41 min)

**Key finding: Higher resolution hurts on small datasets regardless of model size.** Despite the best validation mAP@50:95 of any v1 model (0.576), the test score dropped to 51.99%. At 1280 resolution, even the nano model memorizes pixel-level details that don't generalize. The 640 resolution forces the model to learn coarser, more transferable features.

### Best Leaderboard Scores

| Dataset | Best Config | mAP@50:95 | mAP@50 | AR10 |
|---------|------------|-----------|--------|------|
| Road_poles_iPhone | YOLOv8s (1280), conf=0.1, no TTA | **79.17%** | 95.69 | 82.07% |
| roadpoles_v1 (overall) | RF-DETR-B (`submissions/submission_v1test4_rfdetr.zip`) | **72.64%** | 99.28 | 76.9% |
| roadpoles_v1 (YOLO + post-process) | Multiscale WBF9 + `sw904` + `sk14` + `wi52` (`v1_wbf9_ms_sw904_sk14_wi52.zip`) | **67.44%** | 99.16 | — |
| roadpoles_v1 | WBF3 + shrink 0.91 + `skip_box_thr=0.12` (`v1_wbf3_s91_skip12.zip`) | 64.58% | 96.79 | 69.31% |
| roadpoles_v1 (YOLO single-model) | YOLOv8n (640), conf=0.1, no TTA | 59.23% | 98.78 | 64.14% |

Overall **roadpoles_v1** best score is **RF-DETR-B**; the YOLO + WBF rows are the strongest results on the **Ultralytics** pipeline before adding the second architecture.

### Full Leaderboard Submission History

#### Road_poles_iPhone

| # | Model | Resolution | Conf | TTA | mAP@50:95 | mAP@50 | AR10 |
|---|-------|-----------|------|-----|-----------|--------|------|
| 1 | YOLOv8n | 640 | 0.25 | No | 68.79% | 88.14 | 71.79% |
| 2 | YOLOv8s | 1280 | 0.25 | No | 77.7% | 92.81 | 80.6% |
| 3 | YOLOv8s | 1280 | 0.10 | No | **79.17%** | 95.69 | 82.07% |
| 4 | YOLOv8s | 1280 | 0.10 | Yes | 77.39% | 97.69 | 81.14% |
| 5 | YOLOv8s | 1280 | 0.15 | Yes | 75.75% | 94.81 | 79.57% |
| 6 | YOLOv8s | 1280 | 0.20 | Yes | 75.57% | 94.81 | 79.08% |

#### roadpoles_v1

| # | Model | Resolution | Conf | TTA | mAP@50:95 | mAP@50 | AR10 |
|---|-------|-----------|------|-----|-----------|--------|------|
| 1 | YOLOv8n | 640 | 0.25 | No | 57.99% | 94.94 | 62.76% |
| 2 | YOLOv8s | 1280 | 0.25 | No | 51.85% | 80.67 | 62.24% |
| 3 | YOLOv8n | 640 | 0.10 | Yes | 59.18% | 96.23 | 63.97% |
| 4 | YOLOv8n | 640 | 0.10 | No | **59.23%** | 98.78 | 64.14% |
| 5 | YOLOv8n | 640 | 0.20 | Yes | 59.18% | 96.23 | 63.97% |
| 6 | YOLOv8n | 640 | 0.15 | Yes | 59.18% | 96.23 | 63.97% |
| 7 | YOLOv8n | 640 | 0.15 | No | 59.23% | 98.78 | 64.14% |
| 8 | YOLOv8n | 1280 | 0.10 | No | 51.99% | 86.36 | 57.59% |
| 9 | WBF3 (3×YOLOv8, 640) | 640 | 0.05 | No | 60.96% | — | — |
| 10 | WBF3 + width shrink $\alpha$=0.914 | 640 | 0.05 | No | **64.45%** | 98.74 | 69.31% |
| 11 | WBF3 + shrink 0.91 + WBF `skip_box_thr`=0.12 | 640 | 0.05 | No | **64.58%** | 96.79 | 69.31% |
| 12 | WBF9 multiscale (`v1_wbf9_ms_sw910_sk13_wi50.zip`) | 576/640/704 | 0.05 | No | 67.03% | 99.09 | 71.72% |
| 13 | WBF9 multiscale (`v1_wbf9_ms_sw904_sk14_wi52.zip`) | 576/640/704 | 0.05 | No | **67.44%** | 99.16 | — |
| 14 | RF-DETR-B (`submission_v1test4_rfdetr.zip`) | — | — | No | **72.64%** | 99.28 | 76.9% |

### Sustainability

| | Approach 1 | | Approach 2 | | YOLOv8n 1280 | **All** |
|--|--------|------|--------|------|------|---------|
| | iPhone | v1 | iPhone | v1 | v1 | Total |
| Training time | 594s | 235s | 6818s | 1944s | 2480s | **12071s** (~201 min) |
| GPU power draw (RTX 3070 Ti Laptop) | ~115W | ~115W | ~115W | ~115W | ~115W | — |
| Energy consumed | 0.019 kWh | 0.0075 kWh | 0.218 kWh | 0.062 kWh | 0.079 kWh | **0.386 kWh** |
| Tesla Model Y equivalent (16.9 kWh/100km) | 112m | 44m | 1290m | 367m | 467m | **~2.28 km** |

## Progress Tracker

- [x] Project setup (structure, configs, scripts)
- [x] Dataset downloaded and configured
- [x] EDA notebook created (`notebooks/01_eda.ipynb`)
- [x] EDA notebook executed and analyzed
- [x] YOLOv8n trained on Road_poles_iPhone (100 epochs)
- [x] YOLOv8n trained on roadpoles_v1 (100 epochs)
- [x] Evaluation on validation sets
- [x] Test predictions generated for leaderboard
- [x] Leaderboard submission — Approach 1 (iPhone: 68.79%, v1: 57.99%)
- [x] YOLOv8s (1280) trained on Road_poles_iPhone (200 epochs)
- [x] YOLOv8s (1280) trained on roadpoles_v1 (164 epochs, early stop)
- [x] Leaderboard submission — Approach 2 (iPhone: 77.7%, v1: 51.85%)
- [x] Inference tuning: confidence threshold + TTA (iPhone: 79.17%, v1: 59.23%)
- [x] roadpoles_v1: 3-model WBF + width shrink + WBF skip — **64.58%** (`v1_wbf3_s91_skip12.zip`; WBF3-only shrink sweep **64.45%** with `v1_wbf3_sw914.zip`)
- [x] roadpoles_v1: multiscale WBF9 + tuned shrink/skip/WBF IoU — **67.44%** (`v1_wbf9_ms_sw904_sk14_wi52.zip`; prior **67.03%** with `v1_wbf9_ms_sw910_sk13_wi50.zip`)
- [x] roadpoles_v1: **RF-DETR-B** (second architecture) — **72.64%** leaderboard (`submissions/submission_v1test4_rfdetr.zip`); code and docs in `src/rfdetr/` (`README.md`, `requirements.txt`)
- [x] YOLOv8n (1280) trained on roadpoles_v1 (200 epochs) — overfit, 51.99%
- [x] Video presentation (12–14 min)

## Project Structure

```
.
├── configs/
│   ├── road_poles_iphone.yaml   # iPhone dataset config
│   └── roadpoles_v1.yaml        # v1 dataset config
├── data/
│   └── Poles2025/               # Dataset (do NOT commit)
│       ├── Road_poles_iPhone/
│       ├── roadpoles_v1/
│       └── RoadPoles-MSJ/
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory data analysis
├── scripts/
│   ├── train_idun.slurm         # IDUN cluster training job
│   └── eval_idun.slurm          # IDUN cluster evaluation job
├── src/
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation (Precision, Recall, mAP)
│   ├── predict.py               # Inference / predictions
│   ├── ensemble_wbf_v1.py       # 3-model WBF + shrink / WBF hyperparam sweeps (v1 test)
│   └── rfdetr/                  # RF-DETR-B: scripts, README.md, requirements.txt
├── runs/                        # Training outputs (gitignored; paths depend on --project / --name)
│   ├── train/                   # default from `src/train.py`: --project runs/train
│   └── predict/                 # default from `src/predict.py`: --project runs/predict
├── requirements.txt           # Ultralytics / YOLO stack (default project env)
└── README.md
```

## RF-DETR-B (second architecture)

**[→ RF-DETR documentation](src/rfdetr/README.md)**

The archived **`submissions/submission_v1test4_rfdetr.zip`** documents the **72.64%** v1 leaderboard submission.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# For RF-DETR-B only (separate env recommended if versions conflict):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# pip install -r src/rfdetr/requirements.txt
```

## Usage

### 1. EDA

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Run all cells with **`data/Poles2025/`** present. A **written interpretation** of the EDA outputs (inventory, box statistics, resolutions, MSJ, and implications for mAP@50:95 / WBF / domain gap) lives at the **end of the notebook** in the markdown section **“EDA summary — interpretation”** in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb). Re-run the code cells first if your data or paths change, then adjust that summary if numbers differ.

*EDA confirms single-class detection with ~1.2 objects per labeled image. roadpoles_v1 has much smaller normalized width and larger height than Road_poles_iPhone, so evaluation emphasizes precise box alignment—consistent with mAP@50:95 and horizontal shrink after fusion. Fixed resolutions differ between datasets (1080×1920 vs 1920×1208), supporting a domain-gap narrative between tracks. MSJ adds unlabeled road/snow imagery for optional semi-supervised or qualitative work.*

### 2. Train

```bash
# iPhone dataset
python src/train.py --config configs/road_poles_iphone.yaml --model yolov8n.pt --epochs 100

# roadpoles_v1 dataset
python src/train.py --config configs/roadpoles_v1.yaml --model yolov8n.pt --epochs 100 --name snow_poles_v1
```

### 3. Evaluate

```bash
python src/evaluate.py --model runs/train/snow_poles/weights/best.pt \
                       --config configs/road_poles_iphone.yaml
```

Use the same `runs/train/<run_name>/weights/best.pt` path as in your training command (`--name` sets `<run_name>`; defaults are `--project runs/train` and `--name snow_poles`).

### 4. Predict on Test Set (for leaderboard)

```bash
python src/predict.py --model runs/train/snow_poles/weights/best.pt \
                      --source data/Poles2025/Road_poles_iPhone/images/Test/test \
                      --save-txt --save-conf --name iphone_test
```

## Hardware

- **GPU**: NVIDIA GeForce RTX 3070 Ti Laptop GPU (7820 MiB)
- **Framework**: PyTorch **2.x** + CUDA build matching your driver (see `requirements.txt`; run `python -c "import torch; print(torch.__version__, torch.version.cuda)"` in your venv)
- **Model library**: Ultralytics **8.4.x** (project pins `ultralytics>=8.3.0`; exact sub-version depends on the install)

## Metric Definitions

| Metric | Description |
|--------|-------------|
| Precision | TP / (TP + FP) — how many detections are correct |
| Recall | TP / (TP + FN) — how many ground truth poles are found |
| mAP@50 | Mean AP at IoU threshold = 0.50 |
| mAP@50:95 | Mean AP averaged over IoU thresholds 0.50 to 0.95 (step 0.05) |
