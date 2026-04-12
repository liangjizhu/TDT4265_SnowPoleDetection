# TDT4265 Mini-Project: Snow Pole Detection

Real-time object detection of snow poles for autonomous driving in winter conditions, using the **Poles2025** dataset from the Trondheim/Trondelag region.

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
| roadpoles_v1 | YOLOv8n (640), conf=0.1, no TTA | **59.23%** | 98.78 | 64.14% |

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
- [ ] EDA notebook executed and analyzed
- [x] YOLOv8n trained on Road_poles_iPhone (100 epochs)
- [x] YOLOv8n trained on roadpoles_v1 (100 epochs)
- [x] Evaluation on validation sets
- [x] Test predictions generated for leaderboard
- [x] Leaderboard submission — Approach 1 (iPhone: 68.79%, v1: 57.99%)
- [x] YOLOv8s (1280) trained on Road_poles_iPhone (200 epochs)
- [x] YOLOv8s (1280) trained on roadpoles_v1 (164 epochs, early stop)
- [x] Leaderboard submission — Approach 2 (iPhone: 77.7%, v1: 51.85%)
- [x] Inference tuning: confidence threshold + TTA (iPhone: 79.17%, v1: 59.23%)
- [x] YOLOv8n (1280) trained on roadpoles_v1 (200 epochs) — overfit, 51.99%
- [ ] Error analysis / failure cases
- [ ] Video presentation (12–14 min)

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
│   └── predict.py               # Inference / predictions
├── runs/                        # Training outputs (gitignored)
│   └── detect/runs/
│       ├── train/snow_poles2/   # iPhone model + results
│       ├── train/snow_poles_v1/ # v1 model + results
│       └── predict/             # Test set predictions
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. EDA

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Train

```bash
# iPhone dataset
python src/train.py --config configs/road_poles_iphone.yaml --model yolov8n.pt --epochs 100

# roadpoles_v1 dataset
python src/train.py --config configs/roadpoles_v1.yaml --model yolov8n.pt --epochs 100 --name snow_poles_v1
```

### 3. Evaluate

```bash
python src/evaluate.py --model runs/detect/runs/train/snow_poles2/weights/best.pt \
                       --config configs/road_poles_iphone.yaml
```

### 4. Predict on Test Set (for leaderboard)

```bash
python src/predict.py --model runs/detect/runs/train/snow_poles2/weights/best.pt \
                      --source data/Poles2025/Road_poles_iPhone/images/Test/test \
                      --save-txt --save-conf --name iphone_test
```

## Hardware

- **GPU**: NVIDIA GeForce RTX 3070 Ti Laptop GPU (7820 MiB)
- **Framework**: PyTorch 2.11.0 + CUDA 13.0
- **Model library**: Ultralytics 8.4.33

## Metric Definitions

| Metric | Description |
|--------|-------------|
| Precision | TP / (TP + FP) — how many detections are correct |
| Recall | TP / (TP + FN) — how many ground truth poles are found |
| mAP@50 | Mean AP at IoU threshold = 0.50 |
| mAP@50:95 | Mean AP averaged over IoU thresholds 0.50 to 0.95 (step 0.05) |
