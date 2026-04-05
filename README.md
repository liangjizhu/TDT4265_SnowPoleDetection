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

### Sustainability

| | iPhone | v1 | Total |
|--|--------|------|-------|
| Training time | 594s | 235s | **829s** (~14 min) |
| GPU power draw (RTX 3070 Ti Laptop) | ~115W | ~115W | — |
| Energy consumed | 0.019 kWh | 0.0075 kWh | **0.027 kWh** |
| Tesla Model Y equivalent (16.9 kWh/100km) | 112m | 44m | **~160 meters** |

## Progress Tracker

- [x] Project setup (structure, configs, scripts)
- [x] Dataset downloaded and configured
- [x] EDA notebook created (`notebooks/01_eda.ipynb`)
- [ ] EDA notebook executed and analyzed
- [x] YOLOv8n trained on Road_poles_iPhone (100 epochs)
- [x] YOLOv8n trained on roadpoles_v1 (100 epochs)
- [x] Evaluation on validation sets
- [x] Test predictions generated for leaderboard
- [x] Leaderboard submission (iPhone: 68.79%, v1: 57.99%)
- [ ] Second model variant (e.g. YOLOv8s, YOLO11n)
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
