# TDT4265 Mini-Project: Snow Pole Detection

Real-time object detection of snow poles for autonomous driving in winter conditions, using the **Poles2025** dataset from the Trøndelag region.

## Dataset Overview

| Subset | Size | Train | Val | Test | Labels | Leaderboard |
|--------|------|-------|-----|------|--------|-------------|
| Road_poles_iPhone | 1.3 GB | 942 | 261 | 138 | Train + Val | iPhone submission |
| roadpoles_v1 | 615 MB | 322 | 92 | 46 | Train + Val | v1 submission |
| RoadPoles-MSJ | 283 MB | ~1904 | — | — | None | — |

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

### 1. EDA (start here)

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Train

```bash
# iPhone dataset (main)
python src/train.py --config configs/road_poles_iphone.yaml --model yolov8n.pt --epochs 100

# roadpoles_v1 dataset
python src/train.py --config configs/roadpoles_v1.yaml --model yolov8n.pt --epochs 100
```

### 3. Evaluate

```bash
python src/evaluate.py --model runs/train/snow_poles/weights/best.pt --config configs/road_poles_iphone.yaml
```

### 4. Predict on Test Set (for leaderboard)

```bash
python src/predict.py --model runs/train/snow_poles/weights/best.pt \
                      --source data/Poles2025/Road_poles_iPhone/images/Test/test \
                      --save-txt --save-conf
```

## Metrics

| Metric | Description |
|--------|-------------|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| mAP@50 | Mean AP at IoU=0.50 |
| mAP@50:95 | Mean AP averaged over IoU 0.50–0.95 |
