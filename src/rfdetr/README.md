English || [Español](README.es.md) || [简体中文](README.zh-CN.md)

# RF-DETR-B Snow Pole Detection

> **Location in this repo:** `src/rfdetr/`. Datasets:  
> `data/Poles2025/roadpoles_v1` and `data/Poles2025/Road_poles_iPhone` (same roots as the YOLO configs).  
> Checkpoints and logs: `runs/rfdetr/...` (gitignored, like Ultralytics `runs/`).  
> **Dependencies:** `requirements.txt` in this folder — from repo root:  
> `pip install -r src/rfdetr/requirements.txt` (after installing PyTorch with a suitable CUDA index; see below).

Training **RF-DETR-B** (Roboflow, 2025) on the `roadpoles_v1` dataset for road/snow pole object detection, targeting mAP@0.5:0.95 ≥ 0.70.

---

## Requirements

| Item | Requirement |
|------|-------------|
| Python | 3.9 |
| CUDA | 12.1 (driver ≥ 525) |
| GPU | RTX 3050 4GB or higher |
| uv | installed |

---

## 1. Install dependencies

Use **`src/rfdetr/requirements.txt`** (this directory). It pins the RF-DETR stack separately from the repo-root **`requirements.txt`** (Ultralytics YOLO).

From the **repository root**:

```bash
# PyTorch: pick the wheel index that matches your CUDA/driver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r src/rfdetr/requirements.txt
```

With **uv** (from repo root):

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -r src/rfdetr/requirements.txt
```

---

## 2. Merge Datasets

Randomly merge 30% of the iPhone dataset (~283 images) into `roadpoles_v1/train/`:

```powershell
cd uv
.venv\Scripts\activate
python merge_dataset.py
```

This expands the training set from 322 to ~605 images.

---

## 3. Convert YOLO to COCO Format

```powershell
python yolo_to_coco.py
```

---

## 4. Train

```powershell
python train.py
```

Training configuration:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | RF-DETR-B | DINOv2 backbone |
| Classes | 1 | pole |
| Input resolution | 560×560 | Stable on 4GB VRAM |
| Batch size | 2 | Gradient accumulation ×8 = effective 16 |
| Learning rate | 1e-4 | AdamW |
| Epochs | 100 (with early stopping) | |
| Pretrained weights | COCO (auto-download) | |

Results are saved to `uv/runs/`.

---

## 5. Dataset Structure

RF-DETR reads YOLO format directly. The `roadpoles_v1/` layout:

```
roadpoles_v1/
├── data.yaml          ← nc: 1, names: ['pole']
├── train/
│   ├── images/        ← V1 images + iphone_*.jpg (after merge)
│   └── labels/        ← YOLO .txt annotations
├── valid/
│   ├── images/        ← V1 validation set (92 images)
│   └── labels/
└── test/
    ├── images/        ← V1 test set (46 images)
    └── labels/
```

---

## 6. Training Results

### 6.1 `runs` — roadpoles_v1 (V1 dataset only)

Training data: `roadpoles_v1` (322 training images)  
Best checkpoint: `uv/runs/checkpoint_best_ema.pth`

#### Final Validation Metrics (Best EMA Checkpoint)

| Metric | Value |
|--------|-------|
| **mAP@0.5** | **0.9670** |
| **mAP@0.5:0.95** | **0.6356** |
| Precision | 0.9725 |
| Recall | 0.9300 |

#### Training Curve — Key Epochs (EMA mAP@0.5:0.95 on validation set)

| Epoch | Train Loss | EMA mAP@0.5:0.95 | EMA mAP@0.5 | EMA Precision | EMA Recall |
|-------|-----------|-------------------|-------------|---------------|------------|
| 0     | 5.4943    | 0.3113            | 0.7829      | 0.8687        | 0.76       |
| 5     | 4.7173    | 0.5324            | 0.8873      | 0.9500        | 0.84       |
| 10    | 4.4386    | 0.5803            | 0.9363      | 0.9450        | 0.91       |
| 15    | 4.5667    | 0.5858            | 0.9286      | 0.9450        | 0.91       |
| 20    | 4.1094    | 0.6078            | 0.9416      | 0.9298        | 0.93       |
| 25    | 3.6630    | 0.6229            | 0.9661      | 0.9643        | 0.95       |
| 30    | 3.5519    | 0.6321            | 0.9699      | 0.9322        | 0.97       |
| **34** | **3.4228** | **0.6453**     | **0.9667**  | **0.9558**    | **0.95**   |
| 40    | 3.6075    | 0.6265            | 0.9700      | 0.9316        | 0.96       |
| 49    | 2.9938    | 0.6283            | 0.9702      | 0.9474        | 0.95       |

> **Best EMA mAP@0.5:0.95 = 0.6453** (Epoch 34)  
> Final `results.json` on validation: mAP@0.5 = **0.9670**, mAP@0.5:0.95 = **0.6357**

---

### 6.2 `runs_iphone` — roadpoles_v1 + iPhone Dataset

Training data: `roadpoles_v1` + 30% of iPhone dataset (~605 training images)  
Best checkpoint: `uv/runs_iphone/checkpoint_best_ema.pth`

#### Final Validation Metrics (Best EMA Checkpoint)

| Metric | Value |
|--------|-------|
| **mAP@0.5** | **0.9970** |
| **mAP@0.5:0.95** | **0.8423** |
| Precision | 0.9820 |
| Recall | 0.9900 |

#### Training Curve — Key Epochs (EMA mAP@0.5:0.95 on validation set)

| Epoch | Train Loss | EMA mAP@0.5:0.95 | EMA mAP@0.5 | EMA Precision | EMA Recall |
|-------|-----------|-------------------|-------------|---------------|------------|
| 0     | 4.0261    | 0.6917            | 0.9841      | 0.9782        | 0.95       |
| 5     | 3.3571    | 0.7856            | 0.9960      | 0.9789        | 0.98       |
| 10    | 3.1672    | 0.8087            | 0.9974      | 0.9789        | 0.98       |
| 15    | 3.1132    | 0.8137            | 0.9971      | 0.9761        | 0.99       |
| 20    | 2.9922    | 0.8148            | 0.9966      | 0.9878        | 0.98       |
| 25    | 3.1080    | 0.8220            | 0.9961      | 0.9732        | 0.99       |
| 28    | 2.8734    | 0.8261            | 0.9962      | 0.9762        | 0.99       |
| 37    | 2.6956    | 0.8303            | 0.9974      | 0.9820        | 0.99       |
| 39    | 2.5601    | 0.8339            | 0.9973      | 0.9791        | 0.99       |
| 49    | 2.4881    | 0.8343            | 0.9970      | 0.9820        | 0.99       |
| 53    | 2.5288    | 0.8368            | 0.9969      | 0.9790        | 0.99       |
| 54    | 2.2641    | 0.8384            | 0.9972      | 0.9791        | 0.99       |
| **62** | **2.4879** | **0.8397**     | **0.9974**  | **0.9820**    | **0.99**   |

> **Best EMA mAP@0.5:0.95 = 0.8397** (Epoch 62)  
> Final `results.json` on validation: mAP@0.5 = **0.9970**, mAP@0.5:0.95 = **0.8423**

---

### 6.3 Comparison

| Metric | `runs` (V1 only) | `runs_iphone` (V1 + iPhone) | Gain |
|--------|-----------------|------------------------------|------|
| Training set size | 322 images | ~605 images | +88% |
| Best EMA mAP@0.5:0.95 | 0.6453 | **0.8397** | **+19.4pp** |
| Final mAP@0.5 | 0.9670 | **0.9970** | +3.0pp |
| Final mAP@0.5:0.95 | 0.6357 | **0.8423** | **+20.7pp** |
| Final Precision | 0.9725 | **0.9820** | +0.95pp |
| Final Recall | 0.9300 | **0.9900** | +6.0pp |

> ✅ **Conclusion**: Merging the iPhone dataset boosted mAP@0.5:0.95 from 0.6357 to **0.8423**, exceeding the 0.70 target by a margin of **+20.7pp**.

---

## 7. Key Factors for Improvement

1. **Data diversity**: iPhone images introduce different devices and scenes
2. **DINOv2 pretraining**: RF-DETR-B leverages strong self-supervised visual features
3. **Gradient accumulation**: Effective large-batch training on 4GB VRAM
4. **Anchor-free design**: Automatically adapts to the tall, narrow aspect ratio of poles
5. **EMA weights**: Exponential moving average weights are more stable on validation

---

## 8. Inference

```powershell
# Run inference on the v1 test set
python predict_v1test.py

# Run inference on video
python predict_video.py
```

Predictions are saved to `uv/predictions_v1test/` and `uv/predictions_video/`.

---

## 9. Submission Files

| File | Description |
|------|-------------|
| `submission.zip` | Baseline predictions |
| `submission_v1test.zip` | V1 test set predictions |
| `submission_iphone.zip` | iPhone dataset predictions |
| `submission_conf005.zip` | Predictions with conf=0.05 threshold |

---

## 10. Workflow Script

Use `rf_detr_workflow.py` to run the full pipeline in one command:

```powershell
# Run all steps: convert → train → predict → evaluate
python rf_detr_workflow.py

# Or run individual steps
python rf_detr_workflow.py --step convert   # YOLO→COCO conversion only
python rf_detr_workflow.py --step train     # Training only
python rf_detr_workflow.py --step predict   # Inference only
python rf_detr_workflow.py --step eval      # Evaluation only
```
