English || [Español](README.es.md) || **简体中文**

# RF-DETR-B 雪杆检测

> **在本仓库中的位置：** `src/rfdetr/`。数据集：  
> `data/Poles2025/roadpoles_v1` 与 `data/Poles2025/Road_poles_iPhone`（与 YOLO 配置使用相同根目录）。  
> Checkpoint 与日志：`runs/rfdetr/...`（已在 `.gitignore` 中忽略，类似 Ultralytics 的 `runs/`）。  
> **依赖：** 本目录下的 `requirements.txt` — 在仓库根目录执行：  
> `pip install -r src/rfdetr/requirements.txt`（在安装匹配 CUDA 的 PyTorch 之后；见下文）。

在 `roadpoles_v1` 数据集上训练 **RF-DETR-B**（Roboflow, 2025）用于道路/雪杆目标检测，目标为 mAP@0.5:0.95 ≥ 0.70。

---

## 环境要求

| 项目 | 要求 |
|------|-------------|
| Python | 3.9 |
| CUDA | 12.1（driver ≥ 525） |
| GPU | RTX 3050 4GB 或更高 |
| uv | 已安装 |

---

## 1. 安装依赖

请使用本目录的 **`src/rfdetr/requirements.txt`**。它与仓库根目录的 **`requirements.txt`**（Ultralytics YOLO）分开管理并独立 pin 版本。

在 **仓库根目录**执行：

```bash
# PyTorch：选择与你的 CUDA/驱动匹配的 wheel index
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r src/rfdetr/requirements.txt
```

使用 **uv**（在仓库根目录）：

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -r src/rfdetr/requirements.txt
```

---

## 2. 合并数据集

随机将 iPhone 数据集的 30%（约 283 张）合并到 `roadpoles_v1/train/`：

```powershell
cd uv
.venv\Scripts\activate
python merge_dataset.py
```

这会将训练集从 322 扩展到约 ~605 张图像。

---

## 3. 将 YOLO 转为 COCO 格式

```powershell
python yolo_to_coco.py
```

---

## 4. 训练

```powershell
python train.py
```

训练配置：

| 参数 | 值 | 说明 |
|-----------|-------|------|
| 模型 | RF-DETR-B | DINOv2 backbone |
| 类别数 | 1 | pole |
| 输入分辨率 | 560×560 | 4GB 显存下稳定 |
| Batch size | 2 | 梯度累积 ×8 → 有效 batch 16 |
| 学习率 | 1e-4 | AdamW |
| 训练轮数 | 100（含 early stopping） | |
| 预训练权重 | COCO（自动下载） | |

结果会保存到 `uv/runs/`。

---

## 5. 数据集结构

RF-DETR 可直接读取 YOLO 格式。`roadpoles_v1/` 的目录结构：

```
roadpoles_v1/
├── data.yaml          ← nc: 1, names: ['pole']
├── train/
│   ├── images/        ← V1 images + iphone_*.jpg（合并后）
│   └── labels/        ← YOLO .txt annotations
├── valid/
│   ├── images/        ← V1 validation set (92 images)
│   └── labels/
└── test/
    ├── images/        ← V1 test set (46 images)
    └── labels/
```

---

## 6. 训练结果

### 6.1 `runs` — roadpoles_v1（仅 V1 数据集）

训练数据：`roadpoles_v1`（322 张训练图像）  
最佳 checkpoint：`uv/runs/checkpoint_best_ema.pth`

#### 最终验证指标（最佳 EMA checkpoint）

| 指标 | 值 |
|--------|-------|
| **mAP@0.5** | **0.9670** |
| **mAP@0.5:0.95** | **0.6356** |
| Precision | 0.9725 |
| Recall | 0.9300 |

#### 训练曲线 — 关键 epoch（验证集 EMA mAP@0.5:0.95）

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

> **最佳 EMA mAP@0.5:0.95 = 0.6453**（Epoch 34）  
> 验证集最终 `results.json`：mAP@0.5 = **0.9670**，mAP@0.5:0.95 = **0.6357**

---

### 6.2 `runs_iphone` — roadpoles_v1 + iPhone 数据集

训练数据：`roadpoles_v1` + iPhone 数据集 30%（约 ~605 张训练图像）  
最佳 checkpoint：`uv/runs_iphone/checkpoint_best_ema.pth`

#### 最终验证指标（最佳 EMA checkpoint）

| 指标 | 值 |
|--------|-------|
| **mAP@0.5** | **0.9970** |
| **mAP@0.5:0.95** | **0.8423** |
| Precision | 0.9820 |
| Recall | 0.9900 |

#### 训练曲线 — 关键 epoch（验证集 EMA mAP@0.5:0.95）

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

> **最佳 EMA mAP@0.5:0.95 = 0.8397**（Epoch 62）  
> 验证集最终 `results.json`：mAP@0.5 = **0.9970**，mAP@0.5:0.95 = **0.8423**

---

### 6.3 对比

| 指标 | `runs`（仅 V1） | `runs_iphone`（V1 + iPhone） | 提升 |
|--------|-----------------|------------------------------|------|
| 训练集大小 | 322 张 | ~605 张 | +88% |
| 最佳 EMA mAP@0.5:0.95 | 0.6453 | **0.8397** | **+19.4pp** |
| 最终 mAP@0.5 | 0.9670 | **0.9970** | +3.0pp |
| 最终 mAP@0.5:0.95 | 0.6357 | **0.8423** | **+20.7pp** |
| 最终 Precision | 0.9725 | **0.9820** | +0.95pp |
| 最终 Recall | 0.9300 | **0.9900** | +6.0pp |

> ✅ **结论**：合并 iPhone 数据将 mAP@0.5:0.95 从 0.6357 提升至 **0.8423**，超过 0.70 目标 **+20.7pp**。

---

## 7. 提升的关键因素

1. **数据多样性**：iPhone 图像引入了不同设备与场景
2. **DINOv2 预训练**：RF-DETR-B 利用强自监督视觉特征
3. **梯度累积**：在 4GB 显存下实现有效的大 batch 训练
4. **无 anchor 设计**：自动适配细长竖直的杆状目标
5. **EMA 权重**：指数滑动平均权重在验证集上更稳定

---

## 8. 推理

```powershell
# Run inference on the v1 test set
python predict_v1test.py

# Run inference on video
python predict_video.py
```

预测结果保存到 `uv/predictions_v1test/` 与 `uv/predictions_video/`。

---

## 9. 提交文件

| 文件 | 说明 |
|------|-------------|
| `submission.zip` | 基线预测 |
| `submission_v1test.zip` | V1 测试集预测 |
| `submission_iphone.zip` | iPhone 数据集预测 |
| `submission_conf005.zip` | conf=0.05 阈值的预测 |

---

## 10. 工作流脚本

使用 `rf_detr_workflow.py` 一条命令跑完整 pipeline：

```powershell
# Run all steps: convert → train → predict → evaluate
python rf_detr_workflow.py

# Or run individual steps
python rf_detr_workflow.py --step convert   # YOLO→COCO conversion only
python rf_detr_workflow.py --step train     # Training only
python rf_detr_workflow.py --step predict   # Inference only
python rf_detr_workflow.py --step eval      # Evaluation only
```
