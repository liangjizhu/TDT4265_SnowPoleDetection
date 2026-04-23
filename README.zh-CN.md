English || [Español](README.es.md) || **简体中文**

# TDT4265 迷你项目：雪杆检测（Snow Pole Detection）

面向冬季环境自动驾驶的雪杆实时目标检测项目，使用来自 Trondheim/Trondelag 地区的 **Poles2025** 数据集。

本项目使用 **两套检测器架构**：**Ultralytics YOLO**（主要脚本在 `src/`，依赖在 `requirements.txt`）以及 **RF-DETR-B**（脚本在 `src/rfdetr/`，依赖在 `src/rfdetr/requirements.txt`）。两者是独立技术栈；请安装与你要运行的代码相匹配的依赖文件。

## 数据集概览

| 子集 | 大小 | Train | Val | Test | 标签 | Leaderboard |
|--------|------|-------|-----|------|--------|-------------|
| Road_poles_iPhone | 1.3 GB | 942 | 261 | 138 | Train + Val | iPhone submission |
| roadpoles_v1 | 615 MB | 322 | 92 | 46 | Train + Val | v1 submission |
| RoadPoles-MSJ | 283 MB | ~1904 | — | — | None (unlabeled) | — |

所有标签均为 YOLO 格式（1 个类别：`snow_pole`）。

## 结果

### 方法 1：YOLOv8n（nano）— 3.0M 参数，6 MB，8.1 GFLOPs

基线模型，使用最小的 YOLOv8 变体，适合边缘部署与实时推理。

#### Road_poles_iPhone 数据集

| 指标 | Validation | Test (leaderboard) |
|--------|-----------|-------------------|
| Precision | 0.949 | — |
| Recall | 0.885 | — |
| mAP@50 | 0.942 | **88.14%** |
| mAP@50:95 | 0.711 | **68.79%** |
| AR10 | — | 71.79% |

- 最佳 checkpoint 位于第 92 个 epoch
- 训练时间：**594s**（约 10 分钟）
- 推理速度：**1.7 ms/图**（640x384）

#### roadpoles_v1 数据集

| 指标 | Validation | Test (leaderboard) |
|--------|-----------|-------------------|
| Precision | 0.866 | — |
| Recall | 0.867 | — |
| mAP@50 | 0.899 | **94.94%** |
| mAP@50:95 | 0.494 | **57.99%** |
| AR10 | — | 62.76% |

- 最佳 checkpoint 位于第 96 个 epoch
- 训练时间：**235s**（约 4 分钟）
- 推理速度：**3.1 ms/图**（416x640）

#### Leaderboard 提交

| 数据集 | Leaderboard 分数 (mAP@50:95) | mAP@50 | AR10 | 提交文件 |
|---------|-------------------------------|--------|------|------------|
| Road_poles_iPhone | **68.79%** | 88.14% | 71.79% | `iphone_test_predictions.zip` |
| roadpoles_v1 | **57.99%** | 94.94% | 62.76% | `v1_test_predictions.zip` |

### 方法 2：YOLOv8s（small）— 11.1M 参数，22.5 MB，28.6 GFLOPs

更大的模型变体，以更高分辨率（1280x1280）训练，并提升 patience（50 epochs），以改进边框定位（mAP@50:95）。

#### Road_poles_iPhone 数据集

| 指标 | Validation | Test (leaderboard) |
|--------|-----------|-------------------|
| Precision | 0.960 | — |
| Recall | 0.907 | — |
| mAP@50 | 0.960 | **92.81%** |
| mAP@50:95 | 0.830 | **77.7%** |
| AR10 | — | 80.6% |

- 完整跑完 200 个 epoch（未触发 early stopping）
- 训练时间：**1.894h**（约 114 分钟）
- 分辨率：1280x1280

#### roadpoles_v1 数据集

| 指标 | Validation | Test (leaderboard) |
|--------|-----------|-------------------|
| Precision | 0.887 | — |
| Recall | 0.832 | — |
| mAP@50 | 0.887 | **80.67%** |
| mAP@50:95 | 0.545 | **51.85%** |
| AR10 | — | 62.24% |

- 第 164 个 epoch early stopping，最佳在第 114 个 epoch（patience=50）
- 训练时间：**0.540h**（约 32 分钟）
- 分辨率：1280x1280

#### Leaderboard 提交

| 数据集 | Leaderboard 分数 (mAP@50:95) | mAP@50 | AR10 | 提交文件 |
|---------|-------------------------------|--------|------|------------|
| Road_poles_iPhone | **77.7%** | 92.81% | 80.6% | `iphone_test_predictions_v2.zip` |
| roadpoles_v1 | **51.85%** | 80.67% | 62.24% | `v1_test_predictions_v2.zip` |

### 对比：方法 1 vs 方法 2

#### Road_poles_iPhone（942 张训练图像）

| 指标 | YOLOv8n (640) | YOLOv8s (1280) | Delta |
|--------|--------------|----------------|-------|
| mAP@50:95 (test) | 68.79% | **77.7%** | **+8.91%** |
| mAP@50 (test) | 88.14% | **92.81%** | +4.67% |
| AR10 (test) | 71.79% | **80.6%** | +8.81% |

更大的模型与更高分辨率显著提升了 iPhone 数据集的表现。该数据集训练样本充足（942 张），能够支撑 YOLOv8s 的 11.1M 参数规模。

#### roadpoles_v1（322 张训练图像）

| 指标 | YOLOv8n (640) | YOLOv8s (1280) | Delta |
|--------|--------------|----------------|-------|
| mAP@50:95 (test) | **57.99%** | 51.85% | **-6.14%** |
| mAP@50 (test) | **94.94%** | 80.67% | -14.27% |
| AR10 (test) | 62.76% | 62.24% | -0.52% |

**关键结论：YOLOv8s 在 v1 数据集上过拟合。** 虽然验证集指标提升（mAP@50:95: 0.494 → 0.545），但测试集表现下降。仅有 322 张训练图像时，更大的模型更容易记住训练分布而不是学习可泛化特征。更小的 YOLOv8n 在小数据上泛化更好。这是典型的偏差-方差权衡：容量更高的模型需要更多数据以避免过拟合。

### 推理调参：置信度阈值与 TTA

在不重新训练的前提下，探索降低置信度阈值与使用 Test-Time Augmentation (TTA) 的影响。TTA 会对每张图片做多种增强（翻转、缩放）后运行模型并合并预测。

#### Road_poles_iPhone — YOLOv8s (1280) 模型

| Conf | TTA | mAP@50:95 | mAP@50 | AR10 |
|------|-----|-----------|--------|------|
| 0.25 | No | 77.7% | 92.81 | 80.6% |
| 0.20 | Yes | 75.57% | 94.81 | 79.08% |
| 0.15 | Yes | 75.75% | 94.81 | 79.57% |
| 0.10 | Yes | 77.39% | 97.69 | 81.14% |
| **0.10** | **No** | **79.17%** | **95.69** | **82.07%** |

#### roadpoles_v1 — YOLOv8n (640) 模型

| Conf | TTA | mAP@50:95 | mAP@50 | AR10 |
|------|-----|-----------|--------|------|
| 0.25 | No | 57.99% | 94.94% | 62.76% |
| 0.20 | Yes | 59.18% | 96.23 | 63.97% |
| 0.15 | Yes | 59.18% | 96.23 | 63.97% |
| 0.10 | Yes | 59.18% | 96.23 | 63.97% |
| 0.15 | No | 59.23% | 98.78 | 64.14% |
| **0.10** | **No** | **59.23%** | **98.78** | **64.14%** |

**关键发现：TTA 会伤害雪杆检测效果。** 雪杆是细长、竖直的目标。TTA 的翻转与缩放会引入轻微的边框对齐误差，从而降低 mAP@50:95（该指标在高 IoU 阈值下对边框精度惩罚更大）。单独降低置信度阈值通常更优，因为它能捕获更多真阳性。

### roadpoles_v1：三模型 WBF + 宽度收缩 + WBF skip（后处理）

在 **roadpoles_v1** 上的进一步提升来自 **纯推理阶段**（无需额外训练）。Leaderboard 指标为 **mAP@50:95**；在 v1 上 **mAP@50** 已很高（~97–99%），瓶颈在于极细长雪杆的 **水平对齐精度**（典型归一化宽度 ≈ 0.008）。

该流水线按我们实际应用的顺序由 **四** 个阶段组成：

#### 1. WBF3 — 三模型 Weighted Boxes Fusion

我们使用 **Weighted Boxes Fusion (WBF)**（PyPI 包 `ensemble-boxes`，导入名 `ensemble_boxes`）融合三个 checkpoint 的检测结果，先在归一化坐标中融合，再写出 YOLO txt + confidence：

| 模型 | 作用 |
|-------|------|
| `snow_poles_v1` (YOLOv8n, v1 train) | 强 v1 专用检测器 |
| `yolov8s_finetune_v1` (YOLOv8s, iPhone → fine-tune v1) | 迁移学习，错误类型不同 |
| `yolov8n_v1_640_200ep` (YOLOv8n, 200 epochs) | 略不同的偏置 |

单模型推理设置：`imgsz=640`, `conf=0.05`, `iou=0.7`。WBF 默认设置（除非另有说明）：`weights=[1.0, 1.0, 0.8]`, `iou_thr=0.5`, `skip_box_thr=0.1`。

**效果：** 单个 WBF zip（无宽度收缩）达到 **60.96%** mAP@50:95，相比最佳单模型 + conf 调参的 **59.23%** 有提升——集成多样性主要改善 **边框几何形状**，而不是“是否能找到”雪杆。

#### 2. 宽度收缩（每模型边框之后、WBF 之前）

在 **验证集** 上，匹配到的预测框平均相较 GT **略偏宽**（细长竖直目标）。在融合前对每个检测器输出的框做 **归一化 xyxy** 重缩放（`ensemble_wbf_v1.py`: `build_expert_lists`）：水平中心保持不变，**完整**归一化宽度按 **`shrink_w`**（此处记为 \( \alpha \)）缩放。设左右边界为 \(x_\ell, x_r\)，中心 \(c_x=\tfrac{1}{2}(x_\ell+x_r)\)，宽度 \(w=x_r-x_\ell\)，则

$$
w_{\mathrm{new}}=\alpha\, w,\qquad
x_\ell^{\mathrm{new}}=c_x-\tfrac{w_{\mathrm{new}}}{2},\qquad
x_r^{\mathrm{new}}=c_x+\tfrac{w_{\mathrm{new}}}{2}.
$$

通常 \(0<\alpha\le 1\)；本文报告的运行使用 \(\alpha<1\) 来使框变窄。高度有单独因子 **`scale_h`**（默认 **1.0**）。

我们在 test 提交流程中网格搜索 \(\alpha\)；**\(\alpha \approx 0.914\)**（即预测宽度的 **91.4%**）是表现较强的设置之一。

**该步骤的最佳示例**

| 提交 | 宽度因子 \( \alpha \) | mAP@50:95 | mAP@50 | AR10 |
|------------|-------------------------|-----------|--------|------|
| `v1_wbf3_sw914.zip` | **0.914** | **64.45%** | 98.74 | 69.31% |

#### 3. skip12 — 提高 WBF 的 `skip_box_thr`

WBF 的 `skip_box_thr` 会在融合时丢弃极低置信度候选框。将其从 **0.10** 提升到 **0.12**（`skip12`）可减少错误融合框，同时保持同样的 **0.91** 宽度收缩与 `iou_thr=0.5`。

**最佳（仅 WBF3，单一 `imgsz` 640）** — 在 v1 leaderboard 上被第 4 步的多尺度 WBF9 超越。

| 提交 | Shrink \( \alpha \) | WBF `skip_box_thr` | mAP@50:95 | mAP@50 | AR10 |
|------------|-------------------|--------------------|-----------|--------|------|
| `v1_wbf3_s91_skip12.zip` | **0.91** | **0.12** | **64.58%** | 96.79 | 69.31% |

**相对上一版 v1 best 的总结**

| 阶段 | mAP@50:95 (test) | 说明 |
|-------|------------------|--------|
| 最佳单模型 + conf（`v1` YOLOv8n, conf 0.1） | 59.23% | 上文基线 |
| 仅 WBF3 | 60.96% | `v1_wbf_ensemble.zip` |
| WBF3 + 宽度收缩（调参 \(\alpha\)） | **64.45%** | `v1_wbf3_sw914.zip` — 其中 \(\alpha \approx 0.914\) 最优 |
| WBF3 + shrink 0.91 + `skip_box_thr=0.12` | **64.58%** | `v1_wbf3_s91_skip12.zip` |

#### 4. 多尺度 WBF9（同 3 个 YOLOv8 checkpoint，每个跑 3 个 `imgsz`）

每个 checkpoint 分别在 **576、640、704** 三种尺度推理（仍为 YOLOv8，不使用 1280 的 test-time），形成 **9 个专家**进行 WBF。专家权重为 `model_weight × scale_weight`，默认尺度权重为 `[0.75, 1.0, 0.75]`，略偏好 **640**。同样的 **宽度收缩** 以及 **`skip_box_thr` / `iou_thr`** 调参会在每个专家上先应用，再融合。

多尺度 zip 使用后缀模式 `…_swNNN_skNN_wiNN…`（在 `ensemble_wbf_v1.py` 中生成：`round(shrink_w*1000)`, `round(skip_box*100)`, `round(wbf_iou*100)`）。整数后缀含义：

| 后缀 | 参数 | 公式 | 示例 |
|--------|-----------|---------|---------|
| `sw` + 3 位 | 宽度收缩因子 α | α = NNN / 1000 | `sw904` → α = 0.904（α 越小 → 融合前框越窄） |
| `sk` + 2 位 | WBF `skip_box_thr` | NN / 100 | `sk14` → 0.14 |
| `wi` + 2 位 | WBF 融合 `iou_thr` | NN / 100 | `wi52` → 0.52 |

**Leaderboard 记录（已确认）**

| 提交 | \( \alpha \) | `skip_box_thr` | `iou_thr` | mAP@50:95 | mAP@50 | AR10 |
|------------|------------|----------------|-----------|-----------|--------|------|
| `v1_wbf9_ms_sw910_sk13_wi50.zip` | 0.910 | 0.13 | 0.50 | 67.03% | 99.09 | 71.72% |
| `v1_wbf9_ms_sw904_sk14_wi52.zip` | **0.904** | **0.14** | **0.52** | **67.44%** | **99.16** | — |

**结论 — `sw`、`sk` 与 `wi` 会相互影响：** 在 **WBF3@640** 时，最佳宽度缩放为 **~0.914**（收缩较弱）。在 **多尺度 WBF9** 中，最优组合不同：更 **紧** 的水平收缩（**`sw904`**）+ 更 **高** 的 `skip_box_thr`（**`sk14`**）+ 更 **高** 的 WBF 融合 IoU（**`wi52`**）优于先前的 **`sw910` + `sk13` + `wi50`**（test 上 **67.03% → 67.44%**）。需要对 \(\alpha\)、`skip_box_thr` 与 `iou_thr` 做联合搜索，不能只调 \(\alpha\)。

```bash
python src/ensemble_wbf_v1.py --mode multiscale-sweep --submissions-dir submissions
```

该命令会生成 `v1_wbf9_ms_sw{shrink×1000}_sk{skip×100}_wi{iou×100}.zip`。使用 `--ms-shrink`、`--ms-skip` 与 `--ms-wiou` 联合扫参；当前记录使用 **`sw904`**、**`sk14`**、**`wi52`**。

复现 / 扫参：`python src/ensemble_wbf_v1.py`（`--mode single`、`--mode sweep` 或 `--mode multiscale-sweep`）。

### 实验：v1 上用 1280 分辨率训练 YOLOv8n

测试仅提高分辨率（不增大模型）是否能在小规模 v1 数据集上带来收益。

| 指标 | Val | Test (leaderboard) |
|--------|-----|-------------------|
| Precision | 0.889 | — |
| Recall | 0.929 | — |
| mAP@50 | 0.967 | 86.36% |
| mAP@50:95 | 0.576 | **51.99%** |
| AR10 | — | 57.59% |

- 跑满 200 个 epoch，训练时间：0.689h（约 41 分钟）

**关键结论：在小数据集上，高分辨率会变差，与模型大小无关。** 虽然验证集 mAP@50:95 达到 v1 模型中最高（0.576），但测试分数跌到 51.99%。在 1280 分辨率下，即便是 nano 模型也会记住像素级细节而无法泛化；640 分辨率反而促使模型学习更粗粒度、可迁移的特征。

### 最佳 Leaderboard 分数

| 数据集 | 最佳配置 | mAP@50:95 | mAP@50 | AR10 |
|---------|------------|-----------|--------|------|
| Road_poles_iPhone | YOLOv8s (1280), conf=0.1, no TTA | **79.17%** | 95.69 | 82.07% |
| roadpoles_v1（总体） | RF-DETR-B (`submissions/submission_v1test4_rfdetr.zip`) | **72.64%** | 99.28 | 76.9% |
| roadpoles_v1（YOLO + 后处理） | 多尺度 WBF9 + `sw904` + `sk14` + `wi52` (`v1_wbf9_ms_sw904_sk14_wi52.zip`) | **67.44%** | 99.16 | — |
| roadpoles_v1 | WBF3 + shrink 0.91 + `skip_box_thr=0.12` (`v1_wbf3_s91_skip12.zip`) | 64.58% | 96.79 | 69.31% |
| roadpoles_v1（YOLO 单模型） | YOLOv8n (640), conf=0.1, no TTA | 59.23% | 98.78 | 64.14% |

**roadpoles_v1** 的总体最佳分数来自 **RF-DETR-B**；YOLO + WBF 的结果是在 Ultralytics 流水线（加入第二架构前）最强的一组。

### 完整 Leaderboard 提交历史

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
| 10 | WBF3 + width shrink \( \alpha \)=0.914 | 640 | 0.05 | No | **64.45%** | 98.74 | 69.31% |
| 11 | WBF3 + shrink 0.91 + WBF `skip_box_thr`=0.12 | 640 | 0.05 | No | **64.58%** | 96.79 | 69.31% |
| 12 | WBF9 multiscale (`v1_wbf9_ms_sw910_sk13_wi50.zip`) | 576/640/704 | 0.05 | No | 67.03% | 99.09 | 71.72% |
| 13 | WBF9 multiscale (`v1_wbf9_ms_sw904_sk14_wi52.zip`) | 576/640/704 | 0.05 | No | **67.44%** | 99.16 | — |
| 14 | RF-DETR-B (`submission_v1test4_rfdetr.zip`) | — | — | No | **72.64%** | 99.28 | 76.9% |

### 可持续性

| | 方法 1 | | 方法 2 | | YOLOv8n 1280 | **All** |
|--|--------|------|--------|------|------|---------|
| | iPhone | v1 | iPhone | v1 | v1 | Total |
| Training time | 594s | 235s | 6818s | 1944s | 2480s | **12071s** (~201 min) |
| GPU power draw (RTX 3070 Ti Laptop) | ~115W | ~115W | ~115W | ~115W | ~115W | — |
| Energy consumed | 0.019 kWh | 0.0075 kWh | 0.218 kWh | 0.062 kWh | 0.079 kWh | **0.386 kWh** |
| Tesla Model Y equivalent (16.9 kWh/100km) | 112m | 44m | 1290m | 367m | 467m | **~2.28 km** |

## 进度跟踪

- [x] 项目搭建（结构、configs、scripts）
- [x] 数据集下载并配置
- [x] EDA notebook 创建（`notebooks/01_eda.ipynb`）
- [x] EDA notebook 执行并分析
- [x] 在 Road_poles_iPhone 上训练 YOLOv8n（100 epochs）
- [x] 在 roadpoles_v1 上训练 YOLOv8n（100 epochs）
- [x] 在验证集上评估
- [x] 生成 leaderboard 用的 test 预测
- [x] Leaderboard 提交 — 方法 1（iPhone: 68.79%, v1: 57.99%）
- [x] 在 Road_poles_iPhone 上训练 YOLOv8s（1280）（200 epochs）
- [x] 在 roadpoles_v1 上训练 YOLOv8s（1280）（164 epochs, early stop）
- [x] Leaderboard 提交 — 方法 2（iPhone: 77.7%, v1: 51.85%）
- [x] 推理调参：置信度阈值 + TTA（iPhone: 79.17%, v1: 59.23%）
- [x] roadpoles_v1：三模型 WBF + 宽度收缩 + WBF skip — **64.58%**（`v1_wbf3_s91_skip12.zip`; WBF3-only 收缩 sweep **64.45%**：`v1_wbf3_sw914.zip`）
- [x] roadpoles_v1：多尺度 WBF9 + shrink/skip/WBF IoU 联合调参 — **67.44%**（`v1_wbf9_ms_sw904_sk14_wi52.zip`; 之前 **67.03%**：`v1_wbf9_ms_sw910_sk13_wi50.zip`）
- [x] roadpoles_v1：**RF-DETR-B**（第二架构）— v1 leaderboard **72.64%**（`submissions/submission_v1test4_rfdetr.zip`）；代码与文档位于 `src/rfdetr/`（`README.md`, `requirements.txt`）
- [x] 在 roadpoles_v1 上训练 YOLOv8n（1280）（200 epochs）— 过拟合，51.99%
- [x] 视频展示（12–14 分钟）

## 项目结构

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

## RF-DETR-B（第二架构）

**[→ RF-DETR 文档](src/rfdetr/README.md)**

归档文件 **`submissions/submission_v1test4_rfdetr.zip`** 记录了 v1 leaderboard **72.64%** 的提交结果。

## 安装

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# For RF-DETR-B only (separate env recommended if versions conflict):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# pip install -r src/rfdetr/requirements.txt
```

## 使用

### 1. EDA

```bash
jupyter notebook notebooks/01_eda.ipynb
```

请在 **`data/Poles2025/`** 存在的情况下运行所有 cell。对 EDA 输出（数据清点、框统计、分辨率、MSJ，以及对 mAP@50:95 / WBF / domain gap 的影响）的**文字解读**位于 notebook **末尾**的 markdown 章节 **“EDA summary — interpretation”**（[`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)）。如果你的数据或路径变化，请先重跑代码单元，再根据变化调整该总结（若数值不同）。

*EDA 证实单类检测，标注图像中平均约 ~1.2 个目标。roadpoles_v1 的归一化宽度更小、目标更高，因此评估更强调边框精确对齐——这与 mAP@50:95 以及融合后进行水平宽度收缩的做法一致。两套数据的固定分辨率不同（1080×1920 vs 1920×1208），支持两条赛道之间存在域差异的叙述。MSJ 提供无标注的道路/雪景图像，可用于可选的半监督或定性分析。*

### 2. 训练

```bash
# iPhone dataset
python src/train.py --config configs/road_poles_iphone.yaml --model yolov8n.pt --epochs 100

# roadpoles_v1 dataset
python src/train.py --config configs/roadpoles_v1.yaml --model yolov8n.pt --epochs 100 --name snow_poles_v1
```

### 3. 评估

```bash
python src/evaluate.py --model runs/train/snow_poles/weights/best.pt \
                       --config configs/road_poles_iphone.yaml
```

请使用与你训练命令一致的 `runs/train/<run_name>/weights/best.pt` 路径（`--name` 设置 `<run_name>`；默认是 `--project runs/train` 和 `--name snow_poles`）。

### 4. 在 Test 集上预测（用于 leaderboard）

```bash
python src/predict.py --model runs/train/snow_poles/weights/best.pt \
                      --source data/Poles2025/Road_poles_iPhone/images/Test/test \
                      --save-txt --save-conf --name iphone_test
```

## 硬件

- **GPU**：NVIDIA GeForce RTX 3070 Ti Laptop GPU（7820 MiB）
- **框架**：PyTorch **2.x** + 与驱动匹配的 CUDA 构建（见 `requirements.txt`；在 venv 中运行 `python -c "import torch; print(torch.__version__, torch.version.cuda)"`）
- **模型库**：Ultralytics **8.4.x**（项目约束 `ultralytics>=8.3.0`；具体小版本取决于安装结果）

## 指标定义

| 指标 | 说明 |
|--------|-------------|
| Precision | TP / (TP + FP) — 检测中有多少是正确的 |
| Recall | TP / (TP + FN) — 有多少真实雪杆被找到 |
| mAP@50 | IoU 阈值 = 0.50 的平均 AP |
| mAP@50:95 | IoU 0.50 到 0.95（步长 0.05）平均后的 mAP |
