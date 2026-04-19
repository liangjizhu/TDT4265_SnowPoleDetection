# RF-DETR-B 路杆检测训练环境

使用 **RF-DETR-B**（Roboflow，2025）在 `roadpoles_v1` 数据集上训练路杆（pole）目标检测模型，目标 mAP@0.5:0.95 ≥ 0.70。

---

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.9 |
| CUDA | 12.1（驱动 ≥ 525） |
| GPU | RTX 3050 4GB（或更高） |
| uv | 已安装 |

---

## 一、安装依赖

> 首次使用，在 `uv/` 目录下执行：

```powershell
# 1. 进入 uv 目录
cd e:\Desktop\Detection\uv

# 2. 安装 PyTorch（CUDA 12.1）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 rfdetr 训练依赖
uv pip install "rfdetr[train]"
```

---

## 二、合并数据集

将 iPhone 数据集的 30%（约 283 张）随机合并进 `roadpoles_v1/train/`：

```powershell
cd e:\Desktop\Detection\uv
.venv\Scripts\activate
python merge_dataset.py
```

执行后训练集从 322 张扩充至约 605 张。

---

## 三、开始训练

```powershell
cd e:\Desktop\Detection\uv
.venv\Scripts\activate
python train.py
```

训练配置：

| 参数 | 值 | 说明 |
|------|----|------|
| 模型 | RF-DETR-B | DINOv2 backbone |
| 类别数 | 1 | pole |
| 输入分辨率 | 640×640 | 4GB 显存下稳定 |
| Batch Size | 4 | 梯度累积 4 步 = 等效 16 |
| 学习率 | 1e-4 | AdamW |
| 训练轮数 | 100 epochs | |
| 预训练权重 | COCO（自动下载） | |

训练结果保存在 `uv/runs/` 目录下。

---

## 四、数据集结构

RF-DETR 自动识别 YOLO 格式，`roadpoles_v1/` 目录结构如下：

```
roadpoles_v1/
├── data.yaml          ← nc: 1, names: ['pole']
├── train/
│   ├── images/        ← V1 原始图片 + iphone_*.jpg（合并后）
│   └── labels/        ← YOLO 格式标注（.txt）
├── valid/
│   ├── images/        ← 仅 V1 验证集（92 张）
│   └── labels/
└── test/
    ├── images/        ← 仅 V1 测试集（46 张）
    └── labels/
```

---

## 五、提升 mAP 的关键策略

1. **数据多样性**：合并 iPhone 数据引入不同设备/场景
2. **DINOv2 预训练**：RF-DETR-B 使用强大的自监督视觉特征
3. **梯度累积**：4GB 显存下等效大 batch 训练
4. **无 Anchor 设计**：自动适应路杆细长宽高比，无需手动调参
