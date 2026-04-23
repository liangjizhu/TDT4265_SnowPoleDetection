English || **Español** || [简体中文](README.zh-CN.md)

# RF-DETR-B Detección de Postes de Nieve

> **Ubicación en este repo:** `src/rfdetr/`. Datasets:  
> `data/Poles2025/roadpoles_v1` y `data/Poles2025/Road_poles_iPhone` (mismos roots que las configs de YOLO).  
> Checkpoints y logs: `runs/rfdetr/...` (gitignored, igual que `runs/` de Ultralytics).  
> **Dependencias:** `requirements.txt` en esta carpeta — desde la raíz del repo:  
> `pip install -r src/rfdetr/requirements.txt` (después de instalar PyTorch con un índice CUDA adecuado; ver abajo).

Entrenamiento de **RF-DETR-B** (Roboflow, 2025) en el dataset `roadpoles_v1` para detección de objetos (postes de carretera/nieve), apuntando a mAP@0.5:0.95 ≥ 0.70.

---

## Requisitos

| Ítem | Requisito |
|------|-------------|
| Python | 3.9 |
| CUDA | 12.1 (driver ≥ 525) |
| GPU | RTX 3050 4GB o superior |
| uv | instalado |

---

## 1. Instalar dependencias

Usa **`src/rfdetr/requirements.txt`** (este directorio). Fija el stack de RF-DETR por separado del **`requirements.txt`** de la raíz del repo (Ultralytics YOLO).

Desde la **raíz del repositorio**:

```bash
# PyTorch: elige el índice de wheels que coincida con tu CUDA/driver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r src/rfdetr/requirements.txt
```

Con **uv** (desde la raíz del repo):

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -r src/rfdetr/requirements.txt
```

---

## 2. Mezclar datasets

Mezcla aleatoriamente 30% del dataset iPhone (~283 imágenes) dentro de `roadpoles_v1/train/`:

```powershell
cd uv
.venv\Scripts\activate
python merge_dataset.py
```

Esto expande el conjunto de entrenamiento de 322 a ~605 imágenes.

---

## 3. Convertir YOLO a formato COCO

```powershell
python yolo_to_coco.py
```

---

## 4. Entrenar

```powershell
python train.py
```

Configuración de entrenamiento:

| Parámetro | Valor | Notas |
|-----------|-------|------|
| Modelo | RF-DETR-B | backbone DINOv2 |
| Clases | 1 | pole |
| Resolución de entrada | 560×560 | Estable con 4GB VRAM |
| Batch size | 2 | Gradient accumulation ×8 = efectivo 16 |
| Learning rate | 1e-4 | AdamW |
| Épocas | 100 (con early stopping) | |
| Pesos preentrenados | COCO (auto-descarga) | |

Los resultados se guardan en `uv/runs/`.

---

## 5. Estructura del dataset

RF-DETR lee formato YOLO directamente. El layout de `roadpoles_v1/`:

```
roadpoles_v1/
├── data.yaml          ← nc: 1, names: ['pole']
├── train/
│   ├── images/        ← V1 images + iphone_*.jpg (después de mezclar)
│   └── labels/        ← YOLO .txt annotations
├── valid/
│   ├── images/        ← V1 validation set (92 images)
│   └── labels/
└── test/
    ├── images/        ← V1 test set (46 images)
    └── labels/
```

---

## 6. Resultados de entrenamiento

### 6.1 `runs` — roadpoles_v1 (solo dataset V1)

Datos de entrenamiento: `roadpoles_v1` (322 imágenes de entrenamiento)  
Mejor checkpoint: `uv/runs/checkpoint_best_ema.pth`

#### Métricas finales de validación (mejor checkpoint EMA)

| Métrica | Valor |
|--------|-------|
| **mAP@0.5** | **0.9670** |
| **mAP@0.5:0.95** | **0.6356** |
| Precision | 0.9725 |
| Recall | 0.9300 |

#### Curva de entrenamiento — épocas clave (EMA mAP@0.5:0.95 en validación)

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

> **Mejor EMA mAP@0.5:0.95 = 0.6453** (Epoch 34)  
> `results.json` final en validación: mAP@0.5 = **0.9670**, mAP@0.5:0.95 = **0.6357**

---

### 6.2 `runs_iphone` — roadpoles_v1 + iPhone Dataset

Datos de entrenamiento: `roadpoles_v1` + 30% del dataset iPhone (~605 imágenes de entrenamiento)  
Mejor checkpoint: `uv/runs_iphone/checkpoint_best_ema.pth`

#### Métricas finales de validación (mejor checkpoint EMA)

| Métrica | Valor |
|--------|-------|
| **mAP@0.5** | **0.9970** |
| **mAP@0.5:0.95** | **0.8423** |
| Precision | 0.9820 |
| Recall | 0.9900 |

#### Curva de entrenamiento — épocas clave (EMA mAP@0.5:0.95 en validación)

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

> **Mejor EMA mAP@0.5:0.95 = 0.8397** (Epoch 62)  
> `results.json` final en validación: mAP@0.5 = **0.9970**, mAP@0.5:0.95 = **0.8423**

---

### 6.3 Comparación

| Métrica | `runs` (solo V1) | `runs_iphone` (V1 + iPhone) | Ganancia |
|--------|-----------------|------------------------------|------|
| Tamaño de entrenamiento | 322 imágenes | ~605 imágenes | +88% |
| Mejor EMA mAP@0.5:0.95 | 0.6453 | **0.8397** | **+19.4pp** |
| mAP@0.5 final | 0.9670 | **0.9970** | +3.0pp |
| mAP@0.5:0.95 final | 0.6357 | **0.8423** | **+20.7pp** |
| Precision final | 0.9725 | **0.9820** | +0.95pp |
| Recall final | 0.9300 | **0.9900** | +6.0pp |

> ✅ **Conclusión**: Mezclar el dataset iPhone elevó mAP@0.5:0.95 de 0.6357 a **0.8423**, superando el objetivo 0.70 por **+20.7pp**.

---

## 7. Factores clave de mejora

1. **Diversidad de datos**: las imágenes iPhone introducen distintos dispositivos y escenas
2. **Preentrenamiento DINOv2**: RF-DETR-B aprovecha fuertes características visuales auto-supervisadas
3. **Gradient accumulation**: entrenamiento con batch efectivo grande en 4GB VRAM
4. **Diseño sin anchors**: se adapta automáticamente al aspecto alto y estrecho de los postes
5. **Pesos EMA**: los pesos de media móvil exponencial son más estables en validación

---

## 8. Inferencia

```powershell
# Run inference on the v1 test set
python predict_v1test.py

# Run inference on video
python predict_video.py
```

Las predicciones se guardan en `uv/predictions_v1test/` y `uv/predictions_video/`.

---

## 9. Archivos de envío

| Archivo | Descripción |
|------|-------------|
| `submission.zip` | Predicciones base |
| `submission_v1test.zip` | Predicciones para el test de V1 |
| `submission_iphone.zip` | Predicciones para el dataset iPhone |
| `submission_conf005.zip` | Predicciones con umbral conf=0.05 |

---

## 10. Script de workflow

Usa `rf_detr_workflow.py` para ejecutar el pipeline completo con un solo comando:

```powershell
# Run all steps: convert → train → predict → evaluate
python rf_detr_workflow.py

# Or run individual steps
python rf_detr_workflow.py --step convert   # YOLO→COCO conversion only
python rf_detr_workflow.py --step train     # Training only
python rf_detr_workflow.py --step predict   # Inference only
python rf_detr_workflow.py --step eval      # Evaluation only
```
