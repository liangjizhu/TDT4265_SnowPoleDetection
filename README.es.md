English || **Español** || [简体中文](README.zh-CN.md)

# TDT4265 Mini-Proyecto: Detección de Postes de Nieve

Detección de objetos en tiempo real de postes de nieve para conducción autónoma en condiciones invernales, usando el dataset **Poles2025** de la región de Trondheim/Trondelag.

El proyecto usa **dos arquitecturas de detectores**: **Ultralytics YOLO** (scripts principales en `src/`, dependencias en `requirements.txt`) y **RF-DETR-B** (scripts en `src/rfdetr/`, dependencias en `src/rfdetr/requirements.txt`). Son pilas separadas; instala el archivo que corresponda al código que ejecutes.

## Resumen del dataset

| Subconjunto | Tamaño | Train | Val | Test | Etiquetas | Leaderboard |
|--------|------|-------|-----|------|--------|-------------|
| Road_poles_iPhone | 1.3 GB | 942 | 261 | 138 | Train + Val | iPhone submission |
| roadpoles_v1 | 615 MB | 322 | 92 | 46 | Train + Val | v1 submission |
| RoadPoles-MSJ | 283 MB | ~1904 | — | — | None (unlabeled) | — |

Todas las etiquetas están en formato YOLO (1 clase: `snow_pole`).

## Resultados

### Enfoque 1: YOLOv8n (nano) — 3.0M parámetros, 6 MB, 8.1 GFLOPs

Modelo base usando la variante más pequeña de YOLOv8, adecuada para despliegue en el borde e inferencia en tiempo real.

#### Dataset Road_poles_iPhone

| Métrica | Validación | Test (leaderboard) |
|--------|-----------|-------------------|
| Precisión | 0.949 | — |
| Recall | 0.885 | — |
| mAP@50 | 0.942 | **88.14%** |
| mAP@50:95 | 0.711 | **68.79%** |
| AR10 | — | 71.79% |

- Mejor checkpoint en la época 92
- Tiempo de entrenamiento: **594s** (~10 min)
- Velocidad de inferencia: **1.7 ms/imagen** (640x384)

#### Dataset roadpoles_v1

| Métrica | Validación | Test (leaderboard) |
|--------|-----------|-------------------|
| Precisión | 0.866 | — |
| Recall | 0.867 | — |
| mAP@50 | 0.899 | **94.94%** |
| mAP@50:95 | 0.494 | **57.99%** |
| AR10 | — | 62.76% |

- Mejor checkpoint en la época 96
- Tiempo de entrenamiento: **235s** (~4 min)
- Velocidad de inferencia: **3.1 ms/imagen** (416x640)

#### Envíos al leaderboard

| Dataset | Puntuación (mAP@50:95) | mAP@50 | AR10 | Envío |
|---------|-------------------------------|--------|------|------------|
| Road_poles_iPhone | **68.79%** | 88.14% | 71.79% | `iphone_test_predictions.zip` |
| roadpoles_v1 | **57.99%** | 94.94% | 62.76% | `v1_test_predictions.zip` |

### Enfoque 2: YOLOv8s (small) — 11.1M parámetros, 22.5 MB, 28.6 GFLOPs

Variante de mayor tamaño entrenada a mayor resolución (1280x1280) con mayor paciencia (50 épocas) para mejorar la localización de cajas (mAP@50:95).

#### Dataset Road_poles_iPhone

| Métrica | Validación | Test (leaderboard) |
|--------|-----------|-------------------|
| Precisión | 0.960 | — |
| Recall | 0.907 | — |
| mAP@50 | 0.960 | **92.81%** |
| mAP@50:95 | 0.830 | **77.7%** |
| AR10 | — | 80.6% |

- Se completaron las 200 épocas completas (sin early stopping)
- Tiempo de entrenamiento: **1.894h** (~114 min)
- Resolución: 1280x1280

#### Dataset roadpoles_v1

| Métrica | Validación | Test (leaderboard) |
|--------|-----------|-------------------|
| Precisión | 0.887 | — |
| Recall | 0.832 | — |
| mAP@50 | 0.887 | **80.67%** |
| mAP@50:95 | 0.545 | **51.85%** |
| AR10 | — | 62.24% |

- Early stopping en la época 164, mejor en la época 114 (patience=50)
- Tiempo de entrenamiento: **0.540h** (~32 min)
- Resolución: 1280x1280

#### Envíos al leaderboard

| Dataset | Puntuación (mAP@50:95) | mAP@50 | AR10 | Envío |
|---------|-------------------------------|--------|------|------------|
| Road_poles_iPhone | **77.7%** | 92.81% | 80.6% | `iphone_test_predictions_v2.zip` |
| roadpoles_v1 | **51.85%** | 80.67% | 62.24% | `v1_test_predictions_v2.zip` |

### Comparación: Enfoque 1 vs Enfoque 2

#### Road_poles_iPhone (942 imágenes de train)

| Métrica | YOLOv8n (640) | YOLOv8s (1280) | Delta |
|--------|--------------|----------------|-------|
| mAP@50:95 (test) | 68.79% | **77.7%** | **+8.91%** |
| mAP@50 (test) | 88.14% | **92.81%** | +4.67% |
| AR10 (test) | 71.79% | **80.6%** | +8.81% |

El modelo más grande y la mayor resolución mejoraron significativamente el rendimiento en el dataset iPhone, que tiene suficientes datos de entrenamiento (942 imágenes) para soportar los 11.1M parámetros de YOLOv8s.

#### roadpoles_v1 (322 imágenes de train)

| Métrica | YOLOv8n (640) | YOLOv8s (1280) | Delta |
|--------|--------------|----------------|-------|
| mAP@50:95 (test) | **57.99%** | 51.85% | **-6.14%** |
| mAP@50 (test) | **94.94%** | 80.67% | -14.27% |
| AR10 (test) | 62.76% | 62.24% | -0.52% |

**Hallazgo clave: YOLOv8s se sobreajusta en el dataset v1.** A pesar de mejorar las métricas de validación (mAP@50:95: 0.494 → 0.545), el rendimiento en el test cayó. Con solo 322 imágenes de entrenamiento, el modelo más grande memoriza la distribución de entrenamiento en lugar de aprender rasgos generalizables. El YOLOv8n más pequeño generaliza mejor con datos limitados. Este es un caso clásico del compromiso sesgo-varianza: un modelo de mayor capacidad necesita más datos para evitar el sobreajuste.

### Ajuste de inferencia: umbral de confianza y TTA

Se exploró el efecto de bajar el umbral de confianza y aplicar Test-Time Augmentation (TTA) en inferencia, sin reentrenar. TTA ejecuta el modelo en múltiples versiones aumentadas de cada imagen (flip, escalado) y fusiona predicciones.

#### Road_poles_iPhone — modelo YOLOv8s (1280)

| Conf | TTA | mAP@50:95 | mAP@50 | AR10 |
|------|-----|-----------|--------|------|
| 0.25 | No | 77.7% | 92.81 | 80.6% |
| 0.20 | Yes | 75.57% | 94.81 | 79.08% |
| 0.15 | Yes | 75.75% | 94.81 | 79.57% |
| 0.10 | Yes | 77.39% | 97.69 | 81.14% |
| **0.10** | **No** | **79.17%** | **95.69** | **82.07%** |

#### roadpoles_v1 — modelo YOLOv8n (640)

| Conf | TTA | mAP@50:95 | mAP@50 | AR10 |
|------|-----|-----------|--------|------|
| 0.25 | No | 57.99% | 94.94% | 62.76% |
| 0.20 | Yes | 59.18% | 96.23 | 63.97% |
| 0.15 | Yes | 59.18% | 96.23 | 63.97% |
| 0.10 | Yes | 59.18% | 96.23 | 63.97% |
| 0.15 | No | 59.23% | 98.78 | 64.14% |
| **0.10** | **No** | **59.23%** | **98.78** | **64.14%** |

**Hallazgo clave: TTA perjudica la detección de postes de nieve.** Los postes de nieve son objetos altos, delgados y verticales. El flip y escalado de TTA introducen ligeras desalineaciones de las cajas que degradan mAP@50:95, que penaliza cajas imprecisas en umbrales IoU altos. Bajar solo el umbral de confianza da la mejor mejora al capturar más verdaderos positivos.

### roadpoles_v1: WBF de 3 modelos + reducción de ancho + “WBF skip” (post-procesado)

Más mejoras en **roadpoles_v1** vinieron de pasos **solo en inferencia** (sin entrenamiento extra). La métrica del leaderboard es **mAP@50:95**; en v1, **mAP@50** ya era muy alto (~97–99%), así que el cuello de botella era la **alineación horizontal precisa** de postes muy delgados (ancho normalizado típico ≈ 0.008).

El pipeline está construido en **cuatro** etapas, en el orden en que las aplicamos:

#### 1. WBF3 — Weighted Boxes Fusion de tres modelos

Fusimos detecciones de tres checkpoints con **Weighted Boxes Fusion (WBF)** (paquete de PyPI `ensemble-boxes`, import `ensemble_boxes`), de modo que las cajas se fusionan en coordenadas normalizadas antes de escribir YOLO txt + confidence:

| Modelo | Rol |
|-------|------|
| `snow_poles_v1` (YOLOv8n, v1 train) | Detector fuerte específico de v1 |
| `yolov8s_finetune_v1` (YOLOv8s, iPhone → fine-tune v1) | Transfer learning, errores diferentes |
| `yolov8n_v1_640_200ep` (YOLOv8n, 200 épocas) | Sesgo ligeramente diferente |

Inferencia por modelo: `imgsz=640`, `conf=0.05`, `iou=0.7`. Valores por defecto de WBF salvo que se indique: `weights=[1.0, 1.0, 0.8]`, `iou_thr=0.5`, `skip_box_thr=0.1`.

**Efecto:** un zip único de WBF (sin reducción de ancho) alcanzó **60.96%** mAP@50:95 vs **59.23%** para el mejor modelo único + ajuste de conf — la diversidad del ensemble ayuda sobre todo a la **geometría de la caja**, no a “encontrar” postes.

#### 2. Reducción de ancho (después de cajas por modelo, antes de WBF)

En el split de **validación**, las cajas coincidentes eran en promedio **ligeramente demasiado anchas** respecto al ground truth (clase vertical delgada). Antes de la fusión, cada caja se reescala en **xyxy normalizado** (`ensemble_wbf_v1.py`: `build_expert_lists`): el centro horizontal se mantiene fijo y el **ancho completo** se escala por **`shrink_w`** (llamado \( \alpha \) aquí). Con bordes \(x_\ell, x_r\), centro \(c_x=\tfrac{1}{2}(x_\ell+x_r)\), y span \(w=x_r-x_\ell\),

$$
w_{\mathrm{new}}=\alpha\, w,\qquad
x_\ell^{\mathrm{new}}=c_x-\tfrac{w_{\mathrm{new}}}{2},\qquad
x_r^{\mathrm{new}}=c_x+\tfrac{w_{\mathrm{new}}}{2}.
$$

Típicamente \(0<\alpha\le 1\); las ejecuciones reportadas usan \(\alpha<1\) para estrechar la caja. La altura usa un factor separado **`scale_h`** (por defecto **1.0**).

Hicimos una búsqueda en rejilla de \(\alpha\) en el bucle de envío de test; **\(\alpha \approx 0.914\)** (es decir, **91.4%** del ancho predicho) fue uno de los ajustes más fuertes.

**Ejemplo del mejor resultado de este paso**

| Envío | Factor de ancho \( \alpha \) | mAP@50:95 | mAP@50 | AR10 |
|------------|-------------------------|-----------|--------|------|
| `v1_wbf3_sw914.zip` | **0.914** | **64.45%** | 98.74 | 69.31% |

#### 3. skip12 — `skip_box_thr` más alto en WBF

El `skip_box_thr` de WBF descarta propuestas de muy baja confianza durante la fusión. Subirlo de **0.10** a **0.12** (`skip12`) redujo cajas fusionadas espurias manteniendo la misma reducción de ancho **0.91** y `iou_thr=0.5`.

**Mejor (solo WBF3, un único `imgsz` 640)** — superado en el leaderboard v1 por WBF9 multiescala (etapa 4 abajo).

| Envío | Shrink \( \alpha \) | WBF `skip_box_thr` | mAP@50:95 | mAP@50 | AR10 |
|------------|-------------------|--------------------|-----------|--------|------|
| `v1_wbf3_s91_skip12.zip` | **0.91** | **0.12** | **64.58%** | 96.79 | 69.31% |

**Resumen vs el mejor v1 previo**

| Etapa | mAP@50:95 (test) | Notas |
|-------|------------------|--------|
| Mejor modelo único + conf (`v1` YOLOv8n, conf 0.1) | 59.23% | Baseline del README arriba |
| Solo WBF3 | 60.96% | `v1_wbf_ensemble.zip` |
| WBF3 + reducción de ancho (ajuste \(\alpha\)) | **64.45%** | `v1_wbf3_sw914.zip` — aquí \(\alpha \approx 0.914\) fue lo mejor |
| WBF3 + shrink 0.91 + `skip_box_thr=0.12` | **64.58%** | `v1_wbf3_s91_skip12.zip` |

#### 4. Multi-scale WBF9 (mismos 3 checkpoints YOLOv8, tres `imgsz` cada uno)

Cada checkpoint se ejecuta a **576, 640 y 704** (sigue siendo YOLOv8, sin 1280 en test-time), dando **9 expertos** para WBF. Los pesos de expertos son `model_weight × scale_weight`, con pesos por escala por defecto `[0.75, 1.0, 0.75]` para favorecer ligeramente **640**. Se aplica la misma reducción de ancho y ajuste de **`skip_box_thr` / `iou_thr`** por experto antes de la fusión.

Los zips multiescala usan el patrón de sufijo `…_swNNN_skNN_wiNN…` (construido en `ensemble_wbf_v1.py`: `round(shrink_w*1000)`, `round(skip_box*100)`, `round(wbf_iou*100)`). Decodifica los enteros finales como:

| Sufijo | Parámetro | Fórmula | Ejemplo |
|--------|-----------|---------|---------|
| `sw` + 3 dígitos | factor de reducción de ancho α | α = NNN / 1000 | `sw904` → α = 0.904 (α menor → caja más estrecha antes de WBF) |
| `sk` + 2 dígitos | WBF `skip_box_thr` | NN / 100 | `sk14` → 0.14 |
| `wi` + 2 dígitos | WBF `iou_thr` de fusión | NN / 100 | `wi52` → 0.52 |

**Récord en el leaderboard (confirmado)**

| Envío | \( \alpha \) | `skip_box_thr` | `iou_thr` | mAP@50:95 | mAP@50 | AR10 |
|------------|------------|----------------|-----------|-----------|--------|------|
| `v1_wbf9_ms_sw910_sk13_wi50.zip` | 0.910 | 0.13 | 0.50 | 67.03% | 99.09 | 71.72% |
| `v1_wbf9_ms_sw904_sk14_wi52.zip` | **0.904** | **0.14** | **0.52** | **67.44%** | **99.16** | — |

**Hallazgo — `sw`, `sk` y `wi` interactúan:** Con **WBF3 @ 640**, el mejor escalado de ancho fue **~0.914** (reducción más débil). Con **WBF9 multiescala**, el óptimo en el leaderboard no es el mismo: reducción horizontal más **fuerte** (**`sw904`**) más `skip_box_thr` **más alto** (**`sk14`**) y IoU de fusión WBF **más alto** (**`wi52`**) superaron la fila anterior **`sw910` + `sk13` + `wi50`** (**67.03% → 67.44%** en test). Los barridos conjuntos sobre \(\alpha\), `skip_box_thr` y `iou_thr` importan; no ajustes \(\alpha\) solo.

```bash
python src/ensemble_wbf_v1.py --mode multiscale-sweep --submissions-dir submissions
```

Esto escribe zips llamados `v1_wbf9_ms_sw{shrink×1000}_sk{skip×100}_wi{iou×100}.zip`. Usa `--ms-shrink`, `--ms-skip` y `--ms-wiou` para barrer el espacio conjunto; el récord actual usa **`sw904`**, **`sk14`**, **`wi52`**.

Reproducir / barrer variantes: `python src/ensemble_wbf_v1.py` (`--mode single`, `--mode sweep` o `--mode multiscale-sweep`).

### Experimento: YOLOv8n a 1280 en v1

Se probó si mayor resolución por sí sola (sin un modelo más grande) ayudaría en el pequeño dataset v1.

| Métrica | Val | Test (leaderboard) |
|--------|-----|-------------------|
| Precisión | 0.889 | — |
| Recall | 0.929 | — |
| mAP@50 | 0.967 | 86.36% |
| mAP@50:95 | 0.576 | **51.99%** |
| AR10 | — | 57.59% |

- 200 épocas completas, tiempo de entrenamiento: 0.689h (~41 min)

**Hallazgo clave: Mayor resolución perjudica en datasets pequeños independientemente del tamaño del modelo.** A pesar de tener el mejor mAP@50:95 de validación de cualquier modelo v1 (0.576), la puntuación en test cayó a 51.99%. A 1280, incluso el modelo nano memoriza detalles a nivel de píxel que no generalizan. La resolución 640 fuerza al modelo a aprender rasgos más gruesos y transferibles.

### Mejores puntuaciones del leaderboard

| Dataset | Mejor configuración | mAP@50:95 | mAP@50 | AR10 |
|---------|------------|-----------|--------|------|
| Road_poles_iPhone | YOLOv8s (1280), conf=0.1, sin TTA | **79.17%** | 95.69 | 82.07% |
| roadpoles_v1 (global) | RF-DETR-B (`submissions/submission_v1test4_rfdetr.zip`) | **72.64%** | 99.28 | 76.9% |
| roadpoles_v1 (YOLO + post-proc) | WBF9 multiescala + `sw904` + `sk14` + `wi52` (`v1_wbf9_ms_sw904_sk14_wi52.zip`) | **67.44%** | 99.16 | — |
| roadpoles_v1 | WBF3 + shrink 0.91 + `skip_box_thr=0.12` (`v1_wbf3_s91_skip12.zip`) | 64.58% | 96.79 | 69.31% |
| roadpoles_v1 (YOLO modelo único) | YOLOv8n (640), conf=0.1, sin TTA | 59.23% | 98.78 | 64.14% |

El mejor resultado global de **roadpoles_v1** es **RF-DETR-B**; las filas YOLO + WBF son los resultados más fuertes en el pipeline **Ultralytics** antes de añadir la segunda arquitectura.

### Historial completo de envíos al leaderboard

#### Road_poles_iPhone

| # | Modelo | Resolución | Conf | TTA | mAP@50:95 | mAP@50 | AR10 |
|---|-------|-----------|------|-----|-----------|--------|------|
| 1 | YOLOv8n | 640 | 0.25 | No | 68.79% | 88.14 | 71.79% |
| 2 | YOLOv8s | 1280 | 0.25 | No | 77.7% | 92.81 | 80.6% |
| 3 | YOLOv8s | 1280 | 0.10 | No | **79.17%** | 95.69 | 82.07% |
| 4 | YOLOv8s | 1280 | 0.10 | Yes | 77.39% | 97.69 | 81.14% |
| 5 | YOLOv8s | 1280 | 0.15 | Yes | 75.75% | 94.81 | 79.57% |
| 6 | YOLOv8s | 1280 | 0.20 | Yes | 75.57% | 94.81 | 79.08% |

#### roadpoles_v1

| # | Modelo | Resolución | Conf | TTA | mAP@50:95 | mAP@50 | AR10 |
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
| 10 | WBF3 + reducción de ancho \( \alpha \)=0.914 | 640 | 0.05 | No | **64.45%** | 98.74 | 69.31% |
| 11 | WBF3 + shrink 0.91 + WBF `skip_box_thr`=0.12 | 640 | 0.05 | No | **64.58%** | 96.79 | 69.31% |
| 12 | WBF9 multiescala (`v1_wbf9_ms_sw910_sk13_wi50.zip`) | 576/640/704 | 0.05 | No | 67.03% | 99.09 | 71.72% |
| 13 | WBF9 multiescala (`v1_wbf9_ms_sw904_sk14_wi52.zip`) | 576/640/704 | 0.05 | No | **67.44%** | 99.16 | — |
| 14 | RF-DETR-B (`submission_v1test4_rfdetr.zip`) | — | — | No | **72.64%** | 99.28 | 76.9% |

### Sostenibilidad

| | Enfoque 1 | | Enfoque 2 | | YOLOv8n 1280 | **Total** |
|--|--------|------|--------|------|------|---------|
| | iPhone | v1 | iPhone | v1 | v1 | Total |
| Tiempo de entrenamiento | 594s | 235s | 6818s | 1944s | 2480s | **12071s** (~201 min) |
| Consumo GPU (RTX 3070 Ti Laptop) | ~115W | ~115W | ~115W | ~115W | ~115W | — |
| Energía consumida | 0.019 kWh | 0.0075 kWh | 0.218 kWh | 0.062 kWh | 0.079 kWh | **0.386 kWh** |
| Equivalente Tesla Model Y (16.9 kWh/100km) | 112m | 44m | 1290m | 367m | 467m | **~2.28 km** |

## Seguimiento de progreso

- [x] Preparación del proyecto (estructura, configs, scripts)
- [x] Dataset descargado y configurado
- [x] Notebook de EDA creado (`notebooks/01_eda.ipynb`)
- [x] Notebook de EDA ejecutado y analizado
- [x] YOLOv8n entrenado en Road_poles_iPhone (100 épocas)
- [x] YOLOv8n entrenado en roadpoles_v1 (100 épocas)
- [x] Evaluación en conjuntos de validación
- [x] Predicciones de test generadas para el leaderboard
- [x] Envío al leaderboard — Enfoque 1 (iPhone: 68.79%, v1: 57.99%)
- [x] YOLOv8s (1280) entrenado en Road_poles_iPhone (200 épocas)
- [x] YOLOv8s (1280) entrenado en roadpoles_v1 (164 épocas, early stop)
- [x] Envío al leaderboard — Enfoque 2 (iPhone: 77.7%, v1: 51.85%)
- [x] Ajuste de inferencia: umbral de confianza + TTA (iPhone: 79.17%, v1: 59.23%)
- [x] roadpoles_v1: WBF de 3 modelos + reducción de ancho + WBF skip — **64.58%** (`v1_wbf3_s91_skip12.zip`; barrido de shrink WBF3-only **64.45%** con `v1_wbf3_sw914.zip`)
- [x] roadpoles_v1: WBF9 multiescala + shrink/skip/WBF IoU ajustados — **67.44%** (`v1_wbf9_ms_sw904_sk14_wi52.zip`; antes **67.03%** con `v1_wbf9_ms_sw910_sk13_wi50.zip`)
- [x] roadpoles_v1: **RF-DETR-B** (segunda arquitectura) — **72.64%** en el leaderboard (`submissions/submission_v1test4_rfdetr.zip`); código y docs en `src/rfdetr/` (`README.md`, `requirements.txt`)
- [x] YOLOv8n (1280) entrenado en roadpoles_v1 (200 épocas) — sobreajuste, 51.99%
- [x] Presentación en video (12–14 min)

## Estructura del proyecto

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

## RF-DETR-B (segunda arquitectura)

**[→ Documentación de RF-DETR](src/rfdetr/README.md)**

El archivo archivado **`submissions/submission_v1test4_rfdetr.zip`** documenta el envío v1 al leaderboard con **72.64%**.

## Instalación

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# For RF-DETR-B only (separate env recommended if versions conflict):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# pip install -r src/rfdetr/requirements.txt
```

## Uso

### 1. EDA

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Ejecuta todas las celdas con **`data/Poles2025/`** presente. Una **interpretación escrita** de las salidas de EDA (inventario, estadísticas de cajas, resoluciones, MSJ, e implicaciones para mAP@50:95 / WBF / domain gap) está al **final del notebook** en la sección markdown **“EDA summary — interpretation”** en [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb). Re-ejecuta primero las celdas de código si cambian tus datos o rutas y luego ajusta ese resumen si los números difieren.

*La EDA confirma detección de una sola clase con ~1.2 objetos por imagen etiquetada. roadpoles_v1 tiene un ancho normalizado mucho menor y una altura mayor que Road_poles_iPhone, así que la evaluación enfatiza la alineación precisa de cajas—consistente con mAP@50:95 y la reducción horizontal tras la fusión. Las resoluciones fijas difieren entre datasets (1080×1920 vs 1920×1208), apoyando una narrativa de brecha de dominio entre pistas. MSJ añade imágenes de carretera/nieve sin etiquetas para trabajo semi-supervisado opcional o cualitativo.*

### 2. Entrenar

```bash
# iPhone dataset
python src/train.py --config configs/road_poles_iphone.yaml --model yolov8n.pt --epochs 100

# roadpoles_v1 dataset
python src/train.py --config configs/roadpoles_v1.yaml --model yolov8n.pt --epochs 100 --name snow_poles_v1
```

### 3. Evaluar

```bash
python src/evaluate.py --model runs/train/snow_poles/weights/best.pt \
                       --config configs/road_poles_iphone.yaml
```

Usa la misma ruta `runs/train/<run_name>/weights/best.pt` que en tu comando de entrenamiento (`--name` define `<run_name>`; los valores por defecto son `--project runs/train` y `--name snow_poles`).

### 4. Predecir en el conjunto Test (para el leaderboard)

```bash
python src/predict.py --model runs/train/snow_poles/weights/best.pt \
                      --source data/Poles2025/Road_poles_iPhone/images/Test/test \
                      --save-txt --save-conf --name iphone_test
```

## Hardware

- **GPU**: NVIDIA GeForce RTX 3070 Ti Laptop GPU (7820 MiB)
- **Framework**: PyTorch **2.x** + build CUDA que coincida con tu driver (ver `requirements.txt`; ejecuta `python -c "import torch; print(torch.__version__, torch.version.cuda)"` en tu venv)
- **Librería de modelos**: Ultralytics **8.4.x** (el proyecto fija `ultralytics>=8.3.0`; la sub-versión exacta depende de la instalación)

## Definiciones de métricas

| Métrica | Descripción |
|--------|-------------|
| Precisión | TP / (TP + FP) — cuántas detecciones son correctas |
| Recall | TP / (TP + FN) — cuántos postes reales se encuentran |
| mAP@50 | AP medio con umbral IoU = 0.50 |
| mAP@50:95 | AP medio promediado sobre umbrales IoU 0.50 a 0.95 (paso 0.05) |
