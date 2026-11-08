# Tarea 2 · Deep Learning (CIT3360): detección de objetos en FjordFish

Entrena y compara seis detectores de objetos sobre el dataset FjordFish, y evalúa la robustez del mejor ante imágenes degradadas. Todos los modelos usan las mismas particiones y se evalúan con el mismo código (`comun.py`).

## Cuadernos

| Cuaderno | Qué hace | Ítems de la rúbrica |
|---|---|---|
| `00_preparacion_dataset.ipynb` | Revisa las etiquetas, crea las particiones, exporta el dataset y hace el análisis exploratorio | 1.2 |
| `01_cnn_propia_desde_cero.ipynb` | Entrena y evalúa la CNN propia | 1.1, 1.2, 1.3 |
| `02_yolo_desde_cero.ipynb` | Entrena y evalúa YOLOv8n desde cero | 1.1, 1.2, 1.3 |
| `02b_yolo_a_mano.ipynb` | Implementa a mano, sin Ultralytics, y entrena una YOLOv3-tiny desde cero | 1.1, 1.2, 1.3 |
| `03_yolo_preentrenado.ipynb` | Entrena y evalúa YOLOv8n con pesos COCO | 2.1, 2.2, 2.3 |
| `03b_yolo_a_mano_preentrenado.ipynb` | Ajusta la YOLOv3-tiny de `02b` desde los pesos de COCO de Darknet (`yolov3-tiny.weights`) | 2.1, 2.2, 2.3 |
| `04_faster_rcnn_preentrenado.ipynb` | Entrena y evalúa Faster R-CNN ResNet50-FPN v2 con pesos COCO | 2.1, 2.2, 2.3 |
| `05_comparacion.ipynb` | Compara los seis modelos con el evaluador común | 1.2, 1.4, 2.4, 3.2 |
| `06_robustez.ipynb` | Evalúa el mejor modelo en test con imágenes degradadas | 3.1, 3.2, 3.3 |

## Entorno

Probado con Windows 11, Python 3.14 y una GPU NVIDIA RTX 4060 Laptop (8 GB). PyTorch se instala primero, desde su propio índice, como indica `requirements.txt`:

```
pip install torch==2.14.0 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

## Datos

El dataset no está en el repositorio. Hay que descargar `FjordFish.zip` (unos 6 GB) de https://zenodo.org/records/17950781 y dejarlo en la raíz del proyecto, sin descomprimir. `00_preparacion_dataset.ipynb` lee el zip directamente y genera `datasets/fjordfish/` con las particiones de `splits/`.

## Reutilizar los entrenamientos

Los archivos `.pt` están en [Git LFS](https://git-lfs.com), que hay que instalar antes de clonar. Para no entrenar de nuevo, se copia el contenido de `entrenamientos/` a `~/tarea2_runs/reales/` y se deja `ENTRENAR = False` en `01` a `04`, en `02b` (donde además van `PROBAR_SOBREAJUSTE = False` y `PROBAR_ESCALA = False`) y en `03b`. La primera vez, `03b` descarga `yolov3-tiny.weights` (35,4 MB) a `~/tarea2_runs/pesos_coco/`. En Git Bash:

```
mkdir -p ~/tarea2_runs/reales
cp -r entrenamientos/* ~/tarea2_runs/reales/
```

## Orden de ejecución

Los cuadernos se ejecutan en orden, porque `05` lee los resultados de `01`, `02`, `02b`, `03`, `03b` y `04`, y `06` lee los de `05`. Los tiempos son aproximados, medidos en la GPU indicada.

| Orden | Cuaderno | Con `ENTRENAR = True` | Con `ENTRENAR = False` |
|---|---|---|---|
| 1 | `00` | 5 min (no entrena) | |
| 2 | `01` | 50 min (20 min más con `PROBAR_SOBREAJUSTE = True`) | Unos minutos |
| 3 | `02` | 50 min | Unos minutos |
| 4 | `02b` | 1 h 40 min | Unos minutos |
| 5 | `03` | 50 min | Unos minutos |
| 6 | `03b` | 1 h 15 min | Unos minutos |
| 7 | `04` | 2 h 45 min | Unos minutos |
| 8 | `05` | Unos 2 min (no entrena) | |
| 9 | `06` | 10 min (no entrena) | |

## Resultados

- `resultados/<modelo>/`: tablas, figuras, detecciones y resumen de cada modelo (`cnn_propia`, `yolo_desde_cero`, `yolo_a_mano`, `yolo_preentrenado`, `yolo_a_mano_preentrenado` y `faster_rcnn`).
- `resultados/comparacion/`: tablas y figuras de `05`.
- `resultados/robustez/`: tablas, figuras y detecciones de `06`.
- `resultados/presentacion/`: gráficos, tablas y el diagrama de la YOLOv3-tiny para las diapositivas, uno por archivo, con los nombres y el orden de la presentación y el mismo formato y ejes en todos los modelos. Los genera `graficos_presentacion.py` (1 min, después de `05`; no usa la GPU). `guia_diapositivas.md`, en la misma carpeta, indica qué archivo va en cada diapositiva y cómo se lee.

## ✍️ Declaración de uso de IA generativa

> Completar según lo que pide la tarea (herramienta y propósito específico).
