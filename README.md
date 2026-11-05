# Tarea 2 · Deep Learning (CIT3360): detección de objetos en FjordFish

Entrena y compara cuatro detectores de objetos sobre el dataset FjordFish, y evalúa la robustez del mejor ante imágenes degradadas. Todos los modelos usan las mismas particiones y se evalúan con el mismo código (`comun.py`).

## Cuadernos

| Cuaderno | Qué hace | Ítems de la rúbrica |
|---|---|---|
| `00_preparacion_dataset.ipynb` | Revisa las etiquetas, crea las particiones, exporta el dataset y hace el análisis exploratorio | 1.2 |
| `01_cnn_propia_desde_cero.ipynb` | Entrena y evalúa la CNN propia | 1.1, 1.2, 1.3 |
| `02_yolo_desde_cero.ipynb` | Entrena y evalúa YOLOv8n desde cero | 1.1, 1.2, 1.3 |
| `03_yolo_preentrenado.ipynb` | Entrena y evalúa YOLOv8n con pesos COCO | 2.1, 2.2, 2.3 |
| `04_faster_rcnn_preentrenado.ipynb` | Entrena y evalúa Faster R-CNN ResNet50-FPN v2 con pesos COCO | 2.1, 2.2, 2.3 |
| `05_comparacion.ipynb` | Compara los cuatro modelos con el evaluador común | 1.2, 1.4, 2.4, 3.2 |
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

Los archivos `.pt` están en [Git LFS](https://git-lfs.com), que hay que instalar antes de clonar. Para no entrenar de nuevo, se copia el contenido de `entrenamientos/` a `~/tarea2_runs/reales/` y se deja `ENTRENAR = False` en `01` a `04`. En Git Bash:

```
mkdir -p ~/tarea2_runs/reales
cp -r entrenamientos/* ~/tarea2_runs/reales/
```

## Orden de ejecución

Los cuadernos se ejecutan en orden, porque `05` lee los resultados de `01` a `04`, y `06` lee los de `05`. Los tiempos son aproximados, medidos en la GPU indicada.

| Orden | Cuaderno | Con `ENTRENAR = True` | Con `ENTRENAR = False` |
|---|---|---|---|
| 1 | `00` | 5 min (no entrena) | |
| 2 | `01` | 50 min (20 min más con `PROBAR_SOBREAJUSTE = True`) | Unos minutos |
| 3 | `02` | 50 min | Unos minutos |
| 4 | `03` | 50 min | Unos minutos |
| 5 | `04` | 2 h 45 min | Unos minutos |
| 6 | `05` | Menos de 1 min (no entrena) | |
| 7 | `06` | 10 min (no entrena) | |

## Resultados

- `resultados/<modelo>/`: tablas, figuras, detecciones y resumen de cada modelo (`cnn_propia`, `yolo_desde_cero`, `yolo_preentrenado` y `faster_rcnn`).
- `resultados/comparacion/`: tablas y figuras de `05`.
- `resultados/robustez/`: tablas, figuras y detecciones de `06`.

## ✍️ Declaración de uso de IA generativa

> Completar según lo que pide la tarea (herramienta y propósito específico).
