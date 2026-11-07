# Entrenamientos

Copia de `~/tarea2_runs/reales/` (`CARPETA_RUNS` de `comun.py`) con los entrenamientos finales de los seis modelos: pesos y registros, sin los `last.pt`.

| Modelo | Pesos |
|---|---|
| CNN propia | `cnn_propia/mejor.pt` |
| YOLO desde cero | `yolo_desde_cero/entrenamiento/weights/best.pt` |
| YOLOv3-tiny a mano | `yolo_a_mano/mejor.pt` |
| YOLO preentrenado | `yolo_preentrenado/entrenamiento/weights/best.pt` |
| YOLOv3-tiny a mano preentrenada | `yolo_a_mano_preentrenado/mejor.pt` |
| Faster R-CNN | `faster_rcnn/mejor.pt` |

Los archivos `.pt` están en [Git LFS](https://git-lfs.com). Hay que instalarlo antes de clonar; si no, se descargan punteros en vez de pesos.

Los cuadernos buscan estos archivos en `~/tarea2_runs/reales/`. Para reutilizarlos con `ENTRENAR = False`, hay que copiar allí el contenido de esta carpeta.
