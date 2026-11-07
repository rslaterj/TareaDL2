"""Funciones comunes a los cuadernos de la Tarea 2 (CIT3360): particiones, cajas reales y evaluación.

Todos los modelos se evalúan con `evaluar_detecciones`, para que las métricas de la comparación se
calculen exactamente igual en todos. Las cajas van en píxeles de las imágenes de `datasets/fjordfish/`.

Formato de las predicciones de un modelo sobre una partición (DataFrame o CSV):
    id, clase_id, confianza, x1, y1, x2, y2
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HUELLA_ESPERADA = "7d4e588f7cc4"              # huella de splits/ que imprime el cuaderno 00
# Pesos y registros de entrenamiento, fuera de OneDrive: la sincronización bloquea los archivos que se reescriben
# en cada época y detiene el entrenamiento (PermissionError). Las tablas y figuras sí van en resultados/.
CARPETA_RUNS = Path.home() / "tarea2_runs"
UMBRALES_IOU = np.linspace(0.50, 0.95, 10)    # 0,50; 0,55; …; 0,95, como en COCO
PUNTOS_RECALL = np.linspace(0, 1, 101)        # el AP promedia la precisión en 101 niveles de recall, como en COCO
MAX_DETECCIONES = 300                         # por imagen, igual que la evaluación de Ultralytics
COLUMNAS_PREDICCIONES = ["id", "clase_id", "confianza", "x1", "y1", "x2", "y2"]
COLUMNAS_CAJA = ["x1", "y1", "x2", "y2"]


# ----------------------------------------------------------------------------- particiones y cajas reales

def cargar_particiones(dir_splits):
    """Lee splits/{train,val,test}.txt y devuelve (ids por partición, huella de 12 caracteres)."""
    particion = {s: (Path(dir_splits) / f"{s}.txt").read_text().split() for s in ("train", "val", "test")}
    huella = hashlib.md5(json.dumps(particion, sort_keys=True).encode()).hexdigest()[:12]
    return particion, huella


def verificar_huella(huella):
    """Detiene el cuaderno si las particiones no son las comunes del grupo (ítem 1.2)."""
    assert huella == HUELLA_ESPERADA, f"Huella {huella} distinta de {HUELLA_ESPERADA}: no son las particiones comunes"


def leer_etiquetas_yolo(ruta):
    """Lee un archivo de etiquetas YOLO: arreglo (n, 5) con clase, x_centro, y_centro, ancho y alto normalizados."""
    texto = Path(ruta).read_text().strip()
    if not texto:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array([fila.split() for fila in texto.splitlines()], dtype=np.float32).reshape(-1, 5)


def cargar_cajas_reales(dir_dataset, split, ids):
    """Cajas reales de una partición en píxeles: DataFrame con id, clase_id, x1, y1, x2, y2."""
    filas = []
    for id_img in ids:
        ancho, alto = Image.open(Path(dir_dataset) / "images" / split / f"{id_img}.jpg").size   # solo lee el encabezado
        for c, x, y, w, h in leer_etiquetas_yolo(Path(dir_dataset) / "labels" / split / f"{id_img}.txt"):
            filas.append((id_img, int(c), (x - w / 2) * ancho, (y - h / 2) * alto,
                          (x + w / 2) * ancho, (y + h / 2) * alto))
    return pd.DataFrame(filas, columns=["id", "clase_id", *COLUMNAS_CAJA])


# ----------------------------------------------------------------------------- evaluación

def iou(cajas_a, cajas_b):
    """Matriz de IoU (intersección sobre unión) entre cajas (n, 4) y (m, 4) en formato x1, y1, x2, y2."""
    a, b = cajas_a[:, None, :], cajas_b[None, :, :]
    ancho = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    alto = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    interseccion = ancho * alto
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return interseccion / (area_a + area_b - interseccion + 1e-9)


def preparar_datos(pred, reales):
    """Valida las tablas, ordena las predicciones de mayor a menor confianza y deja hasta MAX_DETECCIONES por imagen."""
    faltan = set(COLUMNAS_PREDICCIONES) - set(pred.columns)
    assert not faltan, f"Faltan columnas en las predicciones: {faltan}"
    pred = pred[COLUMNAS_PREDICCIONES].astype({"id": str, "clase_id": int})   # al leer un CSV, pandas convierte los ids a números
    reales = reales.astype({"id": str, "clase_id": int})
    desconocidas = set(pred.id) - set(reales.id)
    assert not desconocidas, f"Hay predicciones de imágenes que no están en esta partición, por ejemplo {sorted(desconocidas)[:3]}"
    pred = pred.sort_values("confianza", ascending=False, kind="stable")
    pred = pred.groupby("id", sort=False).head(MAX_DETECCIONES).reset_index(drop=True)
    return pred, reales


def marcar_aciertos(pred, reales):
    """Indica, para cada predicción y cada umbral de IoU, si es un verdadero positivo (criterio de COCO).

    En cada imagen las predicciones se recorren de mayor a menor confianza, y cada una se asigna a la caja
    real de su misma clase, todavía libre, con la que tiene mayor IoU, si ese IoU alcanza el umbral.
    Una caja real se asigna una sola vez: las detecciones duplicadas cuentan como falsos positivos.
    """
    aciertos = np.zeros((len(pred), len(UMBRALES_IOU)), dtype=bool)
    reales_por_imagen = dict(tuple(reales.groupby("id")))
    for id_img, p in pred.groupby("id", sort=False):         # dentro de cada imagen se mantiene el orden por confianza
        g = reales_por_imagen[id_img]
        matriz = iou(p[COLUMNAS_CAJA].to_numpy(float), g[COLUMNAS_CAJA].to_numpy(float))
        matriz[p.clase_id.to_numpy()[:, None] != g.clase_id.to_numpy()[None, :]] = 0.0   # solo cajas de su clase
        libres = np.ones((len(g), len(UMBRALES_IOU)), dtype=bool)
        for fila, iou_fila in zip(p.index, matriz):
            candidatas = np.where(libres & (iou_fila[:, None] >= UMBRALES_IOU), iou_fila[:, None], -1.0)
            elegida = candidatas.argmax(axis=0)                # caja real elegida en cada umbral
            hay = candidatas.max(axis=0) >= 0
            libres[elegida[hay], np.flatnonzero(hay)] = False
            aciertos[fila, hay] = True
    return aciertos


def precision_interpolada(aciertos, n_reales):
    """Precisión de una clase en 101 niveles de recall (filas) para cada umbral de IoU (columnas), como en COCO.

    `aciertos` son las filas de marcar_aciertos de esa clase, de mayor a menor confianza. La precisión se
    reemplaza por su envolvente (la máxima precisión con un recall igual o mayor), y donde el modelo nunca
    alcanza ese recall vale 0. El AP es el promedio de cada columna.
    """
    tabla = np.zeros((len(PUNTOS_RECALL), len(UMBRALES_IOU)))
    if len(aciertos) == 0:
        return tabla
    vp = np.cumsum(aciertos, axis=0)                           # verdaderos positivos acumulados
    fp = np.cumsum(~aciertos, axis=0)                          # falsos positivos acumulados
    recall = vp / n_reales
    precision = np.maximum.accumulate((vp / (vp + fp))[::-1], axis=0)[::-1]
    for t in range(len(UMBRALES_IOU)):
        i = np.searchsorted(recall[:, t], PUNTOS_RECALL, side="left")   # primera predicción que alcanza cada recall
        alcanzado = i < len(precision)
        tabla[alcanzado, t] = precision[i[alcanzado], t]
    return tabla


def precision_recall_en_umbral(pred, aciertos, reales, umbral):
    """Precisión, recall y F1 con IoU 0,5, contando solo las predicciones con confianza ≥ umbral.

    Se calculan por clase y se promedian entre las clases que tienen cajas reales, como hace Ultralytics.
    """
    es_acierto = aciertos[:, 0]
    sobre_umbral = pred.confianza.to_numpy() >= umbral
    clases_pred = pred.clase_id.to_numpy()
    precisiones, recalls = [], []
    for c, n_reales in reales.clase_id.value_counts().sort_index().items():
        de_la_clase = sobre_umbral & (clases_pred == c)
        vp = es_acierto[de_la_clase].sum()
        precisiones.append(vp / de_la_clase.sum() if de_la_clase.any() else 0.0)
        recalls.append(vp / n_reales)
    p, r = float(np.mean(precisiones)), float(np.mean(recalls))
    return p, r, (2 * p * r / (p + r) if p + r > 0 else 0.0)


def evaluar_detecciones(pred, reales, nombres=None, umbral_confianza=None):
    """Métricas de un modelo sobre una partición, calculadas igual para todos los modelos.

    Devuelve un diccionario con:
      - 'mAP50', 'mAP75' y 'mAP50_95': promedio del AP entre las clases que tienen cajas reales;
      - 'precision', 'recall' y 'F1' (solo si se da umbral_confianza): con IoU 0,5 y promediadas entre clases;
      - 'precision_micro' (solo si se da umbral_confianza): fracción de las detecciones con confianza ≥ umbral que son
        acierto con IoU 0,5, sin promediar por clase; 0,0 si no hay ninguna;
      - 'clases_con_detecciones' (solo si se da umbral_confianza): cuántas clases con cajas reales tienen al menos una
        detección con confianza ≥ umbral;
      - 'por_clase': tabla con instancias, predicciones, AP50, AP75 y AP50-95 de cada clase;
      - 'curva_pr': precisión media entre clases en cada nivel de recall, con IoU 0,5.
    """
    pred, reales = preparar_datos(pred, reales)
    aciertos = marcar_aciertos(pred, reales)
    clases_pred = pred.clase_id.to_numpy()
    filas, curvas = [], []
    for c, n_reales in reales.clase_id.value_counts().sort_index().items():
        de_la_clase = clases_pred == c
        tabla = precision_interpolada(aciertos[de_la_clase], n_reales)
        ap = tabla.mean(axis=0)                                # un AP por umbral de IoU
        filas.append({"clase_id": c, "clase": nombres[c] if nombres is not None else str(c),
                      "instancias": int(n_reales), "predicciones": int(de_la_clase.sum()),
                      "AP50": ap[0], "AP75": ap[5], "AP50_95": ap.mean()})
        curvas.append(tabla[:, 0])
    por_clase = pd.DataFrame(filas)
    resultado = {
        "mAP50": float(por_clase.AP50.mean()),
        "mAP75": float(por_clase.AP75.mean()),
        "mAP50_95": float(por_clase.AP50_95.mean()),
        "por_clase": por_clase,
        "curva_pr": pd.DataFrame({"recall": PUNTOS_RECALL, "precision": np.mean(curvas, axis=0)}),
    }
    if umbral_confianza is not None:
        resultado["precision"], resultado["recall"], resultado["F1"] = precision_recall_en_umbral(
            pred, aciertos, reales, umbral_confianza)
        sobre_umbral = pred.confianza.to_numpy() >= umbral_confianza
        resultado["precision_micro"] = float(aciertos[sobre_umbral, 0].mean()) if sobre_umbral.any() else 0.0
        resultado["clases_con_detecciones"] = int(np.isin(por_clase.clase_id, clases_pred[sobre_umbral]).sum())
    return resultado


def curva_segun_umbral(pred, reales, umbrales=None):
    """Precisión, recall y F1 (IoU 0,5) para cada umbral de confianza; sirve para elegir el umbral común (D8)."""
    umbrales = np.round(np.arange(0.05, 0.96, 0.05), 2) if umbrales is None else umbrales
    pred, reales = preparar_datos(pred, reales)
    aciertos = marcar_aciertos(pred, reales)
    filas = [(u, *precision_recall_en_umbral(pred, aciertos, reales, u)) for u in umbrales]
    return pd.DataFrame(filas, columns=["umbral", "precision", "recall", "F1"])
