"""Gráficos, tablas y un diagrama sueltos para la presentación de la Tarea 2 (diapositivas de Canva).

Los cuadernos guardaron sus gráficos en figuras de varios paneles, distintas en cada modelo y con los nombres de 05
(«YOLO desde cero», «YOLOv3-tiny a mano»…). Este script los rehace uno por archivo, con el mismo formato, tamaño y ejes
en todos los modelos, y con los nombres y el orden de la presentación: CNN propia, YOLOv8n, YOLOv3-tiny, YOLOv8n
preentrenado, YOLOv3-tiny preentrenada y Faster R-CNN.

Salidas, en resultados/presentacion/:
- <modelo>/, para los dos YOLO desde cero, los dos preentrenados y Faster R-CNN (la CNN ya tiene los suyos en 01):
  por época, pérdida total, de caja y de clase, mAP50 y mAP50-95 de validación y tasa de aprendizaje; en test, AP50
  por clase, curva precisión-recall, matriz de confusión y cuatro imágenes de ejemplo. Lo que solo registró un modelo
  va en <modelo>/solo_este_modelo/.
- anotacion_real/: las cajas anotadas de esas cuatro imágenes.
- comparacion/: YOLOv8n y YOLOv3-tiny desde cero en el mismo gráfico (validación, test, tipos de detección, tamaño de
  los objetos y costo).
- parte1_cnn_y_yolos/: la CNN y los dos YOLO desde cero, con tablas de test y de costo y gráficos.
- seis_modelos/: los seis modelos: tabla y barras de test, mAP50-95 de test según el tiempo de entrenamiento y según los
  FPS, y mAP50-95 de validación por época y según el tiempo.
- arquitectura/: diagrama y tabla de capas de la YOLOv3-tiny de 02b, y diagrama de la YOLOv8n de 02 (con sus capas en
  CSV), que se construye con Ultralytics en la CPU.
Las métricas salen del evaluador común de comun.py, y el script comprueba que coincidan con las tablas de 05. Los
valores nuevos se guardan también en CSV junto a su gráfico.

Solo lee lo que ya guardaron los entrenamientos y las evaluaciones: no entrena ni usa la GPU.
Uso, desde la raíz del proyecto:  py graficos_presentacion.py
"""
import json
import random
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # solo guarda archivos, sin abrir ventanas
import matplotlib.patheffects as efectos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
from PIL import Image

import comun

# ----------------------------------------------------------------------------- configuración

RAIZ = Path(__file__).resolve().parent     # la raíz del proyecto es la carpeta de este archivo
DIR_RESULTADOS = RAIZ / "resultados"
DIR_ENTRENAMIENTOS = RAIZ / "entrenamientos"   # copia de ~/tarea2_runs/reales/, con los registros completos por época
DIR_DATASET = RAIZ / "datasets" / "fjordfish"
DIR_SALIDA = DIR_RESULTADOS / "presentacion"

# Nombres de la presentación, en el orden en que se presentan; las claves son las carpetas de resultados/
MODELOS = {"cnn_propia": "CNN propia", "yolo_desde_cero": "YOLOv8n", "yolo_a_mano": "YOLOv3-tiny",
           "yolo_preentrenado": "YOLOv8n preentrenado", "yolo_a_mano_preentrenado": "YOLOv3-tiny preentrenada",
           "faster_rcnn": "Faster R-CNN"}
FILAS_05 = {"cnn_propia": "CNN propia", "yolo_desde_cero": "YOLO desde cero", "yolo_a_mano": "YOLOv3-tiny a mano",
            "yolo_preentrenado": "YOLO preentrenado", "yolo_a_mano_preentrenado": "YOLOv3-tiny a mano preentrenada",
            "faster_rcnn": "Faster R-CNN"}   # nombres en las tablas de 05
# Los colores de 05, salvo el rosado de la YOLOv3-tiny: junto al naranja de YOLOv8n, el de 05 (#e87ba4) casi no se
# distingue (diferencia de 12,9 en OKLab × 100, bajo el mínimo de 15). Este rosado más oscuro, del mismo tono, da 16,4,
# y con él los seis colores en el orden de MODELOS pasan las pruebas de separación de la paleta, también con daltonismo.
COLORES = {"cnn_propia": "#2a78d6", "yolo_desde_cero": "#eb6834", "yolo_a_mano": "#c2407a",
           "yolo_preentrenado": "#1baf7a", "yolo_a_mano_preentrenado": "#008300", "faster_rcnn": "#eda100"}
TEXTO_SOBRE_COLOR = {"faster_rcnn": "#0b0b0b"}   # texto de las etiquetas: negro sobre el amarillo, blanco sobre el resto
CON_GRAFICOS_PROPIOS = [m for m in MODELOS if m != "cnn_propia"]   # la CNN ya tiene los suyos en su diapositiva (01)
DESDE_CERO = ["cnn_propia", "yolo_desde_cero", "yolo_a_mano"]
YOLOS_DESDE_CERO = ["yolo_desde_cero", "yolo_a_mano"]
ULTRALYTICS = ["yolo_desde_cero", "yolo_preentrenado"]   # registran su entrenamiento en el results.csv de Ultralytics
# Modelos con la misma función de pérdida: sus gráficos de pérdida comparten el eje vertical
FAMILIA = {"yolo_desde_cero": "yolov8n", "yolo_preentrenado": "yolov8n", "yolo_a_mano": "yolov3_tiny",
           "yolo_a_mano_preentrenado": "yolov3_tiny", "faster_rcnn": "faster_rcnn"}

UMBRAL_CONFIANZA = json.loads((DIR_RESULTADOS / "comparacion" / "umbral.json").read_text())["umbral_confianza"]  # D8
IOU_MATRIZ = 0.5           # IoU mínimo para emparejar una detección con un objeto en la matriz de confusión
SEMILLA, N_EJEMPLOS = 42, 4   # los mismos de los cuadernos: las mismas imágenes de test que la diapositiva de la CNN
EJE_EPOCAS_YOLO = (250, 50)   # (máximo, paso) del eje de épocas, el mismo en los cuatro YOLO (de 193 a 235 épocas)
EJE_EPOCAS = {"faster_rcnn": (55, 10)}   # Faster R-CNN paró en la época 52: con el eje de los YOLO ocuparía un quinto
FUERA_DE_ESCALA = 2.5      # un valor de validación mayor que este múltiplo del máximo de train se dibuja fuera del eje
ANCHO, ALTO = 8.0, 5.6     # pulgadas de los gráficos por época: media diapositiva, para poner dos o tres lado a lado
ANCHO_COMPARACION, ALTO_COMPARACION = 10.0, 6.0   # los comparativos traen varios modelos: más anchos
ANCHO_TABLA = 16.0         # las tablas ocupan el ancho de una diapositiva
DPI = 200                  # 8 pulgadas quedan en 1600 px de ancho
IOU_ACIERTO = 0.5          # desde este IoU con un objeto de su clase, una detección es correcta (como en el mAP50)
IOU_ROZA = 0.1             # bajo este IoU con todos los objetos, la detección cae en el fondo (como en 05 y en TIDE)
LADO_ENTRADA = 640         # cada imagen entra a la red con su lado mayor en 640 px
LIMITES_TAMANO = (32, 96)  # objetos pequeños, medianos y grandes según la raíz de su área en la entrada, como en COCO
TIPOS = ["Correcta", "Duplicada", "Clase equivocada", "Mal ubicada", "Clase y ubicación", "Fondo"]
VARIACION_FPS = "hasta 40 %"   # CLAUDE.md: dos mediciones seguidas de los FPS de los modelos pequeños

FONDO, TINTA, TINTA_SECUNDARIA, GRIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
REJILLA, EJE, RELLENO_NEUTRO = "#e1e0d9", "#c3c2b7", "#f0efec"
COLOR_TRAIN = TINTA_SECUNDARIA   # gris oscuro: se distingue de los colores de modelo, también con daltonismo
MAPA_AZUL = LinearSegmentedColormap.from_list("ap", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])  # el de 05

MEJOR_EPOCA = "Línea punteada: mejor época (mayor mAP50-95 de validación), la que se evalúa en test."
# Términos de la pérdida: título y qué mide en cada familia de modelos. Los que no tienen todos van a solo_este_modelo/.
TERMINOS = {
    "total": ("Pérdida total", {
        "yolov8n": "Caja (CIoU) + DFL + clase (BCE), como la registra Ultralytics.",
        "yolov3_tiny": "Caja + objeto + fondo + clase, por imagen.",
        "faster_rcnn": "Clase y caja de la cabeza ROI + objetividad y caja de la RPN, por lote de 4 imágenes."}),
    "caja": ("Pérdida de caja", {
        "yolov8n": "Término 1 − CIoU, como lo registra Ultralytics.",
        "yolov3_tiny": "Error cuadrático de la caja en las candidatas positivas.",
        "faster_rcnn": "Smooth L1 de la caja en la cabeza ROI, por lote."}),
    "clase": ("Pérdida de clase", {
        "yolov8n": "BCE de las 28 clases, como la registra Ultralytics.",
        "yolov3_tiny": "BCE de las 28 clases en las candidatas positivas.",
        "faster_rcnn": "Entropía cruzada de las 28 clases y el fondo en la cabeza ROI, por lote."}),
    "dfl": ("Pérdida DFL", {
        "yolov8n": "Distribution focal loss: distancia a cada lado de la caja. Solo la tienen las YOLOv8n."}),
    "objeto": ("Pérdida de objeto", {
        "yolov3_tiny": "BCE de objeto en las candidatas positivas. Solo la tienen las YOLOv3-tiny."}),
    "fondo": ("Pérdida de fondo", {
        "yolov3_tiny": "BCE de objeto en las demás candidatas, sin las ignoradas. Solo la tienen las YOLOv3-tiny."}),
    "rpn": ("Pérdida de la RPN", {
        "faster_rcnn": "Objetividad (BCE) y caja (smooth L1) de la red de propuestas. Solo la tiene Faster R-CNN."}),
}
# Nota sobre el eje vertical: completa en los términos que tienen todos los modelos y corta en los de una sola familia
NOTA_ESCALA = {"yolov8n": "Mismo eje en las dos YOLOv8n; no se compara con otros modelos.",
               "yolov3_tiny": "Mismo eje en las dos YOLOv3-tiny; no se compara con otros modelos.",
               "faster_rcnn": "Escala propia: no se compara con otros modelos."}
NOTA_ESCALA_PROPIA = {"yolov8n": "Mismo eje en las dos YOLOv8n.", "yolov3_tiny": "Mismo eje en las dos YOLOv3-tiny.",
                      "faster_rcnn": ""}
SUBTITULO_TASA = {
    "yolov8n": "Tasa de los pesos (SGD): 3 épocas de calentamiento y caída lineal desde 0,01, que llegaría a 0,0001 en "
               "la época 300.",
    "faster_rcnn": "Tasa de los pesos (SGD): constante en 0,005, sin calentamiento ni caída."}
SUBTITULO_TASA["yolov3_tiny"] = SUBTITULO_TASA["yolov8n"]   # 02b reproduce el calentamiento y la caída de Ultralytics

# YOLOv3-tiny de 02b (ARQUITECTURA de su sección 4), por tramos. ("conv", tamaño, filtros) lleva BatchNorm y LeakyReLU;
# ("salida", 1, filtros) es una convolución con sesgo y sin activación; ("mp", stride) es un max-pool 2×2.
TRAMOS_YOLOV3_TINY = {
    "Cuerpo hasta stride 16": [("conv", 3, 16), ("mp", 2), ("conv", 3, 32), ("mp", 2), ("conv", 3, 64), ("mp", 2),
                               ("conv", 3, 128), ("mp", 2), ("conv", 3, 256)],
    "Cuerpo hasta stride 32": [("mp", 2), ("conv", 3, 512), ("mp", 1), ("conv", 3, 1024), ("conv", 1, 256)],
    "Cabeza de stride 32": [("conv", 3, 512), ("salida", 1, 99)],
    "Puente": [("conv", 1, 128), ("ampliar",)],
    "Cabeza de stride 16": [("concatenar",), ("conv", 3, 256), ("salida", 1, 99)],
}
COLOR_YOLOV3_TINY = COLORES["yolo_a_mano"]   # rosado oscuro: se lee texto blanco encima


# ----------------------------------------------------------------------------- formato

def configurar_estilo():
    """Estilo de 05 (paleta, fondo y rejilla), con letra más grande para que se lea en una diapositiva."""
    plt.rcParams.update({
        "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
        "axes.edgecolor": EJE, "axes.labelcolor": TINTA_SECUNDARIA, "text.color": TINTA,
        "xtick.color": TINTA_SECUNDARIA, "ytick.color": TINTA_SECUNDARIA,
        "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.9, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 13, "axes.labelsize": 13.5, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
        "legend.fontsize": 13, "legend.frameon": False,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def decimal(valor, decimales=2):
    """Número con coma decimal, como se escribe en español."""
    return f"{valor:.{decimales}f}".replace(".", ",")


def entero(valor):
    """Entero con punto de miles (1.963)."""
    return f"{round(valor):,}".replace(",", ".")


def con_coma(eje, decimales=None):
    """Marcas de un eje con coma decimal; sin `decimales`, cada marca usa los que necesita (5, 2,5, 10)."""
    formato = (lambda v, _: decimal(v, decimales)) if decimales is not None else (lambda v, _: f"{v:g}".replace(".", ","))
    eje.set_major_formatter(FuncFormatter(formato))


def ancho_texto(texto, puntos, negrita=False):
    """Ancho aproximado (pulgadas) de una línea de texto en la letra de los gráficos (DejaVu Sans)."""
    return len(texto) * puntos * (0.62 if negrita else 0.55) / 72


def envolver(texto, ancho_pulgadas, puntos):
    """Corta un texto en líneas que caben en `ancho_pulgadas` con letra de `puntos`."""
    return textwrap.wrap(texto, width=int(ancho_pulgadas * 72 / (puntos * 0.53)), break_on_hyphens=False)


def escribir_encabezado(fig, ancho, alto, titulo, subtitulo, lineas_reservadas):
    """Título y subtítulo arriba a la izquierda; devuelve cuántas pulgadas ocupan desde el borde superior."""
    lineas = textwrap.wrap(subtitulo, width=int((ancho - 0.3) * 9.8))
    if len(lineas) > lineas_reservadas:   # el mismo gráfico de otro modelo quedaría de otro tamaño
        print(f"    aviso: el subtítulo de «{titulo}» usa {len(lineas)} líneas y se reservaron {lineas_reservadas}")
    fig.text(0.13 / ancho, 1 - 0.16 / alto, titulo, fontsize=17, fontweight="bold", va="top")
    fig.text(0.13 / ancho, 1 - 0.58 / alto, "\n".join(lineas), fontsize=12.5, color=TINTA_SECUNDARIA, va="top",
             linespacing=1.4)
    return 0.58 + 0.245 * max(len(lineas), lineas_reservadas) + 0.1


def crear_figura(titulo, subtitulo, marcas=None, ancho=ANCHO, alto=ALTO, margenes=(0.95, 0.35, 0.75), lineas_reservadas=2,
                 columnas_leyenda=None):
    """Figura de un gráfico con encabezado (título, subtítulo y, si hay más de una serie, leyenda) y los ejes debajo.

    Todo se mide en pulgadas y el encabezado reserva siempre las mismas líneas, para que el mismo gráfico de dos
    modelos quede exactamente del mismo tamaño. `margenes` = (izquierdo, derecho, inferior) de los ejes.
    """
    fig = plt.figure(figsize=(ancho, alto))
    arriba = escribir_encabezado(fig, ancho, alto, titulo, subtitulo, lineas_reservadas)
    if marcas:
        columnas = columnas_leyenda or len(marcas)
        fig.legend(handles=marcas, loc="upper left", bbox_to_anchor=(0.13 / ancho, 1 - arriba / alto), ncol=columnas,
                   handlelength=1.8, handletextpad=0.6, columnspacing=2.4, borderaxespad=0, borderpad=0)
        arriba += 0.32 * -(-len(marcas) // columnas)   # una fila de leyenda cada `columnas` marcas
    arriba += 0.18
    izquierdo, derecho, inferior = margenes
    ax = fig.add_axes((izquierdo / ancho, inferior / alto, 1 - (izquierdo + derecho) / ancho,
                       (alto - arriba - inferior) / alto))
    return fig, ax


def guardar(fig, ruta):
    """Guarda la figura con su tamaño exacto (sin recortar, para que todos los modelos queden iguales) y la cierra."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=DPI)
    plt.close(fig)
    print("  " + str(ruta.relative_to(RAIZ)))


ETIQUETA_SOBRE_LINEAS = dict(boxstyle="square,pad=0.15", fc=FONDO, ec="none", alpha=0.85)   # fondo para leerla sobre una línea


def marcar_mejor_epoca(ax, mejor_epoca, tope):
    """Línea punteada vertical en la mejor época, con su número arriba a la derecha de la línea."""
    ax.axvline(mejor_epoca, color=GRIS, lw=1.3, ls=(0, (5, 4)), zorder=1)
    ax.annotate(f"Mejor época: {mejor_epoca}", (mejor_epoca, tope * 0.98), xytext=(6, 0), textcoords="offset points",
                ha="left", va="top", color=TINTA_SECUNDARIA, fontsize=12, bbox=ETIQUETA_SOBRE_LINEAS)


def eje_de_epocas(ax, m=None):
    """Eje de épocas: el mismo en todos los YOLO; Faster R-CNN, que entrenó muchas menos, tiene el suyo."""
    maximo, paso = EJE_EPOCAS.get(m, EJE_EPOCAS_YOLO)
    ax.set_xlim(0, maximo)
    ax.xaxis.set_major_locator(MultipleLocator(paso))
    ax.set_xlabel("Época")


def orden_de_leyenda(marcas, columnas):
    """Reordena las marcas para que la leyenda se lea por filas: matplotlib la llena columna por columna."""
    filas = -(-len(marcas) // columnas)
    return [marcas[f * columnas + c] for c in range(columnas) for f in range(filas) if f * columnas + c < len(marcas)]


POSICIONES_ETIQUETA = [(-9, 7, "right", "bottom"), (9, 7, "left", "bottom"), (9, -9, "left", "top"),
                       (-9, -9, "right", "top"), (0, 12, "center", "bottom"), (0, -12, "center", "top")]
POSICIONES_NOMBRE = [(10, 0, "left", "center"), (-10, 0, "right", "center"), (0, 10, "center", "bottom"),
                     (0, -10, "center", "top"), (9, 7, "left", "bottom"), (9, -7, "left", "top"),
                     (-9, 7, "right", "bottom"), (-9, -7, "right", "top")]   # para el nombre junto a un punto


def contar_dentro(puntos, caja):
    """Cuántos puntos (en píxeles) caen dentro de un rectángulo."""
    return int(((puntos[:, 0] >= caja.x0) & (puntos[:, 0] <= caja.x1) &
                (puntos[:, 1] >= caja.y0) & (puntos[:, 1] <= caja.y1)).sum())


def anotar_sin_tapar(ax, texto, punto, ocupadas, posiciones=POSICIONES_ETIQUETA, **estilo):
    """Escribe `texto` junto a `punto` donde tape menos lo dibujado y no choque con otras etiquetas.

    Prueba las `posiciones` (desplazamiento en puntos y alineación) y se queda con la que cubre menos puntos de las
    líneas y de los marcadores (cada marcador cuenta como 50 puntos de línea), sin salirse de los ejes. Los límites de
    los ejes deben estar fijos antes de llamarla. `ocupadas` es la lista de rectángulos de las etiquetas ya puestas, y
    se le agrega el de esta.
    """
    renderer = ax.figure.canvas.get_renderer()
    trazos = []
    for linea in ax.lines:   # puntos de cada línea en píxeles, con 3 intermedios por tramo
        xy = ax.transData.transform(linea.get_xydata())
        if len(xy) > 1:
            t, i = np.linspace(0, len(xy) - 1, 4 * len(xy) - 3), np.arange(len(xy))
            trazos.append(np.column_stack([np.interp(t, i, xy[:, 0]), np.interp(t, i, xy[:, 1])]))
    puntos = np.concatenate(trazos) if trazos else np.empty((0, 2))
    marcadores = [ax.transData.transform(c.get_offsets()) for c in ax.collections if len(c.get_offsets())]
    marcadores = np.concatenate(marcadores) if marcadores else np.empty((0, 2))
    caja_ejes = ax.get_window_extent(renderer)
    opciones = []
    for dx, dy, ha, va in posiciones:
        prueba = ax.annotate(texto, punto, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, **estilo)
        caja = prueba.get_window_extent(renderer).expanded(1.06, 1.25)
        prueba.remove()
        tapados = contar_dentro(puntos, caja) + 50 * contar_dentro(marcadores, caja)
        fuera = not (caja_ejes.x0 <= caja.x0 and caja.x1 <= caja_ejes.x1 and caja_ejes.y0 <= caja.y0 and caja.y1 <= caja_ejes.y1)
        choques = sum(caja.overlaps(otra) for otra in ocupadas)
        opciones.append((tapados + 1000 * (fuera + choques), dx, dy, ha, va, caja))
    _, dx, dy, ha, va, caja = min(opciones, key=lambda o: o[0])   # con empate, la primera posición de la lista
    ax.annotate(texto, punto, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, **estilo)
    ocupadas.append(caja)


# ----------------------------------------------------------------------------- datos

def registro_por_epoca(m):
    """Registro por época de un modelo, con nombres comunes para todos.

    Columnas: epoca, tiempo_s, tiempo_entrenamiento_s, perdida_<término>_<train|val>, val_mAP50, val_mAP50_95 y tasa
    (la de los pesos), más las que solo registró ese modelo (YOLOv8n: DFL, precisión y recall de validación;
    YOLOv3-tiny: objeto, fondo y norma del gradiente; Faster R-CNN: RPN). tiempo_entrenamiento_s es el tiempo
    acumulado sin la pasada extra que hace Faster R-CNN para calcular la pérdida de validación, como en 05; en los
    demás modelos es igual a tiempo_s.
    """
    if m in ULTRALYTICS:
        r = pd.read_csv(DIR_ENTRENAMIENTOS / m / "entrenamiento" / "results.csv")
        r.columns = r.columns.str.strip()
        registro = pd.DataFrame({
            "epoca": r["epoch"], "tiempo_s": r["time"],
            "perdida_caja_train": r["train/box_loss"], "perdida_caja_val": r["val/box_loss"],
            "perdida_clase_train": r["train/cls_loss"], "perdida_clase_val": r["val/cls_loss"],
            "perdida_dfl_train": r["train/dfl_loss"], "perdida_dfl_val": r["val/dfl_loss"],
            "val_precision": r["metrics/precision(B)"], "val_recall": r["metrics/recall(B)"],
            "val_mAP50": r["metrics/mAP50(B)"], "val_mAP50_95": r["metrics/mAP50-95(B)"],
            "tasa": r["lr/pg0"],
        })
        for particion in ("train", "val"):   # la total es la suma de los tres términos, como en 02 y 03
            registro[f"perdida_total_{particion}"] = sum(registro[f"perdida_{t}_{particion}"] for t in ("caja", "clase", "dfl"))
    else:
        registro = pd.read_csv(DIR_ENTRENAMIENTOS / m / "historial_completo.csv").rename(
            columns={"perdida_train": "perdida_total_train", "perdida_val": "perdida_total_val"})
        if "tasa" not in registro:   # la CNN y Faster R-CNN usaron una tasa constante, que no registraron por época
            info = json.loads((DIR_ENTRENAMIENTOS / m / "info_entrenamiento.json").read_text(encoding="utf-8"))
            registro["tasa"] = info["configuracion"]["tasa_aprendizaje"]
    # debe coincidir con el historial.csv que guardó el cuaderno del modelo
    historial = pd.read_csv(DIR_RESULTADOS / m / "historial.csv")
    for columna, propia in (("tiempo_s", "tiempo_s"), ("perdida_train", "perdida_total_train"),
                            ("perdida_val", "perdida_total_val"), ("val_mAP50", "val_mAP50"), ("val_mAP50_95", "val_mAP50_95")):
        assert np.allclose(historial[columna], registro[propia], atol=1e-3), f"{m}: {propia} no coincide con historial.csv"
    pasada_extra = registro["segundos_perdida_val"].cumsum() if "segundos_perdida_val" in registro else 0.0
    registro["tiempo_entrenamiento_s"] = registro["tiempo_s"] - pasada_extra
    return registro


def comprobar_con_05(m, evaluacion):
    """Detiene el script si las métricas de test no son las de las tablas de 05 (por ejemplo, tras reentrenar)."""
    tabla = pd.read_csv(DIR_RESULTADOS / "comparacion" / "desempeno_test.csv", index_col=0).loc[FILAS_05[m]]
    detalle = pd.read_csv(DIR_RESULTADOS / "comparacion" / "detalle_precision_test.csv", index_col=0).loc[FILAS_05[m]]
    for fila, columna, clave in [(tabla, "Precisión", "precision"), (tabla, "Recall", "recall"), (tabla, "F1", "F1"),
                                 (tabla, "mAP50", "mAP50"), (tabla, "mAP75", "mAP75"), (tabla, "mAP50-95", "mAP50_95"),
                                 (detalle, "Clases con detecciones", "clases_con_detecciones"),
                                 (detalle, "Precisión micro", "precision_micro")]:
        assert np.isclose(fila[columna], evaluacion[clave], atol=1e-9), \
            f"{m}: {columna} de test da {evaluacion[clave]:.4f} y en 05 {fila[columna]:.4f}; vuelva a ejecutar 05"


def cargar_modelo(m, reales, nombres):
    """Registro por época, mejor época, detecciones y evaluación común de test de un modelo, con sus comprobaciones."""
    resumen = json.loads((DIR_RESULTADOS / m / "resumen.json").read_text(encoding="utf-8"))
    assert resumen["huella_particion"] == comun.HUELLA_ESPERADA, f"{m}: particiones distintas de las comunes"
    registro = registro_por_epoca(m)
    mejor = int(resumen["mejor_epoca"])
    assert registro.loc[registro.val_mAP50_95.idxmax(), "epoca"] == mejor, f"{m}: la mejor época no es la de mayor mAP50-95"
    detecciones = pd.read_csv(DIR_RESULTADOS / m / "predicciones_test.csv").astype({"id": str})
    evaluacion = comun.evaluar_detecciones(detecciones, reales, nombres, UMBRAL_CONFIANZA)
    comprobar_con_05(m, evaluacion)
    return {"registro": registro, "mejor": mejor, "detecciones": detecciones, "evaluacion": evaluacion}


def matriz_confusion(detecciones, reales, n_clases):
    """Matriz de confusión de detección, con el método de Ultralytics: filas = clase predicha, columnas = clase real.

    En cada imagen, las detecciones con confianza ≥ UMBRAL_CONFIANZA se recorren de mayor a menor confianza, y cada una
    toma el objeto todavía libre con el que tiene mayor IoU, sin mirar la clase, si ese IoU es ≥ IOU_MATRIZ. La última
    fila (fondo) cuenta los objetos que ninguna detección tomó, y la última columna, las detecciones sin objeto.
    """
    pred, reales = comun.preparar_datos(detecciones, reales)   # ordena por confianza y valida las tablas
    pred = pred[pred.confianza >= UMBRAL_CONFIANZA]
    pred_por_imagen = dict(tuple(pred.groupby("id", sort=False)))
    matriz = np.zeros((n_clases + 1, n_clases + 1), dtype=int)
    for id_img, objetos in reales.groupby("id"):
        clases_reales = objetos.clase_id.to_numpy()
        libre = np.ones(len(objetos), dtype=bool)
        p = pred_por_imagen.get(id_img)
        if p is not None:
            iou = comun.iou(p[comun.COLUMNAS_CAJA].to_numpy(float), objetos[comun.COLUMNAS_CAJA].to_numpy(float))
            for clase_pred, iou_fila in zip(p.clase_id.to_numpy(), iou):
                candidatas = np.where(libre & (iou_fila >= IOU_MATRIZ), iou_fila, -1.0)
                j = candidatas.argmax()
                if candidatas[j] >= 0:
                    libre[j] = False
                    matriz[clase_pred, clases_reales[j]] += 1
                else:
                    matriz[clase_pred, n_clases] += 1
        np.add.at(matriz, (n_clases, clases_reales[libre]), 1)
    return matriz


# ----------------------------------------------------------------------------- gráficos por época

def tope_perdida(registro, termino):
    """Tope del eje de una pérdida: el máximo de train o de validación, sin los picos de validación muy por encima."""
    train, val = registro[f"perdida_{termino}_train"], registro[f"perdida_{termino}_val"]
    return 1.12 * max(train.max(), val[val <= FUERA_DE_ESCALA * train.max()].max())


def grafico_perdida(m, registro, mejor, termino, tope, ruta):
    """Pérdida de train y de validación de un término, por época, con el eje vertical de su familia de modelos."""
    titulo, descripciones = TERMINOS[termino]
    familia = FAMILIA[m]
    series = {"train": ("Train", COLOR_TRAIN, 1.8), "val": ("Validación", COLORES[m], 2.2)}
    marcas = [Line2D([], [], color=color, lw=3, label=texto) for texto, color, _ in series.values()]
    nota = (NOTA_ESCALA if len(descripciones) > 1 else NOTA_ESCALA_PROPIA)[familia]
    fig, ax = crear_figura(f"{MODELOS[m]}: {titulo.lower()}", f"{descripciones[familia]} {nota}".strip(), marcas)
    val = registro[f"perdida_{termino}_val"]
    for particion, (_, color, grosor) in series.items():
        ax.plot(registro.epoca, registro[f"perdida_{termino}_{particion}"], color=color, lw=grosor)
    for _, fila in registro[val > tope].iterrows():   # picos de validación que no caben en el eje
        ax.annotate(f"Validación en la época {int(fila.epoca)}:\n{decimal(fila[val.name], 1)} (fuera de escala)",
                    xy=(fila.epoca, tope), xytext=(26, -16), textcoords="offset points", va="top", fontsize=12,
                    color=TINTA_SECUNDARIA, arrowprops=dict(arrowstyle="-|>", color=GRIS, lw=1.2))
    marcar_mejor_epoca(ax, mejor, tope)
    eje_de_epocas(ax, m)
    ax.set_ylim(0, tope)
    ax.set_ylabel(titulo)
    con_coma(ax.yaxis)
    guardar(fig, ruta)


def grafico_por_epoca(m, registro, mejor, columna, titulo, subtitulo, etiqueta_y, tope, ruta, decimales_eje,
                      decimales_valor=None, limite=None):
    """Una serie por época (una métrica de validación, la tasa o la norma del gradiente), con la mejor época marcada.

    Con `decimales_valor`, escribe el valor de la serie en la mejor época; con `limite`, dibuja esa línea horizontal.
    """
    fig, ax = crear_figura(f"{MODELOS[m]}: {titulo}", subtitulo, margenes=(1.05, 0.35, 0.75))
    ax.plot(registro.epoca, registro[columna], color=COLORES[m], lw=2)
    if decimales_valor is not None:   # el valor en la mejor época, a la izquierda del punto
        valor = registro.loc[registro.epoca == mejor, columna].iloc[0]
        ax.scatter(mejor, valor, s=80, color=COLORES[m], edgecolor=FONDO, linewidth=2, zorder=4)
        ax.annotate(decimal(valor, decimales_valor), (mejor, valor), xytext=(-9, 7), textcoords="offset points",
                    ha="right", va="bottom", color=TINTA, fontsize=13, fontweight="bold", bbox=ETIQUETA_SOBRE_LINEAS)
    if limite is not None:
        ax.axhline(limite, color=GRIS, lw=1.3, ls=(0, (1.5, 2.5)))
    marcar_mejor_epoca(ax, mejor, tope)
    eje_de_epocas(ax, m)
    ax.set_ylim(0, tope)
    ax.set_ylabel(etiqueta_y)
    con_coma(ax.yaxis, decimales_eje)
    guardar(fig, ruta)


# ----------------------------------------------------------------------------- gráficos de test

def grafico_ap50_por_clase(m, evaluacion, n_clases, ruta):
    """AP50 de cada clase en test, en el mismo orden para todos los modelos (de la clase más a la menos frecuente)."""
    por_clase = evaluacion["por_clase"].sort_values(["instancias", "clase_id"], ascending=[False, True])
    subtitulo = (f"Test, evaluador común (mAP50 = {decimal(evaluacion['mAP50'], 3)}). Entre paréntesis, objetos en test; "
                 f"mismo orden en todos los modelos. {n_clases - len(por_clase)} clases no tienen objetos en test.")
    fig, ax = crear_figura(f"{MODELOS[m]}: AP50 por clase", subtitulo, alto=10, margenes=(2.55, 0.4, 0.75))
    y = np.arange(len(por_clase))[::-1]
    ax.barh(y, por_clase.AP50, height=0.66, color=COLORES[m])
    for yi, valor in zip(y, por_clase.AP50):
        ax.text(valor + 0.012, yi, decimal(valor), va="center", ha="left", fontsize=12, color=TINTA_SECUNDARIA)
    ax.set_yticks(y, [f"{c} ({n})" for c, n in zip(por_clase.clase, por_clase.instancias)], fontsize=12)
    ax.set_ylim(-0.6, len(por_clase) - 0.4)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AP50 en test")
    con_coma(ax.xaxis, 1)
    guardar(fig, ruta)


def grafico_curva_pr(m, evaluacion, ruta):
    """Curva precisión-recall media entre clases en test, con IoU 0,5."""
    curva = evaluacion["curva_pr"]
    subtitulo = ("Test, IoU 0,5, evaluador común. Precisión media entre las clases en cada nivel de recall; el área "
                 f"bajo la curva es el mAP50 ({decimal(evaluacion['mAP50'], 3)}).")
    fig, ax = crear_figura(f"{MODELOS[m]}: curva precisión-recall", subtitulo, alto=6.8, margenes=(1.0, 0.35, 0.75))
    ax.fill_between(curva.recall, curva.precision, color=COLORES[m], alpha=0.12, lw=0)
    ax.plot(curva.recall, curva.precision, color=COLORES[m], lw=2.2)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precisión")
    con_coma(ax.xaxis, 1)
    con_coma(ax.yaxis, 1)
    guardar(fig, ruta)


def grafico_matriz_confusion(m, matriz, nombres, ruta):
    """Matriz de confusión de test normalizada por columna (cada clase real suma 1), como la de Ultralytics."""
    totales = matriz.sum(axis=0, keepdims=True)
    normalizada = np.divide(matriz, totales, out=np.full(matriz.shape, np.nan), where=totales > 0)
    normalizada[normalizada < 0.005] = np.nan   # las celdas vacías quedan del color del fondo
    subtitulo = (f"Test. Cada columna es una clase real y suma 1: la diagonal es la fracción de sus objetos detectada con "
                 f"la clase correcta. Detecciones con confianza ≥ {decimal(UMBRAL_CONFIANZA)}, emparejadas con los objetos "
                 f"por IoU ≥ {decimal(IOU_MATRIZ, 1)} sin mirar la clase, como en Ultralytics. Fila «fondo»: objetos sin "
                 "detección; columna «fondo»: detecciones sin objeto. Las columnas vacías son clases sin objetos en test.")
    fig, ax = crear_figura(f"{MODELOS[m]}: matriz de confusión en test", subtitulo, ancho=12, alto=12.9,
                           margenes=(2.45, 1.25, 2.35), lineas_reservadas=4)
    imagen = ax.imshow(normalizada, cmap=MAPA_AZUL, vmin=0, vmax=1, interpolation="none", aspect="auto")
    for i, j in zip(*np.nonzero(~np.isnan(normalizada))):
        valor = normalizada[i, j]
        ax.text(j, i, decimal(valor), ha="center", va="center", fontsize=8, color="white" if valor >= 0.45 else TINTA)
    etiquetas = [*nombres.values(), "fondo"]
    ax.set_xticks(range(len(etiquetas)), etiquetas, rotation=90, fontsize=11)
    ax.set_yticks(range(len(etiquetas)), etiquetas, fontsize=11)
    ax.set_xlabel("Clase real", labelpad=10)
    ax.set_ylabel("Clase predicha", labelpad=10)
    ax.grid(False)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(True)
    caja = ax.get_position()
    barra = fig.add_axes((caja.x1 + 0.3 / 12, caja.y0, 0.22 / 12, caja.height))
    fig.colorbar(imagen, cax=barra)
    con_coma(barra.yaxis, 1)
    barra.tick_params(labelsize=11.5)
    guardar(fig, ruta)


def dibujar_imagen(ruta_imagen, cajas, color_caja, color_etiqueta, color_texto, titulo, subtitulo, nombres, ruta):
    """Una imagen de test con sus cajas (reales o detectadas), con la clase y, si la hay, la confianza."""
    imagen = Image.open(ruta_imagen)
    ancho_px, alto_px = imagen.size
    alto_imagen = ANCHO * alto_px / ancho_px
    encabezado = 0.58 + 0.245 + 0.25
    alto = alto_imagen + encabezado
    fig = plt.figure(figsize=(ANCHO, alto))
    escribir_encabezado(fig, ANCHO, alto, titulo, subtitulo, lineas_reservadas=1)
    ax = fig.add_axes((0, 0, 1, alto_imagen / alto))
    ax.imshow(imagen)
    ax.axis("off")
    contorno = [efectos.withStroke(linewidth=4.5, foreground=(0, 0, 0, 0.55))]   # se ve sobre fondos claros y oscuros
    px_por_punto = ancho_px / (ANCHO * 72)                   # píxeles de la imagen por punto tipográfico
    alto_etiqueta, margen = 21 * px_por_punto, 2.2 * px_por_punto   # alto de una etiqueta y su margen, en píxeles
    if "confianza" in cajas:
        cajas = cajas.sort_values("confianza", ascending=False)   # las más seguras quedan pegadas a su caja
    ocupadas = []                                             # rectángulos de las etiquetas ya puestas
    for fila in cajas.itertuples():
        ax.add_patch(plt.Rectangle((fila.x1, fila.y1), fila.x2 - fila.x1, fila.y2 - fila.y1, fill=False, ec=color_caja,
                                   lw=2.6, path_effects=contorno))
        texto = nombres[fila.clase_id] + (f" {decimal(fila.confianza)}" if hasattr(fila, "confianza") else "")
        ancho_texto = (7.6 * len(texto) + 6) * px_por_punto
        # Alineada con el borde izquierdo de la caja, o con el derecho si se saldría de la imagen
        x0 = max(fila.x1 if fila.x1 + ancho_texto <= ancho_px else fila.x2 - ancho_texto, 0)
        # Sobre la caja si cabe bajo el borde superior de la imagen; si no, dentro de la caja
        y0, paso = (fila.y1 - alto_etiqueta, -alto_etiqueta) if fila.y1 >= alto_etiqueta else (fila.y1, alto_etiqueta)
        for _ in range(8):   # si choca con otra etiqueta (cajas casi iguales del NMS multietiqueta), se corre una fila
            if not any(x0 < b[2] and b[0] < x0 + ancho_texto and y0 < b[3] and b[1] < y0 + alto_etiqueta for b in ocupadas):
                break
            y0 += paso
            if y0 < 0:       # llegó al borde superior de la imagen: sigue dentro de la caja, hacia abajo
                y0, paso = fila.y1, alto_etiqueta
        ocupadas.append((x0, y0, x0 + ancho_texto, y0 + alto_etiqueta))
        ax.text(x0 + margen, y0 + margen, texto, ha="left", va="top", fontsize=12.5, color=color_texto,
                bbox=dict(fc=color_etiqueta, ec="none", pad=2.2), clip_on=False)
    guardar(fig, ruta)


def ejemplos_de_test():
    """Las mismas imágenes de test que muestran los cuadernos: muestra al azar con su semilla."""
    rutas = sorted((DIR_DATASET / "images" / "test").glob("*.jpg"))
    return random.Random(SEMILLA).sample(rutas, N_EJEMPLOS)


# ----------------------------------------------------------------------------- cálculos de los gráficos comparativos

def map_por_umbral_iou(detecciones, reales):
    """mAP de test en cada umbral de IoU (0,50; 0,55; …; 0,95), con las mismas funciones del evaluador común."""
    pred, reales = comun.preparar_datos(detecciones, reales)
    aciertos = comun.marcar_aciertos(pred, reales)
    clases = pred.clase_id.to_numpy()
    por_clase = [comun.precision_interpolada(aciertos[clases == c], n).mean(axis=0)
                 for c, n in reales.clase_id.value_counts().sort_index().items()]
    return np.mean(por_clase, axis=0)


def clasificar_detecciones(detecciones, reales):
    """Tipo de cada detección con confianza ≥ UMBRAL_CONFIANZA y qué objetos reales quedaron bien detectados.

    Empareja como comun.marcar_aciertos con IoU 0,5: en cada imagen, de mayor a menor confianza, cada detección toma el
    objeto libre de su clase con mayor IoU. Las que no toman ninguno se clasifican, en este orden, por su mayor IoU con
    los objetos: duplicada (≥ 0,5 con un objeto de su clase que ya tomó otra), clase equivocada (≥ 0,5 con uno de otra
    clase), mal ubicada (entre 0,1 y 0,5 con uno de su clase), clase y ubicación (entre 0,1 y 0,5 con uno de otra clase)
    o fondo (< 0,1 con todos). Son los tipos de error de TIDE (Bolya et al., 2020), con sus mismos límites.
    Devuelve (tipos: una Serie por detección; detectado: una Serie booleana con el índice de `reales`).
    """
    pred, reales = comun.preparar_datos(detecciones, reales)   # ordena por confianza y valida las tablas
    pred = pred[pred.confianza >= UMBRAL_CONFIANZA]
    objetos_por_imagen = dict(tuple(reales.groupby("id")))
    detectado = pd.Series(False, index=reales.index)
    tipos = []
    for id_img, p in pred.groupby("id", sort=False):
        objetos = objetos_por_imagen[id_img]
        iou = comun.iou(p[comun.COLUMNAS_CAJA].to_numpy(float), objetos[comun.COLUMNAS_CAJA].to_numpy(float))
        misma_clase = p.clase_id.to_numpy()[:, None] == objetos.clase_id.to_numpy()[None, :]
        libre = np.ones(len(objetos), dtype=bool)
        for iou_fila, misma in zip(iou, misma_clase):
            candidatas = np.where(misma & libre & (iou_fila >= IOU_ACIERTO), iou_fila, -1.0)
            if candidatas.max() >= 0:
                libre[candidatas.argmax()] = False
                tipos.append("Correcta")
                continue
            iou_misma, iou_otra = iou_fila[misma].max(initial=0.0), iou_fila[~misma].max(initial=0.0)
            if iou_misma >= IOU_ACIERTO:
                tipos.append("Duplicada")
            elif iou_otra >= IOU_ACIERTO:
                tipos.append("Clase equivocada")
            elif iou_misma >= IOU_ROZA:
                tipos.append("Mal ubicada")
            elif iou_otra >= IOU_ROZA:
                tipos.append("Clase y ubicación")
            else:
                tipos.append("Fondo")
        detectado[objetos.index[~libre]] = True
    return pd.Series(tipos, dtype=object), detectado


def recall_segun_tamano(detectado, reales, tamanos):
    """Objetos y recall por objeto de cada grupo de tamaño, y de todos juntos.

    El tamaño es la raíz del área de la caja en la imagen de entrada, que llega a la red con su lado mayor en LADO_ENTRADA.
    """
    escala = reales.id.map(lambda id_img: LADO_ENTRADA / max(tamanos[id_img]))
    lado = np.sqrt((reales.x2 - reales.x1) * (reales.y2 - reales.y1)) * escala
    grupo = pd.cut(lado, [0, *LIMITES_TAMANO, np.inf], right=False, labels=["Pequeños", "Medianos", "Grandes"])
    tabla = detectado.groupby(grupo, observed=False).agg(["size", "mean"]).set_axis(["objetos", "recall"], axis=1)
    tabla.index = tabla.index.astype(str)
    tabla.loc["Todos"] = [len(detectado), detectado.mean()]
    return tabla.astype({"objetos": int})


# ----------------------------------------------------------------------------- gráficos comparativos

def marcas_juntos(modelos, tipo="linea", etiquetas=None):
    """Leyenda de varios modelos con los colores de los gráficos comparativos."""
    etiquetas = etiquetas or MODELOS
    if tipo == "linea":
        return [Line2D([], [], color=COLORES[m], lw=3, label=etiquetas[m]) for m in modelos]
    if tipo == "punto":
        return [Line2D([], [], color=COLORES[m], ls="", marker="o", ms=11, label=etiquetas[m]) for m in modelos]
    return [Patch(facecolor=COLORES[m], label=etiquetas[m]) for m in modelos]


def leyenda_de_modelos(modelos, tipo):
    """Marcas de la leyenda y columnas: una fila hasta 4 modelos; con más, 3 columnas que se leen por filas."""
    columnas = 3 if len(modelos) > 4 else len(modelos)
    return orden_de_leyenda(marcas_juntos(modelos, tipo), columnas), columnas


NOTA_PASADA_EXTRA = "En Faster R-CNN, sin la pasada extra que hace para calcular la pérdida de validación."


def etiquetar_barras(ax, x, valores, formato, separacion):
    """Escribe el valor sobre cada barra."""
    for xi, valor in zip(x, valores):
        ax.text(xi, valor + separacion, formato(valor), ha="center", va="bottom", fontsize=12, color=TINTA_SECUNDARIA)


def posiciones_de_barras(centros, n, k):
    """Centro y ancho de la barra del modelo k de n en cada grupo: el grupo ocupa 0,84 y deja un hueco entre barras."""
    paso = 0.84 / n
    return centros + (k - (n - 1) / 2) * paso, paso - (0.04 if n <= 3 else 0.03)


def comparar_map_val(datos, modelos, por_tiempo, ruta):
    """mAP50-95 de validación de varios modelos, por época o según el tiempo acumulado de entrenamiento."""
    punto = "El punto marca la mejor época de cada modelo, la que se evalúa en test."
    if por_tiempo:
        titulo = "mAP50-95 de validación según el tiempo de entrenamiento"
        subtitulo = f"Tiempo acumulado en la misma GPU (RTX 4060 Laptop), con la validación de cada época. {punto}"
        if "faster_rcnn" in modelos:
            subtitulo += f" {NOTA_PASADA_EXTRA}"
    else:
        titulo, subtitulo = "mAP50-95 de validación por época", f"En validación, al final de cada época. {punto}"
    marcas, columnas = leyenda_de_modelos(modelos, "linea")
    fig, ax = crear_figura(titulo, subtitulo, marcas, ancho=ANCHO_COMPARACION, alto=ALTO_COMPARACION,
                           margenes=(1.05, 0.35, 0.75), lineas_reservadas=3 if len(modelos) > 4 else 2,
                           columnas_leyenda=columnas)
    if por_tiempo:
        ax.set_xlim(0, 1.03 * max(datos[m]["registro"].tiempo_entrenamiento_s.max() / 60 for m in modelos))
        ax.set_xlabel("Tiempo de entrenamiento (min)")
    else:
        eje_de_epocas(ax)
    ax.set_ylim(0, 1.18 * max(datos[m]["registro"].val_mAP50_95.max() for m in modelos))
    mejores = {}
    for m in modelos:
        registro = datos[m]["registro"]
        x = registro.tiempo_entrenamiento_s / 60 if por_tiempo else registro.epoca
        ax.plot(x, registro.val_mAP50_95, color=COLORES[m], lw=2)
        es_mejor = registro.epoca == datos[m]["mejor"]
        mejores[m] = (x[es_mejor].iloc[0], registro.val_mAP50_95[es_mejor].iloc[0])
        ax.scatter(*mejores[m], s=80, color=COLORES[m], edgecolor=FONDO, linewidth=2, zorder=4)
    ocupadas = []   # el valor de cada mejor época, donde no tape las curvas (se ubica con todas ya dibujadas)
    for m, (x, y) in mejores.items() if len(modelos) <= 3 else ():   # con más modelos, los valores taparían las curvas
        donde = f"a los {decimal(x, 0)} min" if por_tiempo else f"época {datos[m]['mejor']}"
        anotar_sin_tapar(ax, f"{decimal(y, 3)} ({donde})", (x, y), ocupadas, fontsize=13, fontweight="bold", color=TINTA,
                         bbox=ETIQUETA_SOBRE_LINEAS)
    ax.set_ylabel("mAP50-95 de validación")
    con_coma(ax.yaxis, 2)
    guardar(fig, ruta)


def comparar_map_test_segun(datos, modelos, medida, n_imagenes, ruta):
    """mAP50-95 de test de cada modelo frente al tiempo de entrenamiento o a los FPS, con el nombre junto a su punto.

    Los tiempos y los FPS son los de la tabla de costos de 05 (`medida`: "tiempo" o "fps").
    """
    costo = pd.read_csv(DIR_RESULTADOS / "comparacion" / "costo.csv", index_col=0)
    test = f"Cada punto es un modelo: su mAP50-95 en las {n_imagenes} imágenes de test, con el evaluador común."
    fila, titulo, etiqueta_x, subtitulo = {
        "tiempo": ("Tiempo sin la pasada extra (min)", "según el tiempo de entrenamiento", "Tiempo de entrenamiento (min)",
                   f"{test} Tiempo de entrenamiento hasta la parada temprana en la misma GPU, con la validación de cada "
                   f"época. {NOTA_PASADA_EXTRA}"),
        "fps": ("FPS", "según los FPS", "FPS (lote 1)",
                f"{test} FPS: imágenes por segundo en float32 con lote 1, incluidos el preproceso y el NMS. En los modelos "
                f"pequeños, dos mediciones seguidas variaron {VARIACION_FPS}."),
    }[medida]
    marcas, columnas = leyenda_de_modelos(modelos, "punto")
    fig, ax = crear_figura(f"mAP50-95 de test {titulo}", subtitulo, marcas, ancho=ANCHO_COMPARACION, alto=ALTO_COMPARACION,
                           margenes=(1.05, 0.35, 0.75), lineas_reservadas=3, columnas_leyenda=columnas)
    puntos = {m: (costo.loc[fila, FILAS_05[m]], datos[m]["evaluacion"]["mAP50_95"]) for m in modelos}
    ax.set_xlim(0, 1.25 * max(x for x, _ in puntos.values()))
    ax.set_ylim(0, 1.18 * max(y for _, y in puntos.values()))
    for m, (x, y) in puntos.items():
        ax.scatter(x, y, s=120, color=COLORES[m], edgecolor=FONDO, linewidth=2, zorder=4)
    ocupadas = []   # el nombre de cada modelo, donde no tape otros puntos ni otros nombres
    for m, (x, y) in puntos.items():
        anotar_sin_tapar(ax, MODELOS[m], (x, y), ocupadas, posiciones=POSICIONES_NOMBRE, fontsize=12.5, color=TINTA,
                         bbox=ETIQUETA_SOBRE_LINEAS)
    ax.set_xlabel(etiqueta_x)
    ax.set_ylabel("mAP50-95 en test")
    con_coma(ax.yaxis, 2)
    guardar(fig, ruta)


def comparar_metricas_test(datos, modelos, n_imagenes, ruta, ancho=ANCHO_COMPARACION, alto=ALTO_COMPARACION,
                           columnas=None):
    """Precisión, recall y F1 (con el umbral común) y mAP de test de varios modelos, con el evaluador común.

    Con dos modelos, cada barra lleva su valor; con más, los valores van en la tabla que acompaña al gráfico.
    `columnas` de la leyenda: por omisión, una fila hasta 4 modelos y 3 columnas con más.
    """
    metricas = [("Precisión", "precision"), ("Recall", "recall"), ("F1", "F1"), ("mAP50", "mAP50"), ("mAP75", "mAP75"),
                ("mAP50-95", "mAP50_95")]
    subtitulo = (f"Evaluador común, en las {n_imagenes} imágenes de test. Precisión, recall y F1 con IoU 0,5, "
                 "promediados entre las clases.")
    columnas = columnas or (3 if len(modelos) > 4 else len(modelos))
    marcas = orden_de_leyenda(marcas_juntos(modelos, "barra"), columnas)
    fig, ax = crear_figura("Desempeño en test", subtitulo, marcas, ancho=ancho, alto=alto, margenes=(0.95, 0.35, 0.75),
                           columnas_leyenda=columnas)
    centros = np.array([0, 1, 2, 3.4, 4.4, 5.4])
    tope = 1.32 * max(datos[m]["evaluacion"][clave] for m in modelos for _, clave in metricas)
    for k, m in enumerate(modelos):
        x, ancho_barra = posiciones_de_barras(centros, len(modelos), k)
        valores = [datos[m]["evaluacion"][clave] for _, clave in metricas]
        ax.bar(x, valores, ancho_barra, color=COLORES[m])
        if len(modelos) <= 2:
            etiquetar_barras(ax, x, valores, decimal, 0.012 * tope)
    for x0, x1, texto in ((0, 2, f"Con confianza ≥ {decimal(UMBRAL_CONFIANZA)}"), (3.4, 5.4, "Sin umbral de confianza")):
        ax.plot([x0 - 0.42, x1 + 0.42], [tope * 0.9] * 2, color=EJE, lw=1.2)
        ax.text((x0 + x1) / 2, tope * 0.92, texto, ha="center", va="bottom", fontsize=12.5, color=TINTA_SECUNDARIA,
                bbox=ETIQUETA_SOBRE_LINEAS)
    ax.set_xticks(centros, [nombre for nombre, _ in metricas])
    ax.grid(axis="x", visible=False)
    ax.set(xlim=(-0.6, 6.0), ylim=(0, tope), ylabel="Valor")
    con_coma(ax.yaxis, 1)
    guardar(fig, ruta)


def comparar_map_segun_iou(mapas, ruta):
    """mAP de test de dos modelos en cada umbral de IoU, con los valores en 0,50 y 0,75."""
    subtitulo = ("Test, evaluador común. Cada punto exige ese IoU mínimo entre la caja detectada y la real para contar "
                 "un acierto; el mAP50-95 es el promedio de los diez puntos.")
    fig, ax = crear_figura("mAP en test según el umbral de IoU", subtitulo, marcas_juntos(mapas), ancho=ANCHO_COMPARACION,
                           alto=ALTO_COMPARACION, margenes=(1.0, 0.35, 0.75))
    for m, valores in mapas.items():
        ax.plot(comun.UMBRALES_IOU, valores, color=COLORES[m], lw=2, marker="o", ms=8, mec=FONDO, mew=2)
        otro = next(v for clave, v in mapas.items() if clave != m)
        for i in (0, 5):   # IoU 0,50 y 0,75: el valor más alto va sobre su punto y el más bajo, debajo
            arriba = valores[i] >= otro[i]
            ax.annotate(decimal(valores[i], 3), (comun.UMBRALES_IOU[i], valores[i]), xytext=(0, 11 if arriba else -11),
                        textcoords="offset points", ha="center", va="bottom" if arriba else "top", fontsize=12.5,
                        fontweight="bold", color=TINTA, bbox=ETIQUETA_SOBRE_LINEAS)
    ax.set_xticks(comun.UMBRALES_IOU)
    ax.set(xlim=(0.475, 0.975), ylim=(0, 1.2 * max(v.max() for v in mapas.values())), xlabel="Umbral de IoU",
           ylabel="mAP en test")
    con_coma(ax.xaxis, 2)
    con_coma(ax.yaxis, 2)
    guardar(fig, ruta)


def comparar_tipos_de_deteccion(tipos, ruta):
    """Cuántas detecciones de test (confianza ≥ umbral) caen en cada tipo, para dos modelos."""
    modelos = list(tipos)
    conteos = pd.DataFrame({m: tipos[m].value_counts().reindex(TIPOS, fill_value=0) for m in modelos})
    conteos.rename(columns=MODELOS).rename_axis("tipo").to_csv(ruta.with_suffix(".csv"))
    etiquetas = {m: f"{MODELOS[m]}: {entero(conteos[m].sum())} detecciones" for m in modelos}
    subtitulo = (f"Detecciones de test con confianza ≥ {decimal(UMBRAL_CONFIANZA)}, según su mayor IoU con los objetos "
                 "reales. Correcta: de la clase del objeto, IoU ≥ 0,5. Duplicada: igual, pero el objeto ya tenía una "
                 "detección más segura. Clase equivocada: IoU ≥ 0,5 con un objeto de otra clase. Mal ubicada: de su "
                 "clase, IoU entre 0,1 y 0,5. Clase y ubicación: de otra clase, IoU entre 0,1 y 0,5. Fondo: IoU < 0,1 "
                 "con todos los objetos.")
    fig, ax = crear_figura("Tipos de detección en test", subtitulo, marcas_juntos(modelos, "barra", etiquetas),
                           ancho=ANCHO_COMPARACION, alto=7.0, margenes=(1.1, 0.35, 1.0), lineas_reservadas=5)
    centros = np.arange(len(TIPOS))
    tope = 1.15 * conteos.to_numpy().max()
    for k, m in enumerate(modelos):
        x, ancho_barra = posiciones_de_barras(centros, len(modelos), k)
        ax.bar(x, conteos[m], ancho_barra, color=COLORES[m])
        etiquetar_barras(ax, x, conteos[m], entero, 0.012 * tope)
    ax.set_xticks(centros, [t.replace(" ", "\n", 1) if len(t) > 10 else t for t in TIPOS])
    ax.grid(axis="x", visible=False)
    ax.set(ylim=(0, tope), ylabel="Detecciones")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: entero(v)))
    guardar(fig, ruta)


def comparar_recall_segun_tamano(tablas, ruta):
    """Recall por objeto de dos modelos en cada grupo de tamaño y en total."""
    modelos = list(tablas)
    pd.concat([tablas[modelos[0]].objetos] + [t.recall.rename(f"recall {MODELOS[m]}") for m, t in tablas.items()],
              axis=1).rename_axis("tamaño").to_csv(ruta.with_suffix(".csv"))
    subtitulo = (f"Fracción de los objetos de test con una detección correcta (de su clase, IoU ≥ 0,5 y confianza ≥ "
                 f"{decimal(UMBRAL_CONFIANZA)}). Tamaño: raíz del área de la caja en la imagen de entrada de "
                 f"{LADO_ENTRADA} px, con los límites de COCO. Se cuenta por objeto y no por clase, así que no es el "
                 "recall de las métricas de test.")
    fig, ax = crear_figura("Recall en test según el tamaño del objeto", subtitulo, marcas_juntos(modelos, "barra"),
                           ancho=ANCHO_COMPARACION, alto=6.6, margenes=(1.0, 0.35, 1.3), lineas_reservadas=3)
    centros = np.array([0, 1, 2, 3.4])
    for k, (m, tabla) in enumerate(tablas.items()):
        x, ancho_barra = posiciones_de_barras(centros, len(modelos), k)
        ax.bar(x, tabla.recall, ancho_barra, color=COLORES[m])
        etiquetar_barras(ax, x, tabla.recall, decimal, 0.012)
    pequeno, grande = LIMITES_TAMANO
    rotulos = [f"Pequeños\n< {pequeno} px", f"Medianos\n{pequeno} a {grande} px", f"Grandes\n≥ {grande} px", "Todos\n"]
    ax.set_xticks(centros, [f"{r}\n{entero(n)} objetos" for r, n in zip(rotulos, tablas[modelos[0]].objetos)])
    ax.grid(axis="x", visible=False)
    ax.set(xlim=(-0.6, 4.0), ylim=(0, 1), ylabel="Recall por objeto")
    con_coma(ax.yaxis, 1)
    guardar(fig, ruta)


def comparar_costo(modelos, ruta):
    """Tamaño y costo de entrenamiento de varios modelos (tabla de costos de 05), un panel por medida."""
    costo = pd.read_csv(DIR_RESULTADOS / "comparacion" / "costo.csv", index_col=0)
    paneles = [("Parámetros (millones)", "Parámetros (millones)", 2), ("GFLOPs por imagen", "GFLOPs", 1),
               ("Pesos en float32 (MB)", "Pesos en float32 (MB)", 1), ("Tiempo por época (s)", "Tiempo por época (s)", 1),
               ("Tiempo total (min)", "Tiempo total de entrenamiento (min)", 1),
               ("Memoria de GPU (GB)", "Memoria máxima de GPU (GB)", 2)]
    epocas = " y ".join(str(int(costo.loc["Épocas ejecutadas", FILAS_05[m]])) for m in modelos)
    subtitulo = (f"Misma GPU (RTX 4060 Laptop) y entrada de {LADO_ENTRADA} × {LADO_ENTRADA}. Tiempo por época con la "
                 f"validación incluida; tiempo total hasta la parada temprana ({epocas} épocas). Pesos en float32 en "
                 "todos (Ultralytics los guarda en float16).")
    alto = 6.6
    fig, ax = crear_figura("Tamaño y costo de entrenamiento", subtitulo, marcas_juntos(modelos, "barra"),
                           ancho=ANCHO_COMPARACION, alto=alto, margenes=(1.35, 0.35, 0.25), lineas_reservadas=3)
    caja = ax.get_position()
    ax.remove()   # el espacio de los ejes se reparte en seis paneles
    rejilla = fig.add_gridspec(2, 3, left=caja.x0, right=caja.x1, bottom=caja.y0, top=caja.y1 - 0.32 / alto,
                               hspace=0.45, wspace=0.75)
    y = np.arange(len(modelos))[::-1]
    for k, (titulo, fila, decimales) in enumerate(paneles):
        panel = fig.add_subplot(rejilla[k // 3, k % 3])
        valores = [costo.loc[fila, FILAS_05[m]] for m in modelos]
        panel.barh(y, valores, height=0.62, color=[COLORES[m] for m in modelos])
        for yi, valor in zip(y, valores):
            panel.text(valor, yi, f"  {decimal(valor, decimales)}", va="center", ha="left", fontsize=12.5,
                       color=TINTA_SECUNDARIA)
        panel.set_yticks(y, [MODELOS[m] for m in modelos], fontsize=12)
        panel.set_xlim(0, 1.5 * max(valores))
        panel.set_xticks([])
        panel.grid(False)
        panel.spines["bottom"].set_visible(False)
        panel.set_title(titulo, loc="left", fontsize=13, color=TINTA, pad=8)
    guardar(fig, ruta)


# ----------------------------------------------------------------------------- tablas como imagen

def tabla_imagen(filas, columnas, nota, ruta, alto_fila=0.62):
    """Tabla como imagen, del ancho de una diapositiva: una fila por modelo, con su color, y una columna por medida.

    `filas`: {modelo: [texto de cada columna]}; `columnas`: encabezados, con un salto de línea si van en dos líneas.
    Los números se alinean a la derecha de su columna, como el encabezado.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    letra_encabezado, letra_celda = 15.5, 17
    ancho_nombre = 0.4 + 0.5 + max(ancho_texto(MODELOS[m], letra_celda) for m in filas)   # cuadrado, nombre y espacio
    lineas_nota = envolver(nota, ANCHO_TABLA - 0.5, 12.5)
    alto_encabezado, margen = 0.85, 0.15
    base = margen + alto_encabezado                       # línea bajo el encabezado, desde el borde superior
    alto = base + len(filas) * alto_fila + 0.22 + 0.27 * len(lineas_nota) + margen
    fig = plt.figure(figsize=(ANCHO_TABLA, alto))
    x = lambda pulgadas: pulgadas / ANCHO_TABLA           # noqa: E731 (pulgadas → fracción de la figura)
    y = lambda pulgadas: 1 - pulgadas / alto              # noqa: E731 (pulgadas desde arriba → fracción)
    # cada columna, del ancho de su texto más largo (encabezado o celda) más un espacio; lo que sobra se reparte igual
    necesario = [max(max(ancho_texto(linea, letra_encabezado, True) for linea in columna.split("\n")),
                     max(ancho_texto(textos[k], letra_celda) for textos in filas.values())) + 0.3
                 for k, columna in enumerate(columnas)]
    sobra = ANCHO_TABLA - 0.4 - ancho_nombre - sum(necesario)
    assert sobra >= 0, f"la tabla {ruta.name} no cabe: faltan {-sobra:.2f} pulgadas"
    derechas = list(0.2 + ancho_nombre + np.cumsum([n + sobra / len(columnas) for n in necesario]))
    fig.text(x(0.2), y(base - 0.1), "Modelo", ha="left", va="bottom", fontsize=letra_encabezado, fontweight="bold")
    for derecha, columna in zip(derechas, columnas):
        fig.text(x(derecha), y(base - 0.1), columna, ha="right", va="bottom", fontsize=letra_encabezado, fontweight="bold",
                 linespacing=1.15)
    fig.add_artist(Line2D([x(0.1), x(ANCHO_TABLA - 0.1)], [y(base)] * 2, color=TINTA_SECUNDARIA, lw=1.5))
    for i, (m, textos) in enumerate(filas.items()):
        centro = base + (i + 0.5) * alto_fila
        lado = 0.24                                       # cuadrado del color del modelo, como en la leyenda
        fig.add_artist(Rectangle((x(0.2), y(centro + lado / 2)), lado / ANCHO_TABLA, lado / alto,
                                 transform=fig.transFigure, color=COLORES[m]))
        fig.text(x(0.6), y(centro), MODELOS[m], ha="left", va="center", fontsize=letra_celda)
        for derecha, texto in zip(derechas, textos):
            fig.text(x(derecha), y(centro), texto, ha="right", va="center", fontsize=letra_celda)
        fig.add_artist(Line2D([x(0.1), x(ANCHO_TABLA - 0.1)], [y(base + (i + 1) * alto_fila)] * 2, color=REJILLA, lw=1.1))
    fig.text(x(0.2), y(base + len(filas) * alto_fila + 0.22), "\n".join(lineas_nota), ha="left", va="top",
             fontsize=12.5, color=TINTA_SECUNDARIA, linespacing=1.4)
    guardar(fig, ruta)


def columnas_de_tabla(datos):
    """Columnas posibles de las tablas: encabezado → función que da (texto, valor) de un modelo.

    Las de test salen del evaluador común; las de tamaño y costo, de la tabla de costos de 05.
    """
    costo = pd.read_csv(DIR_RESULTADOS / "comparacion" / "costo.csv", index_col=0)
    de_test = lambda clave, decimales=3: lambda m: (decimal(datos[m]["evaluacion"][clave], decimales),   # noqa: E731
                                                    datos[m]["evaluacion"][clave])
    de_costo = lambda fila, decimales: lambda m: (decimal(costo.loc[fila, FILAS_05[m]], decimales),     # noqa: E731
                                                  costo.loc[fila, FILAS_05[m]])
    return {
        "Precisión": de_test("precision"), "Recall": de_test("recall"), "F1": de_test("F1"), "mAP50": de_test("mAP50"),
        "mAP75": de_test("mAP75"), "mAP50-95": de_test("mAP50_95"),
        "Clases con\ndetecciones": lambda m: (str(datos[m]["evaluacion"]["clases_con_detecciones"]),
                                              datos[m]["evaluacion"]["clases_con_detecciones"]),
        "Precisión\nmicro": de_test("precision_micro"),
        "Parámetros\n(millones)": de_costo("Parámetros (millones)", 2), "GFLOPs": de_costo("GFLOPs", 1),
        "Épocas\n(mejor)": lambda m: (f"{int(costo.loc['Épocas ejecutadas', FILAS_05[m]])} "
                                      f"({int(costo.loc['Mejor época', FILAS_05[m]])})",
                                      f"{int(costo.loc['Épocas ejecutadas', FILAS_05[m]])} "
                                      f"({int(costo.loc['Mejor época', FILAS_05[m]])})"),
        "Tiempo total\n(min)": de_costo("Tiempo total de entrenamiento (min)", 1),
        "Tiempo por\népoca (s)": de_costo("Tiempo por época (s)", 1),
        "Memoria de\nGPU (GB)": de_costo("Memoria máxima de GPU (GB)", 2), "FPS\n(lote 1)": de_costo("FPS", 1),
    }


COLUMNAS_TEST = ["Precisión", "Recall", "F1", "mAP50", "mAP75", "mAP50-95", "Clases con\ndetecciones", "Precisión\nmicro"]
COLUMNAS_COSTO = ["Parámetros\n(millones)", "GFLOPs", "Épocas\n(mejor)", "Tiempo total\n(min)", "Tiempo por\népoca (s)",
                  "Memoria de\nGPU (GB)", "FPS\n(lote 1)"]
COLUMNAS_RESUMEN = ["Precisión", "Recall", "mAP50", "mAP50-95", "Parámetros\n(millones)", "Épocas\n(mejor)",
                    "Tiempo total\n(min)", "FPS\n(lote 1)"]


def notas_de_tabla(n_imagenes, n_clases_test):
    """Qué significa cada grupo de columnas, para la nota bajo la tabla."""
    umbral = decimal(UMBRAL_CONFIANZA)
    return {
        "test": (f"Evaluador común, en las {n_imagenes} imágenes de test. Precisión, recall y F1: con IoU 0,5 y confianza "
                 f"≥ {umbral}, promediados entre clases. mAP: sin umbral de confianza. Clases con detecciones: de las "
                 f"{n_clases_test} clases con objetos en test, cuántas tienen al menos una detección con confianza ≥ "
                 f"{umbral}. Precisión micro: fracción de todas esas detecciones que son correctas, sin promediar por clase."),
        "costo": (f"Misma GPU (RTX 4060 Laptop) y entrada de {LADO_ENTRADA} × {LADO_ENTRADA}. Tiempo total: entrenamiento "
                  "hasta la parada temprana, con la validación de cada época. Memoria: la máxima reservada en la GPU al "
                  "entrenar. FPS: imágenes por segundo en float32 con lote 1, incluidos el preproceso y el NMS; en los "
                  f"modelos pequeños, dos mediciones seguidas variaron {VARIACION_FPS}."),
        "resumen": (f"Test: evaluador común, en las {n_imagenes} imágenes. Precisión y recall con IoU 0,5 y confianza ≥ "
                    f"{umbral}, promediados entre clases; mAP sin umbral. Tiempo total: entrenamiento hasta la parada "
                    "temprana en la misma GPU (RTX 4060 Laptop), con la validación de cada época. FPS en float32 con lote "
                    f"1; en los modelos pequeños, dos mediciones seguidas variaron {VARIACION_FPS}."),
        # la tabla de seis modelos comparte la diapositiva con su gráfico: nota más corta
        "test_corta": (f"Evaluador común, en las {n_imagenes} imágenes de test. Precisión, recall y F1 con IoU 0,5 y "
                       f"confianza ≥ {umbral}, promediados entre clases; mAP sin umbral. Clases con detecciones: de las "
                       f"{n_clases_test} con objetos en test. Precisión micro: fracción de las detecciones con confianza ≥ "
                       f"{umbral} que son correctas."),
    }


def guardar_tabla(columnas, encabezados, modelos, nota, ruta, alto_fila=0.62):
    """Guarda una tabla como imagen (con los valores redondeados) y como CSV (con los valores completos)."""
    celdas = {m: [columnas[c](m) for c in encabezados] for m in modelos}
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({MODELOS[m]: {c.replace("\n", " "): valor for c, (_, valor) in zip(encabezados, celdas[m])}
                  for m in modelos}).T.rename_axis("modelo").to_csv(ruta.with_suffix(".csv"))
    tabla_imagen({m: [texto for texto, _ in celdas[m]] for m in modelos}, encabezados, nota, ruta, alto_fila=alto_fila)


# ----------------------------------------------------------------------------- arquitectura de la YOLOv3-tiny

def capas_yolov3_tiny():
    """Cada capa de la YOLOv3-tiny de 02b con su salida (canales, alto, ancho), sus parámetros y sus GFLOPs a 640 px.

    Parámetros de una convolución: k² · entrada · filtros, más 2 · filtros de BatchNorm (escala y desplazamiento) o,
    en las de salida, el sesgo. GFLOPs: 2 por multiplicación-suma, la convención de los cuadernos.
    """
    filas, formas = [], {}
    forma = (3, LADO_ENTRADA, LADO_ENTRADA)
    for tramo, capas in TRAMOS_YOLOV3_TINY.items():
        if tramo == "Cabeza de stride 32":
            forma = formas["Cuerpo hasta stride 32"]
        elif tramo == "Puente":
            forma = formas["Cuerpo hasta stride 32"]   # el puente parte del mismo mapa que la cabeza de stride 32
        for capa in capas:
            canales, alto, ancho = forma
            parametros, gflops = 0, 0.0
            if capa[0] in ("conv", "salida"):
                _, k, filtros = capa
                parametros = k * k * canales * filtros + (filtros if capa[0] == "salida" else 2 * filtros)
                gflops = 2 * k * k * canales * filtros * alto * ancho / 1e9
                nombre = f"Salida: conv {k}×{k}, {filtros}" if capa[0] == "salida" else f"Conv {k}×{k}, {filtros}"
                forma = (filtros, alto, ancho)
            elif capa[0] == "mp":
                stride = capa[1]
                nombre = "Max-pool 2×2" if stride == 2 else "Max-pool 2×2, stride 1"
                forma = (canales, alto // stride, ancho // stride)
            elif capa[0] == "ampliar":
                nombre = "Ampliar ×2"
                forma = (canales, 2 * alto, 2 * ancho)
            else:                                       # concatenar con la salida del cuerpo hasta stride 16
                guardada = formas["Cuerpo hasta stride 16"]
                assert guardada[1:] == forma[1:], "la ampliación no quedó del tamaño del mapa de stride 16"
                nombre = "Concatenar con el mapa de stride 16"
                forma = (canales + guardada[0], alto, ancho)
            filas.append({"tramo": tramo, "capa": nombre, "salida": forma, "parametros": parametros, "GFLOPs": gflops})
        formas[tramo] = forma
    tabla = pd.DataFrame(filas)
    # deben dar el tamaño que midió 02b con la red construida
    resumen = json.loads((DIR_RESULTADOS / "yolo_a_mano" / "resumen.json").read_text(encoding="utf-8"))
    assert tabla.parametros.sum() == round(resumen["parametros_M"] * 1e6), "los parámetros no son los de 02b"
    assert np.isclose(tabla.GFLOPs.sum(), resumen["GFLOPs"], rtol=1e-6), "los GFLOPs no son los de 02b"
    return tabla


def tabla_capas_yolov3_tiny(tabla, ruta):
    """Tabla de capas de la YOLOv3-tiny, por tramos, como la de la CNN en su diapositiva; también en CSV."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.assign(salida=tabla.salida.map(lambda f: " × ".join(map(str, f)))).to_csv(ruta.with_suffix(".csv"), index=False)
    ancho, alto_fila, alto_tramo = 9.0, 0.33, 0.44
    columnas = [("Capa", 0.3, "left"), ("Salida (canales × alto × ancho)", 6.05, "right"), ("Parámetros", 7.6, "right"),
                ("GFLOPs", 8.75, "right")]
    n_tramos = tabla.tramo.nunique()
    nota = ("Cada convolución, salvo las 2 de salida, lleva BatchNorm y LeakyReLU (pendiente 0,1); sus parámetros incluyen "
            "los de BatchNorm. GFLOPs con una imagen de 640 × 640 (2 por multiplicación-suma).")
    lineas_nota = envolver(nota, ancho - 0.6, 11.5)
    titulos = {"Cabeza de stride 32": "Cabeza de stride 32, desde el mapa de 256 × 20 × 20",
               "Puente": "Puente, desde el mismo mapa de 256 × 20 × 20"}
    alto = 0.15 + 0.55 + n_tramos * alto_tramo + len(tabla) * alto_fila + alto_fila + 0.25 + 0.27 * len(lineas_nota) + 0.1
    fig = plt.figure(figsize=(ancho, alto))
    x = lambda pulgadas: pulgadas / ancho                 # noqa: E731
    y = lambda pulgadas: 1 - pulgadas / alto              # noqa: E731
    cursor = 0.7
    for texto, posicion, alineacion in columnas:
        fig.text(x(posicion), y(cursor - 0.08), texto, ha=alineacion, va="bottom", fontsize=13, fontweight="bold")
    fig.add_artist(Line2D([x(0.15), x(ancho - 0.15)], [y(cursor)] * 2, color=TINTA_SECUNDARIA, lw=1.3))
    for n, (tramo, capas) in enumerate(tabla.groupby("tramo", sort=False)):
        if n:   # línea fina entre tramos
            fig.add_artist(Line2D([x(0.15), x(ancho - 0.15)], [y(cursor + 0.02)] * 2, color=REJILLA, lw=1))
        fig.text(x(0.3), y(cursor + alto_tramo - 0.1), titulos.get(tramo, tramo), ha="left", va="bottom", fontsize=12.5,
                 fontweight="bold", color=COLOR_YOLOV3_TINY)
        cursor += alto_tramo
        for fila in capas.itertuples():
            centro = cursor + alto_fila / 2
            textos = [fila.capa, " × ".join(map(str, fila.salida)), entero(fila.parametros) if fila.parametros else "–",
                      decimal(fila.GFLOPs) if fila.GFLOPs else "–"]
            for texto, (_, posicion, alineacion) in zip(textos, columnas):
                fig.text(x(posicion), y(centro), texto, ha=alineacion, va="center", fontsize=12.5,
                         color=TINTA if texto != "–" else GRIS)
            cursor += alto_fila
    centro = cursor + alto_fila / 2 + 0.05
    for texto, (_, posicion, alineacion) in zip(["Total", "", entero(tabla.parametros.sum()), decimal(tabla.GFLOPs.sum(), 1)],
                                                columnas):
        fig.text(x(posicion), y(centro), texto, ha=alineacion, va="center", fontsize=13, fontweight="bold")
    fig.add_artist(Line2D([x(0.15), x(ancho - 0.15)], [y(cursor + 0.04)] * 2, color=TINTA_SECUNDARIA, lw=1.3))
    fig.text(x(0.3), y(cursor + alto_fila + 0.25), "\n".join(lineas_nota), ha="left", va="top", fontsize=11.5,
             color=TINTA_SECUNDARIA, linespacing=1.4)
    guardar(fig, ruta)


def diagrama_yolov3_tiny(tabla, ruta):
    """Diagrama de bloques de la YOLOv3-tiny: el cuerpo, las dos cabezas y el puente que las une, con el tamaño de cada mapa.

    Pensado para el ancho de una diapositiva. Las coordenadas del dibujo están en pulgadas.
    """
    anclas = pd.read_csv(DIR_RESULTADOS / "yolo_a_mano" / "anclas.csv")
    texto_anclas = {s: ", ".join(f"{round(f['ancho (px)'])}×{round(f['alto (px)'])}" for _, f in g.iterrows())
                    for s, g in anclas.groupby("stride")}
    ancho, alto = 16.0, 4.1
    fig = plt.figure(figsize=(ancho, alto))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, ancho), ylim=(0, alto))
    ax.axis("off")
    bloque, paso, x0 = 0.84, 1.13, 1.2                    # ancho de un bloque, distancia entre bloques y primer bloque
    y_arriba, y_abajo, alto_bloque = 2.95, 1.6, 0.95      # centro de las dos filas y alto de un bloque
    y_puente_guardado = 0.74                               # altura de la línea que lleva el mapa de stride 16 a la concatenación
    izquierda = lambda i: x0 + i * paso                   # noqa: E731
    centro = lambda i: izquierda(i) + bloque / 2           # noqa: E731

    def caja(i, fila_y, linea1, linea2, linea3, destacada=False):
        """Un bloque con el tipo de capa, los filtros (o la operación) y el tamaño de su salida."""
        relleno, borde = (COLOR_YOLOV3_TINY, COLOR_YOLOV3_TINY) if destacada else (RELLENO_NEUTRO, EJE)
        principal, secundario = ("white", "white") if destacada else (TINTA, TINTA_SECUNDARIA)
        ax.add_patch(FancyBboxPatch((izquierda(i), fila_y - alto_bloque / 2), bloque, alto_bloque,
                                    boxstyle="round,pad=0,rounding_size=0.07", fc=relleno, ec=borde, lw=1.2))
        ax.text(centro(i), fila_y + 0.27, linea1, ha="center", va="center", fontsize=10, color=secundario)
        ax.text(centro(i), fila_y, linea2, ha="center", va="center", fontsize=13.5, fontweight="bold", color=principal)
        ax.text(centro(i), fila_y - 0.28, linea3, ha="center", va="center", fontsize=10, color=secundario)

    def flecha(puntos, color=TINTA_SECUNDARIA, grosor=1.4):
        """Línea por los puntos dados, con punta de flecha al final."""
        xs, ys = zip(*puntos)
        ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=grosor, solid_capstyle="butt")
        ax.annotate("", puntos[-1], puntos[-2], arrowprops=dict(arrowstyle="-|>", color=color, lw=grosor,
                                                                shrinkA=0, shrinkB=0, mutation_scale=13))

    # fila de arriba: el cuerpo y la cabeza de stride 32
    cuerpo = [c for t in ("Cuerpo hasta stride 16", "Cuerpo hasta stride 32") for c in
              tabla[(tabla.tramo == t) & tabla.capa.str.startswith("Conv")].itertuples()]
    cabeza_32 = tabla[tabla.tramo == "Cabeza de stride 32"].itertuples()
    tamano = lambda f: f"{f.salida[1]} × {f.salida[2]}"   # noqa: E731
    arriba = [*cuerpo, *cabeza_32]
    for i, f in enumerate(arriba):
        caja(i, y_arriba, f.capa.split(",")[0].replace("Salida: conv", "Salida"), str(f.salida[0]), tamano(f),
             destacada=f.capa.startswith("Salida"))
    # max-pool entre los bloques del cuerpo: dónde va cada uno, según el orden de la tabla
    rotulos_entre, n_conv = {}, -1
    for f in tabla[tabla.tramo.str.startswith("Cuerpo")].itertuples():
        if f.capa.startswith("Conv"):
            n_conv += 1
        else:
            rotulos_entre[n_conv] = "MP" if f.capa == "Max-pool 2×2" else "MP*"
    for i in range(len(arriba) - 1):
        a, b = izquierda(i) + bloque, izquierda(i + 1)
        flecha([(a, y_arriba), (b, y_arriba)])
        if i in rotulos_entre:
            ax.text((a + b) / 2, y_arriba + 0.12, rotulos_entre[i], ha="center", va="bottom", fontsize=9.5,
                    color=TINTA_SECUNDARIA, fontweight="bold")
    ax.text(izquierda(0) - 0.16, y_arriba, "Imagen\n640 × 640\n× 3", ha="right", va="center", fontsize=11,
            color=TINTA, linespacing=1.25)
    # fila de abajo: el puente y la cabeza de stride 16, desde el último bloque del cuerpo
    i_ultimo_cuerpo = len(cuerpo) - 1
    abajo = tabla[tabla.tramo.isin(["Puente", "Cabeza de stride 16"])]
    rotulos_abajo = {"Ampliar ×2": ("Ampliar", "×2"), "Concatenar con el mapa de stride 16": ("Concatenar", None)}
    for k, f in enumerate(abajo.itertuples()):
        i = i_ultimo_cuerpo + 1 + k
        if f.capa in rotulos_abajo:
            linea1, linea2 = rotulos_abajo[f.capa]
            caja(i, y_abajo, linea1, linea2 or str(f.salida[0]), tamano(f))
        else:
            es_salida = f.capa.startswith("Salida")
            caja(i, y_abajo, f.capa.split(",")[0].replace("Salida: conv", "Salida"), str(f.salida[0]), tamano(f),
                 destacada=es_salida)
        if k:
            flecha([(izquierda(i - 1) + bloque, y_abajo), (izquierda(i), y_abajo)])
    flecha([(centro(i_ultimo_cuerpo), y_arriba - alto_bloque / 2), (centro(i_ultimo_cuerpo), y_abajo),
            (izquierda(i_ultimo_cuerpo + 1), y_abajo)])
    # el mapa de stride 16 que se reutiliza: del último bloque de 40 × 40 a la concatenación
    i_guardado = len([f for f in cuerpo if f.tramo == "Cuerpo hasta stride 16"]) - 1
    i_concatenar = i_ultimo_cuerpo + 1 + list(abajo.capa).index("Concatenar con el mapa de stride 16")
    flecha([(centro(i_guardado), y_arriba - alto_bloque / 2), (centro(i_guardado), y_puente_guardado),
            (centro(i_concatenar), y_puente_guardado), (centro(i_concatenar), y_abajo - alto_bloque / 2)],
           color=COLOR_YOLOV3_TINY, grosor=1.8)
    guardado = tabla[(tabla.tramo == "Cuerpo hasta stride 16")].iloc[-1].salida
    ax.text(centro(i_guardado) + 0.15, y_puente_guardado + 0.08,
            f"El mapa de {guardado[1]} × {guardado[2]} × {guardado[0]} se guarda y se concatena con la ampliación",
            ha="left", va="bottom", fontsize=10.5, color=TINTA_SECUNDARIA)
    # tramos: corchetes con su nombre
    def corchete(i_desde, i_hasta, fila_y, texto, hacia_arriba=True):
        """Corchete sobre (o bajo) los bloques i_desde a i_hasta, con su texto."""
        signo = 1 if hacia_arriba else -1
        borde_y = fila_y + signo * (alto_bloque / 2 + 0.16)
        x_a, x_b = izquierda(i_desde), izquierda(i_hasta) + bloque
        ax.plot([x_a, x_a, x_b, x_b], [borde_y - signo * 0.08, borde_y, borde_y, borde_y - signo * 0.08],
                color=GRIS, lw=1.1)
        ax.text((x_a + x_b) / 2, borde_y + signo * 0.08, texto, ha="center", va="bottom" if hacia_arriba else "top",
                fontsize=11.5, color=TINTA)
    corchete(0, i_ultimo_cuerpo, y_arriba, "Cuerpo: 8 convoluciones y 6 max-pool, de 640 × 640 a 20 × 20 (stride 32)")
    corchete(i_ultimo_cuerpo + 1, len(arriba) - 1, y_arriba, "Cabeza de stride 32")
    ax.text(centro(i_ultimo_cuerpo) - 0.2, y_abajo, "Puente y cabeza\nde stride 16", ha="right", va="center",
            fontsize=11.5, color=TINTA, linespacing=1.3)
    # qué predice cada salida
    detalle = {s: f"Stride {s} · {LADO_ENTRADA // s} × {LADO_ENTRADA // s} celdas de {s} px\nanclas: {texto_anclas[s]} px"
               for s in (32, 16)}
    ax.text(izquierda(len(arriba)) - 0.08, y_arriba + 0.1, detalle[32], ha="left", va="center", fontsize=11,
            color=TINTA, linespacing=1.35)
    ax.text(izquierda(i_ultimo_cuerpo + len(abajo)) + bloque, y_abajo + alto_bloque / 2 + 0.14, detalle[16], ha="right",
            va="bottom", fontsize=11, color=TINTA, linespacing=1.35)
    nota = ("Cada convolución, salvo las 2 de salida (rosadas), lleva BatchNorm y LeakyReLU (pendiente 0,1). MP: max-pool "
            "2×2 de stride 2, que reduce el mapa a la mitad. MP*: max-pool 2×2 de stride 1, con 1 px de relleno a la derecha "
            "y abajo, que conserva 20 × 20. Cada salida da, por celda, 3 anclas × 33 valores (4 de caja, 1 de objeto y 28 "
            f"de clase) = 99 canales. En total, {entero(tabla.parametros.sum())} parámetros y "
            f"{decimal(tabla.GFLOPs.sum(), 1)} GFLOPs.")
    ax.text(0.15, 0.1, "\n".join(envolver(nota, ancho - 0.4, 10.5)), ha="left", va="bottom", fontsize=10.5,
            color=TINTA_SECUNDARIA, linespacing=1.4)
    guardar(fig, ruta)


# ----------------------------------------------------------------------------- arquitectura de YOLOv8n

COLOR_YOLOV8N = COLORES["yolo_desde_cero"]   # naranja: se lee texto blanco encima
# Capas de yolov8n.yaml en el orden de Ultralytics; el diagrama ubica cada una por su número
TIPOS_YOLOV8N = ["Conv", "Conv", "C2f", "Conv", "C2f", "Conv", "C2f", "Conv", "C2f", "SPPF",
                 "Upsample", "Concat", "C2f", "Upsample", "Concat", "C2f",
                 "Conv", "Concat", "C2f", "Conv", "Concat", "C2f", "Detect"]


def capas_yolov8n(n_clases):
    """Cada capa de la YOLOv8n de 02 (yolov8n.yaml de Ultralytics, con las clases del dataset) con su salida a 640 px.

    Construye la red en la CPU, como la crea 02 antes de entrenar, y le pasa una imagen vacía para medir cada mapa.
    Devuelve las capas, los canales de las dos ramas de cada salida y los GFLOPs, y comprueba el tamaño con el de 02.
    """
    import logging

    import torch
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import get_flops

    logging.getLogger("ultralytics").setLevel(logging.WARNING)   # sin el aviso de que cambia las 80 clases de COCO
    red = DetectionModel("yolov8n.yaml", nc=n_clases, verbose=False).eval()
    salidas = {}
    ganchos = [capa.register_forward_hook(lambda _, __, salida, i=i: salidas.__setitem__(i, salida))
               for i, capa in enumerate(red.model)]
    with torch.no_grad():
        red(torch.zeros(1, 3, LADO_ENTRADA, LADO_ENTRADA))
    for gancho in ganchos:
        gancho.remove()
    filas = []
    for i, capa in enumerate(red.model):
        tipo = capa.type.split(".")[-1]
        filas.append({"capa": i, "tipo": tipo, "desde": capa.f,
                      "nucleo": capa.conv.kernel_size[0] if tipo == "Conv" else None,
                      "stride": capa.conv.stride[0] if tipo == "Conv" else None,
                      "n": len(capa.m) if tipo == "C2f" else None,
                      "atajo": capa.m[0].add if tipo == "C2f" else None,
                      "salida": tuple(salidas[i].shape[1:]) if torch.is_tensor(salidas[i]) else None,
                      "parametros": sum(p.numel() for p in capa.parameters())})
    tabla = pd.DataFrame(filas).astype({"nucleo": "Int64", "stride": "Int64", "n": "Int64"})
    assert list(tabla.tipo) == TIPOS_YOLOV8N, "las capas no son las de yolov8n.yaml que espera el diagrama"
    # lo que dice la nota del diagrama
    convoluciones, c2f = tabla[tabla.tipo == "Conv"], tabla[tabla.tipo == "C2f"]
    assert (convoluciones.nucleo == 3).all() and (convoluciones.stride == 2).all(), "hay una Conv que no es 3×3 de stride 2"
    assert list(c2f.atajo) == [c < 10 for c in c2f.capa], "las C2f no llevan atajo solo en el cuerpo"
    assert red.model[9].m.kernel_size == 5, "el max-pool del SPPF no es de 5×5"
    cabeza = red.model[-1]
    ramas = {"caja": cabeza.cv2[0][-1].out_channels, "clase": cabeza.cv3[0][-1].out_channels,
             "valores_dfl": cabeza.reg_max, "strides": [int(s) for s in cabeza.stride]}
    gflops = get_flops(red, LADO_ENTRADA)
    # deben dar el tamaño que midió 02 con la red que entrenó
    resumen = json.loads((DIR_RESULTADOS / "yolo_desde_cero" / "resumen.json").read_text(encoding="utf-8"))
    assert tabla.parametros.sum() == round(resumen["parametros_M"] * 1e6), "los parámetros no son los de 02"
    assert np.isclose(gflops, resumen["GFLOPs"], rtol=1e-6), "los GFLOPs no son los de 02"
    return tabla, ramas, gflops


def diagrama_yolov8n(capas, ramas, gflops, ruta):
    """Diagrama de bloques de YOLOv8n con el formato del de la YOLOv3-tiny.

    Arriba va el cuerpo. Debajo, los dos caminos del cuello PAN en zigzag: el que baja de 20 × 20 a 80 × 80, de derecha a
    izquierda, y el que vuelve a subir. Abajo, las tres salidas, cada una bajo el mapa del que sale. Las coordenadas del
    dibujo están en pulgadas. Guarda también las capas en CSV.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    capas.assign(salida=capas.salida.map(lambda f: " × ".join(map(str, f)) if f else "")).to_csv(
        ruta.with_name("yolov8n_capas.csv"), index=False)
    ancho, alto = 14.3, 6.75     # la misma escala del de la YOLOv3-tiny (16 pulgadas), sin el margen vacío a la derecha
    fig = plt.figure(figsize=(ancho, alto))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, ancho), ylim=(0, alto))
    ax.axis("off")
    bloque, paso, x0, alto_bloque = 0.84, 1.13, 1.2, 0.95    # los mismos del diagrama de la YOLOv3-tiny
    y_fila = {"cuerpo": 5.75, "baja": 4.3, "sube": 2.85, "salidas": 1.6}
    medio = alto_bloque / 2
    hueco_cuerpo_baja = (y_fila["cuerpo"] + y_fila["baja"]) / 2   # altura de las líneas que cruzan entre dos filas
    hueco_baja_sube = (y_fila["baja"] + y_fila["sube"]) / 2
    izquierda = lambda i: x0 + i * paso                  # noqa: E731
    centro = lambda i: izquierda(i) + bloque / 2          # noqa: E731
    # (fila, columna) de cada capa por su número en Ultralytics. El camino que baja va bajo el cuerpo, de derecha a
    # izquierda, para que cada concatenación quede cerca del mapa del cuerpo que recibe; el que sube vuelve a la derecha.
    lugar = {**{i: ("cuerpo", i) for i in range(10)},
             **{i: ("baja", 18 - i) for i in range(10, 16)},
             **{i: ("sube", i - 12) for i in range(16, 22)}}
    salidas = capas.desde.iloc[-1]                        # las capas de las que sale cada escala: 15, 18 y 21
    assert capas.desde[11] == [-1, 6] and capas.desde[14] == [-1, 4] and capas.desde[17] == [-1, 12] \
        and capas.desde[20] == [-1, 9] and salidas == [15, 18, 21], "las conexiones no son las que dibuja el diagrama"

    def caja(i, fila_y, linea1, linea2, linea3, destacada=False):
        """Un bloque con el tipo de capa, los canales (o la operación) y el tamaño de su salida."""
        relleno, borde = (COLOR_YOLOV8N, COLOR_YOLOV8N) if destacada else (RELLENO_NEUTRO, EJE)
        principal, secundario = ("white", "white") if destacada else (TINTA, TINTA_SECUNDARIA)
        ax.add_patch(FancyBboxPatch((izquierda(i), fila_y - medio), bloque, alto_bloque,
                                    boxstyle="round,pad=0,rounding_size=0.07", fc=relleno, ec=borde, lw=1.2))
        ax.text(centro(i), fila_y + 0.27, linea1, ha="center", va="center", fontsize=10, color=secundario)
        ax.text(centro(i), fila_y, linea2, ha="center", va="center", fontsize=13.5, fontweight="bold", color=principal)
        ax.text(centro(i), fila_y - 0.28, linea3, ha="center", va="center", fontsize=10, color=secundario)

    def flecha(puntos, color=TINTA_SECUNDARIA, grosor=1.4):
        """Línea por los puntos dados, con punta de flecha al final."""
        xs, ys = zip(*puntos)
        ax.plot(xs, ys, color=color, lw=grosor, solid_capstyle="butt")
        ax.annotate("", puntos[-1], puntos[-2], arrowprops=dict(arrowstyle="-|>", color=color, lw=grosor,
                                                                shrinkA=0, shrinkB=0, mutation_scale=13))

    def textos(f):
        """Las tres líneas de un bloque."""
        canales, h, w = f.salida
        tamano = f"{h} × {w}"
        return {"Conv": (f"Conv {f.nucleo}×{f.nucleo}", str(canales), tamano), "C2f": (f"C2f ×{f.n}", str(canales), tamano),
                "SPPF": ("SPPF", str(canales), tamano), "Upsample": ("Ampliar", "×2", tamano),
                "Concat": ("Concatenar", str(canales), tamano)}[f.tipo]

    for f in capas.itertuples():
        if f.tipo != "Detect":
            fila, i = lugar[f.capa]
            caja(i, y_fila[fila], *textos(f))
    # el camino principal: el cuerpo hacia la derecha, la bajada hacia la izquierda y la subida hacia la derecha
    for capa in range(1, 22):
        (fila_a, i_a), (fila_b, i_b) = lugar[capa - 1], lugar[capa]
        y = y_fila[fila_b]
        if fila_a == fila_b and i_b > i_a:
            flecha([(izquierda(i_a) + bloque, y), (izquierda(i_b), y)])
        elif fila_a == fila_b:
            flecha([(izquierda(i_a), y), (izquierda(i_b) + bloque, y)])
        elif fila_b == "baja":   # del SPPF a la primera ampliación
            flecha([(centro(i_a), y_fila[fila_a] - medio), (centro(i_a), y), (izquierda(i_b) + bloque, y)])
        else:                    # de la última C2f de la bajada a la primera convolución de la subida
            flecha([(centro(i_a), y_fila["sube"]), (izquierda(i_b), y)])
    ax.text(izquierda(0) - 0.16, y_fila["cuerpo"], "Imagen\n640 × 640\n× 3", ha="right", va="center", fontsize=11,
            color=TINTA, linespacing=1.25)
    # salidas: bajo la última C2f de cada escala
    ancho_rama = ramas["caja"] + ramas["clase"]
    for capa, stride in zip(salidas, ramas["strides"]):
        fila, i = lugar[capa]
        h, w = capas.salida[capa][1:]
        caja(i, y_fila["salidas"], "Salida", str(ancho_rama), f"{h} × {w}", destacada=True)
        flecha([(centro(i), y_fila[fila] - medio), (centro(i), y_fila["salidas"] + medio)])
        ax.text(izquierda(i) + bloque + 0.14, y_fila["salidas"], f"Stride {stride} · {h} × {w}\nceldas de {stride} px",
                ha="left", va="center", fontsize=11, color=TINTA, linespacing=1.35)
    # mapas que se guardan y se concatenan después (naranjas)
    guardadas = dict(color=COLOR_YOLOV8N, grosor=1.8)
    _, i6 = lugar[6]
    _, i11 = lugar[11]
    flecha([(centro(i6), y_fila["cuerpo"] - medio), (centro(i6), hueco_cuerpo_baja), (centro(i11), hueco_cuerpo_baja),
            (centro(i11), y_fila["baja"] + medio)], **guardadas)
    flecha([(centro(lugar[4][1]), y_fila["cuerpo"] - medio), (centro(lugar[14][1]), y_fila["baja"] + medio)],
           **guardadas)
    _, i12 = lugar[12]
    _, i17 = lugar[17]
    flecha([(centro(i12), y_fila["baja"] - medio), (centro(i12), hueco_baja_sube), (centro(i17), hueco_baja_sube),
            (centro(i17), y_fila["sube"] + medio)], **guardadas)
    _, i9 = lugar[9]
    _, i20 = lugar[20]
    x_lado = izquierda(i9) + bloque + 0.25                # sale por el costado del SPPF, fuera de la columna de la bajada
    flecha([(izquierda(i9) + bloque, y_fila["cuerpo"]), (x_lado, y_fila["cuerpo"]), (x_lado, hueco_baja_sube),
            (centro(i20), hueco_baja_sube), (centro(i20), y_fila["sube"] + medio)], **guardadas)
    # nombre de cada parte
    borde_y = y_fila["cuerpo"] + medio + 0.16
    ax.plot([izquierda(0), izquierda(0), izquierda(9) + bloque, izquierda(9) + bloque],
            [borde_y - 0.08, borde_y, borde_y, borde_y - 0.08], color=GRIS, lw=1.1)
    cuerpo = capas.iloc[:10]
    ax.text((izquierda(0) + izquierda(9) + bloque) / 2, borde_y + 0.08,
            f"Cuerpo: {(cuerpo.tipo == 'Conv').sum()} convoluciones de stride 2, {(cuerpo.tipo == 'C2f').sum()} C2f y SPPF, "
            f"de 640 × 640 a {cuerpo.salida.iloc[-1][1]} × {cuerpo.salida.iloc[-1][2]} (stride 32)",
            ha="center", va="bottom", fontsize=11.5, color=TINTA)
    x_rotulos = izquierda(lugar[15][1]) - 0.2
    for fila, texto in [("baja", "Cuello PAN, bajada:\nde 20 × 20 a 80 × 80"), ("sube", "Cuello PAN, subida:\nde 80 × 80 a 20 × 20"),
                        ("salidas", "Cabeza: una salida\npor escala, sin anclas")]:
        ax.text(x_rotulos, y_fila[fila], texto, ha="right", va="center", fontsize=11.5, color=TINTA, linespacing=1.3)
    candidatas = sum(capas.salida[c][1] * capas.salida[c][2] for c in salidas)
    nota = (f"Conv: convolución de stride 2, con BatchNorm y SiLU, que reduce el mapa a la mitad. C2f ×n: divide los "
            f"canales en dos, pasa una mitad por n bloques de dos convoluciones 3×3 (con atajo solo en el cuerpo) y "
            f"concatena todo. SPPF: tres max-pool 5×5 seguidos, concatenados. Líneas naranjas: mapas que se guardan y se "
            f"concatenan más adelante. Cada salida (naranja) da {ancho_rama} valores por celda, de dos ramas de tres "
            f"convoluciones: caja, {ramas['caja']} (4 lados × {ramas['valores_dfl']} valores de la DFL), y clase, "
            f"{ramas['clase']}. Sin anclas: {entero(candidatas)} candidatas por imagen. En total, "
            f"{entero(capas.parametros.sum())} parámetros y {decimal(gflops, 1)} GFLOPs.")
    ax.text(0.15, 0.1, "\n".join(envolver(nota, ancho - 0.4, 10.5)), ha="left", va="bottom", fontsize=10.5,
            color=TINTA_SECUNDARIA, linespacing=1.4)
    guardar(fig, ruta)


# ----------------------------------------------------------------------------- programa

def topes_de_perdida(datos):
    """Tope del eje de cada pérdida en cada familia: el mayor de sus modelos, para que compartan la escala."""
    topes = {}
    for m in CON_GRAFICOS_PROPIOS:
        for termino in TERMINOS:
            if f"perdida_{termino}_train" in datos[m]["registro"]:
                clave = (FAMILIA[m], termino)
                topes[clave] = max(topes.get(clave, 0), tope_perdida(datos[m]["registro"], termino))
    return topes


def graficos_del_modelo(m, datos, tope_map, topes, nombres, reales):
    """Todos los gráficos de un modelo: los comunes en <modelo>/ y los que solo él registró en solo_este_modelo/."""
    registro, mejor, evaluacion = datos["registro"], datos["mejor"], datos["evaluacion"]
    familia = FAMILIA[m]
    carpeta = DIR_SALIDA / m
    print(MODELOS[m])
    for numero, termino in enumerate(("total", "caja", "clase"), start=1):
        grafico_perdida(m, registro, mejor, termino, topes[familia, termino], carpeta / f"0{numero}_perdida_{termino}.png")
    for numero, columna, nombre in ((4, "val_mAP50", "mAP50"), (5, "val_mAP50_95", "mAP50-95")):
        grafico_por_epoca(m, registro, mejor, columna, f"{nombre} de validación",
                          f"En validación, al final de cada época. {MEJOR_EPOCA}", f"{nombre} de validación",
                          tope_map[columna], carpeta / f"0{numero}_{columna.removeprefix('val_').lower()}_val.png",
                          decimales_eje=2, decimales_valor=3)
    grafico_por_epoca(m, registro, mejor, "tasa", "tasa de aprendizaje", SUBTITULO_TASA[familia], "Tasa de aprendizaje",
                      0.0105, carpeta / "06_tasa_aprendizaje.png", decimales_eje=3)
    grafico_ap50_por_clase(m, evaluacion, len(nombres), carpeta / "07_ap50_por_clase_test.png")
    grafico_curva_pr(m, evaluacion, carpeta / "08_curva_pr_test.png")
    matriz = matriz_confusion(datos["detecciones"], reales, len(nombres))
    etiquetas = [*nombres.values(), "fondo"]
    pd.DataFrame(matriz, index=pd.Index(etiquetas, name="predicha \\ real"), columns=etiquetas).to_csv(
        carpeta / "09_matriz_confusion_test.csv")
    grafico_matriz_confusion(m, matriz, nombres, carpeta / "09_matriz_confusion_test.png")
    detecciones = datos["detecciones"]
    for numero, ruta_imagen in enumerate(ejemplos_de_test(), start=10):
        cajas = detecciones[(detecciones.id == ruta_imagen.stem) & (detecciones.confianza >= UMBRAL_CONFIANZA)]
        cuantas = f"{len(cajas)} detección" if len(cajas) == 1 else f"{len(cajas)} detecciones"
        dibujar_imagen(ruta_imagen, cajas, COLORES[m], COLORES[m], TEXTO_SOBRE_COLOR.get(m, "white"),
                       f"{MODELOS[m]}: imagen {ruta_imagen.stem}", f"Test: {cuantas} con confianza ≥ "
                       f"{decimal(UMBRAL_CONFIANZA)}.", nombres, carpeta / f"{numero}_ejemplo_{ruta_imagen.stem}.png")
    # lo que solo registró este modelo
    propios = carpeta / "solo_este_modelo"
    if familia == "yolov8n":
        grafico_perdida(m, registro, mejor, "dfl", topes[familia, "dfl"], propios / "perdida_dfl.png")
        for columna, nombre in (("val_precision", "precisión"), ("val_recall", "recall")):
            grafico_por_epoca(m, registro, mejor, columna, f"{nombre} de validación",
                              "Media entre clases de Ultralytics, en el umbral de confianza que maximiza su F1 (no en "
                              "0,25). Los demás modelos no la registraron.", f"{nombre.capitalize()} de validación", 1.0,
                              propios / f"{columna.removeprefix('val_')}_val.png", decimales_eje=1, decimales_valor=2)
    elif familia == "yolov3_tiny":
        for termino in ("objeto", "fondo"):
            grafico_perdida(m, registro, mejor, termino, topes[familia, termino], propios / f"perdida_{termino}.png")
        info = json.loads((DIR_ENTRENAMIENTOS / m / "info_entrenamiento.json").read_text(encoding="utf-8"))
        limite = info["configuracion"]["norma_maxima_gradiente"]
        grafico_por_epoca(m, registro, mejor, "norma_gradiente", "norma del gradiente",
                          f"Media por época, antes del recorte; línea de puntos: límite del recorte ({decimal(limite, 0)}). "
                          "Ultralytics no registra este valor.", "Norma del gradiente",
                          1.12 * max(registro.norma_gradiente.max(), limite), propios / "norma_gradiente.png",
                          decimales_eje=None, limite=limite)
    else:
        grafico_perdida(m, registro, mejor, "rpn", topes[familia, "rpn"], propios / "perdida_rpn.png")


def graficos_comparativos(datos, reales, tamanos, n_imagenes, n_clases_test):
    """Los gráficos y tablas con varios modelos juntos."""
    carpeta = DIR_SALIDA / "comparacion"
    carpeta.mkdir(parents=True, exist_ok=True)
    print("YOLOv8n y YOLOv3-tiny desde cero")
    yolos = {m: datos[m] for m in YOLOS_DESDE_CERO}
    comparar_map_val(datos, YOLOS_DESDE_CERO, False, carpeta / "01_map50_95_val_por_epoca.png")
    comparar_map_val(datos, YOLOS_DESDE_CERO, True, carpeta / "02_map50_95_val_segun_tiempo.png")
    comparar_metricas_test(datos, YOLOS_DESDE_CERO, n_imagenes, carpeta / "03_metricas_test.png")
    mapas = {m: map_por_umbral_iou(d["detecciones"], reales) for m, d in yolos.items()}
    for m, valores in mapas.items():   # en 0,50, en 0,75 y en promedio deben dar las métricas del evaluador común
        evaluacion = datos[m]["evaluacion"]
        assert np.allclose([valores[0], valores[5], valores.mean()],
                           [evaluacion["mAP50"], evaluacion["mAP75"], evaluacion["mAP50_95"]]), f"{m}: mAP por IoU distinto"
    pd.DataFrame({MODELOS[m]: v for m, v in mapas.items()},
                 index=pd.Index(np.round(comun.UMBRALES_IOU, 2), name="IoU")).to_csv(carpeta / "04_map_segun_iou_test.csv")
    comparar_map_segun_iou(mapas, carpeta / "04_map_segun_iou_test.png")
    tipos, recall = {}, {}
    for m, d in yolos.items():
        tipos[m], detectado = clasificar_detecciones(d["detecciones"], reales)
        correctas = int((tipos[m] == "Correcta").sum())
        # las correctas deben ser los aciertos del evaluador común sobre el umbral, y cada una detecta un objeto
        assert np.isclose(correctas / len(tipos[m]), d["evaluacion"]["precision_micro"]), f"{m}: correctas distintas"
        assert correctas == detectado.sum(), f"{m}: objetos detectados distintos de las detecciones correctas"
        recall[m] = recall_segun_tamano(detectado, reales, tamanos)
    comparar_tipos_de_deteccion(tipos, carpeta / "05_tipos_de_deteccion_test.png")
    comparar_recall_segun_tamano(recall, carpeta / "06_recall_segun_tamano_test.png")
    comparar_costo(YOLOS_DESDE_CERO, carpeta / "07_costo.png")

    columnas, notas = columnas_de_tabla(datos), notas_de_tabla(n_imagenes, n_clases_test)
    carpeta = DIR_SALIDA / "parte1_cnn_y_yolos"
    print("Parte 1: la CNN y los dos YOLO desde cero")
    guardar_tabla(columnas, COLUMNAS_RESUMEN, DESDE_CERO, notas["resumen"], carpeta / "01_tabla_resumen.png")
    guardar_tabla(columnas, COLUMNAS_TEST, DESDE_CERO, notas["test"], carpeta / "02_tabla_test.png")
    guardar_tabla(columnas, COLUMNAS_COSTO, DESDE_CERO, notas["costo"], carpeta / "03_tabla_costo.png")
    comparar_metricas_test(datos, DESDE_CERO, n_imagenes, carpeta / "04_metricas_test.png")
    comparar_map_val(datos, DESDE_CERO, False, carpeta / "05_map50_95_val_por_epoca.png")
    comparar_map_val(datos, DESDE_CERO, True, carpeta / "06_map50_95_val_segun_tiempo.png")

    carpeta = DIR_SALIDA / "seis_modelos"
    print("Los seis modelos")
    # la tabla y el gráfico van juntos en una diapositiva: filas más bajas, nota corta y gráfico ancho con la leyenda en
    # una fila
    guardar_tabla(columnas, COLUMNAS_TEST, list(MODELOS), notas["test_corta"], carpeta / "01_tabla_test.png",
                  alto_fila=0.52)
    comparar_metricas_test(datos, list(MODELOS), n_imagenes, carpeta / "02_metricas_test.png", ancho=16.0, alto=5.8,
                           columnas=len(MODELOS))
    # los tiempos sin la pasada extra deben ser los de la tabla de costos de 05
    costo = pd.read_csv(DIR_RESULTADOS / "comparacion" / "costo.csv", index_col=0)
    for m in MODELOS:
        registro = datos[m]["registro"]
        hasta_la_mejor = registro.loc[registro.epoca == datos[m]["mejor"], "tiempo_entrenamiento_s"].iloc[0] / 60
        assert np.isclose(hasta_la_mejor, costo.loc["Tiempo hasta la mejor época sin la pasada extra (min)", FILAS_05[m]],
                          atol=1e-6), f"{m}: el tiempo sin la pasada extra no es el de 05"
    comparar_map_test_segun(datos, list(MODELOS), "tiempo", n_imagenes, carpeta / "03_map50_95_test_segun_tiempo.png")
    comparar_map_test_segun(datos, list(MODELOS), "fps", n_imagenes, carpeta / "04_map50_95_test_segun_fps.png")
    comparar_map_val(datos, list(MODELOS), False, carpeta / "05_map50_95_val_por_epoca.png")
    comparar_map_val(datos, list(MODELOS), True, carpeta / "06_map50_95_val_segun_tiempo.png")


def main():
    configurar_estilo()
    particion, huella = comun.cargar_particiones(RAIZ / "splits")
    comun.verificar_huella(huella)
    nombres = yaml.safe_load((DIR_DATASET / "fjordfish.yaml").read_text(encoding="utf-8"))["names"]
    reales = comun.cargar_cajas_reales(DIR_DATASET, "test", particion["test"])
    print("Evaluando las detecciones de test con el evaluador comun...")
    datos = {m: cargar_modelo(m, reales, nombres) for m in MODELOS}
    # mismo eje vertical de mAP para todos los modelos con gráficos propios, para que se comparen a simple vista
    tope_map = {c: 1.15 * max(datos[m]["registro"][c].max() for m in CON_GRAFICOS_PROPIOS)
                for c in ("val_mAP50", "val_mAP50_95")}
    topes = topes_de_perdida(datos)
    print("Graficos:")
    for m in CON_GRAFICOS_PROPIOS:
        graficos_del_modelo(m, datos[m], tope_map, topes, nombres, reales)
    print("Anotaciones reales")
    for ruta_imagen in ejemplos_de_test():
        dibujar_imagen(ruta_imagen, reales[reales.id == ruta_imagen.stem], "white", TINTA_SECUNDARIA, "white",
                       f"Anotación real: imagen {ruta_imagen.stem}", "Test: cajas anotadas en el dataset.",
                       nombres, DIR_SALIDA / "anotacion_real" / f"ejemplo_{ruta_imagen.stem}.png")
    tamanos = {i: Image.open(DIR_DATASET / "images" / "test" / f"{i}.jpg").size for i in particion["test"]}   # solo encabezados
    graficos_comparativos(datos, reales, tamanos, len(particion["test"]), reales.clase_id.nunique())
    print("Arquitectura de la YOLOv3-tiny")
    capas = capas_yolov3_tiny()
    diagrama_yolov3_tiny(capas, DIR_SALIDA / "arquitectura" / "yolov3_tiny_diagrama.png")
    tabla_capas_yolov3_tiny(capas, DIR_SALIDA / "arquitectura" / "yolov3_tiny_capas.png")
    print("Arquitectura de YOLOv8n")
    capas, ramas, gflops = capas_yolov8n(len(nombres))
    diagrama_yolov8n(capas, ramas, gflops, DIR_SALIDA / "arquitectura" / "yolov8n_diagrama.png")


if __name__ == "__main__":
    main()
