# Guía de los gráficos para las diapositivas

Qué archivo va en cada diapositiva de la presentación de Canva, cómo se lee cada gráfico y la arquitectura de la YOLOv3-tiny para la diapositiva 11. Las rutas son relativas a esta carpeta. Todo lo genera `graficos_presentacion.py`, desde la raíz del proyecto, en 1 minuto y sin la GPU.

Los modelos se llaman igual en todos los archivos y van siempre en este orden: CNN propia, YOLOv8n, YOLOv3-tiny, YOLOv8n preentrenado, YOLOv3-tiny preentrenada y Faster R-CNN. Cada modelo tiene su color en todos los gráficos. Las métricas de test salen del evaluador común (`comun.py`), y el script comprueba que coincidan con las de `05`.

## La copia en Canva

Los gráficos ya están puestos en la presentación «Tarea 2 FjordFish · 20 min · gráficos nuevos», una copia de la de 20 minutos; la original no cambió. La copia tiene 28 diapositivas: hay dos nuevas, así que las siguientes se corren.

| En la copia | En el PDF original | Qué tiene |
|---|---|---|
| 11 | 11 | El texto de los dos YOLO, subido un poco, y el diagrama de la YOLOv3-tiny debajo |
| 12 (nueva) | — | «YOLOv3-tiny: arquitectura»: viñetas (escritas a partir de `02b`; conviene revisarlas) y la tabla de capas |
| 13 y 14 | 12 y 13 | YOLOv8n y YOLOv3-tiny con sus tres gráficos |
| 15 | 14 | Comparación entre la CNN y las YOLO, con los tres gráficos nuevos |
| 16 (nueva) | — | Tabla de desempeño y costo de la CNN y los dos YOLO |
| 17 | 15 | ✍️ Parte 1, sin cambios |
| 18, 19 y 20 | 16, 17 y 18 | Transfer learning con los mismos tres gráficos |
| 21 y 22 | 19 y 20 | Los seis modelos, en el orden nuevo |
| 23 a 28 | 21 a 26 | Sin cambios |

La tabla que sigue usa la numeración del PDF original.

## Qué va en cada diapositiva

| Diapositiva | Archivos | Cómo ubicarlos |
|---|---|---|
| 11 · YOLO: 2 modelos | `arquitectura/yolov3_tiny_diagrama.png` | A todo el ancho, bajo el texto. Mide 16 × 4,1 pulgadas: para que se lea, conviene acortar el texto de YOLOv8n o pasar la YOLOv3-tiny a una diapositiva propia. `arquitectura/yolov3_tiny_capas.png` es la tabla de capas, como la de la CNN en la diapositiva 7. |
| 12 · YOLOv8n | `yolo_desde_cero/01_perdida_total.png`, `05_map50_95_val.png` y `10_ejemplo_305.png` | Los tres en una fila, bajo los puntos. Tienen el mismo tamaño. |
| 13 · YOLOv3-tiny | Los mismos tres, de `yolo_a_mano/` | Igual que la 12. |
| 14 · Comparación entre la CNN y las YOLO | `parte1_cnn_y_yolos/04_metricas_test.png`, `05_map50_95_val_por_epoca.png` y `06_map50_95_val_segun_tiempo.png` | Reemplazan la figura de `05` que tiene hoy: son los mismos tres paneles, por separado y con los nombres nuevos. |
| 15 · Tabla de la comparación | `parte1_cnn_y_yolos/01_tabla_resumen.png` | Desempeño en test y costo en una sola tabla. A todo el ancho ocupa casi la mitad del alto, y el texto ✍️ cabe debajo. Con más espacio, se puede reemplazar por `02_tabla_test.png` y `03_tabla_costo.png`. |
| 16 · YOLOv8n preentrenado | Los mismos tres de la 12, de `yolo_preentrenado/` | Reemplazan los 8 paneles de hoy. |
| 17 · YOLOv3-tiny preentrenada | Los mismos tres, de `yolo_a_mano_preentrenado/` | Reemplazan los 8 paneles de hoy. |
| 18 · Faster R-CNN | Los mismos tres, de `faster_rcnn/` | Reemplazan los 5 paneles de hoy. |
| 19 · Los seis modelos | `seis_modelos/01_tabla_test.png` y `02_metricas_test.png` | La tabla arriba y el gráfico debajo, como hoy, en el orden nuevo. |
| 20 · Tiempo, FPS y validación | `seis_modelos/03_map50_95_test_segun_tiempo.png`, `04_map50_95_test_segun_fps.png`, `05_map50_95_val_por_epoca.png` y `06_map50_95_val_segun_tiempo.png` | Reemplazan los cuatro paneles de `05` que tiene hoy, en la misma disposición de 2 × 2, con los nombres, el orden y los colores nuevos. |

Así, las cinco diapositivas de modelos (12, 13, 16, 17 y 18) muestran los mismos tres gráficos, con el mismo tamaño y el mismo eje de mAP.

Otros archivos que pueden servir para las secciones ✍️ o para una diapositiva extra:
- En la carpeta de cada modelo: `02_perdida_caja`, `03_perdida_clase`, `04_map50_val`, `06_tasa_aprendizaje`, `07_ap50_por_clase_test`, `08_curva_pr_test`, `09_matriz_confusion_test` y las otras tres imágenes de ejemplo. En `solo_este_modelo/` están los registros que no tienen todos los modelos.
- `anotacion_real/`: las cajas anotadas de las cuatro imágenes de ejemplo.
- `comparacion/`: YOLOv8n y YOLOv3-tiny desde cero juntos. Incluye el mAP según el umbral de IoU, los tipos de detección, el recall según el tamaño del objeto y el costo.
- `arquitectura/yolov8n_diagrama.png`: el diagrama de YOLOv8n, con el mismo formato y la misma escala que el de la YOLOv3-tiny. Sus capas, con la salida y los parámetros de cada una, están en `yolov8n_capas.csv`. Es el mismo en las dos particiones.

## Cómo se lee cada gráfico

### Pérdida total (`01_perdida_total`)
- El eje horizontal son las épocas. La línea gris es la pérdida de train, y la del color del modelo, la de validación. La línea punteada marca la mejor época.
- El subtítulo dice qué suma esa pérdida en cada modelo. Solo comparten escala los modelos con la misma pérdida: las dos YOLOv8n entre sí y las dos YOLOv3-tiny entre sí. Faster R-CNN tiene su propia escala, y su pérdida es por lote de 4 imágenes, no por imagen. Los valores de pérdida de modelos distintos no se comparan.
- Qué señalar: cuánto baja cada curva, desde qué época se aplana y cómo cambia la distancia entre train y validación.

### mAP50-95 de validación (`05_map50_95_val`)
- El eje vertical es el mismo en los cinco modelos (de 0 a 0,31), así que las alturas se pueden comparar entre diapositivas. El eje de épocas es el mismo en los cuatro YOLO (de 0 a 250). Faster R-CNN tiene el suyo (de 0 a 55) porque entrenó 52 épocas.
- El punto marca la mejor época y su valor. Ese es el criterio de la parada temprana y el modelo que se evalúa en test. Cada curva termina 50 épocas después de su mejor época (25 en Faster R-CNN): es la paciencia.
- Qué señalar: la mejor época y su valor, cuánto sube la curva en las primeras épocas y desde cuándo se aplana.

### Imagen de ejemplo (`10_ejemplo_305`)
- Es la imagen 305 de test con las detecciones de confianza ≥ 0,25 (el umbral D8), con su clase y su confianza. Es la misma imagen en los cinco modelos y la primera de la diapositiva de la CNN.
- `anotacion_real/ejemplo_305.png` muestra las 6 cajas anotadas: 3 corkwing_female, 1 corkwing_male y 2 labrids_unknown, varias pequeñas.
- Qué señalar: cuántos de los 6 peces detecta, con qué clase, y si hay cajas repetidas o sin pez.

### Barras de desempeño en test (`04_metricas_test` y `seis_modelos/02_metricas_test`)
- La precisión, el recall y el F1 usan solo las detecciones con confianza ≥ 0,25. El mAP no usa umbral. Todas salen del evaluador común en las 916 imágenes de test.
- En el de seis modelos, la leyenda se lee por filas: arriba los entrenados desde cero y abajo los preentrenados. Los valores exactos están en la tabla que lo acompaña.

### mAP50-95 de validación de varios modelos (`05_map50_95_val_por_epoca` y `06_map50_95_val_segun_tiempo`)
- Son las curvas de cada modelo en un mismo gráfico, y el punto marca su mejor época. Con tres modelos (diapositiva 14) el punto lleva su valor. Con seis (diapositiva 20) no, porque los valores taparían las curvas; el valor de cada uno está en su propio gráfico `05_map50_95_val`.
- En el de tiempo, el eje horizontal son los minutos de entrenamiento en la misma GPU, con la validación de cada época incluida. Muestra cuánto tiempo tomó llegar a cada valor.

### mAP50-95 de test según el tiempo y según los FPS (`seis_modelos/03` y `04`)
- Cada punto es un modelo, con su nombre al lado. La altura es su mAP50-95 en test.
- En el de tiempo, el eje horizontal es el tiempo de entrenamiento hasta la parada temprana: más a la izquierda es menos tiempo. En el de FPS, son las imágenes por segundo al predecir: más a la derecha es más rápido.
- Qué señalar: dónde queda cada modelo en las dos medidas a la vez.

### Columnas de las tablas
- **Precisión, recall y F1:** con IoU 0,5 y confianza ≥ 0,25, promediados entre clases.
- **mAP50, mAP75 y mAP50-95:** sin umbral de confianza. El mAP50-95 es el promedio de los 10 umbrales de IoU, de 0,50 a 0,95.
- **Clases con detecciones:** de las 24 clases con objetos en test, cuántas tienen al menos una detección con confianza ≥ 0,25.
- **Precisión micro:** fracción de todas las detecciones con confianza ≥ 0,25 que son correctas, sin promediar por clase.
- **Épocas (mejor):** las épocas que se ejecutaron y, entre paréntesis, la mejor.
- **Tiempo total:** el entrenamiento hasta la parada temprana, con la validación de cada época.
- **FPS (lote 1):** imágenes por segundo en float32, incluidos el preproceso y el NMS.

## Precauciones al presentarlos

- **El mAP de las curvas no es el de las tablas.** Las curvas muestran la validación que hizo cada entrenamiento al final de cada época, en float16. Las tablas son de test, otra partición, y salen del evaluador común en float32. Por eso el mejor valor de una curva no coincide con ningún número de las tablas.
- **Los FPS de los modelos pequeños varían.** En dos mediciones seguidas variaron hasta 40 %, y `05` usa la segunda. Las tablas y el gráfico de FPS lo indican.
- **El tiempo de Faster R-CNN no incluye la pasada extra.** En los gráficos de tiempo de la diapositiva 20, igual que en `05`, Faster R-CNN aparece sin la pasada que hace por validación solo para calcular la pérdida: 153,1 min en vez de los 163,8 que dice la diapositiva 18.
- **Las pérdidas no se comparan entre familias de modelos:** cada una se define distinto (ver el subtítulo de cada gráfico).
- **Los colores son los de `05`, salvo la YOLOv3-tiny.** Junto al naranja de YOLOv8n, el rosado claro de `05` casi no se distingue, así que en todos los gráficos nuevos la YOLOv3-tiny es rosado oscuro. Si alguna diapositiva conserva una figura de `05`, tendrá el rosado claro y los nombres antiguos («YOLO desde cero», «YOLOv3-tiny a mano»…).

## ✍️ Pendiente del grupo

- Las diapositivas ✍️ (15, 21, 22, 25 y 26) son del grupo: la justificación, el análisis, las conclusiones y la declaración de IA. Esta guía explica cómo leer los gráficos, pero no los interpreta.
- La declaración de uso de IA debe mencionar que la IA generó `graficos_presentacion.py`, que produce estos gráficos, tablas y los dos diagramas, y la explicación de la arquitectura de la YOLOv3-tiny de esta guía.

## Diapositiva 11: la YOLOv3-tiny, punto por punto

Es la red de `02b_yolo_a_mano.ipynb`, escrita en PyTorch sin Ultralytics. El diagrama la muestra completa; estos son sus partes, en el orden en que pasa la imagen.

1. **Entrada.** Una imagen de 640 × 640 × 3 con *letterbox*, igual que en los demás modelos.
2. **Cuerpo (8 convoluciones y 6 *max-pool*).** Son 7 convoluciones 3×3, de 16 a 1.024 filtros, y una 1×1 de 256. Cada una lleva BatchNorm y LeakyReLU con pendiente 0,1, que multiplica los valores negativos por 0,1 en vez de anularlos. Hay 5 *max-pool* 2×2 de stride 2 que reducen el mapa de 640 a 20 (640 / 2⁵ = 20, es decir, stride 32). Un sexto *max-pool* de stride 1, con 1 px de relleno a la derecha y abajo, combina cada posición con sus vecinas sin achicar el mapa. Es una cadena simple de convoluciones, sin los bloques C2f ni SPPF de YOLOv8n.
3. **Cabeza de stride 32 (objetos grandes).** Una convolución 3×3 de 512 filtros y una de salida 1×1 sobre la grilla de 20 × 20. Cada celda cubre 32 × 32 px y usa las 3 anclas más grandes.
4. **Puente y cabeza de stride 16 (objetos pequeños).** El puente toma el mismo mapa de 20 × 20 × 256, lo reduce a 128 canales con una convolución 1×1 y lo amplía ×2 hasta 40 × 40, copiando cada valor a sus vecinos. Después lo concatena con el mapa de 40 × 40 × 256 que se guardó del cuerpo, y quedan 384 canales. Siguen una convolución 3×3 de 256 y una de salida 1×1. Cada celda cubre 16 × 16 px y usa las 3 anclas más pequeñas. Así combina la información de las capas profundas con un mapa de más resolución. Es la misma idea del cuello PAN de YOLOv8n, pero en un solo sentido y una sola vez.
5. **Anclas.** Son 6 cajas de referencia que se calcularon con k-means sobre el ancho y el alto de las cajas de train, con la distancia 1 − IoU. Las de stride 32 miden 96 × 36, 140 × 75 y 295 × 148 px, y las de stride 16, 21 × 11, 43 × 21 y 54 × 56 px. En promedio, cada caja de train tiene un IoU de 0,62 con su ancla más parecida.
6. **Salida.** Por cada celda y ancla hay 33 valores: 4 de caja, 1 de objeto y 28 de clase. Son 3 × 33 = 99 canales por celda y 20² × 3 + 40² × 3 = 6.000 cajas candidatas por imagen (la CNN propia tiene 4.000).
   - Caja: centro x = (columna + sigmoide(t_x)) × stride, e igual para y; ancho = ancho del ancla × exp(t_w), e igual para el alto. La red predice cuánto ajustar el ancla.
   - Objeto: la probabilidad de que la candidata contenga un objeto. Cada una de las 28 clases tiene su propia sigmoide. La confianza final es objeto × clase.
   - Al inicio del entrenamiento, cada candidata parte con probabilidad de objeto 0,01 y probabilidad 1/28 en cada clase.
7. **Asignación al entrenar.** Cada caja real tiene una sola candidata positiva: el ancla con mayor IoU de forma con ella, que define la escala, en la celda que contiene su centro. Las demás candidatas aprenden a predecir «sin objeto», salvo las que ya cubren bien un objeto (punto 8).
8. **Pérdida (la de YOLOv3).** Es la suma de cuatro términos, cada uno con peso 1:
   - Caja: error cuadrático de sigmoide(t_x), sigmoide(t_y), t_w y t_h, multiplicado por 2 − ancho · alto, para dar más peso a las cajas pequeñas.
   - Objeto: BCE en las positivas.
   - Fondo: BCE en las demás candidatas, sin contar las que ya cubren un objeto con IoU ≥ 0,5.
   - Clase: BCE en las positivas.
   La pérdida se multiplica por 0,25 (D11).
9. **Tamaño.** Tiene 8,73 M de parámetros y 13,0 GFLOPs. La convolución 3×3 de 1.024 filtros concentra 4,72 M de parámetros (54 %) y 3,77 GFLOPs (29 %). La convolución 3×3 de 256 de la cabeza de stride 16 suma 2,83 GFLOPs (22 %), porque trabaja sobre la grilla de 40 × 40.

Diferencias con YOLOv8n (según la tabla de la introducción de `02b`):

| | YOLOv8n (`02`) | YOLOv3-tiny (`02b`) |
|---|---|---|
| Cuerpo | Bloques C2f y SPPF | Convoluciones 3×3 en cadena y *max-pool* |
| Escalas de salida | 3 (stride 8, 16 y 32) | 2 (stride 16 y 32) |
| Caja | Sin anclas: distancias a cada lado, con DFL | 3 anclas por celda, con ajuste del centro, el ancho y el alto |
| Asignación | Varias positivas por caja real (*task-aligned*) | Una positiva por caja real |
| Pérdida | CIoU + DFL + BCE de clase | Error cuadrático de caja + BCE de objeto, de fondo y de clase |
| Confianza | Probabilidad de clase | Probabilidad de objeto × probabilidad de clase |
| Tamaño | 3,02 M de parámetros y 8,2 GFLOPs | 8,73 M de parámetros y 13,0 GFLOPs |
