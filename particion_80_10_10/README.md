# Variante con la partición 80/10/10

Copia de los cuadernos de la tarea para entrenar y evaluar los seis modelos con otra partición: las 3002 imágenes de
FjordFish repartidas en 80 % train, 10 % val y 10 % test, con la misma semilla (42) y la misma estratificación de `00`.
Nada de la carpeta principal se modifica.

| | Carpeta principal | Esta variante |
|---|---|---|
| Partición | train 1773, val 313 y test 916 (huella `7d4e588f7cc4`) | train 2400, val 301 y test 301 (huella `84ea7ae5b688`) |
| Test | la carpeta `val` del zip | un 10 % sorteado de las dos carpetas del zip |
| NMS al evaluar | IoU 0,7 | IoU 0,5 (desde el 9 de noviembre de 2026) |
| Pesos y registros | `~/tarea2_runs/` | `~/tarea2_runs_80_10_10/` |
| Dataset, particiones y resultados | `datasets/`, `splits/` y `resultados/` | los mismos nombres, dentro de esta carpeta |

Qué cambia en las copias:
- `00`: junta las dos carpetas del zip y hace dos cortes estratificados (primero test y después val). Lee el zip de la
  carpeta principal.
- `comun.py`: la huella nueva y `CARPETA_RUNS`.
- `01` a `04`: las rutas y los textos que dependen de la partición. El diseño, los hiperparámetros y las decisiones del
  grupo son los mismos. Se entrenaron con `ENTRENAR = True` y quedaron con `ENTRENAR = False`, como los originales.
- `01` y `02b`: las pruebas de la fase de diseño (sobreajuste y escala de la pérdida) no se repiten; se muestran las
  guardadas en la carpeta principal.
- `05` y `06`: la nota del inicio y el NMS. `06` evalúa el mismo modelo del original (`CLAVE_MODELO`).
- NMS de la evaluación: IoU 0,5 en vez de 0,7 (cambio del 9 de noviembre de 2026), en `IOU_NMS` de `01` a `06`. Solo
  cambia la evaluación: los entrenamientos no se repitieron, y la validación de cada época, con la que se eligió la mejor
  época, se hizo con 0,7. En `05`, la sección B compara ahora el 0,5 con límites de 0,3 y 0,1. El umbral de confianza
  D8 se mantiene en 0,25 (decisión del 9 de noviembre de 2026), aunque con NMS 0,5 el F1 promedio de validación es
  máximo en 0,15 (0,341, contra 0,334 en 0,25).
- `graficos_presentacion.py`: lee los entrenamientos de `~/tarea2_runs_80_10_10/reales/` y ajusta el eje de épocas a las
  que corrieron.

Orden de ejecución, con la GPU libre: `00`, `01`, `02`, `02b`, `03`, `03b`, `04`, `05`, `06` y, al final,
`py graficos_presentacion.py`.

Estado: los seis modelos se entrenaron entre el 8 y el 9 de noviembre de 2026 (10,6 h en total). Después se ejecutaron
`05`, `06` y los gráficos, sin errores. Los pesos quedaron en `~/tarea2_runs_80_10_10/reales/` del computador donde se
entrenó y no están en el repositorio; con `ENTRENAR = False`, los cuadernos los leen desde ahí. El 9 de noviembre, con el NMS en 0,5,
se volvieron a ejecutar `01` a `06` con `ENTRENAR = False` y los gráficos, en unos 12 minutos y sin errores. Las salidas
del entrenamiento que mostraban los cuadernos quedaron en el commit `78215d9`. Como el test y el NMS son otros, las
métricas de esta carpeta no se comparan directamente con las de la carpeta principal.
