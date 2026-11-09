# Variante con la partición 80/10/10

Copia de los cuadernos de la tarea para entrenar y evaluar los seis modelos con otra partición: las 3002 imágenes de
FjordFish repartidas en 80 % train, 10 % val y 10 % test, con la misma semilla (42) y la misma estratificación de `00`.
Nada de la carpeta principal se modifica.

| | Carpeta principal | Esta variante |
|---|---|---|
| Partición | train 1773, val 313 y test 916 (huella `7d4e588f7cc4`) | train 2400, val 301 y test 301 (huella `84ea7ae5b688`) |
| Test | la carpeta `val` del zip | un 10 % sorteado de las dos carpetas del zip |
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
- `05` y `06`: solo la nota del inicio. `06` evalúa el mismo modelo del original (`CLAVE_MODELO`).
- `graficos_presentacion.py`: lee los entrenamientos de `~/tarea2_runs_80_10_10/reales/` y ajusta el eje de épocas a las
  que corrieron.

Orden de ejecución, con la GPU libre: `00`, `01`, `02`, `02b`, `03`, `03b`, `04`, `05`, `06` y, al final,
`py graficos_presentacion.py`.

Estado: los seis modelos se entrenaron entre el 8 y el 9 de noviembre de 2026 (10,6 h en total). Después se ejecutaron
`05`, `06` y los gráficos, sin errores. Los pesos quedaron en `~/tarea2_runs_80_10_10/reales/` del computador donde se
entrenó y no están en el repositorio; con `ENTRENAR = False`, los cuadernos los leen desde ahí. Como el test es otro,
las métricas de esta carpeta no se comparan directamente con las de la carpeta principal.
