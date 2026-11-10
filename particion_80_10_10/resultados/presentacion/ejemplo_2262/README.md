# Ejemplo escogido: imagen 2262 de test

Imagen 2262 del test de la partición 80/10/10. Tiene 5 objetos anotados: 4 spiny_dogfish y 1 goldsinny pequeño. Hay una
imagen con las cajas reales y una por cada YOLO y por Faster R-CNN con sus detecciones de confianza ≥ 0,25 (el umbral
D8), con el NMS de IoU 0,5 de esta carpeta. El formato es el mismo de los ejemplos de cada modelo: la función `dibujar_imagen` de
`graficos_presentacion.py` de esta variante.

- Se escogió porque YOLOv8n preentrenado acierta los 5 objetos (clase correcta e IoU ≥ 0,5), sin cajas de más.
- YOLOv8n pierde el goldsinny.
- La YOLOv3-tiny pierde el spiny_dogfish pequeño que está junto a la red y agrega una caja de cod donde no hay objeto.
- La YOLOv3-tiny preentrenada pierde el goldsinny, y su caja del segundo spiny_dogfish desde la izquierda queda con
  IoU 0,48, bajo el 0,5 que se exige.
- Faster R-CNN da 14 detecciones: acierta 4 objetos, su mejor caja del segundo spiny_dogfish queda justo bajo IoU 0,5,
  y las otras 9 caen sobre el fondo o repiten un pez (6 de ellas son goldsinny).
- No es representativa. En este test, solo 6 de las 131 imágenes con 2 o más objetos salen perfectas en las cuatro YOLO
  a la vez (por ejemplo, la 61), y esta no es una de ellas.
- Ninguna imagen de train es de la misma escena. La más parecida, la 2263, tiene una correlación de 0,90, bajo el
  umbral de 0,95 que se usó para decir que dos imágenes son de la misma escena.
