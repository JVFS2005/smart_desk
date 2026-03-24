import cv2
import numpy as np

dicionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker = cv2.aruco.generateImageMarker(dicionario, 4, 200)

# Canvas maior: marker + régua ao lado
canvas = np.ones((200, 500), dtype=np.uint8) * 255
canvas[0:200, 0:200] = marker

# Régua: traços a cada 50px = 1cm (se impresso a 50px/cm = 127dpi)
# Ajuste a escala conforme sua impressora
for i, cm in enumerate(range(0, 6)):
    x = 220 + i * 118
    altura = 40 if cm % 2 == 0 else 25
    cv2.line(canvas, (x, 180), (x, 180 - altura), 0, 2)
    if cm % 2 == 0:
        cv2.putText(canvas, f"{cm}cm", (x - 10, 175 - altura),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)

cv2.imwrite("marker_4_regua.png", canvas)