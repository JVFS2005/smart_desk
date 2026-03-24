import cv2

# Gera os 4 marcadores de canto (IDs 0, 1, 2, 3) e o de régua (ID 4)
dicionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

for marker_id in range(5):
    img = cv2.aruco.generateImageMarker(dicionario, marker_id, 200)
    cv2.imwrite(f"marker_{marker_id}.png", img)
    print(f"Salvo marker_{marker_id}.png")