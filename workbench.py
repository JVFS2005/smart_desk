import cv2
import numpy as np
import subprocess

FOTO_PATH = "/data/data/com.termux/files/home/frame.jpg"
OUTPUT_PATH = "/data/data/com.termux/files/home/resultado.jpg"

DICIONARIO = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
PARAMETROS = cv2.aruco.DetectorParameters()
DETECTOR = cv2.aruco.ArucoDetector(DICIONARIO, PARAMETROS)

# IDs dos cantos: 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
# (ajuste a ordem conforme você colou na bancada)
ORDEM_CANTOS = [0, 1, 2, 3]
ID_REGUA = 4
CM_REGUA = 5.0  # distância conhecida entre início e fim da régua em cm

def capturar():
    subprocess.run(["termux-camera-photo", "-c", "0", FOTO_PATH],
                   capture_output=True, timeout=10)
    return cv2.imread(FOTO_PATH)

def detectar_markers(frame):
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cantos, ids, _ = DETECTOR.detectMarkers(cinza)
    if ids is None:
        return {}
    return {int(ids[i]): cantos[i][0] for i in range(len(ids))}

def centro(corners):
    return corners.mean(axis=0)

def aplicar_homografia(frame, markers):
    # Pega o centro de cada marker de canto na ordem definida
    pts_src = np.array([
        centro(markers[id]) for id in ORDEM_CANTOS
    ], dtype=np.float32)

    # Tamanho da imagem de saída (bancada retificada)
    W, H = 800, 600
    pts_dst = np.array([
        [0, 0], [W, 0], [W, H], [0, H]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    retificada = cv2.warpPerspective(frame, M, (W, H))
    return retificada, M

def calcular_escala(markers, M):
    # Transforma os cantos do marker de régua para o espaço retificado
    corners_regua = markers[ID_REGUA]
    ones = np.ones((4, 1))
    pts = np.hstack([corners_regua, ones]).T  # 3x4
    pts_transformados = M @ pts
    pts_transformados /= pts_transformados[2]  # normaliza

    # Usa o lado esquerdo e direito do marker como referência
    x_esq = pts_transformados[0, 0]
    x_dir = pts_transformados[0, 1]
    px_por_cm = abs(x_dir - x_esq) / CM_REGUA
    return px_por_cm

def medir_objeto(retificada, px_por_cm):
    # Subtração de fundo simples: detecta contornos sobre fundo claro
    cinza = cv2.cvtColor(retificada, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(cinza, 100, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

    resultado = retificada.copy()
    for c in contornos:
        area = cv2.contourArea(c)
        if area < 500:  # ignora ruído pequeno
            continue
        x, y, w, h = cv2.boundingRect(c)
        largura_cm = w / px_por_cm
        altura_cm = h / px_por_cm
        cv2.rectangle(resultado, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(resultado, f"{largura_cm:.1f}x{altura_cm:.1f}cm",
                    (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return resultado

def main():
    print("Capturando frame...")
    frame = capturar()

    print("Detectando markers...")
    markers = detectar_markers(frame)
    print(f"Markers encontrados: {list(markers.keys())}")

    ids_necessarios = set(ORDEM_CANTOS + [ID_REGUA])
    faltando = ids_necessarios - set(markers.keys())
    if faltando:
        print(f"Markers não encontrados: {faltando}")
        print("Verifique iluminação e posição da câmera.")
        return

    print("Aplicando homografia...")
    retificada, M = aplicar_homografia(frame, markers)

    print("Calculando escala...")
    px_por_cm = calcular_escala(markers, M)
    print(f"Escala: {px_por_cm:.1f} px/cm")

    print("Medindo objetos...")
    resultado = medir_objeto(retificada, px_por_cm)

    cv2.imwrite(OUTPUT_PATH, resultado)
    print(f"Salvo em {OUTPUT_PATH}")

if __name__ == "__main__":
    main()