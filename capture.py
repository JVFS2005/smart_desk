import cv2
import subprocess
import os

FOTO_PATH = "/data/data/com.termux/files/home/frame.jpg"

def capturar_frame(camera=0):
    result = subprocess.run(
        ["termux-camera-photo", "-c", str(camera), FOTO_PATH],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Erro ao capturar: {result.stderr}")
    
    frame = cv2.imread(FOTO_PATH)
    
    if frame is None:
        raise ValueError("Frame inválido")
    
    return frame

def main():
    print("Capturando frame via Termux:API...")
    frame = capturar_frame(camera=0)
    
    altura, largura = frame.shape[:2]
    print(f"Frame capturado: {largura}x{altura} pixels")
    
    cv2.imwrite("frame_teste.jpg", frame)
    print("Salvo em frame_teste.jpg")

if __name__ == "__main__":
    main()