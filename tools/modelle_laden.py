#!/usr/bin/env python3
"""
Lädt die Real-ESRGAN-Gewichte von GitHub und wandelt sie in ONNX-Modelle für Bildwerk um.

Aufruf (im Projektordner):   python tools/modelle_laden.py
Benötigt:                    pip install onnx numpy
"""
import os
import subprocess
import sys
import tempfile
import urllib.request

BASE = "https://github.com/xinntao/Real-ESRGAN/releases/download"
MODELLE = [
    "v0.2.5.0/realesr-general-x4v3.pth",
    "v0.2.5.0/realesr-general-wdn-x4v3.pth",
    "v0.1.0/RealESRGAN_x4plus.pth",
    "v0.2.1/RealESRGAN_x2plus.pth",
    "v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
]

# Gesichtsrestaurierung: fertige ONNX-Dateien
GESICHT = {
    "gfpgan_1.4.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx",
    "face_detection_yunet_2023mar.onnx": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
}

hier = os.path.dirname(os.path.abspath(__file__))
ziel = os.path.join(hier, "..", "models")
os.makedirs(ziel, exist_ok=True)

with tempfile.TemporaryDirectory() as tmp:
    for rel in MODELLE:
        name = os.path.basename(rel)
        onnx_pfad = os.path.join(ziel, name.replace(".pth", ".onnx"))
        if os.path.exists(onnx_pfad):
            print(f"✓ {os.path.basename(onnx_pfad)} ist schon vorhanden")
            continue
        pth = os.path.join(tmp, name)
        print(f"↓ {name} …")
        urllib.request.urlretrieve(f"{BASE}/{rel}", pth)
        subprocess.run([sys.executable, os.path.join(hier, "pth_to_onnx.py"), pth, onnx_pfad], check=True)

face_dir = os.path.join(ziel, "face")
os.makedirs(face_dir, exist_ok=True)
for name, url in GESICHT.items():
    pfad = os.path.join(face_dir, name)
    if os.path.exists(pfad):
        print(f"✓ face/{name} ist schon vorhanden")
        continue
    print(f"↓ face/{name} …")
    urllib.request.urlretrieve(url, pfad + ".part")
    os.replace(pfad + ".part", pfad)

print("Fertig. Modelle liegen in", os.path.abspath(ziel))
