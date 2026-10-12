#!/usr/bin/env bash
# Baut unter Linux (Ubuntu/Debian) die Windows-Version von Bildwerk und den Installer.
#
# Voraussetzungen:
#   sudo apt install mingw-w64 nsis python3-pip
#   rustup target add x86_64-pc-windows-gnu      (oder rust-src für -Zbuild-std, siehe unten)
#   python3 tools/modelle_laden.py               (alle KI-Modelle in models/)
#
# Ergebnis: dist/Bildwerk-Setup-<Version>.exe
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
TARGET=x86_64-pc-windows-gnu
WORK=target/windows-build
STAGE=$WORK/stage
mkdir -p "$WORK" dist

# ---- 1. Laufzeitbibliotheken (ONNX Runtime mit DirectML, Visual-C++-Laufzeit) aus PyPI holen
RT=$WORK/runtime
if [ ! -f "$RT/onnxruntime.dll" ]; then
  mkdir -p "$RT/whl"
  pip download onnxruntime-directml==1.24.4 msvc-runtime --no-deps --only-binary=:all: \
      --platform win_amd64 --python-version 3.12 -d "$RT/whl" -q
  python3 - "$RT" <<'EOF'
import glob, os, sys, zipfile
rt = sys.argv[1]
want = {
    "onnxruntime/capi/onnxruntime.dll", "onnxruntime/capi/onnxruntime_providers_shared.dll",
    "onnxruntime/capi/DirectML.dll", "onnxruntime/LICENSE", "onnxruntime/ThirdPartyNotices.txt",
}
crt = {"msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll"}
for whl in glob.glob(os.path.join(rt, "whl", "*.whl")):
    z = zipfile.ZipFile(whl)
    for n in z.namelist():
        base = n.split("/")[-1]
        if n in want or (n.split("/")[-2:-1] == ["Scripts"] and base in crt):
            open(os.path.join(rt, base), "wb").write(z.read(n))
EOF
fi

# ---- 2. Programm kreuzkompilieren
export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc
if rustup target list --installed 2>/dev/null | grep -q "$TARGET"; then
  cargo build --release --target $TARGET --no-default-features --features windows-installer
else
  # Ohne vorinstalliertes Windows-Target: Standardbibliothek aus rust-src mitbauen
  RUSTC_BOOTSTRAP=1 cargo build --release -Zbuild-std --target $TARGET \
      --no-default-features --features windows-installer
fi

# ---- 3. Programmordner zusammenstellen
rm -rf "$STAGE"
mkdir -p "$STAGE/models/face" "$STAGE/licenses"
cp target/$TARGET/release/bildwerk.exe "$STAGE/Bildwerk.exe"
cp "$RT"/*.dll "$STAGE/"
cp models/*.onnx "$STAGE/models/"
cp models/face/*.onnx "$STAGE/models/face/"
for m in RealESRGAN_x4plus RealESRGAN_x2plus RealESRGAN_x4plus_anime_6B; do
  [ -f "$STAGE/models/$m.onnx" ] || { echo "Modell $m fehlt – erst tools/modelle_laden.py ausführen"; exit 1; }
done
[ -f "$STAGE/models/face/gfpgan_1.4.onnx" ] || { echo "GFPGAN fehlt – erst tools/modelle_laden.py ausführen"; exit 1; }
cp installer/licenses/*.txt "$STAGE/licenses/"
cp "$RT/LICENSE" "$STAGE/licenses/ONNX-Runtime_MIT.txt"
cp "$RT/ThirdPartyNotices.txt" "$STAGE/licenses/ONNX-Runtime_ThirdPartyNotices.txt"

# Textdateien für Windows: UTF-16 mit BOM (NSIS-Lizenzseite) bzw. CRLF
python3 - "$STAGE" <<'EOF'
import sys, os
stage = sys.argv[1]
for name in ("LIZENZ.txt", "LIESMICH.txt"):
    text = open(os.path.join("installer", name), encoding="utf-8").read().replace("\n", "\r\n")
    open(os.path.join(stage, name), "w", encoding="utf-8-sig", newline="").write(text)
lic = open("installer/LIZENZ.txt", encoding="utf-8").read().replace("\n", "\r\n")
open("target/windows-build/LIZENZ-utf16.txt", "wb").write(b"\xff\xfe" + lic.encode("utf-16-le"))
EOF

# ---- 4. Installer bauen
cp installer/icon.ico installer/welcome.bmp installer/header.bmp "$WORK/"
cp installer/bildwerk.nsi "$WORK/"
cp "$WORK/LIZENZ-utf16.txt" "$WORK/LIZENZ.txt"
OUT="$(pwd)/dist/Bildwerk-Setup-$VERSION.exe"
( cd "$WORK" && makensis -V2 -DVERSION="$VERSION" -DSTAGE="stage" -DOUTFILE="$OUT" bildwerk.nsi )
ls -lh "$OUT"
