# Bildwerk

KI-Bildverbesserung in Rust – Vergrößern, Entrauschen und Nachschärfen, mit grafischer Oberfläche und Vorher/Nachher-Vergleich. Läuft komplett offline auf Windows, macOS und Linux.

## Funktionen

- **KI-Upscaling ×2 / ×4** mit Real-ESRGAN-Modellen (Foto, Allgemein, Anime/Grafik), oder **1×**: nur verbessern, Größe bleibt
- **Entrauschen** über das Modell „Allgemein + Entrauschen“
- **Gesichtsrestaurierung** mit GFPGAN 1.4: Gesichter werden automatisch gefunden (auch gedreht, mehrere pro Bild), ausgerichtet, restauriert und weich zurückgeblendet – mit eigenem Stärke-Regler
- **Nachschärfen** live: Stärke, Radius, Schwellenwert, Halo-Schutz, „nur Kanten“ – geschärft wird nur die Helligkeit, daher keine Farbsäume
- **KI-Stärke**: mischt das KI-Ergebnis mit dem klassisch vergrößerten Original, gegen „Plastik-Look“
- **Kachelverarbeitung**: auch sehr große Bilder, nahtlos, Kachelgröße einstellbar
- **Vergleichsansicht** mit verschiebbarem Teiler, Zoom (Mausrad), Verschieben (Ziehen), Doppelklick = Einpassen
- Laden: PNG, JPEG, TIFF, WebP, BMP (auch 16 Bit) · Speichern: PNG/TIFF in 16 Bit, JPEG, WebP · Transparenz bleibt erhalten
- Drag & Drop, Strg/Cmd+O, Strg/Cmd+S
- **Kommandozeile** mit Stapelverarbeitung ganzer Ordner

## Installation

1. Rust installieren: <https://rustup.rs>
2. Modelle: `models/` enthält bereits die zwei schnellen Modelle und den Gesichtsdetektor. Die großen Modelle (Foto ×4/×2, Anime, GFPGAN mit 340 MB) lädt dieses Skript von GitHub und wandelt sie bei Bedarf um:
   ```
   pip install onnx numpy
   python tools/modelle_laden.py
   ```
3. Bauen und starten:
   ```
   cargo run --release
   ```
   Beim ersten Bauen lädt das Crate `ort` die passende ONNX Runtime automatisch herunter.

### GPU-Beschleunigung

Ohne Zusatz läuft die KI auf der CPU. Zur Orientierung, gemessen auf nur 2 CPU-Kernen: Das schnelle Modell braucht für ein 1000 × 1000-Bild (→ 4000 × 4000) etwa 17 Sekunden, das große Foto-Modell etwa 10-mal so lange. Ein aktueller 8-Kern-Prozessor ist spürbar schneller, eine Grafikkarte nochmals um ein Vielfaches. Für die Grafikkarte:

| System | Befehl |
|---|---|
| Windows (NVIDIA, AMD, Intel) | `cargo run --release --features directml` |
| Windows/Linux mit NVIDIA + CUDA 12 | `cargo run --release --features cuda` |
| macOS (Apple Silicon) | `cargo run --release --features coreml` |

Steht die GPU nicht zur Verfügung, fällt das Programm automatisch auf die CPU zurück.

### Windows-Installer

`dist/Bildwerk-Setup-<Version>.exe` enthält Programm, alle KI-Modelle, ONNX Runtime mit DirectML (GPU-Beschleunigung für NVIDIA, AMD und Intel) und die Visual-C++-Laufzeit. Er legt Startmenü- und optional Desktop-Verknüpfung an, trägt Bildwerk in „Öffnen mit“ und ins Explorer-Kontextmenü („Mit Bildwerk verbessern“) ein und lässt sich über „Apps & Features“ wieder vollständig entfernen.

Gebaut wird er unter Linux mit `tools/build_windows.sh` (MinGW + NSIS, siehe Kopf des Skripts).

Da der Installer nicht digital signiert ist, zeigt Windows beim ersten Start „Der Computer wurde durch Windows geschützt“. Über „Weitere Informationen“ → „Trotzdem ausführen“ geht es weiter.

### Weitergeben

Die fertige Programmdatei liegt in `target/release/`. Kopiere sie zusammen mit der danebenliegenden ONNX-Runtime-Bibliothek (`onnxruntime.dll` / `libonnxruntime.*`) und dem Ordner `models/` in einen gemeinsamen Ordner.

Unter Linux werden Dateidialoge über das XDG-Desktop-Portal angezeigt (bei GNOME/KDE vorhanden).

## Kommandozeile

```
bildwerk --cli -i foto.jpg -o foto_gross.png --model RealESRGAN_x4plus --sharpen 0.6
bildwerk --cli -i Ordner/ -o Ausgabe/ --model realesr-general-wdn-x4v3 --scale 2
bildwerk --cli -i familie.jpg -o familie_neu.png --model realesr-general-wdn-x4v3 --face 0.7
bildwerk --cli -i scan.tif -o scan_scharf.tif --model none --scale 1 --sharpen 1.2 --edges
bildwerk --list-models
bildwerk --help
```

## Aufbau

| Datei | Inhalt |
|---|---|
| `src/app.rs` | Oberfläche (egui), Hintergrund-Threads, Vergleichsansicht |
| `src/engine.rs` | Modelle finden/laden (ONNX Runtime), Kachel-Inferenz, Bild-Ein-/Ausgabe, Verarbeitungskette |
| `src/face.rs` | Gesichtserkennung (YuNet), Ausrichtung per Ähnlichkeitstransformation, GFPGAN, Rückprojektion mit weicher Maske |
| `src/sharpen.rs` | Nachschärfen (Unscharf maskieren auf Luminanz, Schwelle, Kantenmaske, Halo-Begrenzung) |
| `installer/` | NSIS-Skript, Grafiken, Lizenztexte für den Windows-Installer |
| `tools/build_windows.sh` | Baut Windows-EXE und Installer unter Linux |
| `tools/pth_to_onnx.py` | Wandelt Real-ESRGAN-Gewichte ohne PyTorch in ONNX um |

### Wie die Gesichtsrestaurierung arbeitet

1. YuNet sucht Gesichter und fünf Merkmalspunkte (Augen, Nase, Mundwinkel) im Originalbild.
2. Jedes Gesicht wird über diese Punkte auf eine Standardposition (512 × 512) gedreht und skaliert.
3. GFPGAN rekonstruiert Augen, Haut, Zähne und Haare.
4. Das Ergebnis wird mit weich auslaufendem Rand zurück ins vergrößerte Bild geblendet.

Gesichter, die im Ergebnis schon deutlich größer als 512 Pixel sind, werden automatisch schwächer behandelt – dort würde GFPGAN eher Details verlieren als gewinnen. Bei sehr großen Bildern mit winzigen Gesichtern kann die Erkennung einzelne Gesichter übersehen, weil sie auf einer 640-Pixel-Version des Bildes läuft.

Eigene Modelle: Jede `.onnx`-Datei in `models/` mit Eingang `[1,3,H,W]` (RGB, 0..1) und Ausgang `[1,3,H·s,W·s]` wird erkannt.

## Lizenzen

- Real-ESRGAN-Modelle: © Xintao Wang et al., BSD-3-Clause – <https://github.com/xinntao/Real-ESRGAN>
- GFPGAN 1.4: © Tencent, Apache-2.0 **mit Ausnahmen**: Die Lizenzdatei nennt Drittkomponenten unter nicht-kommerziellen Lizenzen (NVIDIA StyleGAN2, DFDNet) – <https://github.com/TencentARC/GFPGAN> (ONNX-Fassung aus <https://github.com/facefusion/facefusion-assets>)
- YuNet-Gesichtsdetektor: OpenCV Zoo, MIT – <https://github.com/opencv/opencv_zoo>

Real-ESRGAN und YuNet erlauben kommerzielle Nutzung. Bei GFPGAN ist das wegen der genannten Drittkomponenten nicht eindeutig – für private Nutzung unproblematisch, vor einem kommerziellen Einsatz der Gesichtsrestaurierung bitte die Lizenzlage prüfen. Lizenzhinweise müssen beim Weitergeben mitgeliefert werden.
