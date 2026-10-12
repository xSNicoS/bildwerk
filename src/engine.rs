//! KI-Inferenz (ONNX Runtime) mit Kachelverarbeitung und die komplette Verarbeitungskette.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use image::{imageops::FilterType, DynamicImage, GrayImage, ImageBuffer, Luma, Rgb32FImage};
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;

use crate::face::FaceRestorer;
use crate::sharpen::{sharpen, SharpenParams};

// ------------------------------------------------------------------ Modellverzeichnis

/// Bekannte Modelle mit verständlicher Beschreibung. Unbekannte .onnx-Dateien werden trotzdem angezeigt.
const KNOWN: &[(&str, &str, &str)] = &[
    ("realesr-general-x4v3", "Allgemein – schnell (×4)", "Kleines, schnelles Modell für Fotos aller Art. Gute Wahl auf der CPU."),
    ("realesr-general-wdn-x4v3", "Allgemein + Entrauschen (×4)", "Wie „Allgemein“, entfernt aber zusätzlich Rauschen und JPEG-Artefakte."),
    ("RealESRGAN_x4plus", "Foto – hohe Qualität (×4)", "Großes Modell mit kräftiger Detailrekonstruktion. Deutlich langsamer, GPU empfohlen."),
    ("RealESRGAN_x2plus", "Foto – hohe Qualität (×2)", "Großes Modell für moderate Vergrößerung, natürlicher Look."),
    ("RealESRGAN_x4plus_anime_6B", "Anime / Grafik / KI-Bilder (×4)", "Für Zeichnungen, Anime, Logos und KI-generierte Bilder mit glatten Flächen."),
];

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub path: PathBuf,
    pub label: String,
    pub description: String,
}

/// Sucht nach `models/` neben der ausführbaren Datei und im aktuellen Verzeichnis.
pub fn models_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            dirs.push(d.join("models"));
            // bei `cargo run`: target/release/../../models
            dirs.push(d.join("../../models"));
        }
    }
    dirs.push(PathBuf::from("models"));
    dirs
}

/// Absoluter Pfad ohne das Windows-Präfix `\\?\`, das `canonicalize` dort erzeugt.
fn clean_path(p: PathBuf) -> PathBuf {
    let c = p.canonicalize().unwrap_or(p);
    let s = c.to_string_lossy();
    match s.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC") => PathBuf::from(rest),
        _ => c,
    }
}

pub fn discover_models() -> Vec<ModelInfo> {
    let mut found: Vec<ModelInfo> = Vec::new();
    for dir in models_dirs() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) != Some("onnx") {
                continue;
            }
            let stem = p.file_stem().unwrap_or_default().to_string_lossy().to_string();
            if found.iter().any(|m| m.path.file_stem() == p.file_stem()) {
                continue;
            }
            let (label, desc) = KNOWN
                .iter()
                .find(|(k, _, _)| *k == stem)
                .map(|(_, l, d)| (l.to_string(), d.to_string()))
                .unwrap_or_else(|| (stem.clone(), "Eigenes ONNX-Modell".into()));
            found.push(ModelInfo { path: clean_path(p), label, description: desc });
        }
    }
    let order = |m: &ModelInfo| {
        let s = m.path.file_stem().unwrap_or_default().to_string_lossy().to_string();
        KNOWN.iter().position(|(k, _, _)| *k == s).unwrap_or(usize::MAX)
    };
    found.sort_by_key(order);
    found
}

// ------------------------------------------------------------------ Upscaler

pub struct Upscaler {
    session: Session,
    pub scale: u32,
    pub path: PathBuf,
    pub backend: &'static str,
}

/// Name der GPU-Schnittstelle, mit der dieses Programm gebaut wurde.
fn gpu_backend() -> Option<&'static str> {
    if cfg!(feature = "cuda") {
        Some("GPU (CUDA)")
    } else if cfg!(feature = "directml") {
        Some("GPU (DirectML)")
    } else if cfg!(feature = "coreml") {
        Some("GPU/NPU (CoreML)")
    } else {
        None
    }
}

/// Bei der Windows-Installation liegt onnxruntime.dll im Programmordner. Sie muss ausdrücklich von dort
/// geladen werden – Windows 11 bringt in System32 eine eigene, ältere onnxruntime.dll mit.
#[cfg(feature = "windows-installer")]
fn init_runtime() -> Result<()> {
    static INIT: std::sync::OnceLock<std::result::Result<(), String>> = std::sync::OnceLock::new();
    INIT.get_or_init(|| {
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let dll = exe.parent().unwrap_or(Path::new(".")).join("onnxruntime.dll");
        ort::init_from(&dll)
            .map_err(|e| format!("ONNX Runtime ({}) konnte nicht geladen werden: {e}", dll.display()))?
            .with_name("bildwerk")
            .commit();
        Ok(())
    })
    .clone()
    .map_err(|e| anyhow!(e))
}

#[cfg(not(feature = "windows-installer"))]
fn init_runtime() -> Result<()> {
    Ok(())
}

fn builder() -> Result<ort::session::builder::SessionBuilder> {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    Session::builder()
        .map_err(|e| anyhow!("{e}"))?
        .with_intra_threads(threads)
        .map_err(|e| anyhow!("{e}"))
}

/// Erstellt eine ONNX-Runtime-Sitzung. Ist das Programm mit GPU-Unterstützung gebaut, wird zuerst die
/// Grafikkarte versucht; klappt das nicht (z. B. alter Treiber), läuft das Modell auf der CPU.
/// Zurückgegeben wird auch, worauf es tatsächlich läuft.
pub fn new_session_with_backend(path: &Path) -> Result<(Session, &'static str)> {
    init_runtime()?;
    if !path.is_file() {
        bail!("Modelldatei fehlt: {}", path.display());
    }

    if let Some(gpu) = gpu_backend() {
        #[allow(unused_mut)]
        let mut eps: Vec<ort::ep::ExecutionProviderDispatch> = Vec::new();
        #[cfg(feature = "cuda")]
        eps.push(ort::ep::CUDA::default().build().error_on_failure());
        #[cfg(feature = "directml")]
        eps.push(ort::ep::DirectML::default().build().error_on_failure());
        #[cfg(feature = "coreml")]
        eps.push(ort::ep::CoreML::default().build().error_on_failure());

        let attempt = || -> Result<Session> {
            let mut b = builder()?;
            if cfg!(feature = "directml") {
                // DirectML verträgt weder Speichermuster noch parallele Ausführung
                b = b.with_memory_pattern(false).map_err(|e| anyhow!("{e}"))?;
                b = b.with_parallel_execution(false).map_err(|e| anyhow!("{e}"))?;
            }
            b.with_execution_providers(eps)
                .map_err(|e| anyhow!("{e}"))?
                .commit_from_file(path)
                .map_err(|e| anyhow!("{e}"))
        };
        match attempt() {
            Ok(s) => return Ok((s, gpu)),
            Err(e) => eprintln!("{gpu} nicht verfügbar, nutze CPU: {e}"),
        }
    }

    let session = builder()?
        .commit_from_file(path)
        .map_err(|e| anyhow!("Modell {} konnte nicht geladen werden: {e}", path.display()))?;
    Ok((session, "CPU"))
}

pub fn new_session(path: &Path) -> Result<Session> {
    new_session_with_backend(path).map(|(s, _)| s)
}

impl Upscaler {
    pub fn load(path: &Path) -> Result<Self> {
        let (mut session, backend) = new_session_with_backend(path)?;
        let meta_scale = session
            .metadata()
            .ok()
            .and_then(|m| m.custom("scale"))
            .and_then(|s| s.parse::<u32>().ok());
        let scale = match meta_scale {
            Some(s) => s,
            None => {
                // Skalierungsfaktor durch einen Probelauf ermitteln
                let probe = Array4::<f32>::zeros((1, 3, 16, 16));
                let out = session.run(ort::inputs![TensorRef::from_array_view(&probe)?])?;
                let arr = out[0].try_extract_array::<f32>()?;
                (arr.shape()[2] / 16) as u32
            }
        };
        if !(1..=8).contains(&scale) {
            bail!("Unerwarteter Skalierungsfaktor {scale}");
        }
        Ok(Self { session, scale, path: path.to_path_buf(), backend })
    }

    /// Führt das Modell auf einem Bildausschnitt aus (Werte 0..1).
    fn run_tile(&mut self, tile: &Rgb32FImage) -> Result<Rgb32FImage> {
        let (w, h) = (tile.width() as usize, tile.height() as usize);
        let mut input = Array4::<f32>::zeros((1, 3, h, w));
        for (x, y, p) in tile.enumerate_pixels() {
            for c in 0..3 {
                input[[0, c, y as usize, x as usize]] = p[c];
            }
        }
        let outputs = self.session.run(ort::inputs![TensorRef::from_array_view(&input)?])?;
        let arr = outputs[0].try_extract_array::<f32>()?;
        let (oh, ow) = (arr.shape()[2], arr.shape()[3]);
        let mut out = Rgb32FImage::new(ow as u32, oh as u32);
        for (x, y, p) in out.enumerate_pixels_mut() {
            for c in 0..3 {
                p[c] = arr[[0, c, y as usize, x as usize]];
            }
        }
        Ok(out)
    }

    /// Kachelweise Verarbeitung: jede Kachel wird mit Rand (`pad`) gerechnet, nur der Kern übernommen.
    /// So bleiben keine sichtbaren Nähte, und auch riesige Bilder passen in den Speicher.
    pub fn upscale(&mut self, img: &Rgb32FImage, tile: u32, progress: &Progress) -> Result<Rgb32FImage> {
        let s = self.scale;
        let (w, h) = img.dimensions();
        let tile = tile.max(32);
        let pad = 16u32;
        let tiles_x = w.div_ceil(tile);
        let tiles_y = h.div_ceil(tile);
        progress.add_total((tiles_x * tiles_y) as usize);

        let mut out = Rgb32FImage::new(w * s, h * s);
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                if progress.cancelled() {
                    bail!("Abgebrochen");
                }
                let x0 = tx * tile;
                let y0 = ty * tile;
                let x1 = (x0 + tile).min(w);
                let y1 = (y0 + tile).min(h);
                let px0 = x0.saturating_sub(pad);
                let py0 = y0.saturating_sub(pad);
                let px1 = (x1 + pad).min(w);
                let py1 = (y1 + pad).min(h);
                // auf gerade Kantenlängen auffüllen (für ×2-Modelle mit Pixel-Unshuffle)
                let cw = px1 - px0;
                let ch = py1 - py0;
                let tw = cw + (cw % 2);
                let th = ch + (ch % 2);
                let src = Rgb32FImage::from_fn(tw, th, |x, y| *img.get_pixel(px0 + x.min(cw - 1), py0 + y.min(ch - 1)));

                let res = self.run_tile(&src)?;
                let ox = (x0 - px0) * s;
                let oy = (y0 - py0) * s;
                for y in 0..(y1 - y0) * s {
                    for x in 0..(x1 - x0) * s {
                        out.put_pixel(x0 * s + x, y0 * s + y, *res.get_pixel(ox + x, oy + y));
                    }
                }
                progress.inc();
            }
        }
        Ok(out)
    }
}

// ------------------------------------------------------------------ Fortschritt

#[derive(Default)]
pub struct Progress {
    done: AtomicUsize,
    total: AtomicUsize,
    cancel: AtomicBool,
}

impl Progress {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
    pub fn inc(&self) {
        self.done.fetch_add(1, Ordering::Relaxed);
    }
    pub fn done(&self) -> usize {
        self.done.load(Ordering::Relaxed)
    }
    pub fn add_total(&self, t: usize) {
        self.total.fetch_add(t, Ordering::Relaxed);
    }
    pub fn fraction(&self) -> f32 {
        let t = self.total.load(Ordering::Relaxed);
        if t == 0 { 0.0 } else { self.done() as f32 / t as f32 }
    }
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

// ------------------------------------------------------------------ Bilder

/// Geladenes Bild: Farbe als 32-Bit-Float (verlustfrei auch für 16-Bit-Quellen) plus optionaler Alphakanal.
#[derive(Clone)]
pub struct Picture {
    pub rgb: Rgb32FImage,
    pub alpha: Option<GrayImage>,
}

impl Picture {
    pub fn load(path: &Path) -> Result<Self> {
        let dynimg = image::ImageReader::open(path)
            .with_context(|| format!("Datei nicht lesbar: {}", path.display()))?
            .with_guessed_format()?
            .decode()
            .with_context(|| format!("Bildformat nicht unterstützt: {}", path.display()))?;
        let alpha = dynimg.color().has_alpha().then(|| {
            let rgba = dynimg.to_rgba8();
            ImageBuffer::from_fn(rgba.width(), rgba.height(), |x, y| Luma([rgba.get_pixel(x, y)[3]]))
        });
        Ok(Self { rgb: dynimg.to_rgb32f(), alpha })
    }

    pub fn dimensions(&self) -> (u32, u32) {
        self.rgb.dimensions()
    }

    /// Speichert je nach Dateiendung: PNG/TIFF in 16 Bit, JPEG mit Qualität 95, WebP verlustfrei.
    pub fn save(&self, path: &Path) -> Result<()> {
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("png").to_lowercase();
        let (w, h) = self.rgb.dimensions();
        let clamp8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        let clamp16 = |v: f32| (v.clamp(0.0, 1.0) * 65535.0).round() as u16;

        match ext.as_str() {
            "jpg" | "jpeg" => {
                let rgb8 = image::RgbImage::from_fn(w, h, |x, y| {
                    let p = self.rgb.get_pixel(x, y);
                    image::Rgb([clamp8(p[0]), clamp8(p[1]), clamp8(p[2])])
                });
                let f = std::io::BufWriter::new(std::fs::File::create(path)?);
                image::codecs::jpeg::JpegEncoder::new_with_quality(f, 95).encode_image(&rgb8)?;
            }
            "png" | "tif" | "tiff" => {
                let img: DynamicImage = match &self.alpha {
                    Some(a) => DynamicImage::ImageRgba16(ImageBuffer::from_fn(w, h, |x, y| {
                        let p = self.rgb.get_pixel(x, y);
                        let al = a.get_pixel(x, y)[0] as u16 * 257;
                        image::Rgba([clamp16(p[0]), clamp16(p[1]), clamp16(p[2]), al])
                    })),
                    None => DynamicImage::ImageRgb16(ImageBuffer::from_fn(w, h, |x, y| {
                        let p = self.rgb.get_pixel(x, y);
                        image::Rgb([clamp16(p[0]), clamp16(p[1]), clamp16(p[2])])
                    })),
                };
                img.save(path)?;
            }
            _ => {
                let img: DynamicImage = match &self.alpha {
                    Some(a) => DynamicImage::ImageRgba8(ImageBuffer::from_fn(w, h, |x, y| {
                        let p = self.rgb.get_pixel(x, y);
                        image::Rgba([clamp8(p[0]), clamp8(p[1]), clamp8(p[2]), a.get_pixel(x, y)[0]])
                    })),
                    None => DynamicImage::ImageRgb8(ImageBuffer::from_fn(w, h, |x, y| {
                        let p = self.rgb.get_pixel(x, y);
                        image::Rgb([clamp8(p[0]), clamp8(p[1]), clamp8(p[2])])
                    })),
                };
                img.save(path)?;
            }
        }
        Ok(())
    }
}

fn resize_rgb(img: &Rgb32FImage, w: u32, h: u32) -> Rgb32FImage {
    if img.dimensions() == (w, h) {
        img.clone()
    } else {
        image::imageops::resize(img, w, h, FilterType::Lanczos3)
    }
}

// ------------------------------------------------------------------ Verarbeitungskette

#[derive(Clone, Debug, PartialEq)]
pub struct EnhanceSettings {
    /// Ausgabegröße relativ zum Original (1.0 = Originalgröße, nur verbessern)
    pub output_scale: f32,
    /// Mischung KI-Ergebnis ↔ klassisch vergrößertes Original, 0..1
    pub ai_strength: f32,
    /// Kachelgröße in Eingabepixeln (kleiner = weniger Speicher)
    pub tile: u32,
}

impl Default for EnhanceSettings {
    fn default() -> Self {
        Self { output_scale: 4.0, ai_strength: 1.0, tile: 256 }
    }
}

/// Schritt 1 (teuer): KI-Vergrößerung bzw. klassische Vergrößerung ohne Modell.
pub fn enhance(
    pic: &Picture,
    model: Option<&mut Upscaler>,
    faces: Option<(&mut FaceRestorer, f32)>,
    st: &EnhanceSettings,
    progress: &Progress,
) -> Result<(Picture, usize)> {
    let (w, h) = pic.dimensions();
    let tw = ((w as f32 * st.output_scale).round() as u32).max(1);
    let th = ((h as f32 * st.output_scale).round() as u32).max(1);
    let classic = resize_rgb(&pic.rgb, tw, th);

    // Gesichter vorab auf dem (kleineren) Original suchen – schnell, und der Fortschritt stimmt von Anfang an
    let face_job = match faces {
        Some((fr, strength)) => {
            let found = fr.detect(&pic.rgb, 0.6)?;
            progress.add_total(found.len());
            Some((fr, strength, found))
        }
        None => None,
    };

    let mut rgb = match model {
        None => classic,
        Some(m) => {
            let ai = m.upscale(&pic.rgb, st.tile, progress)?;
            let ai = resize_rgb(&ai, tw, th);
            let s = st.ai_strength.clamp(0.0, 1.0);
            if s >= 0.999 {
                ai
            } else {
                let mut mixed = ai;
                for (o, c) in mixed.as_mut().iter_mut().zip(classic.as_raw()) {
                    *o = *o * s + *c * (1.0 - s);
                }
                mixed
            }
        }
    };

    // Gesichter im Ergebnis restaurieren
    let mut n_faces = 0;
    if let Some((fr, strength, found)) = face_job {
        n_faces = found.len();
        fr.restore(&mut rgb, &found, tw as f32 / w as f32, strength, progress)?;
    }

    let alpha = pic.alpha.as_ref().map(|a| image::imageops::resize(a, tw, th, FilterType::Lanczos3));
    Ok((Picture { rgb, alpha }, n_faces))
}

/// Schritt 2 (schnell): Nachschärfen auf dem Ergebnis.
pub fn finish(pic: &Picture, sharpen_on: bool, sp: &SharpenParams) -> Picture {
    if !sharpen_on {
        return pic.clone();
    }
    Picture { rgb: sharpen(&pic.rgb, sp), alpha: pic.alpha.clone() }
}
