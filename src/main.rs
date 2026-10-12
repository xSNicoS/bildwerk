#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

mod app;
mod engine;
mod face;
mod sharpen;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};

use engine::{discover_models, enhance, finish, EnhanceSettings, Picture, Progress, Upscaler};
use sharpen::SharpenParams;

const HILFE: &str = "\
Bildwerk – KI-Bildverbesserung

Ohne Argumente startet die grafische Oberfläche.

Kommandozeile:
  bildwerk --cli -i EINGABE -o AUSGABE [Optionen]

  EINGABE/AUSGABE können Dateien oder Ordner sein (Ordner = Stapelverarbeitung).

Optionen:
  --model NAME|PFAD|none   KI-Modell (Dateiname ohne .onnx aus models/), Standard: erstes gefundenes
  --scale F                Ausgabegröße relativ zum Original (1 = nur verbessern), Standard: Modellfaktor
  --strength S             KI-Stärke 0..1, Standard 1
  --tile N                 Kachelgröße in Pixeln, Standard 256
  --sharpen A              Nachschärfen mit Stärke A (z. B. 0.8), Standard: aus
  --radius R               Schärferadius in Pixeln, Standard 1.0
  --threshold T            Schwellenwert 0..255, Standard 2
  --halo H                 Halo-Schutz 0..1, Standard 0.7
  --edges                  nur an Kanten schärfen
  --face S                 Gesichtsrestaurierung (GFPGAN) mit Stärke S 0..1, z. B. 0.7
  --list-models            verfügbare Modelle anzeigen
";

/// Unter Windows hat das Programm (ohne Konsolenfenster gebaut) keine Textausgabe. Bei Aufruf aus der
/// Eingabeaufforderung hängen wir uns an deren Konsole an, damit --cli und --help etwas anzeigen.
#[cfg(windows)]
fn attach_console() {
    #[link(name = "kernel32")]
    extern "system" {
        fn AttachConsole(pid: u32) -> i32;
    }
    unsafe {
        AttachConsole(u32::MAX); // ATTACH_PARENT_PROCESS
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    #[cfg(windows)]
    if args.iter().any(|a| a.starts_with("--") || a == "-h") {
        attach_console();
    }
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{HILFE}");
        return Ok(());
    }
    if args.iter().any(|a| a == "--list-models") {
        for m in discover_models() {
            println!("{:<30} {}", m.path.file_stem().unwrap().to_string_lossy(), m.label);
        }
        return Ok(());
    }
    if args.iter().any(|a| a == "--cli") {
        return run_cli(&args);
    }
    app::run()
}

fn run_cli(args: &[String]) -> Result<()> {
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let num = |k: &str, d: f32| -> Result<f32> {
        get(k).map(|v| v.replace(',', ".").parse::<f32>().with_context(|| format!("{k}: keine Zahl"))).unwrap_or(Ok(d))
    };
    let input = PathBuf::from(get("-i").context("-i EINGABE fehlt")?);
    let output = PathBuf::from(get("-o").context("-o AUSGABE fehlt")?);

    let models = discover_models();
    let mut upscaler = match get("--model").as_deref() {
        Some("none") => None,
        Some(name) => {
            let p = PathBuf::from(name);
            let path = if p.exists() {
                p
            } else {
                models
                    .iter()
                    .find(|m| m.path.file_stem().map(|s| s == name).unwrap_or(false))
                    .map(|m| m.path.clone())
                    .with_context(|| format!("Modell „{name}“ nicht gefunden (siehe --list-models)"))?
            };
            Some(Upscaler::load(&path)?)
        }
        None => match models.first() {
            Some(m) => Some(Upscaler::load(&m.path)?),
            None => None,
        },
    };
    if let Some(u) = &upscaler {
        eprintln!("Modell: {} (×{}, {})", u.path.display(), u.scale, u.backend);
    } else {
        eprintln!("Kein KI-Modell – nur klassische Verarbeitung");
    }

    let default_scale = upscaler.as_ref().map(|u| u.scale as f32).unwrap_or(1.0);
    let settings = EnhanceSettings {
        output_scale: num("--scale", default_scale)?,
        ai_strength: num("--strength", 1.0)?,
        tile: num("--tile", 256.0)? as u32,
    };
    let sharpen_amount = num("--sharpen", 0.0)?;
    let sp = SharpenParams {
        amount: sharpen_amount,
        radius: num("--radius", 1.0)?,
        threshold: num("--threshold", 2.0)? / 255.0,
        halo_protection: num("--halo", 0.7)?,
        edges_only: args.iter().any(|a| a == "--edges"),
    };

    let face_strength = num("--face", 0.0)?;
    let mut restorer = if face_strength > 0.0 {
        let r = face::FaceRestorer::load()?;
        eprintln!("Gesichtsrestaurierung: Stärke {:.0} %", face_strength * 100.0);
        Some(r)
    } else {
        None
    };

    let jobs: Vec<(PathBuf, PathBuf)> = if input.is_dir() {
        std::fs::create_dir_all(&output)?;
        let mut v = Vec::new();
        for e in std::fs::read_dir(&input)?.flatten() {
            let p = e.path();
            if is_image(&p) {
                let name = p.file_stem().unwrap().to_string_lossy().to_string();
                v.push((p, output.join(format!("{name}_bildwerk.png"))));
            }
        }
        v.sort();
        v
    } else {
        vec![(input, output)]
    };
    if jobs.is_empty() {
        bail!("Keine Bilder gefunden");
    }

    for (i, (src, dst)) in jobs.iter().enumerate() {
        let t0 = Instant::now();
        let pic = Picture::load(src)?;
        let progress = Progress::new();
        let (res, n_faces) =
            enhance(&pic, upscaler.as_mut(), restorer.as_mut().map(|r| (r, face_strength)), &settings, &progress)?;
        let res = finish(&res, sharpen_amount > 0.0, &sp);
        res.save(dst)?;
        let (w, h) = pic.dimensions();
        let (ow, oh) = res.dimensions();
        eprintln!(
            "[{}/{}] {} ({w}×{h}) → {} ({ow}×{oh}){} in {:.1} s",
            i + 1,
            jobs.len(),
            src.display(),
            dst.display(),
            if restorer.is_some() { format!(", {n_faces} Gesicht(er)") } else { String::new() },
            t0.elapsed().as_secs_f32()
        );
    }
    Ok(())
}

pub fn is_image(p: &Path) -> bool {
    matches!(
        p.extension().and_then(|s| s.to_str()).map(|s| s.to_lowercase()).as_deref(),
        Some("png" | "jpg" | "jpeg" | "tif" | "tiff" | "webp" | "bmp")
    )
}
