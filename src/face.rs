//! Gesichtsrestaurierung: YuNet findet Gesichter samt 5 Merkmalspunkten, jedes Gesicht wird auf die
//! FFHQ-Vorlage (512 × 512) ausgerichtet, von GFPGAN restauriert und mit weicher Maske zurückgeblendet.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use image::{imageops::FilterType, Rgb32FImage};
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;
use rayon::prelude::*;

use crate::engine::{models_dirs, new_session, Progress};

const DET_FILE: &str = "face_detection_yunet_2023mar.onnx";
const GFP_FILE: &str = "gfpgan_1.4.onnx";
const DET_SIZE: u32 = 640;
const FACE: usize = 512;

/// Lage der fünf Merkmalspunkte (Augen, Nase, Mundwinkel) im 512er-FFHQ-Ausschnitt.
const TEMPLATE: [[f32; 2]; 5] = [
    [192.981_38, 239.947_08],
    [318.902_77, 240.193_6],
    [256.634_16, 314.019_35],
    [201.261_17, 371.410_43],
    [313.089_05, 371.151_18],
];

#[derive(Clone, Debug)]
pub struct Face {
    pub score: f32,
    /// x, y, Breite, Höhe im Originalbild
    pub bbox: [f32; 4],
    pub landmarks: [[f32; 2]; 5],
}

/// Findet die Modelldateien in `models/face/`.
pub fn find_models() -> Option<(PathBuf, PathBuf)> {
    models_dirs().into_iter().map(|d| d.join("face")).find_map(|d| {
        let (a, b) = (d.join(DET_FILE), d.join(GFP_FILE));
        (a.is_file() && b.is_file()).then_some((a, b))
    })
}

pub struct FaceRestorer {
    det: Session,
    gfp: Session,
}

impl FaceRestorer {
    pub fn load() -> Result<Self> {
        let Some((det, gfp)) = find_models() else {
            bail!("Gesichtsmodelle fehlen in models/face/ ({DET_FILE}, {GFP_FILE})");
        };
        Self::load_from(&det, &gfp)
    }

    pub fn load_from(det: &Path, gfp: &Path) -> Result<Self> {
        Ok(Self { det: new_session(det)?, gfp: new_session(gfp)? })
    }

    // -------------------------------------------------------------- Erkennung

    pub fn detect(&mut self, img: &Rgb32FImage, min_score: f32) -> Result<Vec<Face>> {
        let (w, h) = img.dimensions();
        let s = DET_SIZE as f32 / w.max(h) as f32;
        let (rw, rh) = (((w as f32 * s).round() as u32).max(1), ((h as f32 * s).round() as u32).max(1));
        let small = image::imageops::resize(img, rw, rh, FilterType::Triangle);

        // YuNet erwartet BGR, 0..255, 640 × 640 (Rest mit Schwarz aufgefüllt)
        let n = DET_SIZE as usize;
        let mut input = Array4::<f32>::zeros((1, 3, n, n));
        for (x, y, p) in small.enumerate_pixels() {
            for c in 0..3 {
                input[[0, 2 - c, y as usize, x as usize]] = p[c].clamp(0.0, 1.0) * 255.0;
            }
        }
        let out = self.det.run(ort::inputs!["input" => TensorRef::from_array_view(&input)?])?;

        let mut faces = Vec::new();
        for stride in [8usize, 16, 32] {
            let cls = out[format!("cls_{stride}").as_str()].try_extract_array::<f32>()?;
            let obj = out[format!("obj_{stride}").as_str()].try_extract_array::<f32>()?;
            let bbox = out[format!("bbox_{stride}").as_str()].try_extract_array::<f32>()?;
            let kps = out[format!("kps_{stride}").as_str()].try_extract_array::<f32>()?;
            let cols = n / stride;
            let st = stride as f32;
            for i in 0..cls.shape()[1] {
                let score = (cls[[0, i, 0]].clamp(0.0, 1.0) * obj[[0, i, 0]].clamp(0.0, 1.0)).sqrt();
                if score < min_score {
                    continue;
                }
                let (r, c) = ((i / cols) as f32, (i % cols) as f32);
                let cx = (c + bbox[[0, i, 0]]) * st;
                let cy = (r + bbox[[0, i, 1]]) * st;
                let bw = bbox[[0, i, 2]].exp() * st;
                let bh = bbox[[0, i, 3]].exp() * st;
                let mut lm = [[0.0f32; 2]; 5];
                for (k, p) in lm.iter_mut().enumerate() {
                    *p = [(kps[[0, i, 2 * k]] + c) * st / s, (kps[[0, i, 2 * k + 1]] + r) * st / s];
                }
                faces.push(Face { score, bbox: [(cx - bw / 2.0) / s, (cy - bh / 2.0) / s, bw / s, bh / s], landmarks: lm });
            }
        }
        Ok(nms(faces, 0.3))
    }

    // -------------------------------------------------------------- Restaurierung

    /// Restauriert alle Gesichter in `target`. `landmark_scale` rechnet Koordinaten aus dem Bild, auf dem
    /// erkannt wurde, in `target` um (z. B. 4.0 nach ×4-Vergrößerung).
    pub fn restore(
        &mut self,
        target: &mut Rgb32FImage,
        faces: &[Face],
        landmark_scale: f32,
        strength: f32,
        progress: &Progress,
    ) -> Result<()> {
        for face in faces {
            if progress.cancelled() {
                bail!("Abgebrochen");
            }
            let pts: Vec<[f32; 2]> =
                face.landmarks.iter().map(|p| [p[0] * landmark_scale, p[1] * landmark_scale]).collect();
            let m = similarity(&pts, &TEMPLATE); // Bild → Vorlage
            let inv = invert(&m); // Vorlage → Bild

            let crop = warp_to_template(target, &inv);
            let restored = self.run_gfpgan(&crop)?;

            // Bereits große, scharfe Gesichter weniger stark behandeln (GFPGAN arbeitet mit 512 px).
            let template_per_px = (m[0] * m[0] + m[3] * m[3]).sqrt();
            let s = strength * smoothstep(0.25, 0.6, template_per_px);
            if s > 0.0 {
                paste_back(target, &restored, &m, &inv, s);
            }
            progress.inc();
        }
        Ok(())
    }

    fn run_gfpgan(&mut self, crop: &Rgb32FImage) -> Result<Rgb32FImage> {
        let mut input = Array4::<f32>::zeros((1, 3, FACE, FACE));
        for (x, y, p) in crop.enumerate_pixels() {
            for c in 0..3 {
                input[[0, c, y as usize, x as usize]] = p[c].clamp(0.0, 1.0) * 2.0 - 1.0;
            }
        }
        let out = self.gfp.run(ort::inputs![TensorRef::from_array_view(&input)?])?;
        let arr = out[0].try_extract_array::<f32>()?;
        Ok(Rgb32FImage::from_fn(FACE as u32, FACE as u32, |x, y| {
            let v = |c: usize| ((arr[[0, c, y as usize, x as usize]] + 1.0) * 0.5).clamp(0.0, 1.0);
            image::Rgb([v(0), v(1), v(2)])
        }))
    }
}

// ------------------------------------------------------------------ Geometrie

/// Affine 2×3-Matrix [a, b, tx, c, d, ty]: x' = a·x + b·y + tx, y' = c·x + d·y + ty
type Affine = [f32; 6];

/// Ähnlichkeitstransformation (Drehung, Skalierung, Verschiebung) nach kleinsten Quadraten.
fn similarity(src: &[[f32; 2]], dst: &[[f32; 2]]) -> Affine {
    let n = src.len() as f32;
    let mean = |p: &[[f32; 2]]| {
        let (sx, sy) = p.iter().fold((0.0, 0.0), |a, q| (a.0 + q[0], a.1 + q[1]));
        (sx / n, sy / n)
    };
    let (msx, msy) = mean(src);
    let (mdx, mdy) = mean(dst);
    let (mut sxx, mut sxy, mut ss) = (0.0f32, 0.0f32, 0.0f32);
    for (s, d) in src.iter().zip(dst) {
        let (ux, uy) = (s[0] - msx, s[1] - msy);
        let (vx, vy) = (d[0] - mdx, d[1] - mdy);
        sxx += ux * vx + uy * vy;
        sxy += ux * vy - uy * vx;
        ss += ux * ux + uy * uy;
    }
    let a = sxx / ss;
    let b = sxy / ss;
    // [a -b; b a]
    [a, -b, mdx - (a * msx - b * msy), b, a, mdy - (b * msx + a * msy)]
}

fn invert(m: &Affine) -> Affine {
    let det = m[0] * m[4] - m[1] * m[3];
    let (a, b, c, d) = (m[4] / det, -m[1] / det, -m[3] / det, m[0] / det);
    [a, b, -(a * m[2] + b * m[5]), c, d, -(c * m[2] + d * m[5])]
}

#[inline]
fn apply(m: &Affine, x: f32, y: f32) -> (f32, f32) {
    (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])
}

#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn bilinear(img: &Rgb32FImage, x: f32, y: f32) -> [f32; 3] {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (x.floor() as i32, y.floor() as i32);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let p = |xx: i32, yy: i32| img.get_pixel(xx as u32, yy as u32).0;
    let (a, b, c, d) = (p(x0, y0), p(x1, y0), p(x0, y1), p(x1, y1));
    let mut o = [0.0; 3];
    for k in 0..3 {
        let top = a[k] + (b[k] - a[k]) * fx;
        let bot = c[k] + (d[k] - c[k]) * fx;
        o[k] = top + (bot - top) * fy;
    }
    o
}

/// Schneidet das Gesicht ausgerichtet aus (Vorlagekoordinaten → Bildkoordinaten über `inv`).
fn warp_to_template(img: &Rgb32FImage, inv: &Affine) -> Rgb32FImage {
    let mut out = Rgb32FImage::new(FACE as u32, FACE as u32);
    out.as_mut().par_chunks_mut(FACE * 3).enumerate().for_each(|(y, row)| {
        for x in 0..FACE {
            let (sx, sy) = apply(inv, x as f32, y as f32);
            row[x * 3..x * 3 + 3].copy_from_slice(&bilinear(img, sx, sy));
        }
    });
    out
}

/// Blendet das restaurierte Gesicht mit weich auslaufender Maske zurück.
fn paste_back(target: &mut Rgb32FImage, face: &Rgb32FImage, m: &Affine, inv: &Affine, strength: f32) {
    let (w, h) = target.dimensions();
    let f = FACE as f32 - 1.0;
    // Begrenzungsrechteck des Ausschnitts im Zielbild
    let corners = [(0.0, 0.0), (f, 0.0), (0.0, f), (f, f)].map(|(x, y)| apply(inv, x, y));
    let x0 = corners.iter().map(|c| c.0).fold(f32::MAX, f32::min).floor().max(0.0) as u32;
    let y0 = corners.iter().map(|c| c.1).fold(f32::MAX, f32::min).floor().max(0.0) as u32;
    let x1 = (corners.iter().map(|c| c.0).fold(f32::MIN, f32::max).ceil() as u32).min(w.saturating_sub(1));
    let y1 = (corners.iter().map(|c| c.1).fold(f32::MIN, f32::max).ceil() as u32).min(h.saturating_sub(1));
    if x0 > x1 || y0 > y1 {
        return;
    }
    let feather = 56.0; // Breite des weichen Rands in Vorlagepixeln
    let width = w as usize;
    target.as_mut().par_chunks_mut(width * 3).enumerate().for_each(|(y, row)| {
        let y = y as u32;
        if y < y0 || y > y1 {
            return;
        }
        for x in x0..=x1 {
            let (u, v) = apply(m, x as f32, y as f32);
            let edge = u.min(f - u).min(v).min(f - v);
            if edge <= 0.0 {
                continue;
            }
            let a = smoothstep(0.0, feather, edge) * strength;
            let src = bilinear(face, u, v);
            let px = &mut row[x as usize * 3..x as usize * 3 + 3];
            for k in 0..3 {
                px[k] += (src[k] - px[k]) * a;
            }
        }
    });
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let ix = (a[0] + a[2]).min(b[0] + b[2]) - a[0].max(b[0]);
    let iy = (a[1] + a[3]).min(b[1] + b[3]) - a[1].max(b[1]);
    if ix <= 0.0 || iy <= 0.0 {
        return 0.0;
    }
    let inter = ix * iy;
    inter / (a[2] * a[3] + b[2] * b[3] - inter)
}

fn nms(mut faces: Vec<Face>, thr: f32) -> Vec<Face> {
    faces.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<Face> = Vec::new();
    for f in faces {
        if keep.iter().all(|k| iou(&k.bbox, &f.bbox) < thr) {
            keep.push(f);
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_roundtrip() {
        // bekannte Transformation: Skalierung 2, Drehung 30°, Verschiebung
        let (s, t) = (2.0f32, 30f32.to_radians());
        let m: Affine = [s * t.cos(), -s * t.sin(), 10.0, s * t.sin(), s * t.cos(), -5.0];
        let dst: Vec<[f32; 2]> = TEMPLATE.iter().map(|p| { let (x, y) = apply(&m, p[0], p[1]); [x, y] }).collect();
        let est = similarity(&TEMPLATE, &dst);
        for (a, b) in m.iter().zip(est.iter()) {
            assert!((a - b).abs() < 1e-2, "{m:?} vs {est:?}");
        }
        let back = invert(&est);
        let (x, y) = apply(&back, dst[2][0], dst[2][1]);
        assert!((x - TEMPLATE[2][0]).abs() < 1e-2 && (y - TEMPLATE[2][1]).abs() < 1e-2);
    }
}
