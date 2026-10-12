//! Klassisches Nachschärfen (Unscharf maskieren) auf dem Helligkeitskanal.
//!
//! * Geschärft wird nur die Luminanz → keine Farbsäume.
//! * Schwellenwert: kleine Helligkeitsunterschiede (Rauschen) bleiben unangetastet.
//! * Kantenmaske: optional nur an deutlichen Kanten schärfen, glatte Flächen bleiben ruhig.
//! * Halo-Schutz: begrenzt Über-/Unterschwinger auf die lokale Helligkeitsspanne.

use image::Rgb32FImage;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharpenParams {
    /// Stärke, 0.0 – 3.0 (1.0 ≈ 100 %)
    pub amount: f32,
    /// Radius (Sigma der Gauß-Unschärfe) in Pixeln, 0.3 – 5.0
    pub radius: f32,
    /// Schwellenwert in Helligkeitseinheiten 0 – 1 (UI: 0 – 255)
    pub threshold: f32,
    /// Halo-Schutz 0.0 (aus) – 1.0 (voll)
    pub halo_protection: f32,
    /// Nur an Kanten schärfen
    pub edges_only: bool,
}

impl Default for SharpenParams {
    fn default() -> Self {
        Self { amount: 0.8, radius: 1.0, threshold: 2.0 / 255.0, halo_protection: 0.7, edges_only: false }
    }
}

const KR: f32 = 0.2126;
const KG: f32 = 0.7152;
const KB: f32 = 0.0722;

#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    if e1 <= e0 {
        return if x >= e0 { 1.0 } else { 0.0 };
    }
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let r = (sigma * 3.0).ceil().max(1.0) as i32;
    let mut k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    k
}

/// Separierbare Gauß-Unschärfe auf einem Einkanal-Puffer (Ränder werden fortgesetzt).
pub fn gaussian_blur(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let k = gaussian_kernel(sigma);
    let r = (k.len() / 2) as isize;

    // horizontal
    let mut tmp = vec![0.0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let row = &src[y * w..(y + 1) * w];
        for (x, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (i, kv) in k.iter().enumerate() {
                let xx = (x as isize + i as isize - r).clamp(0, w as isize - 1) as usize;
                acc += kv * row[xx];
            }
            *o = acc;
        }
    });

    // vertikal (zeilenweise akkumulieren → cache-freundlich)
    let mut dst = vec![0.0f32; w * h];
    dst.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        for (i, kv) in k.iter().enumerate() {
            let yy = (y as isize + i as isize - r).clamp(0, h as isize - 1) as usize;
            let row = &tmp[yy * w..(yy + 1) * w];
            for (o, v) in out.iter_mut().zip(row) {
                *o += kv * v;
            }
        }
    });
    dst
}

/// Wendet das Nachschärfen an und gibt ein neues Bild zurück.
pub fn sharpen(img: &Rgb32FImage, p: &SharpenParams) -> Rgb32FImage {
    let (w, h) = (img.width() as usize, img.height() as usize);
    if p.amount <= 0.0 || w < 3 || h < 3 {
        return img.clone();
    }
    let px = img.as_raw();

    let luma: Vec<f32> = px.par_chunks(3).map(|c| KR * c[0] + KG * c[1] + KB * c[2]).collect();
    let blur = gaussian_blur(&luma, w, h, p.radius.max(0.3));

    // Kantenmaske aus dem Gradienten der geglätteten Luminanz
    let edge_mask: Option<Vec<f32>> = p.edges_only.then(|| {
        let mut m = vec![0.0f32; w * h];
        m.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(h - 1);
            for (x, o) in row.iter_mut().enumerate() {
                let x0 = x.saturating_sub(1);
                let x1 = (x + 1).min(w - 1);
                let gx = blur[y * w + x1] - blur[y * w + x0];
                let gy = blur[y1 * w + x] - blur[y0 * w + x];
                *o = smoothstep(0.015, 0.08, (gx * gx + gy * gy).sqrt());
            }
        });
        // Maske leicht weichzeichnen, damit keine harten Übergänge entstehen
        gaussian_blur(&m, w, h, 1.5)
    });

    let thr = p.threshold;
    let mut out = img.clone();
    out.as_mut()
        .par_chunks_mut(w * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(h - 1);
            for x in 0..w {
                let i = y * w + x;
                let l = luma[i];
                let detail = l - blur[i];

                let mut weight = if thr > 0.0 { smoothstep(thr * 0.5, thr * 1.5, detail.abs()) } else { 1.0 };
                if let Some(m) = &edge_mask {
                    weight *= m[i];
                }
                let mut nl = l + p.amount * detail * weight;

                if p.halo_protection > 0.0 {
                    let x0 = x.saturating_sub(1);
                    let x1 = (x + 1).min(w - 1);
                    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
                    for yy in y0..=y1 {
                        for xx in x0..=x1 {
                            let v = luma[yy * w + xx];
                            lo = lo.min(v);
                            hi = hi.max(v);
                        }
                    }
                    let clamped = nl.clamp(lo, hi);
                    nl += (clamped - nl) * p.halo_protection;
                }

                let d = nl - l;
                let c = &mut row[x * 3..x * 3 + 3];
                c[0] += d;
                c[1] += d;
                c[2] += d;
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_image_unchanged() {
        let img = Rgb32FImage::from_pixel(32, 32, image::Rgb([0.5, 0.4, 0.3]));
        let out = sharpen(&img, &SharpenParams { amount: 2.0, ..Default::default() });
        for (a, b) in img.as_raw().iter().zip(out.as_raw()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn edge_gets_more_contrast() {
        let img = Rgb32FImage::from_fn(32, 8, |x, _| {
            let v = if x < 16 { 0.3 } else { 0.7 };
            image::Rgb([v, v, v])
        });
        let blurred = {
            let l: Vec<f32> = img.as_raw().chunks(3).map(|c| c[0]).collect();
            let b = gaussian_blur(&l, 32, 8, 1.5);
            Rgb32FImage::from_fn(32, 8, |x, y| {
                let v = b[(y * 32 + x) as usize];
                image::Rgb([v, v, v])
            })
        };
        let p = SharpenParams { amount: 1.5, halo_protection: 0.0, threshold: 0.0, ..Default::default() };
        let out = sharpen(&blurred, &p);
        let before = blurred.get_pixel(17, 4)[0] - blurred.get_pixel(14, 4)[0];
        let after = out.get_pixel(17, 4)[0] - out.get_pixel(14, 4)[0];
        assert!(after > before);
    }
}
