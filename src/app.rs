//! Grafische Oberfläche (egui/eframe).

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use eframe::egui::{
    self, Align, Align2, Color32, ColorImage, CornerRadius, FontId, Layout, Pos2, Rect, RichText, Sense, Stroke,
    TextureFilter, TextureHandle, TextureOptions, Vec2,
};

use crate::face::FaceRestorer;
use crate::engine::{discover_models, enhance, finish, EnhanceSettings, ModelInfo, Picture, Progress, Upscaler};
use crate::sharpen::SharpenParams;

// ------------------------------------------------------------------ Farben

const BG: Color32 = Color32::from_rgb(0x13, 0x14, 0x15);
const PANEL: Color32 = Color32::from_rgb(0x1b, 0x1c, 0x1e);
const LINE: Color32 = Color32::from_rgb(0x2b, 0x2d, 0x30);
const TEXT: Color32 = Color32::from_rgb(0xdc, 0xdb, 0xd7);
const DIM: Color32 = Color32::from_rgb(0x8b, 0x8d, 0x91);
const ACCENT: Color32 = Color32::from_rgb(0xe9, 0xa4, 0x3c);
const ACCENT_DARK: Color32 = Color32::from_rgb(0x2a, 0x1d, 0x08);

pub fn run() -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Bildwerk")
            .with_inner_size([1360.0, 860.0])
            .with_min_inner_size([900.0, 560.0])
            .with_drag_and_drop(true)
            .with_icon(eframe::icon_data::from_png_bytes(include_bytes!("../assets/icon.png")).unwrap_or_default()),
        ..Default::default()
    };
    eframe::run_native("Bildwerk", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
        .map_err(|e| anyhow::anyhow!("{e}"))
}

// ------------------------------------------------------------------ Hintergrundaufgaben

enum Msg {
    Loaded(PathBuf, Arc<Picture>),
    Enhanced { pic: Arc<Picture>, secs: f32, info: String },
    Sharpened { gen: u64, pic: Arc<Picture> },
    Saved(PathBuf),
    Error(String),
}

#[derive(Clone, Copy, PartialEq)]
enum View {
    Compare,
    Before,
    After,
}

struct App {
    tx: Sender<Msg>,
    rx: Receiver<Msg>,

    models: Vec<ModelInfo>,
    model_idx: Option<usize>, // None = ohne KI
    model_scale: u32,
    upscaler: Arc<Mutex<Option<Upscaler>>>,
    face_restorer: Arc<Mutex<Option<FaceRestorer>>>,
    face_available: bool,
    face_on: bool,
    face_strength: f32,

    settings: EnhanceSettings,
    sharpen_on: bool,
    sharpen: SharpenParams,

    path: Option<PathBuf>,
    original: Option<Arc<Picture>>,
    enhanced: Option<Arc<Picture>>,
    result: Option<Arc<Picture>>,
    tex_before: Option<TextureHandle>,
    tex_after: Option<TextureHandle>,

    job: Option<Arc<Progress>>,
    job_started: Option<Instant>,
    sharpen_gen: u64,
    sharpen_applied: Option<(bool, SharpenParams, u64)>, // (an, Parameter, Basis-Version)
    sharpen_busy: bool,
    base_version: u64,
    saving: bool,

    status: String,
    error: Option<String>,

    view: View,
    split: f32,
    zoom: Option<f32>, // None = einpassen
    pan: Vec2,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        setup_style(&cc.egui_ctx);
        let (tx, rx) = channel();
        let models = discover_models();
        let model_idx = if models.is_empty() { None } else { Some(0) };
        let mut app = Self {
            tx,
            rx,
            models,
            model_idx,
            model_scale: 4,
            upscaler: Arc::new(Mutex::new(None)),
            face_restorer: Arc::new(Mutex::new(None)),
            face_available: crate::face::find_models().is_some(),
            face_on: false,
            face_strength: 0.7,
            settings: EnhanceSettings::default(),
            sharpen_on: true,
            sharpen: SharpenParams::default(),
            path: None,
            original: None,
            enhanced: None,
            result: None,
            tex_before: None,
            tex_after: None,
            job: None,
            job_started: None,
            sharpen_gen: 0,
            sharpen_applied: None,
            sharpen_busy: false,
            base_version: 0,
            saving: false,
            status: String::new(),
            error: None,
            view: View::Compare,
            split: 0.5,
            zoom: None,
            pan: Vec2::ZERO,
        };
        app.update_model_scale();
        if app.models.is_empty() {
            app.status = "Keine KI-Modelle im Ordner „models“ gefunden – nur Nachschärfen verfügbar.".into();
        }
        // Bild direkt als Argument übergeben (z. B. „Öffnen mit …“)
        if let Some(p) = std::env::args().nth(1).map(PathBuf::from).filter(|p| p.is_file()) {
            app.open(p);
        }
        app
    }

    fn update_model_scale(&mut self) {
        self.model_scale = match self.model_idx.map(|i| &self.models[i]) {
            Some(m) => {
                let n = m.path.file_stem().unwrap_or_default().to_string_lossy();
                if n.contains("x2") { 2 } else { 4 }
            }
            None => 4,
        };
        if self.settings.output_scale > self.model_scale as f32 {
            self.settings.output_scale = self.model_scale as f32;
        }
    }

    fn open(&mut self, path: PathBuf) {
        let tx = self.tx.clone();
        self.status = format!("Lade {} …", path.display());
        self.error = None;
        std::thread::spawn(move || match Picture::load(&path) {
            Ok(p) => drop(tx.send(Msg::Loaded(path, Arc::new(p)))),
            Err(e) => drop(tx.send(Msg::Error(format!("{e:#}")))),
        });
    }

    fn open_dialog(&mut self) {
        if let Some(p) = rfd::FileDialog::new()
            .add_filter("Bilder", &["png", "jpg", "jpeg", "tif", "tiff", "webp", "bmp"])
            .pick_file()
        {
            self.open(p);
        }
    }

    fn save_dialog(&mut self) {
        let Some(res) = self.result.clone() else { return };
        let stem = self
            .path
            .as_ref()
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "bild".into());
        let mut dlg = rfd::FileDialog::new()
            .set_file_name(format!("{stem}_bildwerk.png"))
            .add_filter("PNG (16 Bit)", &["png"])
            .add_filter("JPEG", &["jpg", "jpeg"])
            .add_filter("TIFF (16 Bit)", &["tif", "tiff"])
            .add_filter("WebP", &["webp"]);
        if let Some(dir) = self.path.as_ref().and_then(|p| p.parent()) {
            dlg = dlg.set_directory(dir);
        }
        if let Some(dst) = dlg.save_file() {
            let tx = self.tx.clone();
            self.saving = true;
            self.status = "Speichere …".into();
            std::thread::spawn(move || match res.save(&dst) {
                Ok(()) => drop(tx.send(Msg::Saved(dst))),
                Err(e) => drop(tx.send(Msg::Error(format!("Speichern fehlgeschlagen: {e:#}")))),
            });
        }
    }

    fn start_enhance(&mut self, ctx: &egui::Context) {
        let Some(orig) = self.original.clone() else { return };
        let progress = Progress::new();
        self.job = Some(progress.clone());
        self.job_started = Some(Instant::now());
        self.error = None;
        self.status = "Verarbeite …".into();

        let model_path = self.model_idx.map(|i| self.models[i].path.clone());
        let settings = self.settings.clone();
        let cache = self.upscaler.clone();
        let face_cache = self.face_restorer.clone();
        let face_strength = (self.face_on && self.face_available).then_some(self.face_strength);
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let t0 = Instant::now();
            let run = || -> anyhow::Result<(Picture, String)> {
                let mut up_guard = cache.lock().unwrap();
                let mut info = match &model_path {
                    None => {
                        *up_guard = None;
                        "klassisch (Lanczos)".to_string()
                    }
                    Some(path) => {
                        if up_guard.as_ref().map(|u| &u.path != path).unwrap_or(true) {
                            *up_guard = None;
                            *up_guard = Some(Upscaler::load(path)?);
                        }
                        let up = up_guard.as_ref().unwrap();
                        format!("×{} auf {}", up.scale, up.backend)
                    }
                };
                let up = if model_path.is_some() { up_guard.as_mut() } else { None };

                let mut face_guard = face_cache.lock().unwrap();
                if face_strength.is_some() && face_guard.is_none() {
                    *face_guard = Some(FaceRestorer::load()?);
                }
                let faces = match face_strength {
                    Some(s) => face_guard.as_mut().map(|f| (f, s)),
                    None => None,
                };
                let (pic, n) = enhance(&orig, up, faces, &settings, &progress)?;
                if face_strength.is_some() {
                    info += &match n {
                        0 => " · kein Gesicht gefunden".to_string(),
                        1 => " · 1 Gesicht".to_string(),
                        n => format!(" · {n} Gesichter"),
                    };
                }
                Ok((pic, info))
            };
            let msg = match run() {
                Ok((pic, info)) => Msg::Enhanced { pic: Arc::new(pic), secs: t0.elapsed().as_secs_f32(), info },
                Err(e) => Msg::Error(format!("{e:#}")),
            };
            let _ = tx.send(msg);
            ctx.request_repaint();
        });
    }

    /// Nachschärfen läuft live im Hintergrund, sobald sich ein Regler ändert.
    fn maybe_resharpen(&mut self, ctx: &egui::Context) {
        let Some(base) = self.enhanced.clone().or_else(|| self.original.clone()) else { return };
        let want = (self.sharpen_on, self.sharpen, self.base_version);
        if self.sharpen_applied.map(|a| a == want).unwrap_or(false) || self.sharpen_busy {
            return;
        }
        self.sharpen_applied = Some(want);
        self.sharpen_busy = true;
        self.sharpen_gen += 1;
        let gen = self.sharpen_gen;
        let (on, sp) = (self.sharpen_on, self.sharpen);
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let pic = Arc::new(finish(&base, on, &sp));
            let _ = tx.send(Msg::Sharpened { gen, pic });
            ctx.request_repaint();
        });
    }

    fn handle_messages(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::Loaded(path, pic) => {
                    let (w, h) = pic.dimensions();
                    self.status = format!("{w} × {h} px geladen");
                    self.tex_before = Some(make_texture(ctx, "vorher", &pic));
                    self.tex_after = None;
                    self.path = Some(path);
                    self.original = Some(pic);
                    self.enhanced = None;
                    self.result = None;
                    self.base_version += 1;
                    self.sharpen_applied = None;
                    self.zoom = None;
                    self.pan = Vec2::ZERO;
                }
                Msg::Enhanced { pic, secs, info } => {
                    let (w, h) = pic.dimensions();
                    self.status = format!("Fertig in {secs:.1} s · {info} · {w} × {h} px");
                    self.enhanced = Some(pic);
                    self.base_version += 1;
                    self.sharpen_applied = None;
                    self.job = None;
                }
                Msg::Sharpened { gen, pic } => {
                    self.sharpen_busy = false;
                    if gen == self.sharpen_gen {
                        self.tex_after = Some(make_texture(ctx, "nachher", &pic));
                        self.result = Some(pic);
                    }
                }
                Msg::Saved(p) => {
                    self.saving = false;
                    self.status = format!("Gespeichert: {}", p.display());
                }
                Msg::Error(e) => {
                    self.job = None;
                    self.saving = false;
                    self.status.clear();
                    self.error = Some(e);
                }
            }
        }
    }

    // -------------------------------------------------------------- Seitenleiste

    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        let busy = self.job.is_some();

        section(ui, "KI-Verbesserung");
        let current = self.model_idx.map(|i| self.models[i].label.clone()).unwrap_or_else(|| "Ohne KI (klassisch)".into());
        let mut changed = false;
        ui.add_enabled_ui(!busy, |ui| {
            egui::ComboBox::from_id_salt("model")
                .width(ui.available_width())
                .selected_text(current)
                .show_ui(ui, |ui| {
                    for i in 0..self.models.len() {
                        let l = self.models[i].label.clone();
                        changed |= ui.selectable_value(&mut self.model_idx, Some(i), l).changed();
                    }
                    changed |= ui.selectable_value(&mut self.model_idx, None, "Ohne KI (klassisch)").changed();
                });
        });
        if changed {
            self.update_model_scale();
        }
        let desc = match self.model_idx {
            Some(i) => self.models[i].description.clone(),
            None => "Vergrößert mit Lanczos-Filter. Sinnvoll in Kombination mit Nachschärfen.".into(),
        };
        ui.add_space(4.0);
        ui.label(RichText::new(desc).color(DIM).size(12.0));
        ui.add_space(12.0);

        ui.label("Ausgabegröße");
        ui.add_space(2.0);
        ui.add_enabled_ui(!busy, |ui| {
            ui.horizontal(|ui| {
                let max = self.model_scale.max(4);
                for f in 1..=max {
                    let sel = (self.settings.output_scale - f as f32).abs() < 0.01;
                    let allowed = self.model_idx.is_none() || f <= self.model_scale;
                    let txt = if f == 1 { "1× ".to_string() } else { format!("{f}×") };
                    let b = egui::Button::selectable(sel, RichText::new(txt).size(13.0)).min_size(Vec2::new(44.0, 26.0));
                    if ui.add_enabled(allowed, b).clicked() {
                        self.settings.output_scale = f as f32;
                    }
                }
            });
        });
        if let Some(o) = &self.original {
            let (w, h) = o.dimensions();
            let s = self.settings.output_scale;
            let txt = if s == 1.0 {
                format!("{w} × {h} px – nur verbessern, Größe bleibt")
            } else {
                format!("{} × {} px", (w as f32 * s).round(), (h as f32 * s).round())
            };
            ui.label(RichText::new(txt).color(DIM).size(12.0));
        }
        ui.add_space(10.0);

        ui.add_enabled_ui(!busy && self.model_idx.is_some(), |ui| {
            let mut pct = self.settings.ai_strength * 100.0;
            if param(ui, "KI-Stärke", &mut pct, 0.0..=100.0, "%", 0) {
                self.settings.ai_strength = pct / 100.0;
            }
        });

        ui.add_space(4.0);
        egui::CollapsingHeader::new(RichText::new("Erweitert").color(DIM).size(12.5))
            .default_open(false)
            .show(ui, |ui| {
                ui.add_enabled_ui(!busy, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Kachelgröße");
                        egui::ComboBox::from_id_salt("tile")
                            .selected_text(format!("{} px", self.settings.tile))
                            .show_ui(ui, |ui| {
                                for t in [128u32, 192, 256, 384, 512, 768, 1024] {
                                    ui.selectable_value(&mut self.settings.tile, t, format!("{t} px"));
                                }
                            });
                    });
                    ui.label(
                        RichText::new("Kleiner = weniger Grafikspeicher, größer = schneller.").color(DIM).size(11.5),
                    );
                });
            });

        ui.add_space(14.0);
        ui.horizontal(|ui| {
            section_inline(ui, "Gesichter");
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.add_enabled_ui(self.face_available && !busy, |ui| toggle(ui, &mut self.face_on));
            });
        });
        ui.add_space(6.0);
        if self.face_available {
            ui.add_enabled_ui(self.face_on && !busy, |ui| {
                let mut pct = self.face_strength * 100.0;
                if param(ui, "Stärke", &mut pct, 0.0..=100.0, "%", 0) {
                    self.face_strength = pct / 100.0;
                }
            });
            ui.label(
                RichText::new("Restauriert unscharfe, kleine Gesichter mit GFPGAN. Niedrigere Stärke wirkt natürlicher.")
                    .color(DIM)
                    .size(11.5),
            );
        } else {
            ui.label(
                RichText::new("Modelle fehlen. Mit „python tools/modelle_laden.py“ herunterladen.").color(DIM).size(11.5),
            );
        }

        ui.add_space(14.0);
        ui.horizontal(|ui| {
            section_inline(ui, "Nachschärfen");
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                toggle(ui, &mut self.sharpen_on);
            });
        });
        ui.add_space(6.0);
        ui.add_enabled_ui(self.sharpen_on, |ui| {
            let sp = &mut self.sharpen;
            let mut amount = sp.amount * 100.0;
            if param(ui, "Stärke", &mut amount, 0.0..=300.0, "%", 0) {
                sp.amount = amount / 100.0;
            }
            param(ui, "Radius", &mut sp.radius, 0.3..=5.0, " px", 1);
            let mut thr = sp.threshold * 255.0;
            if param(ui, "Schwellenwert", &mut thr, 0.0..=40.0, "", 0) {
                sp.threshold = thr / 255.0;
            }
            let mut halo = sp.halo_protection * 100.0;
            if param(ui, "Halo-Schutz", &mut halo, 0.0..=100.0, "%", 0) {
                sp.halo_protection = halo / 100.0;
            }
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.checkbox(&mut sp.edges_only, "Nur Kanten schärfen");
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.small_button("Zurücksetzen").clicked() {
                        *sp = SharpenParams::default();
                    }
                });
            });
            ui.label(
                RichText::new("Wirkt live auf das Ergebnis. Schwellenwert schützt glatte Flächen vor Rauschen.")
                    .color(DIM)
                    .size(11.5),
            );
        });

    }

    fn actions_ui(&mut self, ui: &mut egui::Ui) {
        ui.add_space(10.0);
        let has_img = self.original.is_some();
        let full = ui.available_width();
        if let Some(job) = self.job.clone() {
            let f = job.fraction();
            let el = self.job_started.map(|t| t.elapsed().as_secs_f32()).unwrap_or(0.0);
            let eta = if f > 0.02 { format!(" · noch ca. {:.0} s", el / f - el) } else { String::new() };
            ui.add(
                egui::ProgressBar::new(f)
                    .desired_height(40.0)
                    .fill(ACCENT)
                    .text(RichText::new(format!("{:.0} %{eta}", f * 100.0)).color(TEXT)),
            );
            ui.add_space(6.0);
            if ui.add_sized([full, 32.0], egui::Button::new("Abbrechen")).clicked() {
                job.cancel();
            }
        } else {
            let go = egui::Button::new(RichText::new("Verbessern").size(15.0).strong().color(ACCENT_DARK))
                .fill(ACCENT)
                .min_size(Vec2::new(full, 40.0));
            if ui.add_enabled(has_img, go).clicked() {
                self.start_enhance(ui.ctx());
            }
            ui.add_space(6.0);
            let save = egui::Button::new(RichText::new("Speichern unter …").size(14.0)).min_size(Vec2::new(full, 32.0));
            if ui.add_enabled(self.result.is_some() && !self.saving, save).clicked() {
                self.save_dialog();
            }
        }
        ui.add_space(8.0);
        if let Some(e) = &self.error {
            ui.label(RichText::new(e).color(Color32::from_rgb(0xf0, 0x7a, 0x6a)).size(12.0));
        } else if !self.status.is_empty() {
            ui.label(RichText::new(&self.status).color(DIM).size(12.0));
        } else {
            ui.label(RichText::new(" ").size(12.0));
        }
    }

    // -------------------------------------------------------------- Bildansicht

    fn viewer(&mut self, ui: &mut egui::Ui) {
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, BG);

        let Some(before) = self.tex_before.clone() else {
            empty_state(&painter, rect);
            if resp.clicked() {
                self.open_dialog();
            }
            return;
        };
        let after = self.tex_after.clone().unwrap_or_else(|| before.clone());

        // logische Bildgröße = Ergebnisgröße
        let dims = self
            .result
            .as_ref()
            .or(self.original.as_ref())
            .map(|p| p.dimensions())
            .unwrap();
        let img_size = Vec2::new(dims.0 as f32, dims.1 as f32);
        let ppp = ui.ctx().pixels_per_point();
        let fit = ((rect.width() - 32.0) / img_size.x).min((rect.height() - 32.0) / img_size.y).min(1.0 / ppp * 8.0);
        let mut zoom = self.zoom.unwrap_or(fit);

        // Zoomen mit Mausrad (um den Mauszeiger herum)
        if resp.hovered() {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = pinch * (scroll / 400.0).exp();
            if (factor - 1.0).abs() > 1e-4 {
                if let Some(mp) = resp.hover_pos() {
                    let new_zoom = (zoom * factor).clamp(0.02, 32.0);
                    let c = rect.center() + self.pan;
                    self.pan += (mp - c) * (1.0 - new_zoom / zoom);
                    zoom = new_zoom;
                    self.zoom = Some(zoom);
                }
            }
        }

        let size = img_size * zoom;
        let mut img_rect = Rect::from_center_size(rect.center() + self.pan, size);

        // Teiler ziehen oder Bild verschieben
        let split_x = img_rect.left() + self.split * img_rect.width();
        let near_split = self.view == View::Compare
            && resp.hover_pos().map(|p| (p.x - split_x).abs() < 14.0).unwrap_or(false);
        if resp.drag_started() {
            // Entscheidung anhand der Position beim Drücken, nicht der aktuellen
            let origin = ui.input(|i| i.pointer.press_origin());
            let on_split = self.view == View::Compare
                && origin.map(|p| (p.x - split_x).abs() < 16.0).unwrap_or(false);
            ui.memory_mut(|m| m.data.insert_temp(egui::Id::new("split_drag"), on_split));
        }
        let dragging_split = ui.memory(|m| m.data.get_temp::<bool>(egui::Id::new("split_drag")).unwrap_or(false));
        if resp.dragged() {
            if dragging_split {
                if let Some(p) = resp.interact_pointer_pos() {
                    self.split = ((p.x - img_rect.left()) / img_rect.width()).clamp(0.0, 1.0);
                }
            } else {
                self.pan += resp.drag_delta();
                if self.zoom.is_none() {
                    self.zoom = Some(zoom);
                }
                img_rect = img_rect.translate(resp.drag_delta());
            }
        }
        if resp.double_clicked() {
            self.zoom = None;
            self.pan = Vec2::ZERO;
        }
        if near_split || dragging_split {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
        }

        let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        let split_x = img_rect.left() + self.split * img_rect.width();
        match self.view {
            View::Before => {
                painter.image(before.id(), img_rect, uv, Color32::WHITE);
            }
            View::After => {
                painter.image(after.id(), img_rect, uv, Color32::WHITE);
            }
            View::Compare => {
                let left = Rect::from_min_max(img_rect.min, Pos2::new(split_x, img_rect.max.y)).intersect(rect);
                let right = Rect::from_min_max(Pos2::new(split_x, img_rect.min.y), img_rect.max).intersect(rect);
                ui.painter_at(left).image(before.id(), img_rect, uv, Color32::WHITE);
                ui.painter_at(right).image(after.id(), img_rect, uv, Color32::WHITE);

                if split_x > rect.left() && split_x < rect.right() {
                    let top = img_rect.top().max(rect.top());
                    let bot = img_rect.bottom().min(rect.bottom());
                    painter.line_segment(
                        [Pos2::new(split_x, top), Pos2::new(split_x, bot)],
                        Stroke::new(1.5, Color32::from_white_alpha(230)),
                    );
                    let c = Pos2::new(split_x, (top + bot) / 2.0);
                    painter.circle(c, 15.0, Color32::from_black_alpha(150), Stroke::new(1.5, Color32::WHITE));
                    painter.text(c, Align2::CENTER_CENTER, "◀ ▶", FontId::proportional(10.0), Color32::WHITE);
                }
                chip(&painter, Pos2::new(rect.left() + 14.0, rect.top() + 14.0), "VORHER", Align2::LEFT_TOP);
                let after_label = if self.enhanced.is_some() { "NACHHER" } else { "VORSCHAU" };
                chip(&painter, Pos2::new(rect.right() - 14.0, rect.top() + 14.0), after_label, Align2::RIGHT_TOP);
            }
        }

        // Zoomanzeige unten
        let z = format!("{:.0} %", zoom * ppp * 100.0);
        chip(&painter, Pos2::new(rect.right() - 14.0, rect.bottom() - 14.0), &z, Align2::RIGHT_BOTTOM);
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_centered(|ui| {
            ui.label(RichText::new("Bildwerk").size(16.0).strong().color(TEXT));
            ui.add_space(14.0);
            if ui.button("Öffnen …").clicked() {
                self.open_dialog();
            }
            if let Some(p) = &self.path {
                ui.add_space(8.0);
                let name = p.file_name().unwrap_or_default().to_string_lossy();
                ui.label(RichText::new(name).color(DIM));
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let ppp = ui.ctx().pixels_per_point();
                if ui.button("100 %").on_hover_text("Echte Pixel").clicked() {
                    self.zoom = Some(1.0 / ppp);
                    self.pan = Vec2::ZERO;
                }
                if ui.button("Einpassen").on_hover_text("Doppelklick ins Bild").clicked() {
                    self.zoom = None;
                    self.pan = Vec2::ZERO;
                }
                ui.add_space(12.0);
                ui.selectable_value(&mut self.view, View::After, "Nachher");
                ui.selectable_value(&mut self.view, View::Before, "Vorher");
                ui.selectable_value(&mut self.view, View::Compare, "Vergleich");
            });
        });
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.handle_messages(&ctx);
        self.maybe_resharpen(&ctx);

        // Dateien per Drag & Drop
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).filter(|p| !p.as_os_str().is_empty()).collect());
        if let Some(p) = dropped.into_iter().find(|p| crate::is_image(p)) {
            if self.job.is_none() {
                self.open(p);
            }
        }
        // Tastenkürzel
        let (open, save) = ctx.input(|i| {
            (
                i.modifiers.command && i.key_pressed(egui::Key::O),
                i.modifiers.command && i.key_pressed(egui::Key::S),
            )
        });
        if open {
            self.open_dialog();
        }
        if save {
            self.save_dialog();
        }

        egui::Panel::top("top")
            .frame(egui::Frame::new().fill(PANEL).inner_margin(egui::Margin::symmetric(14, 8)).stroke(Stroke::new(1.0, LINE)))
            .exact_size(46.0)
            .show(ui, |ui| self.toolbar(ui));

        egui::Panel::right("side")
            .frame(egui::Frame::new().fill(PANEL).inner_margin(egui::Margin::same(16)).stroke(Stroke::new(1.0, LINE)))
            .exact_size(310.0)
            .resizable(false)
            .show(ui, |ui| {
                egui::Panel::bottom("actions")
                    .frame(egui::Frame::new())
                    .show_separator_line(false)
                    .show(ui, |ui| self.actions_ui(ui));
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| self.settings_ui(ui));
            });

        egui::CentralPanel::no_frame().show(ui, |ui| self.viewer(ui));

        if self.job.is_some() || self.sharpen_busy || self.saving {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
}

// ------------------------------------------------------------------ Hilfsfunktionen

fn make_texture(ctx: &egui::Context, name: &str, pic: &Picture) -> TextureHandle {
    let max = ctx.input(|i| i.max_texture_side).min(16384) as u32;
    let (w, h) = pic.dimensions();
    let rgb = if w > max || h > max {
        let s = max as f32 / w.max(h) as f32;
        image::imageops::resize(
            &pic.rgb,
            ((w as f32 * s) as u32).max(1),
            ((h as f32 * s) as u32).max(1),
            image::imageops::FilterType::Triangle,
        )
    } else {
        pic.rgb.clone()
    };
    let bytes: Vec<u8> = rgb.as_raw().iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
    let img = ColorImage::from_rgb([rgb.width() as usize, rgb.height() as usize], &bytes);
    let opts = TextureOptions {
        magnification: TextureFilter::Nearest,
        minification: TextureFilter::Linear,
        mipmap_mode: Some(TextureFilter::Linear),
        ..Default::default()
    };
    ctx.load_texture(name, img, opts)
}

fn setup_style(ctx: &egui::Context) {
    ctx.set_visuals(egui::Visuals::dark());
    ctx.all_styles_mut(|s| {
        let v = &mut s.visuals;
        v.panel_fill = PANEL;
        v.window_fill = PANEL;
        v.extreme_bg_color = Color32::from_rgb(0x11, 0x12, 0x13);
        v.override_text_color = Some(TEXT);
        v.selection.bg_fill = ACCENT.linear_multiply(0.85);
        v.selection.stroke = Stroke::new(1.0, ACCENT_DARK);
        v.slider_trailing_fill = true;
        for w in [&mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active, &mut v.widgets.open] {
            w.corner_radius = CornerRadius::same(5);
        }
        v.widgets.inactive.weak_bg_fill = Color32::from_rgb(0x26, 0x28, 0x2b);
        v.widgets.inactive.bg_fill = Color32::from_rgb(0x30, 0x32, 0x36);
        v.widgets.hovered.weak_bg_fill = Color32::from_rgb(0x30, 0x32, 0x36);
        v.widgets.active.bg_fill = ACCENT;
        s.spacing.item_spacing = Vec2::new(8.0, 6.0);
        s.spacing.button_padding = Vec2::new(10.0, 5.0);
        s.spacing.slider_rail_height = 4.0;
    });
}

fn section(ui: &mut egui::Ui, title: &str) {
    section_inline(ui, title);
    ui.add_space(8.0);
}

fn section_inline(ui: &mut egui::Ui, title: &str) {
    ui.label(RichText::new(title.to_uppercase()).size(11.5).strong().color(ACCENT).extra_letter_spacing(1.2));
}

/// Regler mit Beschriftung links, Wert rechts und voller Breite darunter.
fn param(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    suffix: &str,
    decimals: usize,
) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(RichText::new(format!("{:.*}{}", decimals, value, suffix)).color(DIM).monospace());
        });
    });
    let w = ui.available_width();
    ui.spacing_mut().slider_width = w;
    let changed = ui.add(egui::Slider::new(value, range).show_value(false)).changed();
    ui.add_space(4.0);
    changed
}

fn toggle(ui: &mut egui::Ui, on: &mut bool) -> egui::Response {
    let size = Vec2::new(34.0, 18.0);
    let (rect, mut resp) = ui.allocate_exact_size(size, Sense::click());
    if resp.clicked() {
        *on = !*on;
        resp.mark_changed();
    }
    let t = ui.ctx().animate_bool(resp.id, *on);
    let bg = Color32::from_rgb(0x3a, 0x3c, 0x40).lerp_to_gamma(ACCENT, t);
    ui.painter().rect_filled(rect, CornerRadius::same(9), bg);
    let x = egui::lerp(rect.left() + 9.0..=rect.right() - 9.0, t);
    ui.painter().circle_filled(Pos2::new(x, rect.center().y), 6.5, Color32::WHITE);
    resp
}

fn chip(painter: &egui::Painter, pos: Pos2, text: &str, align: Align2) {
    let galley = painter.layout_no_wrap(text.to_string(), FontId::proportional(11.0), Color32::WHITE);
    let size = galley.size() + Vec2::new(16.0, 8.0);
    let rect = align.anchor_size(pos, size);
    painter.rect_filled(rect, CornerRadius::same(4), Color32::from_black_alpha(170));
    painter.galley(rect.min + Vec2::new(8.0, 4.0), galley, Color32::WHITE);
}

fn empty_state(painter: &egui::Painter, rect: Rect) {
    let r = Rect::from_center_size(rect.center(), Vec2::new(420.0, 200.0));
    painter.rect_stroke(
        r,
        CornerRadius::same(12),
        Stroke::new(1.5, Color32::from_rgb(0x3a, 0x3c, 0x40)),
        egui::StrokeKind::Inside,
    );
    painter.text(
        r.center() - Vec2::new(0.0, 22.0),
        Align2::CENTER_CENTER,
        "Bild hierher ziehen",
        FontId::proportional(20.0),
        TEXT,
    );
    painter.text(
        r.center() + Vec2::new(0.0, 12.0),
        Align2::CENTER_CENTER,
        "oder klicken zum Öffnen  ·  PNG, JPEG, TIFF, WebP",
        FontId::proportional(13.0),
        DIM,
    );
}
