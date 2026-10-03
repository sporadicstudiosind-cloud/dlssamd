//! The settings panel.

use crate::overlay::{Placement, Session};
use crate::source::{FrameSource, TestPattern};
use crate::store;
use gs_core::catalog::{self, Kind, Lane, Status};
use gs_core::config::*;
use gs_core::pacing::estimate_latency;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Source,
    Upscaling,
    FrameGen,
    Latency,
    Backends,
    About,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SourceKind {
    Window,
    TestPattern,
}

pub struct App {
    settings: Settings,
    saved: Settings,
    last_save: Instant,
    save_error: Option<String>,
    tab: Tab,
    session: Option<Session>,
    source_kind: SourceKind,
    placement: Placement,
    test_size: (u32, u32),
    test_fps: u32,
    display_hz: f32,
    #[cfg_attr(not(windows), allow(dead_code))]
    filter: String,
    #[cfg(windows)]
    windows: Vec<(windows_capture::window::Window, String, String)>,
    #[cfg(windows)]
    selected_window: Option<usize>,
    start_error: Option<String>,
}

impl App {
    pub fn new() -> Self {
        let settings = store::load();
        let mut app = Self {
            saved: settings.clone(),
            settings,
            last_save: Instant::now(),
            save_error: None,
            tab: Tab::Source,
            session: None,
            source_kind: if cfg!(windows) {
                SourceKind::Window
            } else {
                SourceKind::TestPattern
            },
            placement: Placement::MatchTarget,
            test_size: (640, 360),
            test_fps: 30,
            display_hz: 60.0,
            filter: String::new(),
            #[cfg(windows)]
            windows: Vec::new(),
            #[cfg(windows)]
            selected_window: None,
            start_error: None,
        };
        app.refresh_windows();
        app
    }

    fn refresh_windows(&mut self) {
        #[cfg(windows)]
        {
            self.windows = crate::windows_capture_source::list_windows();
            self.selected_window = None;
        }
    }

    fn make_source(&mut self) -> Result<Box<dyn FrameSource>, String> {
        match self.source_kind {
            SourceKind::TestPattern => Ok(Box::new(TestPattern::new(
                self.test_size.0,
                self.test_size.1,
                self.test_fps,
            ))),
            SourceKind::Window => {
                #[cfg(windows)]
                {
                    let i = self.selected_window.ok_or("pick a window first")?;
                    let (w, _, exe) = self.windows[i].clone();
                    if let Some(p) = self.settings.profile_for_exe(&exe) {
                        self.settings.active = p;
                    }
                    Ok(Box::new(crate::windows_capture_source::WindowCapture::new(
                        w,
                    )))
                }
                #[cfg(not(windows))]
                Err("window capture needs Windows (Windows Graphics Capture); use the test pattern here".into())
            }
        }
    }

    fn start(&mut self) {
        self.start_error = None;
        match self.make_source() {
            Ok(src) => {
                let mut profile = self.settings.active_profile().clone();
                profile.sanitize();
                self.session = Some(Session::start(src, profile, self.placement));
            }
            Err(e) => self.start_error = Some(e),
        }
    }

    fn stop(&mut self) {
        if let Some(mut s) = self.session.take() {
            s.stop();
        }
    }

    fn autosave(&mut self) {
        if self.settings != self.saved && self.last_save.elapsed() > Duration::from_millis(800) {
            match store::save(&self.settings) {
                Ok(()) => {
                    self.saved = self.settings.clone();
                    self.save_error = None;
                }
                Err(e) => self.save_error = Some(format!("{e:#}")),
            }
            self.last_save = Instant::now();
        }
    }
}

fn badge(ui: &mut egui::Ui, text: &str, color: egui::Color32) {
    ui.label(egui::RichText::new(text).color(color).small().strong());
}

fn status_text(b: &catalog::Backend) -> (&'static str, egui::Color32) {
    match b.status {
        Status::Builtin => ("built-in", egui::Color32::from_rgb(70, 170, 90)),
        Status::Planned => ("planned", egui::Color32::from_rgb(200, 160, 50)),
        Status::NeedsUserFiles => ("needs your files", egui::Color32::from_rgb(210, 120, 60)),
        Status::ExternalToggle => ("external", egui::Color32::from_rgb(110, 140, 200)),
    }
}

fn lane_text(l: Lane) -> &'static str {
    match l {
        Lane::Capture => "any app (capture)",
        Lane::Injection => "per-game injection",
        Lane::External => "outside this app",
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let running = self.session.as_ref().is_some_and(|s| s.is_running());
        if self.session.is_some() && !running {
            // Thread ended on its own (window closed / error): keep stats for display, drop handle.
        }
        if running {
            ctx.request_repaint_after(Duration::from_millis(250));
        }

        egui::Panel::left("profiles")
            .default_size(200.0)
            .show(ui, |ui| {
                ui.heading("Profiles");
                let mut remove = None;
                for i in 0..self.settings.profiles.len() {
                    ui.horizontal(|ui| {
                        let name = if self.settings.profiles[i].exe_match.is_empty() {
                            self.settings.profiles[i].name.clone()
                        } else {
                            format!(
                                "{}  ({})",
                                self.settings.profiles[i].name, self.settings.profiles[i].exe_match
                            )
                        };
                        ui.selectable_value(&mut self.settings.active, i, name);
                        if self.settings.profiles.len() > 1
                            && ui
                                .small_button("x")
                                .on_hover_text("Delete profile")
                                .clicked()
                        {
                            remove = Some(i);
                        }
                    });
                }
                if let Some(i) = remove {
                    self.settings.profiles.remove(i);
                    self.settings.sanitize();
                }
                ui.horizontal(|ui| {
                    if ui.button("New").clicked() {
                        let name = format!("Profile {}", self.settings.profiles.len() + 1);
                        self.settings.profiles.push(Profile {
                            name,
                            ..Default::default()
                        });
                        self.settings.active = self.settings.profiles.len() - 1;
                    }
                    if ui.button("Duplicate").clicked() {
                        let mut p = self.settings.active_profile().clone();
                        p.name += " copy";
                        p.exe_match.clear();
                        self.settings.profiles.push(p);
                        self.settings.active = self.settings.profiles.len() - 1;
                    }
                });
                ui.separator();
                let p = self.settings.active_profile_mut();
                ui.label("Name");
                ui.text_edit_singleline(&mut p.name);
                ui.label("Auto-apply to exe");
                ui.text_edit_singleline(&mut p.exe_match).on_hover_text(
                    "e.g. game.exe. Applied when you start capturing a window from that program.",
                );
                ui.separator();
                if let Some(e) = &self.save_error {
                    ui.colored_label(egui::Color32::RED, format!("Could not save: {e}"));
                } else {
                    ui.weak(format!("Saved to {}", store::settings_path().display()));
                }
            });

        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                if running {
                    if ui.button("Stop").clicked() {
                        self.stop();
                    }
                } else {
                    if ui.button("Start").clicked() {
                        self.start();
                    }
                    if ui
                        .button("Smooth Motion (2x)")
                        .on_hover_text("One click: fixed 2x frame generation from the frames the app presents, no upscaling, lowest-latency presentation. Clean-room equivalent of the documented behaviour of NVIDIA Smooth Motion, not NVIDIA's code.")
                        .clicked()
                    {
                        self.settings.active = self.settings.ensure_smooth_motion();
                        self.start();
                    }
                }
                if let Some(e) = &self.start_error {
                    ui.colored_label(egui::Color32::RED, e);
                }
                if let Some(s) = &self.session {
                    let st = s.stats.lock().unwrap().clone();
                    if let Some(e) = &st.error {
                        ui.colored_label(egui::Color32::RED, e);
                    } else if st.running {
                        ui.label(format!(
                            "{} | {} | source {:.0} fps -> output {:.0} fps | {}x{} -> {}x{} | GPU submit {:.2} ms | measured capture-to-screen {:.1} ms | dropped {}",
                            s.label, st.adapter, st.source_fps, st.output_fps, st.capture_size.0, st.capture_size.1, st.output_size.0, st.output_size.1, st.process_ms, st.latency_ms, st.dropped_frames
                        ));
                    } else {
                        ui.label("Stopped.");
                    }
                }
            });
        });

        egui::CentralPanel::default().show(ui, |ui| {
            ui.horizontal(|ui| {
                for (t, label) in [
                    (Tab::Source, "Source"),
                    (Tab::Upscaling, "Upscaling"),
                    (Tab::FrameGen, "Frame generation"),
                    (Tab::Latency, "Latency"),
                    (Tab::Backends, "Backends"),
                    (Tab::About, "About"),
                ] {
                    ui.selectable_value(&mut self.tab, t, label);
                }
            });
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| match self.tab {
                Tab::Source => self.tab_source(ui, running),
                Tab::Upscaling => self.tab_upscaling(ui),
                Tab::FrameGen => self.tab_framegen(ui),
                Tab::Latency => self.tab_latency(ui),
                Tab::Backends => self.tab_backends(ui),
                Tab::About => tab_about(ui),
            });
        });

        self.settings.active_profile_mut().sanitize();
        self.autosave();
    }
}

impl App {
    fn tab_source(&mut self, ui: &mut egui::Ui, running: bool) {
        ui.add_enabled_ui(!running, |ui| {
            ui.heading("What to process");
            ui.horizontal(|ui| {
                ui.radio_value(&mut self.source_kind, SourceKind::Window, "A window (Windows only)");
                ui.radio_value(&mut self.source_kind, SourceKind::TestPattern, "Test pattern (any OS, no game needed)");
            });
            match self.source_kind {
                SourceKind::TestPattern => {
                    ui.horizontal(|ui| {
                        ui.label("Size");
                        ui.add(egui::DragValue::new(&mut self.test_size.0).range(64..=3840));
                        ui.label("x");
                        ui.add(egui::DragValue::new(&mut self.test_size.1).range(64..=2160));
                        ui.label("at");
                        ui.add(egui::DragValue::new(&mut self.test_fps).range(1..=240).suffix(" fps"));
                    });
                    ui.weak("A moving scene with a static HUD bar. Use it to compare upscalers and frame generation settings.");
                }
                SourceKind::Window => {
                    #[cfg(windows)]
                    {
                        ui.horizontal(|ui| {
                            ui.label("Filter");
                            ui.text_edit_singleline(&mut self.filter);
                            if ui.button("Refresh").clicked() {
                                self.refresh_windows();
                            }
                        });
                        let f = self.filter.to_lowercase();
                        egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                            for (i, (_, title, exe)) in self.windows.iter().enumerate() {
                                if !f.is_empty() && !title.to_lowercase().contains(&f) && !exe.to_lowercase().contains(&f) {
                                    continue;
                                }
                                ui.selectable_value(&mut self.selected_window, Some(i), format!("{title}   [{exe}]"));
                            }
                        });
                    }
                    #[cfg(not(windows))]
                    ui.colored_label(egui::Color32::from_rgb(210, 120, 60), "Window capture uses the Windows Graphics Capture API and is not available on this OS.");
                }
            }
            ui.add_space(8.0);
            ui.heading("Where to show it");
            ui.radio_value(&mut self.placement, Placement::MatchTarget, "Over the source window (mouse stays aligned)");
            ui.radio_value(&mut self.placement, Placement::FillMonitor, "Fill the monitor (see upscaling at full size; mouse is NOT remapped yet)");
        });
        ui.add_space(8.0);
        ui.weak("The overlay is click-through, so input still reaches the game. Run the game windowed or borderless; exclusive fullscreen can't be captured or overlaid.");
    }

    fn tab_upscaling(&mut self, ui: &mut egui::Ui) {
        let p = self.settings.active_profile_mut();
        ui.checkbox(&mut p.upscale.enabled, "Enable upscaling");
        ui.add_enabled_ui(p.upscale.enabled, |ui| {
            ui.horizontal(|ui| {
                ui.label("Upscaler");
                let current = catalog::by_id(&p.upscaler_backend).map(|b| b.name).unwrap_or("?");
                egui::ComboBox::from_id_salt("upscaler").selected_text(current).width(280.0).show_ui(ui, |ui| {
                    for b in catalog::of_kind(Kind::Upscaler) {
                        let (st, _) = status_text(b);
                        let label = if b.selectable() { b.name.to_string() } else { format!("{}  [{}]", b.name, st) };
                        let resp = ui.add_enabled(b.selectable(), egui::Button::selectable(p.upscaler_backend == b.id, label));
                        let resp = resp.on_hover_text(b.note).on_disabled_hover_text(b.note);
                        if resp.clicked() {
                            p.upscaler_backend = b.id.to_string();
                            if let Some(k) = catalog::upscaler_for_id(b.id) {
                                p.upscale.kind = k;
                            }
                        }
                    }
                });
            });
            if let Some(b) = catalog::by_id(&p.upscaler_backend) {
                ui.weak(b.note);
            }
            ui.add_space(6.0);
            ui.label("Output resolution");
            ui.horizontal(|ui| {
                ui.radio_value(&mut p.upscale.target_height, 0, "Scale factor");
                if ui.radio(p.upscale.target_height != 0, "Fixed height").clicked() && p.upscale.target_height == 0 {
                    p.upscale.target_height = 1080;
                }
            });
            if p.upscale.target_height == 0 {
                ui.add(egui::Slider::new(&mut p.upscale.scale, 1.0..=4.0).text("scale").step_by(0.05));
            } else {
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut p.upscale.target_height).range(144..=8640).suffix(" px high"));
                    for h in [720, 1080, 1440, 2160] {
                        if ui.small_button(format!("{h}p")).clicked() {
                            p.upscale.target_height = h;
                        }
                    }
                });
            }
            ui.add(egui::Slider::new(&mut p.upscale.sharpness, 0.0..=1.0).text("sharpening (CAS)"));
            ui.add(egui::Slider::new(&mut p.upscale.anti_ringing, 0.0..=1.0).text("anti-ringing"));
            ui.add_space(6.0);
            ui.weak("Input resolution is whatever the source renders at. This app can't change a game's internal resolution, so set the game's render scale/resolution lower and let this upscale it.");
        });
    }

    fn tab_framegen(&mut self, ui: &mut egui::Ui) {
        let p = self.settings.active_profile_mut();
        ui.horizontal(|ui| {
            ui.label("Frame generator");
            let current = catalog::by_id(&p.framegen_backend)
                .map(|b| b.name)
                .unwrap_or("Off");
            egui::ComboBox::from_id_salt("framegen")
                .selected_text(if p.framegen.kind == FrameGenKind::Off {
                    "Off"
                } else {
                    current
                })
                .width(280.0)
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(p.framegen.kind == FrameGenKind::Off, "Off")
                        .clicked()
                    {
                        p.framegen.kind = FrameGenKind::Off;
                    }
                    for b in catalog::of_kind(Kind::FrameGen) {
                        let (st, _) = status_text(b);
                        let label = if b.selectable() {
                            b.name.to_string()
                        } else {
                            format!("{}  [{}]", b.name, st)
                        };
                        let resp = ui.add_enabled(
                            b.selectable(),
                            egui::Button::selectable(
                                p.framegen_backend == b.id && p.framegen.kind != FrameGenKind::Off,
                                label,
                            ),
                        );
                        if resp
                            .on_hover_text(b.note)
                            .on_disabled_hover_text(b.note)
                            .clicked()
                        {
                            p.framegen_backend = b.id.to_string();
                            if let Some(k) = catalog::framegen_for_id(b.id) {
                                p.framegen.kind = k;
                            }
                        }
                    }
                });
        });
        if p.framegen.kind == FrameGenKind::Off {
            return;
        }
        if let Some(b) = catalog::by_id(&p.framegen_backend) {
            ui.weak(b.note);
        }
        ui.add_space(6.0);
        ui.add(
            egui::Slider::new(&mut p.framegen.multiplier, 2..=8)
                .text("output frames per real frame"),
        );
        ui.label(format!(
            "Generates {} new frame(s) between each pair of real frames.",
            p.framegen.multiplier - 1
        ));
        ui.horizontal(|ui| {
            ui.label("Interpolate");
            ui.selectable_value(
                &mut p.framegen.stage,
                FrameGenStage::BeforeUpscale,
                "before upscaling (faster)",
            );
            ui.selectable_value(
                &mut p.framegen.stage,
                FrameGenStage::AfterUpscale,
                "after upscaling (sharper warps)",
            );
        });
        if p.framegen.kind == FrameGenKind::OpticalFlow {
            ui.add_space(6.0);
            ui.label("Motion estimation");
            ui.horizontal(|ui| {
                ui.label("Block size");
                for b in [4u32, 8, 16] {
                    ui.selectable_value(&mut p.framegen.block_size, b, format!("{b}px"));
                }
            });
            ui.add(
                egui::Slider::new(&mut p.framegen.search_radius, 2..=64).text("search radius (px)"),
            );
            ui.add(
                egui::Slider::new(&mut p.framegen.sample_stride, 1..=4)
                    .text("sample stride (1 = most accurate)"),
            );
            ui.add(
                egui::Slider::new(&mut p.framegen.confidence_threshold, 0.0..=0.5)
                    .text("fallback threshold"),
            );
            ui.weak("Blocks whose match error is above the threshold fade back to a plain blend instead of showing a wrong warp.");
            ui.add_space(6.0);
            ui.label("HUD protection (regions excluded from warping)");
            let mut remove = None;
            for (i, m) in p.framegen.hud_masks.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    for (v, n) in m.iter_mut().zip(["x", "y", "w", "h"]) {
                        ui.add(
                            egui::DragValue::new(v)
                                .range(0.0..=1.0)
                                .speed(0.005)
                                .prefix(format!("{n} ")),
                        );
                    }
                    if ui.small_button("remove").clicked() {
                        remove = Some(i);
                    }
                });
            }
            if let Some(i) = remove {
                p.framegen.hud_masks.remove(i);
            }
            if p.framegen.hud_masks.len() < 4 && ui.button("Add region").clicked() {
                p.framegen.hud_masks.push([0.0, 0.9, 1.0, 0.1]);
            }
        }
        ui.add_space(8.0);
        ui.colored_label(egui::Color32::from_rgb(210, 160, 60), "Frame generation holds back one real frame, so it adds about one source-frame interval of latency. See the Latency tab.");
    }

    fn tab_latency(&mut self, ui: &mut egui::Ui) {
        let stats = self
            .session
            .as_ref()
            .map(|s| s.stats.lock().unwrap().clone());
        let p = self.settings.active_profile_mut();
        ui.heading("Presentation");
        ui.horizontal(|ui| {
            ui.label("Present mode");
            egui::ComboBox::from_id_salt("present")
                .selected_text(p.latency.present_mode.label())
                .show_ui(ui, |ui| {
                    for m in PresentMode::ALL {
                        ui.selectable_value(&mut p.latency.present_mode, m, m.label());
                    }
                });
        });
        ui.add(
            egui::Slider::new(&mut p.latency.max_frame_latency, 1..=4)
                .text("max frames queued by the swapchain"),
        );
        ui.add(
            egui::Slider::new(&mut p.latency.capture_queue_depth, 1..=8)
                .text("capture queue depth (1 = always newest)"),
        );
        ui.horizontal(|ui| {
            ui.label("FPS cap (0 = none)");
            ui.add(egui::DragValue::new(&mut p.latency.fps_cap).range(0..=1000));
        });
        ui.checkbox(
            &mut p.latency.precise_pacing,
            "Precise pacing (spin-wait the final stretch)",
        );
        ui.add_enabled(
            p.latency.precise_pacing,
            egui::Slider::new(&mut p.latency.spin_us, 0..=5000).text("spin window (us)"),
        );
        ui.checkbox(
            &mut p.latency.high_priority,
            "High-priority render thread + 1 ms timer",
        );
        ui.add_space(8.0);
        ui.collapsing("About Reflex / Anti-Lag", |ui| {
            ui.label(
                "NVIDIA Reflex and AMD Anti-Lag 2 are SDKs a *game* calls to sync its own CPU/GPU work. \
                 An external app can't switch them on for another process. The controls above are the \
                 equivalent knobs for this app's own capture-and-present path, which is the part it can control. \
                 Turn Reflex / Anti-Lag on inside the game as well.",
            );
        });
        ui.add_space(8.0);
        ui.heading("Estimated added latency");
        let source_fps = stats
            .as_ref()
            .filter(|s| s.running && s.source_fps > 1.0)
            .map(|s| s.source_fps)
            .unwrap_or(60.0);
        let proc_ms = stats.as_ref().map(|s| s.process_ms).unwrap_or(2.0);
        let fg = p.framegen.kind != FrameGenKind::Off;
        ui.horizontal(|ui| {
            ui.label("Display refresh");
            ui.add(
                egui::DragValue::new(&mut self.display_hz)
                    .range(30.0..=540.0)
                    .suffix(" Hz"),
            );
        });
        let e = estimate_latency(
            source_fps,
            p.framegen.multiplier,
            fg,
            p.latency.capture_queue_depth,
            p.latency.max_frame_latency,
            self.display_hz,
            proc_ms,
        );
        egui::Grid::new("lat").show(ui, |ui| {
            for (k, v) in [
                ("Capture", e.capture_ms),
                ("Frame-gen hold", e.interpolation_hold_ms),
                ("GPU processing", e.processing_ms),
                ("Present", e.present_ms),
                ("Total (approx.)", e.total_ms),
            ] {
                ui.label(k);
                ui.label(format!("{v:.1} ms"));
                ui.end_row();
            }
        });
        if let Some(st) = stats.as_ref().filter(|s| s.running) {
            ui.label(format!("Measured, capture callback to first presented frame: {:.1} ms (excludes the game's own render time and display scan-out).", st.latency_ms));
        }
        ui.weak(format!("Based on {source_fps:.0} fps source. A rough model, not a measurement: use a high-speed camera or LDAT-style tool for real numbers."));
    }

    fn tab_backends(&mut self, ui: &mut egui::Ui) {
        ui.weak("Everything the app knows about, and why some entries can't be selected. Hover rows for details.");
        egui::Grid::new("backends")
            .striped(true)
            .num_columns(6)
            .show(ui, |ui| {
                for h in ["Name", "Vendor", "Type", "Works on", "GPU", "Status"] {
                    ui.strong(h);
                }
                ui.end_row();
                for b in catalog::BACKENDS {
                    let (st, col) = status_text(b);
                    ui.label(b.name).on_hover_text(b.note);
                    ui.label(b.vendor);
                    ui.label(match b.kind {
                        Kind::Upscaler => "upscaler",
                        Kind::FrameGen => "frame gen",
                    });
                    ui.label(lane_text(b.lane));
                    ui.label(match b.gpu {
                        catalog::GpuReq::Any => "any".to_string(),
                        catalog::GpuReq::NvidiaRtx => "NVIDIA RTX".into(),
                        catalog::GpuReq::Amd => "AMD".into(),
                        catalog::GpuReq::Intel => "Intel".into(),
                        catalog::GpuReq::AnyFasterOn(s) => format!("any (best on {s})"),
                    });
                    badge(ui, st, col);
                    ui.end_row();
                }
            });
    }
}

fn tab_about(ui: &mut egui::Ui) {
    ui.heading("How this works");
    ui.label(
        "This app captures a window, upscales it on the GPU, optionally inserts generated frames, and shows \
         the result in a click-through overlay. It works on any app because it never touches the game.",
    );
    ui.add_space(6.0);
    ui.strong("What that can't do");
    ui.label("- DLSS, FSR 2/3/4, XeSS and vendor frame generation need the game's motion vectors and depth, which a screen capture doesn't have. They're listed under Backends as per-game injection and are not implemented.");
    ui.label("- Generated frames come from image-only motion estimation, so expect artifacts on HUDs, particles and thin geometry, and about one frame of extra latency.");
    ui.label("- Games in exclusive fullscreen can't be captured or overlaid.");
    ui.label("- Anti-cheat may object to overlays. Use it in single-player games.");
    ui.label(
        "- Window capture currently copies frames through the CPU. Fine for 1080p, costly at 4K.",
    );
}
