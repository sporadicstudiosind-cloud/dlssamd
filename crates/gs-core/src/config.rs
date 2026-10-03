//! User-facing settings. Everything is serde-serializable (TOML) and every numeric
//! field is clamped by [`Settings::sanitize`] so a hand-edited file can't crash the pipeline.

use serde::{Deserialize, Serialize};

/// Spatial upscaling kernels implemented by the built-in capture lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpscalerKind {
    Nearest,
    Bilinear,
    CatmullRom,
    Lanczos2,
    Lanczos3,
    /// Direction-aware kernel (original implementation, FSR1-style idea).
    EdgeAdaptive,
}

impl UpscalerKind {
    pub const ALL: [UpscalerKind; 6] = [
        Self::Nearest,
        Self::Bilinear,
        Self::CatmullRom,
        Self::Lanczos2,
        Self::Lanczos3,
        Self::EdgeAdaptive,
    ];
    /// Value passed to the shader's `mode` uniform.
    pub fn shader_mode(self) -> u32 {
        match self {
            Self::Nearest => 0,
            Self::Bilinear => 1,
            Self::CatmullRom => 2,
            Self::Lanczos2 => 3,
            Self::Lanczos3 => 4,
            Self::EdgeAdaptive => 5,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Nearest => "Nearest (pixel-exact)",
            Self::Bilinear => "Bilinear",
            Self::CatmullRom => "Bicubic (Catmull-Rom)",
            Self::Lanczos2 => "Lanczos 2",
            Self::Lanczos3 => "Lanczos 3",
            Self::EdgeAdaptive => "Edge-adaptive (built-in)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameGenKind {
    Off,
    /// Plain cross-fade. Cheap, ghosts on motion; mainly a baseline for comparison.
    Blend,
    /// Block-matching optical flow + motion-compensated warp (built-in).
    OpticalFlow,
}

impl FrameGenKind {
    pub const ALL: [FrameGenKind; 3] = [Self::Off, Self::Blend, Self::OpticalFlow];
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Blend => "Blend (baseline)",
            Self::OpticalFlow => "Optical flow (built-in)",
        }
    }
}

/// Whether interpolation runs on the low-res source or on the upscaled output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameGenStage {
    /// Interpolate at source resolution, then upscale every frame: cheaper, upscales N frames.
    BeforeUpscale,
    /// Upscale once, then interpolate at output resolution: sharper warps, more bandwidth.
    AfterUpscale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PresentMode {
    /// Lowest latency, may tear. Needs the overlay to be borderless fullscreen-sized.
    Immediate,
    /// Replace queued frame with newest; low latency without tearing.
    Mailbox,
    /// Classic vsync.
    Fifo,
}

impl PresentMode {
    pub const ALL: [PresentMode; 3] = [Self::Immediate, Self::Mailbox, Self::Fifo];
    pub fn label(self) -> &'static str {
        match self {
            Self::Immediate => "Immediate (lowest latency, may tear)",
            Self::Mailbox => "Mailbox (low latency, no tearing)",
            Self::Fifo => "FIFO (vsync)",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct UpscaleSettings {
    pub enabled: bool,
    pub kind: UpscalerKind,
    /// Output size = captured size * `scale`. Ignored if `target_height` is non-zero.
    pub scale: f32,
    /// If non-zero, output height in pixels (width follows the capture aspect ratio).
    pub target_height: u32,
    /// Post-upscale contrast-adaptive sharpening strength, 0..1.
    pub sharpness: f32,
    /// 0..1, clamps ringing on Lanczos/bicubic.
    pub anti_ringing: f32,
}

impl Default for UpscaleSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            kind: UpscalerKind::EdgeAdaptive,
            scale: 1.5,
            target_height: 0,
            sharpness: 0.25,
            anti_ringing: 0.8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct FrameGenSettings {
    pub kind: FrameGenKind,
    /// Output frames per real frame (2 = double). Range 2..=8.
    pub multiplier: u32,
    pub stage: FrameGenStage,
    /// Motion block size in source pixels (4, 8, 16). Smaller = finer, slower.
    pub block_size: u32,
    /// Max search distance in source pixels.
    pub search_radius: u32,
    /// Luma sampling stride inside a block (1 = every pixel = best, slowest).
    pub sample_stride: u32,
    /// Blocks whose normalised match error exceeds this fall back to a blend. 0..1.
    pub confidence_threshold: f32,
    /// Rectangles (x, y, w, h as 0..1 fractions) excluded from warping, e.g. HUD.
    pub hud_masks: Vec<[f32; 4]>,
}

impl Default for FrameGenSettings {
    fn default() -> Self {
        Self {
            kind: FrameGenKind::OpticalFlow,
            multiplier: 2,
            stage: FrameGenStage::BeforeUpscale,
            block_size: 8,
            search_radius: 16,
            sample_stride: 2,
            confidence_threshold: 0.12,
            hud_masks: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct LatencySettings {
    pub present_mode: PresentMode,
    /// Frames the swapchain may queue ahead (1 = lowest latency).
    pub max_frame_latency: u32,
    /// Capture-side queue depth. 1 = always process the newest frame, dropping stale ones.
    pub capture_queue_depth: u32,
    /// Hard cap on presented FPS. 0 = uncapped.
    pub fps_cap: u32,
    /// Spin the last ~`spin_us` microseconds of every wait instead of sleeping.
    pub precise_pacing: bool,
    pub spin_us: u32,
    /// Raise process/thread priority of the render thread.
    pub high_priority: bool,
}

impl Default for LatencySettings {
    fn default() -> Self {
        Self {
            present_mode: PresentMode::Mailbox,
            max_frame_latency: 1,
            capture_queue_depth: 1,
            fps_cap: 0,
            precise_pacing: true,
            spin_us: 1500,
            high_priority: true,
        }
    }
}

/// A named bundle of settings, optionally bound to an executable name.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Profile {
    pub name: String,
    /// Case-insensitive exe file name this profile auto-applies to (empty = manual only).
    pub exe_match: String,
    pub upscale: UpscaleSettings,
    pub framegen: FrameGenSettings,
    pub latency: LatencySettings,
    /// Backend ids chosen from the catalog (see `catalog.rs`). Built-ins map onto the
    /// typed fields above; others are recorded for the injection lane.
    pub upscaler_backend: String,
    pub framegen_backend: String,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            name: "Default".into(),
            exe_match: String::new(),
            upscale: Default::default(),
            framegen: Default::default(),
            latency: Default::default(),
            upscaler_backend: "builtin.edge_adaptive".into(),
            framegen_backend: "builtin.optical_flow".into(),
        }
    }
}

impl Profile {
    /// Clamp everything into ranges the GPU code supports.
    pub fn sanitize(&mut self) {
        let u = &mut self.upscale;
        u.scale = u.scale.clamp(1.0, 4.0);
        u.target_height = u.target_height.min(8640);
        u.sharpness = u.sharpness.clamp(0.0, 1.0);
        u.anti_ringing = u.anti_ringing.clamp(0.0, 1.0);

        let f = &mut self.framegen;
        f.multiplier = f.multiplier.clamp(2, 8);
        f.block_size = match f.block_size {
            0..=5 => 4,
            6..=11 => 8,
            _ => 16,
        };
        f.search_radius = f.search_radius.clamp(2, 64);
        f.sample_stride = f.sample_stride.clamp(1, 4);
        f.confidence_threshold = f.confidence_threshold.clamp(0.0, 1.0);
        for m in &mut f.hud_masks {
            for v in m.iter_mut() {
                *v = v.clamp(0.0, 1.0);
            }
        }

        let l = &mut self.latency;
        l.max_frame_latency = l.max_frame_latency.clamp(1, 4);
        l.capture_queue_depth = l.capture_queue_depth.clamp(1, 8);
        l.fps_cap = l.fps_cap.min(1000);
        l.spin_us = l.spin_us.min(10_000);
    }

    /// Output size for a given capture size.
    pub fn output_size(&self, capture: (u32, u32)) -> (u32, u32) {
        let (w, h) = (capture.0.max(1), capture.1.max(1));
        if !self.upscale.enabled {
            return (w, h);
        }
        let (ow, oh) = if self.upscale.target_height > 0 {
            let s = self.upscale.target_height as f32 / h as f32;
            ((w as f32 * s).round(), self.upscale.target_height as f32)
        } else {
            (
                (w as f32 * self.upscale.scale).round(),
                (h as f32 * self.upscale.scale).round(),
            )
        };
        ((ow as u32).clamp(1, 16384), (oh as u32).clamp(1, 16384))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Settings {
    pub profiles: Vec<Profile>,
    pub active: usize,
    /// Show the on-overlay stats readout.
    pub show_stats: bool,
    /// Hotkey description, informational (the app binds Ctrl+Alt+S).
    pub hotkey: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            profiles: vec![Profile::default()],
            active: 0,
            show_stats: true,
            hotkey: "Ctrl+Alt+S".into(),
        }
    }
}

impl Settings {
    pub fn sanitize(&mut self) {
        if self.profiles.is_empty() {
            self.profiles.push(Profile::default());
        }
        self.active = self.active.min(self.profiles.len() - 1);
        for p in &mut self.profiles {
            p.sanitize();
        }
    }

    pub fn active_profile(&self) -> &Profile {
        &self.profiles[self.active.min(self.profiles.len() - 1)]
    }

    pub fn active_profile_mut(&mut self) -> &mut Profile {
        let i = self.active.min(self.profiles.len() - 1);
        &mut self.profiles[i]
    }

    /// Index of the profile bound to `exe_name`, if any.
    pub fn profile_for_exe(&self, exe_name: &str) -> Option<usize> {
        self.profiles
            .iter()
            .position(|p| !p.exe_match.is_empty() && p.exe_match.eq_ignore_ascii_case(exe_name))
    }

    pub fn to_toml(&self) -> anyhow::Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }

    pub fn from_toml(s: &str) -> anyhow::Result<Self> {
        let mut v: Settings = toml::from_str(s)?;
        v.sanitize();
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let mut s = Settings::default();
        s.profiles[0].framegen.multiplier = 4;
        s.profiles[0].framegen.hud_masks.push([0.0, 0.9, 1.0, 0.1]);
        let back = Settings::from_toml(&s.to_toml().unwrap()).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn partial_file_uses_defaults_and_sanitizes() {
        let s = Settings::from_toml(
            "[[profiles]]\nname='x'\n[profiles.framegen]\nmultiplier=99\nblock_size=7\n",
        )
        .unwrap();
        assert_eq!(s.profiles[0].framegen.multiplier, 8);
        assert_eq!(s.profiles[0].framegen.block_size, 8);
        assert_eq!(s.profiles[0].upscale, UpscaleSettings::default());
    }

    #[test]
    fn output_size() {
        let mut p = Profile::default();
        p.upscale.scale = 1.5;
        assert_eq!(p.output_size((1280, 720)), (1920, 1080));
        p.upscale.target_height = 1440;
        assert_eq!(p.output_size((1280, 720)), (2560, 1440));
        p.upscale.enabled = false;
        assert_eq!(p.output_size((1280, 720)), (1280, 720));
    }

    #[test]
    fn exe_match_is_case_insensitive() {
        let mut s = Settings::default();
        s.profiles.push(Profile {
            name: "g".into(),
            exe_match: "Game.EXE".into(),
            ..Default::default()
        });
        assert_eq!(s.profile_for_exe("game.exe"), Some(1));
        assert_eq!(s.profile_for_exe("other.exe"), None);
    }
}
