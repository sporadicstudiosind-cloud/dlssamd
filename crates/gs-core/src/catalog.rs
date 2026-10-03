//! Catalog of upscalers / frame generators the app knows about, with an honest status
//! for each. Only `Status::Builtin` entries run in the capture lane today; the rest are
//! listed so the settings panel can show *why* something isn't selectable.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Upscaler,
    FrameGen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Lane {
    /// Works on any window via screen capture; no game cooperation needed.
    Capture,
    /// Needs the game's motion vectors / depth, i.e. DLL injection into the game.
    Injection,
    /// Lives outside this app (driver or third-party program).
    External,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Status {
    /// Implemented and selectable.
    Builtin,
    /// Planned for the injection lane; the plugin interface exists, no implementation yet.
    Planned,
    /// Cannot be bundled (licence/proprietary); user would have to supply it themselves.
    NeedsUserFiles,
    /// Controlled outside this app; we only link to how to enable it.
    ExternalToggle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuReq {
    Any,
    NvidiaRtx,
    Amd,
    Intel,
    /// Works everywhere but faster on specific hardware (e.g. matrix cores).
    AnyFasterOn(&'static str),
}

#[derive(Debug, Clone, Copy)]
pub struct Backend {
    pub id: &'static str,
    pub name: &'static str,
    pub vendor: &'static str,
    pub kind: Kind,
    pub lane: Lane,
    pub status: Status,
    pub gpu: GpuReq,
    pub note: &'static str,
}

impl Backend {
    pub fn selectable(&self) -> bool {
        self.status == Status::Builtin
    }
}

macro_rules! b {
    ($id:expr, $name:expr, $vendor:expr, $kind:ident, $lane:ident, $status:ident, $gpu:expr, $note:expr) => {
        Backend {
            id: $id,
            name: $name,
            vendor: $vendor,
            kind: Kind::$kind,
            lane: Lane::$lane,
            status: Status::$status,
            gpu: $gpu,
            note: $note,
        }
    };
}

pub const BACKENDS: &[Backend] = &[
    // ---- Built-in capture-lane upscalers -------------------------------------------------
    b!("builtin.nearest", "Nearest", "built-in", Upscaler, Capture, Builtin, GpuReq::Any, "Pixel-exact integer-style scaling; best for pixel art."),
    b!("builtin.bilinear", "Bilinear", "built-in", Upscaler, Capture, Builtin, GpuReq::Any, "Cheapest smooth scaling; soft."),
    b!("builtin.catmull_rom", "Bicubic (Catmull-Rom)", "built-in", Upscaler, Capture, Builtin, GpuReq::Any, "Sharper than bilinear, mild ringing."),
    b!("builtin.lanczos2", "Lanczos 2", "built-in", Upscaler, Capture, Builtin, GpuReq::Any, "Good all-rounder; anti-ringing available."),
    b!("builtin.lanczos3", "Lanczos 3", "built-in", Upscaler, Capture, Builtin, GpuReq::Any, "Sharpest classic kernel; most ringing."),
    b!("builtin.edge_adaptive", "Edge-adaptive", "built-in", Upscaler, Capture, Builtin, GpuReq::Any, "Direction-aware kernel in the spirit of FSR 1's EASU. Original implementation, not AMD's shader."),
    // ---- Built-in capture-lane frame generators ------------------------------------------
    b!("builtin.blend", "Blend", "built-in", FrameGen, Capture, Builtin, GpuReq::Any, "Cross-fade. Baseline only; ghosts on motion."),
    b!("builtin.optical_flow", "Optical flow", "built-in", FrameGen, Capture, Builtin, GpuReq::Any, "Block-matching flow + motion-compensated warp. No game motion vectors, so artifacts around HUD and thin geometry are expected."),
    // ---- Injection lane: need engine data; not implemented ----------------------------------
    b!("nvidia.dlss_sr", "DLSS Super Resolution", "NVIDIA", Upscaler, Injection, NeedsUserFiles, GpuReq::NvidiaRtx, "Needs the game's motion vectors + depth, so it cannot run on captured frames. RTX only. NVIDIA's DLL is not redistributable here."),
    b!("nvidia.dlss_fg", "DLSS Frame Generation", "NVIDIA", FrameGen, Injection, NeedsUserFiles, GpuReq::NvidiaRtx, "Same constraints as DLSS SR; RTX 40+ for FG."),
    b!("amd.fsr1", "FSR 1 (EASU + RCAS)", "AMD", Upscaler, Capture, Builtin, GpuReq::Any, "Port of AMD FidelityFX FSR 1 (MIT). EASU upscale plus RCAS sharpen, which replaces the CAS slider for this upscaler."),
    b!("amd.fsr2", "FSR 2 (temporal)", "AMD", Upscaler, Injection, Planned, GpuReq::Any, "Needs game motion vectors; injection lane."),
    b!("amd.fsr3", "FSR 3 / FSR 3.1", "AMD", Upscaler, Injection, Planned, GpuReq::Any, "Temporal upscale + frame generation via game integration."),
    b!("amd.fsr4", "FSR 4 / 4.1 (INT8 / FP8)", "AMD", Upscaler, Injection, NeedsUserFiles, GpuReq::AnyFasterOn("RDNA 4 (FP8)"), "AMD reportedly ships an official INT8 path for RDNA 3. Needs the game's inputs; injection lane with user-supplied DLL."),
    b!("amd.fsr_fg", "FSR Frame Generation / Redstone ML FG", "AMD", FrameGen, Injection, NeedsUserFiles, GpuReq::AnyFasterOn("RDNA 4"), "Needs game integration. Reportedly officially RDNA 4 only."),
    b!("amd.afmf", "AMD Fluid Motion Frames (driver)", "AMD", FrameGen, External, ExternalToggle, GpuReq::Amd, "Driver-level frame gen. Toggle in Adrenalin; this app cannot control it."),
    b!("intel.xess", "XeSS", "Intel", Upscaler, Injection, NeedsUserFiles, GpuReq::AnyFasterOn("Intel Arc (XMX)"), "Needs game motion vectors; injection lane."),
    b!("nvidia.nis", "NVIDIA Image Scaling", "NVIDIA", Upscaler, Capture, Planned, GpuReq::Any, "Spatial; MIT-licensed shader could be ported."),
    b!("losslessscaling.lsfg", "Lossless Scaling (LSFG / LS1)", "Lossless Scaling", FrameGen, External, ExternalToggle, GpuReq::Any, "Proprietary separate app. Run it alongside if you own it; do not run both on the same window."),
    b!("magpie.shaders", "Magpie-style shader packs", "community", Upscaler, Capture, Planned, GpuReq::Any, "Custom WGSL/HLSL shader loading is a planned plugin."),
];

pub fn by_id(id: &str) -> Option<&'static Backend> {
    BACKENDS.iter().find(|b| b.id == id)
}

pub fn of_kind(kind: Kind) -> impl Iterator<Item = &'static Backend> {
    BACKENDS.iter().filter(move |b| b.kind == kind)
}

/// Map a built-in upscaler id to its typed kernel.
pub fn upscaler_for_id(id: &str) -> Option<crate::config::UpscalerKind> {
    use crate::config::UpscalerKind::*;
    Some(match id {
        "builtin.nearest" => Nearest,
        "builtin.bilinear" => Bilinear,
        "builtin.catmull_rom" => CatmullRom,
        "builtin.lanczos2" => Lanczos2,
        "builtin.lanczos3" => Lanczos3,
        "builtin.edge_adaptive" => EdgeAdaptive,
        "amd.fsr1" => Fsr1,
        _ => return None,
    })
}

pub fn framegen_for_id(id: &str) -> Option<crate::config::FrameGenKind> {
    use crate::config::FrameGenKind::*;
    Some(match id {
        "builtin.blend" => Blend,
        "builtin.optical_flow" => OpticalFlow,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn ids_unique() {
        let mut seen = HashSet::new();
        for b in BACKENDS {
            assert!(seen.insert(b.id), "duplicate {}", b.id);
        }
    }

    #[test]
    fn every_builtin_maps_to_a_typed_option() {
        for b in BACKENDS.iter().filter(|b| b.status == Status::Builtin) {
            match b.kind {
                Kind::Upscaler => assert!(upscaler_for_id(b.id).is_some(), "{}", b.id),
                Kind::FrameGen => assert!(framegen_for_id(b.id).is_some(), "{}", b.id),
            }
        }
    }

    #[test]
    fn default_profile_backends_exist_and_are_selectable() {
        let p = crate::config::Profile::default();
        assert!(by_id(&p.upscaler_backend).unwrap().selectable());
        assert!(by_id(&p.framegen_backend).unwrap().selectable());
    }

    #[test]
    fn temporal_methods_are_never_capture_lane() {
        for id in [
            "nvidia.dlss_sr",
            "amd.fsr2",
            "amd.fsr3",
            "amd.fsr4",
            "intel.xess",
        ] {
            assert_ne!(by_id(id).unwrap().lane, Lane::Capture, "{id}");
        }
    }
}
