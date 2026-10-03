//! Core of the capture-lane upscaler / frame generator: settings, backend catalog,
//! pacing maths and the GPU pipeline. No windowing or capture code lives here.

pub mod catalog;
pub mod config;
pub mod gpu;
pub mod nis;
pub mod nis_coefs;
pub mod pacing;
pub mod shaders;
