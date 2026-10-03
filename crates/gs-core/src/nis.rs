//! NVIDIA Image Scaling configuration (port of `NVScalerUpdateConfig`, SDR path).
//!
//! Original: The MIT License (MIT), Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//! See THIRD_PARTY_LICENSES.md.

use bytemuck::{Pod, Zeroable};

/// Uniform block for `nis_scaler.wgsl` (20 floats).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub struct NisConfig {
    pub detect_ratio: f32,
    pub detect_thres: f32,
    pub min_contrast_ratio: f32,
    pub ratio_norm: f32,
    pub contrast_boost: f32,
    pub eps: f32,
    pub sharp_start_y: f32,
    pub sharp_scale_y: f32,
    pub sharp_strength_min: f32,
    pub sharp_strength_scale: f32,
    pub sharp_limit_min: f32,
    pub sharp_limit_scale: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub in_w: f32,
    pub in_h: f32,
    pub out_w: f32,
    pub out_h: f32,
    pub _pad: [f32; 2],
}

/// NIS only supports upscaling by 1x to 2x per axis (`kScale` must be within 0.5..=1.0).
pub const MAX_SCALE: f32 = 2.0;

/// Returns `None` if the sizes are outside what NIS supports (same rule as the original).
pub fn scaler_config(sharpness: f32, input: (u32, u32), output: (u32, u32)) -> Option<NisConfig> {
    if input.0 == 0 || input.1 == 0 || output.0 == 0 || output.1 == 0 {
        return None;
    }
    let sharpness = sharpness.clamp(0.0, 1.0);
    let sharpen_slider = sharpness - 0.5; // 0..1 -> -0.5..+0.5

    // Different range for 0-50% vs 50-100%: 0% maps to no sharpening while 100% doesn't over-sharpen.
    let pos = sharpen_slider >= 0.0;
    let max_scale = if pos { 1.25 } else { 1.75 };
    let min_scale = if pos { 1.25 } else { 1.0 };
    let limit_scale = if pos { 1.25 } else { 1.0 };

    let detect_ratio = 2.0 * 1127.0 / 1024.0;
    let detect_thres = 64.0 / 1024.0;
    let min_contrast_ratio = 2.0;
    let max_contrast_ratio = 10.0;
    let sharp_start_y = 0.45;
    let sharp_end_y = 0.9;
    let sharp_strength_min = f32::max(0.0, 0.4 + sharpen_slider * min_scale * 1.2);
    let sharp_strength_max = 1.6 + sharpen_slider * max_scale * 1.8;
    let sharp_limit_min = f32::max(0.1, 0.14 + sharpen_slider * limit_scale * 0.32);
    let sharp_limit_max = 0.5 + sharpen_slider * limit_scale * 0.6;

    let scale_x = input.0 as f32 / output.0 as f32;
    let scale_y = input.1 as f32 / output.1 as f32;
    if !(0.5..=1.0).contains(&scale_x) || !(0.5..=1.0).contains(&scale_y) {
        return None;
    }

    Some(NisConfig {
        detect_ratio,
        detect_thres,
        min_contrast_ratio,
        ratio_norm: 1.0 / (max_contrast_ratio - min_contrast_ratio),
        contrast_boost: 1.0,
        eps: 1.0 / 255.0,
        sharp_start_y,
        sharp_scale_y: 1.0 / (sharp_end_y - sharp_start_y),
        sharp_strength_min,
        sharp_strength_scale: sharp_strength_max - sharp_strength_min,
        sharp_limit_min,
        sharp_limit_scale: sharp_limit_max - sharp_limit_min,
        scale_x,
        scale_y,
        in_w: input.0 as f32,
        in_h: input.1 as f32,
        out_w: output.0 as f32,
        out_h: output.1 as f32,
        _pad: [0.0; 2],
    })
}

/// Coefficient banks packed for the shader: `[phase * 2 + (tap >> 2)]` -> vec4.
pub fn packed_coefs(table: &[[f32; 8]; 64]) -> Vec<f32> {
    table.iter().flat_map(|row| row.iter().copied()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nis_coefs::{COEF_SCALE, COEF_USM};

    #[test]
    fn midpoint_sharpness_matches_hand_calculation() {
        // sharpness 0.5 -> slider 0: strength 0.4..1.6, limit 0.14..0.5.
        let c = scaler_config(0.5, (1280, 720), (1920, 1080)).unwrap();
        assert!((c.sharp_strength_min - 0.4).abs() < 1e-6);
        assert!((c.sharp_strength_scale - 1.2).abs() < 1e-6);
        assert!((c.sharp_limit_min - 0.14).abs() < 1e-6);
        assert!((c.sharp_limit_scale - 0.36).abs() < 1e-6);
        assert!((c.scale_x - 1280.0 / 1920.0).abs() < 1e-6);
        assert!((c.ratio_norm - 0.125).abs() < 1e-6);
    }

    #[test]
    fn zero_sharpness_has_lowest_strength() {
        let lo = scaler_config(0.0, (100, 100), (150, 150)).unwrap();
        let hi = scaler_config(1.0, (100, 100), (150, 150)).unwrap();
        assert!(lo.sharp_strength_min < hi.sharp_strength_min);
        assert!(lo.sharp_limit_min < hi.sharp_limit_min);
    }

    #[test]
    fn rejects_scales_nis_cannot_do() {
        assert!(
            scaler_config(0.5, (100, 100), (201, 200)).is_none(),
            ">2x must be refused"
        );
        assert!(
            scaler_config(0.5, (100, 100), (90, 90)).is_none(),
            "downscale must be refused"
        );
        assert!(
            scaler_config(0.5, (100, 100), (200, 200)).is_some(),
            "exactly 2x is fine"
        );
    }

    #[test]
    fn config_block_is_80_bytes() {
        assert_eq!(std::mem::size_of::<NisConfig>(), 80);
    }

    #[test]
    fn coefficient_tables_are_sane() {
        // Phase 0 of the scaler bank is the identity tap (1.0 at index 2); every phase's
        // scaler taps sum to ~1 (a normalised interpolation kernel), USM taps sum to ~0.
        assert_eq!(COEF_SCALE[0][2], 1.0);
        for (p, row) in COEF_SCALE.iter().enumerate() {
            let s: f32 = row[..6].iter().sum();
            assert!((s - 1.0).abs() < 0.01, "scale phase {p} sums to {s}");
        }
        for (p, row) in COEF_USM.iter().enumerate() {
            let s: f32 = row[..6].iter().sum();
            assert!(s.abs() < 0.02, "usm phase {p} sums to {s}");
        }
        assert_eq!(packed_coefs(&COEF_SCALE).len(), 512);
    }
}
