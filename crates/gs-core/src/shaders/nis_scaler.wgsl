// Port of NVIDIA Image Scaling SDK v1.0.3 (NVScaler), https://github.com/NVIDIAGameWorks/NVIDIAImageScaling
// Original: The MIT License (MIT), Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions: The above copyright notice and this
// permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
// OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// Translated from HLSL to WGSL for this project (SDR path, no viewport/NV12/HDR support). The
// original stages luma into group-shared memory tiles for speed; this port computes each output
// pixel independently from a 6x6 luma patch (same maths, more redundant loads, simpler).
// Coefficient filter banks arrive in a uniform buffer: [phase*2 + (tap>>2)][tap&3].

struct Cfg {
    k_detect_ratio: f32,
    k_detect_thres: f32,
    k_min_contrast_ratio: f32,
    k_ratio_norm: f32,
    k_contrast_boost: f32,
    k_eps: f32,
    k_sharp_start_y: f32,
    k_sharp_scale_y: f32,
    k_sharp_strength_min: f32,
    k_sharp_strength_scale: f32,
    k_sharp_limit_min: f32,
    k_sharp_limit_scale: f32,
    k_scale_x: f32,
    k_scale_y: f32,
    in_w: f32,
    in_h: f32,
    out_w: f32,
    out_h: f32,
    _p0: f32,
    _p1: f32,
};

@group(0) @binding(0) var<uniform> cfg: Cfg;
@group(0) @binding(1) var<uniform> coef_scale: array<vec4<f32>, 128>;
@group(0) @binding(2) var<uniform> coef_usm: array<vec4<f32>, 128>;
@group(0) @binding(3) var src: texture_2d<f32>;
@group(0) @binding(4) var dst: texture_storage_2d<rgba8unorm, write>;

const K_PHASE_COUNT: i32 = 64;

// 6x6 luma support, indexed [row (y)][col (x)], as in the original.
var<private> P: array<array<f32, 6>, 6>;

fn get_y(c: vec3<f32>) -> f32 {
    return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
}

fn load_rgb(p: vec2<i32>) -> vec3<f32> {
    let c = clamp(p, vec2<i32>(0, 0), vec2<i32>(i32(cfg.in_w), i32(cfg.in_h)) - vec2<i32>(1, 1));
    return textureLoad(src, c, 0).rgb;
}

fn sat(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

fn coef_s(phase: i32, tap: i32) -> f32 {
    let v = coef_scale[phase * 2 + (tap >> 2)];
    return v[tap & 3];
}
fn coef_u(phase: i32, tap: i32) -> f32 {
    let v = coef_usm[phase * 2 + (tap >> 2)];
    return v[tap & 3];
}

// Edge weights (0deg, 90deg, 45deg, 135deg) for the 3x3 luma window whose top-left is
// P[r0][c0]. (The original's GetEdgeMap(p, i, j) with offsets i = r0, j = c0.)
fn get_edge_map(r0: i32, c0: i32) -> vec4<f32> {
    let g_0 = abs(P[r0][c0] + P[r0][c0 + 1] + P[r0][c0 + 2] - P[r0 + 2][c0] - P[r0 + 2][c0 + 1] - P[r0 + 2][c0 + 2]);
    let g_45 = abs(P[r0 + 1][c0] + P[r0][c0] + P[r0][c0 + 1] - P[r0 + 2][c0 + 1] - P[r0 + 2][c0 + 2] - P[r0 + 1][c0 + 2]);
    let g_90 = abs(P[r0][c0] + P[r0 + 1][c0] + P[r0 + 2][c0] - P[r0][c0 + 2] - P[r0 + 1][c0 + 2] - P[r0 + 2][c0 + 2]);
    let g_135 = abs(P[r0 + 1][c0] + P[r0 + 2][c0] + P[r0 + 2][c0 + 1] - P[r0][c0 + 1] - P[r0][c0 + 2] - P[r0 + 1][c0 + 2]);

    let g_0_90_max = max(g_0, g_90);
    let g_0_90_min = min(g_0, g_90);
    let g_45_135_max = max(g_45, g_135);
    let g_45_135_min = min(g_45, g_135);

    if (g_0_90_max + g_45_135_max == 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let e_0_90 = min(g_0_90_max / (g_0_90_max + g_45_135_max), 1.0);
    let e_45_135 = 1.0 - e_0_90;

    let c_0_90 = (g_0_90_max > (g_0_90_min * cfg.k_detect_ratio)) && (g_0_90_max > cfg.k_detect_thres) && (g_0_90_max > g_45_135_min);
    let c_45_135 = (g_45_135_max > (g_45_135_min * cfg.k_detect_ratio)) && (g_45_135_max > cfg.k_detect_thres) && (g_45_135_max > g_0_90_min);
    let c_g_0_90 = g_0_90_max == g_0;
    let c_g_45_135 = g_45_135_max == g_45;

    var f_e_0_90 = 1.0;
    var f_e_45_135 = 1.0;
    if (c_0_90 && c_45_135) {
        f_e_0_90 = e_0_90;
        f_e_45_135 = e_45_135;
    }

    var weight_0 = 0.0;
    var weight_90 = 0.0;
    var weight_45 = 0.0;
    var weight_135 = 0.0;
    if (c_0_90 && c_g_0_90) { weight_0 = f_e_0_90; }
    if (c_0_90 && !c_g_0_90) { weight_90 = f_e_0_90; }
    if (c_45_135 && c_g_45_135) { weight_45 = f_e_45_135; }
    if (c_45_135 && !c_g_45_135) { weight_135 = f_e_45_135; }
    return vec4<f32>(weight_0, weight_90, weight_45, weight_135);
}

fn calc_lti(p0: f32, p1: f32, p2: f32, p3: f32, p4: f32, p5: f32, phase_index: i32) -> f32 {
    let selector = phase_index <= K_PHASE_COUNT / 2;
    var sel = p3;
    if (selector) { sel = p0; }
    let a_min = min(min(p1, p2), sel);
    let a_max = max(max(p1, p2), sel);
    sel = p5;
    if (selector) { sel = p2; }
    let b_min = min(min(p3, p4), sel);
    let b_max = max(max(p3, p4), sel);

    let a_cont = a_max - a_min;
    let b_cont = b_max - b_min;

    let cont_ratio = max(a_cont, b_cont) / (min(a_cont, b_cont) + cfg.k_eps);
    return (1.0 - sat((cont_ratio - cfg.k_min_contrast_ratio) * cfg.k_ratio_norm)) * cfg.k_contrast_boost;
}

fn eval_poly6(pxl: array<f32, 6>, phase_int: i32) -> f32 {
    var y = 0.0;
    for (var i = 0; i < 6; i = i + 1) {
        y = y + coef_s(phase_int, i) * pxl[i];
    }
    var y_usm = 0.0;
    for (var i = 0; i < 6; i = i + 1) {
        y_usm = y_usm + coef_u(phase_int, i) * pxl[i];
    }

    // piece-wise ramp based on luma
    let y_scale = 1.0 - sat((y - cfg.k_sharp_start_y) * cfg.k_sharp_scale_y);
    // sharpen as a function of luma
    let y_sharpness = y_scale * cfg.k_sharp_strength_scale + cfg.k_sharp_strength_min;
    y_usm = y_usm * y_sharpness;
    // limit USM as a function of luma
    let y_sharpness_limit = (y_scale * cfg.k_sharp_limit_scale + cfg.k_sharp_limit_min) * y;
    y_usm = min(y_sharpness_limit, max(-y_sharpness_limit, y_usm));
    // reduce ringing
    y_usm = y_usm * calc_lti(pxl[0], pxl[1], pxl[2], pxl[3], pxl[4], pxl[5], phase_int);
    return y + y_usm;
}

fn filter_normal(px: i32, py: i32) -> f32 {
    var h_acc = 0.0;
    for (var j = 0; j < 6; j = j + 1) {
        var v_acc = 0.0;
        for (var i = 0; i < 6; i = i + 1) {
            v_acc = v_acc + P[i][j] * coef_s(py, i);
        }
        h_acc = h_acc + v_acc * coef_s(px, j);
    }
    return h_acc;
}

fn add_dir_filters(phase_x_frac: f32, phase_y_frac: f32, px_int: i32, py_int: i32, w: vec4<f32>) -> f32 {
    var f = 0.0;
    if (w.x > 0.0) {
        // 0 deg filter
        var interp: array<f32, 6>;
        for (var i = 0; i < 6; i = i + 1) {
            interp[i] = mix(P[i][2], P[i][3], phase_x_frac);
        }
        f = f + eval_poly6(interp, py_int) * w.x;
    }
    if (w.y > 0.0) {
        // 90 deg filter
        var interp: array<f32, 6>;
        for (var i = 0; i < 6; i = i + 1) {
            interp[i] = mix(P[2][i], P[3][i], phase_y_frac);
        }
        f = f + eval_poly6(interp, px_int) * w.y;
    }
    if (w.z > 0.0) {
        // 45 deg filter
        var pphase_b45 = 0.5 + 0.5 * (phase_x_frac - phase_y_frac);
        var t: array<f32, 7>;
        t[1] = mix(P[2][1], P[1][2], pphase_b45);
        t[3] = mix(P[3][2], P[2][3], pphase_b45);
        t[5] = mix(P[4][3], P[3][4], pphase_b45);
        pphase_b45 = pphase_b45 - 0.5;
        var a = P[2][0];
        var b = P[3][1];
        var c = P[4][2];
        var d = P[5][3];
        if (pphase_b45 >= 0.0) {
            a = P[0][2];
            b = P[1][3];
            c = P[2][4];
            d = P[3][5];
        }
        t[0] = mix(P[1][1], a, abs(pphase_b45));
        t[2] = mix(P[2][2], b, abs(pphase_b45));
        t[4] = mix(P[3][3], c, abs(pphase_b45));
        t[6] = mix(P[4][4], d, abs(pphase_b45));

        var interp: array<f32, 6>;
        var pphase_p45 = phase_x_frac + phase_y_frac;
        if (pphase_p45 >= 1.0) {
            for (var i = 0; i < 6; i = i + 1) { interp[i] = t[i + 1]; }
            pphase_p45 = pphase_p45 - 1.0;
        } else {
            for (var i = 0; i < 6; i = i + 1) { interp[i] = t[i]; }
        }
        f = f + eval_poly6(interp, min(i32(pphase_p45 * 64.0), 63)) * w.z;
    }
    if (w.w > 0.0) {
        // 135 deg filter
        var pphase_b135 = 0.5 * (phase_x_frac + phase_y_frac);
        var t: array<f32, 7>;
        t[1] = mix(P[3][1], P[4][2], pphase_b135);
        t[3] = mix(P[2][2], P[3][3], pphase_b135);
        t[5] = mix(P[1][3], P[2][4], pphase_b135);
        pphase_b135 = pphase_b135 - 0.5;
        var a = P[3][0];
        var b = P[2][1];
        var c = P[1][2];
        var d = P[0][3];
        if (pphase_b135 >= 0.0) {
            a = P[5][2];
            b = P[4][3];
            c = P[3][4];
            d = P[2][5];
        }
        t[0] = mix(P[4][1], a, abs(pphase_b135));
        t[2] = mix(P[3][2], b, abs(pphase_b135));
        t[4] = mix(P[2][3], c, abs(pphase_b135));
        t[6] = mix(P[1][4], d, abs(pphase_b135));

        var interp: array<f32, 6>;
        var pphase_p135 = 1.0 + (phase_x_frac - phase_y_frac);
        if (pphase_p135 >= 1.0) {
            for (var i = 0; i < 6; i = i + 1) { interp[i] = t[i + 1]; }
            pphase_p135 = pphase_p135 - 1.0;
        } else {
            for (var i = 0; i < 6; i = i + 1) { interp[i] = t[i]; }
        }
        f = f + eval_poly6(interp, min(i32(pphase_p135 * 64.0), 63)) * w.w;
    }
    return f;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u32(cfg.out_w) || gid.y >= u32(cfg.out_h)) { return; }

    let src_x = (0.5 + f32(gid.x)) * cfg.k_scale_x - 0.5;
    let src_y = (0.5 + f32(gid.y)) * cfg.k_scale_y - 0.5;
    let fl = vec2<i32>(i32(floor(src_x)), i32(floor(src_y)));
    let fx = src_x - floor(src_x);
    let fy = src_y - floor(src_y);
    let fx_int = i32(fx * f32(K_PHASE_COUNT));
    let fy_int = i32(fy * f32(K_PHASE_COUNT));

    // 6x6 luma support: P[i][j] is the source pixel (floor_x - 2 + j, floor_y - 2 + i).
    for (var i = 0; i < 6; i = i + 1) {
        for (var j = 0; j < 6; j = j + 1) {
            P[i][j] = get_y(load_rgb(fl + vec2<i32>(j - 2, i - 2)));
        }
    }

    // Edge weights at the 2x2 source pixels (floor + {0,1}), interpolated by the sub-pixel offset.
    // edge[i][j] is centred on patch pixel (row 2+i, col 2+j) -> 3x3 window starting at (1+i, 1+j).
    let e00 = get_edge_map(1, 1);
    let e01 = get_edge_map(1, 2);
    let e10 = get_edge_map(2, 1);
    let e11 = get_edge_map(2, 2);
    let h0 = mix(e00, e01, fx);
    let h1 = mix(e10, e11, fx);
    let w = mix(h0, h1, fy);

    let base_weight = 1.0 - w.x - w.y - w.z - w.w;
    var op_y = 0.0;
    op_y = op_y + filter_normal(fx_int, fy_int) * base_weight;
    op_y = op_y + add_dir_filters(fx, fy, fx_int, fy_int, w);

    // Bilinear tap for chroma.
    let a = load_rgb(fl);
    let b = load_rgb(fl + vec2<i32>(1, 0));
    let c = load_rgb(fl + vec2<i32>(0, 1));
    let d = load_rgb(fl + vec2<i32>(1, 1));
    var rgb = mix(mix(a, b, fx), mix(c, d, fx), fy);
    let y = get_y(rgb);
    let corr = op_y - y;
    rgb = rgb + vec3<f32>(corr, corr, corr);
    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
