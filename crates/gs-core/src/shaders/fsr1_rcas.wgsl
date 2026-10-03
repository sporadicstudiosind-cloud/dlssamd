// Port of AMD FidelityFX Super Resolution 1 (FSR 1) v1.20210629, https://github.com/GPUOpen-Effects/FidelityFX-FSR
// Original: Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
// Distributed under the MIT License:
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
// Translated from C/HLSL to WGSL for this project. gather4 is replaced by direct texel loads
// (same texels, same maths); everything else follows the original float (non-packed) path.
//
// RCAS: Robust Contrast-Adaptive Sharpening. One thread per pixel. `strength` 0..1 from the
// UI maps to RCAS "stops" (0 = maximum sharpness): stops = (1 - strength) * 2.

struct Params {
    size: vec2<f32>,
    strength: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba8unorm, write>;

const FSR_RCAS_LIMIT: f32 = 0.1875;   // 0.25 - 1/16

fn prx_med_rcp(a: f32) -> f32 {
    let b = bitcast<f32>(0x7ef19fffu - bitcast<u32>(a));
    return b * (-b * a + 2.0);
}
fn sat1(a: f32) -> f32 { return clamp(a, 0.0, 1.0); }
fn max3(a: f32, b: f32, c: f32) -> f32 { return max(a, max(b, c)); }
fn min3(a: f32, b: f32, c: f32) -> f32 { return min(a, min(b, c)); }
fn min3v(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> { return min(a, min(b, c)); }
fn max3v(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> { return max(a, max(b, c)); }

fn ld(p: vec2<i32>) -> vec3<f32> {
    let c = clamp(p, vec2<i32>(0, 0), vec2<i32>(params.size) - vec2<i32>(1, 1));
    return textureLoad(src, c, 0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<u32>(params.size);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let sp = vec2<i32>(gid.xy);
    //    b
    //  d e f
    //    h
    let b = ld(sp + vec2<i32>(0, -1));
    let d = ld(sp + vec2<i32>(-1, 0));
    let e = ld(sp);
    let f = ld(sp + vec2<i32>(1, 0));
    let h = ld(sp + vec2<i32>(0, 1));

    let sharp = exp2(-((1.0 - clamp(params.strength, 0.0, 1.0)) * 2.0));

    // Luma times 2.
    let bL = b.b * 0.5 + (b.r * 0.5 + b.g);
    let dL = d.b * 0.5 + (d.r * 0.5 + d.g);
    let eL = e.b * 0.5 + (e.r * 0.5 + e.g);
    let fL = f.b * 0.5 + (f.r * 0.5 + f.g);
    let hL = h.b * 0.5 + (h.r * 0.5 + h.g);

    // Noise detection (FSR_RCAS_DENOISE path: reduces sharpening on noisy content).
    var nz = 0.25 * bL + 0.25 * dL + 0.25 * fL + 0.25 * hL - eL;
    nz = sat1(abs(nz) * prx_med_rcp(max3(max3(bL, dL, eL), fL, hL) - min3(min3(bL, dL, eL), fL, hL)));
    nz = -0.5 * nz + 1.0;

    // Min and max of ring.
    let mn4 = min(min3v(b, d, f), h);
    let mx4 = max(max3v(b, d, f), h);

    let peakC = vec2<f32>(1.0, -1.0 * 4.0);
    let hitMin = min(mn4, e) * (vec3<f32>(1.0) / (4.0 * mx4));
    let hitMax = (vec3<f32>(peakC.x) - max(mx4, e)) * (vec3<f32>(1.0) / (4.0 * mn4 + vec3<f32>(peakC.y)));
    let lobe3 = max(-hitMin, hitMax);
    var lobe = max(-FSR_RCAS_LIMIT, min(max3(lobe3.r, lobe3.g, lobe3.b), 0.0)) * sharp;
    lobe = lobe * nz;

    let rcpL = prx_med_rcp(4.0 * lobe + 1.0);
    let pix = (lobe * b + lobe * d + lobe * h + lobe * f + e) * rcpL;
    textureStore(dst, sp, vec4<f32>(clamp(pix, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
