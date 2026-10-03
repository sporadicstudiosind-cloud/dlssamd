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
// EASU: Edge-Adaptive Spatial Upsampling. One thread per output pixel.

struct Params {
    in_size: vec2<f32>,
    out_size: vec2<f32>,
    mode: u32,
    anti_ring: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba8unorm, write>;

fn prx_lo_rcp(a: f32) -> f32 { return bitcast<f32>(0x7ef07ebbu - bitcast<u32>(a)); }
fn prx_lo_rsq(a: f32) -> f32 { return bitcast<f32>(0x5f347d74u - (bitcast<u32>(a) >> 1u)); }
fn sat1(a: f32) -> f32 { return clamp(a, 0.0, 1.0); }

fn tex(p: vec2<i32>) -> vec3<f32> {
    let c = clamp(p, vec2<i32>(0, 0), vec2<i32>(params.in_size) - vec2<i32>(1, 1));
    return textureLoad(src, c, 0).rgb;
}

fn luma2(c: vec3<f32>) -> f32 { return c.b * 0.5 + (c.r * 0.5 + c.g); }

// Filtering for a given tap.
fn easu_tap(aC: ptr<function, vec3<f32>>, aW: ptr<function, f32>, off: vec2<f32>, dir: vec2<f32>,
            len: vec2<f32>, lob: f32, clp: f32, c: vec3<f32>) {
    var v: vec2<f32>;
    v.x = off.x * dir.x + off.y * dir.y;
    v.y = off.x * (-dir.y) + off.y * dir.x;
    v = v * len;
    var d2 = v.x * v.x + v.y * v.y;
    d2 = min(d2, clp);
    var wB = 2.0 / 5.0 * d2 - 1.0;
    var wA = lob * d2 - 1.0;
    wB = wB * wB;
    wA = wA * wA;
    wB = 25.0 / 16.0 * wB - (25.0 / 16.0 - 1.0);
    let w = wB * wA;
    *aC = *aC + c * w;
    *aW = *aW + w;
}

// Accumulate direction and length. (s t / u v) selects which bilinear corner this is.
fn easu_set(dir: ptr<function, vec2<f32>>, len: ptr<function, f32>, pp: vec2<f32>,
            bi_s: bool, bi_t: bool, bi_u: bool, bi_v: bool,
            lA: f32, lB: f32, lC: f32, lD: f32, lE: f32) {
    var w = 0.0;
    if (bi_s) { w = (1.0 - pp.x) * (1.0 - pp.y); }
    if (bi_t) { w = pp.x * (1.0 - pp.y); }
    if (bi_u) { w = (1.0 - pp.x) * pp.y; }
    if (bi_v) { w = pp.x * pp.y; }
    let dc = lD - lC;
    let cb = lC - lB;
    var lenX = max(abs(dc), abs(cb));
    lenX = prx_lo_rcp(lenX);
    let dirX = lD - lB;
    (*dir).x = (*dir).x + dirX * w;
    lenX = sat1(abs(dirX) * lenX);
    lenX = lenX * lenX;
    *len = *len + lenX * w;
    let ec = lE - lC;
    let ca = lC - lA;
    var lenY = max(abs(ec), abs(ca));
    lenY = prx_lo_rcp(lenY);
    let dirY = lE - lA;
    (*dir).y = (*dir).y + dirY * w;
    lenY = sat1(abs(dirY) * lenY);
    lenY = lenY * lenY;
    *len = *len + lenY * w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_px = vec2<u32>(params.out_size);
    if (gid.x >= out_px.x || gid.y >= out_px.y) { return; }

    // FsrEasuCon: output pixel -> input position.
    let scale = params.in_size / params.out_size;
    var pp = vec2<f32>(gid.xy) * scale + (0.5 * scale - vec2<f32>(0.5, 0.5));
    let fp = floor(pp);
    pp = pp - fp;
    let f0 = vec2<i32>(fp);

    // 12-tap kernel:    b c
    //                 e f g h
    //                 i j k l
    //                   n o
    let cb_ = tex(f0 + vec2<i32>(0, -1));   // b
    let cc_ = tex(f0 + vec2<i32>(1, -1));   // c
    let ce_ = tex(f0 + vec2<i32>(-1, 0));   // e
    let cf_ = tex(f0);                      // f
    let cg_ = tex(f0 + vec2<i32>(1, 0));    // g
    let ch_ = tex(f0 + vec2<i32>(2, 0));    // h
    let ci_ = tex(f0 + vec2<i32>(-1, 1));   // i
    let cj_ = tex(f0 + vec2<i32>(0, 1));    // j
    let ck_ = tex(f0 + vec2<i32>(1, 1));    // k
    let cl_ = tex(f0 + vec2<i32>(2, 1));    // l
    let cn_ = tex(f0 + vec2<i32>(0, 2));    // n
    let co_ = tex(f0 + vec2<i32>(1, 2));    // o

    let bL = luma2(cb_); let cL = luma2(cc_); let iL = luma2(ci_); let jL = luma2(cj_);
    let fL = luma2(cf_); let eL = luma2(ce_); let kL = luma2(ck_); let lL = luma2(cl_);
    let hL = luma2(ch_); let gL = luma2(cg_); let oL = luma2(co_); let nL = luma2(cn_);

    var dir = vec2<f32>(0.0, 0.0);
    var len = 0.0;
    easu_set(&dir, &len, pp, true,  false, false, false, bL, eL, fL, gL, jL);
    easu_set(&dir, &len, pp, false, true,  false, false, cL, fL, gL, hL, kL);
    easu_set(&dir, &len, pp, false, false, true,  false, fL, iL, jL, kL, nL);
    easu_set(&dir, &len, pp, false, false, false, true,  gL, jL, kL, lL, oL);

    // Normalize with approximation, and cleanup close to zero.
    let dir2 = dir * dir;
    var dirR = dir2.x + dir2.y;
    let zro = dirR < 1.0 / 32768.0;
    dirR = prx_lo_rsq(dirR);
    if (zro) { dirR = 1.0; dir.x = 1.0; }
    dir = dir * dirR;
    len = len * 0.5;
    len = len * len;
    let stretch = (dir.x * dir.x + dir.y * dir.y) * prx_lo_rcp(max(abs(dir.x), abs(dir.y)));
    let len2 = vec2<f32>(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
    let lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
    let clp = prx_lo_rcp(lob);

    // Min/max of the 4 nearest, for de-ringing.
    let min4 = min(min(min(cf_, ck_), min(cg_, cj_)), vec3<f32>(1e9));
    let max4 = max(max(max(cf_, ck_), max(cg_, cj_)), vec3<f32>(-1e9));

    var aC = vec3<f32>(0.0);
    var aW = 0.0;
    easu_tap(&aC, &aW, vec2<f32>( 0.0, -1.0) - pp, dir, len2, lob, clp, cb_);
    easu_tap(&aC, &aW, vec2<f32>( 1.0, -1.0) - pp, dir, len2, lob, clp, cc_);
    easu_tap(&aC, &aW, vec2<f32>(-1.0,  1.0) - pp, dir, len2, lob, clp, ci_);
    easu_tap(&aC, &aW, vec2<f32>( 0.0,  1.0) - pp, dir, len2, lob, clp, cj_);
    easu_tap(&aC, &aW, vec2<f32>( 0.0,  0.0) - pp, dir, len2, lob, clp, cf_);
    easu_tap(&aC, &aW, vec2<f32>(-1.0,  0.0) - pp, dir, len2, lob, clp, ce_);
    easu_tap(&aC, &aW, vec2<f32>( 1.0,  1.0) - pp, dir, len2, lob, clp, ck_);
    easu_tap(&aC, &aW, vec2<f32>( 2.0,  1.0) - pp, dir, len2, lob, clp, cl_);
    easu_tap(&aC, &aW, vec2<f32>( 2.0,  0.0) - pp, dir, len2, lob, clp, ch_);
    easu_tap(&aC, &aW, vec2<f32>( 1.0,  0.0) - pp, dir, len2, lob, clp, cg_);
    easu_tap(&aC, &aW, vec2<f32>( 1.0,  2.0) - pp, dir, len2, lob, clp, co_);
    easu_tap(&aC, &aW, vec2<f32>( 0.0,  2.0) - pp, dir, len2, lob, clp, cn_);

    let pix = min(max4, max(min4, aC * (1.0 / aW)));
    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(clamp(pix, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
