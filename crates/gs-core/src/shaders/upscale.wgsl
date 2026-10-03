// Spatial upscaler. One thread per output pixel.
// mode: 0 nearest, 1 bilinear, 2 catmull-rom, 3 lanczos2, 4 lanczos3, 5 edge-adaptive.

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

const PI: f32 = 3.14159265;

fn load_px(p: vec2<i32>) -> vec3<f32> {
    let c = clamp(p, vec2<i32>(0, 0), vec2<i32>(params.in_size) - vec2<i32>(1, 1));
    return textureLoad(src, c, 0).rgb;
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

fn lanczos(x: f32, a: f32) -> f32 {
    let ax = abs(x);
    if (ax < 1e-4) { return 1.0; }
    if (ax >= a) { return 0.0; }
    let px = PI * x;
    return a * sin(px) * sin(px / a) / (px * px);
}

fn catrom(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 1.0) { return 1.5 * ax * ax * ax - 2.5 * ax * ax + 1.0; }
    if (ax < 2.0) { return -0.5 * ax * ax * ax + 2.5 * ax * ax - 4.0 * ax + 2.0; }
    return 0.0;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_px = vec2<u32>(params.out_size);
    if (gid.x >= out_px.x || gid.y >= out_px.y) { return; }

    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5, 0.5)) / params.out_size;
    let sp = uv * params.in_size - vec2<f32>(0.5, 0.5);
    let base = floor(sp);
    let f = sp - base;
    let bi = vec2<i32>(base);

    var color = vec3<f32>(0.0);

    if (params.mode == 0u) {
        color = load_px(vec2<i32>(floor(uv * params.in_size)));
    } else if (params.mode == 1u) {
        let a = load_px(bi);
        let b = load_px(bi + vec2<i32>(1, 0));
        let c = load_px(bi + vec2<i32>(0, 1));
        let d = load_px(bi + vec2<i32>(1, 1));
        color = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    } else {
        var radius = 2;
        if (params.mode == 4u) { radius = 3; }

        // Edge direction (mode 5): gradients at the four corners of the source cell,
        // interpolated by the sub-pixel offset.
        var e = vec2<f32>(1.0, 0.0);   // along the edge
        var n = vec2<f32>(0.0, 1.0);   // across the edge
        var edge_amt = 0.0;
        if (params.mode == 5u) {
            let l00 = luma(load_px(bi));
            let l10 = luma(load_px(bi + vec2<i32>(1, 0)));
            let l01 = luma(load_px(bi + vec2<i32>(0, 1)));
            let l11 = luma(load_px(bi + vec2<i32>(1, 1)));
            // Central differences on the cell, widened by one ring for stability.
            let lm0 = luma(load_px(bi + vec2<i32>(-1, 0)));
            let l20 = luma(load_px(bi + vec2<i32>(2, 0)));
            let lm1 = luma(load_px(bi + vec2<i32>(-1, 1)));
            let l21 = luma(load_px(bi + vec2<i32>(2, 1)));
            let l0m = luma(load_px(bi + vec2<i32>(0, -1)));
            let l1m = luma(load_px(bi + vec2<i32>(1, -1)));
            let l02 = luma(load_px(bi + vec2<i32>(0, 2)));
            let l12 = luma(load_px(bi + vec2<i32>(1, 2)));
            let gx0 = (l10 - lm0) * 0.5;
            let gx1 = (l20 - l00) * 0.5;
            let gx2 = (l11 - lm1) * 0.5;
            let gx3 = (l21 - l01) * 0.5;
            let gy0 = (l01 - l0m) * 0.5;
            let gy1 = (l11 - l1m) * 0.5;
            let gy2 = (l02 - l00) * 0.5;
            let gy3 = (l12 - l10) * 0.5;
            let gx = mix(mix(gx0, gx1, f.x), mix(gx2, gx3, f.x), f.y);
            let gy = mix(mix(gy0, gy1, f.x), mix(gy2, gy3, f.x), f.y);
            let mag = sqrt(gx * gx + gy * gy);
            if (mag > 1e-4) {
                n = vec2<f32>(gx, gy) / mag;
                e = vec2<f32>(-n.y, n.x);
            }
            edge_amt = smoothstep(0.02, 0.2, mag);
        }

        var sum = vec3<f32>(0.0);
        var wsum = 0.0;
        var mn = vec3<f32>(1e9);
        var mx = vec3<f32>(-1e9);
        for (var j = 1 - radius; j <= radius; j = j + 1) {
            for (var i = 1 - radius; i <= radius; i = i + 1) {
                let d = vec2<f32>(f32(i) - f.x, f32(j) - f.y);
                var w = 1.0;
                if (params.mode == 2u) {
                    w = catrom(d.x) * catrom(d.y);
                } else if (params.mode == 3u) {
                    w = lanczos(d.x, 2.0) * lanczos(d.y, 2.0);
                } else if (params.mode == 4u) {
                    w = lanczos(d.x, 3.0) * lanczos(d.y, 3.0);
                } else {
                    // Anisotropic radial lanczos2: wider along the edge, narrower across.
                    let along = dot(d, e) * mix(1.0, 0.8, edge_amt);
                    let across = dot(d, n) * mix(1.0, 1.35, edge_amt);
                    w = lanczos(sqrt(along * along + across * across), 2.0);
                }
                let c = load_px(bi + vec2<i32>(i, j));
                sum = sum + c * w;
                wsum = wsum + w;
                if (i >= 0 && i <= 1 && j >= 0 && j <= 1) {
                    mn = min(mn, c);
                    mx = max(mx, c);
                }
            }
        }
        color = sum / max(wsum, 1e-5);
        // Anti-ringing: pull overshoot back toward the inner 2x2 range.
        color = mix(color, clamp(color, mn, mx), params.anti_ring);
    }

    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
