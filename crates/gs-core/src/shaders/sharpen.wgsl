// Contrast-adaptive sharpening (the published CAS idea: adapt strength to local contrast
// so flat areas and already-hard edges aren't over-sharpened).

struct Params {
    size: vec2<f32>,
    strength: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba8unorm, write>;

fn px(p: vec2<i32>) -> vec3<f32> {
    let c = clamp(p, vec2<i32>(0, 0), vec2<i32>(params.size) - vec2<i32>(1, 1));
    return textureLoad(src, c, 0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<u32>(params.size);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let p = vec2<i32>(gid.xy);

    let e = px(p);
    let b = px(p + vec2<i32>(0, -1));
    let d = px(p + vec2<i32>(-1, 0));
    let f = px(p + vec2<i32>(1, 0));
    let h = px(p + vec2<i32>(0, 1));

    let mn = min(min(min(d, e), min(f, b)), h);
    let mx = max(max(max(d, e), max(f, b)), h);
    let amp = sqrt(clamp(min(mn, vec3<f32>(1.0) - mx) / max(mx, vec3<f32>(1e-4)), vec3<f32>(0.0), vec3<f32>(1.0)));
    let peak = -1.0 / mix(8.0, 5.0, clamp(params.strength, 0.0, 1.0));
    let w = amp * peak;
    let out_c = (w * (b + d + f + h) + e) / (vec3<f32>(1.0) + 4.0 * w);
    textureStore(dst, p, vec4<f32>(clamp(out_c, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
