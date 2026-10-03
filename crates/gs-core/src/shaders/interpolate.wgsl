// Builds the in-between frame at time t in (0,1) from prev and curr.
// mode 0: cross-fade. mode 1: motion-compensated.
//
// Flow convention (see flow.wgsl): curr(p) ~ prev(p + v). For an output pixel r at time t:
//   prev sample position = r + t*v        curr sample position = r - (1-t)*v
// (flow is anchored in curr coordinates; a small error for large motion, accepted.)

struct Params {
    size: vec2<u32>,
    flow_dims: vec2<u32>,
    t: f32,
    block: f32,
    scale: f32,      // output px per source px
    conf: f32,       // SAD above which we fall back to cross-fade
    mode: u32,
    mask_count: u32,
    _a: u32,
    _b: u32,
    masks: array<vec4<f32>, 4>,   // x, y, w, h as 0..1 fractions
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var prev_tex: texture_2d<f32>;
@group(0) @binding(2) var curr_tex: texture_2d<f32>;
@group(0) @binding(3) var flow_tex: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;
@group(0) @binding(5) var dst: texture_storage_2d<rgba8unorm, write>;

fn flow_at(c: vec2<i32>) -> vec4<f32> {
    let cc = clamp(c, vec2<i32>(0, 0), vec2<i32>(params.flow_dims) - vec2<i32>(1, 1));
    return textureLoad(flow_tex, cc, 0);
}

fn in_mask(uv: vec2<f32>) -> bool {
    for (var i = 0u; i < params.mask_count; i = i + 1u) {
        let m = params.masks[i];
        if (uv.x >= m.x && uv.x <= m.x + m.z && uv.y >= m.y && uv.y <= m.y + m.w) { return true; }
    }
    return false;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) { return; }
    let size_f = vec2<f32>(params.size);
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5, 0.5)) / size_f;

    var v = vec2<f32>(0.0, 0.0);   // in source pixels
    if (params.mode == 1u && !in_mask(uv)) {
        // Bilinear blend of the four nearest block flows for smooth motion fields.
        let src_pos = (vec2<f32>(gid.xy) + vec2<f32>(0.5, 0.5)) / params.scale;
        let fp = src_pos / params.block - vec2<f32>(0.5, 0.5);
        let i0 = vec2<i32>(floor(fp));
        let fr = fp - floor(fp);
        let a = flow_at(i0);
        let b = flow_at(i0 + vec2<i32>(1, 0));
        let c = flow_at(i0 + vec2<i32>(0, 1));
        let d = flow_at(i0 + vec2<i32>(1, 1));
        let m = mix(mix(a, b, fr.x), mix(c, d, fr.x), fr.y);
        let worst = max(max(a.z, b.z), max(c.z, d.z));
        // Trust fades out smoothly as match error approaches 2x the threshold.
        let trust = 1.0 - smoothstep(params.conf, params.conf * 2.0 + 1e-4, worst);
        v = m.xy * trust;
    }

    // Convert source-pixel displacement to uv displacement.
    let vuv = v * params.scale / size_f;
    let p_uv = uv + params.t * vuv;
    let c_uv = uv - (1.0 - params.t) * vuv;
    let pc = textureSampleLevel(prev_tex, samp, p_uv, 0.0).rgb;
    let cc = textureSampleLevel(curr_tex, samp, c_uv, 0.0).rgb;
    let outc = mix(pc, cc, params.t);
    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(outc, 1.0));
}
