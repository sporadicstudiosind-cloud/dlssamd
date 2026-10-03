// Block-matching motion estimation for ONE pyramid level. One thread per block.
// Finds v minimising SAD(curr(x), prev(x + v)); the object therefore moved by -v.
//
// Called coarse-to-fine. Each level searches an exhaustive (2r+1)^2 window centred on the
// vector inherited from the coarser level (scaled x2), plus the zero vector. The top level
// has no parent and centres on zero with a larger window.
// Output texel = (v.x, v.y, normalised SAD, 0) in this level's pixels.

struct Params {
    level_size: vec2<u32>,
    flow_dims: vec2<u32>,
    block: u32,
    radius: i32,
    stride: u32,
    has_init: u32,
    total_radius: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var prev_tex: texture_2d<f32>;
@group(0) @binding(2) var curr_tex: texture_2d<f32>;
@group(0) @binding(3) var init_flow: texture_2d<f32>;
@group(0) @binding(4) var flow_out: texture_storage_2d<rgba16float, write>;

fn luma_at(t: texture_2d<f32>, p: vec2<i32>) -> f32 {
    let c = clamp(p, vec2<i32>(0, 0), vec2<i32>(params.level_size) - vec2<i32>(1, 1));
    return textureLoad(t, c, 0).r;
}

fn block_cost(origin: vec2<i32>, v: vec2<i32>) -> f32 {
    var sum = 0.0;
    var count = 0.0;
    let b = i32(params.block);
    let s = i32(params.stride);
    for (var j = 0; j < b; j = j + s) {
        for (var i = 0; i < b; i = i + s) {
            let p = origin + vec2<i32>(i, j);
            sum = sum + abs(luma_at(curr_tex, p) - luma_at(prev_tex, p + v));
            count = count + 1.0;
        }
    }
    // Small bias toward zero motion so flat/noisy regions don't invent movement.
    let bias = 0.002 * length(vec2<f32>(v)) / max(params.total_radius, 1.0);
    return sum / count + bias;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.flow_dims.x || gid.y >= params.flow_dims.y) { return; }
    let origin = vec2<i32>(gid.xy) * i32(params.block);

    var center = vec2<i32>(0, 0);
    if (params.has_init != 0u) {
        let id = vec2<i32>(textureDimensions(init_flow));
        let parent = clamp(vec2<i32>(gid.xy) / 2, vec2<i32>(0, 0), id - vec2<i32>(1, 1));
        let pv = textureLoad(init_flow, parent, 0).xy;
        center = vec2<i32>(round(pv * 2.0));
    }

    var best_v = vec2<i32>(0, 0);
    var best_c = block_cost(origin, best_v);

    let r = params.radius;
    for (var dy = -r; dy <= r; dy = dy + 1) {
        for (var dx = -r; dx <= r; dx = dx + 1) {
            let v = center + vec2<i32>(dx, dy);
            if (v.x == 0 && v.y == 0) { continue; }
            let c = block_cost(origin, v);
            if (c < best_c) { best_c = c; best_v = v; }
        }
    }

    textureStore(flow_out, vec2<i32>(gid.xy), vec4<f32>(f32(best_v.x), f32(best_v.y), best_c, 0.0));
}
