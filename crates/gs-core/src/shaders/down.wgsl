// 2x2 box downsample of a luma level (low-pass, which is what makes the coarse motion
// search immune to aliasing on periodic textures).
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = textureDimensions(dst);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let s = vec2<i32>(textureDimensions(src)) - vec2<i32>(1, 1);
    let p = vec2<i32>(gid.xy) * 2;
    let a = textureLoad(src, clamp(p, vec2<i32>(0, 0), s), 0).r;
    let b = textureLoad(src, clamp(p + vec2<i32>(1, 0), vec2<i32>(0, 0), s), 0).r;
    let c = textureLoad(src, clamp(p + vec2<i32>(0, 1), vec2<i32>(0, 0), s), 0).r;
    let e = textureLoad(src, clamp(p + vec2<i32>(1, 1), vec2<i32>(0, 0), s), 0).r;
    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>((a + b + c + e) * 0.25, 0.0, 0.0, 0.0));
}
