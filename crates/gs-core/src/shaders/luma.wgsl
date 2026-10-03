// RGBA -> luma (level 0 of the motion-search pyramid).
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = textureDimensions(dst);
    if (gid.x >= d.x || gid.y >= d.y) { return; }
    let rgb = textureLoad(src, vec2<i32>(gid.xy), 0).rgb;
    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(dot(rgb, vec3<f32>(0.299, 0.587, 0.114)), 0.0, 0.0, 0.0));
}
