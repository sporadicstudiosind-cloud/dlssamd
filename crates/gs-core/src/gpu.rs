//! wgpu pipeline: captured frame in, `multiplier` (or 1) presentable frames out.
//!
//! ```text
//! capture --copy--> src[curr]
//!   frame gen on:   flow(prev,curr) -> for each phase t: interpolate -> upscale -> sharpen -> out[k]
//!   always:         upscale(curr) -> sharpen -> out[last]
//! ```
//! Outputs are in display order; the last one is always the real (newest) frame. Showing
//! generated frames before the real one is what costs one source interval of latency.

use crate::config::{FrameGenKind, FrameGenStage, Profile, UpscalerKind};
use crate::pacing::interpolation_phases;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const FLOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const LUMA_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// Pyramid depth for a given search radius: enough levels that the top-level window is small.
fn pyramid_levels(radius: u32) -> u32 {
    let r = radius.max(1) as f32;
    ((r / 4.0).log2().ceil().max(0.0) as u32 + 1).clamp(1, 4)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UpscaleParams {
    in_size: [f32; 2],
    out_size: [f32; 2],
    mode: u32,
    anti_ring: f32,
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SharpenParams {
    size: [f32; 2],
    strength: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FlowParams {
    level_size: [u32; 2],
    flow_dims: [u32; 2],
    block: u32,
    radius: i32,
    stride: u32,
    has_init: u32,
    total_radius: f32,
    _pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InterpParams {
    size: [u32; 2],
    flow_dims: [u32; 2],
    t: f32,
    block: f32,
    scale: f32,
    conf: f32,
    mode: u32,
    mask_count: u32,
    _a: u32,
    _b: u32,
    masks: [[f32; 4]; 4],
}

struct Stage {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

enum Entry {
    Uniform,
    Tex,
    /// Non-filterable float texture (r32float); only ever read with textureLoad.
    TexLoad,
    Storage(wgpu::TextureFormat),
    Sampler,
}

fn make_stage(device: &wgpu::Device, label: &str, wgsl: &str, entries: &[Entry]) -> Stage {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: match e {
                Entry::Uniform => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                Entry::Tex => wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                Entry::TexLoad => wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                Entry::Storage(f) => wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: *f,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                Entry::Sampler => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            },
            count: None,
        })
        .collect();
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &bgl_entries,
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    Stage { pipeline, layout }
}

fn tex(
    device: &wgpu::Device,
    label: &str,
    size: (u32, u32),
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: size.0.max(1), height: size.1.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

const SAMPLED_STORAGE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING
    .union(wgpu::TextureUsages::STORAGE_BINDING)
    .union(wgpu::TextureUsages::COPY_SRC)
    .union(wgpu::TextureUsages::COPY_DST);

struct PhaseSlot {
    interp_params: wgpu::Buffer,
}

struct State {
    profile: Profile,
    in_size: (u32, u32),
    out_size: (u32, u32),
    src: [wgpu::Texture; 2],
    up: [wgpu::Texture; 2],
    mid: wgpu::Texture,
    scaled: wgpu::Texture,
    /// flows[0] is the full-resolution motion field used by interpolation.
    flows: Vec<wgpu::Texture>,
    /// 1x1 stand-in bound as the parent flow of the top pyramid level.
    dummy_flow: wgpu::Texture,
    /// luma[slot][level]
    luma: [Vec<wgpu::Texture>; 2],
    outputs: Vec<wgpu::Texture>,
    cur: usize,
    frames_seen: u64,
    upscale_params: wgpu::Buffer,
    sharpen_params: wgpu::Buffer,
    flow_params: Vec<wgpu::Buffer>,
    phases: Vec<PhaseSlot>,
}

/// GPU frame processor. Create once per device; call [`configure`](Self::configure)
/// whenever the profile or capture size changes, then [`process`](Self::process) per frame.
pub struct Pipeline {
    device: wgpu::Device,
    queue: wgpu::Queue,
    upscale: Stage,
    sharpen: Stage,
    flow: Stage,
    luma: Stage,
    down: Stage,
    interp: Stage,
    sampler: wgpu::Sampler,
    state: Option<State>,
}

impl Pipeline {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let upscale = make_stage(
            &device,
            "upscale",
            crate::shaders::UPSCALE,
            &[Entry::Uniform, Entry::Tex, Entry::Storage(FORMAT)],
        );
        let sharpen = make_stage(
            &device,
            "sharpen",
            crate::shaders::SHARPEN,
            &[Entry::Uniform, Entry::Tex, Entry::Storage(FORMAT)],
        );
        let flow = make_stage(
            &device,
            "flow",
            crate::shaders::FLOW,
            &[Entry::Uniform, Entry::TexLoad, Entry::TexLoad, Entry::Tex, Entry::Storage(FLOW_FORMAT)],
        );
        let luma = make_stage(&device, "luma", crate::shaders::LUMA, &[Entry::Tex, Entry::Storage(LUMA_FORMAT)]);
        let down = make_stage(&device, "down", crate::shaders::DOWN, &[Entry::TexLoad, Entry::Storage(LUMA_FORMAT)]);
        let interp = make_stage(
            &device,
            "interpolate",
            crate::shaders::INTERPOLATE,
            &[Entry::Uniform, Entry::Tex, Entry::Tex, Entry::Tex, Entry::Sampler, Entry::Storage(FORMAT)],
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self { device, queue, upscale, sharpen, flow, luma, down, interp, sampler, state: None }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Number of frames [`process`](Self::process) returns once a previous frame exists.
    pub fn outputs_per_frame(profile: &Profile) -> u32 {
        if profile.framegen.kind == FrameGenKind::Off {
            1
        } else {
            profile.framegen.multiplier
        }
    }

    /// The most recent motion field (`Rgba16Float`: v.x, v.y, match error), for diagnostics.
    pub fn flow_texture(&self) -> Option<&wgpu::Texture> {
        self.state.as_ref().map(|s| &s.flows[0])
    }

    pub fn output_size(&self) -> Option<(u32, u32)> {
        self.state.as_ref().map(|s| s.out_size)
    }

    /// (Re)allocate everything for `profile` at `capture_size`. Drops frame history.
    pub fn configure(&mut self, profile: &Profile, capture_size: (u32, u32)) {
        let mut profile = profile.clone();
        profile.sanitize();
        let d = &self.device;
        let in_size = (capture_size.0.max(1), capture_size.1.max(1));
        let out_size = profile.output_size(in_size);
        let block = profile.framegen.block_size;
        let flow_dims = (in_size.0.div_ceil(block), in_size.1.div_ceil(block));

        let mk = |label: &str, size, fmt| tex(d, label, size, fmt, SAMPLED_STORAGE);
        let src = [mk("src0", in_size, FORMAT), mk("src1", in_size, FORMAT)];
        let up = [mk("up0", out_size, FORMAT), mk("up1", out_size, FORMAT)];
        let mid = mk("mid", in_size, FORMAT);
        let scaled = mk("scaled", out_size, FORMAT);
        let levels = pyramid_levels(profile.framegen.search_radius) as usize;
        let level_size = |l: usize| ((in_size.0 >> l).max(1), (in_size.1 >> l).max(1));
        let level_flow_dims = |l: usize| {
            let (w, h) = level_size(l);
            (w.div_ceil(block), h.div_ceil(block))
        };
        let flows: Vec<wgpu::Texture> = (0..levels).map(|l| mk(&format!("flow{l}"), level_flow_dims(l), FLOW_FORMAT)).collect();
        let dummy_flow = mk("dummy_flow", (1, 1), FLOW_FORMAT);
        let luma_tex = |slot: usize| -> Vec<wgpu::Texture> { (0..levels).map(|l| mk(&format!("luma{slot}_{l}"), level_size(l), LUMA_FORMAT)).collect() };
        let luma = [luma_tex(0), luma_tex(1)];
        let n = Self::outputs_per_frame(&profile).max(1) as usize;
        let outputs = (0..n).map(|i| mk(&format!("out{i}"), out_size, FORMAT)).collect();

        let mode = if profile.upscale.enabled && out_size != in_size {
            profile.upscale.kind.shader_mode()
        } else {
            UpscalerKind::Nearest.shader_mode()
        };
        let ub = |label: &str, data: &[u8]| {
            d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let upscale_params = ub(
            "upscale_params",
            bytemuck::bytes_of(&UpscaleParams {
                in_size: [in_size.0 as f32, in_size.1 as f32],
                out_size: [out_size.0 as f32, out_size.1 as f32],
                mode,
                anti_ring: profile.upscale.anti_ringing,
                _pad: [0.0; 2],
            }),
        );
        let sharpen_params = ub(
            "sharpen_params",
            bytemuck::bytes_of(&SharpenParams {
                size: [out_size.0 as f32, out_size.1 as f32],
                strength: profile.upscale.sharpness,
                _pad: 0.0,
            }),
        );
        let total_r = profile.framegen.search_radius;
        let flow_params: Vec<wgpu::Buffer> = (0..levels)
            .map(|l| {
                let top = l == levels - 1;
                // Top level: window covers the whole range at 1/2^l scale. Others: +-2 refinement.
                let radius = if top { (total_r >> l) as i32 + 1 } else { 2 };
                ub(
                    "flow_params",
                    bytemuck::bytes_of(&FlowParams {
                        level_size: [level_size(l).0, level_size(l).1],
                        flow_dims: [level_flow_dims(l).0, level_flow_dims(l).1],
                        block,
                        radius,
                        stride: if l == 0 { profile.framegen.sample_stride } else { 1 },
                        has_init: u32::from(!top),
                        total_radius: (total_r >> l).max(1) as f32,
                        _pad: [0.0; 3],
                    }),
                )
            })
            .collect();

        // Interpolation runs at source res (BeforeUpscale) or output res (AfterUpscale).
        let after = profile.framegen.stage == FrameGenStage::AfterUpscale;
        let interp_size = if after { out_size } else { in_size };
        let scale = interp_size.0 as f32 / in_size.0 as f32;
        let mut masks = [[0.0f32; 4]; 4];
        let mask_count = profile.framegen.hud_masks.len().min(4);
        for (i, m) in profile.framegen.hud_masks.iter().take(4).enumerate() {
            masks[i] = *m;
        }
        let interp_mode = u32::from(profile.framegen.kind == FrameGenKind::OpticalFlow);
        let phases = interpolation_phases(Self::outputs_per_frame(&profile))
            .into_iter()
            .map(|t| PhaseSlot {
                interp_params: ub(
                    "interp_params",
                    bytemuck::bytes_of(&InterpParams {
                        size: [interp_size.0, interp_size.1],
                        flow_dims: [flow_dims.0, flow_dims.1],
                        t,
                        block: block as f32,
                        scale,
                        conf: profile.framegen.confidence_threshold,
                        mode: interp_mode,
                        mask_count: mask_count as u32,
                        _a: 0,
                        _b: 0,
                        masks,
                    }),
                ),
            })
            .collect();

        self.state = Some(State {
            profile,
            in_size,
            out_size,
            src,
            up,
            mid,
            scaled,
            flows,
            dummy_flow,
            luma,
            outputs,
            cur: 0,
            frames_seen: 0,
            upscale_params,
            sharpen_params,
            flow_params,
            phases,
        });
    }

    /// Run one captured frame (`FORMAT`, `capture_size`) through the pipeline and return the
    /// frames to present, in order. The last element is the real frame.
    ///
    /// Panics if [`configure`](Self::configure) has not been called.
    pub fn process(&mut self, frame: &wgpu::Texture) -> &[wgpu::Texture] {
        let mut st = self.state.take().expect("Pipeline::configure must be called first");
        let had_prev = st.frames_seen > 0;
        let prev = st.cur;
        let cur = 1 - st.cur;
        st.cur = cur;
        st.frames_seen += 1;

        let fg = st.profile.framegen.kind != FrameGenKind::Off && had_prev;
        let n = if fg { st.profile.framegen.multiplier as usize } else { 1 };
        let after = st.profile.framegen.stage == FrameGenStage::AfterUpscale;
        let sharpen = st.profile.upscale.enabled && st.profile.upscale.sharpness > 0.001;

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        enc.copy_texture_to_texture(
            frame.as_image_copy(),
            st.src[cur].as_image_copy(),
            wgpu::Extent3d { width: st.in_size.0, height: st.in_size.1, depth_or_array_layers: 1 },
        );

        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let bg = |stage: &Stage, entries: &[wgpu::BindingResource]| {
            let e: Vec<_> = entries
                .iter()
                .enumerate()
                .map(|(i, r)| wgpu::BindGroupEntry { binding: i as u32, resource: r.clone() })
                .collect();
            self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &stage.layout, entries: &e })
        };
        let dispatch = |enc: &mut wgpu::CommandEncoder, stage: &Stage, group: &wgpu::BindGroup, size: (u32, u32)| {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&stage.pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
        };

        // upscale(`from`) -> `to`
        let do_upscale = |enc: &mut wgpu::CommandEncoder, from: &wgpu::Texture, to: &wgpu::Texture| {
            let g = bg(
                &self.upscale,
                &[st.upscale_params.as_entire_binding(), wgpu::BindingResource::TextureView(&view(from)), wgpu::BindingResource::TextureView(&view(to))],
            );
            dispatch(enc, &self.upscale, &g, st.out_size);
        };
        let do_sharpen = |enc: &mut wgpu::CommandEncoder, from: &wgpu::Texture, to: &wgpu::Texture| {
            let g = bg(
                &self.sharpen,
                &[st.sharpen_params.as_entire_binding(), wgpu::BindingResource::TextureView(&view(from)), wgpu::BindingResource::TextureView(&view(to))],
            );
            dispatch(enc, &self.sharpen, &g, st.out_size);
        };
        let do_copy = |enc: &mut wgpu::CommandEncoder, from: &wgpu::Texture, to: &wgpu::Texture| {
            enc.copy_texture_to_texture(
                from.as_image_copy(),
                to.as_image_copy(),
                wgpu::Extent3d { width: st.out_size.0, height: st.out_size.1, depth_or_array_layers: 1 },
            );
        };

        // Motion-search pyramid for the new frame (the previous one is kept from last time).
        if st.profile.framegen.kind == FrameGenKind::OpticalFlow {
            let g = bg(&self.luma, &[wgpu::BindingResource::TextureView(&view(&st.src[cur])), wgpu::BindingResource::TextureView(&view(&st.luma[cur][0]))]);
            dispatch(&mut enc, &self.luma, &g, st.in_size);
            for l in 1..st.luma[cur].len() {
                let g = bg(&self.down, &[wgpu::BindingResource::TextureView(&view(&st.luma[cur][l - 1])), wgpu::BindingResource::TextureView(&view(&st.luma[cur][l]))]);
                dispatch(&mut enc, &self.down, &g, (st.luma[cur][l].width(), st.luma[cur][l].height()));
            }
        }

        if fg {
            if st.profile.framegen.kind == FrameGenKind::OpticalFlow {
                let levels = st.flows.len();
                for l in (0..levels).rev() {
                    let parent = if l + 1 < levels { &st.flows[l + 1] } else { &st.dummy_flow };
                    let g = bg(
                        &self.flow,
                        &[
                            st.flow_params[l].as_entire_binding(),
                            wgpu::BindingResource::TextureView(&view(&st.luma[prev][l])),
                            wgpu::BindingResource::TextureView(&view(&st.luma[cur][l])),
                            wgpu::BindingResource::TextureView(&view(parent)),
                            wgpu::BindingResource::TextureView(&view(&st.flows[l])),
                        ],
                    );
                    dispatch(&mut enc, &self.flow, &g, (st.flows[l].width(), st.flows[l].height()));
                }
            }
            if after {
                do_upscale(&mut enc, &st.src[cur], &st.up[cur]);
            }
            for (k, slot) in st.phases.iter().enumerate() {
                let (a, b, isize) = if after { (&st.up[prev], &st.up[cur], st.out_size) } else { (&st.src[prev], &st.src[cur], st.in_size) };
                // Where interpolation lands: straight into the output when nothing follows it.
                let interp_dst = if after {
                    if sharpen { &st.scaled } else { &st.outputs[k] }
                } else {
                    &st.mid
                };
                let g = bg(
                    &self.interp,
                    &[
                        slot.interp_params.as_entire_binding(),
                        wgpu::BindingResource::TextureView(&view(a)),
                        wgpu::BindingResource::TextureView(&view(b)),
                        wgpu::BindingResource::TextureView(&view(&st.flows[0])),
                        wgpu::BindingResource::Sampler(&self.sampler),
                        wgpu::BindingResource::TextureView(&view(interp_dst)),
                    ],
                );
                dispatch(&mut enc, &self.interp, &g, isize);
                if after {
                    if sharpen {
                        do_sharpen(&mut enc, &st.scaled, &st.outputs[k]);
                    }
                } else if sharpen {
                    do_upscale(&mut enc, &st.mid, &st.scaled);
                    do_sharpen(&mut enc, &st.scaled, &st.outputs[k]);
                } else {
                    do_upscale(&mut enc, &st.mid, &st.outputs[k]);
                }
            }
        }

        // The real frame, always last.
        let last = n - 1;
        if after && fg {
            // up[cur] already holds the upscaled real frame.
            if sharpen {
                do_sharpen(&mut enc, &st.up[cur], &st.outputs[last]);
            } else {
                do_copy(&mut enc, &st.up[cur], &st.outputs[last]);
            }
        } else if after {
            do_upscale(&mut enc, &st.src[cur], &st.up[cur]);
            if sharpen {
                do_sharpen(&mut enc, &st.up[cur], &st.outputs[last]);
            } else {
                do_copy(&mut enc, &st.up[cur], &st.outputs[last]);
            }
        } else if sharpen {
            do_upscale(&mut enc, &st.src[cur], &st.scaled);
            do_sharpen(&mut enc, &st.scaled, &st.outputs[last]);
        } else {
            do_upscale(&mut enc, &st.src[cur], &st.outputs[last]);
        }

        self.queue.submit([enc.finish()]);
        self.state = Some(st);
        let st = self.state.as_ref().unwrap();
        &st.outputs[..n]
    }
}
