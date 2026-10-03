//! End-to-end GPU tests. They need a Vulkan device (lavapipe in CI); if none is present
//! they print a notice and pass, so `cargo test` never fails just for lack of a GPU.

use gs_core::config::*;
use gs_core::gpu::{Pipeline, FORMAT};

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .ok()?;
    eprintln!("adapter: {:?}", adapter.get_info().name);
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
}

fn upload(
    dev: &wgpu::Device,
    q: &wgpu::Queue,
    w: u32,
    h: u32,
    f: impl Fn(u32, u32) -> [f32; 3],
) -> wgpu::Texture {
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let c = f(x, y);
            data.extend(c.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8));
            data.push(255);
        }
    }
    let t = dev.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    q.write_texture(
        t.as_image_copy(),
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 4),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    t
}

fn read(dev: &wgpu::Device, q: &wgpu::Queue, t: &wgpu::Texture) -> (u32, u32, Vec<f32>) {
    let (w, h) = (t.width(), t.height());
    let bpr = (w * 4).next_multiple_of(256);
    let buf = dev.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (bpr * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = dev.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        t.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    q.submit([enc.finish()]);
    let slice = buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
    dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let raw = slice.get_mapped_range().unwrap();
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let i = (y * bpr + x * 4) as usize;
            for c in 0..3 {
                out.push(raw[i + c] as f32 / 255.0);
            }
        }
    }
    (w, h, out)
}

fn profile() -> Profile {
    let mut p = Profile::default();
    p.upscale.sharpness = 0.0;
    p
}

#[test]
fn constant_image_stays_constant_in_every_upscaler() {
    let Some((dev, q)) = device() else {
        return eprintln!("no GPU adapter; skipping");
    };
    for kind in UpscalerKind::ALL {
        for sharp in [0.0, 0.8] {
            let mut p = profile();
            p.upscale.kind = kind;
            p.upscale.sharpness = sharp;
            p.upscale.scale = 2.0;
            p.framegen.kind = FrameGenKind::Off;
            let mut pipe = Pipeline::new(dev.clone(), q.clone());
            pipe.configure(&p, (32, 24));
            let input = upload(&dev, &q, 32, 24, |_, _| [0.25, 0.5, 0.75]);
            let outs = pipe.process(&input);
            assert_eq!(outs.len(), 1);
            let (w, h, px) = read(&dev, &q, &outs[0]);
            assert_eq!((w, h), (64, 48));
            for v in px.chunks(3) {
                assert!(
                    (v[0] - 0.25).abs() < 0.01
                        && (v[1] - 0.5).abs() < 0.01
                        && (v[2] - 0.75).abs() < 0.01,
                    "{kind:?} sharp={sharp}: {v:?}"
                );
            }
        }
    }
}

#[test]
fn bilinear_matches_hand_computed_values() {
    let Some((dev, q)) = device() else {
        return eprintln!("no GPU adapter; skipping");
    };
    let mut p = profile();
    p.upscale.kind = UpscalerKind::Bilinear;
    p.upscale.scale = 2.0;
    p.framegen.kind = FrameGenKind::Off;
    let mut pipe = Pipeline::new(dev.clone(), q.clone());
    pipe.configure(&p, (4, 4));
    // Horizontal ramp 0, 1/3, 2/3, 1 (values are exact in u8: 0, 85, 170, 255).
    let input = upload(&dev, &q, 4, 4, |x, _| [x as f32 / 3.0; 3]);
    let outs = pipe.process(&input);
    let (_, _, px) = read(&dev, &q, &outs[0]);
    // Output pixel x=3 (centre 3.5/8 -> source 1.25-0.5... = 1.25): between src 1 and 2 at f=.25.
    // source pos = (3+0.5)*0.5 - 0.5 = 1.25 -> 85 + 0.25*(170-85) = 106.25 -> 0.4167
    let got = px[3 * 3];
    assert!((got - 106.25 / 255.0).abs() < 0.01, "{got}");
    // x=0 clamps to the left edge value.
    assert!(px[0].abs() < 0.01);
}

fn pattern(x: f32, y: f32) -> [f32; 3] {
    [
        0.5 + 0.25 * (0.35 * x).sin() + 0.2 * (0.21 * y).sin(),
        0.5 + 0.25 * (0.27 * y + 0.1 * x).sin(),
        0.5 + 0.2 * (0.17 * (x + y)).sin() + 0.2 * (0.31 * x).cos(),
    ]
}

fn mse(a: &[f32], b: &[f32], w: u32, h: u32, margin: u32) -> f32 {
    let (mut s, mut n) = (0.0, 0.0);
    for y in margin..h - margin {
        for x in margin..w - margin {
            for c in 0..3 {
                let i = ((y * w + x) * 3) as usize + c;
                s += (a[i] - b[i]).powi(2);
                n += 1.0;
            }
        }
    }
    s / n
}

fn run_pan(kind: FrameGenKind, stage: FrameGenStage, motion: (f32, f32)) -> f32 {
    let (dev, q) = device().unwrap();
    let (w, h) = (96u32, 64u32);
    let mut p = profile();
    p.upscale.enabled = false;
    p.framegen.kind = kind;
    p.framegen.stage = stage;
    p.framegen.multiplier = 2;
    p.framegen.block_size = 8;
    p.framegen.search_radius = 16;
    p.framegen.sample_stride = 1;
    let mut pipe = Pipeline::new(dev.clone(), q.clone());
    pipe.configure(&p, (w, h));
    // Content moves by +motion each frame: frame_i(p) = pattern(p - i*motion).
    let f0 = upload(&dev, &q, w, h, |x, y| {
        pattern(x as f32 - 0.0, y as f32 - 0.0)
    });
    let f1 = upload(&dev, &q, w, h, |x, y| {
        pattern(x as f32 - motion.0, y as f32 - motion.1)
    });
    assert_eq!(pipe.process(&f0).len(), 1, "first frame has no history");
    let outs = pipe.process(&f1);
    assert_eq!(outs.len(), 2);
    let (_, _, got) = read(&dev, &q, &outs[0]);
    // Ground truth mid-frame: content displaced by half the motion.
    let (_, _, want) = read(
        &dev,
        &q,
        &upload(&dev, &q, w, h, |x, y| {
            pattern(x as f32 - motion.0 / 2.0, y as f32 - motion.1 / 2.0)
        }),
    );
    // The last output must be the real frame.
    let (_, _, real) = read(&dev, &q, &outs[1]);
    let (_, _, real_want) = read(&dev, &q, &f1);
    assert!(
        mse(&real, &real_want, w, h, 0) < 1e-4,
        "last output isn't the real frame"
    );
    mse(&got, &want, w, h, 12)
}

#[test]
fn optical_flow_beats_blend_on_a_pan() {
    if device().is_none() {
        return eprintln!("no GPU adapter; skipping");
    }
    for motion in [
        (6.0, 0.0),
        (4.0, 2.0),
        (-8.0, 4.0),
        (5.0, 3.0),
        (-7.0, -1.0),
    ] {
        let blend = run_pan(FrameGenKind::Blend, FrameGenStage::BeforeUpscale, motion);
        let flow = run_pan(
            FrameGenKind::OpticalFlow,
            FrameGenStage::BeforeUpscale,
            motion,
        );
        let flow_after = run_pan(
            FrameGenKind::OpticalFlow,
            FrameGenStage::AfterUpscale,
            motion,
        );
        eprintln!("motion {motion:?}: blend mse {blend:.5}  flow mse {flow:.5}  flow(after) mse {flow_after:.5}");
        assert!(
            flow < blend * 0.25,
            "flow {flow} not clearly better than blend {blend} for {motion:?}"
        );
        assert!(
            flow_after < blend * 0.25,
            "after-stage flow {flow_after} vs blend {blend}"
        );
    }
}

#[test]
fn multiplier_controls_output_count_and_sizes() {
    let Some((dev, q)) = device() else {
        return eprintln!("no GPU adapter; skipping");
    };
    let mut p = profile();
    p.upscale.scale = 1.5;
    p.upscale.sharpness = 0.5;
    p.framegen.multiplier = 4;
    let mut pipe = Pipeline::new(dev.clone(), q.clone());
    pipe.configure(&p, (64, 48));
    let a = upload(&dev, &q, 64, 48, |x, y| pattern(x as f32, y as f32));
    assert_eq!(pipe.process(&a).len(), 1);
    let outs = pipe.process(&a);
    assert_eq!(outs.len(), 4);
    for o in outs {
        assert_eq!((o.width(), o.height()), (96, 72));
    }
}

fn f16(bits: u16) -> f32 {
    let s = if bits >> 15 == 1 { -1.0 } else { 1.0 };
    let e = ((bits >> 10) & 0x1f) as i32;
    let m = (bits & 0x3ff) as f32;
    if e == 0 {
        s * m * 2f32.powi(-24)
    } else {
        s * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
    }
}

#[test]
#[ignore]
fn debug_flow_vectors() {
    let (dev, q) = device().unwrap();
    let (w, h) = (96u32, 64u32);
    let mut p = profile();
    p.upscale.enabled = false;
    p.framegen.sample_stride = 1;
    let mut pipe = Pipeline::new(dev.clone(), q.clone());
    pipe.configure(&p, (w, h));
    let m = (5.0, 3.0);
    let f0 = upload(&dev, &q, w, h, |x, y| pattern(x as f32, y as f32));
    let f1 = upload(&dev, &q, w, h, |x, y| {
        pattern(x as f32 - m.0, y as f32 - m.1)
    });
    pipe.process(&f0);
    pipe.process(&f1);
    let t = pipe.flow_texture().unwrap();
    let (fw, fh) = (t.width(), t.height());
    let bpr = (fw * 8).next_multiple_of(256);
    let buf = dev.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (bpr * fh) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = dev.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        t.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(fh),
            },
        },
        wgpu::Extent3d {
            width: fw,
            height: fh,
            depth_or_array_layers: 1,
        },
    );
    q.submit([enc.finish()]);
    buf.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let raw = buf.slice(..).get_mapped_range().unwrap();
    for y in 0..fh {
        let mut line = String::new();
        for x in 0..fw {
            let i = (y * bpr + x * 8) as usize;
            let g = |o: usize| f16(u16::from_le_bytes([raw[i + o], raw[i + o + 1]]));
            line += &format!("({:+.0},{:+.0}|{:.3}) ", g(0), g(2), g(4));
        }
        eprintln!("{line}");
    }
}
