//! Presentation: a borderless always-on-top window that shows the processed frames,
//! paced according to the latency settings.

use crate::platform;
use crate::source::{Frame, FrameQueue, FrameSource, ScreenRect};
use anyhow::{anyhow, Context, Result};
use gs_core::config::{PresentMode, Profile};
use gs_core::gpu::{Pipeline, FORMAT};
use gs_core::pacing::{present_offsets, IntervalEstimator};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};
use winit::window::{Window, WindowAttributes, WindowId, WindowLevel};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Cover exactly the target window (mouse input stays aligned).
    MatchTarget,
    /// Cover the whole monitor the target is on. Upscaling is only visible in this mode,
    /// but the mouse position is NOT remapped yet, so clicks land on the small window below.
    FillMonitor,
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub running: bool,
    pub adapter: String,
    pub source_fps: f32,
    pub output_fps: f32,
    pub process_ms: f32,
    /// Capture callback -> first presented frame, measured.
    pub latency_ms: f32,
    pub capture_size: (u32, u32),
    pub output_size: (u32, u32),
    pub dropped_frames: u64,
    pub error: Option<String>,
}

pub struct Session {
    pub label: String,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    pub stats: Arc<Mutex<Stats>>,
}

impl Session {
    pub fn start(source: Box<dyn FrameSource>, profile: Profile, placement: Placement) -> Session {
        let label = source.label();
        let stop = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Mutex::new(Stats {
            running: true,
            ..Default::default()
        }));
        let (s2, st2) = (stop.clone(), stats.clone());
        let thread = std::thread::Builder::new()
            .name("gpuscale-render".into())
            .spawn(move || {
                let r = run_blocking(source, profile, placement, s2, st2.clone());
                let mut s = st2.lock().unwrap();
                s.running = false;
                if let Err(e) = r {
                    s.error = Some(format!("{e:#}"));
                }
            })
            .expect("spawn render thread");
        Session {
            label,
            stop,
            thread: Some(thread),
            stats,
        }
    }

    pub fn is_running(&self) -> bool {
        self.stats.lock().unwrap().running
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.stop();
    }
}

const BLIT: &str = r#"
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var o: VOut;
    o.pos = vec4<f32>(p[i], 0.0, 1.0);
    o.uv = vec2<f32>((p[i].x + 1.0) * 0.5, (1.0 - p[i].y) * 0.5);
    return o;
}
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
@fragment fn fs(v: VOut) -> @location(0) vec4<f32> { return vec4<f32>(textureSample(t, s, v.uv).rgb, 1.0); }
"#;

struct Gfx {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    blit: wgpu::RenderPipeline,
    blit_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    pipeline: Pipeline,
    input: Option<wgpu::Texture>,
    configured_for: Option<(u32, u32)>,
    adapter_name: String,
}

struct Overlay {
    profile: Profile,
    rect: Option<ScreenRect>,
    window: Option<Arc<Window>>,
    gfx: Option<Gfx>,
    instance: wgpu::Instance,
    closed: bool,
    error: Option<anyhow::Error>,
}

fn pick_present_mode(caps: &wgpu::SurfaceCapabilities, want: PresentMode) -> wgpu::PresentMode {
    use wgpu::PresentMode as W;
    let has = |m: W| caps.present_modes.contains(&m);
    match want {
        PresentMode::Immediate if has(W::Immediate) => W::Immediate,
        PresentMode::Immediate | PresentMode::Mailbox if has(W::Mailbox) => W::Mailbox,
        _ => W::Fifo,
    }
}

impl Overlay {
    fn window_attrs(&self) -> WindowAttributes {
        let r = self.rect;
        let mut a = Window::default_attributes()
            .with_title("GPUScale output")
            .with_visible(true)
            .with_active(r.is_none());
        match r {
            Some(r) => {
                a = a
                    .with_decorations(false)
                    .with_window_level(WindowLevel::AlwaysOnTop)
                    .with_position(PhysicalPosition::new(r.x, r.y))
                    .with_inner_size(PhysicalSize::new(r.w, r.h));
            }
            None => {
                a = a.with_inner_size(PhysicalSize::new(1280u32, 720u32));
            }
        }
        #[cfg(windows)]
        {
            use winit::platform::windows::WindowAttributesExtWindows;
            if r.is_some() {
                a = a.with_skip_taskbar(true);
            }
        }
        a
    }

    fn init_gfx(&mut self, window: Arc<Window>) -> Result<()> {
        let surface = self
            .instance
            .create_surface(window.clone())
            .context("create surface")?;
        let adapter =
            pollster::block_on(self.instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            }))
            .map_err(|e| anyhow!("no suitable GPU adapter: {e}"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gpuscale"),
            ..Default::default()
        }))
        .context("request device")?;

        let caps = surface.get_capabilities(&adapter);
        // Prefer a non-sRGB format: our pixels are already display-encoded, so an sRGB
        // surface would apply the gamma curve a second time.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: Default::default(),
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: pick_present_mode(&caps, self.profile.latency.present_mode),
            desired_maximum_frame_latency: self.profile.latency.max_frame_latency,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT.into()),
        });
        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit"),
            bind_group_layouts: &[Some(&blit_layout)],
            immediate_size: 0,
        });
        let blit = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let pipeline = Pipeline::new(device.clone(), queue.clone());
        log::info!(
            "adapter: {}, present mode {:?}, surface {:?}",
            adapter.get_info().name,
            config.present_mode,
            format
        );
        self.gfx = Some(Gfx {
            surface,
            config,
            device,
            queue,
            blit,
            blit_layout,
            sampler,
            pipeline,
            input: None,
            configured_for: None,
            adapter_name: adapter.get_info().name,
        });
        self.window = Some(window);
        Ok(())
    }
}

impl ApplicationHandler for Overlay {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let res = el
            .create_window(self.window_attrs())
            .context("create overlay window")
            .and_then(|w| {
                let w = Arc::new(w);
                if self.rect.is_some() {
                    // Let mouse/keyboard fall through to the game underneath.
                    let _ = w.set_cursor_hittest(false);
                }
                self.init_gfx(w)
            });
        if let Err(e) = res {
            self.error = Some(e);
            self.closed = true;
        }
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, _id: WindowId, ev: WindowEvent) {
        match ev {
            WindowEvent::CloseRequested => self.closed = true,
            WindowEvent::Resized(sz) => {
                if let Some(g) = &mut self.gfx {
                    g.config.width = sz.width.max(1);
                    g.config.height = sz.height.max(1);
                    g.surface.configure(&g.device, &g.config);
                }
            }
            _ => {}
        }
    }
}

fn wait_until(deadline: Instant, precise: bool, spin: Duration) {
    let now = Instant::now();
    if deadline <= now {
        return;
    }
    let coarse = if precise {
        deadline.saturating_duration_since(now).saturating_sub(spin)
    } else {
        deadline - now
    };
    if !coarse.is_zero() {
        std::thread::sleep(coarse);
    }
    if precise {
        while Instant::now() < deadline {
            std::hint::spin_loop();
        }
    }
}

impl Gfx {
    fn upload(&mut self, f: &Frame, profile: &Profile) {
        let size = (f.width, f.height);
        if self.configured_for != Some(size) {
            self.pipeline.configure(profile, size);
            self.input = Some(self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("capture"),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
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
            }));
            self.configured_for = Some(size);
        }
        self.queue.write_texture(
            self.input.as_ref().unwrap().as_image_copy(),
            &f.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(f.width * 4),
                rows_per_image: Some(f.height),
            },
            wgpu::Extent3d {
                width: f.width,
                height: f.height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Draw `tex` stretched over the surface and present it.
    fn present(&self, tex: &wgpu::Texture) -> Result<()> {
        use wgpu::CurrentSurfaceTexture as C;
        let frame = match self.surface.get_current_texture() {
            C::Success(t) | C::Suboptimal(t) => t,
            C::Timeout | C::Occluded => return Ok(()),
            C::Outdated | C::Lost => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            other => return Err(anyhow!("surface error: {other:?}")),
        };
        let view = frame.texture.create_view(&Default::default());
        let tex_view = tex.create_view(&Default::default());
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.blit_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(&self.blit);
            rp.set_bind_group(0, &bg, &[]);
            rp.draw(0..3, 0..1);
        }
        self.queue.submit([enc.finish()]);
        self.queue.present(frame);
        Ok(())
    }
}

/// Run the whole capture -> process -> present loop on the calling thread until `stop`
/// is set, the window is closed, or an error occurs.
pub fn run_blocking(
    mut source: Box<dyn FrameSource>,
    mut profile: Profile,
    placement: Placement,
    stop: Arc<AtomicBool>,
    stats: Arc<Mutex<Stats>>,
) -> Result<()> {
    profile.sanitize();
    if profile.latency.high_priority {
        platform::boost_current_thread();
    }
    let queue = FrameQueue::new(profile.latency.capture_queue_depth as usize);
    source.start(queue.clone())?;
    let result = render_loop(&*source, &profile, placement, &stop, &stats, &queue);
    source.stop();
    platform::restore_timer_resolution();
    result
}

fn render_loop(
    source: &dyn FrameSource,
    profile: &Profile,
    placement: Placement,
    stop: &AtomicBool,
    stats: &Mutex<Stats>,
    queue: &FrameQueue,
) -> Result<()> {
    let mut builder = EventLoop::builder();
    #[cfg(windows)]
    {
        use winit::platform::windows::EventLoopBuilderExtWindows;
        builder.with_any_thread(true);
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        use winit::platform::x11::EventLoopBuilderExtX11;
        EventLoopBuilderExtX11::with_any_thread(&mut builder, true);
    }
    let mut event_loop = builder.build().context("create event loop")?;

    let mut app = Overlay {
        profile: profile.clone(),
        rect: source.screen_rect(),
        window: None,
        gfx: None,
        instance: wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle()),
        closed: false,
        error: None,
    };

    let mut interval = IntervalEstimator::new(0.1);
    let mut last_real: Option<Instant> = None;
    let mut last_present = Instant::now();
    let mut out_count = 0u32;
    let mut fps_window = Instant::now();
    let mut last_rect_check = Instant::now();
    let precise = profile.latency.precise_pacing;
    let spin = Duration::from_micros(profile.latency.spin_us as u64);

    while !stop.load(Ordering::SeqCst) {
        if let PumpStatus::Exit(_) = event_loop.pump_app_events(Some(Duration::ZERO), &mut app) {
            break;
        }
        if let Some(e) = app.error.take() {
            return Err(e);
        }
        if app.closed {
            break;
        }
        let Some(gfx) = app.gfx.as_mut() else {
            continue;
        };

        // Follow the target window if it moves (Match mode only).
        if placement == Placement::MatchTarget
            && last_rect_check.elapsed() > Duration::from_millis(250)
        {
            last_rect_check = Instant::now();
            if let (Some(r), Some(w)) = (source.screen_rect(), app.window.as_ref()) {
                if app.rect != Some(r) {
                    app.rect = Some(r);
                    w.set_outer_position(PhysicalPosition::new(r.x, r.y));
                    let _ = w.request_inner_size(PhysicalSize::new(r.w, r.h));
                }
            }
        }

        let Some(frame) = queue.pop_timeout(Duration::from_millis(2)) else {
            continue;
        };
        let arrived = Instant::now();
        if let Some(prev) = last_real {
            interval.observe(arrived - prev);
        }
        last_real = Some(arrived);

        gfx.upload(&frame, profile);
        let t_gpu = Instant::now();
        let outs: Vec<wgpu::Texture> = gfx.pipeline.process(gfx.input.as_ref().unwrap()).to_vec();
        let n = outs.len() as u32;
        let out_size = (outs[0].width(), outs[0].height());
        let process_ms = t_gpu.elapsed().as_secs_f32() * 1000.0;

        let step_interval = interval.interval().unwrap_or(Duration::from_millis(16));
        let mut latency_ms = None;
        let offsets = present_offsets(step_interval, n, profile.latency.fps_cap);
        let min_gap = if profile.latency.fps_cap > 0 {
            Duration::from_secs_f64(1.0 / profile.latency.fps_cap as f64)
        } else {
            Duration::ZERO
        };
        for (tex, off) in outs.iter().zip(offsets) {
            let due = (arrived + off).max(last_present + min_gap);
            wait_until(due, precise, spin);
            gfx.present(tex)?;
            last_present = Instant::now();
            latency_ms.get_or_insert(
                last_present.duration_since(frame.captured_at).as_secs_f32() * 1000.0,
            );
            out_count += 1;
        }

        if fps_window.elapsed() >= Duration::from_millis(500) {
            let secs = fps_window.elapsed().as_secs_f32();
            let mut s = stats.lock().unwrap();
            s.adapter = gfx.adapter_name.clone();
            s.output_fps = out_count as f32 / secs;
            s.source_fps = interval
                .interval()
                .map(|d| 1.0 / d.as_secs_f32())
                .unwrap_or(0.0);
            s.process_ms = process_ms;
            s.latency_ms = latency_ms.unwrap_or(s.latency_ms);
            s.capture_size = (frame.width, frame.height);
            s.output_size = out_size;
            s.dropped_frames = queue.dropped.load(Ordering::Relaxed);
            out_count = 0;
            fps_window = Instant::now();
        }
    }
    Ok(())
}
