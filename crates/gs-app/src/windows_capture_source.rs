//! Windows Graphics Capture source (Windows only).
//!
//! Frames arrive as CPU-readable buffers, so every frame costs a GPU->CPU readback and a
//! CPU->GPU upload. That's the simple path; a zero-copy D3D11 shared-texture path is the
//! obvious next optimisation.

use crate::source::{Frame, FrameQueue, FrameSource, ScreenRect};
use anyhow::{anyhow, Result};
use std::sync::Arc;
use std::time::Instant;
use windows_capture::capture::{CaptureControl, Context, GraphicsCaptureApiHandler};
use windows_capture::frame::Frame as WgcFrame;
use windows_capture::graphics_capture_api::InternalCaptureControl;
use windows_capture::settings::{
    ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
    MinimumUpdateIntervalSettings, SecondaryWindowSettings, Settings,
};
use windows_capture::window::Window;

struct Handler {
    queue: Arc<FrameQueue>,
    scratch: Vec<u8>,
}

impl GraphicsCaptureApiHandler for Handler {
    type Flags = Arc<FrameQueue>;
    type Error = Box<dyn std::error::Error + Send + Sync>;

    fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
        Ok(Self {
            queue: ctx.flags,
            scratch: Vec::new(),
        })
    }

    fn on_frame_arrived(
        &mut self,
        frame: &mut WgcFrame,
        _control: InternalCaptureControl,
    ) -> Result<(), Self::Error> {
        let fb = frame.buffer()?;
        let (width, height) = (fb.width(), fb.height());
        let rgba = fb.as_nopadding_buffer(&mut self.scratch).to_vec();
        self.queue.push(Frame {
            width,
            height,
            rgba,
            captured_at: Instant::now(),
        });
        Ok(())
    }
}

pub struct WindowCapture {
    window: Window,
    title: String,
    control: Option<CaptureControl<Handler, Box<dyn std::error::Error + Send + Sync>>>,
}

impl WindowCapture {
    pub fn new(window: Window) -> Self {
        let title = window.title().unwrap_or_default();
        Self {
            window,
            title,
            control: None,
        }
    }
}

impl FrameSource for WindowCapture {
    fn label(&self) -> String {
        format!("Window: {}", self.title)
    }

    fn start(&mut self, queue: Arc<FrameQueue>) -> Result<()> {
        if !self.window.is_valid() {
            return Err(anyhow!("that window no longer exists"));
        }
        let settings = Settings::new(
            self.window,
            CursorCaptureSettings::WithoutCursor,
            DrawBorderSettings::WithoutBorder,
            SecondaryWindowSettings::Default,
            MinimumUpdateIntervalSettings::Default,
            DirtyRegionSettings::Default,
            ColorFormat::Rgba8,
            queue,
        );
        let control = Handler::start_free_threaded(settings)
            .map_err(|e| anyhow!("could not start capture: {e}"))?;
        self.control = Some(control);
        Ok(())
    }

    fn stop(&mut self) {
        if let Some(c) = self.control.take() {
            let _ = c.stop();
        }
    }

    fn screen_rect(&self) -> Option<ScreenRect> {
        let r = self.window.rect().ok()?;
        Some(ScreenRect {
            x: r.left,
            y: r.top,
            w: (r.right - r.left).max(1) as u32,
            h: (r.bottom - r.top).max(1) as u32,
        })
    }
}

/// (title, process name) of every capturable top-level window.
pub fn list_windows() -> Vec<(Window, String, String)> {
    Window::enumerate()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|w| {
            let title = w.title().ok()?;
            if title.trim().is_empty() {
                return None;
            }
            let exe = w.process_name().unwrap_or_default();
            Some((w, title, exe))
        })
        .collect()
}
