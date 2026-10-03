//! Frame sources. A source pushes tightly-packed RGBA8 frames into a [`FrameQueue`].

use anyhow::Result;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// Tightly packed RGBA8, `width * height * 4` bytes.
    pub rgba: Vec<u8>,
    pub captured_at: Instant,
}

/// Bounded "newest wins" queue. With depth 1 a slow consumer always sees the latest frame,
/// which is what you want for latency; deeper queues trade latency for smoothness.
pub struct FrameQueue {
    inner: Mutex<VecDeque<Frame>>,
    cv: Condvar,
    depth: usize,
    pub dropped: std::sync::atomic::AtomicU64,
}

impl FrameQueue {
    pub fn new(depth: usize) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
            depth: depth.max(1),
            dropped: Default::default(),
        })
    }

    pub fn push(&self, f: Frame) {
        let mut q = self.inner.lock().unwrap();
        while q.len() >= self.depth {
            q.pop_front();
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        q.push_back(f);
        self.cv.notify_one();
    }

    pub fn pop_timeout(&self, timeout: Duration) -> Option<Frame> {
        let mut q = self.inner.lock().unwrap();
        if let Some(f) = q.pop_front() {
            return Some(f);
        }
        let (mut q, _) = self
            .cv
            .wait_timeout_while(q, timeout, |q| q.is_empty())
            .unwrap();
        q.pop_front()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScreenRect {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}

pub trait FrameSource: Send {
    fn label(&self) -> String;
    fn start(&mut self, queue: Arc<FrameQueue>) -> Result<()>;
    fn stop(&mut self);
    /// Where the source currently is on screen, so the overlay can follow it.
    /// `None` = no meaningful screen position (the overlay opens as a normal window).
    fn screen_rect(&self) -> Option<ScreenRect>;
}

/// Synthetic moving pattern at a fixed FPS. Lets you exercise the whole pipeline (upscale,
/// frame generation, pacing, presentation) without a game, on any OS.
pub struct TestPattern {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl TestPattern {
    pub fn new(width: u32, height: u32, fps: u32) -> Self {
        Self {
            width,
            height,
            fps: fps.clamp(1, 240),
            stop: Default::default(),
            thread: None,
        }
    }
}

/// Deterministic scene: a textured background panning right plus a bouncing box, so both
/// global and local motion are present.
pub fn render_test_frame(width: u32, height: u32, t: f32) -> Vec<u8> {
    let mut out = vec![255u8; (width * height * 4) as usize];
    let pan = t * 90.0;
    let bx = ((t * 140.0) % (2.0 * (width as f32 - 80.0))).abs();
    let bx = if bx > width as f32 - 80.0 {
        2.0 * (width as f32 - 80.0) - bx
    } else {
        bx
    };
    let by = height as f32 * 0.5 + (t * 3.0).sin() * height as f32 * 0.25;
    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 - pan;
            let fy = y as f32;
            let mut c = [
                0.45 + 0.25 * (fx * 0.045).sin() * (fy * 0.03).cos(),
                0.45 + 0.25 * (fx * 0.03 + fy * 0.02).sin(),
                0.5 + 0.2 * ((fx + fy) * 0.025).cos(),
            ];
            // Thin grid lines stress fine detail.
            if (fx.rem_euclid(48.0)) < 1.5 || fy.rem_euclid(48.0) < 1.5 {
                c = [0.1, 0.1, 0.12];
            }
            if x as f32 >= bx
                && (x as f32) < bx + 80.0
                && (y as f32) >= by - 40.0
                && (y as f32) < by + 40.0
            {
                c = [0.95, 0.35, 0.1];
            }
            // Static HUD bar: should survive frame generation untouched.
            if y < 24 && x < width / 3 {
                c = [0.05, 0.05, 0.05];
            }
            let i = ((y * width + x) * 4) as usize;
            for k in 0..3 {
                out[i + k] = (c[k].clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
    }
    out
}

impl FrameSource for TestPattern {
    fn label(&self) -> String {
        format!(
            "Test pattern {}x{} @ {} fps",
            self.width, self.height, self.fps
        )
    }

    fn start(&mut self, queue: Arc<FrameQueue>) -> Result<()> {
        let (w, h, fps) = (self.width, self.height, self.fps);
        let stop = self.stop.clone();
        stop.store(false, Ordering::SeqCst);
        self.thread = Some(std::thread::spawn(move || {
            let start = Instant::now();
            let mut n = 0u64;
            while !stop.load(Ordering::SeqCst) {
                let due = start + Duration::from_secs_f64(n as f64 / fps as f64);
                let now = Instant::now();
                if due > now {
                    std::thread::sleep(due - now);
                }
                queue.push(Frame {
                    width: w,
                    height: h,
                    rgba: render_test_frame(w, h, n as f32 / fps as f32),
                    captured_at: Instant::now(),
                });
                n += 1;
            }
        }));
        Ok(())
    }

    fn stop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }

    fn screen_rect(&self) -> Option<ScreenRect> {
        None
    }
}

impl Drop for TestPattern {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_depth_one_keeps_newest() {
        let q = FrameQueue::new(1);
        for i in 0..3u8 {
            q.push(Frame {
                width: 1,
                height: 1,
                rgba: vec![i; 4],
                captured_at: Instant::now(),
            });
        }
        assert_eq!(q.pop_timeout(Duration::from_millis(1)).unwrap().rgba[0], 2);
        assert!(q.pop_timeout(Duration::from_millis(1)).is_none());
        assert_eq!(q.dropped.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_scene_moves_and_keeps_hud_static() {
        let a = render_test_frame(160, 90, 0.0);
        let b = render_test_frame(160, 90, 0.5);
        assert_ne!(a, b);
        // HUD pixel (5,5) identical across time.
        let i = ((5 * 160 + 5) * 4) as usize;
        assert_eq!(a[i..i + 3], b[i..i + 3]);
    }
}
