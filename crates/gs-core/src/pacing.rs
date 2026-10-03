//! Frame pacing and latency arithmetic. Pure functions so they're unit-testable.

use std::time::Duration;

/// Exponential moving average of the real (source) frame interval.
#[derive(Debug, Clone)]
pub struct IntervalEstimator {
    ema_us: Option<f64>,
    alpha: f64,
}

impl IntervalEstimator {
    pub fn new(alpha: f64) -> Self {
        Self { ema_us: None, alpha: alpha.clamp(0.01, 1.0) }
    }

    /// Feed the time since the previous real frame. Outliers (>4x the average, e.g. a
    /// stall or a static screen) are ignored so one hitch doesn't skew pacing.
    pub fn observe(&mut self, dt: Duration) {
        let us = dt.as_secs_f64() * 1e6;
        match self.ema_us {
            None => self.ema_us = Some(us),
            Some(avg) if us > avg * 4.0 => {}
            Some(avg) => self.ema_us = Some(avg + self.alpha * (us - avg)),
        }
    }

    pub fn interval(&self) -> Option<Duration> {
        self.ema_us.map(|u| Duration::from_secs_f64(u / 1e6))
    }
}

/// Interpolation positions in (0,1) for `multiplier` output frames per real frame.
/// multiplier=2 -> [0.5]; 4 -> [0.25, 0.5, 0.75].
pub fn interpolation_phases(multiplier: u32) -> Vec<f32> {
    let m = multiplier.max(1);
    (1..m).map(|k| k as f32 / m as f32).collect()
}

/// Present schedule for one real-frame interval: offsets from the moment the newest real
/// frame arrived. Generated frames are spread across the *previous* interval's length
/// (we display `prev -> curr` while `curr` is being received), ending with `curr` itself.
pub fn present_offsets(interval: Duration, multiplier: u32, fps_cap: u32) -> Vec<Duration> {
    let m = multiplier.max(1);
    let mut step = interval / m;
    if fps_cap > 0 {
        let min_step = Duration::from_secs_f64(1.0 / fps_cap as f64);
        if step < min_step {
            step = min_step;
        }
    }
    (0..m).map(|k| step * k).collect()
}

/// Rough end-to-end added latency in milliseconds, for the UI's latency readout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatencyEstimate {
    pub capture_ms: f32,
    pub interpolation_hold_ms: f32,
    pub processing_ms: f32,
    pub present_ms: f32,
    pub total_ms: f32,
}

pub fn estimate_latency(
    source_fps: f32,
    multiplier: u32,
    frame_gen_active: bool,
    capture_queue_depth: u32,
    max_frame_latency: u32,
    display_hz: f32,
    processing_ms: f32,
) -> LatencyEstimate {
    let src_ms = 1000.0 / source_fps.max(1.0);
    // Capture: on average half a source interval plus any queued frames we didn't drop.
    let capture_ms = src_ms * 0.5 + src_ms * (capture_queue_depth.saturating_sub(1)) as f32;
    // Interpolation must hold the newest real frame until the in-between ones are shown.
    let hold = if frame_gen_active { src_ms } else { 0.0 };
    let _ = multiplier;
    let refresh_ms = 1000.0 / display_hz.max(1.0);
    let present_ms = refresh_ms * 0.5 + refresh_ms * (max_frame_latency.saturating_sub(1)) as f32;
    LatencyEstimate {
        capture_ms,
        interpolation_hold_ms: hold,
        processing_ms,
        present_ms,
        total_ms: capture_ms + hold + processing_ms + present_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phases() {
        assert_eq!(interpolation_phases(1), Vec::<f32>::new());
        assert_eq!(interpolation_phases(2), vec![0.5]);
        assert_eq!(interpolation_phases(4), vec![0.25, 0.5, 0.75]);
    }

    #[test]
    fn estimator_ignores_stalls() {
        let mut e = IntervalEstimator::new(0.2);
        for _ in 0..20 {
            e.observe(Duration::from_millis(16));
        }
        e.observe(Duration::from_millis(500));
        let i = e.interval().unwrap().as_secs_f64() * 1000.0;
        assert!((i - 16.0).abs() < 0.5, "{i}");
    }

    #[test]
    fn offsets_respect_cap() {
        let o = present_offsets(Duration::from_millis(33), 3, 0);
        assert_eq!(o.len(), 3);
        assert_eq!(o[0], Duration::ZERO);
        let capped = present_offsets(Duration::from_millis(33), 4, 60);
        assert!(capped[1] >= Duration::from_secs_f64(1.0 / 60.0));
    }

    #[test]
    fn framegen_adds_one_source_interval() {
        let off = estimate_latency(60.0, 2, false, 1, 1, 144.0, 2.0);
        let on = estimate_latency(60.0, 2, true, 1, 1, 144.0, 2.0);
        assert!((on.total_ms - off.total_ms - 1000.0 / 60.0).abs() < 1e-3);
    }
}
