mod app;
mod overlay;
mod platform;
mod source;
mod store;
#[cfg(windows)]
mod windows_capture_source;

/// `gs-app --selftest SECONDS`: run the test pattern through the real pipeline and overlay
/// window with no UI, then print what happened. Used to verify a build on a new machine.
fn selftest(seconds: u64) -> i32 {
    use std::sync::{atomic::AtomicBool, Arc, Mutex};
    // The one-click Smooth Motion preset, i.e. exactly what the UI button runs.
    let profile = gs_core::config::Profile::smooth_motion();
    let stop = Arc::new(AtomicBool::new(false));
    let stats = Arc::new(Mutex::new(overlay::Stats::default()));
    let (s2, st2) = (stop.clone(), stats.clone());
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(seconds));
        s2.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    let src = Box::new(source::TestPattern::new(640, 360, 30));
    let r = overlay::run_blocking(src, profile, overlay::Placement::MatchTarget, stop, st2);
    let s = stats.lock().unwrap().clone();
    println!("{s:#?}");
    match r {
        // Speed depends on the GPU (software rendering is slow), so check behaviour instead:
        // with a 2x multiplier, presented frames must be ~2x captured frames.
        Ok(()) if s.source_fps > 0.5 && (1.7..2.3).contains(&(s.output_fps / s.source_fps)) => {
            println!(
                "SELFTEST OK: {:.1} fps in -> {:.1} fps out (x{:.2})",
                s.source_fps,
                s.output_fps,
                s.output_fps / s.source_fps
            );
            0
        }
        Ok(()) => {
            println!("SELFTEST FAILED: expected ~2x output/source frame ratio");
            1
        }
        Err(e) => {
            println!("SELFTEST FAILED: {e:#}");
            1
        }
    }
}

fn main() -> eframe::Result<()> {
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--selftest") {
        let secs = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(5);
        std::process::exit(selftest(secs));
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([980.0, 680.0])
            .with_title("GPUScale"),
        ..Default::default()
    };
    eframe::run_native(
        "GPUScale",
        options,
        Box::new(|_cc| Ok(Box::new(app::App::new()))),
    )
}
