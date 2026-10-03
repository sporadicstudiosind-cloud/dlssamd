//! The few OS-specific calls the render thread needs.

/// Raise the calling thread's priority and (on Windows) ask for 1 ms timer resolution so
/// `sleep` is accurate enough for frame pacing.
pub fn boost_current_thread() {
    #[cfg(windows)]
    unsafe {
        use windows::Win32::Media::timeBeginPeriod;
        use windows::Win32::System::Threading::{
            GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_HIGHEST,
        };
        let _ = timeBeginPeriod(1);
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    }
}

pub fn restore_timer_resolution() {
    #[cfg(windows)]
    unsafe {
        let _ = windows::Win32::Media::timeEndPeriod(1);
    }
}
