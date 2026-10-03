//! Settings persistence (TOML in the user's config directory).

use anyhow::{Context, Result};
use gs_core::config::Settings;
use std::path::PathBuf;

pub fn settings_path() -> PathBuf {
    let base = std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from))
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("gpuscale").join("settings.toml")
}

pub fn load() -> Settings {
    let path = settings_path();
    match std::fs::read_to_string(&path) {
        Ok(text) => Settings::from_toml(&text).unwrap_or_else(|e| {
            log::warn!("ignoring unreadable settings at {}: {e}", path.display());
            Settings::default()
        }),
        Err(_) => Settings::default(),
    }
}

pub fn save(settings: &Settings) -> Result<()> {
    let path = settings_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    // Write-then-rename so a crash can't leave a half-written file.
    let tmp = path.with_extension("toml.tmp");
    std::fs::write(&tmp, settings.to_toml()?)
        .with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, &path).with_context(|| format!("replace {}", path.display()))?;
    Ok(())
}
