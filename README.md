# GPUScale

Upscale and add frames to **any windowed app** on the GPU, with a settings panel for
choosing the upscaler and frame generator. Rust + wgpu + egui. Windows first.

## What it does

1. **Captures** a window (Windows Graphics Capture).
2. **Upscales** it with a spatial kernel: nearest, bilinear, Catmull-Rom, Lanczos 2/3, a
   built-in edge-adaptive kernel, or **AMD FSR 1** (EASU + RCAS, a port of AMD's MIT-licensed
   shaders), plus contrast-adaptive sharpening.
3. **Generates frames** (2x to 8x; or one click on **Smooth Motion (2x)**, see below) with block-matching optical flow on a luma pyramid, or a
   plain-blend baseline. Regions you mark as HUD are excluded from warping, and blocks the
   matcher isn't confident about fade back to a blend.
4. **Presents** in a click-through, always-on-top overlay, paced by the latency settings
   (present mode, swapchain queue, capture queue depth, FPS cap, spin-wait pacing, priority).

Profiles can be bound to an exe name and auto-apply when you capture that program.

## Smooth Motion (2x)

One button, nothing to configure: fixed 2x frame generation from the frames the app presents
(no engine data), optical-flow based, no upscaling, and the lowest-latency presentation path.
It mirrors the *publicly documented behaviour* of NVIDIA Smooth Motion (a driver feature that
inserts one frame between every two rendered frames, with no per-game integration).

It is **not** NVIDIA's implementation: that is closed source and RTX-only, and nothing was
copied or decompiled. Their flow estimation is reportedly AI-based; this uses block matching,
so image quality will differ. It also runs as a capture overlay, not in the driver, so it
can't match a driver feature's integration or latency. For AMD's own driver-level equivalent,
see AFMF in Adrenalin.

## What it cannot do (read this)

- **DLSS, FSR 2/3/4, XeSS and vendor frame generation are not implemented.** (FSR 1 and NIS are, because they're spatial and need no engine data.) They need the
  game's motion vectors and depth, which a screen capture doesn't have. They are in the
  catalog (Backends tab) with their real status. Supporting them means per-game DLL
  injection (the Optiscaler approach), which is a separate, fragile, single-player-only
  feature. The `Backend` catalog is the extension point.
- **Reflex / Anti-Lag** are SDKs a game calls itself. An external app can't enable them for
  another process. The Latency tab tunes this app's own capture-and-present path.
- **Generated frames come from image-only motion estimation**: expect artifacts on HUDs,
  particles and thin geometry, and about one source-frame interval of added latency.
- **Mouse is not remapped** in "Fill the monitor" mode. Use "Over the source window" for
  anything you click.
- Exclusive-fullscreen games can't be captured; use windowed/borderless. Anti-cheat may flag
  overlays.
- Window capture copies frames through the CPU (GPU readback, then upload). Fine at 1080p;
  a zero-copy shared-texture path is the next optimisation.

## Status of testing

| Part | How it's verified |
|---|---|
| Settings, catalog, pacing maths | Unit tests |
| All four GPU stages | Run on a software Vulkan device (lavapipe) against synthetic images with known answers: constant-colour preservation for every upscaler, hand-computed bilinear values, and optical-flow-vs-blend error on pans with whole- and half-pixel motion |
| Capture -> process -> present loop | `gs-app --selftest` on a virtual X display: checks the 2x output/input frame ratio |
| **Windows capture + overlay** | **Type-checked for `x86_64-pc-windows-msvc` only. Never run on Windows.** Expect first-run bugs. |
| **Performance on a real GPU** | **Not measured.** Numbers from the software renderer mean nothing. |

## Build and run

```
cargo run --release -p gs-app             # GUI
cargo run --release -p gs-app -- --selftest 8
cargo test --workspace
```

Linux needs `libxkbcommon-x11` and a Vulkan driver at runtime. Only the test-pattern source
works off Windows.

## Layout

- `crates/gs-core` - settings, backend catalog, pacing, WGSL shaders, GPU pipeline. No windowing.
- `crates/gs-app` - egui settings panel, capture sources, overlay and render loop.

## Licensing

- `fsr1_easu.wgsl` / `fsr1_rcas.wgsl` are WGSL ports of AMD FidelityFX Super Resolution 1
  (MIT, (c) 2021 Advanced Micro Devices, Inc.). The licence notice is kept in each file.
- The other shaders (Lanczos/bicubic, edge-adaptive, CAS-style sharpen, optical flow,
  interpolation) are original implementations of published techniques.
- `nis_scaler.wgsl`, `nis.rs` and `nis_coefs.rs` are ports of NVIDIA Image Scaling SDK v1.0.3
  (MIT, (c) 2022 NVIDIA CORPORATION & AFFILIATES). The coefficient tables were extracted from
  `NIS_Config.h` by script, not retyped. SDR path only; the original's group-shared-memory
  tiling, viewport, NV12 and HDR paths are not ported.
- Not included, and not going to be: DLSS (proprietary, RTX-only), FSR 4 weights or leaked
  FSR 4 binaries (not open source), Lossless Scaling's LSFG (proprietary).
- Being open source does not make something usable here: FSR 2/3 are MIT but need the game's
  motion vectors and depth, so they can't run on captured frames.
