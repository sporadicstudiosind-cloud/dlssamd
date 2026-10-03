//! WGSL sources, embedded at compile time.

pub const UPSCALE: &str = include_str!("shaders/upscale.wgsl");
pub const SHARPEN: &str = include_str!("shaders/sharpen.wgsl");
pub const FLOW: &str = include_str!("shaders/flow.wgsl");
pub const LUMA: &str = include_str!("shaders/luma.wgsl");
pub const DOWN: &str = include_str!("shaders/down.wgsl");
pub const INTERPOLATE: &str = include_str!("shaders/interpolate.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse + validate every shader without needing a GPU.
    fn validate(name: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{name}: parse error:\n{}", e.emit_to_string(src)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: validation error: {e:?}"));
    }

    #[test]
    fn upscale_valid() {
        validate("upscale", UPSCALE);
    }
    #[test]
    fn sharpen_valid() {
        validate("sharpen", SHARPEN);
    }
    #[test]
    fn flow_valid() {
        validate("flow", FLOW);
    }
    #[test]
    fn luma_valid() {
        validate("luma", LUMA);
    }
    #[test]
    fn down_valid() {
        validate("down", DOWN);
    }
    #[test]
    fn interpolate_valid() {
        validate("interpolate", INTERPOLATE);
    }
}
