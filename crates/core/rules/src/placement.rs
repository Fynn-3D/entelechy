//! Where a picture lands in a window: fitted whole, aspect kept,
//! centered. Presenters, CPU blits and pointer mapping share this rule.

/// The image's rectangle inside the window, in window pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Placement {
    /// Left edge.
    pub x: u32,
    /// Top edge.
    pub y: u32,
    /// Shown width.
    pub width: u32,
    /// Shown height.
    pub height: u32,
}

/// Fits `width x height` into `target_width x target_height`, centered.
pub fn placement(width: u32, height: u32, target_width: u32, target_height: u32) -> Placement {
    if width == 0 || height == 0 || target_width == 0 || target_height == 0 {
        return Placement::default();
    }
    let scale = (target_width as f32 / width as f32).min(target_height as f32 / height as f32);
    let shown_width = ((width as f32 * scale) as u32).max(1);
    let shown_height = ((height as f32 * scale) as u32).max(1);
    Placement {
        x: (target_width - shown_width) / 2,
        y: (target_height - shown_height) / 2,
        width: shown_width,
        height: shown_height,
    }
}
