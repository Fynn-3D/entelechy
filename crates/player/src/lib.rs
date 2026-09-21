#![warn(missing_docs)]
//! Presentation helpers for the Player, kept free of windowing so they can
//! be tested headlessly.

pub mod audio;
pub mod gamepad;
pub mod hud;

pub use orla_gpu::present::{Placement, placement};

/// Maps a window position to output pixel coordinates; positions in the
/// letterbox clamp to the nearest edge.
pub fn to_output(
    placement: &Placement,
    window_x: f32,
    window_y: f32,
    width: u32,
    height: u32,
) -> [f32; 2] {
    if placement.width == 0 || placement.height == 0 {
        return [0.0, 0.0];
    }
    let x = (window_x - placement.x as f32) * width as f32 / placement.width as f32;
    let y = (window_y - placement.y as f32) * height as f32 / placement.height as f32;
    [
        x.clamp(0.0, width.saturating_sub(1) as f32),
        y.clamp(0.0, height.saturating_sub(1) as f32),
    ]
}

/// Converts linear float pixels to `0x00RRGGBB` and scales them with
/// nearest-neighbour sampling into a `target_width x target_height` frame,
/// preserving aspect ratio and letterboxing with black. Returns where the
/// image was placed. This is the CPU path; the GPU backend presents
/// resident images without it.
pub fn present(
    pixels: &[[f32; 4]],
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
    frame: &mut [u32],
) -> Placement {
    frame.fill(0);
    let placed = placement(width, height, target_width, target_height);
    if placed.width == 0 || placed.height == 0 {
        return placed;
    }
    for ty in 0..placed.height {
        let sy = (ty as u64 * height as u64 / placed.height as u64) as usize;
        let row = &pixels[sy * width as usize..(sy + 1) * width as usize];
        let target_row = (placed.y + ty) as usize * target_width as usize;
        for tx in 0..placed.width {
            let sx = (tx as u64 * width as u64 / placed.width as u64) as usize;
            let [r, g, b, _] = row[sx];
            let channel = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u32;
            frame[target_row + (placed.x + tx) as usize] =
                (channel(r) << 16) | (channel(g) << 8) | channel(b);
        }
    }
    placed
}

/// Browser-style key code for a key name as winit reports it: single
/// characters by ASCII uppercase, plus the named keys Kernels commonly use.
pub fn key_code(name: &str) -> Option<i32> {
    let mut chars = name.chars();
    if let (Some(c), None) = (chars.next(), chars.next()) {
        let upper = c.to_ascii_uppercase();
        return match upper {
            'A'..='Z' | '0'..='9' => Some(upper as i32),
            ' ' => Some(32),
            _ => None,
        };
    }
    Some(match name {
        "Space" => 32,
        "Enter" => 13,
        "Escape" => 27,
        "Backspace" => 8,
        "Tab" => 9,
        "Shift" => 16,
        "Control" => 17,
        "Alt" => 18,
        "ArrowLeft" => 37,
        "ArrowUp" => 38,
        "ArrowRight" => 39,
        "ArrowDown" => 40,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterboxes_and_scales_without_distortion() {
        let pixels = vec![[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]];
        let mut frame = vec![0xFFFF_FFFF; 8 * 2];
        let placed = present(&pixels, 2, 1, 8, 2, &mut frame);
        assert_eq!(
            placed,
            Placement {
                x: 2,
                y: 0,
                width: 4,
                height: 2
            }
        );
        assert_eq!(frame[0], 0, "left letterbox column is black");
        assert_eq!(frame[2], 0x00FF_0000);
        assert_eq!(frame[4], 0x0000_00FF);
        assert_eq!(frame[10], 0x00FF_0000, "second row is a scaled copy");
        let mut frame = vec![0xFFFF_FFFF; 4 * 4];
        present(&pixels, 2, 1, 4, 4, &mut frame);
        assert_eq!(frame[0], 0, "top letterbox row is black");
        assert_eq!(frame[4], 0x00FF_0000);
        assert_eq!(frame[7], 0x0000_00FF);
        assert_eq!(frame[12], 0, "bottom letterbox row is black");
    }

    #[test]
    fn window_positions_map_to_output_pixels() {
        let placed = Placement {
            x: 2,
            y: 0,
            width: 4,
            height: 2,
        };
        assert_eq!(to_output(&placed, 2.0, 0.0, 2, 1), [0.0, 0.0]);
        assert_eq!(to_output(&placed, 4.0, 1.0, 2, 1), [1.0, 0.0]);
        assert_eq!(
            to_output(&placed, 0.0, 5.0, 2, 1),
            [0.0, 0.0],
            "letterbox clamps"
        );
        assert_eq!(key_code("a"), Some(65));
        assert_eq!(key_code("ArrowLeft"), Some(37));
        assert_eq!(key_code("F5"), None);
    }
}
