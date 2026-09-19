#![warn(missing_docs)]
//! Presentation helpers for the Player, kept free of windowing so they can
//! be tested headlessly.

/// Converts linear float pixels to `0x00RRGGBB` and scales them with
/// nearest-neighbour sampling into a `target_width x target_height` frame,
/// preserving aspect ratio and letterboxing with black.
pub fn present(
    pixels: &[[f32; 4]],
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
    frame: &mut [u32],
) {
    frame.fill(0);
    if width == 0 || height == 0 || target_width == 0 || target_height == 0 {
        return;
    }
    let scale = (target_width as f32 / width as f32).min(target_height as f32 / height as f32);
    let shown_width = ((width as f32 * scale) as u32).max(1);
    let shown_height = ((height as f32 * scale) as u32).max(1);
    let x0 = (target_width - shown_width) / 2;
    let y0 = (target_height - shown_height) / 2;
    for ty in 0..shown_height {
        let sy = (ty as u64 * height as u64 / shown_height as u64) as usize;
        let row = &pixels[sy * width as usize..(sy + 1) * width as usize];
        let target_row = (y0 + ty) as usize * target_width as usize;
        for tx in 0..shown_width {
            let sx = (tx as u64 * width as u64 / shown_width as u64) as usize;
            let [r, g, b, _] = row[sx];
            let channel = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u32;
            frame[target_row + (x0 + tx) as usize] =
                (channel(r) << 16) | (channel(g) << 8) | channel(b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterboxes_and_scales_without_distortion() {
        let pixels = vec![[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]];
        // 2x1 into 8x2: scale 2 fits the height, so 4x2 is shown, centred.
        let mut frame = vec![0xFFFF_FFFF; 8 * 2];
        present(&pixels, 2, 1, 8, 2, &mut frame);
        assert_eq!(frame[0], 0, "left letterbox column is black");
        assert_eq!(frame[2], 0x00FF_0000);
        assert_eq!(frame[3], 0x00FF_0000);
        assert_eq!(frame[4], 0x0000_00FF);
        assert_eq!(frame[5], 0x0000_00FF);
        assert_eq!(frame[6], 0);
        assert_eq!(frame[10], 0x00FF_0000, "second row is a scaled copy");
        // 2x1 into 4x4: scale 2 fits the width, so 4x2 is shown, centred.
        let mut frame = vec![0xFFFF_FFFF; 4 * 4];
        present(&pixels, 2, 1, 4, 4, &mut frame);
        assert_eq!(frame[0], 0, "top letterbox row is black");
        assert_eq!(frame[4], 0x00FF_0000);
        assert_eq!(frame[5], 0x00FF_0000);
        assert_eq!(frame[6], 0x0000_00FF);
        assert_eq!(frame[7], 0x0000_00FF);
        assert_eq!(frame[12], 0, "bottom letterbox row is black");
    }
}
