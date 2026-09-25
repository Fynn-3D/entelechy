//! One line of frame data over the picture, toggled by F10: rendered on
//! the CPU with a five-by-seven font into `0x00RRGGBB` pixels, which the
//! GPU presenter samples as an overlay and the CPU path stamps into its
//! frame. Nothing here knows about windows, or keeps its own time: the
//! seconds are the host's clock's.

use std::time::Duration;

/// A glyph: seven rows of five bits, the high bit leftmost.
type Glyph = [u8; 7];

const GLYPHS: &[(char, Glyph)] = &[
    ('0', [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E]),
    ('1', [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E]),
    ('2', [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F]),
    ('3', [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E]),
    ('4', [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02]),
    ('5', [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E]),
    ('6', [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E]),
    ('7', [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08]),
    ('8', [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E]),
    ('9', [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C]),
    ('a', [0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F]),
    ('b', [0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x1E]),
    ('c', [0x00, 0x00, 0x0E, 0x10, 0x10, 0x11, 0x0E]),
    ('d', [0x01, 0x01, 0x0D, 0x13, 0x11, 0x11, 0x0F]),
    ('e', [0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E]),
    ('f', [0x06, 0x09, 0x08, 0x1C, 0x08, 0x08, 0x08]),
    ('g', [0x00, 0x0F, 0x11, 0x11, 0x0F, 0x01, 0x0E]),
    ('h', [0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x11]),
    ('i', [0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E]),
    ('j', [0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0C]),
    ('k', [0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12]),
    ('l', [0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E]),
    ('m', [0x00, 0x00, 0x1A, 0x15, 0x15, 0x11, 0x11]),
    ('n', [0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11]),
    ('o', [0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E]),
    ('p', [0x00, 0x00, 0x1E, 0x11, 0x1E, 0x10, 0x10]),
    ('q', [0x00, 0x00, 0x0D, 0x13, 0x0F, 0x01, 0x01]),
    ('r', [0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10]),
    ('s', [0x00, 0x00, 0x0E, 0x10, 0x0E, 0x01, 0x1E]),
    ('t', [0x08, 0x08, 0x1C, 0x08, 0x08, 0x09, 0x06]),
    ('u', [0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0D]),
    ('v', [0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04]),
    ('w', [0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A]),
    ('x', [0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11]),
    ('y', [0x00, 0x00, 0x11, 0x11, 0x0F, 0x01, 0x0E]),
    ('z', [0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F]),
    (' ', [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
    ('.', [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C]),
    (',', [0x00, 0x00, 0x00, 0x00, 0x0C, 0x04, 0x08]),
    (':', [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00]),
    ('|', [0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04]),
    ('-', [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00]),
    ('+', [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00]),
    ('/', [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x00]),
    ('%', [0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03]),
    ('(', [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02]),
    (')', [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08]),
    ('~', [0x00, 0x00, 0x08, 0x15, 0x02, 0x00, 0x00]),
];

/// A cell is the glyph and a column of space; a line is a cell tall plus a
/// row of space above and below.
const CELL: (u32, u32) = (6, 9);
const INK: u32 = 0x00E0_E0E0;
const PAPER: u32 = 0x0000_0000;

/// The rendered line: `0x00RRGGBB` pixels, `width` per row.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Panel {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The pixels, row by row.
    pub pixels: Vec<u32>,
}

/// Renders `text` at `scale` device pixels per font pixel. Uppercase is
/// drawn as lowercase; a character the font lacks is a gap.
pub fn panel(text: &str, scale: u32) -> Panel {
    let scale = scale.max(1);
    let chars: Vec<char> = text.chars().map(|c| c.to_ascii_lowercase()).collect();
    let width = (chars.len() as u32 * CELL.0 + 1) * scale;
    let height = CELL.1 * scale;
    let mut pixels = vec![PAPER; (width * height) as usize];
    for (column, c) in chars.iter().enumerate() {
        let Some((_, glyph)) = GLYPHS.iter().find(|(name, _)| name == c) else {
            continue;
        };
        for (row, bits) in glyph.iter().enumerate() {
            for bit in 0..5 {
                if bits & (0x10 >> bit) == 0 {
                    continue;
                }
                let x0 = (column as u32 * CELL.0 + 1 + bit) * scale;
                let y0 = (row as u32 + 1) * scale;
                for y in y0..y0 + scale {
                    let start = (y * width + x0) as usize;
                    pixels[start..start + scale as usize].fill(INK);
                }
            }
        }
    }
    Panel {
        width,
        height,
        pixels,
    }
}

/// Stamps the panel into the top-left of a `0x00RRGGBB` frame, clipped.
pub fn stamp(panel: &Panel, frame: &mut [u32], frame_width: u32, frame_height: u32) {
    let width = panel.width.min(frame_width) as usize;
    for y in 0..panel.height.min(frame_height) as usize {
        let from = y * panel.width as usize;
        let to = y * frame_width as usize;
        frame[to..to + width].copy_from_slice(&panel.pixels[from..from + width]);
    }
}

/// What a frame took, smoothed, and what the last one did.
#[derive(Debug)]
pub struct Stats {
    last_start: Option<f64>,
    /// Smoothed milliseconds: between frames, evaluating, waiting for
    /// the device to finish what evaluating gave it, and presenting
    /// (which includes waiting for the display's refresh). The device's
    /// own time is `None` for a backend that has no device.
    interval: f64,
    evaluate: f64,
    device: Option<f64>,
    present: f64,
}

impl Default for Stats {
    fn default() -> Self {
        Self {
            last_start: None,
            interval: 0.0,
            evaluate: 0.0,
            device: None,
            present: 0.0,
        }
    }
}

/// An exponential average: the last few frames count, one frame does not.
fn smooth(average: &mut f64, sample: Duration) {
    let millis = sample.as_secs_f64() * 1000.0;
    *average = if *average == 0.0 {
        millis
    } else {
        *average * 0.9 + millis * 0.1
    };
}

impl Stats {
    /// Marks a frame's start; the time since the last one is its interval.
    pub fn begin(&mut self, now: f64) {
        if let Some(last) = self.last_start.replace(now) {
            smooth(
                &mut self.interval,
                Duration::from_secs_f64((now - last).max(0.0)),
            );
        }
    }

    /// Records how long evaluating, waiting for the device and
    /// presenting took this frame.
    pub fn record(&mut self, evaluate: Duration, device: Option<Duration>) {
        smooth(&mut self.evaluate, evaluate);
        if let Some(device) = device {
            smooth(self.device.get_or_insert(0.0), device);
        }
    }

    /// What presenting the frame took, once the device has presented it.
    pub fn presented(&mut self, present: Duration) {
        smooth(&mut self.present, present);
    }

    /// The line: the smoothed times, then what the caller adds.
    pub fn line(&self, rest: &str) -> String {
        let fps = if self.interval > 0.0 {
            1000.0 / self.interval
        } else {
            0.0
        };
        // `device`, not `gpu`: the backend's own name follows, and one
        // line cannot say `gpu` twice and mean two things.
        let device = match self.device {
            Some(device) => format!("device {device:.2} ms | "),
            None => String::new(),
        };
        format!(
            "frame {:.1} ms ({fps:.0} fps) | eval {:.2} ms | {device}present {:.2} ms | {rest}",
            self.interval, self.evaluate, self.present
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panel_draws_glyphs_in_cells_and_stamps_clipped() {
        let one = panel("1", 1);
        assert_eq!((one.width, one.height), (7, 9));
        // The `1`'s stem: column 3 of the glyph, rows 1 to 7 of the line.
        assert_eq!(one.pixels[(2 * 7 + 1 + 2) as usize], INK);
        assert_eq!(one.pixels[0], PAPER, "a row of space above");
        let big = panel("Ab", 2);
        assert_eq!((big.width, big.height), (26, 18));
        assert_eq!(
            big.pixels,
            panel("ab", 2).pixels,
            "uppercase draws as lowercase"
        );
        let mut frame = vec![0x00FF_FFFF; 4 * 4];
        stamp(&one, &mut frame, 4, 4);
        assert_eq!(frame[0], PAPER);
        assert_eq!(frame[3 * 4 + 1], PAPER);
        assert_eq!(frame[3 * 4 + 3], INK, "the stem, clipped to the frame");
        let mut stats = Stats::default();
        stats.begin(0.0);
        stats.record(Duration::from_millis(2), None);
        stats.presented(Duration::from_micros(500));
        assert!(
            stats
                .line("x")
                .starts_with("frame 0.0 ms (0 fps) | eval 2.00 ms | present 0.50 ms | x"),
            "a backend with no device has no time of its own: {}",
            stats.line("x")
        );
        // A backend with a device says what the device took, which is
        // then not part of presenting, and is not called `gpu`: the
        // backend's own name is already that.
        let mut stats = Stats::default();
        stats.begin(0.0);
        stats.record(Duration::from_millis(2), Some(Duration::from_micros(1450)));
        stats.presented(Duration::from_micros(500));
        assert!(
            stats.line("x").starts_with(
                "frame 0.0 ms (0 fps) | eval 2.00 ms | device 1.45 ms | present 0.50 ms | x"
            ),
            "{}",
            stats.line("x")
        );
    }
}
