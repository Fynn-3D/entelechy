//! The F10 frame data, rendered on the CPU with a 5x7 font into
//! `0x00RRGGBB` pixels for a presenter to overlay: a line of the frame's
//! times, then the nodes that kept the device busiest. Times come from the
//! host's clock, and the device's own where it keeps one.

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
    ('_', [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F]),
    ('#', [0x0A, 0x0A, 0x1F, 0x0A, 0x1F, 0x0A, 0x0A]),
];

/// A glyph plus one column of space; a glyph plus a row above and below.
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

/// Renders `text` at `scale` device pixels per font pixel, uppercase as
/// lowercase; a character outside the font is a gap.
pub fn panel(text: &str, scale: u32) -> Panel {
    lines(&[text.to_owned()], scale)
}

/// Renders `texts`, a line each, one under another, as [`panel`] does.
pub fn lines(texts: &[String], scale: u32) -> Panel {
    lines_at_least(texts, scale, 0)
}

/// [`lines`], the panel at least `columns` wide.
pub fn lines_at_least(texts: &[String], scale: u32, columns: usize) -> Panel {
    let scale = scale.max(1);
    let rows: Vec<Vec<char>> = texts
        .iter()
        .map(|text| text.chars().map(|c| c.to_ascii_lowercase()).collect())
        .collect();
    let longest = rows.iter().map(Vec::len).max().unwrap_or(0).max(columns) as u32;
    let width = (longest * CELL.0 + 1) * scale;
    let height = CELL.1 * scale * rows.len().max(1) as u32;
    let mut pixels = vec![PAPER; (width * height) as usize];
    for (line, chars) in rows.iter().enumerate() {
        let top = line as u32 * CELL.1 * scale;
        draw(chars, &mut pixels, width, top, scale);
    }
    Panel {
        width,
        height,
        pixels,
    }
}

fn draw(chars: &[char], pixels: &mut [u32], width: u32, top: u32, scale: u32) {
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
                let y0 = top + (row as u32 + 1) * scale;
                for y in y0..y0 + scale {
                    let start = (y * width + x0) as usize;
                    pixels[start..start + scale as usize].fill(INK);
                }
            }
        }
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

/// Smoothed frame times.
#[derive(Debug)]
pub struct Stats {
    last_start: Option<f64>,
    /// Smoothed milliseconds: frame interval; the processor's work on a
    /// frame, its waits left out; the device's busy time on one by its own
    /// clock (`None` without one); presenting (including vsync).
    interval: f64,
    cpu: f64,
    gpu: Option<f64>,
    present: f64,
    /// Each node's device milliseconds, and those between nodes, as
    /// shown: averaged over the last [`SHOWN_EVERY`] clocked frames told
    /// ([`Stats::told`]) and held until the next as many.
    nodes: Vec<(String, f64)>,
    between: f64,
    /// What the clocked frames told since the last shown add up to, and
    /// how many they are.
    summed: Vec<(String, f64)>,
    summed_between: f64,
    summed_frames: u32,
    /// The latest clocked frame told.
    told: Option<u64>,
    /// The latest frame's start, and the lines shown since `held_at`:
    /// numbers that change every frame cannot be read, so the lines are
    /// made again [`HELD_FOR`] seconds apart.
    now: f64,
    held: Vec<String>,
    held_at: f64,
    /// The most columns a shown line has taken: the panel is never
    /// narrower, so it does not twitch as a line comes and goes.
    widest: usize,
}

/// Seconds the shown lines are held before they are made again.
pub const HELD_FOR: f64 = 0.25;

/// Groups laid left to right, ` | ` between them, a line taking as many
/// as fit in `columns`; a group is never broken.
pub fn flow(groups: &[String], columns: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    for group in groups.iter().filter(|g| !g.is_empty()) {
        match lines.last_mut() {
            Some(line) if line.chars().count() + 3 + group.chars().count() <= columns => {
                line.push_str(" | ");
                line.push_str(group);
            }
            _ => lines.push(group.clone()),
        }
    }
    lines
}

/// The clocked frames each shown line of nodes averages, about half a
/// second: a line that changes every frame cannot be read.
const SHOWN_EVERY: u32 = 30;

impl Default for Stats {
    fn default() -> Self {
        Self {
            last_start: None,
            interval: 0.0,
            cpu: 0.0,
            gpu: None,
            present: 0.0,
            nodes: Vec::new(),
            between: 0.0,
            summed: Vec::new(),
            summed_between: 0.0,
            summed_frames: 0,
            told: None,
            now: 0.0,
            held: Vec::new(),
            held_at: f64::NEG_INFINITY,
            widest: 0,
        }
    }
}

/// An exponential moving average, weight 0.1 per sample.
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
        self.now = now;
        if let Some(last) = self.last_start.replace(now) {
            smooth(
                &mut self.interval,
                Duration::from_secs_f64((now - last).max(0.0)),
            );
        }
    }

    /// Records a frame's processor time and the device's busy time on the
    /// latest frame it has done.
    pub fn record(&mut self, cpu: Duration, gpu: Option<Duration>) {
        smooth(&mut self.cpu, cpu);
        match gpu {
            Some(gpu) => smooth(self.gpu.get_or_insert(0.0), gpu),
            None => self.gpu = None,
        }
    }

    /// Records a clocked frame's device time node by node, once a frame:
    /// a node not in it is taken as idle. The first frame is shown at
    /// once, then each [`SHOWN_EVERY`] frames' average.
    pub fn told(&mut self, times: &crate::DeviceTimes) {
        if self.told == Some(times.frame) {
            return;
        }
        let first = self.told.is_none();
        self.told = Some(times.frame);
        for (node, busy) in &times.nodes {
            let name = node.to_string();
            let millis = busy.as_secs_f64() * 1000.0;
            match self.summed.iter_mut().find(|(n, _)| *n == name) {
                Some((_, sum)) => *sum += millis,
                None => self.summed.push((name, millis)),
            }
        }
        self.summed_between += times.between.as_secs_f64() * 1000.0;
        self.summed_frames += 1;
        if first || self.summed_frames >= SHOWN_EVERY {
            let frames = f64::from(self.summed_frames);
            self.nodes = std::mem::take(&mut self.summed)
                .into_iter()
                .map(|(name, sum)| (name, sum / frames))
                .filter(|(_, average)| *average >= 0.005)
                .collect();
            self.between = self.summed_between / frames;
            self.summed_between = 0.0;
            self.summed_frames = 0;
        }
    }

    /// The nodes that kept the device busiest, `most` of them, each
    /// `name ms` with its milliseconds in a fixed width, then what ran
    /// between nodes; the first headed `gpu by node:`.
    pub fn busiest(&self, most: usize) -> Vec<String> {
        if self.told.is_none() {
            return Vec::new();
        }
        let mut nodes = self.nodes.clone();
        nodes.sort_by(|a, b| b.1.total_cmp(&a.1));
        nodes.truncate(most);
        let mut said: Vec<String> = nodes
            .iter()
            .map(|(name, ms)| format!("{name} {ms:>6.2}"))
            .collect();
        match said.first_mut() {
            Some(first) => *first = format!("gpu by node: {first}"),
            None => said.push("gpu by node:".into()),
        }
        said.push(format!("between nodes {:>6.2}", self.between));
        said
    }

    /// `make`'s lines, made again only [`HELD_FOR`] seconds after the
    /// last were.
    pub fn held(&mut self, make: impl FnOnce(&Self) -> Vec<String>) -> Vec<String> {
        if self.held.is_empty() || self.now - self.held_at >= HELD_FOR || self.now < self.held_at {
            self.held = make(self);
            self.held_at = self.now;
            let longest = self
                .held
                .iter()
                .map(|l| l.chars().count())
                .max()
                .unwrap_or(0);
            self.widest = self.widest.max(longest);
        }
        self.held.clone()
    }

    /// The most columns a shown line has taken.
    pub fn widest(&self) -> usize {
        self.widest
    }

    /// Records this frame's presenting time.
    pub fn presented(&mut self, present: Duration) {
        smooth(&mut self.present, present);
    }

    /// The smoothed times, each number in a fixed width.
    pub fn times(&self) -> String {
        let fps = if self.interval > 0.0 {
            1000.0 / self.interval
        } else {
            0.0
        };
        let gpu = match self.gpu {
            Some(gpu) => format!("gpu {gpu:>6.2} ms | "),
            None => String::new(),
        };
        format!(
            "frame {:>6.1} ms ({fps:>3.0} fps) | cpu {:>6.2} ms | {gpu}present {:>6.2} ms",
            self.interval, self.cpu, self.present
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
        assert_eq!(
            stats.times(),
            "frame    0.0 ms (  0 fps) | cpu   2.00 ms | present   0.50 ms",
            "a backend with no device has no time of its own"
        );
        // A device with a clock reports its busy time apart from presenting.
        let mut stats = Stats::default();
        stats.begin(0.0);
        stats.record(Duration::from_millis(2), Some(Duration::from_micros(1450)));
        stats.presented(Duration::from_micros(500));
        assert_eq!(
            stats.times(),
            "frame    0.0 ms (  0 fps) | cpu   2.00 ms | gpu   1.45 ms | present   0.50 ms"
        );
        // Groups flow left to right, as many to a line as fit, none broken.
        let groups = ["aaaa", "bbbb", "cc", "", "dddddd"].map(String::from);
        assert_eq!(flow(&groups, 11), vec!["aaaa | bbbb", "cc | dddddd"]);
        assert_eq!(flow(&groups, 100), vec!["aaaa | bbbb | cc | dddddd"]);
        assert_eq!(flow(&groups, 3), vec!["aaaa", "bbbb", "cc", "dddddd"]);
        // Lines are held a quarter of a second; the panel never narrows.
        let mut made = 0;
        for (now, want) in [(0.0, 1), (0.1, 1), (0.2, 1), (0.3, 2), (0.4, 2), (0.6, 3)] {
            stats.begin(now);
            stats.held(|_| {
                made += 1;
                vec![format!("{made}")]
            });
            assert_eq!(made, want, "at {now}");
        }
        assert_eq!(stats.widest(), 1);
    }

    #[test]
    fn the_busiest_nodes_follow_the_line_averaged_and_held_half_a_second() {
        let id = |n: &str| entelechy_graph::NodeId(n.to_owned());
        let ms = Duration::from_millis;
        let mut stats = Stats::default();
        assert!(stats.busiest(3).is_empty(), "nothing told, nothing said");
        stats.told(&crate::DeviceTimes {
            frame: 1,
            busy: ms(7),
            nodes: vec![
                (id("look"), ms(4)),
                (id("raster.drawn"), ms(2)),
                (id("hud_top"), ms(1)),
            ],
            between: Duration::ZERO,
        });
        assert_eq!(
            stats.busiest(3),
            [
                "gpu by node: look   4.00",
                "raster.drawn   2.00",
                "hud_top   1.00",
                "between nodes   0.00"
            ]
            .map(String::from)
        );
        // The same frame told again changes nothing; later ones are held
        // until as many as a shown line averages are told, a frame without
        // a node counting it idle.
        stats.told(&crate::DeviceTimes {
            frame: 1,
            busy: ms(7),
            nodes: Vec::new(),
            between: Duration::ZERO,
        });
        assert!(stats.busiest(1)[0].ends_with("4.00"));
        for frame in 2..2 + u64::from(SHOWN_EVERY) {
            let look = if frame % 2 == 0 { ms(1) } else { ms(3) };
            stats.told(&crate::DeviceTimes {
                frame,
                busy: ms(4),
                nodes: vec![(id("look"), look), (id("raster.drawn"), ms(2))],
                between: Duration::ZERO,
            });
            if frame < 1 + u64::from(SHOWN_EVERY) {
                assert!(stats.busiest(1)[0].ends_with("4.00"), "held");
            }
        }
        assert!(stats.busiest(2)[0].ends_with("2.00"));
        // Lines stack, each as wide as the longest, or as asked.
        let two = lines(&["1".to_string(), "_".to_string()], 1);
        assert_eq!((two.width, two.height), (7, 18));
        assert_eq!(lines_at_least(&["1".to_string()], 1, 5).width, 31);
        assert_eq!(
            two.pixels[(16 * 7 + 1) as usize],
            INK,
            "the underscore, on the second line's last row"
        );
    }
}
