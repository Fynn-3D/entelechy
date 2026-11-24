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
    ('=', [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00]),
    ('<', [0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02]),
    ('>', [0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08]),
    ('\'', [0x04, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00]),
    ('`', [0x08, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00]),
    ('"', [0x0A, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00]),
    ('[', [0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E]),
    (']', [0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E]),
    ('!', [0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04]),
    ('?', [0x0E, 0x11, 0x01, 0x06, 0x04, 0x00, 0x04]),
    (';', [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x04, 0x08]),
    ('@', [0x0E, 0x11, 0x17, 0x15, 0x17, 0x10, 0x0E]),
    ('*', [0x00, 0x04, 0x15, 0x0E, 0x15, 0x04, 0x00]),
    ('\u{00B7}', [0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00, 0x00]),
];

/// Bold geometric lowercase letters for the `entelechy` display wordmark
/// (13 rows tall, high bit leftmost of `width` columns).
const WORDMARK_GLYPHS: &[(char, u32, [u8; 13])] = &[
    (
        'e',
        7,
        [
            0b0000000, 0b0000000, 0b0000000, 0b0111110, 0b1100011, 0b1100011, 0b1111111, 0b1100000,
            0b1100011, 0b0111110, 0b0000000, 0b0000000, 0b0000000,
        ],
    ),
    (
        'n',
        7,
        [
            0b0000000, 0b0000000, 0b0000000, 0b1101110, 0b1110011, 0b1100011, 0b1100011, 0b1100011,
            0b1100011, 0b1100011, 0b0000000, 0b0000000, 0b0000000,
        ],
    ),
    (
        't',
        6,
        [
            0b000000, 0b011000, 0b011000, 0b111110, 0b011000, 0b011000, 0b011000, 0b011000,
            0b011000, 0b001111, 0b000000, 0b000000, 0b000000,
        ],
    ),
    (
        'l',
        4,
        [
            0b1100, 0b1100, 0b1100, 0b1100, 0b1100, 0b1100, 0b1100, 0b1100, 0b1100, 0b0111, 0b0000,
            0b0000, 0b0000,
        ],
    ),
    (
        'c',
        7,
        [
            0b0000000, 0b0000000, 0b0000000, 0b0111110, 0b1100011, 0b1100000, 0b1100000, 0b1100000,
            0b1100011, 0b0111110, 0b0000000, 0b0000000, 0b0000000,
        ],
    ),
    (
        'h',
        7,
        [
            0b1100000, 0b1100000, 0b1100000, 0b1101110, 0b1110011, 0b1100011, 0b1100011, 0b1100011,
            0b1100011, 0b1100011, 0b0000000, 0b0000000, 0b0000000,
        ],
    ),
    (
        'y',
        7,
        [
            0b0000000, 0b0000000, 0b0000000, 0b1100011, 0b1100011, 0b1100011, 0b1100011, 0b1100011,
            0b0111111, 0b0000011, 0b0000011, 0b1100011, 0b0111110,
        ],
    ),
];

/// A glyph plus one column of space; a glyph plus a row above and below.
const CELL: (u32, u32) = (6, 9);
const INK: u32 = 0x00E0_E0E0;
const PAPER: u32 = 0x0000_0000;

/// The rendered line or centered stage card: `0x00RRGGBB` pixels, `width` per row.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Panel {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The pixels, row by row.
    pub pixels: Vec<u32>,
    /// Whether the panel is centered in the window instead of placed at `(0, 0)`.
    pub centered: bool,
}

/// Content of the centered Entelechy boot and empty-stage card, matching the
/// web player's Minimalist Soul 6-square Living Seed & Open DAG screen.
#[derive(Clone, Debug, PartialEq)]
pub struct StageCard {
    /// Small Solar-Gold pill badge at the top (`native · gpu` or `live · 127.0.0.1:7788`).
    pub badge: String,
    /// Scene title when loading (`None` on an empty resting stage).
    pub title: Option<String>,
    /// Loading progress in `0.0..=1.0` (`None` on an empty resting stage, where
    /// all three tiers of the emblem stand fully lit).
    pub progress: Option<f32>,
    /// Whether loading faulted.
    pub failed: bool,
    /// Status or instruction line below the progress bar.
    pub detail: String,
}

/// Builds a live [`StageCard`] from `progress` as a scene's Kernels compile
/// and build on the device.
pub fn loading_card(badge: &str, title: &str, progress: &entelechy_kernel::Progress) -> StageCard {
    use entelechy_kernel::Stage;
    let now = progress.now();
    let total = now.seeds.len();
    if total == 0 {
        return StageCard {
            badge: badge.to_owned(),
            title: Some(title.to_owned()),
            progress: Some(0.08),
            failed: false,
            detail: "preparing scene...".to_owned(),
        };
    }
    let built = now.built();
    let weighted: f32 = now
        .seeds
        .iter()
        .map(|seed| match seed.stage {
            Stage::Waiting => 0.0,
            Stage::Compiling => 0.25,
            Stage::Compiled => 0.55,
            Stage::Building => {
                let creep = 1.0 - (-seed.took.as_secs_f32() * 0.55).exp();
                0.72 + 0.24 * creep
            }
            Stage::Built => 1.0,
        })
        .sum();
    let part = (0.08 + 0.90 * (weighted / total as f32)).clamp(0.08, 0.99);
    let detail = match now.longest() {
        Some(seed) => {
            let verb = match seed.stage {
                Stage::Compiling => "compiling",
                _ => "building",
            };
            let secs = seed.took.as_secs_f32();
            if secs >= 0.2 {
                format!("{built}/{total} · {verb} {} · {secs:.1}s", seed.id)
            } else {
                format!("{built}/{total} · {verb} {}", seed.id)
            }
        }
        None => format!("{built}/{total} kernels ready"),
    };
    StageCard {
        badge: badge.to_owned(),
        title: Some(title.to_owned()),
        progress: Some(part),
        failed: false,
        detail,
    }
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
        centered: false,
    }
}

fn draw(chars: &[char], pixels: &mut [u32], width: u32, top: u32, scale: u32) {
    draw_at(chars, pixels, width, scale, top + scale, scale, INK);
}

fn draw_at(
    chars: &[char],
    pixels: &mut [u32],
    width: u32,
    left: u32,
    top: u32,
    scale: u32,
    color: u32,
) {
    for (column, c) in chars.iter().enumerate() {
        let Some((_, glyph)) = GLYPHS.iter().find(|(name, _)| name == c) else {
            continue;
        };
        for (row, bits) in glyph.iter().enumerate() {
            for bit in 0..5 {
                if bits & (0x10 >> bit) == 0 {
                    continue;
                }
                let x0 = left + (column as u32 * CELL.0 + bit) * scale;
                let y0 = top + row as u32 * scale;
                fill_rect(pixels, width, x0, y0, scale, scale, color);
            }
        }
    }
}

fn fill_rect(pixels: &mut [u32], stride: u32, x: u32, y: u32, w: u32, h: u32, color: u32) {
    let total_h = pixels.len() as u32 / stride.max(1);
    let x1 = (x + w).min(stride);
    let y1 = (y + h).min(total_h);
    if x >= x1 || y >= y1 {
        return;
    }
    for row in y..y1 {
        let start = (row * stride + x) as usize;
        let end = (row * stride + x1) as usize;
        pixels[start..end].fill(color);
    }
}

fn truncate_chars(text: &str, max_chars: usize) -> Vec<char> {
    let lower: Vec<char> = text.chars().map(|c| c.to_ascii_lowercase()).collect();
    if lower.len() <= max_chars {
        return lower;
    }
    let keep = max_chars.saturating_sub(3);
    let mut out = lower[..keep].to_vec();
    out.extend(['.', '.', '.']);
    out
}

fn wrap_card_lines(text: &str, max_cols: usize, max_lines: usize) -> Vec<Vec<char>> {
    let mut lines: Vec<String> = Vec::new();
    for raw in text.lines() {
        let mut current = String::new();
        for word in raw.split_whitespace() {
            let wlen = word.chars().count();
            let needed = if current.is_empty() {
                wlen
            } else {
                current.chars().count() + 1 + wlen
            };
            if needed <= max_cols {
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(word);
            } else {
                if !current.is_empty() {
                    lines.push(std::mem::take(&mut current));
                    if lines.len() >= max_lines {
                        break;
                    }
                }
                let mut rest = word;
                while rest.chars().count() > max_cols && lines.len() < max_lines {
                    let split = rest
                        .char_indices()
                        .nth(max_cols)
                        .map_or(rest.len(), |(idx, _)| idx);
                    lines.push(rest[..split].to_owned());
                    rest = &rest[split..];
                }
                if lines.len() >= max_lines {
                    break;
                }
                current.push_str(rest);
            }
        }
        if lines.len() >= max_lines {
            break;
        }
        if !current.is_empty() {
            lines.push(current);
            if lines.len() >= max_lines {
                break;
            }
        }
    }
    lines
        .into_iter()
        .map(|line| line.chars().map(|c| c.to_ascii_lowercase()).collect())
        .collect()
}

/// Renders the centered Entelechy boot / empty-stage card at `scale` device
/// pixels per logical pixel.
pub fn stage_card(card: &StageCard, scale: u32) -> Panel {
    const CARD_BG: u32 = 0x0005_0505;
    const CARD_BORDER: u32 = 0x0026_2626;
    const SOLAR_GOLD: u32 = 0x00FF_B000;
    const PURE_WHITE: u32 = 0x00FF_FFFF;
    const TIER2_DIM: u32 = 0x0037_3737;
    const TIER3_DIM: u32 = 0x0028_2828;
    const HORIZON: u32 = 0x0059_5959;
    const TRACK: u32 = 0x0026_2626;
    const SUBTLE_TRACK: u32 = 0x001A_1A1A;
    const MUTED: u32 = 0x008C_8C8C;
    const FAULT_BAR: u32 = 0x00FF_3B30;
    const FAULT_TEXT: u32 = 0x00FC_A5A5;

    let s = scale.max(1);
    let card_w = 348 * s;
    let pad_y = 28 * s;
    let badge_chars = truncate_chars(&card.badge, 42);
    let badge_text_w = badge_chars.len().saturating_sub(1) as u32 * CELL.0 * s + 5 * s;
    let badge_w = badge_text_w + 16 * s;
    let badge_h = 15 * s;

    let emblem_px = |v: u32| (v * 3 * s) / 2;
    let emblem_w = emblem_px(50);
    let emblem_h = emblem_px(42);

    let wm_unit = 2 * s;
    let wm_letters = "entelechy";
    let wm_units_w: u32 = wm_letters
        .chars()
        .filter_map(|c| WORDMARK_GLYPHS.iter().find(|(ch, _, _)| *ch == c))
        .map(|(_, w, _)| *w + 1)
        .sum::<u32>()
        .saturating_sub(1);
    let wm_w = wm_units_w * wm_unit;
    let wm_h = 13 * wm_unit;

    let title_chars = card.title.as_deref().map(|t| truncate_chars(t, 44));
    let detail_rows = wrap_card_lines(&card.detail, 48, 5);

    let mut y = pad_y;
    let badge_y = y;
    y += badge_h + 16 * s;
    let emblem_y = y;
    y += emblem_h + 12 * s;
    let wm_y = y;
    y += wm_h;
    let title_y = if title_chars.is_some() {
        y += 14 * s;
        let at = y;
        y += 7 * s;
        Some(at)
    } else {
        None
    };
    y += 16 * s;
    let bar_y = y;
    let bar_h = 2 * s;
    y += bar_h + 14 * s;
    let detail_y = y;
    let detail_h = if detail_rows.is_empty() {
        7 * s
    } else {
        (detail_rows.len() as u32 - 1) * 10 * s + 7 * s
    };
    y += detail_h + pad_y;
    let card_h = y;

    let mut pixels = vec![CARD_BG; (card_w * card_h) as usize];
    let border = s.min(2);
    fill_rect(&mut pixels, card_w, 0, 0, card_w, border, CARD_BORDER);
    fill_rect(
        &mut pixels,
        card_w,
        0,
        card_h - border,
        card_w,
        border,
        CARD_BORDER,
    );
    fill_rect(&mut pixels, card_w, 0, 0, border, card_h, CARD_BORDER);
    fill_rect(
        &mut pixels,
        card_w,
        card_w - border,
        0,
        border,
        card_h,
        CARD_BORDER,
    );

    // Top badge (`native · gpu` or `live · 127.0.0.1:7788`).
    if !badge_chars.is_empty() {
        let bx = (card_w - badge_w) / 2;
        fill_rect(
            &mut pixels,
            card_w,
            bx,
            badge_y,
            badge_w,
            badge_h,
            SOLAR_GOLD,
        );
        draw_at(
            &badge_chars,
            &mut pixels,
            card_w,
            bx + 8 * s,
            badge_y + 4 * s,
            s,
            0x0000_0000,
        );
    }

    // 6-Square Living Seed & Open DAG emblem (`viewBox="0 0 50 42"`).
    let ex = (card_w - emblem_w) / 2;
    let t2_color = match card.progress {
        None => PURE_WHITE,
        Some(p) if p >= 0.28 => PURE_WHITE,
        Some(_) => TIER2_DIM,
    };
    let t3_color = match card.progress {
        None => PURE_WHITE,
        Some(p) if p >= 0.55 => PURE_WHITE,
        Some(_) => TIER3_DIM,
    };
    let sq = emblem_px(10);
    for gx in [0, 20, 40] {
        fill_rect(
            &mut pixels,
            card_w,
            ex + emblem_px(gx),
            emblem_y + emblem_px(2),
            sq,
            sq,
            t3_color,
        );
    }
    for gx in [10, 30] {
        fill_rect(
            &mut pixels,
            card_w,
            ex + emblem_px(gx),
            emblem_y + emblem_px(12),
            sq,
            sq,
            t2_color,
        );
    }
    fill_rect(
        &mut pixels,
        card_w,
        ex + emblem_px(20),
        emblem_y + emblem_px(22),
        sq,
        sq,
        SOLAR_GOLD,
    );
    fill_rect(
        &mut pixels,
        card_w,
        ex + emblem_px(15),
        emblem_y + emblem_px(38),
        emblem_px(20),
        emblem_px(2).max(s),
        HORIZON,
    );

    // Bold geometric `entelechy` wordmark.
    let mut wx = (card_w - wm_w) / 2;
    for ch in wm_letters.chars() {
        let Some((_, gw, rows)) = WORDMARK_GLYPHS.iter().find(|(c, _, _)| *c == ch) else {
            continue;
        };
        for (row, bits) in rows.iter().enumerate() {
            for col in 0..*gw {
                if bits & (1 << (gw - 1 - col)) != 0 {
                    fill_rect(
                        &mut pixels,
                        card_w,
                        wx + col * wm_unit,
                        wm_y + row as u32 * wm_unit,
                        wm_unit,
                        wm_unit,
                        PURE_WHITE,
                    );
                }
            }
        }
        wx += (*gw + 1) * wm_unit;
    }

    // Optional scene title.
    if let (Some(at_y), Some(chars)) = (title_y, &title_chars)
        && !chars.is_empty()
    {
        let tw = (chars.len() as u32 * CELL.0 - 1) * s;
        draw_at(
            chars,
            &mut pixels,
            card_w,
            card_w.saturating_sub(tw) / 2,
            at_y,
            s,
            PURE_WHITE,
        );
    }

    // Progress bar (or quiet Solar-Seed horizon tick on an empty resting stage).
    let bar_w = 260 * s;
    let bar_x = (card_w - bar_w) / 2;
    match card.progress {
        Some(part) => {
            fill_rect(&mut pixels, card_w, bar_x, bar_y, bar_w, bar_h, TRACK);
            let fill_color = if card.failed { FAULT_BAR } else { SOLAR_GOLD };
            let fill_w = if card.failed {
                bar_w
            } else {
                ((part.clamp(0.06, 1.0) * bar_w as f32).round() as u32).clamp(1, bar_w)
            };
            fill_rect(&mut pixels, card_w, bar_x, bar_y, fill_w, bar_h, fill_color);
        }
        None => {
            let line_h = s;
            fill_rect(
                &mut pixels,
                card_w,
                bar_x,
                bar_y,
                bar_w,
                line_h,
                SUBTLE_TRACK,
            );
            let tick_w = if card.failed { bar_w } else { 24 * s };
            let tick_color = if card.failed { FAULT_BAR } else { SOLAR_GOLD };
            fill_rect(
                &mut pixels,
                card_w,
                (card_w - tick_w) / 2,
                bar_y,
                tick_w,
                bar_h,
                tick_color,
            );
        }
    }

    // Detail / status lines (wrapped and centered).
    let detail_color = if card.failed { FAULT_TEXT } else { MUTED };
    for (idx, row_chars) in detail_rows.iter().enumerate() {
        if row_chars.is_empty() {
            continue;
        }
        let dw = (row_chars.len() as u32 * CELL.0 - 1) * s;
        draw_at(
            row_chars,
            &mut pixels,
            card_w,
            card_w.saturating_sub(dw) / 2,
            detail_y + idx as u32 * 10 * s,
            s,
            detail_color,
        );
    }

    Panel {
        width: card_w,
        height: card_h,
        pixels,
        centered: true,
    }
}

/// Stamps the panel into the top-left (or center when `panel.centered`) of a
/// `0x00RRGGBB` frame, clipped.
pub fn stamp(panel: &Panel, frame: &mut [u32], frame_width: u32, frame_height: u32) {
    let ox = if panel.centered {
        frame_width.saturating_sub(panel.width) / 2
    } else {
        0
    };
    let oy = if panel.centered {
        frame_height.saturating_sub(panel.height) / 2
    } else {
        0
    };
    let width = panel.width.min(frame_width.saturating_sub(ox)) as usize;
    let height = panel.height.min(frame_height.saturating_sub(oy)) as usize;
    for y in 0..height {
        let from = y * panel.width as usize;
        let to = (oy as usize + y) * frame_width as usize + ox as usize;
        frame[to..to + width].copy_from_slice(&panel.pixels[from..from + width]);
    }
}

/// Smoothed frame times.
#[derive(Debug)]
pub struct Stats {
    last_start: Option<f64>,
    /// Smoothed milliseconds: frame interval; the processor's work on a
    /// frame, its waits left out; its waits for the device; the device's
    /// busy time on one by its own clock (`None` without one); presenting
    /// (including vsync).
    interval: f64,
    cpu: f64,
    wait: f64,
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
    /// Each node's waits for the device a frame, as shown: averaged over
    /// the last [`SHOWN_EVERY`] frames ([`Stats::waited`]), and the waits
    /// at the frame's end besides; then what those frames add up to.
    waits: Vec<(String, f64)>,
    wait_after: f64,
    summed_waits: Vec<(String, f64)>,
    summed_wait_after: f64,
    summed_wait_frames: u32,
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
            wait: 0.0,
            gpu: None,
            present: 0.0,
            nodes: Vec::new(),
            between: 0.0,
            summed: Vec::new(),
            summed_between: 0.0,
            summed_frames: 0,
            waits: Vec::new(),
            wait_after: 0.0,
            summed_waits: Vec::new(),
            summed_wait_after: 0.0,
            summed_wait_frames: 0,
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

    /// Records how long a frame's thread waited for the device in all, and
    /// which nodes waited ([`entelechy_graph::Report::waits`]); what no
    /// node waited is the frame's end. Shown as [`Stats::told`] shows.
    pub fn waited(&mut self, total: Duration, by: &[(entelechy_graph::NodeId, Duration)]) {
        smooth(&mut self.wait, total);
        let mut nodes = 0.0;
        for (node, spent) in by {
            let name = node.to_string();
            let millis = spent.as_secs_f64() * 1000.0;
            nodes += millis;
            match self.summed_waits.iter_mut().find(|(n, _)| *n == name) {
                Some((_, sum)) => *sum += millis,
                None => self.summed_waits.push((name, millis)),
            }
        }
        self.summed_wait_after += (total.as_secs_f64() * 1000.0 - nodes).max(0.0);
        self.summed_wait_frames += 1;
        if self.summed_wait_frames >= SHOWN_EVERY {
            let frames = f64::from(self.summed_wait_frames);
            self.waits = std::mem::take(&mut self.summed_waits)
                .into_iter()
                .map(|(name, sum)| (name, sum / frames))
                .filter(|(_, average)| *average >= 0.005)
                .collect();
            self.wait_after = self.summed_wait_after / frames;
            self.summed_wait_after = 0.0;
            self.summed_wait_frames = 0;
        }
    }

    /// The nodes that kept the frame's thread waiting longest for the
    /// device, `most` of them, then the waits at the frame's end; the
    /// first headed `waited for the device:`. None while nothing waits.
    pub fn waiting(&self, most: usize) -> Vec<String> {
        if self.waits.is_empty() && self.wait_after < 0.005 {
            return Vec::new();
        }
        let mut nodes = self.waits.clone();
        nodes.sort_by(|a, b| b.1.total_cmp(&a.1));
        nodes.truncate(most);
        let mut said: Vec<String> = nodes
            .iter()
            .map(|(name, ms)| format!("{name} {ms:>6.2}"))
            .collect();
        said.push(format!("frame's end {:>6.2}", self.wait_after));
        said[0] = format!("waited for the device: {}", said[0]);
        said
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
        // A backend with a device of its own may be waited for.
        let wait = if self.gpu.is_some() || self.wait > 0.0 {
            format!("wait {:>6.2} ms | ", self.wait)
        } else {
            String::new()
        };
        format!(
            "frame {:>6.1} ms ({fps:>3.0} fps) | cpu {:>6.2} ms | {wait}{gpu}present {:>6.2} ms",
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
            "frame    0.0 ms (  0 fps) | cpu   2.00 ms | wait   0.00 ms | gpu   1.45 ms | present   0.50 ms"
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
    fn the_waits_name_their_nodes_and_the_frames_end_averaged() {
        let mut stats = Stats::default();
        assert!(stats.waiting(8).is_empty(), "nothing said before a wait");
        let faces = entelechy_graph::NodeId("faces".into());
        for _ in 0..SHOWN_EVERY {
            stats.waited(
                Duration::from_millis(4),
                &[(faces.clone(), Duration::from_millis(3))],
            );
        }
        assert_eq!(
            stats.waiting(8),
            vec![
                "waited for the device: faces   3.00".to_owned(),
                "frame's end   1.00".to_owned()
            ],
            "what no node waited is the frame's end"
        );
        stats.record(Duration::from_millis(2), None);
        assert_eq!(
            stats.times(),
            "frame    0.0 ms (  0 fps) | cpu   2.00 ms | wait   4.00 ms | present   0.00 ms"
        );
    }

    #[test]
    fn the_busiest_nodes_follow_the_line_averaged_and_held_half_a_second() {
        let id = |n: &str| entelechy_graph::NodeId(n.to_owned().into());
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
