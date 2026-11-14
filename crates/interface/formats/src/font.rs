//! A font's Fields, as `entelechy import` writes them from a font file and
//! `entelechy export` writes a font file from them: the layouts both
//! sides read. A glyph's outline is line segments in ems, y up, each
//! contour closed and its segments one after another, filled by the
//! nonzero rule.

use entelechy_host::{Domain, FieldData, Scalar, Schema};

/// How finely curves are cut: the largest distance, in ems, a segment may
/// stray from its curve.
pub const TOLERANCE: f32 = 1.0 / 2048.0;

const MAX_COMPACT_ITEMS: usize = 1 << 20;
const MAX_COMPACT_OPS: usize = 4 << 20;

/// A glyph's row, as a Kernel declares it to read one: `code: Int;
/// advance: Int; first: Int; count: Int; lo: Vec2; hi: Vec2`. The advance
/// is in the font's units, so a line's advances sum exactly; the box is
/// in ems. A code of -1 is the font's glyph for a character it lacks.
pub fn glyph_schema() -> Schema {
    let f = Scalar::Float;
    let i = Scalar::Int;
    Schema::new(vec![i, i, i, i, f, f, f, f])
}

/// A segment of an outline, `a.x, a.y, b.x, b.y` in ems: `Rows<Vec4>`.
pub fn segment_schema() -> Schema {
    Schema::new(vec![Scalar::Float; 4])
}

/// The font's measures: `units: Int; ascent: Float; descent: Float; gap:
/// Float`, the font's units in an em and the rest in ems.
pub fn metrics_schema() -> Schema {
    let f = Scalar::Float;
    Schema::new(vec![Scalar::Int, f, f, f])
}

/// Which of a font's Fields a layout is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// Glyph rows, [`glyph_schema`].
    Glyphs,
    /// Outline segments, [`segment_schema`].
    Segments,
    /// The measures, [`metrics_schema`].
    Metrics,
}

impl Part {
    /// The part of a font `schema` lays out, if any.
    pub fn of(schema: &Schema) -> Option<Self> {
        let scalars = schema.scalars();
        [
            (Self::Glyphs, glyph_schema()),
            (Self::Segments, segment_schema()),
            (Self::Metrics, metrics_schema()),
        ]
        .into_iter()
        .find(|(_, layout)| layout.scalars() == scalars)
        .map(|(part, _)| part)
    }
}

/// A glyph's outline flattened to line segments in ems, curves cut until
/// within [`TOLERANCE`] of them.
pub struct Outline {
    em: f32,
    tolerance: f32,
    start: [f32; 2],
    at: [f32; 2],
    segments: Vec<[f32; 4]>,
}

impl Outline {
    /// Starts an outline in a face whose em is `1.0 / em` font units.
    pub fn new(em: f32) -> Self {
        Self {
            em,
            tolerance: TOLERANCE,
            start: [0.0; 2],
            at: [0.0; 2],
            segments: Vec::new(),
        }
    }

    fn line(&mut self, to: [f32; 2]) {
        if to != self.at {
            self.segments.push([self.at[0], self.at[1], to[0], to[1]]);
        }
        self.at = to;
    }

    fn scaled(&self, x: f32, y: f32) -> [f32; 2] {
        [x * self.em, y * self.em]
    }

    fn pieces(&self, bend: f32) -> usize {
        ((bend / self.tolerance).sqrt().ceil() as usize).clamp(1, 64)
    }

    /// Starts a new contour at `(x, y)` in font units, closing any open one.
    pub fn move_to(&mut self, x: f32, y: f32) {
        self.close();
        self.start = self.scaled(x, y);
        self.at = self.start;
    }

    /// Adds a line segment to `(x, y)` in font units.
    pub fn line_to(&mut self, x: f32, y: f32) {
        let to = self.scaled(x, y);
        self.line(to);
    }

    /// Flattens a quadratic curve through `(x1, y1)` to `(x, y)` in font units.
    pub fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let (p0, p1, p2) = (self.at, self.scaled(x1, y1), self.scaled(x, y));
        let bend = 0.25 * (p0[0] - 2.0 * p1[0] + p2[0]).hypot(p0[1] - 2.0 * p1[1] + p2[1]);
        let n = self.pieces(bend);
        for k in 1..=n {
            let t = k as f32 / n as f32;
            let u = 1.0 - t;
            let to = [
                u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0],
                u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1],
            ];
            self.line(if k == n { p2 } else { to });
        }
    }

    /// Flattens a cubic curve through `(x1, y1)` and `(x2, y2)` to `(x, y)` in font units.
    pub fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let (p0, p1, p2, p3) = (
            self.at,
            self.scaled(x1, y1),
            self.scaled(x2, y2),
            self.scaled(x, y),
        );
        let d1 = (p0[0] - 2.0 * p1[0] + p2[0]).hypot(p0[1] - 2.0 * p1[1] + p2[1]);
        let d2 = (p1[0] - 2.0 * p2[0] + p3[0]).hypot(p1[1] - 2.0 * p2[1] + p3[1]);
        let n = self.pieces(0.75 * d1.max(d2));
        for k in 1..=n {
            let t = k as f32 / n as f32;
            let u = 1.0 - t;
            let to = [
                u * u * u * p0[0]
                    + 3.0 * u * u * t * p1[0]
                    + 3.0 * u * t * t * p2[0]
                    + t * t * t * p3[0],
                u * u * u * p0[1]
                    + 3.0 * u * u * t * p1[1]
                    + 3.0 * u * t * t * p2[1]
                    + t * t * t * p3[1],
            ];
            self.line(if k == n { p3 } else { to });
        }
    }

    /// Closes the active contour back to its starting point.
    pub fn close(&mut self) {
        self.line(self.start);
    }

    /// Applies one [`Verb`] in font units.
    pub fn apply(&mut self, verb: Verb) {
        match verb {
            Verb::MoveTo(x, y) => self.move_to(x, y),
            Verb::LineTo(x, y) => self.line_to(x, y),
            Verb::QuadTo(x1, y1, x, y) => self.quad_to(x1, y1, x, y),
            Verb::CurveTo(x1, y1, x2, y2, x, y) => self.curve_to(x1, y1, x2, y2, x, y),
            Verb::Close => self.close(),
        }
    }

    /// Closes the last contour and returns all flattened segments in ems.
    pub fn finish(mut self) -> Vec<[f32; 4]> {
        self.close();
        self.segments
    }
}

/// One step of an unflattened glyph contour, in font units.
#[derive(Clone, Copy, Debug)]
pub enum Verb {
    /// Move to `(x, y)`.
    MoveTo(f32, f32),
    /// Line to `(x, y)`.
    LineTo(f32, f32),
    /// Quadratic curve through `(x1, y1)` to `(x, y)`.
    QuadTo(f32, f32, f32, f32),
    /// Cubic curve through `(x1, y1)` and `(x2, y2)` to `(x, y)`.
    CurveTo(f32, f32, f32, f32, f32, f32),
    /// Close contour.
    Close,
}

impl PartialEq for Verb {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::Close, Self::Close) => true,
            (Self::MoveTo(ax, ay), Self::MoveTo(bx, by))
            | (Self::LineTo(ax, ay), Self::LineTo(bx, by)) => {
                ax.to_bits() == bx.to_bits() && ay.to_bits() == by.to_bits()
            }
            (Self::QuadTo(ax1, ay1, ax, ay), Self::QuadTo(bx1, by1, bx, by)) => {
                ax1.to_bits() == bx1.to_bits()
                    && ay1.to_bits() == by1.to_bits()
                    && ax.to_bits() == bx.to_bits()
                    && ay.to_bits() == by.to_bits()
            }
            (
                Self::CurveTo(ax1, ay1, ax2, ay2, ax, ay),
                Self::CurveTo(bx1, by1, bx2, by2, bx, by),
            ) => {
                ax1.to_bits() == bx1.to_bits()
                    && ay1.to_bits() == by1.to_bits()
                    && ax2.to_bits() == bx2.to_bits()
                    && ay2.to_bits() == by2.to_bits()
                    && ax.to_bits() == bx.to_bits()
                    && ay.to_bits() == by.to_bits()
            }
            _ => false,
        }
    }
}

impl Eq for Verb {}

/// One entry in a [`CompactFont`]'s character table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompactGlyph {
    /// Character code (`-1` for the missing-character row).
    pub code: i32,
    /// Horizontal advance in font units.
    pub advance: i32,
    /// Index into [`CompactFont::shapes`].
    pub shape: u32,
}

/// A font's measures, character rows and unflattened contours, carried in
/// binary scene bundles and flattened to [`FieldData`] on open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactFont {
    /// Font units per em.
    pub units: u16,
    /// Metrics row words (`[units, ascender, descender, gap]`).
    pub metrics: [u32; 4],
    /// Character rows in ascending code order after row 0 (`-1`).
    pub glyphs: Vec<CompactGlyph>,
    /// Distinct glyph contours in first-seen order.
    pub shapes: Vec<Vec<Verb>>,
}

fn half_unit(v: f32) -> Option<i32> {
    let scaled = v * 2.0;
    if !scaled.is_finite() || scaled.abs() > 16_777_216.0 {
        return None;
    }
    let h = scaled.round() as i32;
    (((h as f32) * 0.5).to_bits() == v.to_bits()).then_some(h)
}

fn push_leb(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let mut b = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            b |= 0x80;
        }
        out.push(b);
        if v == 0 {
            break;
        }
    }
}

fn push_zig(out: &mut Vec<u8>, v: i64) {
    push_leb(out, ((v << 1) ^ (v >> 63)) as u64);
}

fn read_leb(rest: &mut &[u8]) -> Result<u64, String> {
    let mut val = 0u64;
    let mut shift = 0u32;
    loop {
        let (&b, tail) = rest
            .split_first()
            .ok_or_else(|| "compact font ended unexpectedly".to_owned())?;
        *rest = tail;
        if shift >= 64 || (shift == 63 && (b & 0x7f) > 1) {
            return Err("compact font integer overflow".to_owned());
        }
        val |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return Ok(val);
        }
        shift += 7;
    }
}

fn read_zig(rest: &mut &[u8]) -> Result<i64, String> {
    let u = read_leb(rest)?;
    Ok(((u >> 1) as i64) ^ -((u & 1) as i64))
}

fn read_f32(rest: &mut &[u8]) -> Result<f32, String> {
    if rest.len() < 4 {
        return Err("compact font truncated in float".to_owned());
    }
    let (head, tail) = rest.split_at(4);
    *rest = tail;
    Ok(f32::from_le_bytes(head.try_into().unwrap()))
}

fn read_u32_le(rest: &mut &[u8]) -> Result<u32, String> {
    if rest.len() < 4 {
        return Err("compact font truncated in metrics".to_owned());
    }
    let (head, tail) = rest.split_at(4);
    *rest = tail;
    Ok(u32::from_le_bytes(head.try_into().unwrap()))
}

fn read_slice<'a>(rest: &mut &'a [u8]) -> Result<&'a [u8], String> {
    let len =
        usize::try_from(read_leb(rest)?).map_err(|_| "compact font stream length overflow")?;
    if len > rest.len() {
        return Err("compact font stream truncated".to_owned());
    }
    let (head, tail) = rest.split_at(len);
    *rest = tail;
    Ok(head)
}

impl CompactFont {
    /// Flattens every shape at [`TOLERANCE`] and builds `(glyphs, segments, metrics)`.
    pub fn to_fields(&self) -> Result<(FieldData, FieldData, FieldData), String> {
        if self.units == 0 {
            return Err("compact font has zero units per em".to_owned());
        }
        let em = 1.0 / f32::from(self.units);
        let mut segments: Vec<u32> = Vec::new();
        let mut boxes: Vec<[u32; 6]> = Vec::with_capacity(self.shapes.len());
        let mut first = 0usize;
        for verbs in &self.shapes {
            let mut outline = Outline::new(em);
            for &verb in verbs {
                outline.apply(verb);
            }
            let flat = outline.finish();
            let (mut lo, mut hi) = ([0.0f32; 2], [0.0f32; 2]);
            if let Some(s) = flat.first() {
                lo = [s[0], s[1]];
                hi = lo;
            }
            for s in &flat {
                for (x, y) in [(s[0], s[1]), (s[2], s[3])] {
                    lo = [lo[0].min(x), lo[1].min(y)];
                    hi = [hi[0].max(x), hi[1].max(y)];
                }
                segments.extend(s.iter().map(|v| v.to_bits()));
            }
            let count = flat.len();
            let [lo_x, lo_y, hi_x, hi_y] = [lo[0], lo[1], hi[0], hi[1]].map(f32::to_bits);
            boxes.push([first as u32, count as u32, lo_x, lo_y, hi_x, hi_y]);
            first += count;
        }
        let mut glyph_words = Vec::with_capacity(self.glyphs.len().max(1) * 8);
        if self.glyphs.is_empty() {
            glyph_words.extend_from_slice(&[-1i32 as u32, 0, 0, 0, 0, 0, 0, 0]);
        }
        for g in &self.glyphs {
            let b = boxes
                .get(g.shape as usize)
                .ok_or_else(|| "compact font glyph shape out of bounds".to_owned())?;
            glyph_words.extend_from_slice(&[
                g.code as u32,
                g.advance as u32,
                b[0],
                b[1],
                b[2],
                b[3],
                b[4],
                b[5],
            ]);
        }
        if segments.is_empty() {
            segments = vec![0; 4];
        }
        let glyph_rows = (glyph_words.len() / 8) as u32;
        let seg_rows = first.max(1) as u32;
        let field = |schema: Schema, rows: u32, words: Vec<u32>| {
            FieldData::from_storage(schema, Domain::Rows(rows), words).map_err(|e| e.to_string())
        };
        Ok((
            field(glyph_schema(), glyph_rows, glyph_words)?,
            field(segment_schema(), seg_rows, segments)?,
            field(metrics_schema(), 1, self.metrics.to_vec())?,
        ))
    }

    /// Flattens only the shapes requested by `codes` at [`TOLERANCE`] and builds
    /// `(glyphs, segments, metrics)`, matching `entelechy_import::font::glyphs_of`
    /// word for word without pre-flattening unused characters.
    pub fn glyphs_of(&self, codes: &[i32]) -> Result<(FieldData, FieldData, FieldData), String> {
        if self.units == 0 {
            return Err("compact font has zero units per em".to_owned());
        }
        let em = 1.0 / f32::from(self.units);
        let missing_advance = self.glyphs.first().map_or(0, |g| g.advance as u32);
        let missing = [-1i32 as u32, missing_advance, 0, 0, 0, 0, 0, 0];
        let sorted = self.glyphs.get(1..).unwrap_or(&[]);

        let mut glyph_words = Vec::with_capacity(codes.len().max(1) * 8);
        let mut segments: Vec<u32> = Vec::new();
        let mut drawn: std::collections::HashMap<u32, [u32; 6]> = std::collections::HashMap::new();
        let mut first = 0usize;

        if codes.is_empty() {
            glyph_words.extend_from_slice(&missing);
        }
        for &code in codes {
            if code <= 0 {
                glyph_words.extend_from_slice(&missing);
                continue;
            }
            let Ok(found) = sorted.binary_search_by_key(&code, |g| g.code) else {
                glyph_words.extend_from_slice(&missing);
                continue;
            };
            let g = sorted[found];
            let row = if let Some(&existing) = drawn.get(&g.shape) {
                existing
            } else {
                let verbs = self
                    .shapes
                    .get(g.shape as usize)
                    .ok_or_else(|| "compact font glyph shape out of bounds".to_owned())?;
                let mut outline = Outline::new(em);
                for &verb in verbs {
                    outline.apply(verb);
                }
                let flat = outline.finish();
                let (mut lo, mut hi) = ([0.0f32; 2], [0.0f32; 2]);
                if let Some(s) = flat.first() {
                    lo = [s[0], s[1]];
                    hi = lo;
                }
                for s in &flat {
                    for (x, y) in [(s[0], s[1]), (s[2], s[3])] {
                        lo = [lo[0].min(x), lo[1].min(y)];
                        hi = [hi[0].max(x), hi[1].max(y)];
                    }
                    segments.extend(s.iter().map(|v| v.to_bits()));
                }
                let count = flat.len();
                let [lo_x, lo_y, hi_x, hi_y] = [lo[0], lo[1], hi[0], hi[1]].map(f32::to_bits);
                let built = [first as u32, count as u32, lo_x, lo_y, hi_x, hi_y];
                drawn.insert(g.shape, built);
                first += count;
                built
            };
            glyph_words.extend_from_slice(&[
                code as u32,
                g.advance as u32,
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
            ]);
        }
        if segments.is_empty() {
            segments = vec![0; 4];
        }
        let glyph_rows = (glyph_words.len() / 8) as u32;
        let seg_rows = first.max(1) as u32;
        let field = |schema: Schema, rows: u32, words: Vec<u32>| {
            FieldData::from_storage(schema, Domain::Rows(rows), words).map_err(|e| e.to_string())
        };
        Ok((
            field(glyph_schema(), glyph_rows, glyph_words)?,
            field(segment_schema(), seg_rows, segments)?,
            field(metrics_schema(), 1, self.metrics.to_vec())?,
        ))
    }

    /// Appends the compact binary encoding of this font to `out`.
    pub fn write_to(&self, out: &mut Vec<u8>) {
        push_leb(out, u64::from(self.units));
        for word in self.metrics {
            out.extend_from_slice(&word.to_le_bytes());
        }
        push_leb(out, self.glyphs.len() as u64);
        let mut prev_code = -1i64;
        for g in &self.glyphs {
            push_zig(out, i64::from(g.code) - prev_code);
            prev_code = i64::from(g.code);
        }
        let mut prev_adv = 0i64;
        for g in &self.glyphs {
            push_zig(out, i64::from(g.advance) - prev_adv);
            prev_adv = i64::from(g.advance);
        }
        let mut prev_shape = -1i64;
        for g in &self.glyphs {
            push_zig(out, i64::from(g.shape) - prev_shape);
            prev_shape = i64::from(g.shape);
        }
        push_leb(out, self.shapes.len() as u64);
        let mut ops = Vec::new();
        let mut dxs = Vec::new();
        let mut dys = Vec::new();
        let mut raw = Vec::new();
        for shape in &self.shapes {
            let (mut cx, mut cy) = (0i64, 0i64);
            for &verb in shape {
                match verb {
                    Verb::Close => ops.push(5),
                    Verb::MoveTo(x, y) => match (half_unit(x), half_unit(y)) {
                        (Some(hx), Some(hy)) => {
                            ops.push(1);
                            push_zig(&mut dxs, i64::from(hx) - cx);
                            push_zig(&mut dys, i64::from(hy) - cy);
                            (cx, cy) = (i64::from(hx), i64::from(hy));
                        }
                        _ => {
                            ops.push(6);
                            for v in [x, y] {
                                raw.extend_from_slice(&v.to_le_bytes());
                            }
                        }
                    },
                    Verb::LineTo(x, y) => match (half_unit(x), half_unit(y)) {
                        (Some(hx), Some(hy)) => {
                            ops.push(2);
                            push_zig(&mut dxs, i64::from(hx) - cx);
                            push_zig(&mut dys, i64::from(hy) - cy);
                            (cx, cy) = (i64::from(hx), i64::from(hy));
                        }
                        _ => {
                            ops.push(7);
                            for v in [x, y] {
                                raw.extend_from_slice(&v.to_le_bytes());
                            }
                        }
                    },
                    Verb::QuadTo(x1, y1, x, y) => {
                        match (half_unit(x1), half_unit(y1), half_unit(x), half_unit(y)) {
                            (Some(hx1), Some(hy1), Some(hx), Some(hy)) => {
                                ops.push(3);
                                push_zig(&mut dxs, i64::from(hx1) - cx);
                                push_zig(&mut dys, i64::from(hy1) - cy);
                                push_zig(&mut dxs, i64::from(hx) - i64::from(hx1));
                                push_zig(&mut dys, i64::from(hy) - i64::from(hy1));
                                (cx, cy) = (i64::from(hx), i64::from(hy));
                            }
                            _ => {
                                ops.push(8);
                                for v in [x1, y1, x, y] {
                                    raw.extend_from_slice(&v.to_le_bytes());
                                }
                            }
                        }
                    }
                    Verb::CurveTo(x1, y1, x2, y2, x, y) => match (
                        half_unit(x1),
                        half_unit(y1),
                        half_unit(x2),
                        half_unit(y2),
                        half_unit(x),
                        half_unit(y),
                    ) {
                        (Some(hx1), Some(hy1), Some(hx2), Some(hy2), Some(hx), Some(hy)) => {
                            ops.push(4);
                            push_zig(&mut dxs, i64::from(hx1) - cx);
                            push_zig(&mut dys, i64::from(hy1) - cy);
                            push_zig(&mut dxs, i64::from(hx2) - i64::from(hx1));
                            push_zig(&mut dys, i64::from(hy2) - i64::from(hy1));
                            push_zig(&mut dxs, i64::from(hx) - i64::from(hx2));
                            push_zig(&mut dys, i64::from(hy) - i64::from(hy2));
                            (cx, cy) = (i64::from(hx), i64::from(hy));
                        }
                        _ => {
                            ops.push(9);
                            for v in [x1, y1, x2, y2, x, y] {
                                raw.extend_from_slice(&v.to_le_bytes());
                            }
                        }
                    },
                }
            }
            ops.push(0);
        }
        for stream in [&ops, &dxs, &dys, &raw] {
            push_leb(out, stream.len() as u64);
            out.extend_from_slice(stream);
        }
    }

    /// Reads a [`CompactFont`] written by [`CompactFont::write_to`] from `rest`.
    pub fn read_from(rest: &mut &[u8]) -> Result<Self, String> {
        let units = u16::try_from(read_leb(rest)?).map_err(|_| "bad compact font units")?;
        if units == 0 {
            return Err("compact font has zero units per em".to_owned());
        }
        let metrics = [
            read_u32_le(rest)?,
            read_u32_le(rest)?,
            read_u32_le(rest)?,
            read_u32_le(rest)?,
        ];
        let glyph_count =
            usize::try_from(read_leb(rest)?).map_err(|_| "bad compact glyph count")?;
        if glyph_count > MAX_COMPACT_ITEMS {
            return Err("compact font glyph table too large".to_owned());
        }
        let mut codes = Vec::with_capacity(glyph_count.min(4096));
        let mut prev_code = -1i64;
        for _ in 0..glyph_count {
            prev_code = prev_code
                .checked_add(read_zig(rest)?)
                .ok_or_else(|| "compact font code overflow".to_owned())?;
            codes.push(i32::try_from(prev_code).map_err(|_| "compact font code out of range")?);
        }
        let mut advances = Vec::with_capacity(glyph_count.min(4096));
        let mut prev_adv = 0i64;
        for _ in 0..glyph_count {
            prev_adv = prev_adv
                .checked_add(read_zig(rest)?)
                .ok_or_else(|| "compact font advance overflow".to_owned())?;
            advances
                .push(i32::try_from(prev_adv).map_err(|_| "compact font advance out of range")?);
        }
        let mut shape_ids = Vec::with_capacity(glyph_count.min(4096));
        let mut prev_shape = -1i64;
        for _ in 0..glyph_count {
            prev_shape = prev_shape
                .checked_add(read_zig(rest)?)
                .ok_or_else(|| "compact font shape overflow".to_owned())?;
            shape_ids
                .push(u32::try_from(prev_shape).map_err(|_| "compact font shape out of range")?);
        }
        let shape_count =
            usize::try_from(read_leb(rest)?).map_err(|_| "bad compact shape count")?;
        if shape_count > MAX_COMPACT_ITEMS {
            return Err("compact font shape table too large".to_owned());
        }
        for &s in &shape_ids {
            if (s as usize) >= shape_count {
                return Err("compact font glyph shape index out of bounds".to_owned());
            }
        }
        let mut ops = read_slice(rest)?;
        if ops.len() > MAX_COMPACT_OPS {
            return Err("compact font outline stream too large".to_owned());
        }
        let mut dxs = read_slice(rest)?;
        let mut dys = read_slice(rest)?;
        let mut raw = read_slice(rest)?;

        let step = |cx: &mut i64,
                    cy: &mut i64,
                    dxs: &mut &[u8],
                    dys: &mut &[u8]|
         -> Result<(f32, f32), String> {
            *cx = cx
                .checked_add(read_zig(dxs)?)
                .ok_or_else(|| "compact font x overflow".to_owned())?;
            *cy = cy
                .checked_add(read_zig(dys)?)
                .ok_or_else(|| "compact font y overflow".to_owned())?;
            Ok(((*cx as f32) * 0.5, (*cy as f32) * 0.5))
        };

        let mut shapes = Vec::with_capacity(shape_count.min(4096));
        for _ in 0..shape_count {
            let (mut cx, mut cy) = (0i64, 0i64);
            let mut verbs = Vec::new();
            loop {
                let (&op, tail) = ops
                    .split_first()
                    .ok_or_else(|| "compact font ops ended unexpectedly".to_owned())?;
                ops = tail;
                match op {
                    0 => break,
                    1 => {
                        let (x, y) = step(&mut cx, &mut cy, &mut dxs, &mut dys)?;
                        verbs.push(Verb::MoveTo(x, y));
                    }
                    2 => {
                        let (x, y) = step(&mut cx, &mut cy, &mut dxs, &mut dys)?;
                        verbs.push(Verb::LineTo(x, y));
                    }
                    3 => {
                        let (x1, y1) = step(&mut cx, &mut cy, &mut dxs, &mut dys)?;
                        let (x, y) = step(&mut cx, &mut cy, &mut dxs, &mut dys)?;
                        verbs.push(Verb::QuadTo(x1, y1, x, y));
                    }
                    4 => {
                        let (x1, y1) = step(&mut cx, &mut cy, &mut dxs, &mut dys)?;
                        let (x2, y2) = step(&mut cx, &mut cy, &mut dxs, &mut dys)?;
                        let (x, y) = step(&mut cx, &mut cy, &mut dxs, &mut dys)?;
                        verbs.push(Verb::CurveTo(x1, y1, x2, y2, x, y));
                    }
                    5 => verbs.push(Verb::Close),
                    6 => verbs.push(Verb::MoveTo(read_f32(&mut raw)?, read_f32(&mut raw)?)),
                    7 => verbs.push(Verb::LineTo(read_f32(&mut raw)?, read_f32(&mut raw)?)),
                    8 => verbs.push(Verb::QuadTo(
                        read_f32(&mut raw)?,
                        read_f32(&mut raw)?,
                        read_f32(&mut raw)?,
                        read_f32(&mut raw)?,
                    )),
                    9 => verbs.push(Verb::CurveTo(
                        read_f32(&mut raw)?,
                        read_f32(&mut raw)?,
                        read_f32(&mut raw)?,
                        read_f32(&mut raw)?,
                        read_f32(&mut raw)?,
                        read_f32(&mut raw)?,
                    )),
                    other => return Err(format!("unknown compact font opcode {other}")),
                }
            }
            shapes.push(verbs);
        }
        if !ops.is_empty() || !dxs.is_empty() || !dys.is_empty() || !raw.is_empty() {
            return Err("compact font stream has trailing bytes".to_owned());
        }
        let glyphs = codes
            .into_iter()
            .zip(advances)
            .zip(shape_ids)
            .map(|((code, advance), shape)| CompactGlyph {
                code,
                advance,
                shape,
            })
            .collect();
        Ok(Self {
            units,
            metrics,
            glyphs,
            shapes,
        })
    }
}
