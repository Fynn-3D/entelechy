//! A TrueType or OpenType font as Fields a scene's Kernels lay out and
//! draw: each glyph's outline flattened to line segments in ems, y up,
//! filled by the nonzero rule. `ttf-parser` reads
//! the font (quadratic and cubic outlines, the character map). What is
//! left out is named in the notes.
//!
//! Three files are written:
//!
//! - `<stem>.glyphs.field`: a row a glyph in order of code, the layout
//!   [`glyph_schema`] gives; the first, of code -1, is the font's glyph for
//!   a character it lacks;
//! - `<stem>.segments.field`: `Rows<Vec4>`, a segment `a.x, a.y, b.x, b.y`;
//! - `<stem>.metrics.field`: `Rows<Metrics>`, one row, [`metrics_schema`].

use std::path::{Path, PathBuf};

use entelechy_host::{Domain, FieldData, Schema};

/// How finely curves are cut: the largest distance, in ems, a segment may
/// stray from its curve.
const TOLERANCE: f32 = 1.0 / 2048.0;

pub use entelechy_formats::font::{glyph_schema, metrics_schema, segment_schema};

/// What a font import produced.
#[derive(Debug)]
pub struct Imported {
    /// The files written.
    pub files: Vec<PathBuf>,
    /// Characters imported, the missing glyph not counted.
    pub glyphs: usize,
    /// Segments over all glyphs.
    pub segments: usize,
    /// Characters asked for that the font lacks.
    pub missing: Vec<char>,
    /// What was left out, in words.
    pub notes: Vec<String>,
}

/// The three Fields of a font.
#[derive(Debug)]
pub struct Fields {
    /// A row a glyph, in order of code.
    pub glyphs: FieldData,
    /// `Rows<Vec4>`, the segments.
    pub segments: FieldData,
    /// `Rows<Metrics>`, the font's measures.
    pub metrics: FieldData,
}

/// Reads the font at `path` and writes the Fields of `chars`, or of every
/// character it maps when none are given, into `out_dir`.
pub fn import(path: &Path, out_dir: &Path, chars: Option<&str>) -> Result<Imported, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("font")
        .to_owned();
    let (fields, imported) = read(&bytes, chars)?;
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    let mut files = Vec::new();
    for (name, data) in [
        ("glyphs", &fields.glyphs),
        ("segments", &fields.segments),
        ("metrics", &fields.metrics),
    ] {
        let file = out_dir.join(format!("{stem}.{name}.field"));
        std::fs::write(&file, entelechy_formats::file::encode_data(data))
            .map_err(|e| format!("{}: {e}", file.display()))?;
        files.push(file);
    }
    Ok(Imported { files, ..imported })
}

/// Reads a font into the Fields of `chars` (every mapped character when
/// `None`) and what was left out.
pub fn read(bytes: &[u8], chars: Option<&str>) -> Result<(Fields, Imported), String> {
    let face = ttf_parser::Face::parse(bytes, 0)
        .map_err(|e| format!("not a TrueType or OpenType font this reads: {e}"))?;
    let mut codes: Vec<char> = match chars {
        Some(text) => text.chars().collect(),
        None => {
            let mut all = Vec::new();
            if let Some(table) = face.tables().cmap {
                for sub in table.subtables {
                    if sub.is_unicode() {
                        sub.codepoints(|c| all.extend(char::from_u32(c)));
                    }
                }
            }
            all
        }
    };
    codes.sort_unstable();
    codes.dedup();
    let mut missing = Vec::new();
    let mut entries: Vec<(i32, ttf_parser::GlyphId)> = vec![(-1, ttf_parser::GlyphId(0))];
    for c in codes {
        match face.glyph_index(c) {
            Some(id) => entries.push((c as i32, id)),
            None if c.is_control() => {}
            None => missing.push(c),
        }
    }
    if entries.len() == 1 {
        return Err("the font maps none of the characters asked for".into());
    }
    let mut drawer = Drawer::new(&face);
    let mut glyphs: Vec<u32> = Vec::new();
    for &(code, id) in &entries {
        glyphs.extend(drawer.row(code, id));
    }
    let first = drawer.first;
    let fields = drawer.fields(glyphs, entries.len())?;
    let mut notes = vec!["kerning, ligatures and shaping are left out; each character is its own glyph at its own advance".to_owned()];
    if face.is_variable() {
        notes.push("a variable font is read at its default instance".into());
    }
    let imported = Imported {
        files: Vec::new(),
        glyphs: entries.len() - 1,
        segments: first,
        missing,
        notes,
    };
    Ok((fields, imported))
}

/// The Fields of the characters `codes` asks for, a glyph row each in the
/// order asked (row `k` codes[k]'s), for a scene's `font` node: a code
/// the font lacks, or none (zero or below), is the missing glyph's row
/// with its code -1 and no outline; no codes, one such row.
pub fn glyphs_of(bytes: &[u8], codes: &[i32]) -> Result<Fields, String> {
    let face = ttf_parser::Face::parse(bytes, 0)
        .map_err(|e| format!("not a TrueType or OpenType font this reads: {e}"))?;
    let mut drawer = Drawer::new(&face);
    let missing_advance =
        i32::from(face.glyph_hor_advance(ttf_parser::GlyphId(0)).unwrap_or(0)) as u32;
    let mut glyphs: Vec<u32> = Vec::new();
    for &code in codes {
        let id = u32::try_from(code)
            .ok()
            .filter(|&c| c > 0)
            .and_then(char::from_u32)
            .and_then(|c| face.glyph_index(c));
        match id {
            Some(id) => glyphs.extend(drawer.row(code, id)),
            None => glyphs.extend([-1i32 as u32, missing_advance, 0, 0, 0, 0, 0, 0]),
        }
    }
    let rows = codes.len().max(1);
    if codes.is_empty() {
        glyphs.extend([-1i32 as u32, missing_advance, 0, 0, 0, 0, 0, 0]);
    }
    drawer.fields(glyphs, rows)
}

/// A font node's Fields before its font is read, or when it cannot be:
/// one row of the missing glyph's (code -1, no outline), one segment of
/// zeros, and zero measures.
pub fn placeholder() -> Fields {
    let field = |schema: Schema, words: Vec<u32>| {
        FieldData::from_storage(schema, Domain::Rows(1), words)
            .expect("one row of the schema's words")
    };
    Fields {
        glyphs: field(glyph_schema(), vec![-1i32 as u32, 0, 0, 0, 0, 0, 0, 0]),
        segments: field(segment_schema(), vec![0; 4]),
        metrics: field(metrics_schema(), vec![0; 4]),
    }
}

/// Draws a face's glyphs into one run of segments, a glyph drawn once
/// however many characters or rows share it.
struct Drawer<'a> {
    face: &'a ttf_parser::Face<'a>,
    em: f32,
    /// Each glyph drawn: where its segments begin, how many, its box.
    drawn: std::collections::HashMap<u16, [u32; 6]>,
    segments: Vec<u32>,
    first: usize,
}

impl<'a> Drawer<'a> {
    fn new(face: &'a ttf_parser::Face<'a>) -> Self {
        Self {
            face,
            em: 1.0 / f32::from(face.units_per_em()),
            drawn: std::collections::HashMap::new(),
            segments: Vec::new(),
            first: 0,
        }
    }

    /// Glyph `id`'s row for `code`, in [`glyph_schema`]'s layout, its
    /// outline drawn if no row before drew it.
    fn row(&mut self, code: i32, id: ttf_parser::GlyphId) -> [u32; 8] {
        let advance = i32::from(self.face.glyph_hor_advance(id).unwrap_or(0));
        if let Some(&[at, count, lo_x, lo_y, hi_x, hi_y]) = self.drawn.get(&id.0) {
            return [
                code as u32,
                advance as u32,
                at,
                count,
                lo_x,
                lo_y,
                hi_x,
                hi_y,
            ];
        }
        let mut outline = Outline {
            em: self.em,
            tolerance: TOLERANCE,
            start: [0.0; 2],
            at: [0.0; 2],
            segments: Vec::new(),
        };
        self.face.outline_glyph(id, &mut outline);
        outline.close();
        let (mut lo, mut hi) = ([0.0f32; 2], [0.0f32; 2]);
        if let Some(s) = outline.segments.first() {
            lo = [s[0], s[1]];
            hi = lo;
        }
        for s in &outline.segments {
            for (x, y) in [(s[0], s[1]), (s[2], s[3])] {
                lo = [lo[0].min(x), lo[1].min(y)];
                hi = [hi[0].max(x), hi[1].max(y)];
            }
            self.segments.extend(s.iter().map(|v| v.to_bits()));
        }
        let count = outline.segments.len();
        let [lo_x, lo_y, hi_x, hi_y] = [lo[0], lo[1], hi[0], hi[1]].map(f32::to_bits);
        let row = [self.first as u32, count as u32, lo_x, lo_y, hi_x, hi_y];
        self.drawn.insert(id.0, row);
        self.first += count;
        [
            code as u32,
            advance as u32,
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
        ]
    }

    /// The three Fields: `rows` glyph rows of `glyphs`, the segments drawn
    /// (one of zeros when none), the face's measures.
    fn fields(self, glyphs: Vec<u32>, rows: usize) -> Result<Fields, String> {
        let face = self.face;
        let units = face.units_per_em();
        let em = self.em;
        let mut segments = self.segments;
        if segments.is_empty() {
            segments = vec![0; 4];
        }
        let metrics = [
            i32::from(units) as u32,
            (f32::from(face.ascender()) * em).to_bits(),
            (f32::from(face.descender()) * em).to_bits(),
            (f32::from(face.line_gap()) * em).to_bits(),
        ];
        let field = |schema: Schema, rows: usize, words: Vec<u32>| {
            FieldData::from_storage(schema, Domain::Rows(rows as u32), words)
                .map_err(|e| e.to_string())
        };
        Ok(Fields {
            glyphs: field(glyph_schema(), rows, glyphs)?,
            segments: field(segment_schema(), self.first.max(1), segments)?,
            metrics: field(metrics_schema(), 1, metrics.to_vec())?,
        })
    }
}

/// A glyph's outline as line segments in ems, curves cut until within
/// `tolerance` of them.
struct Outline {
    em: f32,
    tolerance: f32,
    start: [f32; 2],
    at: [f32; 2],
    segments: Vec<[f32; 4]>,
}

impl Outline {
    fn line(&mut self, to: [f32; 2]) {
        if to != self.at {
            self.segments.push([self.at[0], self.at[1], to[0], to[1]]);
        }
        self.at = to;
    }

    fn close(&mut self) {
        self.line(self.start);
    }

    fn scaled(&self, x: f32, y: f32) -> [f32; 2] {
        [x * self.em, y * self.em]
    }

    // How many pieces a curve bending `bend` from its chord is cut into.
    fn pieces(&self, bend: f32) -> usize {
        ((bend / self.tolerance).sqrt().ceil() as usize).clamp(1, 64)
    }
}

impl ttf_parser::OutlineBuilder for Outline {
    fn move_to(&mut self, x: f32, y: f32) {
        self.close();
        self.start = self.scaled(x, y);
        self.at = self.start;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let to = self.scaled(x, y);
        self.line(to);
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
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

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
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

    fn close(&mut self) {
        Outline::close(self);
    }
}

/// A small TrueType font built from the format's tables, so tests read a
/// font without carrying one: 1000 units an em, a box for a missing
/// character, a space, `I`, `l`, and an `o` of two loops of quadratic
/// curves that dips just below the baseline.
pub mod sample {
    /// A point of a TrueType outline: x, y, and whether it is on the curve.
    type Point = (i16, i16, bool);

    /// A simple glyph from its contours, coordinates as deltas of `i16`.
    fn glyph(contours: &[&[Point]]) -> Vec<u8> {
        let points: Vec<Point> = contours.iter().flat_map(|c| c.iter().copied()).collect();
        if points.is_empty() {
            return Vec::new();
        }
        let mut out = (contours.len() as i16).to_be_bytes().to_vec();
        let lo = |f: fn(&Point) -> i16| points.iter().map(f).min().unwrap();
        let hi = |f: fn(&Point) -> i16| points.iter().map(f).max().unwrap();
        for bound in [lo(|p| p.0), lo(|p| p.1), hi(|p| p.0), hi(|p| p.1)] {
            out.extend(bound.to_be_bytes());
        }
        let mut end = 0u16;
        for contour in contours {
            end += contour.len() as u16;
            out.extend((end - 1).to_be_bytes());
        }
        out.extend(0u16.to_be_bytes()); // no instructions
        out.extend(points.iter().map(|p| u8::from(p.2)));
        for axis in [|p: &Point| p.0, |p: &Point| p.1] {
            let mut last = 0i16;
            for p in &points {
                out.extend((axis(p) - last).to_be_bytes());
                last = axis(p);
            }
        }
        out
    }

    /// The font.
    pub fn latin() -> Vec<u8> {
        let bar = |x0: i16, x1: i16, top: i16| {
            glyph(&[&[
                (x0, 0, true),
                (x0, top, true),
                (x1, top, true),
                (x1, 0, true),
            ]])
        };
        // A loop about (300, 260): on the curve at its extremes, the
        // corners of its box between.
        let ring = |r: i16| -> Vec<Point> {
            let (x, y) = (300, 260);
            vec![
                (x + r, y, true),
                (x + r, y + r, false),
                (x, y + r, true),
                (x - r, y + r, false),
                (x - r, y, true),
                (x - r, y - r, false),
                (x, y - r, true),
                (x + r, y - r, false),
            ]
        };
        let mut inner = ring(170);
        inner.reverse();
        // (advance, outline), in glyph order.
        let glyphs = [
            (500u16, bar(100, 400, 700)),
            (260, Vec::new()),
            (300, bar(100, 200, 714)),
            (290, bar(100, 190, 760)),
            (600, glyph(&[&ring(270), &inner])),
        ];
        let cmap_groups: [(u32, u32); 4] = [
            (' ' as u32, 1),
            ('I' as u32, 2),
            ('l' as u32, 3),
            ('o' as u32, 4),
        ];

        let be16 = |v: u16| v.to_be_bytes();
        let mut glyf = Vec::new();
        let mut loca = Vec::new();
        let mut hmtx = Vec::new();
        for (advance, outline) in &glyphs {
            loca.extend((glyf.len() as u32).to_be_bytes());
            glyf.extend(outline);
            while !glyf.len().is_multiple_of(4) {
                glyf.push(0);
            }
            hmtx.extend(be16(*advance));
            hmtx.extend(0i16.to_be_bytes());
        }
        loca.extend((glyf.len() as u32).to_be_bytes());
        let count = glyphs.len() as u16;

        let mut head = Vec::new();
        head.extend(0x0001_0000u32.to_be_bytes()); // version 1.0
        head.extend(0x0001_0000u32.to_be_bytes()); // revision
        head.extend(0u32.to_be_bytes()); // checksum adjustment
        head.extend(0x5F0F_3CF5u32.to_be_bytes()); // magic
        head.extend(be16(0)); // flags
        head.extend(be16(1000)); // units an em
        head.extend([0; 16]); // created, modified
        for bound in [0i16, -10, 570, 760] {
            head.extend(bound.to_be_bytes());
        }
        head.extend(be16(0)); // style
        head.extend(be16(8)); // smallest readable size
        head.extend(2i16.to_be_bytes()); // direction
        head.extend(1i16.to_be_bytes()); // long offsets in `loca`
        head.extend(0i16.to_be_bytes()); // glyph data format

        let mut hhea = Vec::new();
        hhea.extend(0x0001_0000u32.to_be_bytes());
        for v in [800i16, -200, 90] {
            hhea.extend(v.to_be_bytes()); // ascent, descent, gap
        }
        hhea.extend(be16(600)); // widest advance
        hhea.extend([0; 22]); // bearings, extent, caret, reserved, format
        hhea.extend(be16(count)); // advances in `hmtx`

        let mut maxp = 0x0000_5000u32.to_be_bytes().to_vec();
        maxp.extend(be16(count));

        // One subtable, Windows full Unicode, format 12: ranges of codes.
        let mut cmap = Vec::new();
        cmap.extend(be16(0));
        cmap.extend(be16(1));
        cmap.extend(be16(3));
        cmap.extend(be16(10));
        cmap.extend(12u32.to_be_bytes());
        cmap.extend(be16(12));
        cmap.extend(be16(0));
        cmap.extend((16 + 12 * cmap_groups.len() as u32).to_be_bytes());
        cmap.extend(0u32.to_be_bytes()); // language
        cmap.extend((cmap_groups.len() as u32).to_be_bytes());
        for (code, id) in cmap_groups {
            for v in [code, code, id] {
                cmap.extend(v.to_be_bytes());
            }
        }

        // The table directory, sorted by tag, then the tables.
        let tables: [(&[u8; 4], Vec<u8>); 7] = [
            (b"cmap", cmap),
            (b"glyf", glyf),
            (b"head", head),
            (b"hhea", hhea),
            (b"hmtx", hmtx),
            (b"loca", loca),
            (b"maxp", maxp),
        ];
        let mut font = 0x0001_0000u32.to_be_bytes().to_vec();
        for v in [7u16, 64, 2, 48] {
            font.extend(be16(v)); // tables, search range, selector, shift
        }
        let mut offset = 12 + 16 * tables.len();
        for (tag, table) in &tables {
            font.extend(*tag);
            font.extend(0u32.to_be_bytes()); // checksum
            font.extend((offset as u32).to_be_bytes());
            font.extend((table.len() as u32).to_be_bytes());
            offset += table.len().div_ceil(4) * 4;
        }
        for (_, table) in &tables {
            font.extend(table);
            while !font.len().is_multiple_of(4) {
                font.push(0);
            }
        }
        font
    }
}

#[cfg(test)]
mod tests {
    use super::sample::latin;
    use super::*;

    fn words(data: &FieldData) -> Vec<u32> {
        data.storage().to_vec()
    }

    #[test]
    fn the_glyphs_asked_for_are_in_order_of_code_after_the_missing_one() {
        let (fields, imported) = read(&latin(), Some("oIl o")).unwrap();
        assert_eq!(imported.glyphs, 4);
        let glyphs = words(&fields.glyphs);
        let codes: Vec<i32> = glyphs.chunks(8).map(|g| g[0] as i32).collect();
        assert_eq!(
            codes,
            vec![-1, ' ' as i32, 'I' as i32, 'l' as i32, 'o' as i32]
        );
        // A space has an advance and no outline.
        let space = &glyphs[8..16];
        assert!(space[1] as i32 > 0);
        assert_eq!(space[3], 0);
    }

    #[test]
    fn an_outline_is_closed_and_within_its_box() {
        let (fields, _) = read(&latin(), Some("o")).unwrap();
        let glyphs = words(&fields.glyphs);
        let o = &glyphs[8..16];
        let (first, count) = (o[2] as usize, o[3] as usize);
        let lo = [f32::from_bits(o[4]), f32::from_bits(o[5])];
        let hi = [f32::from_bits(o[6]), f32::from_bits(o[7])];
        let segments: Vec<f32> = words(&fields.segments)
            .into_iter()
            .map(f32::from_bits)
            .collect();
        let mine = &segments[first * 4..(first + count) * 4];
        // An `o` is two closed loops: every end is some segment's start.
        for s in mine.chunks(4) {
            assert!(mine.chunks(4).any(|t| t[0] == s[2] && t[1] == s[3]));
            for v in [s[0], s[2]] {
                assert!(v >= lo[0] && v <= hi[0]);
            }
        }
        // It sits on the baseline and rises about half an em.
        assert!(lo[1] < 0.0 && lo[1] > -0.03);
        assert!(hi[1] > 0.5 && hi[1] < 0.6);
    }

    #[test]
    fn a_character_the_font_lacks_is_named() {
        let (_, imported) = read(&latin(), Some("o中")).unwrap();
        assert_eq!(imported.missing, vec!['中']);
        assert_eq!(imported.glyphs, 1);
    }

    #[test]
    fn what_is_not_a_font_is_refused() {
        assert!(read(b"not a font", None).is_err());
    }

    #[test]
    fn a_font_node_has_a_row_for_each_code_asked_in_order() {
        let fields = glyphs_of(&latin(), &['o' as i32, 0x4E2D, 'I' as i32, 'o' as i32]).unwrap();
        let glyphs = words(&fields.glyphs);
        let codes: Vec<i32> = glyphs.chunks(8).map(|g| g[0] as i32).collect();
        assert_eq!(codes, vec!['o' as i32, -1, 'I' as i32, 'o' as i32]);
        // The second `o` shares the first's segments; the missing row has none.
        assert_eq!(glyphs[2..4], glyphs[26..28]);
        assert_eq!(glyphs[11], 0);
        let empty = glyphs_of(&latin(), &[]).unwrap();
        assert_eq!(words(&empty.glyphs)[0] as i32, -1);
        assert_eq!(empty.glyphs.storage().len(), 8);
    }
}
