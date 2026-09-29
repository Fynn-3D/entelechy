//! A TrueType or OpenType font as the Fields the `font` package lays out
//! and the `vector` package draws: each glyph's outline flattened to line
//! segments in ems, y up, filled by the nonzero rule. `ttf-parser` reads
//! the font (quadratic and cubic outlines, the character map). What is
//! left out is named in the notes.
//!
//! Three files are written:
//!
//! - `<stem>.glyphs.field`: `Rows<Letterform>` in order of code, the layout
//!   [`glyph_schema`] gives; the first, of code -1, is the font's glyph for
//!   a character it lacks;
//! - `<stem>.segments.field`: `Rows<Vec4>`, a segment `a.x, a.y, b.x, b.y`;
//! - `<stem>.metrics.field`: `Rows<Metrics>`, one row, [`metrics_schema`].

use std::path::{Path, PathBuf};

use orla_host::{Domain, FieldData, Scalar, Schema};

/// How finely curves are cut: the largest distance, in ems, a segment may
/// stray from its curve.
const TOLERANCE: f32 = 1.0 / 2048.0;

/// A glyph's layout, as the `font` package declares it: `code: Int;
/// advance: Int; first: Int; count: Int; lo: Vec2; hi: Vec2`. The advance
/// is in the font's units, so a line's advances sum exactly; the box is
/// in ems.
pub fn glyph_schema() -> Schema {
    let f = Scalar::Float;
    let i = Scalar::Int;
    Schema::new(vec![i, i, i, i, f, f, f, f])
}

/// The font's measures: `units: Int; ascent: Float; descent: Float; gap:
/// Float`, the font's units in an em and the rest in ems.
pub fn metrics_schema() -> Schema {
    let f = Scalar::Float;
    Schema::new(vec![Scalar::Int, f, f, f])
}

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
    /// `Rows<Letterform>`, in order of code.
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
        std::fs::write(&file, orla_formats::file::encode_data(data))
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
    let units = face.units_per_em();
    let em = 1.0 / f32::from(units);
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
    let mut glyphs: Vec<u32> = Vec::new();
    let mut segments: Vec<u32> = Vec::new();
    let mut first = 0usize;
    for &(code, id) in &entries {
        let mut outline = Outline {
            em,
            tolerance: TOLERANCE,
            start: [0.0; 2],
            at: [0.0; 2],
            segments: Vec::new(),
        };
        face.outline_glyph(id, &mut outline);
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
            segments.extend(s.iter().map(|v| v.to_bits()));
        }
        let count = outline.segments.len();
        let advance = i32::from(face.glyph_hor_advance(id).unwrap_or(0));
        glyphs.extend([code as u32, advance as u32, first as u32, count as u32]);
        glyphs.extend([lo[0], lo[1], hi[0], hi[1]].map(f32::to_bits));
        first += count;
    }
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
        FieldData::from_storage(schema, Domain::Rows(rows as u32), words).map_err(|e| e.to_string())
    };
    let fields = Fields {
        glyphs: field(glyph_schema(), entries.len(), glyphs)?,
        segments: field(Schema::new(vec![Scalar::Float; 4]), first.max(1), segments)?,
        metrics: field(metrics_schema(), 1, metrics.to_vec())?,
    };
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

#[cfg(test)]
mod tests {
    use super::*;

    fn latin() -> Vec<u8> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../examples/type/source/NotoSans-Regular.ttf");
        std::fs::read(path).unwrap()
    }

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
        let (_, imported) = read(&latin(), Some("a中")).unwrap();
        assert_eq!(imported.missing, vec!['中']);
        assert_eq!(imported.glyphs, 1);
    }

    #[test]
    fn what_is_not_a_font_is_refused() {
        assert!(read(b"not a font", None).is_err());
    }
}
