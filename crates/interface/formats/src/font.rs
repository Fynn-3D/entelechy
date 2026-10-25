//! A font's Fields, as `entelechy import` writes them from a font file and
//! `entelechy export` writes a font file from them: the layouts both
//! sides read. A glyph's outline is line segments in ems, y up, each
//! contour closed and its segments one after another, filled by the
//! nonzero rule.

use entelechy_host::{Scalar, Schema};

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
