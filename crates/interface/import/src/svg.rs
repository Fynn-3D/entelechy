//! An SVG document as the Fields the `vector` package draws: its paths,
//! each fill and each stroke a shape in paint order, flattened to line
//! segments in document units. `usvg` resolves the document (CSS, units,
//! the viewBox, basic shapes, `use`, transforms, gradient units); a
//! stroke becomes the outline it paints. What is left out is named in the
//! notes.
//!
//! Four files are written:
//!
//! - `<stem>.segments.field`: `Rows<Vec4>`, a segment `a.x, a.y, b.x, b.y`;
//! - `<stem>.shapes.field`: `Rows<Shape>` in paint order, the layout
//!   [`shape_schema`] gives;
//! - `<stem>.stops.field`: `Rows<Stop>`, `offset` and `rgba`, the gradients'
//!   stops (one unused stop when no shape has a gradient);
//! - `<stem>.view.field`: `Rows<Vec2>`, one row: the document's size.

use std::path::{Path, PathBuf};

use orla_host::{Domain, FieldData, Scalar, Schema};
use usvg::tiny_skia_path::{self, PathSegment, Point, Transform};

/// How finely curves are cut: the largest distance, in parts of the
/// document's diagonal, a segment may stray from its curve.
const TOLERANCE: f32 = 1.0 / 8192.0;

/// A shape's layout, as the `vector` package declares it:
/// `color: Vec4; first: Int; count: Int; rule: Int; paint: Int; lo: Vec2;
/// hi: Vec2; gradient: Vec4; inverse: Vec4; shift: Vec2; stops: Int;
/// stop_count: Int; spread: Int`.
pub fn shape_schema() -> Schema {
    let f = Scalar::Float;
    let i = Scalar::Int;
    Schema::new(vec![
        f, f, f, f, i, i, i, i, f, f, f, f, f, f, f, f, f, f, f, f, f, f, i, i, i,
    ])
}

/// A stop's layout: `offset: Float; color: Vec4`.
pub fn stop_schema() -> Schema {
    Schema::new(vec![Scalar::Float; 5])
}

/// What an SVG import produced.
#[derive(Debug)]
pub struct Imported {
    /// The files written.
    pub files: Vec<PathBuf>,
    /// The document's size in its own units.
    pub size: (f32, f32),
    /// Shapes drawn.
    pub shapes: usize,
    /// Segments over all shapes.
    pub segments: usize,
    /// What was left out, in words.
    pub notes: Vec<String>,
}

/// A shape before it is written: its paint, rule, box and segments.
struct Shape {
    color: [f32; 4],
    rule: i32,
    paint: i32,
    gradient: [f32; 4],
    inverse: [f32; 6],
    stops: (usize, usize),
    spread: i32,
    segments: Vec<[f32; 4]>,
}

/// What is collected while the tree is walked.
#[derive(Default)]
struct Drawing {
    shapes: Vec<Shape>,
    stops: Vec<[f32; 5]>,
    notes: Vec<String>,
    tolerance: f32,
}

impl Drawing {
    fn note(&mut self, note: impl Into<String>) {
        let note = note.into();
        if !self.notes.contains(&note) {
            self.notes.push(note);
        }
    }
}

/// Reads the SVG at `path` and writes its Fields into `out_dir`.
pub fn import(path: &Path, out_dir: &Path) -> Result<Imported, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("drawing")
        .to_owned();
    let (fields, imported) = read(&bytes)?;
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    let mut files = Vec::new();
    for (name, data) in [
        ("segments", &fields.segments),
        ("shapes", &fields.shapes),
        ("stops", &fields.stops),
        ("view", &fields.view),
    ] {
        let file = out_dir.join(format!("{stem}.{name}.field"));
        std::fs::write(&file, orla_formats::file::encode_data(data))
            .map_err(|e| format!("{}: {e}", file.display()))?;
        files.push(file);
    }
    Ok(Imported { files, ..imported })
}

/// The four Fields of a drawing.
#[derive(Debug)]
pub struct Fields {
    /// `Rows<Vec4>`, the segments.
    pub segments: FieldData,
    /// `Rows<Shape>`, the shapes in paint order.
    pub shapes: FieldData,
    /// `Rows<Stop>`, the gradients' stops.
    pub stops: FieldData,
    /// `Rows<Vec2>`, the document's size.
    pub view: FieldData,
}

/// Reads an SVG document into its Fields and what was left out.
pub fn read(bytes: &[u8]) -> Result<(Fields, Imported), String> {
    let tree = usvg::Tree::from_data(bytes, &usvg::Options::default())
        .map_err(|e| format!("not an SVG document this reads: {e}"))?;
    let size = tree.size();
    let (width, height) = (size.width(), size.height());
    let mut drawing = Drawing {
        tolerance: (width * width + height * height).sqrt() * TOLERANCE,
        ..Drawing::default()
    };
    walk(tree.root(), 1.0, &mut drawing);
    // Text is dropped as the document is read, fonts being left out: the
    // source says whether there was any.
    let text = std::str::from_utf8(bytes)
        .ok()
        .and_then(|source| usvg::roxmltree::Document::parse(source).ok())
        .is_some_and(|doc| doc.descendants().any(|node| node.has_tag_name("text")));
    if text {
        drawing.note("text is left out; convert it to paths first");
    }
    if drawing.shapes.is_empty() {
        return Err("the document draws no filled or stroked path".into());
    }
    let mut segments: Vec<u32> = Vec::new();
    let mut shapes: Vec<u32> = Vec::new();
    let mut first = 0usize;
    for shape in &drawing.shapes {
        let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
        for s in &shape.segments {
            for (x, y) in [(s[0], s[1]), (s[2], s[3])] {
                lo = [lo[0].min(x), lo[1].min(y)];
                hi = [hi[0].max(x), hi[1].max(y)];
            }
            segments.extend(s.iter().map(|v| v.to_bits()));
        }
        let count = shape.segments.len();
        let f = |v: f32| v.to_bits();
        let i = |v: i32| v as u32;
        shapes.extend(shape.color.iter().map(|v| f(*v)));
        shapes.extend([
            i(first as i32),
            i(count as i32),
            i(shape.rule),
            i(shape.paint),
        ]);
        shapes.extend([f(lo[0]), f(lo[1]), f(hi[0]), f(hi[1])]);
        shapes.extend(shape.gradient.iter().map(|v| f(*v)));
        let m = shape.inverse;
        shapes.extend([f(m[0]), f(m[1]), f(m[2]), f(m[3]), f(m[4]), f(m[5])]);
        shapes.extend([
            i(shape.stops.0 as i32),
            i(shape.stops.1 as i32),
            i(shape.spread),
        ]);
        first += count;
    }
    let mut stops: Vec<u32> = drawing
        .stops
        .iter()
        .flatten()
        .map(|v| v.to_bits())
        .collect();
    let stop_rows = drawing.stops.len().max(1);
    if stops.is_empty() {
        stops = vec![0; 5];
    }
    let field = |schema: Schema, rows: usize, words: Vec<u32>| {
        FieldData::from_storage(schema, Domain::Rows(rows as u32), words).map_err(|e| e.to_string())
    };
    let fields = Fields {
        segments: field(Schema::new(vec![Scalar::Float; 4]), first, segments)?,
        shapes: field(shape_schema(), drawing.shapes.len(), shapes)?,
        stops: field(stop_schema(), stop_rows, stops)?,
        view: field(
            Schema::new(vec![Scalar::Float; 2]),
            1,
            vec![width.to_bits(), height.to_bits()],
        )?,
    };
    let imported = Imported {
        files: Vec::new(),
        size: (width, height),
        shapes: drawing.shapes.len(),
        segments: first,
        notes: drawing.notes,
    };
    Ok((fields, imported))
}

fn walk(group: &usvg::Group, opacity: f32, drawing: &mut Drawing) {
    let opacity = opacity * group.opacity().get();
    if group.opacity().get() < 1.0 && group.children().len() > 1 {
        drawing.note("a group's opacity is taken into each of its shapes, so where they overlap they show through one another");
    }
    if group.clip_path().is_some() {
        drawing.note("clip paths are left out; what they clip is drawn whole");
    }
    if group.mask().is_some() {
        drawing.note("masks are left out; what they mask is drawn whole");
    }
    if !group.filters().is_empty() {
        drawing.note("filters are left out; what they filter is drawn plain");
    }
    if group.blend_mode() != usvg::BlendMode::Normal {
        drawing.note("blend modes are left out; shapes are drawn over what is beneath");
    }
    for node in group.children() {
        match node {
            usvg::Node::Group(inner) => walk(inner, opacity, drawing),
            usvg::Node::Path(path) => shape(path, opacity, drawing),
            usvg::Node::Image(_) => drawing.note("images are left out"),
            usvg::Node::Text(_) => drawing.note("text is left out; convert it to paths first"),
        }
    }
}

fn shape(path: &usvg::Path, opacity: f32, drawing: &mut Drawing) {
    if !path.is_visible() {
        return;
    }
    let transform = path.abs_transform();
    let fill = path.fill().map(|fill| {
        let rule = match fill.rule() {
            usvg::FillRule::NonZero => 0,
            usvg::FillRule::EvenOdd => 1,
        };
        (
            path.data().clone(),
            fill.paint().clone(),
            fill.opacity().get(),
            rule,
        )
    });
    let stroke = path.stroke().and_then(|stroke| {
        let ts = stroke.to_tiny_skia();
        // Curves of the outline are cut finely enough for the document
        // drawn at its size and some; the segments below are cut again.
        let scale = transform.get_scale();
        let res = (scale.0.max(scale.1) * 4.0).max(1.0);
        let dashed = match &ts.dash {
            Some(dash) => path.data().dash(dash, res),
            None => Some(path.data().clone()),
        }?;
        let outline = dashed.stroke(&ts, res)?;
        Some((outline, stroke.paint().clone(), stroke.opacity().get(), 0))
    });
    let order = match path.paint_order() {
        usvg::PaintOrder::FillAndStroke => [fill, stroke],
        usvg::PaintOrder::StrokeAndFill => [stroke, fill],
    };
    for (data, paint, alpha, rule) in order.into_iter().flatten() {
        let segments = flatten(&data, transform, drawing.tolerance);
        if segments.is_empty() {
            continue;
        }
        let alpha = alpha * opacity;
        let mut made = Shape {
            color: [0.0, 0.0, 0.0, alpha],
            rule,
            paint: 0,
            gradient: [0.0; 4],
            inverse: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            stops: (0, 0),
            spread: 0,
            segments,
        };
        match &paint {
            usvg::Paint::Color(c) => {
                made.color = [
                    f32::from(c.red) / 255.0,
                    f32::from(c.green) / 255.0,
                    f32::from(c.blue) / 255.0,
                    alpha,
                ];
            }
            usvg::Paint::LinearGradient(g) => {
                made.paint = 1;
                made.gradient = [g.x1(), g.y1(), g.x2(), g.y2()];
                gradient(&mut made, g, transform, drawing);
            }
            usvg::Paint::RadialGradient(g) => {
                made.paint = 2;
                made.gradient = [g.cx(), g.cy(), g.r().get(), 0.0];
                if (g.fx(), g.fy()) != (g.cx(), g.cy()) || g.fr().get() > 0.0 {
                    drawing.note("a radial gradient's focal point is taken at its centre");
                }
                gradient(&mut made, g, transform, drawing);
            }
            usvg::Paint::Pattern(_) => {
                drawing.note("patterns are left out; a patterned shape is not drawn");
                continue;
            }
        }
        drawing.shapes.push(made);
    }
}

/// A gradient's stops, spread and the way from the document to the
/// gradient's own space.
fn gradient(
    shape: &mut Shape,
    g: &usvg::BaseGradient,
    transform: Transform,
    drawing: &mut Drawing,
) {
    let first = drawing.stops.len();
    for stop in g.stops() {
        let c = stop.color();
        drawing.stops.push([
            stop.offset().get(),
            f32::from(c.red) / 255.0,
            f32::from(c.green) / 255.0,
            f32::from(c.blue) / 255.0,
            stop.opacity().get(),
        ]);
    }
    shape.stops = (first, drawing.stops.len() - first);
    shape.spread = match g.spread_method() {
        usvg::SpreadMethod::Pad => 0,
        usvg::SpreadMethod::Reflect => 1,
        usvg::SpreadMethod::Repeat => 2,
    };
    let to_document = transform.pre_concat(g.transform());
    if let Some(back) = to_document.invert() {
        shape.inverse = [back.sx, back.ky, back.kx, back.sy, back.tx, back.ty];
    }
    // A gradient of no stops paints nothing; of one, its colour.
    if shape.stops.1 == 0 {
        shape.color[3] = 0.0;
    }
}

/// A path as line segments in document units, each subpath closed, its
/// curves cut until no segment strays further than `tolerance`.
fn flatten(data: &tiny_skia_path::Path, transform: Transform, tolerance: f32) -> Vec<[f32; 4]> {
    let map = |p: Point| {
        let mut q = p;
        transform.map_point(&mut q);
        q
    };
    let mut out = Vec::new();
    let mut start = Point::zero();
    let mut at = Point::zero();
    let mut open = false;
    let line = |out: &mut Vec<[f32; 4]>, a: Point, b: Point| {
        if a != b {
            out.push([a.x, a.y, b.x, b.y]);
        }
    };
    for segment in data.segments() {
        match segment {
            PathSegment::MoveTo(p) => {
                if open {
                    line(&mut out, at, start);
                }
                start = map(p);
                at = start;
                open = true;
            }
            PathSegment::LineTo(p) => {
                let b = map(p);
                line(&mut out, at, b);
                at = b;
            }
            PathSegment::QuadTo(c, p) => {
                let (c, b) = (map(c), map(p));
                let dd = (at.x - 2.0 * c.x + b.x).hypot(at.y - 2.0 * c.y + b.y);
                let n = ((0.25 * dd / tolerance).sqrt().ceil() as usize).clamp(1, 1024);
                let mut prev = at;
                for k in 1..=n {
                    let t = k as f32 / n as f32;
                    let u = 1.0 - t;
                    let q = Point::from_xy(
                        u * u * at.x + 2.0 * u * t * c.x + t * t * b.x,
                        u * u * at.y + 2.0 * u * t * c.y + t * t * b.y,
                    );
                    line(&mut out, prev, q);
                    prev = q;
                }
                at = b;
            }
            PathSegment::CubicTo(c1, c2, p) => {
                let (c1, c2, b) = (map(c1), map(c2), map(p));
                let d1 = (at.x - 2.0 * c1.x + c2.x).hypot(at.y - 2.0 * c1.y + c2.y);
                let d2 = (c1.x - 2.0 * c2.x + b.x).hypot(c1.y - 2.0 * c2.y + b.y);
                let n = ((0.75 * d1.max(d2) / tolerance).sqrt().ceil() as usize).clamp(1, 1024);
                let mut prev = at;
                for k in 1..=n {
                    let t = k as f32 / n as f32;
                    let u = 1.0 - t;
                    let (w0, w1, w2, w3) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
                    let q = Point::from_xy(
                        w0 * at.x + w1 * c1.x + w2 * c2.x + w3 * b.x,
                        w0 * at.y + w1 * c1.y + w2 * c2.y + w3 * b.y,
                    );
                    line(&mut out, prev, q);
                    prev = q;
                }
                at = b;
            }
            PathSegment::Close => {
                line(&mut out, at, start);
                at = start;
                open = false;
            }
        }
    }
    if open {
        line(&mut out, at, start);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields(svg: &str) -> (Fields, Imported) {
        read(svg.as_bytes()).unwrap_or_else(|e| panic!("{e}"))
    }

    fn shape_words(f: &Fields, k: usize) -> Vec<u32> {
        f.shapes.storage()[k * 25..(k + 1) * 25].to_vec()
    }

    #[test]
    fn a_rectangle_is_four_segments_in_its_colour() {
        let (f, i) = fields(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><rect x="10" y="20" width="30" height="10" fill="#ff8000"/></svg>"##,
        );
        assert_eq!(i.size, (100.0, 50.0));
        assert_eq!((i.shapes, i.segments), (1, 4));
        let s = shape_words(&f, 0);
        let color: Vec<f32> = s[0..4].iter().map(|w| f32::from_bits(*w)).collect();
        assert_eq!(color, [1.0, 128.0 / 255.0, 0.0, 1.0]);
        assert_eq!((s[4], s[5], s[6], s[7]), (0, 4, 0, 0));
        let bounds: Vec<f32> = s[8..12].iter().map(|w| f32::from_bits(*w)).collect();
        assert_eq!(bounds, [10.0, 20.0, 40.0, 30.0]);
        assert!(i.notes.is_empty(), "{:?}", i.notes);
    }

    #[test]
    fn curves_are_cut_finely_and_close() {
        let (f, i) = fields(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200"><circle cx="100" cy="100" r="80"/></svg>"#,
        );
        assert!(i.segments > 64, "{}", i.segments);
        // Every point lies on the circle, to the tolerance.
        let words = f.segments.storage();
        for s in words.chunks_exact(4) {
            let (x, y) = (f32::from_bits(s[0]), f32::from_bits(s[1]));
            let r = (x - 100.0).hypot(y - 100.0);
            assert!((r - 80.0).abs() < 0.05, "{r}");
        }
        // The segments join end to start, round the whole circle.
        let first = (f32::from_bits(words[0]), f32::from_bits(words[1]));
        let n = words.len();
        let last = (f32::from_bits(words[n - 2]), f32::from_bits(words[n - 1]));
        assert!((first.0 - last.0).abs() < 1e-3 && (first.1 - last.1).abs() < 1e-3);
    }

    #[test]
    fn strokes_are_outlines_and_rules_gradients_and_transforms_carry() {
        let (f, i) = fields(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
              <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#000"/><stop offset="1" stop-color="#fff" stop-opacity="0.5"/></linearGradient></defs>
              <g transform="translate(10 0)" opacity="0.5">
                <path d="M0 0 H40 V40 H0 Z M10 10 H30 V30 H10 Z" fill-rule="evenodd" fill="url(#g)" stroke="red" stroke-width="4"/>
              </g>
              <text x="0" y="90">hi</text>
            </svg>"##,
        );
        assert_eq!(i.shapes, 2, "a fill and its stroke");
        let fill = shape_words(&f, 0);
        assert_eq!((fill[6], fill[7]), (1, 1), "even-odd, linear");
        assert_eq!(f32::from_bits(fill[3]), 0.5, "the group's opacity");
        // The gradient's space: the box's width is its unit, from x = 10.
        let inverse: Vec<f32> = fill[16..22].iter().map(|w| f32::from_bits(*w)).collect();
        assert!(
            (inverse[0] - 1.0 / 40.0).abs() < 1e-6 && (inverse[4] + 10.0 / 40.0).abs() < 1e-6,
            "{inverse:?}"
        );
        assert_eq!((fill[22], fill[23]), (0, 2));
        let stops = f.stops.storage();
        assert_eq!(f32::from_bits(stops[9]), 0.5, "the second stop's opacity");
        let stroke = shape_words(&f, 1);
        assert_eq!(f32::from_bits(stroke[0]), 1.0, "red");
        let lo = (f32::from_bits(stroke[8]), f32::from_bits(stroke[9]));
        assert!(
            (lo.0 - 8.0).abs() < 0.01 && (lo.1 + 2.0).abs() < 0.01,
            "{lo:?}"
        );
        assert!(i.notes.iter().any(|n| n.contains("text")), "{:?}", i.notes);
    }

    #[test]
    fn what_is_no_drawing_is_refused() {
        assert!(read(b"not svg").is_err());
        let refused = read(br#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>"#)
            .unwrap_err();
        assert!(refused.contains("draws no"), "{refused}");
    }
}
