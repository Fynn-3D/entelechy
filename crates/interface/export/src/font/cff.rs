//! The `CFF ` table of an OpenType font: each glyph a Type 2 charstring,
//! a contour's quadratics as the cubics they are.

use super::Point;

/// A number as a charstring (or, `dict`, a DICT) writes it; a large one in
/// a DICT is four bytes.
fn number(v: i32, dict: bool) -> Vec<u8> {
    match v {
        -107..=107 => vec![(v + 139) as u8],
        108..=1131 => vec![((v - 108) / 256 + 247) as u8, ((v - 108) % 256) as u8],
        -1131..=-108 => vec![((-v - 108) / 256 + 251) as u8, ((-v - 108) % 256) as u8],
        _ if !dict && (-32768..=32767).contains(&v) => {
            let mut out = vec![28];
            out.extend((v as i16).to_be_bytes());
            out
        }
        _ => {
            let mut out = vec![29];
            out.extend(v.to_be_bytes());
            out
        }
    }
}

/// A DICT entry holding a four-byte number, whatever its size, so an
/// offset can be written before it is known.
fn long(v: i32) -> Vec<u8> {
    let mut out = vec![29];
    out.extend(v.to_be_bytes());
    out
}

/// An INDEX of `items`.
fn index(items: &[Vec<u8>]) -> Vec<u8> {
    if items.is_empty() {
        return vec![0, 0];
    }
    let mut offsets = vec![1usize];
    for item in items {
        offsets.push(offsets.last().expect("an offset") + item.len());
    }
    let last = *offsets.last().expect("an offset");
    let size = if last < 1 << 8 {
        1
    } else if last < 1 << 16 {
        2
    } else if last < 1 << 24 {
        3
    } else {
        4
    };
    let mut out = (items.len() as u16).to_be_bytes().to_vec();
    out.push(size as u8);
    for offset in offsets {
        out.extend(&(offset as u32).to_be_bytes()[4 - size..]);
    }
    for item in items {
        out.extend(item);
    }
    out
}

/// A contour's pieces from an on-curve point: lines to a point, and
/// quadratics through a control to a point, the points two controls imply
/// made real.
enum Piece {
    Line((f64, f64)),
    Quad((f64, f64), (f64, f64)),
}

fn pieces(contour: &[Point]) -> Option<((f64, f64), Vec<Piece>)> {
    let n = contour.len();
    let mut seq: Vec<(f64, f64, bool)> = Vec::new();
    for (i, &(x, y, on)) in contour.iter().enumerate() {
        seq.push((f64::from(x), f64::from(y), on));
        let (nx, ny, next_on) = contour[(i + 1) % n];
        if !on && !next_on {
            seq.push((
                (f64::from(x) + f64::from(nx)) / 2.0,
                (f64::from(y) + f64::from(ny)) / 2.0,
                true,
            ));
        }
    }
    let first = seq.iter().position(|p| p.2)?;
    seq.rotate_left(first);
    let m = seq.len();
    let mut out = Vec::new();
    let mut k = 1;
    while k <= m {
        let p = seq[k % m];
        if p.2 {
            out.push(Piece::Line((p.0, p.1)));
            k += 1;
        } else {
            let q = seq[(k + 1) % m];
            out.push(Piece::Quad((p.0, p.1), (q.0, q.1)));
            k += 2;
        }
    }
    Some(((seq[0].0, seq[0].1), out))
}

/// A charstring's operators, arguments gathered under one operator while
/// it repeats.
struct Charstring {
    out: Vec<u8>,
    pending: Option<u8>,
    args: Vec<i32>,
    lead: Vec<i32>,
}

impl Charstring {
    fn flush(&mut self) {
        if let Some(op) = self.pending.take() {
            for v in self.lead.drain(..).chain(self.args.drain(..)) {
                self.out.extend(number(v, false));
            }
            self.out.push(op);
        }
        self.args.clear();
    }

    fn op(&mut self, code: u8, values: &[i32]) {
        if self.pending != Some(code) || self.args.len() + values.len() > 48 {
            self.flush();
        }
        self.pending = Some(code);
        self.args.extend(values);
    }
}

/// A glyph as a Type 2 charstring: each contour turned the way CFF fills
/// (counterclockwise), its width given where it is not `default`.
pub fn charstring(contours: &[Vec<Point>], width: u16, default: u16) -> Vec<u8> {
    let mut cs = Charstring {
        out: Vec::new(),
        pending: None,
        args: Vec::new(),
        lead: if width == default {
            Vec::new()
        } else {
            vec![i32::from(width)]
        },
    };
    let (mut px, mut py) = (0i32, 0i32);
    for contour in contours {
        let reversed: Vec<Point> = contour.iter().rev().copied().collect();
        let Some((start, pieces)) = pieces(&reversed) else {
            continue;
        };
        let (sx, sy) = (start.0.round() as i32, start.1.round() as i32);
        cs.flush();
        cs.op(21, &[sx - px, sy - py]);
        cs.flush();
        (px, py) = (sx, sy);
        let last = pieces.len() - 1;
        for (k, piece) in pieces.iter().enumerate() {
            match *piece {
                Piece::Line((x, y)) => {
                    let (x, y) = (x.round() as i32, y.round() as i32);
                    if k == last && (x, y) == (sx, sy) {
                        break; // the contour closes itself
                    }
                    cs.op(5, &[x - px, y - py]);
                    (px, py) = (x, y);
                }
                Piece::Quad((cx, cy), (x, y)) => {
                    let (fx, fy) = (f64::from(px), f64::from(py));
                    let (x1, y1) = (
                        (fx + 2.0 * (cx - fx) / 3.0).round() as i32,
                        (fy + 2.0 * (cy - fy) / 3.0).round() as i32,
                    );
                    let (x2, y2) = (
                        (x + 2.0 * (cx - x) / 3.0).round() as i32,
                        (y + 2.0 * (cy - y) / 3.0).round() as i32,
                    );
                    let (x, y) = (x.round() as i32, y.round() as i32);
                    cs.op(8, &[x1 - px, y1 - py, x2 - x1, y2 - y1, x - x2, y - y2]);
                    (px, py) = (x, y);
                }
            }
        }
    }
    cs.flush();
    let lead = std::mem::take(&mut cs.lead);
    for v in lead {
        cs.out.extend(number(v, false));
    }
    cs.out.push(14);
    cs.out
}

/// The `CFF ` table: one font named `postscript`, its glyphs' charstrings
/// (glyph 0 first), `units` an em, a whole em its default width.
pub fn table(
    charstrings: &[Vec<u8>],
    family: &str,
    postscript: &str,
    bounds: [i16; 4],
    units: u16,
) -> Vec<u8> {
    let mut strings: Vec<Vec<u8>> = vec![family.as_bytes().to_vec(), family.as_bytes().to_vec()];
    strings.extend((1..charstrings.len()).map(|k| format!("g{k}").into_bytes()));
    let top = |charset: i32, glyphs: i32, private: i32, private_size: i32| {
        let mut out = Vec::new();
        out.extend(number(391, true));
        out.push(2); // FullName
        out.extend(number(392, true));
        out.push(3); // FamilyName
        for v in bounds {
            out.extend(number(i32::from(v), true));
        }
        out.push(5); // FontBBox
        out.extend(long(charset));
        out.push(15);
        out.extend(long(glyphs));
        out.push(17);
        out.extend(long(private_size));
        out.extend(long(private));
        out.push(18);
        out
    };
    let mut private = number(i32::from(units), true);
    private.push(20); // defaultWidthX
    private.extend(number(0, true));
    private.push(21); // nominalWidthX
    let mut charset = vec![2u8];
    charset.extend(393u16.to_be_bytes());
    charset.extend((charstrings.len().saturating_sub(2) as u16).to_be_bytes());
    let glyphs = index(charstrings);
    let mut head = vec![1u8, 0, 4, 4];
    head.extend(index(&[postscript.as_bytes().to_vec()]));
    let mut rest = index(&strings);
    rest.extend(index(&[]));
    let before = (head.len() + index(&[top(0, 0, 0, 0)]).len() + rest.len()) as i32;
    let at_glyphs = before + charset.len() as i32;
    let at_private = at_glyphs + glyphs.len() as i32;
    let mut out = head;
    out.extend(index(&[top(
        before,
        at_glyphs,
        at_private,
        private.len() as i32,
    )]));
    out.extend(rest);
    out.extend(charset);
    out.extend(glyphs);
    out.extend(private);
    out
}
