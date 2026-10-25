//! A font file from a font's Fields (`entelechy_formats::font`), the
//! Fields `entelechy import` reads a font into: TrueType outlines in a
//! `.ttf`, the same tables compressed in a `.woff`, or CFF outlines in an
//! `.otf`.
//!
//! A glyph row's segments are its contours, each closed and its segments
//! one after another, their ends the points the contour passes (taken in
//! whole font units). They are written as curves through those points
//! ([`curves`]): quadratics in TrueType, the same as cubics in CFF, within
//! a font unit of the points, corners kept and straight runs as their
//! ends. Rows that share an outline and an advance share a glyph; the row
//! of code -1, if any, is the glyph for a character the font lacks.
//! Kerning, ligatures and hints are not written.

mod cff;
mod curves;

use std::collections::{BTreeMap, BTreeSet, HashMap};

use entelechy_formats::font::Part;
use entelechy_host::FieldData;

/// A font ready to write: its glyphs and which characters map to them.
#[derive(Debug)]
pub struct Face {
    family: String,
    units: u16,
    ascent: i16,
    descent: i16,
    gap: i16,
    /// Glyph 0 is the glyph for a missing character.
    glyphs: Vec<Glyph>,
    /// Each character and its glyph, in order of code.
    map: Vec<(u32, u16)>,
}

/// A point of a written contour: x, y, and whether it is on the curve.
type Point = (i16, i16, bool);

#[derive(Debug, Default)]
struct Glyph {
    advance: u16,
    contours: Vec<Vec<Point>>,
}

impl Glyph {
    fn bounds(&self) -> Option<[i16; 4]> {
        let mut points = self.contours.iter().flatten();
        let &(x, y, _) = points.next()?;
        Some(points.fold([x, y, x, y], |b, &(x, y, _)| {
            [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)]
        }))
    }

    fn points(&self) -> usize {
        self.contours.iter().map(Vec::len).sum()
    }
}

impl Face {
    /// Characters mapped.
    pub fn characters(&self) -> usize {
        self.map.len()
    }

    /// Glyphs, the missing character's counted.
    pub fn glyphs(&self) -> usize {
        self.glyphs.len()
    }
}

/// The font the Fields `inputs` hold (glyph rows, segments and, if given,
/// the measures; told apart by layout), named `family`: of the characters
/// of `chars`, or of every one the rows map.
pub fn face(inputs: &[FieldData], family: &str, chars: Option<&str>) -> Result<Face, String> {
    let wanted: Option<BTreeSet<u32>> = chars.map(|text| text.chars().map(u32::from).collect());
    let mut glyphs = None;
    let mut segments = None;
    let mut metrics = None;
    for data in inputs {
        match Part::of(data.schema()) {
            Some(Part::Glyphs) => glyphs = Some(data),
            Some(Part::Segments) => segments = Some(data),
            Some(Part::Metrics) => metrics = Some(data),
            None => {
                return Err(format!(
                    "a {} is no part of a font: glyph rows, segments (Rows<Vec4>) and the measures are",
                    data.schema()
                ));
            }
        }
    }
    let glyphs =
        glyphs.ok_or("a font needs its glyph rows (code, advance, first, count, lo, hi)")?;
    let segments = segments.ok_or("a font needs its outline segments (Rows<Vec4>)")?;
    let rows = glyphs.values();
    let lines: Vec<[f64; 4]> = segments
        .values()
        .chunks_exact(4)
        .map(|s| [0, 1, 2, 3].map(|k| f64::from(f32::from_bits(s[k]))))
        .collect();
    let (units, ascent, descent, gap) = match metrics {
        Some(data) => {
            let words = data.values();
            let float = |k: usize| f64::from(f32::from_bits(words[k]));
            (words[0] as i32, float(1), float(2), float(3))
        }
        None => {
            // Without measures: a thousand units, the outlines' own height.
            let top = lines.iter().map(|s| s[1].max(s[3])).fold(0.0, f64::max);
            let foot = lines.iter().map(|s| s[1].min(s[3])).fold(0.0, f64::min);
            (1000, top, foot, 0.0)
        }
    };
    let units = u16::try_from(units)
        .ok()
        .filter(|u| (16..=16384).contains(u))
        .ok_or(format!("{units} units an em: a font has 16 to 16384"))?;
    let scale = f64::from(units);
    let unit = |v: f64| (v * scale).round().clamp(-32768.0, 32767.0) as i16;
    let mut face = Face {
        family: family.to_owned(),
        units,
        ascent: unit(ascent),
        descent: unit(descent),
        gap: unit(gap),
        glyphs: vec![Glyph::default()],
        map: Vec::new(),
    };
    let mut shared: HashMap<(u32, u32, u32), u16> = HashMap::new();
    let mut missing = false;
    for row in rows.chunks_exact(8) {
        let code = row[0] as i32;
        let advance = u16::try_from((row[1] as i32).max(0)).unwrap_or(u16::MAX);
        let (first, count) = (row[2] as usize, row[3] as usize);
        let outline = || -> Result<Glyph, String> {
            let run = lines.get(first..first + count).ok_or(format!(
                "the glyph of code {code} reads segments {first} to {}, past the {} there are",
                first + count,
                lines.len()
            ))?;
            Ok(Glyph {
                advance,
                contours: contours(run, scale),
            })
        };
        if code < 0 {
            if !missing {
                face.glyphs[0] = outline()?;
                missing = true;
            }
            continue;
        }
        let Some(code) = u32::try_from(code)
            .ok()
            .filter(|&c| char::from_u32(c).is_some())
        else {
            continue;
        };
        if wanted.as_ref().is_some_and(|w| !w.contains(&code)) {
            continue;
        }
        let key = (row[1], row[2], row[3]);
        let id = match shared.get(&key) {
            Some(&id) => id,
            None => {
                let id = u16::try_from(face.glyphs.len())
                    .map_err(|_| "a font holds at most 65535 glyphs".to_owned())?;
                face.glyphs.push(outline()?);
                shared.insert(key, id);
                id
            }
        };
        face.map.push((code, id));
    }
    face.map.sort_unstable_by_key(|&(code, _)| code);
    face.map.dedup_by_key(|&mut (code, _)| code);
    if face.map.is_empty() {
        return Err("no glyph row maps a character asked for".into());
    }
    if !missing {
        face.glyphs[0].advance = units / 2;
    }
    Ok(face)
}

/// A run of segments as contours in font units: a contour ends where a
/// segment comes back to its start, or where the next segment does not
/// begin at the last one's end.
fn contours(run: &[[f64; 4]], scale: f64) -> Vec<Vec<Point>> {
    let mut out = Vec::new();
    let mut ring: Vec<(f64, f64)> = Vec::new();
    let mut finish = |ring: &mut Vec<(f64, f64)>| {
        if let Some(points) = written(ring, scale) {
            out.push(points);
        }
        ring.clear();
    };
    for s in run {
        let (a, b) = ((s[0], s[1]), (s[2], s[3]));
        if ring.last() != Some(&a) {
            finish(&mut ring);
            ring.push(a);
        }
        if b == ring[0] {
            finish(&mut ring);
        } else {
            ring.push(b);
        }
    }
    finish(&mut ring);
    out
}

/// A closed ring of points in ems as a contour in font units, through its
/// points taken in whole units: none when it encloses no area.
fn written(ring: &[(f64, f64)], scale: f64) -> Option<Vec<Point>> {
    let mut points: Vec<(i32, i32)> = Vec::with_capacity(ring.len());
    for &(x, y) in ring {
        let p = ((x * scale).round() as i32, (y * scale).round() as i32);
        if points.last() != Some(&p) {
            points.push(p);
        }
    }
    while points.len() > 1 && points.first() == points.last() {
        points.pop();
    }
    let n = points.len();
    let twice_area: i64 = (0..n)
        .map(|k| {
            let (a, b) = (points[k], points[(k + 1) % n]);
            i64::from(a.0) * i64::from(b.1) - i64::from(b.0) * i64::from(a.1)
        })
        .sum();
    if n < 3 || twice_area.abs() <= 2 {
        return None;
    }
    let clamp = |v: i32| v.clamp(-32768, 32767) as i16;
    Some(
        curves::curved(&points)
            .into_iter()
            .map(|(x, y, on)| (clamp(x), clamp(y), on))
            .collect(),
    )
}

// ---------------------------------------------------------------- tables

fn be16(v: u16) -> [u8; 2] {
    v.to_be_bytes()
}

fn bi16(v: i16) -> [u8; 2] {
    v.to_be_bytes()
}

fn be32(v: u32) -> [u8; 4] {
    v.to_be_bytes()
}

/// A glyph's outline as a `glyf` entry: no contours, no bytes.
fn glyf_entry(glyph: &Glyph) -> Vec<u8> {
    let Some([x0, y0, x1, y1]) = glyph.bounds() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    out.extend(bi16(glyph.contours.len() as i16));
    for v in [x0, y0, x1, y1] {
        out.extend(bi16(v));
    }
    let mut end = 0usize;
    for contour in &glyph.contours {
        end += contour.len();
        out.extend(be16((end - 1) as u16));
    }
    out.extend(be16(0)); // no instructions
    // A move that fits a byte is a byte (its sign in the flag), none a
    // repeat of the coordinate; equal flags in a row are one and a count.
    let mut flags: Vec<u8> = Vec::new();
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    let (mut px, mut py) = (0i32, 0i32);
    for &(x, y, on) in glyph.contours.iter().flatten() {
        let mut flag = u8::from(on);
        for (d, short, same, data) in [
            (i32::from(x) - px, 0x02u8, 0x10u8, &mut xs),
            (i32::from(y) - py, 0x04, 0x20, &mut ys),
        ] {
            if d == 0 {
                flag |= same;
            } else if (-255..=255).contains(&d) {
                flag |= short | if d > 0 { same } else { 0 };
                data.push(d.unsigned_abs() as u8);
            } else {
                data.extend((d as i16).to_be_bytes());
            }
        }
        flags.push(flag);
        (px, py) = (i32::from(x), i32::from(y));
    }
    let mut k = 0;
    while k < flags.len() {
        let mut run = 1;
        while k + run < flags.len() && flags[k + run] == flags[k] && run < 256 {
            run += 1;
        }
        if run > 1 {
            out.push(flags[k] | 0x08);
            out.push((run - 1) as u8);
        } else {
            out.push(flags[k]);
        }
        k += run;
    }
    out.extend(xs);
    out.extend(ys);
    out
}

/// `cmap`: format 4 for the characters of the Basic Multilingual Plane
/// (when it fits the format's 64 KiB) and format 12 for every one.
fn cmap(map: &[(u32, u16)]) -> Vec<u8> {
    let mut groups: Vec<(u32, u32, u32)> = Vec::new();
    for &(code, id) in map {
        match groups.last_mut() {
            Some(g) if g.1 + 1 == code && g.2 + (code - g.0) == u32::from(id) => g.1 = code,
            _ => groups.push((code, code, u32::from(id))),
        }
    }
    let mut twelve = Vec::new();
    twelve.extend(be16(12));
    twelve.extend(be16(0));
    twelve.extend(be32(16 + 12 * groups.len() as u32));
    twelve.extend(be32(0));
    twelve.extend(be32(groups.len() as u32));
    for (start, end, id) in &groups {
        twelve.extend(be32(*start));
        twelve.extend(be32(*end));
        twelve.extend(be32(*id));
    }
    let bmp: Vec<(u32, u16)> = map.iter().copied().filter(|&(c, _)| c < 0xFFFF).collect();
    let four = format4(&bmp);
    let mut subtables: Vec<(u16, u16, Vec<u8>)> = Vec::new();
    if let Some(four) = four {
        subtables.push((3, 1, four));
    }
    subtables.push((3, 10, twelve));
    let mut out = Vec::new();
    out.extend(be16(0));
    out.extend(be16(subtables.len() as u16));
    let mut offset = 4 + 8 * subtables.len() as u32;
    for (platform, encoding, table) in &subtables {
        out.extend(be16(*platform));
        out.extend(be16(*encoding));
        out.extend(be32(offset));
        offset += table.len() as u32;
    }
    for (_, _, table) in subtables {
        out.extend(table);
    }
    out
}

/// A format 4 subtable: runs of consecutive codes as segments, a segment
/// whose glyphs are consecutive by its delta and the rest by an array;
/// the runs nearest each other merged (the codes between mapping to
/// glyph 0) until the subtable fits. None if it cannot.
fn format4(map: &[(u32, u16)]) -> Option<Vec<u8>> {
    let mut runs: Vec<(u32, Vec<u16>)> = Vec::new();
    for &(code, id) in map {
        match runs.last_mut() {
            Some((start, ids)) if *start + ids.len() as u32 == code => ids.push(id),
            _ => runs.push((code, vec![id])),
        }
    }
    let delta = |ids: &[u16]| ids.windows(2).all(|w| w[1] == w[0].wrapping_add(1));
    let size = |segments: &[(u32, Vec<u16>)]| {
        16 + 8 * (segments.len() + 1)
            + segments
                .iter()
                .filter(|(_, ids)| !delta(ids))
                .map(|(_, ids)| 2 * ids.len())
                .sum::<usize>()
    };
    let merged = |within: u32| {
        let mut out: Vec<(u32, Vec<u16>)> = Vec::new();
        for (start, ids) in &runs {
            match out.last_mut() {
                Some((s, have)) if start - (*s + have.len() as u32) <= within => {
                    let gap = start - (*s + have.len() as u32);
                    have.extend(std::iter::repeat_n(0, gap as usize));
                    have.extend(ids);
                }
                _ => out.push((*start, ids.clone())),
            }
        }
        out
    };
    let mut gaps: Vec<u32> = runs
        .windows(2)
        .map(|w| w[1].0 - (w[0].0 + w[0].1.len() as u32))
        .collect();
    gaps.sort_unstable();
    gaps.dedup();
    let mut segments = merged(0);
    let mut tries = gaps.into_iter();
    while size(&segments) > 0xFFFF {
        segments = merged(tries.next()?);
    }
    let count = segments.len() + 1;
    let mut ends = Vec::new();
    let mut starts = Vec::new();
    let mut deltas = Vec::new();
    let mut offsets = Vec::new();
    let mut array: Vec<u16> = Vec::new();
    for (k, (start, ids)) in segments.iter().enumerate() {
        starts.push(*start as u16);
        ends.push((*start + ids.len() as u32 - 1) as u16);
        if delta(ids) {
            deltas.push(ids[0].wrapping_sub(*start as u16));
            offsets.push(0u16);
        } else {
            deltas.push(0);
            offsets.push((2 * (count - k + array.len())) as u16);
            array.extend(ids);
        }
    }
    starts.push(0xFFFF);
    ends.push(0xFFFF);
    deltas.push(1);
    offsets.push(0);
    let length = 16 + 8 * count + 2 * array.len();
    let search = 2 * (1usize << count.ilog2());
    let mut out = Vec::new();
    for v in [
        4,
        length as u16,
        0,
        (2 * count) as u16,
        search as u16,
        count.ilog2() as u16,
        (2 * count - search) as u16,
    ] {
        out.extend(be16(v));
    }
    out.extend(ends.iter().flat_map(|v| be16(*v)));
    out.extend(be16(0));
    out.extend(starts.iter().flat_map(|v| be16(*v)));
    out.extend(deltas.iter().flat_map(|v| be16(*v)));
    out.extend(offsets.iter().flat_map(|v| be16(*v)));
    out.extend(array.iter().flat_map(|v| be16(*v)));
    Some(out)
}

/// `name`: the family, its one style (Regular), the full and PostScript
/// names, in UTF-16 for Windows' Unicode platform.
/// A family's name as PostScript takes one: printable ASCII, no spaces
/// nor delimiters.
fn postscript_name(family: &str) -> String {
    let name: String = family
        .chars()
        .filter(|c| c.is_ascii_graphic() && !"[](){}<>/%".contains(*c))
        .collect();
    if name.is_empty() {
        "Font".to_owned()
    } else {
        name
    }
}

fn name(family: &str) -> Vec<u8> {
    let postscript = postscript_name(family);
    let full = family.to_owned();
    let records: [(u16, &str); 5] = [
        (1, family),
        (2, "Regular"),
        (3, &full),
        (4, &full),
        (6, &postscript),
    ];
    let strings: Vec<Vec<u8>> = records
        .iter()
        .map(|(_, s)| s.encode_utf16().flat_map(u16::to_be_bytes).collect())
        .collect();
    let mut out = Vec::new();
    out.extend(be16(0));
    out.extend(be16(records.len() as u16));
    out.extend(be16(6 + 12 * records.len() as u16));
    let mut offset = 0u16;
    for ((id, _), bytes) in records.iter().zip(&strings) {
        for v in [3, 1, 0x409, *id, bytes.len() as u16, offset] {
            out.extend(be16(v));
        }
        offset += bytes.len() as u16;
    }
    out.extend(strings.concat());
    out
}

/// The font's tables, by tag: TrueType outlines (`glyf`), or CFF ones.
fn tables(face: &Face, cff: bool) -> BTreeMap<[u8; 4], Vec<u8>> {
    let glyphs = &face.glyphs;
    let bounds: Vec<Option<[i16; 4]>> = glyphs.iter().map(Glyph::bounds).collect();
    let all = bounds
        .iter()
        .flatten()
        .fold([i16::MAX, i16::MAX, i16::MIN, i16::MIN], |b, g| {
            [
                b[0].min(g[0]),
                b[1].min(g[1]),
                b[2].max(g[2]),
                b[3].max(g[3]),
            ]
        });
    let all = if all[0] > all[2] { [0; 4] } else { all };
    let n = glyphs.len() as u16;
    let mut out = BTreeMap::new();

    let mut glyf = Vec::new();
    let mut loca = Vec::new();
    for glyph in glyphs {
        loca.extend(be32(glyf.len() as u32));
        glyf.extend(glyf_entry(glyph));
        while glyf.len() % 4 != 0 {
            glyf.push(0);
        }
    }
    loca.extend(be32(glyf.len() as u32));

    let mut head = Vec::new();
    head.extend(be32(0x0001_0000));
    head.extend(be32(0x0001_0000));
    head.extend(be32(0)); // the whole font's checksum adjustment, set last
    head.extend(be32(0x5F0F_3CF5));
    head.extend(be16(0b1011)); // baseline at 0, left side bearing at x = xMin, whole ppem
    head.extend(be16(face.units));
    head.extend([0; 16]); // created, modified
    for v in all {
        head.extend(bi16(v));
    }
    for v in [0u16, 8] {
        head.extend(be16(v)); // style, smallest readable size
    }
    for v in [2i16, 1, 0] {
        head.extend(bi16(v)); // direction hint, long offsets, glyph format
    }
    out.insert(*b"head", head);

    let advance_max = glyphs.iter().map(|g| g.advance).max().unwrap_or(0);
    let lsb = |k: usize| bounds[k].map_or(0, |b| b[0]);
    let rsb = |k: usize| bounds[k].map_or(0, |b| glyphs[k].advance as i16 - b[2]);
    let mut hhea = Vec::new();
    hhea.extend(be32(0x0001_0000));
    for v in [face.ascent, face.descent, face.gap] {
        hhea.extend(bi16(v));
    }
    hhea.extend(be16(advance_max));
    let drawn = (0..glyphs.len()).filter(|&k| bounds[k].is_some());
    hhea.extend(bi16(drawn.clone().map(lsb).min().unwrap_or(0)));
    hhea.extend(bi16(drawn.clone().map(rsb).min().unwrap_or(0)));
    hhea.extend(bi16(
        drawn
            .map(|k| bounds[k].map_or(0, |b| b[2]))
            .max()
            .unwrap_or(0),
    ));
    for v in [1i16, 0, 0, 0, 0, 0, 0, 0] {
        hhea.extend(bi16(v)); // caret rise, run, offset, reserved, metric format
    }
    hhea.extend(be16(n));
    out.insert(*b"hhea", hhea);

    let mut maxp = Vec::new();
    if cff {
        maxp.extend(be32(0x0000_5000));
        maxp.extend(be16(n));
    } else {
        maxp.extend(be32(0x0001_0000));
        maxp.extend(be16(n));
        maxp.extend(be16(
            glyphs.iter().map(Glyph::points).max().unwrap_or(0) as u16
        ));
        maxp.extend(be16(
            glyphs.iter().map(|g| g.contours.len()).max().unwrap_or(0) as u16,
        ));
        for v in [0u16, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0] {
            maxp.extend(be16(v));
        }
    }
    out.insert(*b"maxp", maxp);

    // Each glyph's advance and left side bearing, its leftmost x: a reader
    // that places a glyph by it (FreeType) keeps it where its points are.
    let mut hmtx = Vec::new();
    for (k, glyph) in glyphs.iter().enumerate() {
        hmtx.extend(be16(glyph.advance));
        hmtx.extend(bi16(lsb(k)));
    }
    out.insert(*b"hmtx", hmtx);

    let em = i32::from(face.units);
    let part = |share: i32| (em * share / 100) as i16;
    let advances: Vec<i32> = glyphs
        .iter()
        .filter(|g| g.advance > 0)
        .map(|g| i32::from(g.advance))
        .collect();
    let average = if advances.is_empty() {
        0
    } else {
        advances.iter().sum::<i32>() / advances.len() as i32
    };
    let first = face.map.first().map_or(0, |m| m.0.min(0xFFFF)) as u16;
    let last = face.map.last().map_or(0, |m| m.0.min(0xFFFF)) as u16;
    let mut os2 = Vec::new();
    os2.extend(be16(4));
    os2.extend(bi16(average as i16));
    for v in [400u16, 5, 0] {
        os2.extend(be16(v)); // weight, width, embedding allowed
    }
    for v in [
        part(65),
        part(70),
        0,
        part(14),
        part(65),
        part(70),
        0,
        part(48),
        part(5),
        part(25),
        0,
    ] {
        os2.extend(bi16(v)); // sub and superscripts, strikeout, family class
    }
    os2.extend([0; 10]); // panose
    os2.extend([0; 16]); // Unicode ranges
    os2.extend(*b"NONE");
    for v in [0x40u16, first, last] {
        os2.extend(be16(v)); // regular, first and last character
    }
    for v in [face.ascent, face.descent, face.gap] {
        os2.extend(bi16(v));
    }
    os2.extend(be16(face.ascent.max(all[3]).max(0) as u16));
    os2.extend(be16((-face.descent).max(-all[1]).max(0) as u16));
    os2.extend([0; 8]); // code pages
    for v in [part(50), part(70)] {
        os2.extend(bi16(v)); // x and cap heights
    }
    for v in [0u16, 32, 1] {
        os2.extend(be16(v)); // default and break characters, context
    }
    out.insert(*b"OS/2", os2);

    out.insert(*b"cmap", cmap(&face.map));
    if cff {
        let charstrings: Vec<Vec<u8>> = glyphs
            .iter()
            .map(|g| cff::charstring(&g.contours, g.advance, face.units))
            .collect();
        let postscript = postscript_name(&face.family);
        out.insert(
            *b"CFF ",
            cff::table(&charstrings, &face.family, &postscript, all, face.units),
        );
    } else {
        out.insert(*b"loca", loca);
        out.insert(*b"glyf", glyf);
    }
    out.insert(*b"name", name(&face.family));
    let mut post = Vec::new();
    post.extend(be32(0x0003_0000));
    post.extend(be32(0));
    post.extend(bi16(-part(10)));
    post.extend(bi16(part(5)));
    post.extend([0; 20]);
    out.insert(*b"post", post);
    out
}

fn checksum(bytes: &[u8]) -> u32 {
    bytes.chunks(4).fold(0u32, |sum, chunk| {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        sum.wrapping_add(u32::from_be_bytes(word))
    })
}

fn padded(bytes: &[u8]) -> Vec<u8> {
    let mut out = bytes.to_vec();
    out.resize(bytes.len().next_multiple_of(4), 0);
    out
}

/// The tables as one font file, `head`'s checksum adjustment set, and
/// where `head` begins in it.
fn sfnt(tables: &BTreeMap<[u8; 4], Vec<u8>>) -> (Vec<u8>, usize) {
    let n = tables.len() as u16;
    let search = 16 * (1u16 << n.ilog2());
    let mut out = Vec::new();
    out.extend(be32(flavor(tables)));
    for v in [n, search, n.ilog2() as u16, 16 * n - search] {
        out.extend(be16(v));
    }
    let mut offset = 12 + 16 * u32::from(n);
    let mut head_at = 0;
    for (tag, table) in tables {
        if tag == b"head" {
            head_at = offset as usize;
        }
        out.extend(tag);
        out.extend(be32(checksum(table)));
        out.extend(be32(offset));
        out.extend(be32(table.len() as u32));
        offset += table.len().next_multiple_of(4) as u32;
    }
    for table in tables.values() {
        out.extend(padded(table));
    }
    let adjustment = 0xB1B0_AFBAu32.wrapping_sub(checksum(&out));
    out[head_at + 8..head_at + 12].copy_from_slice(&be32(adjustment));
    (out, head_at)
}

/// The version a font file with `tables` begins with: `OTTO` for CFF
/// outlines.
fn flavor(tables: &BTreeMap<[u8; 4], Vec<u8>>) -> u32 {
    if tables.contains_key(b"CFF ") {
        0x4F54_544F
    } else {
        0x0001_0000
    }
}

/// The face as a TrueType file.
pub fn ttf(face: &Face) -> Vec<u8> {
    sfnt(&tables(face, false)).0
}

/// The face as an OpenType file of CFF outlines.
pub fn otf(face: &Face) -> Vec<u8> {
    sfnt(&tables(face, true)).0
}

/// The face as a WOFF file: the TrueType file's tables, each compressed
/// (zlib) where that makes it smaller, `head` with the TrueType file's
/// checksum adjustment.
pub fn woff(face: &Face) -> Vec<u8> {
    let tables = tables(face, false);
    let (whole, head_at) = sfnt(&tables);
    let n = tables.len();
    let mut directory = Vec::new();
    let mut data = Vec::new();
    let mut offset = 44 + 20 * n;
    let mut sfnt_size = 12 + 16 * n;
    for (tag, table) in &tables {
        // A table's checksum is the TrueType file's: head's taken with its
        // adjustment at zero, as there.
        let sum = checksum(table);
        let table = if tag == b"head" {
            whole[head_at..head_at + table.len()].to_vec()
        } else {
            table.clone()
        };
        let packed = entelechy_formats::deflate::zlib(&table);
        let stored = if packed.len() < table.len() {
            packed
        } else {
            table.clone()
        };
        directory.extend(tag);
        directory.extend(be32(offset as u32));
        directory.extend(be32(stored.len() as u32));
        directory.extend(be32(table.len() as u32));
        directory.extend(be32(sum));
        offset += stored.len().next_multiple_of(4);
        sfnt_size += table.len().next_multiple_of(4);
        data.extend(padded(&stored));
    }
    let mut out = Vec::new();
    out.extend(*b"wOFF");
    out.extend(be32(0x0001_0000));
    out.extend(be32((44 + directory.len() + data.len()) as u32));
    out.extend(be16(n as u16));
    out.extend(be16(0));
    out.extend(be32(sfnt_size as u32));
    out.extend(be16(1));
    out.extend(be16(0));
    out.extend([0; 20]); // no metadata, no private data
    out.extend(directory);
    out.extend(data);
    out
}
