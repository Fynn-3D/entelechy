//! A font written from its Fields and read back by the importer: the
//! characters, advances, outlines and measures come back.

use entelechy_export::font;
use entelechy_host::{Domain, FieldData};
use entelechy_import::font::{Fields, glyph_schema, metrics_schema, read, sample, segment_schema};

fn field(schema: entelechy_host::Schema, rows: usize, words: Vec<u32>) -> FieldData {
    FieldData::from_storage(schema, Domain::Rows(rows as u32), words).unwrap()
}

fn all(fields: Fields) -> Vec<FieldData> {
    vec![fields.glyphs, fields.segments, fields.metrics]
}

/// A glyph row's code, advance and box.
type Row = (i32, i32, [f32; 4]);

/// Each glyph row's code, advance and box, and the measures.
fn summary(fields: &Fields) -> (Vec<Row>, [f32; 3]) {
    let rows = fields
        .glyphs
        .values()
        .chunks_exact(8)
        .map(|r| {
            let f = |k: usize| f32::from_bits(r[k]);
            (r[0] as i32, r[1] as i32, [f(4), f(5), f(6), f(7)])
        })
        .collect();
    let m = fields.metrics.values();
    let f = |k: usize| f32::from_bits(m[k]);
    (rows, [f(1), f(2), f(3)])
}

#[test]
fn a_font_comes_back_as_it_was_read() {
    let (fields, _) = read(&sample::latin(), None).unwrap();
    let (before, measures) = summary(&fields);
    let face = font::face(&all(fields), "Sample", None).unwrap();
    let (again, _) = read(&font::ttf(&face), None).unwrap();
    let (after, measures_again) = summary(&again);
    assert_eq!(before.len(), after.len());
    // Curves come back as cut again, within a unit or so of a thousand.
    for (b, a) in before.iter().zip(&after) {
        assert_eq!((b.0, b.1), (a.0, a.1));
        for k in 0..4 {
            assert!(
                (b.2[k] - a.2[k]).abs() <= 0.002,
                "code {}: {:?} against {:?}",
                b.0,
                b.2,
                a.2
            );
        }
    }
    for k in 0..3 {
        assert!((measures[k] - measures_again[k]).abs() <= 0.001);
    }
}

#[test]
fn a_straight_outline_comes_back_point_for_point() {
    // `A`: a box of five segments (one edge cut in two), a thousand units an em.
    let points = [
        (0.1f32, 0.1f32),
        (0.35, 0.1),
        (0.6, 0.1),
        (0.6, 0.7),
        (0.1, 0.7),
    ];
    let mut segments = Vec::new();
    for k in 0..points.len() {
        let (a, b) = (points[k], points[(k + 1) % points.len()]);
        segments.extend([a.0, a.1, b.0, b.1].map(f32::to_bits));
    }
    let row = [
        65u32,
        700,
        0,
        5,
        0.1f32.to_bits(),
        0.1f32.to_bits(),
        0.6f32.to_bits(),
        0.7f32.to_bits(),
    ];
    let metrics = [
        1000u32,
        0.8f32.to_bits(),
        (-0.2f32).to_bits(),
        0f32.to_bits(),
    ];
    let inputs = vec![
        field(glyph_schema(), 1, row.to_vec()),
        field(segment_schema(), 5, segments),
        field(metrics_schema(), 1, metrics.to_vec()),
    ];
    let face = font::face(&inputs, "Box", None).unwrap();
    let (back, _) = read(&font::ttf(&face), Some("A")).unwrap();
    let rows = back.glyphs.values();
    let a = &rows[8..16];
    assert_eq!(
        (a[0], a[1], a[3]),
        (65, 700, 4),
        "the point along the bottom edge is left out"
    );
    let segments: Vec<f32> = back
        .segments
        .values()
        .iter()
        .map(|w| f32::from_bits(*w))
        .collect();
    let corners: Vec<(f32, f32)> = segments[a[2] as usize * 4..][..16]
        .chunks_exact(4)
        .map(|s| (s[0], s[1]))
        .collect();
    for corner in [(0.1, 0.1), (0.6, 0.1), (0.6, 0.7), (0.1, 0.7)] {
        assert!(
            corners
                .iter()
                .any(|c| (c.0 - corner.0).abs() < 1e-6 && (c.1 - corner.1).abs() < 1e-6),
            "{corner:?} among {corners:?}"
        );
    }
}

/// A font file's tables, by tag.
fn tables(sfnt: &[u8]) -> Vec<([u8; 4], Vec<u8>)> {
    let n = u16::from_be_bytes([sfnt[4], sfnt[5]]) as usize;
    (0..n)
        .map(|k| {
            let r = &sfnt[12 + 16 * k..][..16];
            let word = |at: usize| u32::from_be_bytes(r[at..at + 4].try_into().unwrap()) as usize;
            (
                r[..4].try_into().unwrap(),
                sfnt[word(8)..][..word(12)].to_vec(),
            )
        })
        .collect()
}

#[test]
fn a_woff_holds_the_truetype_files_tables() {
    let (fields, _) = read(&sample::latin(), None).unwrap();
    let face = font::face(&all(fields), "Sample", None).unwrap();
    let ttf = font::ttf(&face);
    let woff = font::woff(&face);
    assert_eq!(&woff[..4], b"wOFF");
    assert_eq!(
        u32::from_be_bytes(woff[8..12].try_into().unwrap()) as usize,
        woff.len()
    );
    let n = u16::from_be_bytes([woff[12], woff[13]]) as usize;
    let expected = tables(&ttf);
    assert_eq!(n, expected.len());
    for (k, (tag, table)) in expected.iter().enumerate() {
        let r = &woff[44 + 20 * k..][..20];
        let word = |at: usize| u32::from_be_bytes(r[at..at + 4].try_into().unwrap()) as usize;
        assert_eq!(&r[..4], tag);
        let stored = &woff[word(4)..][..word(8)];
        let unpacked = if word(8) < word(12) {
            entelechy_formats::inflate::unzlib(stored, word(12)).unwrap()
        } else {
            stored.to_vec()
        };
        assert_eq!(&unpacked, table, "{}", String::from_utf8_lossy(tag));
    }
}

#[test]
fn characters_far_apart_map_to_their_glyphs() {
    // Nine thousand characters a code or two apart: as many runs as
    // characters, more than a format 4 map holds one segment each.
    let n = 9000u32;
    let mut rows = Vec::new();
    for k in 0..n {
        rows.extend([0x4E00 + 3 * k, 100 + k, 0, 0, 0, 0, 0, 0]);
    }
    let inputs = vec![
        field(glyph_schema(), n as usize, rows),
        field(segment_schema(), 1, vec![0; 4]),
    ];
    let face = font::face(&inputs, "Far", None).unwrap();
    assert_eq!(
        (face.characters(), face.glyphs()),
        (n as usize, n as usize + 1)
    );
    let text: String = (0..n)
        .map(|k| char::from_u32(0x4E00 + 3 * k).unwrap())
        .collect();
    let (back, imported) = read(&font::ttf(&face), Some(&text)).unwrap();
    assert!(
        imported.missing.is_empty(),
        "{} missing",
        imported.missing.len()
    );
    let rows = back.glyphs.values();
    for k in 0..n as usize {
        assert_eq!(
            rows[8 * (k + 1) + 1],
            100 + k as u32,
            "the advance of character {k}"
        );
    }
}

#[test]
fn cff_outlines_come_back_as_truetype_ones_do() {
    let (fields, _) = read(&sample::latin(), None).unwrap();
    let face = font::face(&all(fields), "Sample", None).unwrap();
    let (from_ttf, _) = read(&font::ttf(&face), None).unwrap();
    let (from_otf, _) = read(&font::otf(&face), None).unwrap();
    let (ttf, measures) = summary(&from_ttf);
    let (otf, measures_otf) = summary(&from_otf);
    assert_eq!(measures, measures_otf);
    assert_eq!(ttf.len(), otf.len());
    // A quadratic as a cubic, its controls in whole units: within a unit.
    for (t, o) in ttf.iter().zip(&otf) {
        assert_eq!((t.0, t.1), (o.0, o.1));
        for k in 0..4 {
            assert!(
                (t.2[k] - o.2[k]).abs() <= 0.002,
                "code {}: {:?} against {:?}",
                t.0,
                t.2,
                o.2
            );
        }
    }
}

#[test]
fn a_font_of_some_characters_holds_those_alone() {
    let (fields, _) = read(&sample::latin(), None).unwrap();
    let face = font::face(&all(fields), "Sample", Some("lo")).unwrap();
    assert_eq!((face.characters(), face.glyphs()), (2, 3));
    let (back, _) = read(&font::ttf(&face), None).unwrap();
    let codes: Vec<i32> = back
        .glyphs
        .values()
        .chunks_exact(8)
        .map(|r| r[0] as i32)
        .collect();
    assert_eq!(codes, vec![-1, 'l' as i32, 'o' as i32]);
}
