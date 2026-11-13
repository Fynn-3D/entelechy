//! Compiled scene bundles: validated IR modules, lowered graph, initial
//! input Fields, device bindings and pre-baked font atlases in one compact
//! binary stream.
//!
//! An exported page carries a [`Bundle`] instead of `.ent`/`.esn` sources and
//! raw asset codecs, so playback runs over `entelechy-ir`, `entelechy-kernel`
//! and `entelechy-graph` directly without the compiler or document loader.

mod decode;
mod encode;

use std::collections::{BTreeMap, HashMap};

use entelechy_abi::{Domain, Value};
use entelechy_graph::{Graph, Source};
use entelechy_ir::{BinaryOp, MAX_TYPE_DEPTH, Module};
use entelechy_kernel::FieldData;

pub(super) const MAGIC: &[u8; 16] = b"ENTELECHYBUNDLE\0";
pub(super) const VERSION: u32 = 1;
pub(super) const MAX_ITEMS: usize = 1 << 20;
pub(super) const MAX_NESTING: usize = MAX_TYPE_DEPTH + 8;

/// A compiled scene ready to open and play without source documents.
#[derive(Clone, Debug, PartialEq)]
pub struct Bundle {
    /// Compiled Kernels the graph's nodes and state migrations run.
    pub modules: Vec<Module>,
    /// The lowered graph, source locations cleared.
    pub graph: Graph,
    /// Graph parameter overrides active when bundled.
    pub params: BTreeMap<String, Value>,
    /// Static external Fields (assets and bound images), by external name.
    pub inputs: Vec<(String, FieldData)>,
    /// Scene outputs sent `to` a Host device: `(output, device)`.
    pub presented: Vec<(String, String)>,
    /// Scene inputs fed `=` by a Host device: `(input, device)`.
    pub feeds: Vec<(String, String)>,
    /// Pre-baked font atlases for each `font` node in the scene.
    pub fonts: Vec<BundledFont>,
}

/// How a `font` node's atlas is handed to [`crate::Session::bundle_with_fonts`].
#[derive(Clone, Debug, PartialEq)]
pub enum BundledAtlas {
    /// Unflattened contours (`entelechy_import::font::read_compact`), encoded
    /// compactly in the bundle and flattened on open.
    Compact(entelechy_formats::font::CompactFont),
    /// Pre-flattened `(glyphs, segments, metrics)` Fields.
    Fields {
        /// Glyph rows.
        glyphs: FieldData,
        /// Flattened segments.
        segments: FieldData,
        /// Font metrics.
        metrics: FieldData,
    },
}

/// One `font` node's pre-baked glyph table, curve segments and metrics.
#[derive(Clone, Debug, PartialEq)]
pub struct BundledFont {
    /// Name of the `font` node, for diagnostics.
    pub node: String,
    /// Source of the `Rows<Int>` character codes read after each frame.
    pub codes: Source,
    /// External names fed with `[glyphs, segments, metrics]`.
    pub outputs: [String; 3],
    /// Atlas glyph rows (`[code, advance, first, count, lo_x, lo_y, hi_x, hi_y]`),
    /// row 0 holding `.notdef` (`code == -1`) and rows `1..` sorted ascending by `code`.
    pub glyphs: FieldData,
    /// Atlas line segments (`Vec4` per segment).
    pub segments: FieldData,
    /// Font metrics (`[units, ascender, descender, line_gap]`).
    pub metrics: FieldData,
    /// Unflattened contours when available, so `Bundle::encode` carries curves
    /// instead of flattened segment `f32`s.
    pub compact: Option<entelechy_formats::font::CompactFont>,
}

/// Slices the glyph rows and curve segments for `codes` out of a pre-baked
/// [`BundledFont`] atlas, matching `entelechy_import::font::glyphs_of` word
/// for word without parsing the font file at run time.
pub fn glyphs_from_atlas(
    atlas: &BundledFont,
    codes: &[i32],
) -> Result<(FieldData, FieldData, FieldData), String> {
    let gw = atlas.glyphs.storage();
    if gw.len() < 8 || !gw.len().is_multiple_of(8) {
        return Err(format!("font `{}`: malformed glyph atlas", atlas.node));
    }
    let sw = atlas.segments.storage();
    if !sw.len().is_multiple_of(4) {
        return Err(format!("font `{}`: malformed segment atlas", atlas.node));
    }
    let sorted_rows: Vec<&[u32]> = gw[8..].chunks_exact(8).collect();
    let missing_advance = gw[1];
    let missing = [-1i32 as u32, missing_advance, 0, 0, 0, 0, 0, 0];

    let mut glyphs = Vec::with_capacity(codes.len().max(1) * 8);
    let mut segments: Vec<u32> = Vec::new();
    let mut drawn: HashMap<(u32, u32), u32> = HashMap::new();
    let mut next_seg = 0u32;

    if codes.is_empty() {
        glyphs.extend_from_slice(&missing);
    }
    for &code in codes {
        if code <= 0 {
            glyphs.extend_from_slice(&missing);
            continue;
        }
        let Ok(found) = sorted_rows.binary_search_by_key(&code, |row| row[0] as i32) else {
            glyphs.extend_from_slice(&missing);
            continue;
        };
        let row = sorted_rows[found];
        let (src_first, count) = (row[2], row[3]);
        let first = if let Some(&existing) = drawn.get(&(src_first, count)) {
            existing
        } else {
            let dst = next_seg;
            if count > 0 {
                let start = (src_first as usize).saturating_mul(4);
                let end = start.saturating_add((count as usize).saturating_mul(4));
                let slice = sw
                    .get(start..end)
                    .ok_or_else(|| format!("font `{}`: segment range out of bounds", atlas.node))?;
                segments.extend_from_slice(slice);
                next_seg = next_seg.saturating_add(count);
            }
            drawn.insert((src_first, count), dst);
            dst
        };
        glyphs.extend_from_slice(&[
            code as u32,
            row[1],
            first,
            count,
            row[4],
            row[5],
            row[6],
            row[7],
        ]);
    }
    let seg_rows = next_seg.max(1);
    if segments.is_empty() {
        segments.resize(4, 0);
    }
    let glyph_rows = (glyphs.len() / 8) as u32;
    let glyphs_field = FieldData::from_storage(
        atlas.glyphs.schema().clone(),
        Domain::Rows(glyph_rows),
        glyphs,
    )
    .map_err(|e| e.to_string())?;
    let segments_field = FieldData::from_storage(
        atlas.segments.schema().clone(),
        Domain::Rows(seg_rows),
        segments,
    )
    .map_err(|e| e.to_string())?;
    Ok((glyphs_field, segments_field, atlas.metrics.clone()))
}

pub(super) const BINARIES: [BinaryOp; 19] = [
    BinaryOp::Add,
    BinaryOp::Sub,
    BinaryOp::Mul,
    BinaryOp::Div,
    BinaryOp::Rem,
    BinaryOp::Less,
    BinaryOp::LessEqual,
    BinaryOp::Greater,
    BinaryOp::GreaterEqual,
    BinaryOp::Equal,
    BinaryOp::NotEqual,
    BinaryOp::And,
    BinaryOp::Or,
    BinaryOp::BitAnd,
    BinaryOp::BitOr,
    BinaryOp::BitXor,
    BinaryOp::Shl,
    BinaryOp::Shr,
    BinaryOp::Ushr,
];
