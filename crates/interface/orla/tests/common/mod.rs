//! What a rig is, as rows: the one statement of it the tests hold
//! everything to. [`docs/RIGS.md`](../../../../docs/RIGS.md) says the
//! same in words.

use std::path::Path;

/// Every kind of row a rig comes in as: the file's suffix and its
/// layout, `i` an `Int` and `f` a `Float`, in the order the ABI lays
/// them out.
pub const RIG_ROWS: &[(&str, &str)] = &[
    // position, normal, uv — the mesh at rest, in the skin's space
    ("vertices.field", "f,f,f,f,f,f,f,f"),
    ("triangles.field", "i,i,i"),
    // four joints and how much of each, one row a vertex
    ("weights.field", "i,i,i,i,f,f,f,f"),
    // the parent, then the inverse bind matrix by columns
    ("skeleton.field", "i,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f"),
    // a local translation, rotation (x y z w) and scale, a frame's
    // joints together
    ("clip.field", "f,f,f,f,f,f,f,f,f,f"),
];

/// The layout a Field file was written with, from its header.
pub fn schema_of(path: &Path) -> String {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let text = String::from_utf8_lossy(&bytes[..bytes.len().min(256)]).to_string();
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("ORLAFIELD 1"), "{}", path.display());
    lines
        .next()
        .and_then(|line| line.split_whitespace().nth(4))
        .unwrap_or_else(|| panic!("{} has no layout", path.display()))
        .to_owned()
}

/// Holds every rig file in a directory to the contract, and says how
/// many it found of each kind.
pub fn rows_keep_the_contract(dir: &Path) -> Vec<(String, String)> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let Some((kind, layout)) = RIG_ROWS
            .iter()
            .find(|(suffix, _)| name.ends_with(suffix))
            .copied()
        else {
            continue;
        };
        assert_eq!(
            schema_of(&path),
            layout,
            "{name} is not laid out as a {kind}"
        );
        found.push((kind.to_owned(), name));
    }
    found
}
