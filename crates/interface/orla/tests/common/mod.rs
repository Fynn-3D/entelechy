//! A rig as rows: the one statement of it the tests hold everything to.
//! [`docs/RIGS.md`](../../../../docs/RIGS.md) says the same in words.

use std::path::Path;

/// Every kind of rig row: the file's suffix and its layout, `i` an `Int`
/// and `f` a `Float`, in ABI order.
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

/// Holds every rig file in a directory to the contract, returning the
/// count of each kind.
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

/// Writes rows of `layout` as a `.field` file: the header, then each
/// value as a little-endian word, an `i` as an `Int` and an `f` as a
/// `Float`.
pub fn rows(path: &Path, layout: &str, values: &[f32]) {
    let scalars: Vec<&str> = layout.split(',').collect();
    let mut bytes = format!(
        "ORLAFIELD 1\nRows {} 1 1 {layout}\n",
        values.len() / scalars.len()
    )
    .into_bytes();
    for (scalar, value) in scalars.iter().cycle().zip(values) {
        let word = if *scalar == "i" {
            *value as i32 as u32
        } else {
            value.to_bits()
        };
        bytes.extend(word.to_le_bytes());
    }
    std::fs::write(path, bytes).unwrap();
}

/// A two-joint arm as the five kinds of rig row, in `dir` as `arm.*`: a
/// square tube two units tall, four sides of nine rings of two vertices,
/// the lower half bound to the root, the upper to the joint at its
/// middle, the band between blended; and two clips of 33 frames, a swing
/// of the root and a curl of the joint.
pub fn arm(dir: &Path) {
    let mut vertices = Vec::new();
    let mut weights = Vec::new();
    let mut triangles = Vec::new();
    let rings = 9;
    let corners = [(-0.2f32, -0.2f32), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2)];
    for side in 0..4 {
        let (a, b) = (corners[side], corners[(side + 1) % 4]);
        // Outward, one of the axes: the side's middle over its distance.
        let normal = [(a.0 + b.0) / 0.4, 0.0, (a.1 + b.1) / 0.4];
        for ring in 0..rings {
            let y = ring as f32 * 0.25;
            let t = ((y - 0.75) / 0.5).clamp(0.0, 1.0);
            for (u, (x, z)) in [a, b].into_iter().enumerate() {
                vertices.extend([x, y, z]);
                vertices.extend(normal);
                vertices.extend([u as f32, y / 2.0]);
                weights.extend([0.0, 1.0, 0.0, 0.0, 1.0 - t, t, 0.0, 0.0]);
            }
            if ring + 1 < rings {
                let at = ((side * rings + ring) * 2) as f32;
                triangles.extend([at, at + 1.0, at + 3.0, at, at + 3.0, at + 2.0]);
            }
        }
    }
    let moved = |y: f32| {
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, y, 0.0, 1.0,
        ]
    };
    let mut skeleton = vec![-1.0];
    skeleton.extend(moved(0.0));
    skeleton.push(0.0);
    skeleton.extend(moved(-1.0));
    let clip = |joint: usize| {
        let mut rows = Vec::new();
        for frame in 0..33 {
            for this in 0..2 {
                let angle = if this == joint {
                    0.6 * (frame as f32 / 32.0 * std::f32::consts::TAU).sin()
                } else {
                    0.0
                };
                rows.extend([0.0, this as f32, 0.0]);
                rows.extend([0.0, 0.0, (angle / 2.0).sin(), (angle / 2.0).cos()]);
                rows.extend([1.0, 1.0, 1.0]);
            }
        }
        rows
    };
    let path = |name: &str| dir.join(format!("arm.{name}.field"));
    rows(&path("vertices"), "f,f,f,f,f,f,f,f", &vertices);
    rows(&path("triangles"), "i,i,i", &triangles);
    rows(&path("weights"), "i,i,i,i,f,f,f,f", &weights);
    rows(
        &path("skeleton"),
        "i,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f",
        &skeleton,
    );
    let pose = "f,f,f,f,f,f,f,f,f,f";
    rows(&path("swing.clip"), pose, &clip(0));
    rows(&path("curl.clip"), pose, &clip(1));
}
