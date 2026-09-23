//! A mesh as GLB: the vertex rows `{position, normal, uv}` and the
//! triangle rows the runtime uses, and a PNG for the base colour when
//! there is one, become one glTF 2.0 binary with one mesh, one node and
//! one scene, which Blender, three.js and the rest open as they are.
//!
//! With the rest of [a rig's rows](../../../docs/RIGS.md) — what every
//! vertex hangs off, the joints, and a clip or several — it writes a
//! skin and an animation apiece instead, so a rig that came in through
//! `orla import` or the Blender bridge goes back out the way it came.

use std::path::Path;

use orla_runtime::{FieldData, Scalar};
use serde_json::{Value as Json, json};

/// What makes a mesh a rig: the rows beside it.
pub struct Rig<'a> {
    /// One row a vertex: four joints, then how much of each.
    pub weights: &'a FieldData,
    /// One row a joint: its parent, then its inverse bind matrix.
    pub skeleton: &'a FieldData,
    /// Each clip by name, its rows, and the rate it was sampled at.
    pub clips: Vec<(String, &'a FieldData, f32)>,
}

/// The inverse of an affine matrix, by columns as the ABI lays one out.
fn affine_inverse(m: &[f32; 16]) -> [f32; 16] {
    let a = [[m[0], m[4], m[8]], [m[1], m[5], m[9]], [m[2], m[6], m[10]]];
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let scale = if det.abs() > 0.0 { 1.0 / det } else { 0.0 };
    // The adjugate over the determinant, by rows.
    let inverse = [
        [
            (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * scale,
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * scale,
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * scale,
        ],
        [
            (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * scale,
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * scale,
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * scale,
        ],
        [
            (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * scale,
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * scale,
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * scale,
        ],
    ];
    let t = [m[12], m[13], m[14]];
    let moved = [
        -(inverse[0][0] * t[0] + inverse[0][1] * t[1] + inverse[0][2] * t[2]),
        -(inverse[1][0] * t[0] + inverse[1][1] * t[1] + inverse[1][2] * t[2]),
        -(inverse[2][0] * t[0] + inverse[2][1] * t[1] + inverse[2][2] * t[2]),
    ];
    [
        inverse[0][0],
        inverse[1][0],
        inverse[2][0],
        0.0,
        inverse[0][1],
        inverse[1][1],
        inverse[2][1],
        0.0,
        inverse[0][2],
        inverse[1][2],
        inverse[2][2],
        0.0,
        moved[0],
        moved[1],
        moved[2],
        1.0,
    ]
}

/// `a * b`, both by columns.
fn multiply(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for column in 0..4 {
        for row in 0..4 {
            out[column * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[column * 4 + k]).sum();
        }
    }
    out
}

/// An affine matrix as the translation, rotation and scale a glTF node
/// carries.
fn decompose(m: &[f32; 16]) -> ([f32; 3], [f32; 4], [f32; 3]) {
    let column = |c: usize| [m[c * 4], m[c * 4 + 1], m[c * 4 + 2]];
    let length = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let mut scale = [length(column(0)), length(column(1)), length(column(2))];
    // A mirrored matrix has a negative determinant; one axis carries it.
    let det = {
        let (x, y, z) = (column(0), column(1), column(2));
        x[0] * (y[1] * z[2] - y[2] * z[1]) - x[1] * (y[0] * z[2] - y[2] * z[0])
            + x[2] * (y[0] * z[1] - y[1] * z[0])
    };
    if det < 0.0 {
        scale[0] = -scale[0];
    }
    let axis = |c: usize| {
        let v = column(c);
        let s = if scale[c].abs() > 0.0 { scale[c] } else { 1.0 };
        [v[0] / s, v[1] / s, v[2] / s]
    };
    let (x, y, z) = (axis(0), axis(1), axis(2));
    // The rotation as a quaternion, by the largest of the four terms so
    // no square root is taken of something near zero.
    let trace = x[0] + y[1] + z[2];
    let turn = if trace > 0.0 {
        let r = (1.0 + trace).sqrt();
        let s = 0.5 / r;
        [
            (y[2] - z[1]) * s,
            (z[0] - x[2]) * s,
            (x[1] - y[0]) * s,
            0.5 * r,
        ]
    } else if x[0] > y[1] && x[0] > z[2] {
        let r = (1.0 + x[0] - y[1] - z[2]).sqrt();
        let s = 0.5 / r;
        [
            0.5 * r,
            (x[1] + y[0]) * s,
            (z[0] + x[2]) * s,
            (y[2] - z[1]) * s,
        ]
    } else if y[1] > z[2] {
        let r = (1.0 + y[1] - x[0] - z[2]).sqrt();
        let s = 0.5 / r;
        [
            (x[1] + y[0]) * s,
            0.5 * r,
            (y[2] + z[1]) * s,
            (z[0] - x[2]) * s,
        ]
    } else {
        let r = (1.0 + z[2] - x[0] - y[1]).sqrt();
        let s = 0.5 / r;
        [
            (z[0] + x[2]) * s,
            (y[2] + z[1]) * s,
            0.5 * r,
            (x[1] - y[0]) * s,
        ]
    };
    ([m[12], m[13], m[14]], turn, scale)
}

/// Writes `vertices` (rows of eight floats) and `triangles` (rows of
/// three ints) as a GLB, with `texture` as the base colour when given.
pub fn write_glb(
    vertices: &FieldData,
    triangles: &FieldData,
    texture: Option<&[u8]>,
    name: &str,
) -> Result<Vec<u8>, String> {
    write_rig(vertices, triangles, texture, None, name)
}

/// The same, with the rest of a rig's rows: a skin and one animation a
/// clip beside the mesh.
pub fn write_rig(
    vertices: &FieldData,
    triangles: &FieldData,
    texture: Option<&[u8]>,
    rig: Option<&Rig<'_>>,
    name: &str,
) -> Result<Vec<u8>, String> {
    if vertices.schema().words() != 8 || vertices.domain().shape() != orla_runtime::Shape::Rows {
        return Err("vertices must be rows of {position: Vec3, normal: Vec3, uv: Vec2}".into());
    }
    if triangles.schema().words() != 3 || triangles.domain().shape() != orla_runtime::Shape::Rows {
        return Err("triangles must be rows of {a: Int, b: Int, c: Int}".into());
    }
    let count = vertices.domain().count();
    let mut positions = Vec::with_capacity(count * 12);
    let mut normals = Vec::with_capacity(count * 12);
    let mut uvs = Vec::with_capacity(count * 8);
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for vertex in vertices.words().chunks_exact(8) {
        let floats: Vec<f32> = vertex.iter().map(|w| f32::from_bits(*w)).collect();
        for k in 0..3 {
            lo[k] = lo[k].min(floats[k]);
            hi[k] = hi[k].max(floats[k]);
        }
        positions.extend(floats[..3].iter().flat_map(|v| v.to_le_bytes()));
        normals.extend(floats[3..6].iter().flat_map(|v| v.to_le_bytes()));
        uvs.extend(floats[6..8].iter().flat_map(|v| v.to_le_bytes()));
    }
    let mut indices = Vec::with_capacity(triangles.domain().count() * 12);
    for index in triangles.words() {
        if *index as usize >= count {
            return Err(format!(
                "triangle index {index} is past the {count} vertices"
            ));
        }
        indices.extend(index.to_le_bytes());
    }
    let pad = |bytes: &mut Vec<u8>| {
        while !bytes.len().is_multiple_of(4) {
            bytes.push(0);
        }
    };
    let mut bin = Vec::new();
    let mut views = Vec::new();
    let mut view = |bin: &mut Vec<u8>, data: &[u8], target: Option<u32>| -> usize {
        let offset = bin.len();
        bin.extend_from_slice(data);
        pad(bin);
        let mut entry = json!({ "buffer": 0, "byteOffset": offset, "byteLength": data.len() });
        if let Some(target) = target {
            entry["target"] = json!(target);
        }
        views.push(entry);
        views.len() - 1
    };
    let position_view = view(&mut bin, &positions, Some(34962));
    let normal_view = view(&mut bin, &normals, Some(34962));
    let uv_view = view(&mut bin, &uvs, Some(34962));
    let index_view = view(&mut bin, &indices, Some(34963));
    let texture_view = texture.map(|png| view(&mut bin, png, None));
    // The rig: what every vertex hangs off, the joints as nodes, and a
    // clip apiece as an animation.
    let mut accessors = Vec::new();
    let mut skin_attributes = json!({});
    let mut joints: Vec<Json> = Vec::new();
    let mut animations: Vec<Json> = Vec::new();
    let mut skins: Vec<Json> = Vec::new();
    if let Some(rig) = rig {
        let scalars = rig.skeleton.schema().scalars();
        if rig.skeleton.schema().words() != 17 || scalars.first() != Some(&Scalar::Int) {
            return Err("a skeleton is rows of {parent: Int, bind: Mat4}".into());
        }
        let weight_scalars = rig.weights.schema().scalars();
        if rig.weights.schema().words() != 8
            || weight_scalars[..4] != [Scalar::Int; 4]
            || weight_scalars[4..] != [Scalar::Float; 4]
        {
            return Err("weights are rows of {joints: [Int; 4], weights: Vec4}".into());
        }
        if rig.weights.domain().count() != count {
            return Err(format!(
                "there are {} vertices and {} rows of weights",
                count,
                rig.weights.domain().count()
            ));
        }
        let joint_count = rig.skeleton.domain().count();
        let mut joint_indices = Vec::with_capacity(count * 8);
        let mut joint_weights = Vec::with_capacity(count * 16);
        for row in rig.weights.words().chunks_exact(8) {
            for word in &row[..4] {
                let index = *word as i32;
                if index < 0 || index as usize >= joint_count {
                    return Err(format!("a vertex hangs off joint {index} of {joint_count}"));
                }
                joint_indices.extend((index as u16).to_le_bytes());
            }
            for word in &row[4..] {
                joint_weights.extend(f32::from_bits(*word).to_le_bytes());
            }
        }
        // The bind matrices as they are, and the rest pose they imply:
        // a joint's place in its parent is what the binds say, undone.
        let mut binds = Vec::with_capacity(joint_count * 64);
        let parents: Vec<i32> = rig
            .skeleton
            .words()
            .chunks_exact(17)
            .map(|row| row[0] as i32)
            .collect();
        let bind_matrices: Vec<[f32; 16]> = rig
            .skeleton
            .words()
            .chunks_exact(17)
            .map(|row| {
                let mut m = [0.0f32; 16];
                for (slot, word) in m.iter_mut().zip(&row[1..]) {
                    *slot = f32::from_bits(*word);
                }
                m
            })
            .collect();
        for matrix in &bind_matrices {
            binds.extend(matrix.iter().flat_map(|v| v.to_le_bytes()));
        }
        let world: Vec<[f32; 16]> = bind_matrices.iter().map(affine_inverse).collect();
        for (index, parent) in parents.iter().enumerate() {
            let local = if *parent >= 0 {
                multiply(&affine_inverse(&world[*parent as usize]), &world[index])
            } else {
                world[index]
            };
            let (at, turn, grow) = decompose(&local);
            let children: Vec<usize> = parents
                .iter()
                .enumerate()
                .filter(|(_, up)| **up == index as i32)
                .map(|(child, _)| child + 1)
                .collect();
            let mut node = json!({
                "name": format!("joint{index}"),
                "translation": at,
                "rotation": turn,
                "scale": grow,
            });
            if !children.is_empty() {
                node["children"] = json!(children);
            }
            joints.push(node);
        }
        let joints_view = view(&mut bin, &joint_indices, Some(34962));
        let weights_view = view(&mut bin, &joint_weights, Some(34962));
        let binds_view = view(&mut bin, &binds, None);
        accessors.push(json!({ "bufferView": joints_view, "componentType": 5123, "count": count, "type": "VEC4" }));
        accessors.push(json!({ "bufferView": weights_view, "componentType": 5126, "count": count, "type": "VEC4" }));
        accessors.push(json!({ "bufferView": binds_view, "componentType": 5126, "count": joint_count, "type": "MAT4" }));
        skin_attributes = json!({ "JOINTS_0": 4, "WEIGHTS_0": 5 });
        skins.push(json!({
            "name": name,
            "joints": (1..=joint_count).collect::<Vec<usize>>(),
            "inverseBindMatrices": 6,
        }));
        // One animation a clip: a time apiece, and a joint's
        // translation, rotation and scale as they were sampled.
        for (clip_name, clip, rate) in &rig.clips {
            if clip.schema().words() != 10 {
                return Err(format!("clip `{clip_name}` is not rows of ten floats"));
            }
            if clip.domain().count() % joint_count != 0 {
                return Err(format!(
                    "clip `{clip_name}` holds {} rows, which is not a whole number of {joint_count}-joint frames",
                    clip.domain().count()
                ));
            }
            let frames = clip.domain().count() / joint_count;
            let times: Vec<u8> = (0..frames)
                .flat_map(|frame| (frame as f32 / rate).to_le_bytes())
                .collect();
            let times_view = view(&mut bin, &times, None);
            let time_accessor = accessors.len() + 4;
            accessors.push(json!({
                "bufferView": times_view, "componentType": 5126, "count": frames,
                "type": "SCALAR", "min": [0.0], "max": [(frames.saturating_sub(1)) as f32 / rate]
            }));
            let mut samplers = Vec::new();
            let mut channels = Vec::new();
            let words = clip.words();
            for joint in 0..joint_count {
                let take = |offset: usize, lanes: usize| -> Vec<u8> {
                    (0..frames)
                        .flat_map(|frame| {
                            let base = (frame * joint_count + joint) * 10 + offset;
                            words[base..base + lanes]
                                .iter()
                                .flat_map(|w| f32::from_bits(*w).to_le_bytes())
                                .collect::<Vec<u8>>()
                        })
                        .collect()
                };
                for (offset, lanes, path) in [
                    (0usize, 3usize, "translation"),
                    (3, 4, "rotation"),
                    (7, 3, "scale"),
                ] {
                    let data = take(offset, lanes);
                    let data_view = view(&mut bin, &data, None);
                    accessors.push(json!({
                        "bufferView": data_view, "componentType": 5126, "count": frames,
                        "type": if lanes == 4 { "VEC4" } else { "VEC3" }
                    }));
                    samplers.push(json!({
                        "input": time_accessor,
                        "output": accessors.len() + 3,
                        "interpolation": "LINEAR"
                    }));
                    channels.push(json!({
                        "sampler": samplers.len() - 1,
                        "target": { "node": joint + 1, "path": path }
                    }));
                }
            }
            animations
                .push(json!({ "name": clip_name, "samplers": samplers, "channels": channels }));
        }
    }
    let mut document = json!({
        "asset": { "version": "2.0", "generator": "orla-export" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0, "name": name }],
        "meshes": [{
            "name": name,
            "primitives": [{
                "attributes": { "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2 },
                "indices": 3,
                "mode": 4
            }]
        }],
        "buffers": [{ "byteLength": bin.len() }],
        "bufferViews": views,
        "accessors": [
            { "bufferView": position_view, "componentType": 5126, "count": count, "type": "VEC3", "min": lo, "max": hi },
            { "bufferView": normal_view, "componentType": 5126, "count": count, "type": "VEC3" },
            { "bufferView": uv_view, "componentType": 5126, "count": count, "type": "VEC2" },
            { "bufferView": index_view, "componentType": 5125, "count": triangles.domain().count() * 3, "type": "SCALAR" }
        ]
    });
    if !joints.is_empty() {
        // The skinned mesh node is placed by its joints, so it stands
        // beside them in the scene rather than under them.
        document["nodes"][0]["skin"] = json!(0);
        let roots: Vec<usize> = joints
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                rig.map(|rig| {
                    rig.skeleton
                        .words()
                        .chunks_exact(17)
                        .nth(*index)
                        .map(|row| row[0] as i32)
                        == Some(-1)
                })
                .unwrap_or(false)
            })
            .map(|(index, _)| index + 1)
            .collect();
        let mut nodes = document["nodes"].as_array().cloned().unwrap_or_default();
        nodes.extend(joints);
        document["nodes"] = json!(nodes);
        let scene_nodes: Vec<usize> = std::iter::once(0).chain(roots).collect();
        document["scenes"] = json!([{ "nodes": scene_nodes }]);
        for (key, value) in skin_attributes.as_object().into_iter().flatten() {
            document["meshes"][0]["primitives"][0]["attributes"][key] = value.clone();
        }
        document["skins"] = json!(skins);
        let mut all = document["accessors"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        all.extend(accessors);
        document["accessors"] = json!(all);
        if !animations.is_empty() {
            document["animations"] = json!(animations);
        }
    }
    if let Some(texture_view) = texture_view {
        document["images"] = json!([{ "bufferView": texture_view, "mimeType": "image/png" }]);
        document["samplers"] =
            json!([{ "magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497 }]);
        document["textures"] = json!([{ "source": 0, "sampler": 0 }]);
        document["materials"] = json!([{
            "name": name,
            "pbrMetallicRoughness": { "baseColorTexture": { "index": 0 }, "metallicFactor": 0.0, "roughnessFactor": 0.8 }
        }]);
        document["meshes"][0]["primitives"][0]["material"] = json!(0);
    }
    let mut text = serde_json::to_vec(&document).map_err(|e| e.to_string())?;
    while !text.len().is_multiple_of(4) {
        text.push(b' ');
    }
    let mut out = Vec::with_capacity(28 + text.len() + bin.len());
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&((28 + text.len() + bin.len()) as u32).to_le_bytes());
    out.extend_from_slice(&(text.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&text);
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&bin);
    Ok(out)
}

/// Reads the vertex and triangle Field files, and the texture when
/// given, and writes the GLB at `out`.
pub fn export(
    vertices: &Path,
    triangles: &Path,
    texture: Option<&Path>,
    out: &Path,
) -> Result<(), String> {
    export_rig(vertices, triangles, texture, &[], &[], 30.0, out)
}

/// The same, with a rig: `weights` and `skeleton` beside the mesh, and
/// a clip file apiece, each named by what stands before `.clip.field`.
pub fn export_rig(
    vertices: &Path,
    triangles: &Path,
    texture: Option<&Path>,
    rig: &[&Path],
    clips: &[&Path],
    rate: f32,
    out: &Path,
) -> Result<(), String> {
    let read = |path: &Path| -> Result<FieldData, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        orla_runtime::file::decode(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    };
    let png = match texture {
        Some(path) => {
            let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            orla_session::png::decode(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
            Some(bytes)
        }
        None => None,
    };
    let name = out.file_stem().and_then(|s| s.to_str()).unwrap_or("mesh");
    let rig_rows: Vec<FieldData> = rig
        .iter()
        .map(|path| read(path))
        .collect::<Result<_, _>>()?;
    let clip_rows: Vec<FieldData> = clips
        .iter()
        .map(|path| read(path))
        .collect::<Result<_, _>>()?;
    let mut weights = None;
    let mut skeleton = None;
    for (path, data) in rig.iter().zip(&rig_rows) {
        match data.schema().words() {
            8 => weights = Some(data),
            17 => skeleton = Some(data),
            other => {
                return Err(format!(
                    "{}: {other} words a row is no part of a rig",
                    path.display()
                ));
            }
        }
    }
    let rig = match (weights, skeleton) {
        (Some(weights), Some(skeleton)) => Some(Rig {
            weights,
            skeleton,
            clips: clips
                .iter()
                .zip(&clip_rows)
                .map(|(path, data)| (clip_name(path), data, rate))
                .collect(),
        }),
        (None, None) if clips.is_empty() => None,
        _ => {
            return Err(
                "a rig needs both a weights .field and a skeleton .field beside the mesh".into(),
            );
        }
    };
    let glb = write_rig(
        &read(vertices)?,
        &read(triangles)?,
        png.as_deref(),
        rig.as_ref(),
        name,
    )?;
    std::fs::write(out, glb).map_err(|e| format!("{}: {e}", out.display()))
}

/// A clip's name: what stands before `.clip.field`, or the stem.
fn clip_name(path: &Path) -> String {
    let stem = path.file_name().and_then(|s| s.to_str()).unwrap_or("clip");
    let stem = stem.strip_suffix(".clip.field").unwrap_or(stem);
    stem.rsplit('.').next().unwrap_or(stem).to_owned()
}
