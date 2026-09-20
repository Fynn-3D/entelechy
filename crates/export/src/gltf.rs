//! A mesh as GLB: the vertex rows `{position, normal, uv}` and the
//! triangle rows the runtime uses, and a PNG for the base colour when
//! there is one, become one glTF 2.0 binary with one mesh, one node and
//! one scene, which Blender, three.js and the rest open as they are.

use std::path::Path;

use orla_runtime::FieldData;
use serde_json::json;

/// Writes `vertices` (rows of eight floats) and `triangles` (rows of
/// three ints) as a GLB, with `texture` as the base colour when given.
pub fn write_glb(
    vertices: &FieldData,
    triangles: &FieldData,
    texture: Option<&[u8]>,
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
    let read = |path: &Path| -> Result<FieldData, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        orla_runtime::file::decode(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    };
    let png = match texture {
        Some(path) => {
            let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            orla::png::decode(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
            Some(bytes)
        }
        None => None,
    };
    let name = out.file_stem().and_then(|s| s.to_str()).unwrap_or("mesh");
    let glb = write_glb(&read(vertices)?, &read(triangles)?, png.as_deref(), name)?;
    std::fs::write(out, glb).map_err(|e| format!("{}: {e}", out.display()))
}
