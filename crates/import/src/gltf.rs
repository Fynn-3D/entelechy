//! glTF 2.0 as Fields: `orla import` reads a model and writes what a
//! scene takes as inputs: vertices as rows of `{position, normal, uv}`,
//! triangles as rows of three indices, the base colour texture as a PNG.
//! Node transforms are baked into positions and normals, so the model
//! stands as its author placed it; every mesh of the scene lands in one
//! pair of rows. Skins, animations, sparse accessors and JPEG textures
//! are reported and left where they are.

use std::path::{Path, PathBuf};

use serde_json::Value as Json;

use orla_runtime::{Domain, Scalar, Schema, file};

use orla_formats::base64::unbase64;

/// What an import produced.
pub struct Imported {
    /// Rows of vertices written.
    pub vertices: usize,
    /// Rows of triangles written.
    pub triangles: usize,
    /// The texture file, when the model has a base colour texture in PNG.
    pub texture: Option<PathBuf>,
    /// Notes about what was left out.
    pub notes: Vec<String>,
    /// Corners of the model's bounding box after transforms.
    pub bounds: ([f32; 3], [f32; 3]),
    /// Joints of the skin, when the model has one.
    pub joints: usize,
    /// Each animation as a clip: its name and how many frames it was
    /// sampled into.
    pub clips: Vec<(String, usize)>,
    /// Frames a second the clips were sampled at.
    pub rate: f32,
    /// Every file written.
    pub files: Vec<PathBuf>,
}

type Matrix = [f32; 16];

const IDENTITY: Matrix = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

/// Column-major product `a * b`.
fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = [0.0; 16];
    for column in 0..4 {
        for row in 0..4 {
            out[column * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[column * 4 + k]).sum();
        }
    }
    out
}

fn transform_point(m: &Matrix, p: [f32; 3]) -> [f32; 3] {
    let w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
    let w = if w.abs() < 1e-12 { 1.0 } else { w };
    [
        (m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12]) / w,
        (m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13]) / w,
        (m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]) / w,
    ]
}

/// Normals transform by the inverse transpose of the upper 3×3, which
/// is the adjugate over the determinant, transposed: the cofactors.
fn transform_normal(m: &Matrix, n: [f32; 3]) -> [f32; 3] {
    let (a, b, c) = (m[0], m[4], m[8]);
    let (d, e, f) = (m[1], m[5], m[9]);
    let (g, h, i) = (m[2], m[6], m[10]);
    let cofactors = [
        e * i - f * h,
        f * g - d * i,
        d * h - e * g,
        c * h - b * i,
        a * i - c * g,
        b * g - a * h,
        b * f - c * e,
        c * d - a * f,
        a * e - b * d,
    ];
    let out = [
        cofactors[0] * n[0] + cofactors[3] * n[1] + cofactors[6] * n[2],
        cofactors[1] * n[0] + cofactors[4] * n[1] + cofactors[7] * n[2],
        cofactors[2] * n[0] + cofactors[5] * n[1] + cofactors[8] * n[2],
    ];
    normalize(out)
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if length > 0.0 {
        [v[0] / length, v[1] / length, v[2] / length]
    } else {
        [0.0, 0.0, 0.0]
    }
}

fn local_matrix(node: &Json) -> Matrix {
    if let Some(m) = node.get("matrix").and_then(Json::as_array)
        && m.len() == 16
    {
        let mut out = [0.0; 16];
        for (k, v) in m.iter().enumerate() {
            out[k] = v.as_f64().unwrap_or(0.0) as f32;
        }
        return out;
    }
    let floats = |key: &str, n: usize, default: &[f32]| -> Vec<f32> {
        node.get(key)
            .and_then(Json::as_array)
            .filter(|a| a.len() == n)
            .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect())
            .unwrap_or_else(|| default.to_vec())
    };
    let t = floats("translation", 3, &[0.0, 0.0, 0.0]);
    let r = floats("rotation", 4, &[0.0, 0.0, 0.0, 1.0]);
    let s = floats("scale", 3, &[1.0, 1.0, 1.0]);
    let (x, y, z, w) = (r[0], r[1], r[2], r[3]);
    let rotation: Matrix = [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + z * w),
        2.0 * (x * z - y * w),
        0.0,
        2.0 * (x * y - z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + x * w),
        0.0,
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ];
    let scale: Matrix = [
        s[0], 0.0, 0.0, 0.0, 0.0, s[1], 0.0, 0.0, 0.0, 0.0, s[2], 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let mut translation = IDENTITY;
    translation[12] = t[0];
    translation[13] = t[1];
    translation[14] = t[2];
    multiply(&translation, &multiply(&rotation, &scale))
}

/// The document: its JSON and its buffers, resolved.
struct Model {
    json: Json,
    buffers: Vec<Vec<u8>>,
}

impl Model {
    fn open(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        let (json, glb_bin): (Json, Option<Vec<u8>>) = if bytes.starts_with(b"glTF") {
            if bytes.len() < 20 {
                return Err("GLB header ends early".into());
            }
            let mut position = 12;
            let mut json = None;
            let mut bin = None;
            while position + 8 <= bytes.len() {
                let length = u32::from_le_bytes([
                    bytes[position],
                    bytes[position + 1],
                    bytes[position + 2],
                    bytes[position + 3],
                ]) as usize;
                let kind = &bytes[position + 4..position + 8];
                let body = bytes
                    .get(position + 8..position + 8 + length)
                    .ok_or("GLB chunk ends early")?;
                match kind {
                    b"JSON" => {
                        json = Some(serde_json::from_slice(body).map_err(|e| e.to_string())?)
                    }
                    b"BIN\0" => bin = Some(body.to_vec()),
                    _ => {}
                }
                position += 8 + length;
            }
            (json.ok_or("GLB has no JSON chunk")?, bin)
        } else {
            (
                serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?,
                None,
            )
        };
        let mut buffers = Vec::new();
        for (index, buffer) in json
            .get("buffers")
            .and_then(Json::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            let data = match buffer.get("uri").and_then(Json::as_str) {
                Some(uri) => read_uri(dir, uri)?,
                None => glb_bin.clone().ok_or_else(|| {
                    format!("buffer {index} has no uri and there is no GLB chunk")
                })?,
            };
            buffers.push(data);
        }
        Ok(Self { json, buffers })
    }

    fn view(&self, index: usize) -> Result<(&[u8], Option<usize>), String> {
        let view = self.json["bufferViews"]
            .get(index)
            .ok_or_else(|| format!("bufferView {index} is missing"))?;
        let buffer = view["buffer"].as_u64().unwrap_or(0) as usize;
        let offset = view["byteOffset"].as_u64().unwrap_or(0) as usize;
        let length = view["byteLength"].as_u64().unwrap_or(0) as usize;
        let stride = view["byteStride"].as_u64().map(|s| s as usize);
        let data = self
            .buffers
            .get(buffer)
            .and_then(|b| b.get(offset..offset + length))
            .ok_or_else(|| format!("bufferView {index} is out of its buffer"))?;
        Ok((data, stride))
    }

    /// An accessor's elements as floats, `components` per element;
    /// normalized integers scale to `0..=1` or `-1..=1`.
    fn floats(&self, index: usize) -> Result<(Vec<f32>, usize), String> {
        let accessor = self.json["accessors"]
            .get(index)
            .ok_or_else(|| format!("accessor {index} is missing"))?;
        if accessor.get("sparse").is_some() {
            return Err(format!(
                "accessor {index} is sparse, which is not supported"
            ));
        }
        let components = match accessor["type"].as_str().unwrap_or("") {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            "MAT4" => 16,
            other => return Err(format!("accessor {index} has type {other}")),
        };
        let component = accessor["componentType"].as_u64().unwrap_or(0);
        let normalized = accessor["normalized"].as_bool().unwrap_or(false);
        let count = accessor["count"].as_u64().unwrap_or(0) as usize;
        let size = match component {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            5125 | 5126 => 4,
            other => return Err(format!("accessor {index} has component type {other}")),
        };
        let Some(view_index) = accessor["bufferView"].as_u64() else {
            return Ok((vec![0.0; count * components], components));
        };
        let (data, stride) = self.view(view_index as usize)?;
        let start = accessor["byteOffset"].as_u64().unwrap_or(0) as usize;
        let stride = stride.unwrap_or(size * components);
        let mut out = Vec::with_capacity(count * components);
        for element in 0..count {
            for k in 0..components {
                let at = start + element * stride + k * size;
                let bytes = data
                    .get(at..at + size)
                    .ok_or_else(|| format!("accessor {index} runs past its view"))?;
                let value = match component {
                    5120 => {
                        let v = bytes[0] as i8 as f32;
                        if normalized { (v / 127.0).max(-1.0) } else { v }
                    }
                    5121 => {
                        let v = bytes[0] as f32;
                        if normalized { v / 255.0 } else { v }
                    }
                    5122 => {
                        let v = i16::from_le_bytes([bytes[0], bytes[1]]) as f32;
                        if normalized {
                            (v / 32767.0).max(-1.0)
                        } else {
                            v
                        }
                    }
                    5123 => {
                        let v = u16::from_le_bytes([bytes[0], bytes[1]]) as f32;
                        if normalized { v / 65535.0 } else { v }
                    }
                    5125 => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32,
                    _ => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                };
                out.push(value);
            }
        }
        Ok((out, components))
    }

    /// An index accessor's elements.
    fn indices(&self, index: usize) -> Result<Vec<u32>, String> {
        let accessor = self.json["accessors"]
            .get(index)
            .ok_or_else(|| format!("accessor {index} is missing"))?;
        let component = accessor["componentType"].as_u64().unwrap_or(0);
        let count = accessor["count"].as_u64().unwrap_or(0) as usize;
        let size = match component {
            5121 => 1,
            5123 => 2,
            5125 => 4,
            other => return Err(format!("index accessor {index} has component type {other}")),
        };
        let view_index = accessor["bufferView"]
            .as_u64()
            .ok_or_else(|| format!("index accessor {index} has no bufferView"))?;
        let (data, stride) = self.view(view_index as usize)?;
        let start = accessor["byteOffset"].as_u64().unwrap_or(0) as usize;
        let stride = stride.unwrap_or(size);
        (0..count)
            .map(|element| {
                let at = start + element * stride;
                let bytes = data
                    .get(at..at + size)
                    .ok_or_else(|| format!("index accessor {index} runs past its view"))?;
                Ok(match size {
                    1 => bytes[0] as u32,
                    2 => u16::from_le_bytes([bytes[0], bytes[1]]) as u32,
                    _ => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                })
            })
            .collect()
    }
}

fn read_uri(dir: &Path, uri: &str) -> Result<Vec<u8>, String> {
    if let Some(rest) = uri.strip_prefix("data:") {
        let comma = rest.find(',').ok_or("data URI has no comma")?;
        if !rest[..comma].ends_with(";base64") {
            return Err("data URI is not base64".into());
        }
        return unbase64(&rest[comma + 1..]);
    }
    let path = dir.join(uri.replace("%20", " "));
    std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))
}

/// A vertex as the runtime takes it.
#[derive(Clone, Copy)]
struct Point {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

/// What a vertex is bound to: four joints of the skin and how much of
/// each. All four weights zero means the vertex is bound to nothing,
/// which is how a model that mixes skinned and unskinned meshes says
/// "leave this one where it is".
#[derive(Clone, Copy, Default)]
struct Bound {
    joints: [u32; 4],
    weights: [f32; 4],
}

/// A skin as the rows a scene reads: a parent and an inverse bind
/// matrix per joint, parents before children.
struct Skeleton {
    /// Which node each joint is, in the skin's own order.
    nodes: Vec<usize>,
    /// The joint's parent in this same row, or -1 at a root.
    parents: Vec<i32>,
    binds: Vec<Matrix>,
}

/// How a sampler reads between its keys.
#[derive(Clone, Copy, PartialEq)]
enum Between {
    Step,
    Linear,
    Cubic,
}

/// One channel of an animation: the times, the values, and how to read
/// between them.
struct Track {
    times: Vec<f32>,
    values: Vec<f32>,
    stride: usize,
    between: Between,
}

impl Track {
    /// The value at `t`, held at the ends.
    fn at(&self, t: f32, rotation: bool) -> Vec<f32> {
        let stride = self.stride;
        let keys = self.times.len();
        if keys == 0 {
            return vec![0.0; stride];
        }
        let value = |k: usize| -> &[f32] {
            let base = if self.between == Between::Cubic {
                k * stride * 3 + stride
            } else {
                k * stride
            };
            &self.values[base..base + stride]
        };
        if keys == 1 || t <= self.times[0] {
            return value(0).to_vec();
        }
        if t >= self.times[keys - 1] {
            return value(keys - 1).to_vec();
        }
        let next = self.times.iter().position(|&time| time > t).unwrap_or(1);
        let first = next - 1;
        let span = self.times[next] - self.times[first];
        let u = if span > 0.0 {
            (t - self.times[first]) / span
        } else {
            0.0
        };
        match self.between {
            Between::Step => value(first).to_vec(),
            Between::Linear if rotation => slerp(value(first), value(next), u),
            Between::Linear => value(first)
                .iter()
                .zip(value(next))
                .map(|(a, b)| a + (b - a) * u)
                .collect(),
            Between::Cubic => {
                // The spline glTF defines: each key is an in-tangent,
                // the value, and an out-tangent.
                let out = |k: usize| &self.values[k * stride * 3 + stride * 2..][..stride];
                let into = |k: usize| &self.values[k * stride * 3..][..stride];
                let (t2, t3) = (u * u, u * u * u);
                let a = 2.0 * t3 - 3.0 * t2 + 1.0;
                let b = t3 - 2.0 * t2 + u;
                let c = -2.0 * t3 + 3.0 * t2;
                let d = t3 - t2;
                let mixed: Vec<f32> = (0..stride)
                    .map(|k| {
                        a * value(first)[k]
                            + b * span * out(first)[k]
                            + c * value(next)[k]
                            + d * span * into(next)[k]
                    })
                    .collect();
                if rotation { normalize4(&mixed) } else { mixed }
            }
        }
    }
}

/// The shorter way round between two rotations.
fn slerp(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let mut d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let sign = if d < 0.0 { -1.0 } else { 1.0 };
    d = d.abs();
    if d > 0.9995 {
        let mixed: Vec<f32> = a
            .iter()
            .zip(b)
            .map(|(x, y)| x + (y * sign - x) * t)
            .collect();
        return normalize4(&mixed);
    }
    let angle = d.acos();
    let (first, second) = (
        ((1.0 - t) * angle).sin() / angle.sin(),
        (t * angle).sin() / angle.sin(),
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| x * first + y * sign * second)
        .collect()
}

fn normalize4(v: &[f32]) -> Vec<f32> {
    let length = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if length > 0.0 {
        v.iter().map(|x| x / length).collect()
    } else {
        vec![0.0, 0.0, 0.0, 1.0]
    }
}

/// A node's own transform as the ten floats a `Pose` holds.
fn rest_pose(node: &Json) -> [f32; 10] {
    let three = |key: &str, fallback: [f32; 3]| -> [f32; 3] {
        node.get(key)
            .and_then(Json::as_array)
            .filter(|v| v.len() == 3)
            .map(|v| {
                let mut out = fallback;
                for (k, slot) in out.iter_mut().enumerate() {
                    *slot = v[k].as_f64().unwrap_or(0.0) as f32;
                }
                out
            })
            .unwrap_or(fallback)
    };
    let at = three("translation", [0.0; 3]);
    let grow = three("scale", [1.0; 3]);
    let turn = node
        .get("rotation")
        .and_then(Json::as_array)
        .filter(|v| v.len() == 4)
        .map(|v| {
            let mut out = [0.0, 0.0, 0.0, 1.0];
            for (k, slot) in out.iter_mut().enumerate() {
                *slot = v[k].as_f64().unwrap_or(0.0) as f32;
            }
            out
        })
        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
    [
        at[0], at[1], at[2], turn[0], turn[1], turn[2], turn[3], grow[0], grow[1], grow[2],
    ]
}

/// The skin's joints as rows: each one's parent in the same row and its
/// inverse bind matrix. The order is the skin's own, which is what a
/// vertex's `JOINTS_0` indexes.
fn skeleton_of(model: &Model, skin: usize) -> Result<Skeleton, String> {
    let skin = model.json["skins"]
        .get(skin)
        .ok_or_else(|| format!("skin {skin} is missing"))?;
    let nodes: Vec<usize> = skin["joints"]
        .as_array()
        .ok_or("a skin has no joints")?
        .iter()
        .filter_map(|j| j.as_u64().map(|n| n as usize))
        .collect();
    if nodes.is_empty() {
        return Err("a skin has no joints".into());
    }
    // Who holds whom, over every node there is: a joint's parent may be
    // any node, and only the ones that are joints of this skin count.
    let mut parent_of: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for (index, node) in model.json["nodes"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        for child in node["children"].as_array().into_iter().flatten() {
            if let Some(child) = child.as_u64() {
                parent_of.insert(child as usize, index);
            }
        }
    }
    let place = |node: usize| nodes.iter().position(|&j| j == node);
    let parents: Vec<i32> = nodes
        .iter()
        .map(|&node| {
            parent_of
                .get(&node)
                .and_then(|&up| place(up))
                .map(|index| index as i32)
                .unwrap_or(-1)
        })
        .collect();
    let binds = match skin["inverseBindMatrices"].as_u64() {
        Some(accessor) => {
            let (values, components) = model.floats(accessor as usize)?;
            if components != 16 {
                return Err("inverse bind matrices are not 4x4".into());
            }
            nodes
                .iter()
                .enumerate()
                .map(|(index, _)| {
                    let mut m = IDENTITY;
                    m.copy_from_slice(&values[index * 16..index * 16 + 16]);
                    m
                })
                .collect()
        }
        None => vec![IDENTITY; nodes.len()],
    };
    Ok(Skeleton {
        nodes,
        parents,
        binds,
    })
}

/// One animation as a clip: every joint's local transform at every
/// frame, a frame's joints together, sampled at `rate`.
fn clip_of(
    model: &Model,
    animation: &Json,
    skeleton: &Skeleton,
    rate: f32,
) -> Result<(Vec<f32>, usize), String> {
    let samplers = animation["samplers"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    // A track per joint per path, and how long the whole thing runs.
    let mut tracks: std::collections::BTreeMap<(usize, String), Track> =
        std::collections::BTreeMap::new();
    let mut seconds: f32 = 0.0;
    for channel in animation["channels"].as_array().into_iter().flatten() {
        let Some(node) = channel["target"]["node"].as_u64() else {
            continue;
        };
        let Some(joint) = skeleton.nodes.iter().position(|&j| j == node as usize) else {
            continue;
        };
        let path = channel["target"]["path"].as_str().unwrap_or("").to_owned();
        if !matches!(path.as_str(), "translation" | "rotation" | "scale") {
            continue;
        }
        let Some(sampler) = channel["sampler"]
            .as_u64()
            .and_then(|index| samplers.get(index as usize))
        else {
            continue;
        };
        let (times, _) = model.floats(sampler["input"].as_u64().unwrap_or(0) as usize)?;
        let (values, stride) = model.floats(sampler["output"].as_u64().unwrap_or(0) as usize)?;
        let between = match sampler["interpolation"].as_str().unwrap_or("LINEAR") {
            "STEP" => Between::Step,
            "CUBICSPLINE" => Between::Cubic,
            _ => Between::Linear,
        };
        seconds = seconds.max(times.last().copied().unwrap_or(0.0));
        tracks.insert(
            (joint, path),
            Track {
                times,
                values,
                stride,
                between,
            },
        );
    }
    let frames = ((seconds * rate).ceil() as usize + 1).max(1);
    let joints = skeleton.nodes.len();
    let rest: Vec<[f32; 10]> = skeleton
        .nodes
        .iter()
        .map(|&node| {
            model.json["nodes"]
                .get(node)
                .map(rest_pose)
                .unwrap_or([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        })
        .collect();
    let mut out = Vec::with_capacity(frames * joints * 10);
    for frame in 0..frames {
        let t = frame as f32 / rate;
        for (joint, base) in rest.iter().enumerate() {
            let mut pose = *base;
            if let Some(track) = tracks.get(&(joint, "translation".to_owned())) {
                pose[..3].copy_from_slice(&track.at(t, false)[..3]);
            }
            if let Some(track) = tracks.get(&(joint, "rotation".to_owned())) {
                pose[3..7].copy_from_slice(&track.at(t, true)[..4]);
            }
            if let Some(track) = tracks.get(&(joint, "scale".to_owned())) {
                pose[7..10].copy_from_slice(&track.at(t, false)[..3]);
            }
            out.extend_from_slice(&pose);
        }
    }
    Ok((out, frames))
}

/// Reads `path` and writes `<stem>.vertices.field`, `<stem>.triangles.field`
/// and, when there is one in PNG, `<stem>.albedo.png` into `out_dir`.
///
/// A model with a skin writes two more — `<stem>.weights.field`, the four
/// joints and weights of every vertex, and `<stem>.skeleton.field`, each
/// joint's parent and inverse bind matrix — and one `<stem>.<name>.clip.field`
/// per animation, sampled at `rate` frames a second: rows of a local
/// translation, rotation and scale, a frame's joints together. The
/// `skin` package poses a mesh from those.
pub fn import(path: &Path, out_dir: &Path, rate: f32) -> Result<Imported, String> {
    let model = Model::open(path)?;
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_owned();
    let mut notes = Vec::new();
    let mut points: Vec<Point> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut bounds: Vec<Bound> = Vec::new();
    let skins = model
        .json
        .get("skins")
        .and_then(Json::as_array)
        .map(|s| s.len())
        .unwrap_or(0);
    if skins > 1 {
        notes.push(format!(
            "the model has {skins} skins; the first one comes in and the rest are left out"
        ));
    }
    let skeleton = if skins > 0 {
        Some(skeleton_of(&model, 0)?)
    } else {
        None
    };
    // What a model can carry that these rows do not, said rather than
    // dropped: a reader that loses something in silence is worse than
    // one that refuses.
    let morphs = model.json["meshes"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|mesh| mesh["primitives"].as_array().into_iter().flatten())
        .any(|primitive| primitive.get("targets").is_some());
    if morphs {
        notes.push("morph targets are left out; the mesh comes in at its own shape".into());
    }
    let textured = model.json["materials"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|material| {
            material["pbrMetallicRoughness"]["baseColorTexture"]["index"].is_number()
        })
        .count();
    if textured > 1 {
        notes.push(format!(
            "the model has {textured} materials with a base colour texture; the first comes in and the rest are left out"
        ));
    }
    // Walk the scene: nodes carry meshes and transforms.
    let scene_index = model.json["scene"].as_u64().unwrap_or(0) as usize;
    let roots: Vec<usize> = model.json["scenes"]
        .get(scene_index)
        .and_then(|s| s["nodes"].as_array())
        .map(|nodes| {
            nodes
                .iter()
                .filter_map(|n| n.as_u64())
                .map(|n| n as usize)
                .collect()
        })
        .unwrap_or_else(|| {
            // No scenes: every node stands on its own.
            (0..model.json["nodes"].as_array().map_or(0, Vec::len)).collect()
        });
    let mut stack: Vec<(usize, Matrix)> = roots.iter().map(|&n| (n, IDENTITY)).collect();
    let mut visited = std::collections::BTreeSet::new();
    while let Some((index, parent)) = stack.pop() {
        if !visited.insert(index) {
            continue;
        }
        let Some(node) = model.json["nodes"].get(index) else {
            continue;
        };
        let world = multiply(&parent, &local_matrix(node));
        for child in node["children"].as_array().into_iter().flatten() {
            if let Some(child) = child.as_u64() {
                stack.push((child as usize, world));
            }
        }
        let Some(mesh_index) = node["mesh"].as_u64() else {
            continue;
        };
        let Some(mesh) = model.json["meshes"].get(mesh_index as usize) else {
            continue;
        };
        for primitive in mesh["primitives"].as_array().into_iter().flatten() {
            let mode = primitive["mode"].as_u64().unwrap_or(4);
            if mode != 4 {
                notes.push(format!(
                    "a primitive of mode {mode} is left out; only triangles come in"
                ));
                continue;
            }
            let attributes = &primitive["attributes"];
            let Some(position_index) = attributes["POSITION"].as_u64() else {
                notes.push("a primitive without positions is left out".into());
                continue;
            };
            let (positions, _) = model.floats(position_index as usize)?;
            let count = positions.len() / 3;
            let normals = match attributes["NORMAL"].as_u64() {
                Some(i) => Some(model.floats(i as usize)?.0),
                None => None,
            };
            let uvs = match attributes["TEXCOORD_0"].as_u64() {
                Some(i) => Some(model.floats(i as usize)?.0),
                None => None,
            };
            let joints = match attributes["JOINTS_0"].as_u64() {
                Some(i) => Some(model.floats(i as usize)?.0),
                None => None,
            };
            let weights = match attributes["WEIGHTS_0"].as_u64() {
                Some(i) => Some(model.floats(i as usize)?.0),
                None => None,
            };
            // A skinned mesh is placed by its joints, so glTF says to
            // ignore the node's own transform; an unskinned one is
            // placed where its author put it.
            let skinned = node.get("skin").is_some() && joints.is_some() && weights.is_some();
            let world = if skinned { IDENTITY } else { world };
            let base = points.len() as u32;
            for k in 0..count {
                let position = transform_point(
                    &world,
                    [positions[k * 3], positions[k * 3 + 1], positions[k * 3 + 2]],
                );
                let normal = normals
                    .as_ref()
                    .map(|n| transform_normal(&world, [n[k * 3], n[k * 3 + 1], n[k * 3 + 2]]))
                    .unwrap_or([0.0, 0.0, 0.0]);
                let uv = uvs
                    .as_ref()
                    .map(|t| [t[k * 2], t[k * 2 + 1]])
                    .unwrap_or([0.0, 0.0]);
                points.push(Point {
                    position,
                    normal,
                    uv,
                });
                if skeleton.is_some() {
                    bounds.push(match (&joints, &weights) {
                        (Some(j), Some(w)) => Bound {
                            joints: [
                                j[k * 4] as u32,
                                j[k * 4 + 1] as u32,
                                j[k * 4 + 2] as u32,
                                j[k * 4 + 3] as u32,
                            ],
                            weights: [w[k * 4], w[k * 4 + 1], w[k * 4 + 2], w[k * 4 + 3]],
                        },
                        // Bound to nothing, which the `skin` package
                        // reads as "leave it where it is".
                        _ => Bound::default(),
                    });
                }
            }
            let indices = match primitive["indices"].as_u64() {
                Some(i) => model.indices(i as usize)?,
                None => (0..count as u32).collect(),
            };
            let first_triangle = triangles.len();
            for face in indices.chunks_exact(3) {
                if face.iter().any(|i| *i as usize >= count) {
                    return Err("an index points past its primitive's vertices".into());
                }
                triangles.push([base + face[0], base + face[1], base + face[2]]);
            }
            if normals.is_none() {
                // Area-weighted vertex normals from the faces.
                for face in &triangles[first_triangle..] {
                    let [a, b, c] = [
                        points[face[0] as usize].position,
                        points[face[1] as usize].position,
                        points[face[2] as usize].position,
                    ];
                    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                    let n = [
                        u[1] * v[2] - u[2] * v[1],
                        u[2] * v[0] - u[0] * v[2],
                        u[0] * v[1] - u[1] * v[0],
                    ];
                    for &i in face {
                        let p = &mut points[i as usize];
                        p.normal = [p.normal[0] + n[0], p.normal[1] + n[1], p.normal[2] + n[2]];
                    }
                }
                for p in &mut points[base as usize..] {
                    p.normal = normalize(p.normal);
                }
            }
        }
    }
    if points.is_empty() || triangles.is_empty() {
        return Err("the model has no triangles".into());
    }
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for p in &points {
        for k in 0..3 {
            lo[k] = lo[k].min(p.position[k]);
            hi[k] = hi[k].max(p.position[k]);
        }
    }
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    let mut files = Vec::new();
    let vertex_words: Vec<u32> = points
        .iter()
        .flat_map(|p| {
            [
                p.position[0],
                p.position[1],
                p.position[2],
                p.normal[0],
                p.normal[1],
                p.normal[2],
                p.uv[0],
                p.uv[1],
            ]
            .map(f32::to_bits)
        })
        .collect();
    let vertices_path = out_dir.join(format!("{stem}.vertices.field"));
    std::fs::write(
        &vertices_path,
        file::encode(
            &Schema::new(vec![Scalar::Float; 8]),
            Domain::Rows(points.len() as u32),
            &vertex_words,
        ),
    )
    .map_err(|e| format!("{}: {e}", vertices_path.display()))?;
    files.push(vertices_path);
    let triangle_words: Vec<u32> = triangles.iter().flat_map(|t| [t[0], t[1], t[2]]).collect();
    let triangles_path = out_dir.join(format!("{stem}.triangles.field"));
    std::fs::write(
        &triangles_path,
        file::encode(
            &Schema::new(vec![Scalar::Int; 3]),
            Domain::Rows(triangles.len() as u32),
            &triangle_words,
        ),
    )
    .map_err(|e| format!("{}: {e}", triangles_path.display()))?;
    files.push(triangles_path);
    // The base colour texture of the first material that has one.
    let mut texture = None;
    let image_index = model.json["materials"]
        .as_array()
        .into_iter()
        .flatten()
        .find_map(|m| m["pbrMetallicRoughness"]["baseColorTexture"]["index"].as_u64())
        .and_then(|t| model.json["textures"].get(t as usize)?["source"].as_u64());
    if let Some(image_index) = image_index
        && let Some(image) = model.json["images"].get(image_index as usize)
    {
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        let bytes = if let Some(uri) = image["uri"].as_str() {
            read_uri(dir, uri)?
        } else if let Some(view) = image["bufferView"].as_u64() {
            model.view(view as usize)?.0.to_vec()
        } else {
            Vec::new()
        };
        if bytes.starts_with(b"\x89PNG") {
            orla_formats::png::decode(&bytes).map_err(|e| format!("the texture: {e}"))?;
            let texture_path = out_dir.join(format!("{stem}.albedo.png"));
            std::fs::write(&texture_path, &bytes)
                .map_err(|e| format!("{}: {e}", texture_path.display()))?;
            files.push(texture_path.clone());
            texture = Some(texture_path);
        } else if !bytes.is_empty() {
            notes.push("the base colour texture is not PNG (JPEG, likely); convert it to PNG and it comes in".into());
        }
    }
    // The skin: what every vertex is bound to, what the joints are, and
    // every animation sampled into a clip.
    let mut clips = Vec::new();
    let joints = skeleton.as_ref().map(|s| s.nodes.len()).unwrap_or(0);
    if let Some(skeleton) = &skeleton {
        let weight_words: Vec<u32> = bounds
            .iter()
            .flat_map(|b| {
                [
                    b.joints[0],
                    b.joints[1],
                    b.joints[2],
                    b.joints[3],
                    b.weights[0].to_bits(),
                    b.weights[1].to_bits(),
                    b.weights[2].to_bits(),
                    b.weights[3].to_bits(),
                ]
            })
            .collect();
        let weights_path = out_dir.join(format!("{stem}.weights.field"));
        std::fs::write(
            &weights_path,
            file::encode(
                &Schema::new([vec![Scalar::Int; 4], vec![Scalar::Float; 4]].concat()),
                Domain::Rows(bounds.len() as u32),
                &weight_words,
            ),
        )
        .map_err(|e| format!("{}: {e}", weights_path.display()))?;
        files.push(weights_path);

        let skeleton_words: Vec<u32> = skeleton
            .parents
            .iter()
            .zip(&skeleton.binds)
            .flat_map(|(parent, bind)| {
                let mut row = vec![*parent as u32];
                row.extend(bind.iter().map(|v| v.to_bits()));
                row
            })
            .collect();
        let skeleton_path = out_dir.join(format!("{stem}.skeleton.field"));
        std::fs::write(
            &skeleton_path,
            file::encode(
                &Schema::new([vec![Scalar::Int], vec![Scalar::Float; 16]].concat()),
                Domain::Rows(joints as u32),
                &skeleton_words,
            ),
        )
        .map_err(|e| format!("{}: {e}", skeleton_path.display()))?;
        files.push(skeleton_path);

        for (index, animation) in model.json["animations"]
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
        {
            let (values, frames) = clip_of(&model, animation, skeleton, rate)?;
            let name: String = animation["name"]
                .as_str()
                .unwrap_or("")
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
                .collect();
            let name = if name.trim_matches('-').is_empty() {
                format!("clip{index}")
            } else {
                name.trim_matches('-').to_lowercase()
            };
            let clip_path = out_dir.join(format!("{stem}.{name}.clip.field"));
            std::fs::write(
                &clip_path,
                file::encode(
                    &Schema::new(vec![Scalar::Float; 10]),
                    Domain::Rows((frames * joints) as u32),
                    &values.iter().map(|v| v.to_bits()).collect::<Vec<u32>>(),
                ),
            )
            .map_err(|e| format!("{}: {e}", clip_path.display()))?;
            files.push(clip_path);
            clips.push((name, frames));
        }
    }
    Ok(Imported {
        vertices: points.len(),
        triangles: triangles.len(),
        texture,
        notes,
        bounds: (lo, hi),
        joints,
        clips,
        rate,
        files,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A quad of two triangles, positions and uvs, indices as u16, in a
    /// data URI; one node scales it by two and lifts it by one.
    fn quad_gltf() -> String {
        let mut bin = Vec::new();
        for p in [
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ] {
            bin.extend(p.iter().flat_map(|v| v.to_le_bytes()));
        }
        for t in [[0.0f32, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]] {
            bin.extend(t.iter().flat_map(|v| v.to_le_bytes()));
        }
        for i in [0u16, 1, 2, 0, 2, 3] {
            bin.extend(i.to_le_bytes());
        }
        let uri = format!(
            "data:application/octet-stream;base64,{}",
            orla_formats::base64::base64(&bin)
        );
        format!(
            r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],
            "nodes":[{{"mesh":0,"scale":[2,2,2],"translation":[0,1,0]}}],
            "meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"TEXCOORD_0":1}},"indices":2}}]}}],
            "buffers":[{{"byteLength":{},"uri":"{uri}"}}],
            "bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":48}},{{"buffer":0,"byteOffset":48,"byteLength":32}},{{"buffer":0,"byteOffset":80,"byteLength":12}}],
            "accessors":[{{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3"}},{{"bufferView":1,"componentType":5126,"count":4,"type":"VEC2"}},{{"bufferView":2,"componentType":5123,"count":6,"type":"SCALAR"}}]}}"#,
            bin.len()
        )
    }

    /// Two vertices on a chain of two joints, the upper one turning a
    /// quarter circle over a second: the skin, the skeleton and the clip
    /// a scene reads.
    fn rigged_gltf() -> String {
        let mut bin = Vec::new();
        // positions
        for p in [[0.0f32, 0.0, 0.0], [0.0, 2.0, 0.0]] {
            bin.extend(p.iter().flat_map(|v| v.to_le_bytes()));
        }
        let joints_at = bin.len();
        for j in [[0u16, 0, 0, 0], [1, 0, 0, 0]] {
            bin.extend(j.iter().flat_map(|v| v.to_le_bytes()));
        }
        let weights_at = bin.len();
        for w in [[1.0f32, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]] {
            bin.extend(w.iter().flat_map(|v| v.to_le_bytes()));
        }
        let indices_at = bin.len();
        for i in [0u16, 1, 0] {
            bin.extend(i.to_le_bytes());
        }
        let binds_at = bin.len();
        // Column-major: the identity, then a lift of one undone.
        for m in [
            [
                1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0,
            ],
        ] {
            bin.extend(m.iter().flat_map(|v| v.to_le_bytes()));
        }
        let times_at = bin.len();
        for t in [0.0f32, 1.0] {
            bin.extend(t.to_le_bytes());
        }
        let turns_at = bin.len();
        let half = std::f32::consts::FRAC_PI_4;
        for q in [[0.0f32, 0.0, 0.0, 1.0], [half.sin(), 0.0, 0.0, half.cos()]] {
            bin.extend(q.iter().flat_map(|v| v.to_le_bytes()));
        }
        let uri = format!(
            "data:application/octet-stream;base64,{}",
            orla_formats::base64::base64(&bin)
        );
        format!(
            r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0,1]}}],
            "nodes":[{{"mesh":0,"skin":0,"translation":[9,9,9]}},
                     {{"children":[2]}},
                     {{"translation":[0,1,0]}}],
            "skins":[{{"joints":[1,2],"inverseBindMatrices":4}}],
            "meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"JOINTS_0":1,"WEIGHTS_0":2}},"indices":3}}]}}],
            "animations":[{{"name":"Bend","channels":[{{"sampler":0,"target":{{"node":2,"path":"rotation"}}}}],
                            "samplers":[{{"input":5,"output":6,"interpolation":"LINEAR"}}]}}],
            "buffers":[{{"byteLength":{},"uri":"{uri}"}}],
            "bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":24}},
                           {{"buffer":0,"byteOffset":{joints_at},"byteLength":16}},
                           {{"buffer":0,"byteOffset":{weights_at},"byteLength":32}},
                           {{"buffer":0,"byteOffset":{indices_at},"byteLength":6}},
                           {{"buffer":0,"byteOffset":{binds_at},"byteLength":128}},
                           {{"buffer":0,"byteOffset":{times_at},"byteLength":8}},
                           {{"buffer":0,"byteOffset":{turns_at},"byteLength":32}}],
            "accessors":[{{"bufferView":0,"componentType":5126,"count":2,"type":"VEC3"}},
                         {{"bufferView":1,"componentType":5123,"count":2,"type":"VEC4"}},
                         {{"bufferView":2,"componentType":5126,"count":2,"type":"VEC4"}},
                         {{"bufferView":3,"componentType":5123,"count":3,"type":"SCALAR"}},
                         {{"bufferView":4,"componentType":5126,"count":2,"type":"MAT4"}},
                         {{"bufferView":5,"componentType":5126,"count":2,"type":"SCALAR"}},
                         {{"bufferView":6,"componentType":5126,"count":2,"type":"VEC4"}}]}}"#,
            bin.len()
        )
    }

    /// What a model carries that these rows do not is said, not dropped.
    #[test]
    fn what_is_left_out_is_named() {
        let dir = std::env::temp_dir().join(format!("orla-gltf-notes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let text = quad_gltf()
            .replace(
                r#""attributes":{"POSITION":0,"TEXCOORD_0":1},"indices":2"#,
                r#""attributes":{"POSITION":0,"TEXCOORD_0":1},"indices":2,"targets":[{"POSITION":0}]"#,
            )
            .replace(
                r#""nodes":[{"mesh":0"#,
                r#""materials":[{"pbrMetallicRoughness":{"baseColorTexture":{"index":0}}},{"pbrMetallicRoughness":{"baseColorTexture":{"index":1}}}],
                "nodes":[{"mesh":0"#,
            );
        std::fs::write(dir.join("quad.gltf"), text).unwrap();
        let imported = import(&dir.join("quad.gltf"), &dir.join("out"), 30.0).unwrap();
        assert!(
            imported
                .notes
                .iter()
                .any(|note| note.contains("morph targets")),
            "{:?}",
            imported.notes
        );
        assert!(
            imported
                .notes
                .iter()
                .any(|note| note.contains("2 materials with a base colour texture")),
            "{:?}",
            imported.notes
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_rig_imports_as_a_skeleton_a_binding_and_a_clip() {
        let dir = std::env::temp_dir().join(format!("orla-gltf-rig-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("arm.gltf"), rigged_gltf()).unwrap();
        let out = dir.join("out");
        let imported = import(&dir.join("arm.gltf"), &out, 30.0).unwrap();
        assert_eq!(imported.joints, 2);
        assert_eq!(imported.clips, vec![("bend".to_owned(), 31)]);
        assert!(imported.notes.is_empty(), "{:?}", imported.notes);

        let read = |name: &str| {
            let bytes = std::fs::read(out.join(name)).unwrap();
            let data = file::decode(&bytes).unwrap();
            (data.values().to_vec(), data.domain())
        };
        // A skinned mesh is placed by its joints, so the node's own
        // translation of nine is not in the positions.
        let (vertices, _) = read("arm.vertices.field");
        assert_eq!(f32::from_bits(vertices[1]), 0.0, "the first vertex");
        assert_eq!(f32::from_bits(vertices[9]), 2.0, "the second vertex");

        // The second vertex hangs off the second joint.
        let (weights, domain) = read("arm.weights.field");
        assert_eq!(domain, Domain::Rows(2));
        assert_eq!(weights[8], 1, "the joint it is bound to");
        assert_eq!(f32::from_bits(weights[12]), 1.0, "how much of it");

        // The root has no parent; the second joint hangs off it.
        let (skeleton, domain) = read("arm.skeleton.field");
        assert_eq!(domain, Domain::Rows(2));
        assert_eq!(skeleton[0] as i32, -1);
        assert_eq!(skeleton[17] as i32, 0);

        // The clip: the joint at rest to start, a quarter turn about x
        // by the last frame, and the halfway frame halfway round.
        let (clip, domain) = read("arm.bend.clip.field");
        assert_eq!(domain, Domain::Rows(31 * 2));
        let pose = |frame: usize, joint: usize| -> Vec<f32> {
            clip[(frame * 2 + joint) * 10..][..10]
                .iter()
                .map(|w| f32::from_bits(*w))
                .collect()
        };
        assert_eq!(pose(0, 1)[..3], [0.0, 1.0, 0.0], "where the joint rests");
        assert!(
            pose(0, 1)[3..7]
                .iter()
                .zip([0.0, 0.0, 0.0, 1.0])
                .all(|(a, b)| (a - b).abs() < 1e-5)
        );
        let end = pose(30, 1);
        let quarter = std::f32::consts::FRAC_PI_4;
        assert!((end[3] - quarter.sin()).abs() < 1e-4, "{end:?}");
        assert!((end[6] - quarter.cos()).abs() < 1e-4, "{end:?}");
        // Halfway is halfway round, which is what slerp is for.
        let middle = pose(15, 1);
        let eighth = std::f32::consts::FRAC_PI_8;
        assert!((middle[3] - eighth.sin()).abs() < 1e-3, "{middle:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_quad_imports_transformed_with_computed_normals() {
        let dir = std::env::temp_dir().join(format!("orla-gltf-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("quad.gltf"), quad_gltf()).unwrap();
        let imported = import(&dir.join("quad.gltf"), &dir.join("out"), 30.0).unwrap();
        assert_eq!((imported.vertices, imported.triangles), (4, 2));
        assert!(
            imported.texture.is_none() && imported.notes.is_empty(),
            "{:?}",
            imported.notes
        );
        assert_eq!(imported.bounds, ([0.0, 1.0, 0.0], [2.0, 3.0, 0.0]));
        let vertices = file::decode(&std::fs::read(&imported.files[0]).unwrap()).unwrap();
        let words: Vec<f32> = vertices
            .values()
            .iter()
            .map(|w| f32::from_bits(*w))
            .collect();
        // Vertex 2: position (2, 3, 0), normal +z from the winding, uv (1, 0).
        assert_eq!(&words[16..24], &[2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
        let triangles = file::decode(&std::fs::read(&imported.files[1]).unwrap()).unwrap();
        assert_eq!(&*triangles.values(), &[0, 1, 2, 0, 2, 3]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
