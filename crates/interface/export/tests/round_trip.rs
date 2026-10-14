//! What the exporter writes, the importer and the decoders read back.

use std::path::Path;

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-export-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Writes rows of `layout` as a `.field` file at `path`.
fn rows(path: &Path, layout: &str, values: &[f32]) {
    let schema: entelechy_host::Schema = layout.parse().unwrap();
    let width = layout.split(',').count();
    let words: Vec<u32> = layout
        .split(',')
        .cycle()
        .zip(values)
        .map(|(scalar, v)| {
            if scalar == "i" {
                *v as i32 as u32
            } else {
                v.to_bits()
            }
        })
        .collect();
    let domain = entelechy_host::Domain::Rows((values.len() / width) as u32);
    std::fs::write(
        path,
        entelechy_formats::file::encode(&schema, domain, &words),
    )
    .unwrap();
}

/// A unit cube, four vertices a face so each keeps its face's normal,
/// and a checker to paint it: `crate.vertices.field`,
/// `crate.triangles.field` and `crate.albedo.png` in `dir`.
fn cube(dir: &Path) {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    for axis in 0..3 {
        for side in [-1.0f32, 1.0] {
            let mut normal = [0.0f32; 3];
            normal[axis] = side;
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            let base = (vertices.len() / 8) as f32;
            for (a, b) in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)] {
                let mut position = [normal[0] * 0.5, normal[1] * 0.5, normal[2] * 0.5];
                position[u] = a;
                position[v] = b;
                vertices.extend(position);
                vertices.extend(normal);
                vertices.extend([a + 0.5, b + 0.5]);
            }
            triangles.extend([base, base + 1.0, base + 2.0, base, base + 2.0, base + 3.0]);
        }
    }
    rows(
        &dir.join("crate.vertices.field"),
        "f,f,f,f,f,f,f,f",
        &vertices,
    );
    rows(&dir.join("crate.triangles.field"), "i,i,i", &triangles);
    let checker: Vec<[f32; 4]> = (0..16)
        .map(|i| {
            let dark = (i % 4 + i / 4) % 2 == 0;
            if dark {
                [0.3, 0.2, 0.1, 1.0]
            } else {
                [0.8, 0.6, 0.3, 1.0]
            }
        })
        .collect();
    std::fs::write(
        dir.join("crate.albedo.png"),
        entelechy_formats::png::encode(4, 4, &checker),
    )
    .unwrap();
}

/// A two-joint arm as the five kinds of rig row, in `dir` as `arm.*`: a
/// square tube two units tall, the lower half bound to the root, the
/// upper to the joint at its middle, the band between blended; and two
/// clips of 33 frames, a swing of the root and a curl of the joint.
fn arm(dir: &Path) {
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
    let identity = |y: f32| {
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, y, 0.0, 1.0,
        ]
    };
    let mut skeleton = vec![-1.0];
    skeleton.extend(identity(0.0));
    skeleton.push(0.0);
    skeleton.extend(identity(-1.0));
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

#[test]
fn the_crate_leaves_as_glb_and_comes_back_the_same() {
    let dir = scratch("glb");
    let assets = dir.join("assets");
    std::fs::create_dir_all(&assets).unwrap();
    cube(&assets);
    let out = dir.join("crate.glb");
    entelechy_export::gltf::export(
        &assets.join("crate.vertices.field"),
        &assets.join("crate.triangles.field"),
        Some(&assets.join("crate.albedo.png")),
        &out,
    )
    .unwrap();
    let imported = entelechy_import::gltf::import(&out, &dir.join("back"), 30.0).unwrap();
    assert_eq!((imported.vertices, imported.triangles), (24, 12));
    assert!(
        imported.texture.is_some() && imported.notes.is_empty(),
        "{:?}",
        imported.notes
    );
    let before = std::fs::read(assets.join("crate.vertices.field")).unwrap();
    let after = std::fs::read(&imported.files[0]).unwrap();
    assert_eq!(
        before, after,
        "the vertex rows survived the round trip bit for bit"
    );
    let before = std::fs::read(assets.join("crate.triangles.field")).unwrap();
    let after = std::fs::read(&imported.files[1]).unwrap();
    assert_eq!(before, after);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_tone_leaves_as_wav_and_decodes_back() {
    let dir = scratch("sound");
    // A 440 Hz tone at half amplitude, half a second, stereo, 22050 Hz.
    let rate = 22050;
    let samples: Vec<f32> = (0..rate / 2)
        .flat_map(|i| {
            let v = 0.5 * (i as f32 / rate as f32 * 440.0 * std::f32::consts::TAU).sin();
            [v, v]
        })
        .collect();
    let name = "tone.wav";
    let path = dir.join(name);
    let words: Vec<u32> = samples.iter().map(|v| v.to_bits()).collect();
    std::fs::write(&path, entelechy_formats::wav::encode(&words, 2, rate)).unwrap();
    let imported = entelechy_import::audio::import(&path, &dir.join("back")).unwrap();
    assert_eq!((imported.rate, imported.channels), (rate, 2), "{name}");
    assert!(
        imported.frames >= rate as usize / 2 && imported.frames < rate as usize,
        "{name}: {} frames",
        imported.frames
    );
    let (field, _) = entelechy_formats::read(&imported.file, None).unwrap();
    let left: Vec<f32> = field
        .storage()
        .iter()
        .step_by(2)
        .map(|w| f32::from_bits(*w))
        .collect();
    let peak = left.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(peak > 0.4 && peak < 0.6, "{name}: peak {peak}");
    let start = imported.frames / 2;
    let window = &left[start..start + 2205];
    let crossings = window
        .windows(2)
        .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
        .count();
    assert!(
        (84..=92).contains(&crossings),
        "{name}: {crossings} crossings"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A rig round-trips: mesh, weights and joints bit for bit, clips within
/// a quaternion's resampling rounding.
#[test]
fn a_rig_leaves_as_glb_and_comes_back_posing_the_same() {
    let dir = scratch("rig");
    let assets = dir.join("assets");
    std::fs::create_dir_all(&assets).unwrap();
    arm(&assets);
    let out = dir.join("arm.glb");
    let path = |name: &str| assets.join(format!("arm.{name}.field"));
    entelechy_export::gltf::export_rig(
        &path("vertices"),
        &path("triangles"),
        &[],
        &[&path("weights"), &path("skeleton")],
        &[&path("swing.clip"), &path("curl.clip")],
        30.0,
        &out,
        &entelechy_export::gltf::Options::default(),
    )
    .unwrap();
    let back = dir.join("back");
    let imported = entelechy_import::gltf::import(&out, &back, 30.0).unwrap();
    assert_eq!(imported.joints, 2);
    assert_eq!(
        imported.clips,
        vec![("swing".to_owned(), 33), ("curl".to_owned(), 33)]
    );
    assert!(imported.notes.is_empty(), "{:?}", imported.notes);

    // Rows passed through untouched are exact.
    for name in ["vertices", "triangles", "weights", "skeleton"] {
        assert_eq!(
            std::fs::read(path(name)).unwrap(),
            std::fs::read(back.join(format!("arm.{name}.field"))).unwrap(),
            "{name} survived the round trip bit for bit"
        );
    }
    // The clips are resampled, so they come within a rounding.
    for name in ["swing", "curl"] {
        let before =
            entelechy_formats::file::decode(&std::fs::read(path(&format!("{name}.clip"))).unwrap())
                .unwrap();
        let after = entelechy_formats::file::decode(
            &std::fs::read(back.join(format!("arm.{name}.clip.field"))).unwrap(),
        )
        .unwrap();
        assert_eq!(before.domain(), after.domain(), "{name}");
        let worst = before
            .storage()
            .iter()
            .zip(after.storage())
            .map(|(a, b)| (f32::from_bits(*a) - f32::from_bits(*b)).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "{name} drifted by {worst}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A GLB's JSON chunk.
fn document(glb: &Path) -> serde_json::Value {
    let bytes = std::fs::read(glb).unwrap();
    let length = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    serde_json::from_slice(&bytes[20..20 + length]).unwrap()
}

/// The texture is sampled as asked: blended with mipmaps by default,
/// the nearest texel for flat colors.
#[test]
fn a_texture_is_sampled_linear_or_nearest_as_asked() {
    let dir = scratch("filter");
    cube(&dir);
    let texture = dir.join("crate.albedo.png");
    let sampler = |filter: entelechy_export::gltf::Filter, name: &str| {
        let out = dir.join(name);
        entelechy_export::gltf::export_rig(
            &dir.join("crate.vertices.field"),
            &dir.join("crate.triangles.field"),
            &[&texture],
            &[],
            &[],
            30.0,
            &out,
            &entelechy_export::gltf::Options {
                filter,
                joints: Vec::new(),
                double_sided: false,
            },
        )
        .unwrap();
        document(&out)["samplers"][0].clone()
    };
    let linear = sampler(entelechy_export::gltf::Filter::Linear, "linear.glb");
    assert_eq!(
        (linear["magFilter"].as_u64(), linear["minFilter"].as_u64()),
        (Some(9729), Some(9987))
    );
    let nearest = sampler(entelechy_export::gltf::Filter::Nearest, "nearest.glb");
    assert_eq!(
        (nearest["magFilter"].as_u64(), nearest["minFilter"].as_u64()),
        (Some(9728), Some(9728))
    );
}

/// Joints take the names given, one a skeleton row; a count that is not
/// the skeleton's is refused, and a rig comes back the same whatever its
/// joints are called.
#[test]
fn joints_take_the_names_given_in_skeleton_order() {
    let dir = scratch("joints");
    let assets = dir.join("assets");
    std::fs::create_dir_all(&assets).unwrap();
    arm(&assets);
    let path = |name: &str| assets.join(format!("arm.{name}.field"));
    let export = |joints: &[&str], out: &Path| {
        entelechy_export::gltf::export_rig(
            &path("vertices"),
            &path("triangles"),
            &[],
            &[&path("weights"), &path("skeleton")],
            &[&path("swing.clip")],
            30.0,
            out,
            &entelechy_export::gltf::Options {
                filter: entelechy_export::gltf::Filter::Linear,
                joints: joints.iter().map(|name| (*name).to_owned()).collect(),
                double_sided: false,
            },
        )
    };
    let out = dir.join("arm.glb");
    export(&["shoulder", "elbow"], &out).unwrap();
    let nodes = document(&out)["nodes"].clone();
    let names: Vec<&str> = nodes
        .as_array()
        .unwrap()
        .iter()
        .skip(1)
        .map(|node| node["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["shoulder", "elbow"]);
    let refused = export(&["shoulder"], &dir.join("short.glb")).unwrap_err();
    assert!(
        refused.contains("1 joint name(s) for a skeleton of 2"),
        "{refused}"
    );
    let blank = export(&["shoulder", " "], &dir.join("blank.glb")).unwrap_err();
    assert!(blank.contains("joint 1 has no name"), "{blank}");
    let imported = entelechy_import::gltf::import(&out, &dir.join("back"), 30.0).unwrap();
    assert_eq!(imported.joints, 2);
    for name in ["weights", "skeleton"] {
        assert_eq!(
            std::fs::read(path(name)).unwrap(),
            std::fs::read(dir.join("back").join(format!("arm.{name}.field"))).unwrap(),
            "{name} survived the named round trip bit for bit"
        );
    }
}

/// Vertices with a tangent carry it as the mesh's tangents, and PNGs
/// named for a normal map and an occlusion map are the material's; the
/// importer reads the position, normal and uv back bit for bit.
#[test]
fn tangents_and_maps_go_with_the_material() {
    let dir = scratch("maps");
    cube(&dir);
    let plain =
        entelechy_formats::file::decode(&std::fs::read(dir.join("crate.vertices.field")).unwrap())
            .unwrap();
    let mut with_tangent = Vec::new();
    for row in plain.storage().chunks_exact(8) {
        with_tangent.extend(row.iter().map(|w| f32::from_bits(*w)));
        with_tangent.extend([1.0, 0.0, 0.0, -1.0]);
    }
    rows(
        &dir.join("lit.vertices.field"),
        &["f"; 12].join(","),
        &with_tangent,
    );
    let flat = vec![[0.5f32, 0.5, 1.0, 1.0]; 4];
    std::fs::write(
        dir.join("lit.normal.png"),
        entelechy_formats::png::encode(2, 2, &flat),
    )
    .unwrap();
    std::fs::write(
        dir.join("lit.occlusion.png"),
        entelechy_formats::png::encode(2, 2, &[[0.7, 0.7, 0.7, 1.0]; 4]),
    )
    .unwrap();
    let out = dir.join("lit.glb");
    entelechy_export::gltf::export_rig(
        &dir.join("lit.vertices.field"),
        &dir.join("crate.triangles.field"),
        &[
            &dir.join("crate.albedo.png"),
            &dir.join("lit.normal.png"),
            &dir.join("lit.occlusion.png"),
        ],
        &[],
        &[],
        30.0,
        &out,
        &entelechy_export::gltf::Options::default(),
    )
    .unwrap();
    let json = document(&out);
    let primitive = &json["meshes"][0]["primitives"][0];
    let tangent = &json["accessors"][primitive["attributes"]["TANGENT"].as_u64().unwrap() as usize];
    assert_eq!(
        (tangent["type"].as_str(), tangent["count"].as_u64()),
        (Some("VEC4"), Some(24))
    );
    let material = &json["materials"][0];
    assert_eq!(
        material["pbrMetallicRoughness"]["baseColorTexture"]["index"].as_u64(),
        Some(0)
    );
    assert_eq!(material["normalTexture"]["index"].as_u64(), Some(1));
    assert_eq!(material["occlusionTexture"]["index"].as_u64(), Some(2));
    assert_eq!(json["images"].as_array().map(Vec::len), Some(3));
    // A map is blended whatever the base color asks.
    assert_eq!(json["textures"][1]["sampler"].as_u64(), Some(1));

    let imported = entelechy_import::gltf::import(&out, &dir.join("back"), 30.0).unwrap();
    assert_eq!((imported.vertices, imported.triangles), (24, 12));
    let before = std::fs::read(dir.join("crate.vertices.field")).unwrap();
    let after = std::fs::read(&imported.files[0]).unwrap();
    assert_eq!(
        before, after,
        "position, normal and uv came back bit for bit"
    );

    // An ORM map: the metallic-roughness texture, and the occlusion where
    // none is given apart.
    std::fs::write(
        dir.join("lit.orm.png"),
        entelechy_formats::png::encode(2, 2, &[[0.7, 0.5, 0.0, 1.0]; 4]),
    )
    .unwrap();
    let orm = dir.join("orm.glb");
    entelechy_export::gltf::export_rig(
        &dir.join("lit.vertices.field"),
        &dir.join("crate.triangles.field"),
        &[
            &dir.join("crate.albedo.png"),
            &dir.join("lit.normal.png"),
            &dir.join("lit.orm.png"),
        ],
        &[],
        &[],
        30.0,
        &orm,
        &entelechy_export::gltf::Options::default(),
    )
    .unwrap();
    let json = document(&orm);
    let pbr = &json["materials"][0]["pbrMetallicRoughness"];
    assert_eq!(pbr["metallicRoughnessTexture"]["index"].as_u64(), Some(2));
    assert_eq!(
        (
            pbr["metallicFactor"].as_f64(),
            pbr["roughnessFactor"].as_f64()
        ),
        (Some(1.0), Some(1.0))
    );
    assert_eq!(
        json["materials"][0]["occlusionTexture"]["index"].as_u64(),
        Some(2)
    );
    assert_eq!(json["images"].as_array().map(Vec::len), Some(3));

    // Two normal maps are one too many.
    std::fs::copy(dir.join("lit.normal.png"), dir.join("other.normal.png")).unwrap();
    let twice = entelechy_export::gltf::export_rig(
        &dir.join("lit.vertices.field"),
        &dir.join("crate.triangles.field"),
        &[&dir.join("lit.normal.png"), &dir.join("other.normal.png")],
        &[],
        &[],
        30.0,
        &dir.join("twice.glb"),
        &entelechy_export::gltf::Options::default(),
    );
    assert!(twice.unwrap_err().contains("a second normal map"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// A material asked double-sided says so, with maps or without; one not
/// asked says nothing (glTF's default: one side).
#[test]
fn a_material_is_double_sided_when_asked() {
    let dir = scratch("double_sided");
    cube(&dir);
    let texture = dir.join("crate.albedo.png");
    let material = |textures: &[&Path], double_sided: bool, name: &str| {
        let out = dir.join(name);
        entelechy_export::gltf::export_rig(
            &dir.join("crate.vertices.field"),
            &dir.join("crate.triangles.field"),
            textures,
            &[],
            &[],
            30.0,
            &out,
            &entelechy_export::gltf::Options {
                double_sided,
                ..Default::default()
            },
        )
        .unwrap();
        document(&out)["materials"][0].clone()
    };
    assert_eq!(
        material(&[&texture], true, "leaf.glb")["doubleSided"],
        serde_json::json!(true)
    );
    assert_eq!(
        material(&[], true, "plain.glb")["doubleSided"],
        serde_json::json!(true)
    );
    assert!(material(&[&texture], false, "one.glb")["doubleSided"].is_null());
    let _ = std::fs::remove_dir_all(&dir);
}
