//! What the exporter writes, the importer and the decoders read back.

use std::path::Path;

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("orla-export-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn the_crate_leaves_as_glb_and_comes_back_the_same() {
    let dir = scratch("glb");
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/model/assets");
    let out = dir.join("crate.glb");
    orla_export::gltf::export(
        &assets.join("crate.vertices.field"),
        &assets.join("crate.triangles.field"),
        Some(&assets.join("crate.albedo.png")),
        &out,
    )
    .unwrap();
    let imported = orla_import::gltf::import(&out, &dir.join("back"), 30.0).unwrap();
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
    let pcm = orla_export::sound::Pcm {
        samples: &samples,
        channels: 2,
        rate,
    };
    let name = "tone.wav";
    let path = dir.join(name);
    std::fs::write(&path, orla_export::sound::wav(&pcm)).unwrap();
    let imported = orla_import::audio::import(&path, &dir.join("back")).unwrap();
    assert_eq!((imported.rate, imported.channels), (rate, 2), "{name}");
    assert!(
        imported.frames >= rate as usize / 2 && imported.frames < rate as usize,
        "{name}: {} frames",
        imported.frames
    );
    let (field, _) = orla_session::Session::load_field(&imported.file, None).unwrap();
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

/// A rig goes out and comes back: the mesh, what every vertex hangs
/// off and the joints bit for bit, and the clips to the rounding a
/// quaternion takes when it is sampled again.
#[test]
fn a_rig_leaves_as_glb_and_comes_back_posing_the_same() {
    let dir = scratch("rig");
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/skin/assets");
    let out = dir.join("arm.glb");
    let path = |name: &str| assets.join(format!("arm.{name}.field"));
    orla_export::gltf::export_rig(
        &path("vertices"),
        &path("triangles"),
        None,
        &[&path("weights"), &path("skeleton")],
        &[&path("swing.clip"), &path("curl.clip")],
        30.0,
        &out,
    )
    .unwrap();
    let back = dir.join("back");
    let imported = orla_import::gltf::import(&out, &back, 30.0).unwrap();
    assert_eq!(imported.joints, 2);
    assert_eq!(
        imported.clips,
        vec![("swing".to_owned(), 33), ("curl".to_owned(), 33)]
    );
    assert!(imported.notes.is_empty(), "{:?}", imported.notes);

    // The rows that pass through untouched do so exactly.
    for name in ["vertices", "triangles", "weights", "skeleton"] {
        assert_eq!(
            std::fs::read(path(name)).unwrap(),
            std::fs::read(back.join(format!("arm.{name}.field"))).unwrap(),
            "{name} survived the round trip bit for bit"
        );
    }
    // The clips are sampled again on the way back, so they come within
    // a rounding of where they were.
    for name in ["swing", "curl"] {
        let before =
            orla_runtime::file::decode(&std::fs::read(path(&format!("{name}.clip"))).unwrap())
                .unwrap();
        let after = orla_runtime::file::decode(
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
