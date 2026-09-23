//! The bridges are clients outside the workspace, each with a runner of
//! its own: a page driven headless through Chrome's DevTools protocol,
//! and a Godot project run with `-s`. Both are run from here when the
//! tools they need are on this machine, and skipped with a word when
//! they are not, so a change to the protocol that breaks a bridge
//! breaks this rather than waiting for someone to open one.

use std::path::{Path, PathBuf};
use std::process::Command;

use orla::png;

/// The workspace root, from this crate.
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A tool this machine has, by the paths it is usually at, or nothing.
fn tool(paths: &[&str]) -> Option<PathBuf> {
    paths
        .iter()
        .map(PathBuf::from)
        .find(|path| path.exists())
        .or_else(|| {
            let found = Command::new("which").arg(paths[0]).output().ok()?;
            found
                .status
                .success()
                .then(|| PathBuf::from(String::from_utf8_lossy(&found.stdout).trim()))
        })
}

/// Runs a bridge's own runner and says what it said. The runners print
/// a line for every thing they checked and end with `RESULT: ok`.
fn passed(mut command: Command, what: &str, marker: &str) {
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("{what}: {error}"));
    let said = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.status.success() && said.contains("RESULT: ok"),
        "{what} did not pass:\n{said}"
    );
    // One of the things it checked, so a runner that said `ok` without
    // running is not taken for a pass.
    assert!(
        said.contains(marker),
        "{what} did not check `{marker}`:\n{said}"
    );
}

#[test]
fn the_web_bridge_passes_its_own_test() {
    let Some(node) = tool(&["node", "/usr/local/bin/node", "/opt/homebrew/bin/node"]) else {
        eprintln!("skipped: no node on this machine");
        return;
    };
    let Some(chrome) = tool(&[
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "google-chrome",
        "chromium",
    ]) else {
        eprintln!("skipped: no Chrome on this machine");
        return;
    };
    let mut command = Command::new(node);
    command
        .current_dir(root())
        .arg("bridges/web/test.mjs")
        .arg("--orla")
        .arg(env!("CARGO_BIN_EXE_orla"))
        .arg("--chrome")
        .arg(chrome);
    passed(command, "the web bridge", "voxel: a click places a voxel");
}

#[test]
fn the_godot_bridge_passes_its_own_test() {
    let Some(godot) = tool(&["godot", "/usr/local/bin/godot"]) else {
        eprintln!("skipped: no Godot on this machine");
        return;
    };
    let mut command = Command::new(godot);
    command
        .current_dir(root())
        .args(["--headless", "--path", "bridges/godot", "-s", "test.gd"])
        .args(["--", "--orla", env!("CARGO_BIN_EXE_orla")]);
    passed(
        command,
        "the Godot bridge",
        "mesh: the torus is an ArrayMesh",
    );
}

/// The two ways a rig comes in — the Blender bridge writing the rows
/// itself, and `orla import` reading the glTF the same rig exports —
/// must pose the mesh the same way. Both are built from one rig in one
/// headless Blender, then posed at the same moment and compared as
/// pictures, which is the only comparison that does not care that the
/// two write their vertices in different orders.
#[test]
fn a_rig_pushed_from_blender_poses_as_its_gltf_does() {
    let Some(blender) = tool(&[
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "blender",
    ]) else {
        eprintln!("skipped: no Blender on this machine");
        return;
    };
    let dir = std::env::temp_dir().join(format!("orla-rig-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let mut command = Command::new(blender);
    command
        .current_dir(root())
        .args(["-b", "--python", "bridges/blender/test_rig.py", "--"])
        .arg("--out")
        .arg(&dir);
    passed(command, "the Blender rig", "pushed Arm.swing.clip.field");

    // The converter is another crate's binary; `cargo test --workspace`
    // builds it beside this one.
    let importer = Path::new(env!("CARGO_BIN_EXE_orla"))
        .parent()
        .unwrap()
        .join("orla-import");
    if !importer.exists() {
        eprintln!("skipped: orla-import is not built beside orla");
        return;
    }
    let imported = dir.join("imported");
    let import = Command::new(&importer)
        .arg(dir.join("arm.glb"))
        .arg("--out")
        .arg(&imported)
        .args(["--rate", "24"])
        .output()
        .unwrap();
    let said = String::from_utf8_lossy(&import.stdout).to_string();
    assert!(import.status.success(), "{said}");
    assert!(said.contains("2 joints"), "{said}");

    // One scene per set of rows, posed at the same moment.
    let picture = |assets: &Path, stem: &str, name: &str| -> png::Decoded {
        let scene = assets.join("pose.osn");
        std::fs::write(
            &scene,
            format!(
                "use \"{root}/examples/skin/kernels/\";\n\
                 use \"{root}/packages/skin.orlapkg/kernels/skin.orla\";\n\
                 input mesh: Rows<Point> = \"{stem}.vertices.field\";\n\
                 input faces: Triangles = \"{stem}.triangles.field\";\n\
                 input weights: Rows<Bound> = \"{stem}.weights.field\";\n\
                 input skeleton: Skeleton = \"{stem}.skeleton.field\";\n\
                 input swing: Clip = \"{stem}.swing.clip.field\";\n\
                 node clock: kernel {{ requires {{ field.write; }} param at: Float = 0.35; \
                 output result: Rows<Float>; domain = 1; compute {{ return at; }} }}\n\
                 node poses: skin.sample@1 {{ skeleton <- skeleton; clip <- swing; \
                 at <- clock.result; rate = 24.0; }}\n\
                 node palette: skin.palette@1 {{ poses <- poses.result; skeleton <- skeleton; }}\n\
                 node posed: skin.apply@1 {{ mesh <- mesh; weights <- weights; \
                 palette <- palette.result; }}\n\
                 node project: arm.project@1 {{ mesh <- posed.result; spin = 0.0; }}\n\
                 node raster: native.rasterize@1 {{ vertices <- project.result; \
                 triangles <- faces; domain = vec2(200, 120); }}\n\
                 node view: arm.shade@1 {{ surface <- raster.surface; domain = vec2(200, 120); }}\n\
                 output result <- view.result;\n",
                root = root().canonicalize().unwrap().display(),
            ),
        )
        .unwrap();
        let out = dir.join(name);
        let run = Command::new(env!("CARGO_BIN_EXE_orla"))
            .arg("run")
            .arg(&scene)
            .args(["--backend", "cpu"])
            .arg("--save")
            .arg(format!("view.result={}", out.display()))
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "{}{}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
        png::decode(&std::fs::read(&out).unwrap()).unwrap()
    };
    let pushed = picture(&dir.join("pushed"), "Arm", "pushed.png");
    let read = picture(&imported, "arm", "imported.png");

    assert_eq!((pushed.width, pushed.height), (read.width, read.height));
    // Where the arm is, and how much of the picture it covers: two
    // pictures of one pose agree about both, whatever order their
    // vertices were written in, and a rest pose against a bent one does
    // not.
    let arm = |picture: &png::Decoded| -> (usize, f32, f32) {
        let mut count = 0usize;
        let (mut x, mut y) = (0.0f32, 0.0f32);
        for (index, pixel) in picture.samples.chunks(picture.channels).enumerate() {
            if pixel[0] > 0.3 {
                count += 1;
                x += (index as u32 % picture.width) as f32;
                y += (index as u32 / picture.width) as f32;
            }
        }
        let n = count.max(1) as f32;
        (count, x / n, y / n)
    };
    let (lit, ax, ay) = arm(&pushed);
    let (also, bx, by) = arm(&read);
    assert!(
        lit > (pushed.width * pushed.height / 100) as usize,
        "the arm covers {lit} pixels of the pushed rig's picture"
    );
    let bigger = lit.max(also) as f32;
    assert!(
        (lit as f32 - also as f32).abs() / bigger < 0.02,
        "the arm covers {lit} pixels pushed and {also} imported"
    );
    assert!(
        (ax - bx).abs() < 1.0 && (ay - by).abs() < 1.0,
        "the arm is at ({ax:.2}, {ay:.2}) pushed and ({bx:.2}, {by:.2}) imported"
    );
    // The shading agrees too, away from the silhouette, where a pixel
    // is either arm or sky and half a pixel of difference flips it.
    let differ = pushed
        .samples
        .iter()
        .zip(&read.samples)
        .filter(|(a, b)| (*a - *b).abs() > 0.03)
        .count();
    let samples = pushed.samples.len();
    assert!(
        differ * 50 < samples,
        "{differ} of {samples} samples differ between the pushed rig and its glTF"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
