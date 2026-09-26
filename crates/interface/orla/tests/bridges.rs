//! The bridges are clients outside the workspace, each with its own
//! runner: a page driven headless through Chrome's DevTools protocol, and
//! a Godot project run with `-s`. Each runs when its tools are on this
//! machine and is skipped with a note otherwise.

mod common;

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use orla_formats::png;

/// The workspace root, from this crate.
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// A tool at one of its usual paths, if this machine has it.
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

/// Orla built for a page (`wasm32-unknown-unknown`) in its own target
/// directory, with `wasm-bindgen`'s glue: the glue's path, or why there is
/// none (no such target, or no `wasm-bindgen`).
fn page_module() -> Result<PathBuf, String> {
    let sysroot = Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .map_err(|e| e.to_string())?;
    let sysroot = PathBuf::from(String::from_utf8_lossy(&sysroot.stdout).trim());
    if !sysroot.join("lib/rustlib/wasm32-unknown-unknown").exists() {
        return Err("no wasm32-unknown-unknown target".into());
    }
    let Some(bindgen) = tool(&["wasm-bindgen"]) else {
        return Err(
            "no wasm-bindgen (cargo install wasm-bindgen-cli at the version Cargo.lock holds)"
                .into(),
        );
    };
    let target = root().join("target/web");
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let built = Command::new(cargo)
        .current_dir(root())
        .args(["build", "--release", "-p", "orla-ffi"])
        .args(["--target", "wasm32-unknown-unknown", "--target-dir"])
        .arg(&target)
        .output()
        .expect("cargo runs");
    assert!(
        built.status.success(),
        "orla-ffi does not build for a page:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let page = target.join("page");
    let glued = Command::new(bindgen)
        .args(["--target", "web", "--keep-lld-exports", "--no-typescript"])
        .arg("--out-dir")
        .arg(&page)
        .arg(target.join("wasm32-unknown-unknown/release/orla_ffi.wasm"))
        .output()
        .expect("wasm-bindgen runs");
    assert!(
        glued.status.success(),
        "wasm-bindgen:\n{}",
        String::from_utf8_lossy(&glued.stderr)
    );
    Ok(page.join("orla_ffi.js"))
}

/// Runs a bridge's runner and reports its output. Runners print a line per
/// check and end with `RESULT: ok`.
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
    // One of its checks, so an `ok` without running is no pass.
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
    match page_module() {
        Ok(glue) => {
            command.arg("--glue").arg(glue);
            passed(
                command,
                "the web bridge",
                "play.html keys: F5 and Tab reached the play",
            );
        }
        Err(why) => {
            eprintln!("in the page: skipped, {why}");
            // Nothing stale stands in for what was not built.
            command.arg("--glue").arg("nothing-built");
            passed(command, "the web bridge", "voxel: a click places a voxel");
        }
    }
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

/// The bridge talks to a running session: it pushes files, pulls rows and
/// a picture, watches a node, and sets a parameter.
#[test]
fn the_blender_bridge_passes_its_own_test() {
    let Some(blender) = tool(&[
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "blender",
    ]) else {
        eprintln!("skipped: no Blender on this machine");
        return;
    };
    let dir = std::env::temp_dir().join(format!("orla-blender-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut session = Command::new(env!("CARGO_BIN_EXE_orla"))
        .current_dir(root())
        .arg("serve")
        .arg("examples/mesh/scene.osn")
        .args(["--listen", "127.0.0.1:0"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stderr = BufReader::new(session.stderr.take().unwrap());
    let port = stderr
        .lines()
        .map_while(Result::ok)
        .find_map(|line| {
            line.strip_prefix("listening on 127.0.0.1:")
                .map(str::to_owned)
        })
        .expect("the session names its port");
    let mut command = Command::new(blender);
    command
        .current_dir(root())
        .args(["-b", "--python", "bridges/blender/test_bridge.py", "--"])
        .args(["--port", &port])
        .arg("--out")
        .arg(&dir);
    let checked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        passed(command, "the Blender bridge", "pull: ok");
    }));
    let _ = session.kill();
    let _ = session.wait();
    let _ = std::fs::remove_dir_all(&dir);
    if let Err(panic) = checked {
        std::panic::resume_unwind(panic);
    }
}

/// Both ways a rig comes in, the Blender bridge writing rows and `orla
/// import` reading the same rig's glTF, pose the mesh alike. Both come
/// from one rig in one headless Blender and are compared as pictures,
/// which ignore vertex order.
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

    // What it pushed, against the one table every rig row is held to.
    let pushed_rows = common::rows_keep_the_contract(&dir.join("pushed"));
    assert_eq!(
        pushed_rows.len(),
        6,
        "five kinds, two of them clips: {pushed_rows:?}"
    );
    let live = common::rows_keep_the_contract(&dir.join("live"));
    assert!(
        live.iter()
            .any(|(kind, name)| kind == "clip.field" && name.ends_with("pose.clip.field")),
        "Live pushes the pose it stands on: {live:?}"
    );

    // The importer is another crate's binary, built beside this one by
    // `cargo test --workspace`.
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
                 palette <- palette.result; skeleton <- skeleton; }}\n\
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
    // Where the arm is and how much it covers: two pictures of one pose
    // agree on both, a rest pose and a bent one do not.
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
    // The shading agrees away from the silhouette, where half a pixel
    // flips arm and sky.
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

/// The way out: a rig's rows become a GLB that Blender opens with its
/// bones, weights and clips.
#[test]
fn a_rig_orla_wrote_opens_in_blender() {
    let Some(blender) = tool(&[
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "blender",
    ]) else {
        eprintln!("skipped: no Blender on this machine");
        return;
    };
    let exporter = Path::new(env!("CARGO_BIN_EXE_orla"))
        .parent()
        .unwrap()
        .join("orla-export");
    if !exporter.exists() {
        eprintln!("skipped: orla-export is not built beside orla");
        return;
    }
    let dir = std::env::temp_dir().join(format!("orla-rig-out-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let assets = root().join("examples/skin/assets");
    let glb = dir.join("arm.glb");
    let rows: Vec<PathBuf> = [
        "vertices",
        "triangles",
        "weights",
        "skeleton",
        "swing.clip",
        "curl.clip",
    ]
    .iter()
    .map(|name| assets.join(format!("arm.{name}.field")))
    .collect();
    let wrote = Command::new(&exporter)
        .args(&rows)
        .arg("--out")
        .arg(&glb)
        .args(["--rate", "30"])
        .output()
        .unwrap();
    let said = String::from_utf8_lossy(&wrote.stdout).to_string();
    assert!(
        wrote.status.success(),
        "{said}{}",
        String::from_utf8_lossy(&wrote.stderr)
    );
    assert!(said.contains("a skin and 2 clip(s)"), "{said}");

    let mut command = Command::new(blender);
    command
        .current_dir(root())
        .args(["-b", "--python", "bridges/blender/test_back.py", "--"])
        .arg("--glb")
        .arg(&glb);
    passed(
        command,
        "the rig Orla wrote",
        "opened: 1734 vertices, 2 bones",
    );
    let _ = std::fs::remove_dir_all(&dir);
}
