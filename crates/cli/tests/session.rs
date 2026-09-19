use std::path::{Path, PathBuf};
use std::process::Command;

use orla::session::{Event, Session};
use orla_graph::EvaluateError;
use orla_runtime::Context;

fn example() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/grayscale")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orla-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("kernels")).unwrap();
    for entry in std::fs::read_dir(example().join("kernels")).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), dir.join("kernels").join(entry.file_name())).unwrap();
    }
    std::fs::copy(example().join("scene.osn"), dir.join("scene.osn")).unwrap();
    dir
}

fn published(events: &[Event]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Published { id, .. } => Some(id.to_string()),
            _ => None,
        })
        .collect()
}

fn rejected(events: &[Event]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Rejected { error, .. } => Some(error.clone()),
            _ => None,
        })
        .collect()
}

const INVERT: &str = "kernel image.invert@1 { requires { image.read; image.write; } input source: Image; output result: Image; compute { let c = sample(source); return vec4(1.0 - c.r, 1.0 - c.g, 1.0 - c.b, c.a); } }";

#[test]
fn a_saved_edit_is_published_and_a_broken_edit_keeps_the_running_kernel() {
    let dir = scratch("session");
    let mut session = Session::open(&dir.join("scene.osn"), None, 1 << 20);
    let events = session.sync();
    assert!(events.contains(&Event::SceneLoaded));
    assert_eq!(
        published(&events),
        ["image.gradient@1", "image.grayscale@1", "image.pixelate@1"]
    );
    let report = session.evaluate(&Context::default()).unwrap();
    assert_eq!(report.executed.len(), 3);
    let out = dir.join("out.ppm");
    session.export(report.output, &out).unwrap();
    assert!(
        std::fs::read(&out)
            .unwrap()
            .starts_with(b"P6\n128 128\n255\n")
    );
    let pixel = session
        .runtime()
        .host()
        .read_pixel(report.output, 100, 20)
        .unwrap();
    assert!((pixel[0] - pixel[1]).abs() < 1e-5 && (pixel[1] - pixel[2]).abs() < 1e-5);

    assert!(session.sync().is_empty(), "nothing changed on disk");

    // Edit: turn grayscale into a red-channel pass-through.
    let grayscale = dir.join("kernels/grayscale.orla");
    let source = std::fs::read_to_string(&grayscale).unwrap();
    std::fs::write(
        &grayscale,
        source.replace(
            "return mix(c, vec4(g, g, g, c.a), amount);",
            "return vec4(c.r, 0.0, 0.0, c.a);",
        ),
    )
    .unwrap();
    let events = session.sync();
    assert_eq!(published(&events), ["image.grayscale@1"]);
    let report = session.evaluate(&Context::default()).unwrap();
    assert_eq!(
        report
            .executed
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        ["gray", "blocks"]
    );
    let pixel = session
        .runtime()
        .host()
        .read_pixel(report.output, 100, 20)
        .unwrap();
    assert_eq!(pixel[1], 0.0);
    assert!(pixel[0] > 0.5);
    let generation = session
        .runtime()
        .kernel(&"image.grayscale@1".parse().unwrap())
        .unwrap()
        .generation();

    // Break it: the previous version keeps running and the diagnostic has a position.
    std::fs::write(
        &grayscale,
        "kernel image.grayscale@1 {\n  requires { image.write; }\n  output result: Image;\n  compute { return oops; }\n}",
    )
    .unwrap();
    let events = session.sync();
    let errors = rejected(&events);
    assert_eq!(errors.len(), 1);
    assert!(
        errors[0].contains("4:20: `oops` is not defined"),
        "{}",
        errors[0]
    );
    assert_eq!(
        session
            .runtime()
            .kernel(&"image.grayscale@1".parse().unwrap())
            .unwrap()
            .generation(),
        generation
    );
    let report = session.evaluate(&Context::default()).unwrap();
    assert!(report.executed.is_empty());

    // Remove it: the kernel retires and the graph reports the missing binding.
    std::fs::remove_file(&grayscale).unwrap();
    let events = session.sync();
    assert!(
        matches!(&events[..], [Event::Retired { id, .. }] if id.to_string() == "image.grayscale@1"),
        "{events:?}"
    );
    let error = session.evaluate(&Context::default()).unwrap_err();
    assert!(
        matches!(error, EvaluateError::UnknownKernel { .. }),
        "{error}"
    );

    // Two documents claiming one id: the second is rejected, the first stays.
    std::fs::write(
        &grayscale,
        "kernel image.pixelate@1 { requires { image.write; } output result: Image; compute { return vec4(0.0, 0.0, 0.0, 1.0); } }",
    )
    .unwrap();
    let events = session.sync();
    let errors = rejected(&events);
    assert!(errors[0].contains("already published"), "{}", errors[0]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn scene_edits_inline_kernels_and_externals_flow_through_the_session() {
    let dir = scratch("scene");
    let mut session = Session::open(&dir.join("scene.osn"), None, 1 << 20);
    session.sync();
    session.evaluate(&Context::default()).unwrap();

    let scene = dir.join("scene.osn");
    let text = std::fs::read_to_string(&scene).unwrap();
    std::fs::write(&scene, text.replace("size = 8;", "size = 32;")).unwrap();
    assert_eq!(session.sync(), [Event::SceneLoaded]);
    let report = session.evaluate(&Context::default()).unwrap();
    assert_eq!(
        report
            .executed
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        ["blocks"]
    );

    std::fs::write(&scene, "output r <- nowhere.x;\nnode {").unwrap();
    let events = session.sync();
    assert!(
        matches!(&events[..], [Event::SceneRejected(_)]),
        "{events:?}"
    );
    let report = session.evaluate(&Context::default()).unwrap();
    assert!(
        report.executed.is_empty(),
        "previous graph is still evaluated"
    );

    // A self-contained scene: an inline named kernel, an anonymous one, and
    // an external input. Nothing is used from the kernels directory.
    let photo = dir.join("photo.ppm");
    session.export(report.output, &photo).unwrap();
    std::fs::write(
        &scene,
        format!(
            "input photo: Image;
node inv: {}
node dim: kernel {{
    requires {{ image.read; image.write; }}
    input source: Image <- inv.result;
    param amount: Float = 0.5;
    output result: Image;
    compute {{ let c = sample(source); return vec4(c.r * amount, c.g * amount, c.b * amount, c.a); }}
}}
output result <- dim.result;",
            INVERT.replacen("input source: Image;", "input source: Image <- photo;", 1)
        ),
    )
    .unwrap();
    session.bind_external("photo", &photo).unwrap();
    let events = session.sync();
    assert_eq!(published(&events), ["image.invert@1", "scene.dim@1"]);
    assert!(
        events.iter().any(
            |e| matches!(e, Event::Retired { id, .. } if id.to_string() == "image.gradient@1")
        ),
        "{events:?}"
    );
    let report = session.evaluate(&Context::default()).unwrap();
    let inverted = session
        .runtime()
        .host()
        .read_pixel(report.output, 5, 5)
        .unwrap();
    assert!(inverted[3] == 1.0 && inverted[0] <= 0.5);

    // Editing only the anonymous kernel republishes only it.
    let text = std::fs::read_to_string(&scene).unwrap();
    std::fs::write(
        &scene,
        text.replace("amount: Float = 0.5", "amount: Float = 0.25"),
    )
    .unwrap();
    let events = session.sync();
    assert_eq!(published(&events), ["scene.dim@1"]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_binary_checks_and_runs() {
    let dir = scratch("binary");
    let orla = env!("CARGO_BIN_EXE_orla");
    let output = Command::new(orla)
        .args(["check", "--ir"])
        .arg(dir.join("kernels/grayscale.orla"))
        .arg(dir.join("scene.osn"))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    assert!(stdout.contains("kernel image.grayscale@1"), "{stdout}");
    assert!(stdout.contains("host image.sample input0"), "{stdout}");
    assert!(
        stdout.contains("graph `scene` with 3 node(s), 1 use(s)"),
        "{stdout}"
    );

    std::fs::write(
        dir.join("kernels/bad.orla"),
        "kernel bad@1 {\n  compute { return 1; }\n}",
    )
    .unwrap();
    let output = Command::new(orla)
        .arg("check")
        .arg(dir.join("kernels/bad.orla"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("bad.orla:1:"), "{stdout}");
    std::fs::remove_file(dir.join("kernels/bad.orla")).unwrap();

    let out = dir.join("result.ppm");
    let output = Command::new(orla)
        .arg("run")
        .arg(dir.join("scene.osn"))
        .arg("--out")
        .arg(&out)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(
        stderr.contains("executed [source, gray, blocks]"),
        "{stderr}"
    );
    assert!(
        std::fs::read(&out)
            .unwrap()
            .starts_with(b"P6\n128 128\n255\n")
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn every_example_replay_passes_and_failures_are_reported() {
    let orla = env!("CARGO_BIN_EXE_orla");
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
    let mut replays = Vec::new();
    for example in std::fs::read_dir(&examples).unwrap().flatten() {
        let tests = example.path().join("tests");
        let Ok(entries) = std::fs::read_dir(&tests) else {
            continue;
        };
        for entry in entries.flatten() {
            if entry.path().extension().is_some_and(|ext| ext == "replay") {
                replays.push(entry.path());
            }
        }
    }
    assert!(!replays.is_empty(), "examples ship replay tests");
    let output = Command::new(orla)
        .arg("replay")
        .args(&replays)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("blinker.replay: ok"), "{stdout}");

    let dir = scratch("replay");
    std::fs::write(
        dir.join("wrong.replay"),
        "scene scene.osn\nframe 0.0\nexpect 0 0 = 9.0 9.0 9.0\n",
    )
    .unwrap();
    let output = Command::new(orla)
        .arg("replay")
        .arg(dir.join("wrong.replay"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("line 3: pixel (0, 0) is"), "{stdout}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn every_example_document_checks() {
    let orla = env!("CARGO_BIN_EXE_orla");
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
    let mut documents = Vec::new();
    for example in std::fs::read_dir(&examples).unwrap().flatten() {
        let scene = example.path().join("scene.osn");
        if scene.exists() {
            documents.push(scene);
        }
    }
    let output = Command::new(orla)
        .arg("check")
        .args(&documents)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
}
