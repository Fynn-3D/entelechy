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

const INVERT: &str = "kernel image.invert@1 { requires { field.read; field.write; } input source: Image; output result: Image; compute { let c = sample(source); return vec4(1.0 - c.r, 1.0 - c.g, 1.0 - c.b, c.a); } }";

#[test]
fn a_saved_edit_is_published_and_a_broken_edit_keeps_the_running_kernel() {
    let dir = scratch("session");
    let mut session = Session::open(&dir.join("scene.osn"), None, 1 << 20);
    session.set_fusion(false);
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
        "kernel image.grayscale@1 {\n  requires { field.write; }\n  output result: Image;\n  compute { return oops; }\n}",
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
        "kernel image.pixelate@1 { requires { field.write; } output result: Image; compute { return vec4(0.0, 0.0, 0.0, 1.0); } }",
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
    requires {{ field.read; field.write; }}
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
    assert!(stdout.contains("host sample input0"), "{stdout}");
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
    // A scene is checked through its libraries: the bad Kernel fails it.
    let output = Command::new(orla)
        .arg("check")
        .arg(dir.join("scene.osn"))
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
        stderr.contains("executed [gray+source, blocks]"),
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
    // The same replays on the JIT: native code, the interpreter's words.
    let output = Command::new(orla)
        .arg("replay")
        .args(["--backend", "jit"])
        .args(&replays)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("voxel.replay: ok"), "{stdout}");

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

#[test]
fn the_graph_command_prints_the_plan() {
    let dir = scratch("graph");
    let orla = env!("CARGO_BIN_EXE_orla");
    let output = Command::new(orla)
        .arg("graph")
        .arg(dir.join("scene.osn"))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    assert!(
        stdout.contains("node gray: image.grayscale@1 {"),
        "{stdout}"
    );
    assert!(stdout.contains("// plan: 2 group(s)"), "{stdout}");
    assert!(
        stdout.contains("//   gray + source: fused.gray@1 vec2(128, 128)"),
        "{stdout}"
    );
    assert!(stdout.contains("//   blocks: image.pixelate@1"), "{stdout}");
    let output = Command::new(orla)
        .args(["graph", "--no-fusion"])
        .arg(dir.join("scene.osn"))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("// plan: 3 group(s)"), "{stdout}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_background_republish_keeps_the_running_kernel_until_the_worker_is_done() {
    let dir = scratch("background");
    let mut session = Session::open(&dir.join("scene.osn"), None, 1 << 20);
    session.set_fusion(false);
    let events = session.sync();
    assert_eq!(published(&events).len(), 3);
    session.set_background(true);
    let first = session.evaluate(&Context::default()).unwrap();
    let before = session
        .runtime()
        .host()
        .read_pixel(first.output, 100, 20)
        .unwrap();
    let generation = session
        .runtime()
        .kernel(&"image.grayscale@1".parse().unwrap())
        .unwrap()
        .generation();

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
    // The edit is noticed at once; the publish lands on a later sync, and
    // every frame in between still runs the old version.
    let mut events = session.sync();
    assert!(
        matches!(events.first(), Some(Event::Preparing { id, .. }) if id.to_string() == "image.grayscale@1"),
        "{events:?}"
    );
    let mut frames = 0;
    while published(&events).is_empty() {
        let report = session.evaluate(&Context::default()).unwrap();
        let pixel = session
            .runtime()
            .host()
            .read_pixel(report.output, 100, 20)
            .unwrap();
        assert_eq!(pixel, before, "the running version keeps its output");
        assert_eq!(
            session
                .runtime()
                .kernel(&"image.grayscale@1".parse().unwrap())
                .unwrap()
                .generation(),
            generation
        );
        frames += 1;
        assert!(frames < 5000, "the worker never delivered");
        std::thread::sleep(std::time::Duration::from_millis(1));
        events = session.sync();
    }
    assert_eq!(published(&events), ["image.grayscale@1"]);
    let report = session.evaluate(&Context::default()).unwrap();
    let pixel = session
        .runtime()
        .host()
        .read_pixel(report.output, 100, 20)
        .unwrap();
    assert_eq!(pixel[1], 0.0);
    assert!(pixel[0] > 0.5);

    // A broken edit is rejected by the compiler at once, off no thread.
    std::fs::write(
        &grayscale,
        "kernel image.grayscale@1 {\n  requires { field.write; }\n  output result: Image;\n  compute { return oops; }\n}",
    )
    .unwrap();
    let events = session.sync();
    assert!(rejected(&events)[0].contains("`oops` is not defined"));
    assert!(session.evaluate(&Context::default()).is_ok());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn libraries_share_declarations_through_use_and_refuse_conflicts() {
    let dir = std::env::temp_dir().join(format!("orla-shared-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("lib")).unwrap();
    std::fs::write(
        dir.join("lib/types.orla"),
        "struct Cell { kind: Int; ttl: Int; }\ntype Board = Grid<Cell>;\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("lib/rules.orla"),
        "use \"types.orla\";\nkernel t.seed@1 { requires { field.write; } output result: Board; compute { return Cell { kind: 1, ttl: int(coord().x) }; } }\n",
    )
    .unwrap();
    // The scene's inline Kernel sees the declarations through `use` as well.
    std::fs::write(
        dir.join("scene.osn"),
        "use \"lib/\";\nnode seed: t.seed@1 { domain = vec2(4, 2); }\nnode show: kernel { requires { field.read; field.write; } input board: Board <- seed.result; output result: Image; compute { let c = sample(board); return vec4(float(c.ttl) / 4.0, float(c.kind), 0.0, 1.0); } }\noutput result <- show.result;\n",
    )
    .unwrap();
    let mut session = Session::open(&dir.join("scene.osn"), None, 1 << 20);
    let events = session.sync();
    assert_eq!(rejected(&events), Vec::<String>::new(), "{events:?}");
    let report = session.evaluate(&Context::default()).unwrap();
    let pixel = session.read_pixel(report.output, 3, 0).unwrap();
    assert_eq!(pixel, [0.75, 1.0, 0.0, 1.0]);

    // A second declaration of the same struct with other fields is refused.
    std::fs::write(
        dir.join("lib/other.orla"),
        "struct Cell { kind: Int; }\nkernel t.other@1 { requires { field.write; } output result: Rows<Cell>; compute { return Cell { kind: 0 }; } }\n",
    )
    .unwrap();
    let events = session.sync();
    assert!(
        events
            .iter()
            .any(|e| e.to_string().contains("declared differently")),
        "{events:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn functions_are_shared_through_use_and_conflicts_are_refused() {
    let dir = scratch("functions");
    std::fs::write(
        dir.join("kernels/math.orla"),
        "fn twice(x: Float) -> Float { return x * 2.0; }
fn peek(f: Image, c: Vec2) -> Vec4 { return at(f, c); }
",
    )
    .unwrap();
    std::fs::write(
        dir.join("scene.osn"),
        "use \"kernels/\";
node seed: life.seed@1 { domain = vec2(8, 8); }
node view: kernel { requires { field.read; field.write; } input cells: Image <- seed.result; output result: Image;
    compute { let c = peek(cells, coord()); return vec4(twice(c.r), c.g, c.b, 1.0); } }
output result <- view.result;
",
    )
    .unwrap();
    let orla = env!("CARGO_BIN_EXE_orla");
    let output = Command::new(orla)
        .arg("check")
        .arg(dir.join("scene.osn"))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    // The same name from a second file is a conflict, named by both files.
    std::fs::write(
        dir.join("kernels/more.orla"),
        "fn twice(x: Float) -> Float { return x + x; }\n",
    )
    .unwrap();
    let output = Command::new(orla)
        .arg("check")
        .arg(dir.join("scene.osn"))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!output.status.success(), "{stdout}");
    assert!(
        stdout.contains("fn `twice` is declared by both") && stdout.contains("more.orla"),
        "{stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn inputs_read_files_the_host_watches_and_outputs_save_as_files() {
    let dir = scratch("assets");
    let orla = env!("CARGO_BIN_EXE_orla");
    // A 2x1 picture as a file, read by an input and copied by a Kernel.
    let picture = |left: [u8; 3]| {
        let mut bytes = b"P6\n2 1\n255\n".to_vec();
        bytes.extend_from_slice(&left);
        bytes.extend_from_slice(&[0, 0, 255]);
        bytes
    };
    std::fs::write(dir.join("photo.ppm"), picture([255, 0, 0])).unwrap();
    std::fs::write(
        dir.join("assets.osn"),
        "input photo: Image = \"photo.ppm\";
node copy: kernel { requires { field.read; field.write; } input source: Image <- photo; output result: Image; compute { return sample(source); } }
output result <- copy.result;
",
    )
    .unwrap();
    let mut session = Session::open(&dir.join("assets.osn"), None, 1 << 20);
    let events = session.sync();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::Loaded { name, .. } if name == "photo")),
        "{events:?}"
    );
    let report = session.evaluate(&Context::default()).unwrap();
    assert_eq!(session.read_pixel(report.output, 0, 0).unwrap()[0], 1.0);
    // The file changes: the next sync reads it again and the picture follows.
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::write(dir.join("photo.ppm"), picture([0, 255, 0])).unwrap();
    let events = session.sync();
    assert!(
        events.iter().any(|e| matches!(e, Event::Loaded { .. })),
        "{events:?}"
    );
    let report = session.evaluate(&Context::default()).unwrap();
    assert_eq!(session.read_pixel(report.output, 0, 0).unwrap()[1], 1.0);
    // A file that is not a Field is rejected with its path, and the
    // previous value stays bound.
    std::fs::write(dir.join("photo.ppm"), b"not a picture").unwrap();
    let events = session.sync();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::Rejected { path, .. } if path.ends_with("photo.ppm"))),
        "{events:?}"
    );
    assert!(session.evaluate(&Context::default()).is_ok());

    // The binary saves any node's output as a Field file and a picture.
    std::fs::write(dir.join("photo.ppm"), picture([255, 0, 0])).unwrap();
    let output = Command::new(orla)
        .arg("run")
        .arg(dir.join("assets.osn"))
        .arg("--save")
        .arg(format!("copy.result={}", dir.join("copy.field").display()))
        .arg("--save")
        .arg(format!("copy={}", dir.join("copy.ppm").display()))
        .arg("--out")
        .arg(dir.join("out.ppm"))
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(stderr.contains("loaded photo from"), "{stderr}");
    let (field, _) = Session::load_field(&dir.join("copy.field"), None).unwrap();
    assert_eq!(field.domain(), orla_runtime::Domain::Grid(2, 1));
    assert_eq!(f32::from_bits(field.words()[0]), 1.0);
    assert_eq!(
        std::fs::read(dir.join("copy.ppm")).unwrap(),
        picture([255, 0, 0])
    );

    // PNG comes in as pixels or as a Grid<Float> by the input's type, and
    // goes out as a file any viewer opens; WAV comes in as rows with its
    // rate reported.
    let pixels = vec![[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]];
    std::fs::write(dir.join("photo.png"), orla::png::encode(2, 1, &pixels)).unwrap();
    let (as_pixels, detail) = Session::load_field(&dir.join("photo.png"), Some("Image")).unwrap();
    assert!(
        as_pixels.schema().is_pixel() && detail.contains("Grid 2"),
        "{detail}"
    );
    let (as_gray, _) = Session::load_field(&dir.join("photo.png"), Some("Grid<Float>")).unwrap();
    assert_eq!(as_gray.schema().words(), 1);
    assert!((f32::from_bits(as_gray.words()[0]) - 0.2126).abs() < 1e-5);
    let sound: Vec<u32> = [0.0f32, 0.5, -0.5, 0.25]
        .iter()
        .map(|v| v.to_bits())
        .collect();
    std::fs::write(dir.join("hit.wav"), orla::wav::encode(&sound, 2, 22050)).unwrap();
    let (rows, detail) = Session::load_field(&dir.join("hit.wav"), Some("Rows<Vec2>")).unwrap();
    assert_eq!(rows.domain(), orla_runtime::Domain::Rows(2));
    assert!(detail.ends_with("at 22050 Hz"), "{detail}");
    let output = Command::new(orla)
        .arg("run")
        .arg(dir.join("assets.osn"))
        .arg("--save")
        .arg(format!("copy={}", dir.join("copy.png").display()))
        .arg("--out")
        .arg(dir.join("out.ppm"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let (again, _) = Session::load_field(&dir.join("copy.png"), Some("Image")).unwrap();
    assert_eq!(f32::from_bits(again.words()[0]), 1.0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_param_that_sizes_the_graph_lowers_it_again() {
    let dir = scratch("times");
    std::fs::write(
        dir.join("scene.osn"),
        "use \"kernels/\";
param level: Int = 1;
node seed: image.gradient@1 { domain = vec2(8, 8); }
node gray: image.grayscale@1 { times = level; carry source <- result; source <- seed.result; }
node blocks: image.pixelate@1 { source <- gray.result; }
output result <- blocks.result;
",
    )
    .unwrap();
    let mut session = Session::open(&dir.join("scene.osn"), None, 1 << 20);
    session.sync();
    assert_eq!(session.graph().unwrap().nodes.len(), 3);
    assert!(session.evaluate(&Context::default()).is_ok());
    session.set_param("level", orla_runtime::Value::Int(3));
    let events = session.sync();
    assert!(
        events.iter().any(|e| matches!(e, Event::SceneLoaded)),
        "{events:?}"
    );
    let ids: Vec<String> = session
        .graph()
        .unwrap()
        .nodes
        .iter()
        .map(|n| n.id.0.clone())
        .collect();
    assert_eq!(ids, ["seed", "gray/1", "gray/2", "gray/3", "blocks"]);
    let report = session
        .evaluate(&Context::default())
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(
        report.executed.iter().any(|n| n.0 == "gray/3"),
        "{:?}",
        report.executed
    );
    // A param that does not size the graph does not lower it again.
    session.set_param("nothing", orla_runtime::Value::Int(1));
    assert!(
        !session
            .sync()
            .iter()
            .any(|e| matches!(e, Event::SceneLoaded))
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn inputs_hold_text_as_rows_of_codes_and_follow_the_document() {
    let dir = scratch("texts");
    let write = |text: &str| {
        std::fs::write(
            dir.join("label.osn"),
            format!(
                "input label: Rows<Int> = text \"{text}\";
node count: kernel {{ requires {{ field.read; field.write; }} input codes: Rows<Int> <- label; output result: Rows<Int>; domain = 2; compute {{ return if element() == 0 {{ count(codes) }} else {{ at(codes, 1) }}; }} }}
output result <- count.result;
"
            ),
        )
        .unwrap();
    };
    write("Hi");
    let mut session = Session::open(&dir.join("label.osn"), None, 1 << 20);
    let events = session.sync();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::Text { name, chars: 2 } if name == "label")),
        "{events:?}"
    );
    session.pin(orla_graph::NodeId("count".into()));
    session.evaluate(&Context::default()).unwrap();
    let handle = session
        .cached(&orla_graph::NodeId("count".into()), "result")
        .unwrap();
    let host = session.runtime().host();
    assert_eq!(host.element(handle, 0).unwrap()[0], 2);
    assert_eq!(host.element(handle, 1).unwrap()[0], 'i' as u32);
    // The document changes its text: the next sync writes it again and
    // the rows follow, count included.
    std::thread::sleep(std::time::Duration::from_millis(20));
    write("Hey!");
    let events = session.sync();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::Text { chars: 4, .. })),
        "{events:?}"
    );
    session.evaluate(&Context::default()).unwrap();
    let handle = session
        .cached(&orla_graph::NodeId("count".into()), "result")
        .unwrap();
    let host = session.runtime().host();
    assert_eq!(host.element(handle, 0).unwrap()[0], 4);
    assert_eq!(host.element(handle, 1).unwrap()[0], 'e' as u32);
    // An empty text is refused where it is written.
    write("");
    let events = session.sync();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::SceneRejected(message) if message.contains("at least one character"))),
        "{events:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_binary_traces_one_step_of_a_scan_and_profiles_a_run() {
    let orla = env!("CARGO_BIN_EXE_orla");
    let dir = scratch("trace");
    // A tally over four cells of memory, six steps a frame: the first
    // evaluation leaves [2, 2, 1, 1]; the next one's step 5 finds cell 1
    // at 3, writes 4, and its `if` takes the branch with the product.
    std::fs::write(
        dir.join("kernels/tally.orla"),
        "kernel tally.step@1 {
    requires { field.read; field.write; }
    input ticks: Rows<Int>;
    memory ram: Rows<Int>;
    output result: Rows<Int>;
    scan with acc = 0 {
        let cell = index() % 4;
        let v = at(ram, cell) + 1;
        put(ram, cell, v);
        let big = if v > 1 { let product = v * 100; product } else { let none = 0; none };
        let kind = match cell { 0 | 1 => { let low = 1; low }, 3 => { let high = 3; high }, _ => 2 };
        return acc + v + kind - 1;
    }
}
",
    )
    .unwrap();
    std::fs::write(
        dir.join("tally.osn"),
        "use \"kernels/\";
node ticks: kernel { requires { field.write; } output result: Rows<Int>; domain = 6; compute { return 0; } }
node zeros: kernel { requires { field.write; } output result: Rows<Int>; domain = 4; compute { return 0; } }
node tally: tally.step@1 { ticks <- ticks.result; ram <~ tally.ram from zeros.result; }
output result <- tally.result;
",
    )
    .unwrap();
    let output = Command::new(orla)
        .args(["trace", "--node", "tally", "--element", "5"])
        .arg(dir.join("tally.osn"))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.starts_with("tally (tally.step@1), step 5"),
        "{stdout}"
    );
    for line in [
        "  cell: Int = 1",
        "  v: Int = 4",
        "  product: Int = 400 (0x190)",
        "  none: Int = - (not run)",
        "  big: Int = 400 (0x190)",
        "  low: Int = 1",
        "  high: Int = - (not run)",
        "  kind: Int = 1",
        "  -> result: Int = 21 (0x15)",
    ] {
        assert!(stdout.contains(line), "{line:?} in {stdout}");
    }
    // A step outside the frame is refused, not computed.
    let output = Command::new(orla)
        .args(["trace", "--node", "tally", "--element", "6"])
        .arg(dir.join("tally.osn"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("outside the domain"));
    // --profile names every executed node with its time.
    let output = Command::new(orla)
        .args(["run", "--profile", "--out"])
        .arg(dir.join("out.ppm"))
        .arg(dir.join("tally.osn"))
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(
        stderr.contains("  tally ") && stderr.contains(" ms"),
        "{stderr}"
    );
}

#[test]
fn help_lists_the_builtins_and_shows_one() {
    let orla = env!("CARGO_BIN_EXE_orla");
    let output = Command::new(orla).arg("help").output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    for name in [
        "floor",
        "atan2",
        "at_linear",
        "put",
        "key",
        "vec4",
        "add_wrap",
        "match",
    ] {
        assert_eq!(
            stdout.contains(&format!("  {name} ")),
            name != "match",
            "{name} in {stdout}"
        );
    }
    let output = Command::new(orla)
        .args(["help", "smoothstep"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("smoothstep(low: Float, high: Float, x: Vec3) -> Vec3"),
        "{stdout}"
    );
    assert!(
        stdout.contains("smoothstep(low: Vec2, high: Vec2, x: Vec2) -> Vec2"),
        "{stdout}"
    );
    let output = Command::new(orla)
        .args(["help", "sample"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Needs `field.read`") && stdout.contains("Only the entry"),
        "{stdout}"
    );
    let output = Command::new(orla).args(["help", "atan3"]).output().unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("did you mean") && stderr.contains("atan2"),
        "{stderr}"
    );
}
