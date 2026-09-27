use std::path::{Path, PathBuf};
use std::process::Command;

use orla_graph::EvaluateError;
use orla_kernel::Context;
use orla_session::{Event, Session};

fn example() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/grayscale")
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
            Event::Rejected { error, .. } => Some(error.to_string()),
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
    session.allow_write_to(&out).unwrap();
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

    // Edit: grayscale becomes a red-channel pass-through.
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

    // Broken: the previous version keeps running; the diagnostic has a
    // position.
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

    // Removed: the Kernel retires and the graph reports the missing binding.
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

    // Two documents claim one id: the second is rejected, the first stays.
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
    // A text reading a missing node is refused; the running graph goes on.
    let dangling = format!(
        "{}\noutput extra <- nowhere.result;\n",
        text.replace("size = 8;", "size = 32;")
    );
    std::fs::write(&scene, dangling).unwrap();
    let events = session.sync();
    assert!(
        matches!(&events[..], [Event::SceneRejected(why)] if why.message.contains("unknown source `nowhere.result`")),
        "{events:?}"
    );
    let report = session.evaluate(&Context::default()).unwrap();
    assert!(
        report.executed.is_empty(),
        "the graph running is still evaluated"
    );

    // A self-contained scene: an inline named Kernel, an anonymous one, and
    // an external input.
    let photo = dir.join("photo.ppm");
    session.allow_write_to(&photo).unwrap();
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

    // Editing only the anonymous Kernel republishes only it.
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
    // The IR shown is the IR a backend runs, block counting included.
    std::fs::write(
        dir.join("kernels/counted.orla"),
        "kernel t.counted@1 { requires { field.write; } param n: Int = 3; output result: Rows<Int>; compute { return loop n with a = 0 { a + 1 }; } }",
    )
    .unwrap();
    let counted = Command::new(orla)
        .args(["check", "--ir"])
        .arg(dir.join("kernels/counted.orla"))
        .output()
        .unwrap();
    let shown = String::from_utf8_lossy(&counted.stdout);
    assert!(counted.status.success(), "{shown}");
    assert!(
        shown.contains("work spend") && shown.contains("work look"),
        "{shown}"
    );
    assert!(
        !stdout.contains("work spend"),
        "grayscale has no loop: {stdout}"
    );
    std::fs::remove_file(dir.join("kernels/counted.orla")).unwrap();

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

/// `orla player|import|export` runs `orla-<tool>` beside `orla` with the
/// same arguments, and exits as it does; with none beside it or on `PATH`,
/// it says what it looked for.
#[test]
fn a_tool_beside_orla_runs_as_an_orla_command() {
    let dir = std::env::temp_dir().join(format!("orla-beside-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let orla = dir.join("orla");
    std::fs::copy(env!("CARGO_BIN_EXE_orla"), &orla).unwrap();
    // A stand-in for `orla-export`: `orla` itself, which answers `version`.
    std::fs::copy(env!("CARGO_BIN_EXE_orla"), dir.join("orla-export")).unwrap();
    let output = Command::new(&orla)
        .args(["export", "version"])
        .env("PATH", "")
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.starts_with("orla = "), "{stdout}");
    let output = Command::new(&orla)
        .args(["export", "check"])
        .env("PATH", "")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1), "its exit status is orla's");
    let output = Command::new(&orla)
        .arg("player")
        .env("PATH", "")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("runs `orla-player`, beside `orla` or on PATH"),
        "{stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `run` writes only the files named, each in the format its extension
/// names.
#[test]
fn run_writes_only_the_files_named_each_as_its_name_says() {
    let dir = scratch("run-writes");
    let orla = env!("CARGO_BIN_EXE_orla");
    let files = |dir: &Path| {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|name| name != "kernels" && name != "scene.osn")
            .collect();
        names.sort();
        names
    };
    let run = |args: &[&str]| {
        let output = Command::new(orla)
            .current_dir(&dir)
            .arg("run")
            .arg("scene.osn")
            .args(args)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        (output.status.success(), stderr)
    };

    let (ok, stderr) = run(&[]);
    assert!(ok, "{stderr}");
    assert!(
        stderr.contains("nothing written: name a file with `--out <file.png>`"),
        "{stderr}"
    );
    assert!(files(&dir).is_empty(), "{:?}", files(&dir));

    let (ok, stderr) = run(&["--save", "blocks.result=saved.png"]);
    assert!(ok, "{stderr}");
    assert_eq!(files(&dir), ["saved.png"]);

    let (ok, stderr) = run(&["--out", "frame.png"]);
    assert!(ok, "{stderr}");
    let png = std::fs::read(dir.join("frame.png")).unwrap();
    assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"), "a PNG");
    assert_eq!(
        png,
        std::fs::read(dir.join("saved.png")).unwrap(),
        "one encoding for both"
    );
    let (ok, stderr) = run(&["--out", "frame.ppm"]);
    assert!(ok, "{stderr}");
    assert!(
        std::fs::read(dir.join("frame.ppm"))
            .unwrap()
            .starts_with(b"P6\n128 128\n255\n")
    );

    let (ok, stderr) = run(&["--out", "frame.txt"]);
    assert!(!ok);
    assert!(
        stderr.contains("cannot write") && stderr.contains("use .field, .png"),
        "{stderr}"
    );
    assert_eq!(files(&dir), ["frame.png", "frame.ppm", "saved.png"]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn every_example_replay_passes_and_failures_are_reported() {
    let orla = env!("CARGO_BIN_EXE_orla");
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples");
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
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples");
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
    // frames in between run the old version.
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

    // A broken edit is rejected by the compiler at once.
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
    // The scene's inline Kernel sees the declarations through `use`.
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

    // A second declaration of the struct with other fields is refused.
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
    // The same name from a second file is a conflict naming both files.
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
    // A 2x1 picture file, read by an input and copied by a Kernel.
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
    // The file changes: the next sync rereads it and the picture follows.
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::write(dir.join("photo.ppm"), picture([0, 255, 0])).unwrap();
    let events = session.sync();
    assert!(
        events.iter().any(|e| matches!(e, Event::Loaded { .. })),
        "{events:?}"
    );
    let report = session.evaluate(&Context::default()).unwrap();
    assert_eq!(session.read_pixel(report.output, 0, 0).unwrap()[1], 1.0);
    // A file that is not a Field is rejected with its path; the previous
    // value stays bound.
    std::fs::write(dir.join("photo.ppm"), b"not a picture").unwrap();
    let events = session.sync();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::Rejected { path, .. } if path.ends_with("photo.ppm"))),
        "{events:?}"
    );
    assert!(session.evaluate(&Context::default()).is_ok());

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
    let (field, _) = orla_formats::read(&dir.join("copy.field"), None).unwrap();
    assert_eq!(field.domain(), orla_kernel::Domain::Grid(2, 1));
    assert_eq!(f32::from_bits(field.storage()[0]), 1.0);
    assert_eq!(
        std::fs::read(dir.join("copy.ppm")).unwrap(),
        picture([255, 0, 0])
    );

    // PNG reads as pixels or Grid<Float> by the input's type and writes as
    // a standard file; WAV reads as rows with its rate reported.
    let pixels = vec![[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]];
    std::fs::write(
        dir.join("photo.png"),
        orla_formats::png::encode(2, 1, &pixels),
    )
    .unwrap();
    let (as_pixels, detail) = orla_formats::read(&dir.join("photo.png"), Some("Image")).unwrap();
    assert!(
        as_pixels.schema().is_pixel() && detail.contains("Grid 2"),
        "{detail}"
    );
    let (as_gray, _) = orla_formats::read(&dir.join("photo.png"), Some("Grid<Float>")).unwrap();
    assert_eq!(as_gray.schema().value_words(), 1);
    assert!((f32::from_bits(as_gray.storage()[0]) - 0.2126).abs() < 1e-5);
    let sound: Vec<u32> = [0.0f32, 0.5, -0.5, 0.25]
        .iter()
        .map(|v| v.to_bits())
        .collect();
    std::fs::write(
        dir.join("hit.wav"),
        orla_formats::wav::encode(&sound, 2, 22050),
    )
    .unwrap();
    let (rows, detail) = orla_formats::read(&dir.join("hit.wav"), Some("Rows<Vec2>")).unwrap();
    assert_eq!(rows.domain(), orla_kernel::Domain::Rows(2));
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
    let (again, _) = orla_formats::read(&dir.join("copy.png"), Some("Image")).unwrap();
    assert_eq!(f32::from_bits(again.storage()[0]), 1.0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_param_that_sizes_the_graph_lowers_it_again() {
    let dir = scratch("times");
    std::fs::write(
        dir.join("scene.osn"),
        "use \"kernels/\";
param level: Int = 1;
param other: Int = 0;
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
    session
        .set_param("level", orla_kernel::Value::Int(3))
        .unwrap();
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
    // A param that does not size the graph does not relower it.
    session
        .set_param("other", orla_kernel::Value::Int(1))
        .unwrap();
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
    // The document's text changes: the next sync rewrites the rows, count
    // included.
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
    // An empty text is refused at its position.
    write("");
    let events = session.sync();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::SceneRejected(message) if message.message.contains("at least one character"))),
        "{events:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_binary_traces_one_step_of_a_scan_and_profiles_a_run() {
    let orla = env!("CARGO_BIN_EXE_orla");
    let dir = scratch("trace");
    // A tally over four memory cells, six steps a frame: the first
    // evaluation leaves [2, 2, 1, 1]; the next one's step 5 writes 4 to
    // cell 1 and takes the `if` branch with the product.
    std::fs::write(
        dir.join("kernels/tally.orla"),
        "kernel tally.step@1 {
    requires { field.read; field.write; }
    input ticks: Rows<Int>;
    memory ram: Rows<Int>;
    output result: Rows<Int>;
    fn bump(m: Rows<Int>, a: Int, v: Int) -> Int { let doubled = v * 2; put(m, a, v); return doubled; }
    scan with acc = 0 {
        let cell = index() % 4;
        let v = at(ram, cell) + 1;
        bump(ram, cell, v);
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
    // The first evaluation's step 5: `ram` as the evaluation begins it,
    // all zeros, which step 1 set to 1 at cell 1.
    for line in [
        "  cell: Int = 1",
        "  v: Int = 2",
        "  product: Int = 200 (0xC8)",
        "  none: Int = - (not run)",
        "  big: Int = 200 (0xC8)",
        "  low: Int = 1",
        "  high: Int = - (not run)",
        "  kind: Int = 1",
        "  call bump<ram> #1",
        "    doubled: Int = 4",
        "    -> result: Int = 4",
        "  -> result: Int = 11",
    ] {
        assert!(stdout.contains(line), "{line:?} in {stdout}");
    }
    // A step outside the frame is refused.
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

/// `orla help` is an index of sections; a section lists its items; a
/// name shows every item of it in full.
#[test]
fn help_lists_the_sections_their_items_and_shows_one() {
    let orla = env!("CARGO_BIN_EXE_orla");
    let output = Command::new(orla).arg("help").output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    for section in [
        "builtins", "kernel", "scene", "replay", "protocol", "commands",
    ] {
        assert!(
            stdout.contains(&format!("  {section} ")),
            "{section} in {stdout}"
        );
    }
    let listed = |section: &str| {
        let output = Command::new(orla).args(["help", section]).output().unwrap();
        String::from_utf8_lossy(&output.stdout).into_owned()
    };
    let builtins = listed("builtins");
    for name in [
        "floor",
        "atan2",
        "at_linear",
        "put",
        "key",
        "vec4",
        "add_wrap",
    ] {
        assert!(
            builtins.contains(&format!("  {name} ")),
            "{name} in {builtins}"
        );
    }
    let forms = format!("{}{}", listed("kernel"), listed("scene"));
    for name in ["match", "loop", "each", "scatter", "fill"] {
        assert!(forms.contains(&format!("  {name} ")), "{name} in {forms}");
    }
    // `replay` names a section and a command: the answer is both.
    let replay = listed("replay");
    assert!(
        replay.contains("  expect ") && replay.contains("orla replay <file.replay>"),
        "{replay}"
    );
    assert!(listed("protocol").contains("  publish "));
    assert!(listed("commands").contains("  serve "));
    let output = Command::new(orla).args(["help", "each"]).output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("[each n { body }]"), "{stdout}");
    assert!(stdout.contains("element by element"), "{stdout}");
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
        stderr.contains("did you mean") && stderr.contains("atan"),
        "{stderr}"
    );
    let output = Command::new(orla)
        .args(["help", "publish", "--json"])
        .output()
        .unwrap();
    let answer: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(answer["found"][0]["fields"][0]["name"], "source");
}

/// `pack` makes one file, `unpack` restores the directory, `package` lists
/// it, and a scene's `use` brings its Kernels in when versions fit.
#[test]
fn packages_pack_unpack_describe_and_refuse_a_newer_language() {
    let orla = env!("CARGO_BIN_EXE_orla");
    let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../packages/pixelfx.orlapkg");
    let dir = std::env::temp_dir().join(format!("orla-packages-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let archive = dir.join("pixelfx.orlapkg");
    let output = Command::new(orla)
        .arg("pack")
        .arg(&example)
        .arg(&archive)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("packed 3 file(s)"));
    let unpacked = dir.join("unpacked.orlapkg");
    let output = Command::new(orla)
        .arg("unpack")
        .arg(&archive)
        .arg(&unpacked)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    for file in ["package.osn", "kernels/pixelfx.orla", "graphs/nes.osn"] {
        assert_eq!(
            std::fs::read(example.join(file)).unwrap(),
            std::fs::read(unpacked.join(file)).unwrap(),
            "{file} came back as it went"
        );
    }
    let output = Command::new(orla)
        .arg("package")
        .arg(&unpacked)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    assert!(
        stdout.contains("package pixelfx@1 (needs language 1, ir 1): 1 kernel file(s), 1 graph preset(s), 0 other file(s)"),
        "{stdout}"
    );
    // A scene beside the unpacked package uses it by name through
    // ORLA_PACKAGES.
    let elsewhere = dir.join("elsewhere");
    std::fs::create_dir_all(&elsewhere).unwrap();
    std::fs::write(
        elsewhere.join("scene.osn"),
        "use \"unpacked.orlapkg\";\nnode source: kernel { requires { field.write; } output result: Image; domain = vec2(8, 8); compute { return vec4(0.5, 0.5, 0.5, 1.0); } }\nnode toned: pixelfx.palette@1 { source <- source.result; }\noutput result <- toned.result;\n",
    )
    .unwrap();
    let output = Command::new(orla)
        .arg("check")
        .arg(elsewhere.join("scene.osn"))
        .env("ORLA_PACKAGES", &dir)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    assert!(
        stdout.contains("graph `scene` with 2 node(s), 1 use(s)"),
        "{stdout}"
    );
    // A preset inside it is placed by the package name, and the preset's
    // own `use` resolves from the package's real location.
    std::fs::write(
        elsewhere.join("preset.osn"),
        "use \"unpacked.orlapkg\";\nnode source: kernel { requires { field.write; } output result: Image; domain = vec2(8, 8); compute { return vec4(0.5, 0.5, 0.5, 1.0); } }\nnode look: graph \"unpacked.orlapkg/graphs/nes.osn\" { source <- source.result; cell = 4.0; }\noutput result <- look.result;\n",
    )
    .unwrap();
    let output = Command::new(orla)
        .arg("check")
        .arg(elsewhere.join("preset.osn"))
        .env("ORLA_PACKAGES", &dir)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    assert!(stdout.contains("graph `preset` with 3 node(s)"), "{stdout}");

    // A package needing a newer language is refused with both versions.
    let manifest = unpacked.join("package.osn");
    let text = std::fs::read_to_string(&manifest).unwrap();
    std::fs::write(&manifest, text.replace("language = 1", "language = 99")).unwrap();
    let output = Command::new(orla)
        .arg("check")
        .arg(elsewhere.join("scene.osn"))
        .env("ORLA_PACKAGES", &dir)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!output.status.success(), "{stdout}");
    assert!(
        stdout.contains("package pixelfx@1 needs language 99, and this Orla has language 1"),
        "{stdout}"
    );
    let output = Command::new(orla).arg("version").output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.starts_with(&format!("orla = {}\n", env!("CARGO_PKG_VERSION")))
            && stdout.contains("orla.language = 1")
            && stdout.contains("orla.package = 1"),
        "{stdout}"
    );
}

#[test]
fn render_writes_a_png_per_frame_and_one_wav_of_every_block() {
    let orla = env!("CARGO_BIN_EXE_orla");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let out = std::env::temp_dir().join(format!("orla-render-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    let output = Command::new(orla)
        .args(["render", "--frames", "3", "--out"])
        .arg(&out)
        .arg(root.join("examples/audio/scene.osn"))
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(
        stderr.contains("rendered 3 picture(s) of 320x240"),
        "{stderr}"
    );
    assert!(stderr.contains("rendered 0.05 s of sound"), "{stderr}");
    for index in 0..3 {
        let bytes = std::fs::read(orla::render::picture_path(&out, index)).unwrap();
        let picture = orla_formats::png::decode(&bytes).unwrap();
        assert_eq!((picture.width, picture.height), (320, 240));
    }
    assert!(!out.join("0003.png").exists());
    let sound =
        orla_formats::wav::decode(&std::fs::read(out.join(orla::render::SOUND_NAME)).unwrap())
            .unwrap();
    assert_eq!(sound.rate, 48_000);
    assert_eq!(
        sound.field.domain().count(),
        3 * 800,
        "three blocks of 800 frames"
    );
    // The same frames give the same bytes: the clock is the count.
    let first = std::fs::read(orla::render::picture_path(&out, 2)).unwrap();
    let again = std::env::temp_dir().join(format!("orla-render-again-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&again);
    assert!(
        Command::new(orla)
            .args(["render", "--frames", "3", "--out"])
            .arg(&again)
            .arg(root.join("examples/audio/scene.osn"))
            .status()
            .unwrap()
            .success()
    );
    assert_eq!(
        std::fs::read(orla::render::picture_path(&again, 2)).unwrap(),
        first
    );
    // An async node is settled after every frame: each picture shows the
    // step before it, none skipped.
    let machine = std::env::temp_dir().join(format!("orla-render-async-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&machine);
    let output = Command::new(orla)
        .args([
            "render",
            "--frames",
            "2",
            "--window",
            "32x32",
            "--backend",
            "jit",
            "--out",
        ])
        .arg(&machine)
        .arg(root.join("examples/async/scene.osn"))
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(
        stderr.contains("rendered 2 picture(s) of 32x32"),
        "{stderr}"
    );
    for dir in [&out, &again, &machine] {
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[test]
fn a_frames_input_brings_in_the_frame_of_the_moment_and_wraps() {
    let dir = std::env::temp_dir().join(format!("orla-frames-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("clip")).unwrap();
    for (index, color) in [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
        .iter()
        .enumerate()
    {
        let pixels = vec![*color; 4];
        std::fs::write(
            dir.join(format!("clip/{index}.png")),
            orla_formats::png::encode(2, 2, &pixels),
        )
        .unwrap();
    }
    std::fs::write(
        dir.join("scene.osn"),
        "input clip: Image = frames \"clip/\" at 1;\nnode view: kernel { requires { field.read; field.write; } input clip: Image <- clip; output result: Image; compute { return sample(clip); } }\noutput result <- view.result;\n",
    )
    .unwrap();
    let mut session = Session::open(
        &dir.join("scene.osn"),
        None,
        orla_session::DEFAULT_BUDGET_WORDS,
    );
    let events = session.sync();
    assert!(
        events.iter().any(|event| matches!(event, Event::Loaded { name, detail, .. } if name == "clip" && detail == "2 frame(s) at 1 a second")),
        "{events:?}"
    );
    let red = |session: &mut Session, time: f32| {
        let report = session
            .evaluate(&Context {
                time,
                ..Context::default()
            })
            .unwrap();
        session.read_pixel(report.output, 0, 0).unwrap()[0]
    };
    assert_eq!(red(&mut session, 0.0), 1.0);
    assert_eq!(red(&mut session, 0.9), 1.0);
    assert_eq!(red(&mut session, 1.0), 0.0);
    assert_eq!(red(&mut session, 2.0), 1.0, "wraps after the last frame");
    // A new frame file is seen at the next sync.
    std::fs::write(
        dir.join("clip/2.png"),
        orla_formats::png::encode(2, 2, &[[0.0, 0.0, 1.0, 1.0]; 4]),
    )
    .unwrap();
    let events = session.sync();
    assert!(events.iter().any(|event| matches!(event, Event::Loaded { detail, .. } if detail == "3 frame(s) at 1 a second")), "{events:?}");
    assert_eq!(red(&mut session, 2.0), 0.0);
    // An empty directory is reported, and evaluation names what is missing.
    std::fs::remove_dir_all(dir.join("clip")).unwrap();
    std::fs::create_dir_all(dir.join("clip")).unwrap();
    let events = session.sync();
    assert!(events.iter().any(|event| matches!(event, Event::Rejected { error, .. } if error.message.contains("holds no frame"))), "{events:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_filled_field_node_is_a_field_of_its_type_and_the_loop_count_may_be_a_param() {
    let dir = std::env::temp_dir().join(format!("orla-fill-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("scene.osn"),
        "param level: Float = 0.25;\n\
         param depth: Int = 3;\n\
         node zeros: Rows<Float> { domain = 4; }\n\
         node set: Rows<Float> { domain = 4; value = level; }\n\
         node counted: kernel {\n\
             requires { field.read; field.write; }\n\
             input base: Rows<Float> <- set.result;\n\
             param depth: Int = depth;\n\
             output result: Rows<Float>;\n\
             domain = 4;\n\
             compute { return loop depth with acc = at(base, element()) { acc + 1.0 }; }\n\
         }\n\
         output result <- counted.result;\n\
         output zeros <- zeros.result;\n",
    )
    .unwrap();
    let mut session = Session::open(
        &dir.join("scene.osn"),
        None,
        orla_session::DEFAULT_BUDGET_WORDS,
    );
    session.sync();
    let read = |session: &mut Session, handle| {
        session.field(handle).unwrap();
        session
            .runtime()
            .host()
            .data(handle)
            .unwrap()
            .storage()
            .iter()
            .map(|w| f32::from_bits(*w))
            .collect::<Vec<_>>()
    };
    let report = session.evaluate(&Context::default()).unwrap();
    assert_eq!(read(&mut session, report.output), vec![3.25; 4]);
    let zeros = report
        .outputs
        .iter()
        .find(|(name, _)| name == "zeros")
        .unwrap()
        .1;
    assert_eq!(read(&mut session, zeros), vec![0.0; 4]);
    session
        .set_param("level", orla_kernel::Value::Float(1.0))
        .unwrap();
    session
        .set_param("depth", orla_kernel::Value::Int(10))
        .unwrap();
    session.sync();
    let report = session.evaluate(&Context::default()).unwrap();
    assert_eq!(read(&mut session, report.output), vec![11.0; 4]);
    // A count of zero yields what the loop carries.
    session
        .set_param("depth", orla_kernel::Value::Int(0))
        .unwrap();
    session.sync();
    let report = session.evaluate(&Context::default()).unwrap();
    assert_eq!(read(&mut session, report.output), vec![1.0; 4]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_name_a_kernel_cannot_find_names_the_file_that_declares_it() {
    let dir = std::env::temp_dir().join(format!("orla-located-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("kernels")).unwrap();
    std::fs::write(
        dir.join("kernels/util.orla"),
        "fn wobble(x: Float) -> Float { return x * 1.5; }\nkernel util.nothing@1 { requires { field.write; } output result: Rows<Float>; compute { return 0.0; } }\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("kernels/main.orla"),
        "kernel main.k@1 { requires { field.write; } output result: Rows<Float>; compute { return wobble(2.0); } }\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("scene.osn"),
        "use \"kernels/main.orla\";\nnode k: main.k@1 { domain = 2; }\noutput result <- k.result;\n",
    )
    .unwrap();
    let mut session = Session::open(
        &dir.join("scene.osn"),
        None,
        orla_session::DEFAULT_BUDGET_WORDS,
    );
    let events = session.sync();
    let reported = rejected(&events).join("\n");
    assert!(reported.contains("`wobble` is not a builtin"), "{reported}");
    assert!(
        reported.contains("fn `wobble` is declared in util.orla"),
        "{reported}"
    );
    assert!(reported.contains("`use \"util.orla\";`"), "{reported}");
    // Two documents using each other are reported as a cycle.
    std::fs::write(
        dir.join("kernels/main.orla"),
        "use \"util.orla\";\nstruct Tint { r: Float; }\nkernel main.k@1 { requires { field.write; } output result: Rows<Float>; compute { return wobble(2.0); } }\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("kernels/util.orla"),
        "use \"main.orla\";\nfn wobble(x: Float) -> Float { return x * 1.5; }\nkernel util.nothing@1 { requires { field.write; } output result: Rows<Float>; compute { return 0.0; } }\n",
    )
    .unwrap();
    let events = session.sync();
    let reported = rejected(&events).join("\n");
    assert!(
        reported.contains("use each other") && reported.contains("a third they both use"),
        "{reported}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_bytes_input_reads_a_file_as_rows_of_one_integer() {
    let dir = std::env::temp_dir().join(format!("orla-bytes-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("blob.bin"), [79u8, 82, 76, 65, 0, 255, 128]).unwrap();
    std::fs::write(
        dir.join("scene.osn"),
        "input blob: Rows<Int> = bytes \"blob.bin\";\n\
         node read: kernel {\n\
             requires { field.read; field.write; }\n\
             input data: Rows<Int> <- blob;\n\
             output result: Rows<Int>;\n\
             domain = 7;\n\
             compute { return at(data, element()); }\n\
         }\n\
         output result <- read.result;\n",
    )
    .unwrap();
    let mut session = Session::open(
        &dir.join("scene.osn"),
        None,
        orla_session::DEFAULT_BUDGET_WORDS,
    );
    let events = session.sync();
    assert!(
        events.iter().any(|event| matches!(event, Event::Loaded { name, detail, .. } if name == "blob" && detail == "7 byte(s)")),
        "{events:?}"
    );
    let words = |session: &mut Session, handle| {
        session.field(handle).unwrap();
        session
            .runtime()
            .host()
            .data(handle)
            .unwrap()
            .storage()
            .to_vec()
    };
    let report = session.evaluate(&Context::default()).unwrap();
    assert_eq!(
        words(&mut session, report.output),
        vec![79, 82, 76, 65, 0, 255, 128]
    );
    // The file is watched: the next sync takes an edit.
    std::fs::write(dir.join("blob.bin"), [1u8, 2, 3, 4, 5, 6, 7]).unwrap();
    session.sync();
    let report = session.evaluate(&Context::default()).unwrap();
    assert_eq!(
        words(&mut session, report.output),
        vec![1, 2, 3, 4, 5, 6, 7]
    );
    std::fs::write(dir.join("blob.bin"), []).unwrap();
    let events = session.sync();
    assert!(
        events
            .iter()
            .any(|event| matches!(event, Event::Rejected { error, .. } if error.message.contains("empty"))),
        "{events:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A placed graph's port follows the last dot; any other name is reported
/// as what it is.
#[test]
fn a_save_names_a_node_a_placed_graphs_node_or_an_output() {
    let orla = env!("CARGO_BIN_EXE_orla");
    let scene = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/collapse/scene.osn");
    let dir = scratch("saves");
    let run = |target: &str, file: &str| {
        Command::new(orla)
            .arg("run")
            .arg(&scene)
            .args(["--window", "160x100", "--backend", "cpu"])
            .arg("--save")
            .arg(format!("{target}={}", dir.join(file).display()))
            .arg("--out")
            .arg(dir.join("picture.ppm"))
            .output()
            .unwrap()
    };
    let output = run("grid", "by-output.field");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = run("steps/8.spread/6.result", "by-node.field");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // The same Field either way: an output is the node it is wired to.
    assert_eq!(
        std::fs::read(dir.join("by-output.field")).unwrap(),
        std::fs::read(dir.join("by-node.field")).unwrap()
    );
    let output = run("steps.result", "nothing.field");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        stderr.contains("is not a node: it is placed as 64 of them"),
        "{stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `bench` reports the least of several runs; its `--json` output is read
/// back by `--compare`.
#[test]
fn bench_measures_frames_and_compares_with_what_it_wrote() {
    let orla = env!("CARGO_BIN_EXE_orla");
    let dir = scratch("bench");
    let json = dir.join("before.json");
    // A scene that changes with the clock, so its nodes run each frame.
    let scene = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/collapse/scene.osn");
    let measure = |extra: &[&str]| {
        Command::new(orla)
            .arg("bench")
            .arg(&scene)
            .args([
                "--frames",
                "2",
                "--repeat",
                "2",
                "--backend",
                "cpu",
                "--window",
                "80x60",
            ])
            .args(extra)
            .output()
            .unwrap()
    };
    let output = measure(&["--nodes", "--json", json.to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}{stdout}");
    assert!(stdout.contains("2 frame(s) a run, 2 run(s)"), "{stdout}");
    assert!(stdout.contains("vs cpu"), "{stdout}");
    assert!(stdout.contains("ms a frame a node"), "{stdout}");
    // The scene's nodes are named, fused groups as the graph runs them.
    assert!(stdout.contains("steps/*.spread/*"), "{stdout}");

    let output = measure(&["--compare", json.to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    assert!(
        stdout.contains("ms +") || stdout.contains("ms -"),
        "{stdout}"
    );

    let output = measure(&["--compare", dir.join("nothing.json").to_str().unwrap()]);
    assert!(!output.status.success());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_scene_that_does_not_load_stops_a_command_with_why() {
    let orla = env!("CARGO_BIN_EXE_orla");
    let dir = scratch("unloaded");
    let scene = dir.join("broken.osn");
    std::fs::write(
        &scene,
        "param n: Int = 1;\nnode a: t.k@1 { times = n; domain = 4; }\noutput result <- a.result;\n",
    )
    .unwrap();
    for command in [
        vec!["run", "--param", "n=-1"],
        vec![
            "bench",
            "--backend",
            "cpu",
            "--frames",
            "1",
            "--repeat",
            "1",
            "--param",
            "n=-1",
        ],
    ] {
        let output = Command::new(orla)
            .arg(command[0])
            .arg(&scene)
            .args(&command[1..])
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{command:?}: {stderr}");
        assert!(stderr.contains("`times` must be"), "{command:?}: {stderr}");
        assert!(!stderr.contains("<no scene>"), "{command:?}: {stderr}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A thousand republishes leave one version, nothing awaiting collection,
/// and only the frame's Fields held.
#[test]
fn a_session_that_republishes_a_thousand_times_keeps_one_of_each() {
    let dir = scratch("long");
    let mut session = Session::open(&dir.join("scene.osn"), None, 1 << 20);
    session.sync();
    let kernels = session.runtime().kernels().count();
    assert!(kernels >= 3, "the scene publishes its Kernels: {kernels}");
    let context = Context::default();
    // What one frame leaves live is what a thousand leave.
    session.evaluate(&context).unwrap();
    let live = session.runtime().host().live();
    for step in 0..1000 {
        let source = format!(
            "kernel image.grayscale@1 {{ requires {{ field.read; field.write; }} input source: Image; param amount: Float = 1.0; output result: Image; fn luma(c: Vec4) -> Float {{ return c.r * {} + c.g * 0.7152 + c.b * 0.0722; }} compute {{ let c = sample(source); let g = luma(c); return mix(c, vec4(g, g, g, c.a), amount); }} }}",
            0.2126 + f64::from(step) * 0.000_001
        );
        session
            .publish_source(&source, orla_kernel::Expect::Any)
            .unwrap_or_else(|error| panic!("{error}"));
        if step % 4 == 0 {
            session.evaluate(&context).unwrap();
        }
    }
    assert_eq!(
        session.runtime().kernels().count(),
        kernels,
        "one version of each Kernel, whatever was published over it"
    );
    assert_eq!(
        session.runtime().uncollected(),
        0,
        "nothing prepared and left uncollected"
    );
    assert_eq!(
        session.runtime().host().live(),
        live,
        "the Host holds what the frame needs and no more"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Every `orla new` template is made, checked and replayed.
#[test]
fn what_new_writes_checks_and_replays() {
    let orla = env!("CARGO_BIN_EXE_orla");
    let root = scratch("new");
    for kind in ["picture", "sound", "game"] {
        let dir = root.join(kind);
        let made = Command::new(orla)
            .args(["new", dir.to_str().unwrap(), "--kind", kind])
            .output()
            .unwrap();
        assert!(
            made.status.success(),
            "{}",
            String::from_utf8_lossy(&made.stderr)
        );
        let listed = String::from_utf8_lossy(&made.stdout);
        assert_eq!(listed.lines().count(), 4, "{listed}");
        assert!(dir.join("scene.osn").exists(), "{listed}");

        let checked = Command::new(orla)
            .arg("check")
            .arg(dir.join("scene.osn"))
            .output()
            .unwrap();
        assert!(
            checked.status.success(),
            "{kind}: {}{}",
            String::from_utf8_lossy(&checked.stdout),
            String::from_utf8_lossy(&checked.stderr)
        );

        let replay = std::fs::read_dir(dir.join("tests"))
            .unwrap()
            .flatten()
            .next()
            .expect("a replay")
            .path();
        let tested = Command::new(orla)
            .arg("replay")
            .arg(&replay)
            .args(["--backend", "cpu"])
            .output()
            .unwrap();
        assert!(
            tested.status.success(),
            "{kind}: {}{}",
            String::from_utf8_lossy(&tested.stdout),
            String::from_utf8_lossy(&tested.stderr)
        );

        let again = Command::new(orla)
            .args(["new", dir.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(!again.status.success());
        assert!(
            String::from_utf8_lossy(&again.stderr).contains("already holds a scene"),
            "{}",
            String::from_utf8_lossy(&again.stderr)
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// The default budget holds millions of triangles.
#[test]
fn a_session_holds_what_its_budget_says_and_says_so_when_it_cannot() {
    let dir = std::env::temp_dir().join(format!("orla-budget-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // A grid of a million pixels: four million words, copied once more by
    // the second node.
    std::fs::write(
        dir.join("scene.osn"),
        "node fill: kernel { requires { field.write; } output result: Image; domain = vec2(1024, 1024); compute { return vec4(coord().x, coord().y, 0.0, 1.0); } }\n\
         node keep: kernel { requires { field.read; field.write; } input source: Image <- fill.result; output result: Image; domain = vec2(1024, 1024); compute { return sample(source) * 0.5; } }\n\
         output result <- keep.result;\n",
    )
    .unwrap();
    let scene = dir.join("scene.osn");

    // Two mebibytes hold neither.
    let mut small = Session::open(&scene, None, orla_session::budget_of_megabytes(2));
    small.sync();
    let refused = small.evaluate(&Context::default()).unwrap_err().to_string();
    assert!(
        refused.contains("resource budget exceeded") && refused.contains("available"),
        "{refused}"
    );

    // Sixty-four hold both, with the words the Kernel wrote.
    let mut enough = Session::open(&scene, None, orla_session::budget_of_megabytes(64));
    enough.sync();
    let report = enough.evaluate(&Context::default()).unwrap();
    let image = enough.runtime().host().image(report.output).unwrap();
    assert_eq!(image.width(), 1024);
    assert_eq!(image.pixels()[100][0], 50.0, "half of that pixel's own x");

    assert_eq!(orla_session::budget_of_megabytes(1), 256 * 1024);
    let _ = std::fs::remove_dir_all(&dir);
}

/// `times` resolves when the document is lowered.
#[test]
fn a_parents_param_reaches_a_placed_graphs_times() {
    let orla = env!("CARGO_BIN_EXE_orla");
    let dir = std::env::temp_dir().join(format!("orla-times-param-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("step.orla"),
        "kernel t.zero@1 { requires { field.write; } output result: Rows<Int>; compute { return 0; } }\nkernel t.inc@1 { requires { field.read; field.write; } input previous: Rows<Int>; output result: Rows<Int>; compute { return sample(previous) + 1; } }\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("chain.osn"),
        "use \"step.orla\";\nparam n: Int = 5;\ninput seed: Rows<Int>;\nnode step: t.inc@1 { times = n; carry previous <- result; previous <- seed; }\noutput result <- step.result;\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("scene.osn"),
        "use \"step.orla\";\nparam rounds: Int = 2;\nnode seed: t.zero@1 { domain = 4; }\nnode chain: graph \"chain.osn\" { seed <- seed.result; n = rounds; }\noutput result <- chain.result;\n",
    )
    .unwrap();
    let output = Command::new(orla)
        .arg("check")
        .arg(dir.join("scene.osn"))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    // The parent says two, so two, not the sub-graph's five.
    assert!(
        stdout.contains("graph `scene` with 3 node(s)"),
        "the parent's default reaches `times`: {stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn narrow_fields_land_live_and_leave_as_their_bytes() {
    // The packed example: file data and Kernel outputs are narrow, and a
    // narrow output saved as a Field file round-trips its bytes.
    let scene = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/packed/scene.osn");
    let mut session = Session::open(&scene, None, 1 << 22);
    session.set_fusion(false);
    let events = session.sync();
    assert!(events.contains(&Event::SceneLoaded), "{events:?}");
    session.evaluate(&Context::default()).unwrap();
    let host = session.runtime().host();
    let words = |handle| host.data(handle).unwrap().storage().len();
    let albedo = session.external("albedo").unwrap();
    assert_eq!(
        host.schema(albedo).unwrap().to_string(),
        "f:unorm8,f:unorm8,f:unorm8,f:unorm8"
    );
    assert_eq!(words(albedo), 64 * 64, "a word a pixel, not four");
    let tiles = session.external("tiles").unwrap();
    assert_eq!(words(tiles), 64 / 4, "a byte a tile");
    let node = |name: &str| {
        session
            .cached(&orla_graph::NodeId(name.into()), "result")
            .unwrap()
    };
    assert_eq!(words(node("height")), 160 * 90 * 2 / 4);
    assert_eq!(words(node("normal")), 160 * 90 * 3 / 4);
    assert_eq!(words(node("light")), 160 * 90 * 8 / 4);
    let normal = node("normal");

    let dir = scratch("narrow-save");
    let file = dir.join("normal.field");
    session.allow_write_to(&file).unwrap();
    session
        .save(&orla_graph::NodeId("normal".into()), "result", &file)
        .unwrap();
    let bytes = std::fs::read(&file).unwrap();
    let header = b"ORLAFIELD 1\nGrid 160 90 1 f:snorm8,f:snorm8,f:snorm8\n";
    assert!(bytes.starts_with(header));
    assert_eq!(bytes.len(), header.len() + 160 * 90 * 3);
    let back = orla_formats::file::decode(&bytes).unwrap();
    let host = session.runtime().host();
    assert_eq!(&back, host.data(normal).unwrap());
}

#[test]
fn a_parameter_of_any_value_type_is_set_by_its_layout() {
    // The cube's camera is a Mat4 and its style a struct: a value of their
    // layout is taken in their shape; another layout is refused by name.
    let scene = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/cube/scene.osn");
    let mut session = Session::open(&scene, None, 1 << 22);
    assert!(session.sync().contains(&Event::SceneLoaded));
    use orla_kernel::Value;
    let columns = Value::Record(vec![
        Value::Record(vec![
            Value::Int(3),
            Value::Int(0),
            Value::Int(0),
            Value::Int(0),
        ]),
        Value::Vec4([0.0, 3.0, 0.0, 0.0]),
        Value::Vec4([0.0, 0.0, 1.0, -1.0]),
        Value::Vec4([0.0; 4]),
    ]);
    session.set_param("lens", columns).unwrap();
    let Some(Value::Mat4(lens)) = session.params().get("lens") else {
        panic!("a Mat4: {:?}", session.params().get("lens"));
    };
    assert_eq!(lens[0], 3.0);
    assert_eq!(lens[11], -1.0);
    session
        .set_param(
            "style",
            Value::Record(vec![
                Value::Vec3([0.9, 0.9, 0.9]),
                Value::Vec3([0.8, 0.1, 0.1]),
                Value::Int(2),
            ]),
        )
        .unwrap();
    assert_eq!(
        session.params()["style"].to_string(),
        "[vec3(0.9, 0.9, 0.9), vec3(0.8, 0.1, 0.1), 2.0]"
    );
    session.evaluate(&Context::default()).unwrap();

    let error = session
        .set_param("lens", Value::Vec3([1.0, 2.0, 3.0]))
        .unwrap_err();
    assert!(error.contains("param `lens` is of type Mat4"), "{error}");
    let error = session
        .set_param("style", Value::Record(vec![Value::Bool(true)]))
        .unwrap_err();
    assert!(error.contains("param `style` is of type Style"), "{error}");
}

/// The interpreter declining every module holding a `Double`: a device
/// without doubles, for the session to route around.
struct NoDoubles(orla_cpu::CpuBackend);

impl orla_kernel::Backend for NoDoubles {
    fn name(&self) -> &'static str {
        "device"
    }
    fn declines(&self, module: &orla_ir::Validated) -> Option<String> {
        module
            .holds(orla_ir::Type::Double)
            .then(|| "it holds a Double".to_owned())
    }
    fn prepare(
        &self,
        module: &orla_ir::Validated,
    ) -> Result<Box<dyn orla_kernel::Prepared>, String> {
        self.0.prepare(module)
    }
    fn execute(
        &self,
        prepared: &dyn orla_kernel::Prepared,
        module: &orla_ir::Validated,
        invocation: &orla_kernel::Invocation,
        scope: &mut orla_kernel::Scope<'_>,
    ) -> Result<Vec<orla_kernel::Handle>, orla_kernel::ExecutionError> {
        orla_kernel::Backend::execute(&self.0, prepared, module, invocation, scope)
    }
}

#[test]
fn a_node_off_the_main_backend_is_said_when_the_scene_loads() {
    let dir = scratch("routed");
    std::fs::write(
        dir.join("scene.osn"),
        "node wide: kernel { requires { field.write; } output result: Rows<Double>; domain = 4;
            compute { return double(element()) * 0.1; } }
node plain: kernel { requires { field.write; } output result: Rows<Float>; domain = 4;
            compute { return float(element()); } }
output result <- plain.result;
",
    )
    .unwrap();
    let runtime = orla_kernel::Runtime::with_backends(
        1 << 20,
        Box::new(NoDoubles(orla_cpu::CpuBackend)),
        Some(Box::new(orla_cpu::CpuBackend)),
    );
    let mut session = Session::open_with(&dir.join("scene.osn"), None, runtime);
    let events = session.sync();
    let routed: Vec<String> = events
        .iter()
        .filter(|event| matches!(event, Event::Routed { .. }))
        .map(ToString::to_string)
        .collect();
    assert_eq!(
        routed,
        ["node `wide` runs on cpu (not on device: it holds a Double)"]
    );
    let placements = session.placements();
    let plain = placements.iter().find(|(node, _)| node == "plain").unwrap();
    assert_eq!(
        (plain.1.backend, plain.1.routed.as_deref()),
        ("device", None)
    );
    // Reported once: a sync that changes nothing reports nothing.
    assert!(
        !session
            .sync()
            .iter()
            .any(|event| matches!(event, Event::Routed { .. }))
    );
}

#[test]
fn a_device_failure_fails_the_evaluation_as_the_reference_does() {
    let Ok(device) = Session::runtime_for(
        "gpu",
        orla_session::Limits {
            words: 1 << 22,
            ..Default::default()
        },
    ) else {
        return;
    };
    let dir = scratch("device-failure");
    std::fs::write(
        dir.join("scene.osn"),
        "param k: Int = 10;
node n: kernel { requires { field.write; } param k: Int = k; output result: Rows<Int>; domain = 4;
    compute { return 2147483647 - k + element(); } }
node m: kernel { requires { field.read; field.write; } input rows: Rows<Int> <- n.result; output result: Rows<Int>;
    compute { return sample(rows) / 2; } }
output result <- m.result;
",
    )
    .unwrap();
    // The same scene on the reference and the device, through the same
    // values: every answer equal, failure for failure.
    let reference = Session::runtime_for(
        "cpu",
        orla_session::Limits {
            words: 1 << 22,
            ..Default::default()
        },
    )
    .unwrap();
    let mut sessions = [
        Session::open_with(&dir.join("scene.osn"), None, reference),
        Session::open_with(&dir.join("scene.osn"), None, device),
    ];
    let mut answers: [Vec<Result<(), String>>; 2] = [Vec::new(), Vec::new()];
    for (session, answers) in sessions.iter_mut().zip(&mut answers) {
        session.sync();
        let evaluate = |session: &mut Session| {
            session
                .evaluate(&Context::default())
                .map(|_| ())
                .map_err(|error| error.to_string())
        };
        answers.push(evaluate(session));
        session.set_param("k", orla_kernel::Value::Int(1)).unwrap();
        // Element 2 overflows each time: nothing from the failed run is
        // kept.
        answers.push(evaluate(session));
        answers.push(evaluate(session));
        session.set_param("k", orla_kernel::Value::Int(10)).unwrap();
        answers.push(evaluate(session));
    }
    assert_eq!(answers[1], answers[0]);
    let failure = answers[0][1].clone().unwrap_err();
    assert!(
        failure.contains("at element (2, 0): integer overflow"),
        "{failure}"
    );
    assert!(answers[0][0].is_ok() && answers[0][3].is_ok());
}

/// Elements each within their own budget fail the run once together they
/// pass its bound; the count is exact, and exactly the bound passes.
#[test]
fn a_run_past_the_hosts_work_fails_alike_on_every_backend() {
    // A map, a grid map, a scan and a reduction: each element loops S
    // times.
    let scenes = [
        "node w: kernel { requires { field.write; } param steps: Int = S; output result: Rows<Int>; domain = N;
            compute { return loop steps with a = 0 { a + 1 }; } }
output result <- w.result;",
        "node w: kernel { requires { field.write; } param steps: Int = S; output result: Grid<Int>; domain = vec2(25, N / 25);
            compute { return loop steps with a = 0 { a + 1 }; } }
output result <- w.result;",
        "node s: Rows<Int> { domain = N; }
node w: kernel { requires { field.read; field.write; } param steps: Int = S; input t: Rows<Int> <- s.result; output result: Rows<Int>;
            scan with c = 0 { return c + loop steps with a = 0 { a + 1 }; } }
output result <- w.result;",
        "node s: Rows<Int> { domain = N; }
node w: kernel { requires { field.read; field.write; } param steps: Int = S; input t: Rows<Int> <- s.result; output result: Rows<Int>;
            compute { return loop steps with a = 0 { a + 1 }; } merge(a, b) { return a + b; } }
output result <- w.result;",
    ];
    let dir = scratch("run-work");
    // (elements, iterations an element, the run's bound, whether it fails)
    let cases = [
        ("1000", "5000", 3_000_000, true),
        ("500", "5000", 3_000_000, false),
        ("1000", "800", 799_999, true),
        ("1000", "800", 800_000, false),
    ];
    for (k, scene) in scenes.iter().enumerate() {
        for (count, steps, work, fails) in cases {
            let text = scene.replace('N', count).replace('S', steps);
            std::fs::write(dir.join("scene.osn"), text).unwrap();
            let limits = orla_session::Limits {
                words: 1 << 22,
                work,
            };
            for backend in ["cpu", "jit", "gpu"] {
                let Ok(runtime) = Session::runtime_for(backend, limits) else {
                    continue;
                };
                let mut session = Session::open_with(&dir.join("scene.osn"), None, runtime);
                let events = session.sync();
                assert!(rejected(&events).is_empty(), "{:?}", rejected(&events));
                let answer = session
                    .evaluate(&Context::default())
                    .map(|_| ())
                    .map_err(|error| error.to_string());
                let case = format!("scene {k}, {count} x {steps} within {work} on {backend}");
                match fails {
                    true => assert!(
                        answer.as_ref().is_err_and(|error| error
                            .contains(&format!("together took more than {work} loop iterations"))),
                        "{case}: {answer:?}"
                    ),
                    false => assert!(answer.is_ok(), "{case}: {answer:?}"),
                }
            }
        }
    }
}

#[test]
fn work_bounds_hold_per_node_whether_or_not_nodes_fuse() {
    let dir = scratch("work-fused");
    // Counts are params, so the loops run rather than fold.
    let pair = "node a: kernel { requires { field.write; } BUDGET param n: Int = STEPS; output result: Rows<Float>; domain = 700;
    compute { return float(loop n with x = 0 { x + 1 }); } }
node b: kernel { requires { field.read; field.write; } param n: Int = 1000; input t: Rows<Float> <- a.result; output result: Rows<Float>;
    compute { return sample(t) + float(loop n with x = 0 { x + 1 }); } }
output result <- b.result;";
    // (the producer's budget, its iterations, the run's bound, what fails)
    let cases = [
        ("", "1000", 1_000_000, None),
        ("budget 1500;", "2000", 1_000_000_000, Some("node `a`")),
    ];
    for (budget, steps, work, fails) in cases {
        let text = pair.replace("BUDGET", budget).replace("STEPS", steps);
        std::fs::write(dir.join("scene.osn"), text).unwrap();
        for backend in ["cpu", "jit", "gpu"] {
            for (fusion, pinned) in [(true, false), (false, false), (true, true)] {
                let limits = orla_session::Limits {
                    words: 1 << 22,
                    work,
                };
                let Ok(runtime) = Session::runtime_for(backend, limits) else {
                    continue;
                };
                let mut session = Session::open_with(&dir.join("scene.osn"), None, runtime);
                session.set_fusion(fusion);
                if pinned {
                    session.pin(orla_graph::NodeId("a".into()));
                }
                let events = session.sync();
                assert!(rejected(&events).is_empty(), "{:?}", rejected(&events));
                let answer = session
                    .evaluate(&Context::default())
                    .map(|_| ())
                    .map_err(|error| error.to_string());
                let case =
                    format!("{budget} {steps} on {backend}, fusion {fusion}, pinned {pinned}");
                match fails {
                    None => assert!(answer.is_ok(), "{case}: {answer:?}"),
                    Some(node) => assert!(
                        answer.as_ref().is_err_and(|error| error.contains(node)),
                        "{case}: {answer:?}"
                    ),
                }
            }
        }
    }
}

#[test]
fn an_element_past_its_budget_fails_alike_on_every_backend() {
    // Element n runs n + n * n iterations; with a budget of a thousand,
    // element 32 is the first past it (31 runs 992).
    let scenes = [
        "node w: kernel { requires { field.write; } budget 1000; output result: Rows<Int>; domain = 64;
            compute { let n = element(); return loop n with a = 0 { loop n with b = a { b + 1 } }; } }
output result <- w.result;",
        "node w: kernel { requires { field.write; } budget 1000; output result: Grid<Int>; domain = vec2(8, 8);
            compute { let c = icoord(); let n = c.x + c.y * 8; return loop n with a = 0 { loop n with b = a { b + 1 } }; } }
output result <- w.result;",
        "node ticks: Rows<Int> { domain = 64; }
node w: kernel { requires { field.read; field.write; } budget 1000; input t: Rows<Int> <- ticks.result; output result: Rows<Int>;
            scan with s = 0 { let n = index(); return s + loop n with a = 0 { loop n with b = a { b + 1 } }; } }
output result <- w.result;",
        "node ticks: Rows<Int> { domain = 64; }
node w: kernel { requires { field.read; field.write; } budget 50; input t: Rows<Int> <- ticks.result; output result: Rows<Int>;
            compute { return 3; } merge(a, b) { return loop a with x = b { x + 1 }; } }
output result <- w.result;",
        // Element e's body runs e times before `until` stops the loop, so
        // 1001 is the first past a thousand.
        "node w: kernel { requires { field.write; } budget 1000; output result: Rows<Int>; domain = 1100;
            compute { return loop 2000 with a = 0 until index() >= element() { a + 1 }; } }
output result <- w.result;",
        // A loop bounded where written is never counted and never fails.
        "node w: kernel { requires { field.write; } budget 1000; output result: Rows<Int>; domain = 64;
            compute { return loop 30 with a = 0 { loop 30 with b = a { b + 1 } }; } }
output result <- w.result;",
    ];
    let dir = scratch("work-budget");
    for (k, scene) in scenes.iter().enumerate() {
        std::fs::write(dir.join("scene.osn"), scene).unwrap();
        let mut answers = Vec::new();
        for backend in ["cpu", "jit", "gpu"] {
            let Ok(runtime) = Session::runtime_for(
                backend,
                orla_session::Limits {
                    words: 1 << 22,
                    ..Default::default()
                },
            ) else {
                continue;
            };
            let mut session = Session::open_with(&dir.join("scene.osn"), None, runtime);
            let events = session.sync();
            assert!(rejected(&events).is_empty(), "{:?}", rejected(&events));
            let answer = session
                .evaluate(&Context::default())
                .map(|_| ())
                .map_err(|error| error.to_string());
            answers.push((backend, answer));
        }
        let (_, reference) = &answers[0];
        for (backend, answer) in &answers {
            assert_eq!(answer, reference, "scene {k} on {backend}");
        }
        match k {
            0 => assert!(
                reference
                    .as_ref()
                    .unwrap_err()
                    .contains("at element (32, 0): more than 1000 loop iterations"),
                "{reference:?}"
            ),
            4 => assert!(
                reference
                    .as_ref()
                    .unwrap_err()
                    .contains("at element (1001, 0): more than 1000 loop iterations"),
                "{reference:?}"
            ),
            5 => assert!(reference.is_ok(), "{reference:?}"),
            _ => assert!(reference.is_err(), "scene {k}: {reference:?}"),
        }
    }
}

/// A domain is charged before any allocation: past the budget or 32-bit
/// indices, the node fails with why and the session goes on.
#[test]
fn a_domain_past_the_budget_fails_the_node_not_the_process() {
    let scenes = [
        (
            "domain = 2000000000;",
            "Rows<Int>",
            "resource budget exceeded",
        ),
        (
            "domain = vec3(4194304, 2097152, 2097152);",
            "Volume<Int>",
            "holds more than 4294967295 elements",
        ),
        (
            "domain = vec2(4294967295, 4294967295);",
            "Grid<Int>",
            "holds more than 4294967295 elements",
        ),
    ];
    let dir = scratch("huge-domain");
    for backend in ["cpu", "jit", "gpu"] {
        let Ok(runtime) = Session::runtime_for(backend, Default::default()) else {
            continue;
        };
        std::fs::write(dir.join("scene.osn"), "node w: kernel { requires { field.write; } output result: Rows<Int>; domain = 4; compute { return 1; } }\noutput result <- w.result;").unwrap();
        let mut session = Session::open_with(&dir.join("scene.osn"), None, runtime);
        for (domain, ty, why) in scenes {
            std::fs::write(
                dir.join("scene.osn"),
                format!(
                    "node w: kernel {{ requires {{ field.write; }} output result: {ty}; {domain} compute {{ return 1; }} }}\noutput result <- w.result;"
                ),
            )
            .unwrap();
            let events = session.sync();
            assert!(rejected(&events).is_empty(), "{:?}", rejected(&events));
            let error = session
                .evaluate(&Context::default())
                .map(|_| ())
                .unwrap_err()
                .to_string();
            assert!(error.contains(why), "{backend}, {domain}: {error}");
            assert_eq!(session.runtime().host().used_words(), 0, "{backend}");
        }
        std::fs::write(dir.join("scene.osn"), "node w: kernel { requires { field.write; } output result: Rows<Int>; domain = 8; compute { return 2; } }\noutput result <- w.result;").unwrap();
        session.sync();
        assert!(session.evaluate(&Context::default()).is_ok(), "{backend}");
    }
}

/// The raster package's cutout and `native.rasterize@2` draw the same
/// surface on the interpreter and the JIT: the same triangle at every
/// pixel, its depth to the bit.
#[test]
fn the_package_cutout_is_the_fast_paths_to_the_bit_off_the_device() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let base = root.join("examples/mesh/tests");
    let replay = orla::replay::parse(
        "scene cutout.osn\nwindow 320 240\nframe 0.0\nexpect differ.result[0] = 0\nexpect deeper.result[0] = 0\nexpect taken.result[0] > 1000\n",
        &base,
    )
    .unwrap();
    for backend in ["cpu", "jit"] {
        let runtime = Session::runtime_for(backend, Default::default()).unwrap();
        let outcome = orla::replay::run(&replay, runtime).unwrap();
        assert!(
            outcome.failures.is_empty(),
            "{backend}: {:?}",
            outcome.failures
        );
        assert_eq!(outcome.checks, 3, "{backend}");
    }
}

/// A document's Kernels change together: when one of two Kernels sharing
/// a changed struct does not compile, neither publishes and the running
/// pair stays. Fixed, both land in the same sync, on the worker and
/// inline.
#[test]
fn a_documents_kernels_publish_together_or_not_at_all() {
    let dir = std::env::temp_dir().join(format!("orla-together-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let pair = |field: &str, broken: bool| {
        format!(
            "struct P {{ {field}: Int; }}\n\
             kernel t.make@1 {{ requires {{ field.write; }} output result: Rows<P>; compute {{ return P {{ {field}: element() + 1 }}; }} }}\n\
             kernel t.use@1 {{ requires {{ field.read; field.write; }} input p: Rows<P>; output result: Rows<Int>; compute {{ return sample(p).{} * 10; }} }}\n",
            if broken { "nope" } else { field }
        )
    };
    std::fs::write(dir.join("pair.orla"), pair("a", false)).unwrap();
    std::fs::write(
        dir.join("scene.osn"),
        "use \"pair.orla\";\nnode make: t.make@1 { domain = 3; }\nnode use: t.use@1 { p <- make.result; }\noutput result <- use.result;\n",
    )
    .unwrap();
    for background in [false, true] {
        let mut session = Session::open(&dir.join("scene.osn"), None, 1 << 20);
        session.set_background(background);
        std::fs::write(dir.join("pair.orla"), pair("a", false)).unwrap();
        let events = session.sync();
        assert!(rejected(&events).is_empty(), "{:?}", rejected(&events));
        let generation = |session: &Session, id: &str| {
            session
                .runtime()
                .kernel(&id.parse().unwrap())
                .unwrap()
                .generation()
        };
        let (make, used) = (
            generation(&session, "t.make@1"),
            generation(&session, "t.use@1"),
        );
        let value = |session: &mut Session| {
            let report = session.evaluate(&Context::default()).unwrap();
            session.runtime().host().element(report.output, 2).unwrap()[0] as i32
        };
        assert_eq!(value(&mut session), 30);

        // The field is renamed in both, and the reader is broken.
        std::fs::write(dir.join("pair.orla"), pair("b", true)).unwrap();
        let mut events = session.sync();
        for _ in 0..20 {
            std::thread::sleep(std::time::Duration::from_millis(5));
            events.extend(session.sync());
        }
        assert!(published(&events).is_empty(), "{events:?}");
        let said = rejected(&events).join("\n");
        assert!(said.contains("nope"), "{said}");
        assert!(said.contains("change together"), "{said}");
        assert_eq!(
            generation(&session, "t.make@1"),
            make,
            "background {background}"
        );
        assert_eq!(generation(&session, "t.use@1"), used);
        assert_eq!(value(&mut session), 30, "the pair that ran runs");

        std::fs::write(dir.join("pair.orla"), pair("b", false)).unwrap();
        // On the worker, the replacements land in a later sync.
        let mut landed = published(&session.sync());
        for _ in 0..1000 {
            if landed.len() == 2 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
            landed.extend(published(&session.sync()));
        }
        landed.sort();
        assert_eq!(landed, ["t.make@1", "t.use@1"], "background {background}");
        assert_eq!(value(&mut session), 30);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The Player's way (the whole scene each frame, one frame before the
/// window opens) holds what `orla replay` holds.
#[test]
fn a_replay_played_as_a_window_plays_it_holds_as_it_does_headless() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/voxel/tests");
    for name in ["voxel.replay"] {
        let text = std::fs::read_to_string(base.join(name)).unwrap();
        let replay = orla::replay::parse(&text, &base).unwrap();
        let runtime = Session::runtime_for("jit", Default::default()).unwrap();
        let mut session = Session::open_with(&replay.scene, replay.kernels.as_deref(), runtime);
        session.sync();
        let size = replay.steps.iter().find_map(|step| match step {
            orla::replay::Step::Window { size, scale } => Some((*size, *scale)),
            _ => None,
        });
        let (window, scale) = size.unwrap_or(([0.0, 0.0], 1.0));
        session
            .evaluate(&Context {
                window,
                scale,
                ..Context::default()
            })
            .unwrap();
        let mut replayer = orla::replay::Replayer::start(&replay, &mut session);
        while let Some(frame) = replayer.next_frame(&mut session).unwrap() {
            let report = session.evaluate(&frame.context).unwrap();
            replayer.evaluated(report);
        }
        let outcome = &replayer.outcome;
        assert!(
            outcome.failures.is_empty(),
            "{name}: {:?}",
            outcome.failures
        );
        assert!(outcome.checks > 30, "{name}: {} checks", outcome.checks);
    }
}

/// An undeclared param set before load stops the load with why; a held
/// value goes when its declaration goes.
#[test]
fn a_param_is_one_the_document_declares() {
    let dir = scratch("declared");
    let scene = dir.join("scene.osn");
    let text = "use \"kernels/\";
param level: Float = 0.5;
node seed: image.gradient@1 { domain = vec2(8, 8); }
node gray: image.grayscale@1 { source <- seed.result; amount = level; }
output result <- gray.result;
";
    std::fs::write(&scene, text).unwrap();
    // On the command line, a node's own value is not a param.
    let orla = env!("CARGO_BIN_EXE_orla");
    let output = Command::new(orla)
        .arg("run")
        .arg(&scene)
        .args(["--param", "amount=1.0", "--out"])
        .arg(dir.join("out.ppm"))
        .output()
        .unwrap();
    let said = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        said.contains(
            "`amount` is not a param the scene declares; node `gray` sets its own `amount`"
        ),
        "{said}"
    );
    let output = Command::new(orla)
        .arg("run")
        .arg(&scene)
        .args(["--param", "level=1.0", "--out"])
        .arg(dir.join("out.ppm"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let mut session = Session::open(&scene, None, 1 << 20);
    session.sync();
    session
        .set_param("level", orla_kernel::Value::Float(1.0))
        .unwrap();
    assert_eq!(
        session.params().get("level"),
        Some(&orla_kernel::Value::Float(1.0))
    );
    // The document drops the declaration: the held value goes with it.
    std::fs::write(
        &scene,
        text.replace("param level: Float = 0.5;\n", "")
            .replace("amount = level;", "amount = 0.5;"),
    )
    .unwrap();
    assert!(session.sync().contains(&Event::SceneLoaded));
    assert!(session.params().is_empty(), "{:?}", session.params());
    let refused = session
        .set_param("level", orla_kernel::Value::Float(1.0))
        .unwrap_err();
    assert!(
        refused.contains("`level` is not a param the scene declares"),
        "{refused}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A named file allows that file alone, not its directory.
#[test]
fn save_and_export_write_only_where_the_session_may() {
    let dir = scratch("guarded-writes");
    let mut session = Session::open(&dir.join("scene.osn"), None, 1 << 20);
    session.sync();
    session.pin(orla_graph::NodeId("show".into()));
    let report = session.evaluate(&Context::default()).unwrap();
    let out = dir.join("out.ppm");
    let refused = session.export(report.output, &out).unwrap_err().to_string();
    assert!(
        refused.contains("writing files is not allowed"),
        "{refused}"
    );
    let field = dir.join("show.field");
    let refused = session
        .save(&orla_graph::NodeId("show".into()), "result", &field)
        .unwrap_err();
    assert!(
        refused.contains("writing files is not allowed"),
        "{refused}"
    );
    assert!(!out.exists() && !field.exists(), "nothing was written");
    // Named, the file is written; its neighbour is not.
    session.allow_write_to(&out).unwrap();
    session.export(report.output, &out).unwrap();
    assert!(out.exists());
    let refused = session
        .save(&orla_graph::NodeId("show".into()), "result", &field)
        .unwrap_err();
    assert!(
        refused.contains("outside every directory allowed"),
        "{refused}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The Kernels the running scene uses stay published.
#[test]
fn a_refused_scene_edit_leaves_the_scene_running() {
    let dir = scratch("atomic-scene");
    let scene = dir.join("scene.osn");
    let original = std::fs::read_to_string(&scene).unwrap();
    let mut runtime = Session::runtime_for("cpu", orla_session::Limits::default()).unwrap();
    runtime.set_grants(
        [
            orla_kernel::Capability::FieldRead,
            orla_kernel::Capability::FieldWrite,
        ]
        .into_iter()
        .collect(),
    );
    let mut session = Session::open_with(&scene, None, runtime);
    session.sync();
    assert!(original.contains("output result <- blocks.result;"));
    let before = session.evaluate(&Context::default()).unwrap();
    let nodes = session.graph().unwrap().nodes.len();
    std::fs::write(
        &scene,
        original.replace(
            "output result <- blocks.result;",
            "node t: kernel { requires { field.read; field.write; time.read; } input source: Image <- blocks.result; output result: Image; compute { let c = sample(source); return vec4(c.r * time(), c.g, c.b, 1.0); } }\noutput result <- t.result;",
        ),
    )
    .unwrap();
    let events = session.sync();
    assert!(
        events.iter().any(|e| matches!(e, Event::SceneRejected(why) if why.message.contains("does not replace the scene running"))),
        "{events:?}"
    );
    assert!(!events.contains(&Event::SceneLoaded), "{events:?}");
    assert_eq!(
        session.graph().unwrap().nodes.len(),
        nodes,
        "the graph running stays"
    );
    let after = session.evaluate(&Context::default()).unwrap();
    assert_eq!(after.outputs.len(), before.outputs.len());
    let _ = std::fs::remove_dir_all(&dir);
}
