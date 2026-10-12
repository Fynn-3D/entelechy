use std::path::{Path, PathBuf};
use std::process::Command;

use entelechy_graph::EvaluateError;
use entelechy_kernel::Context;
use entelechy_session::{Event, Session};

/// The scene most tests start from: a gradient, turned gray, in blocks.
const SCENE: &str = "// gradient -> grayscale -> pixelate
use \"kernels/\";

node source: image.gradient@1 {
    domain = vec2(128, 128);
}

node gray: image.grayscale@1 {
    source <- source.result;
    amount = 1.0;
}

node blocks: image.pixelate@1 {
    source <- gray.result;
    size = 8;
}

output result <- blocks.result;
";

/// The scene's Kernels, a file each.
const KERNELS: &[(&str, &str)] = &[
    (
        "gradient.ent",
        "/// A generator: no inputs, so the scene must give the node a domain.
kernel image.gradient@1 {
    requires { field.write; }
    param depth: Float = 0.5;
    output result: Image;
    compute {
        let uv = coord() / (domain() - vec2(1.0, 1.0));
        return vec4(uv.x, uv.y, depth, 1.0);
    }
}
",
    ),
    (
        "grayscale.ent",
        "kernel image.grayscale@1 {
    requires { field.read; field.write; }
    input source: Image;
    param amount: Float = 1.0;
    output result: Image;
    fn luma(c: Vec4) -> Float {
        return c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722;
    }
    compute {
        let c = sample(source);
        let g = luma(c);
        return mix(c, vec4(g, g, g, c.a), amount);
    }
}
",
    ),
    (
        "pixelate.ent",
        "kernel image.pixelate@1 {
    requires { field.read; field.write; }
    input source: Image;
    param size: Int = 8;
    output result: Image;
    compute {
        let block = float(max(size, 1));
        let origin = floor(coord() / block) * block;
        return fetch(source, origin);
    }
}
",
    ),
];

/// A directory of its own holding [`SCENE`] and [`KERNELS`].
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("kernels")).unwrap();
    for (file, text) in KERNELS {
        std::fs::write(dir.join("kernels").join(file), text).unwrap();
    }
    std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
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
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
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
    let grayscale = dir.join("kernels/grayscale.ent");
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
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    session.sync();
    session.evaluate(&Context::default()).unwrap();

    let scene = dir.join("scene.esn");
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
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    let output = Command::new(entelechy)
        .args(["check", "--ir"])
        .arg(dir.join("kernels/grayscale.ent"))
        .arg(dir.join("scene.esn"))
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
        dir.join("kernels/counted.ent"),
        "kernel t.counted@1 { requires { field.write; } param n: Int = 3; output result: Rows<Int>; compute { return loop n with a = 0 { a + 1 }; } }",
    )
    .unwrap();
    let counted = Command::new(entelechy)
        .args(["check", "--ir"])
        .arg(dir.join("kernels/counted.ent"))
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
    std::fs::remove_file(dir.join("kernels/counted.ent")).unwrap();

    std::fs::write(
        dir.join("kernels/bad.ent"),
        "kernel bad@1 {\n  compute { return 1; }\n}",
    )
    .unwrap();
    let output = Command::new(entelechy)
        .arg("check")
        .arg(dir.join("kernels/bad.ent"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("bad.ent:1:"), "{stdout}");
    // A scene is checked through its libraries: the bad Kernel fails it.
    let output = Command::new(entelechy)
        .arg("check")
        .arg(dir.join("scene.esn"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("bad.ent:1:"), "{stdout}");
    std::fs::remove_file(dir.join("kernels/bad.ent")).unwrap();

    let out = dir.join("result.ppm");
    let output = Command::new(entelechy)
        .arg("run")
        .arg(dir.join("scene.esn"))
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

/// `entelechy player|import|export` runs `entelechy-<tool>` beside `entelechy` with the
/// same arguments, and exits as it does; with none beside it or on `PATH`,
/// it says what it looked for.
#[test]
fn a_tool_beside_entelechy_runs_as_an_entelechy_command() {
    let dir = std::env::temp_dir().join(format!("entelechy-beside-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let entelechy = dir.join("entelechy");
    std::fs::copy(env!("CARGO_BIN_EXE_entelechy"), &entelechy).unwrap();
    // A stand-in for `entelechy-export`: `entelechy` itself, which answers `version`.
    std::fs::copy(
        env!("CARGO_BIN_EXE_entelechy"),
        dir.join("entelechy-export"),
    )
    .unwrap();
    let output = Command::new(&entelechy)
        .args(["export", "version"])
        .env("PATH", "")
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.starts_with("entelechy = "), "{stdout}");
    let output = Command::new(&entelechy)
        .args(["export", "check"])
        .env("PATH", "")
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(1),
        "its exit status is entelechy's"
    );
    let output = Command::new(&entelechy)
        .arg("player")
        .env("PATH", "")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("runs `entelechy-player`, beside `entelechy` or on PATH"),
        "{stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `run` writes only the files named, each in the format its extension
/// names.
#[test]
fn run_writes_only_the_files_named_each_as_its_name_says() {
    let dir = scratch("run-writes");
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    let files = |dir: &Path| {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|name| name != "kernels" && name != "scene.esn")
            .collect();
        names.sort();
        names
    };
    let run = |args: &[&str]| {
        let output = Command::new(entelechy)
            .current_dir(&dir)
            .arg("run")
            .arg("scene.esn")
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
        stderr.contains("names no format Entelechy writes") && stderr.contains("use .field, .png"),
        "{stderr}"
    );
    assert_eq!(files(&dir), ["frame.png", "frame.ppm", "saved.png"]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A replay that holds says ok with what it ran; one that does not says
/// which line failed and what it found.
#[test]
fn a_replay_says_ok_or_what_failed() {
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    let dir = scratch("replay");
    std::fs::write(dir.join("right.replay"), "scene scene.esn\nframe 0.0\n").unwrap();
    let output = Command::new(entelechy)
        .arg("replay")
        .arg(dir.join("right.replay"))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    assert!(
        stdout.contains("right.replay: ok (1 frames, 0 checks)"),
        "{stdout}"
    );
    std::fs::write(
        dir.join("wrong.replay"),
        "scene scene.esn\nframe 0.0\nexpect 0 0 = 9.0 9.0 9.0\n",
    )
    .unwrap();
    let output = Command::new(entelechy)
        .arg("replay")
        .arg(dir.join("wrong.replay"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("line 3: pixel (0, 0) is"), "{stdout}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// `--profile` on `render` and on `replay` times each node over the frames
/// after the first, each a line in milliseconds a frame, and their sum.
#[test]
fn render_and_replay_profile_their_nodes() {
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    let dir = scratch("profile");
    std::fs::write(
        dir.join("clocked.esn"),
        "node ticks: kernel { requires { field.write; time.read; } output result: Image; domain = vec2(8, 8); compute { return vec4(fract(time()), 0.0, 0.0, 1.0); } }\noutput result <- ticks.result;\n",
    )
    .unwrap();
    let output = Command::new(entelechy)
        .args(["render", "--frames", "3", "--profile", "--out"])
        .arg(dir.join("frames"))
        .arg(dir.join("clocked.esn"))
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(
        stderr.contains("profile of 2 frame(s) after the first"),
        "{stderr}"
    );
    assert!(
        stderr
            .lines()
            .any(|line| line.trim_start().starts_with("ticks ")),
        "{stderr}"
    );
    assert!(stderr.contains("(sum)"), "{stderr}");
    std::fs::write(
        dir.join("timed.replay"),
        "scene clocked.esn\nframe 0.0\nframe 0.1\nframe 0.2\n",
    )
    .unwrap();
    let output = Command::new(entelechy)
        .args(["replay", "--profile"])
        .arg(dir.join("timed.replay"))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    assert!(
        stdout.contains("profile of 2 frame(s) after the first"),
        "{stdout}"
    );
    assert!(
        stdout
            .lines()
            .any(|line| line.trim_start().starts_with("ticks ")),
        "{stdout}"
    );
    assert!(
        stdout.contains("timed.replay: ok (3 frames, 0 checks)"),
        "{stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_graph_command_prints_the_plan() {
    let dir = scratch("graph");
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    let output = Command::new(entelechy)
        .arg("graph")
        .arg(dir.join("scene.esn"))
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
    let output = Command::new(entelechy)
        .args(["graph", "--no-fusion"])
        .arg(dir.join("scene.esn"))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("// plan: 3 group(s)"), "{stdout}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_background_republish_keeps_the_running_kernel_until_the_worker_is_done() {
    let dir = scratch("background");
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
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

    let grayscale = dir.join("kernels/grayscale.ent");
    let source = std::fs::read_to_string(&grayscale).unwrap();
    std::fs::write(
        &grayscale,
        source.replace(
            "return mix(c, vec4(g, g, g, c.a), amount);",
            "return vec4(c.r, 0.0, 0.0, c.a);",
        ),
    )
    .unwrap();
    // The edit is compiled off the frames and said once compiled; the
    // publish lands on a later sync, and frames in between run the old
    // version.
    let mut events = session.sync();
    let mut frames = 0;
    while events.is_empty() {
        let report = session.evaluate(&Context::default()).unwrap();
        let pixel = session
            .runtime()
            .host()
            .read_pixel(report.output, 100, 20)
            .unwrap();
        assert_eq!(pixel, before, "the running version keeps its output");
        frames += 1;
        assert!(frames < 5000, "the document never compiled");
        std::thread::sleep(std::time::Duration::from_millis(1));
        events = session.sync();
    }
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
        assert!(frames < 10000, "the worker never delivered");
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

    // A broken edit is rejected by the compiler once compiled.
    std::fs::write(
        &grayscale,
        "kernel image.grayscale@1 {\n  requires { field.write; }\n  output result: Image;\n  compute { return oops; }\n}",
    )
    .unwrap();
    let events = session.sync_settled();
    assert!(rejected(&events)[0].contains("`oops` is not defined"));
    assert!(session.evaluate(&Context::default()).is_ok());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn libraries_share_declarations_through_use_and_keep_their_own() {
    let dir = std::env::temp_dir().join(format!("entelechy-shared-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("lib")).unwrap();
    std::fs::write(
        dir.join("lib/types.ent"),
        "struct Cell { kind: Int; ttl: Int; }\ntype Board = Grid<Cell>;\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("lib/rules.ent"),
        "use \"types.ent\";\nkernel t.seed@1 { requires { field.write; } output result: Board; compute { return Cell { kind: 1, ttl: int(coord().x) }; } }\n",
    )
    .unwrap();
    // The scene's inline Kernel sees the declarations through `use`.
    std::fs::write(
        dir.join("scene.esn"),
        "use \"lib/\";\nnode seed: t.seed@1 { domain = vec2(4, 2); }\nnode show: kernel { requires { field.read; field.write; } input board: Board <- seed.result; output result: Image; compute { let c = sample(board); return vec4(float(c.ttl) / 4.0, float(c.kind), 0.0, 1.0); } }\noutput result <- show.result;\n",
    )
    .unwrap();
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let events = session.sync();
    assert_eq!(rejected(&events), Vec::<String>::new(), "{events:?}");
    let report = session.evaluate(&Context::default()).unwrap();
    let pixel = session.read_pixel(report.output, 3, 0).unwrap();
    assert_eq!(pixel, [0.75, 1.0, 0.0, 1.0]);

    // A second library declares its own `Cell`: each document keeps its
    // own, and the scene, which names `Cell` only through `Board`, runs
    // as it did.
    std::fs::write(
        dir.join("lib/other.ent"),
        "struct Cell { kind: Int; }\nkernel t.other@1 { requires { field.write; } output result: Rows<Cell>; compute { return Cell { kind: 0 }; } }\n",
    )
    .unwrap();
    let events = session.sync();
    assert_eq!(rejected(&events), Vec::<String>::new(), "{events:?}");
    let report = session.evaluate(&Context::default()).unwrap();
    let pixel = session.read_pixel(report.output, 3, 0).unwrap();
    assert_eq!(pixel, [0.75, 1.0, 0.0, 1.0]);

    // A Kernel of the scene that names `Cell` itself reaches two, and is
    // refused, naming both.
    std::fs::write(
        dir.join("scene.esn"),
        "use \"lib/\";\nnode seed: t.seed@1 { domain = vec2(4, 2); }\nnode show: kernel { requires { field.read; field.write; } input board: Rows<Cell> <- seed.result; output result: Image; compute { let c = sample(board); return vec4(float(c.kind), 0.0, 0.0, 1.0); } }\noutput result <- show.result;\n",
    )
    .unwrap();
    let events = session.sync();
    let refused = rejected(&events).join("\n");
    assert!(refused.contains("`Cell` is declared by"), "{refused}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_type_written_alike_over_different_structs_is_two() {
    let dir = std::env::temp_dir().join(format!("entelechy-alike-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("a")).unwrap();
    std::fs::create_dir_all(dir.join("b")).unwrap();
    // Two libraries write `Styles` alike, each over its own `Style`.
    std::fs::write(
        dir.join("a/a.ent"),
        "struct Style { x: Float; }\ntype Styles = Rows<Style>;\nkernel t.a@1 { requires { field.write; } output result: Styles; compute { return Style { x: 1.0 }; } }\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("b/b.ent"),
        "struct Style { y: Int; z: Int; }\ntype Styles = Rows<Style>;\nkernel t.b@1 { requires { field.write; } output result: Styles; compute { return Style { y: 2, z: 3 }; } }\n",
    )
    .unwrap();
    // A Kernel that uses both and names `Styles` reaches two, whichever
    // it uses first.
    for (first, second) in [("a", "b"), ("b", "a")] {
        std::fs::write(
            dir.join("scene.esn"),
            format!("use \"{first}/\";\nuse \"{second}/\";\nnode seed: t.a@1 {{ domain = 4; }}\nnode show: kernel {{ requires {{ field.read; field.write; }} input styles: Styles <- seed.result; output result: Image; domain = vec2(1, 1); compute {{ return vec4(float(count(styles)), 0.0, 0.0, 1.0); }} }}\noutput result <- show.result;\n"),
        )
        .unwrap();
        let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
        let refused = rejected(&session.sync()).join("\n");
        assert!(
            refused.contains("`Styles` is declared by"),
            "{first} first: {refused}"
        );
    }
    // Alike over alike structs, the two are one.
    std::fs::write(
        dir.join("b/b.ent"),
        "struct Style { x: Float; }\ntype Styles = Rows<Style>;\n",
    )
    .unwrap();
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let events = session.sync();
    assert_eq!(rejected(&events), Vec::<String>::new(), "{events:?}");
    let report = session.evaluate(&Context::default()).unwrap();
    assert_eq!(
        session.read_pixel(report.output, 0, 0).unwrap(),
        [4.0, 0.0, 0.0, 1.0]
    );

    // A document writing `Styles` alike over a `Style` of its own has its
    // own: a library's function of the library's `Styles` refuses it
    // where it is passed.
    std::fs::write(
        dir.join("a/a.ent"),
        "struct Style { x: Float; }\ntype Styles = Rows<Style>;\nfn first_x(s: Styles) -> Float { return at(s, 0).x; }\nkernel t.a@1 { requires { field.write; } output result: Styles; compute { return Style { x: 1.0 }; } }\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("own.ent"),
        "use \"a/\";\nstruct Style { q: Int; }\ntype Styles = Rows<Style>;\nkernel t.own@1 { requires { field.read; field.write; } input styles: Styles; output result: Image; compute { return vec4(first_x(styles), 0.0, 0.0, 1.0); } }\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "use \"own.ent\";\nnode show: t.own@1 { domain = vec2(1, 1); }\noutput result <- show.result;\n",
    )
    .unwrap();
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let refused = rejected(&session.sync()).join("\n");
    assert!(
        refused.starts_with("4:") && refused.contains("`first_x` expects"),
        "{refused}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn functions_are_shared_through_use_and_scoped_by_document() {
    let dir = scratch("functions");
    std::fs::write(
        dir.join("kernels/math.ent"),
        "fn twice(x: Float) -> Float { return x * 2.0; }
fn peek(f: Image, c: Vec2) -> Vec4 { return at(f, c); }
",
    )
    .unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "use \"kernels/\";
node seed: image.gradient@1 { domain = vec2(8, 8); }
node view: kernel { requires { field.read; field.write; } input cells: Image <- seed.result; output result: Image;
    compute { let c = peek(cells, coord()); return vec4(twice(c.r), c.g, c.b, 1.0); } }
output result <- view.result;
",
    )
    .unwrap();
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    let check = |scene: &str| {
        let output = Command::new(entelechy)
            .arg("check")
            .arg(dir.join(scene))
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        (output.status.success(), stdout)
    };
    let (ok, stdout) = check("scene.esn");
    assert!(ok, "{stdout}");
    // A second file declaring the name: the scene's own call reaches
    // neither, and the refusal names both files.
    std::fs::write(
        dir.join("kernels/more.ent"),
        "fn twice(x: Float) -> Float { return x + x; }\n",
    )
    .unwrap();
    let (ok, stdout) = check("scene.esn");
    assert!(!ok, "{stdout}");
    assert!(
        stdout.contains("fn `twice` is declared by")
            && stdout.contains("math.ent")
            && stdout.contains("more.ent"),
        "{stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A replay says how many pads are connected, and a Kernel reads it.
#[test]
fn a_replay_connects_pads_a_kernel_counts() {
    let dir = scratch("pads");
    std::fs::write(
        dir.join("pads.esn"),
        "node k: kernel { requires { field.write; input.read; } output result: Rows<Int>; compute { return pads(); } domain = 1; }\noutput result <- k.result;\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("pads.replay"),
        "scene pads.esn\nframe 0.0\nexpect k.result[0] = 0\npads 2\nframe 0.1\nexpect k.result[0] = 2\npads 0\nframe 0.2\nexpect k.result[0] = 0\n",
    )
    .unwrap();
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    for backend in ["cpu", "jit", "gpu"] {
        let output = Command::new(entelechy)
            .args(["replay", "--backend", backend])
            .arg(dir.join("pads.replay"))
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{backend}: {stdout}{stderr}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Two libraries each keep a helper of one name; a document using both
/// calls each library's functions, and each reaches its own helper. A
/// document's own function of a used one's name is its own, and the
/// used function keeps calling the one it sees.
#[test]
fn each_document_calls_the_functions_it_sees() {
    let dir = scratch("scoped");
    let lib = dir.join("lib");
    std::fs::create_dir_all(&lib).unwrap();
    std::fs::write(
        lib.join("a.ent"),
        "fn helper(x: Float) -> Float { return x * 2.0; }\nfn by_a(x: Float) -> Float { return helper(x); }\n",
    )
    .unwrap();
    std::fs::write(
        lib.join("b.ent"),
        "fn helper(x: Float) -> Float { return x + 10.0; }\nfn by_b(x: Float) -> Float { return helper(x); }\n",
    )
    .unwrap();
    std::fs::write(
        lib.join("c.ent"),
        "use \"a.ent\";
fn helper(x: Float) -> Float { return x * 3.0; }
kernel t.own@1 { requires { field.write; } output result: Rows<Float>; compute { return helper(1.0) + by_a(1.0) * 10.0; } }
",
    )
    .unwrap();
    std::fs::write(
        dir.join("scoped.esn"),
        "use \"lib/a.ent\";
use \"lib/b.ent\";
use \"lib/c.ent\";
node both: kernel { requires { field.write; } output result: Rows<Float>; compute { return by_a(1.0) + by_b(1.0) * 100.0; } domain = 1; }
node own: t.own@1 { domain = 1; }
output both <- both.result;
output own <- own.result;
",
    )
    .unwrap();
    std::fs::write(
        dir.join("scoped.replay"),
        "scene scoped.esn\nframe 0.0\nexpect both.result[0] = 1102.0\nexpect own.result[0] = 23.0\n",
    )
    .unwrap();
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    let output = Command::new(entelechy)
        .arg("replay")
        .arg(dir.join("scoped.replay"))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}{stderr}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn inputs_read_files_the_host_watches_and_outputs_save_as_files() {
    let dir = scratch("assets");
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    // A 2x1 picture file, read by an input and copied by a Kernel.
    let picture = |left: [u8; 3]| {
        let mut bytes = b"P6\n2 1\n255\n".to_vec();
        bytes.extend_from_slice(&left);
        bytes.extend_from_slice(&[0, 0, 255]);
        bytes
    };
    std::fs::write(dir.join("photo.ppm"), picture([255, 0, 0])).unwrap();
    std::fs::write(
        dir.join("assets.esn"),
        "input photo: Image = \"photo.ppm\";
node copy: kernel { requires { field.read; field.write; } input source: Image <- photo; output result: Image; compute { return sample(source); } }
output result <- copy.result;
",
    )
    .unwrap();
    let mut session = Session::open(&dir.join("assets.esn"), None, 1 << 20);
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
    let output = Command::new(entelechy)
        .arg("run")
        .arg(dir.join("assets.esn"))
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
    let (field, _) = entelechy_formats::read(&dir.join("copy.field"), None).unwrap();
    assert_eq!(field.domain(), entelechy_kernel::Domain::Grid(2, 1));
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
        entelechy_formats::png::encode(2, 1, &pixels),
    )
    .unwrap();
    let (as_pixels, detail) =
        entelechy_formats::read(&dir.join("photo.png"), Some("Image")).unwrap();
    assert!(
        as_pixels.schema().is_pixel() && detail.contains("Grid 2"),
        "{detail}"
    );
    let (as_gray, _) =
        entelechy_formats::read(&dir.join("photo.png"), Some("Grid<Float>")).unwrap();
    assert_eq!(as_gray.schema().value_words(), 1);
    assert!((f32::from_bits(as_gray.storage()[0]) - 0.2126).abs() < 1e-5);
    let sound: Vec<u32> = [0.0f32, 0.5, -0.5, 0.25]
        .iter()
        .map(|v| v.to_bits())
        .collect();
    std::fs::write(
        dir.join("hit.wav"),
        entelechy_formats::wav::encode(&sound, 2, 22050),
    )
    .unwrap();
    let (rows, detail) = entelechy_formats::read(&dir.join("hit.wav"), Some("Rows<Vec2>")).unwrap();
    assert_eq!(rows.domain(), entelechy_kernel::Domain::Rows(2));
    assert!(detail.ends_with("at 22050 Hz"), "{detail}");
    let output = Command::new(entelechy)
        .arg("run")
        .arg(dir.join("assets.esn"))
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
    let (again, _) = entelechy_formats::read(&dir.join("copy.png"), Some("Image")).unwrap();
    assert_eq!(f32::from_bits(again.storage()[0]), 1.0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_param_that_sizes_the_graph_lowers_it_again() {
    let dir = scratch("times");
    std::fs::write(
        dir.join("scene.esn"),
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
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    session.sync();
    assert_eq!(session.graph().unwrap().nodes.len(), 3);
    assert!(session.evaluate(&Context::default()).is_ok());
    session
        .set_param("level", entelechy_kernel::Value::Int(3))
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
        .set_param("other", entelechy_kernel::Value::Int(1))
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
            dir.join("label.esn"),
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
    let mut session = Session::open(&dir.join("label.esn"), None, 1 << 20);
    let events = session.sync();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::Text { name, chars: 2 } if name == "label")),
        "{events:?}"
    );
    session.pin(entelechy_graph::NodeId("count".into()));
    session.evaluate(&Context::default()).unwrap();
    let handle = session
        .cached(&entelechy_graph::NodeId("count".into()), "result")
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
        .cached(&entelechy_graph::NodeId("count".into()), "result")
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

/// A node whose element fails is traced up to its failure, and the
/// failure is placed where it is written, through a function call and
/// the loop accounting that renumbers the Kernel.
#[test]
fn a_failing_element_is_traced_to_where_it_is_written() {
    let dir = scratch("trace-fault");
    let scene = dir.join("fault.esn");
    std::fs::write(
        &scene,
        "node k: kernel {
    requires { field.write; }
    output result: Rows<Int>;
    budget 10;
    fn grow(x: Int) -> Int {
        let big = x * 1000000;
        return big * 1000;
    }
    compute {
        let i = element();
        let steps = loop 20 with t = 0 until t > 3 { t + 1 };
        let v = grow(i + steps - 3);
        let w = v + 1;
        return w;
    }
    domain = 8;
}
output result <- k.result;
",
    )
    .unwrap();
    let mut session = Session::open(&scene, None, 1 << 20);
    session.sync();
    let context = Context::default();
    let node = entelechy_graph::NodeId("k".into());
    let error = session.evaluate(&context).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("at element (2, 0): integer overflow"),
        "{error}"
    );
    let place = session.failure_place(&error, &context).expect("placed");
    assert_eq!((place.line, place.column), (7, 16), "{place}");
    assert_eq!(place.function.as_deref(), Some("grow"));
    assert_eq!(place.path, scene);

    // The node itself fails, and its element is still traced.
    let traced = session.trace(&node, [2, 0, 0], &context).unwrap();
    let fault = traced.trace.fault.expect("element 2 fails");
    assert_eq!(fault.function.as_deref(), Some("grow"));
    let value = |values: &[entelechy_session::TraceValue], name: &str| {
        values.iter().find(|v| v.name == name).cloned().unwrap()
    };
    assert_eq!(value(&fault.bindings, "big").words, vec![3_000_000]);
    assert!(value(&traced.trace.bindings, "steps").ran);
    assert!(!value(&traced.trace.bindings, "v").ran);
    assert!(!value(&traced.trace.bindings, "w").ran);
    assert_eq!(traced.place, Some(place));
    // An element that holds has no failure.
    let clean = session.trace(&node, [1, 0, 0], &context).unwrap();
    assert!(clean.trace.fault.is_none() && clean.place.is_none());

    // The command line says the same.
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    let output = Command::new(entelechy)
        .args(["trace", "--node", "k", "--element", "2"])
        .arg(&scene)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    for line in [
        "  v: Int = - (not run)",
        "  failed: at element (2, 0): integer overflow",
        ":7:16, in `grow`",
        "    big: Int = 3000000",
    ] {
        assert!(stdout.contains(line), "{line:?} in {stdout}");
    }
    let output = Command::new(entelechy)
        .args(["run", "--backend", "cpu"])
        .arg(&scene)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        stderr.contains("the failing operation is written at")
            && stderr.contains(":7:16, in `grow`"),
        "{stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A failure in a function a used document declares is placed in that
/// document, and so is one in a function another used document also
/// names.
#[test]
fn a_failure_in_a_used_function_is_placed_in_its_document() {
    let dir = scratch("trace-used");
    let lib = dir.join("lib");
    std::fs::create_dir_all(&lib).unwrap();
    std::fs::write(
        lib.join("a.ent"),
        "fn grow(x: Int) -> Int {\n    let big = x * 1000000;\n    return big * 1000;\n}\nfn by_a(x: Int) -> Int { return grow(x); }\n",
    )
    .unwrap();
    std::fs::write(
        lib.join("b.ent"),
        "fn grow(x: Int) -> Int { return x; }\nfn by_b(x: Int) -> Int { return grow(x); }\n",
    )
    .unwrap();
    let scene = dir.join("used.esn");
    let write = |call: &str| {
        std::fs::write(
            &scene,
            format!(
                "use \"lib/a.ent\";\n{}node k: kernel {{ requires {{ field.write; }} output result: Rows<Int>; compute {{ return {call}(element()) + by_a(0); }} domain = 4; }}\noutput result <- k.result;\n",
                if call == "by_b" { "use \"lib/b.ent\";\n" } else { "" }
            ),
        )
        .unwrap();
    };
    for call in ["by_a", "by_b"] {
        write(call);
        let mut session = Session::open(&scene, None, 1 << 20);
        session.sync();
        let context = Context::default();
        let error = session.evaluate(&context);
        if call == "by_b" {
            // b's `grow` passes its element through: nothing fails.
            assert!(error.is_ok(), "{error:?}");
            continue;
        }
        let error = error.unwrap_err();
        let place = session.failure_place(&error, &context).expect("placed");
        assert_eq!(
            place.path,
            lib.join("a.ent").canonicalize().unwrap(),
            "{place}"
        );
        assert_eq!((place.line, place.column), (3, 12), "{place}");
        assert_eq!(place.function.as_deref(), Some("grow"));
    }
    // With both libraries used, a's `grow` is renamed in the scene's
    // reach and still placed as a's `grow`.
    std::fs::write(
        &scene,
        "use \"lib/a.ent\";\nuse \"lib/b.ent\";\nnode k: kernel { requires { field.write; } output result: Rows<Int>; compute { return by_a(element()) + by_b(1); } domain = 4; }\noutput result <- k.result;\n",
    )
    .unwrap();
    let mut session = Session::open(&scene, None, 1 << 20);
    session.sync();
    let context = Context::default();
    let error = session.evaluate(&context).unwrap_err();
    let place = session.failure_place(&error, &context).expect("placed");
    assert_eq!(
        place.path,
        lib.join("a.ent").canonicalize().unwrap(),
        "{place}"
    );
    assert_eq!((place.line, place.column), (3, 12), "{place}");
    assert_eq!(place.function.as_deref(), Some("grow"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// An element found past its budget as it ends is placed at the loop
/// that ran the most iterations in all, here one inside a function.
#[test]
fn an_element_past_its_budget_is_placed_at_its_busiest_loop() {
    let dir = scratch("trace-budget");
    let scene = dir.join("budget.esn");
    std::fs::write(
        &scene,
        "node k: kernel {
    requires { field.write; }
    output result: Rows<Int>;
    budget 100;
    fn spin(t: Int) -> Int {
        return loop 30 with u = t { (u ^ index()) & 1023 };
    }
    compute {
        let small = loop 5 with a = element() { (a ^ 5) & 255 };
        let total = loop 4 with t = small { spin(t) };
        return total;
    }
    domain = 2;
}
output result <- k.result;
",
    )
    .unwrap();
    let mut session = Session::open(&scene, None, 1 << 20);
    session.sync();
    let context = Context::default();
    let error = session.evaluate(&context).unwrap_err();
    assert!(error.to_string().contains("loop iterations"), "{error}");
    let place = session.failure_place(&error, &context).expect("placed");
    assert_eq!((place.line, place.column), (6, 16), "{place}");
    assert_eq!(place.function.as_deref(), Some("spin"));
    // The element ran to its end: every `let` holds its value.
    let traced = session
        .trace(&entelechy_graph::NodeId("k".into()), [0, 0, 0], &context)
        .unwrap();
    assert!(traced.trace.bindings.iter().all(|value| value.ran));
    assert_eq!(traced.place, Some(place));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_binary_traces_one_step_of_a_scan_and_profiles_a_run() {
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    let dir = scratch("trace");
    // A tally over four memory cells, six steps a frame: the first
    // evaluation leaves [2, 2, 1, 1]; the next one's step 5 writes 4 to
    // cell 1 and takes the `if` branch with the product.
    std::fs::write(
        dir.join("kernels/tally.ent"),
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
        dir.join("tally.esn"),
        "use \"kernels/\";
node ticks: kernel { requires { field.write; } output result: Rows<Int>; domain = 6; compute { return 0; } }
node zeros: kernel { requires { field.write; } output result: Rows<Int>; domain = 4; compute { return 0; } }
node tally: tally.step@1 { ticks <- ticks.result; ram <~ tally.ram from zeros.result; }
output result <- tally.result;
",
    )
    .unwrap();
    let output = Command::new(entelechy)
        .args(["trace", "--node", "tally", "--element", "5"])
        .arg(dir.join("tally.esn"))
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
    let output = Command::new(entelechy)
        .args(["trace", "--node", "tally", "--element", "6"])
        .arg(dir.join("tally.esn"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("outside the domain"));
    // --profile names every executed node with its time.
    let output = Command::new(entelechy)
        .args(["run", "--profile", "--out"])
        .arg(dir.join("out.ppm"))
        .arg(dir.join("tally.esn"))
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(
        stderr.contains("  tally ") && stderr.contains(" ms"),
        "{stderr}"
    );
}

/// `entelechy help` prints the whole reference and the commands; a name shows
/// every item of it in full.
#[test]
fn help_prints_the_reference_and_shows_one() {
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    let output = Command::new(entelechy).arg("help").output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    for item in [
        "**atan2**",
        "**put**",
        "**each**",
        "**fill**",
        "**publish**",
        "**expect**",
        "**serve**",
    ] {
        assert!(stdout.contains(item), "{item}");
    }
    let output = Command::new(entelechy)
        .args(["help", "each"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("[each n { body }]"), "{stdout}");
    assert!(stdout.contains("element by element"), "{stdout}");
    // `replay` names a replay line and a command: the answer is both.
    let output = Command::new(entelechy)
        .args(["help", "replay"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("entelechy replay <file.replay>"),
        "{stdout}"
    );
    let output = Command::new(entelechy)
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
    let output = Command::new(entelechy)
        .args(["help", "sample"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Needs `field.read`") && stdout.contains("Only the entry"),
        "{stdout}"
    );
    let output = Command::new(entelechy)
        .args(["help", "atan3"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("did you mean") && stderr.contains("atan"),
        "{stderr}"
    );
    let output = Command::new(entelechy)
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
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    let dir = std::env::temp_dir().join(format!("entelechy-packages-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    // A package of two Kernels and a preset graph made of both.
    let example = dir.join("source/pixelfx.entpkg");
    std::fs::create_dir_all(example.join("kernels")).unwrap();
    std::fs::create_dir_all(example.join("graphs")).unwrap();
    std::fs::write(
        example.join("package.esn"),
        "package pixelfx@1 { language = 1; ir = 1; }\nuse \"kernels/\";\n",
    )
    .unwrap();
    std::fs::write(
        example.join("kernels/pixelfx.ent"),
        "kernel pixelfx.pixelate@1 { requires { field.read; field.write; } input source: Image; param cell: Float = 8.0; output result: Image; compute { return at(source, floor(coord() / cell) * cell + cell * 0.5); } }\n\
         kernel pixelfx.palette@1 { requires { field.read; field.write; } input source: Image; output result: Image; compute { let c = sample(source); let k = clamp(int((c.r + c.g + c.b) / 3.0 * 4.0), 0, 3); return vec4(float(k) / 3.0, float(k) / 3.0, float(k) / 3.0, 1.0); } }\n",
    )
    .unwrap();
    std::fs::write(
        example.join("graphs/nes.esn"),
        "use \"../kernels/\";\ninput source: Image;\nparam cell: Float = 8.0;\nnode coarse: pixelfx.pixelate@1 { source <- source; cell = cell; }\nnode toned: pixelfx.palette@1 { source <- coarse.result; }\noutput result <- toned.result;\n",
    )
    .unwrap();
    let archive = dir.join("pixelfx.entpkg");
    let output = Command::new(entelechy)
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
    let installed = dir.join("installed");
    let unpacked = installed.join("pixelfx.entpkg");
    let output = Command::new(entelechy)
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
    // A directory is named for the package it holds: unpacked under
    // another name, it is refused and nothing is left.
    let misnamed = dir.join("other.entpkg");
    let output = Command::new(entelechy)
        .arg("unpack")
        .arg(&archive)
        .arg(&misnamed)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("a package's directory is named for it, `pixelfx.entpkg`"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!misnamed.exists());
    for file in ["package.esn", "kernels/pixelfx.ent", "graphs/nes.esn"] {
        assert_eq!(
            std::fs::read(example.join(file)).unwrap(),
            std::fs::read(unpacked.join(file)).unwrap(),
            "{file} came back as it went"
        );
    }
    let output = Command::new(entelechy)
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
    // ENTELECHY_PACKAGES.
    let elsewhere = dir.join("elsewhere");
    std::fs::create_dir_all(&elsewhere).unwrap();
    std::fs::write(
        elsewhere.join("scene.esn"),
        "use \"pixelfx.entpkg\";\nnode source: kernel { requires { field.write; } output result: Image; domain = vec2(8, 8); compute { return vec4(0.5, 0.5, 0.5, 1.0); } }\nnode toned: pixelfx.palette@1 { source <- source.result; }\noutput result <- toned.result;\n",
    )
    .unwrap();
    let output = Command::new(entelechy)
        .arg("check")
        .arg(elsewhere.join("scene.esn"))
        .env("ENTELECHY_PACKAGES", &installed)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    assert!(
        stdout.contains("graph `scene` with 2 node(s), 1 use(s)"),
        "{stdout}"
    );
    // `bench` finds it the same way.
    let output = Command::new(entelechy)
        .arg("bench")
        .arg(elsewhere.join("scene.esn"))
        .args(["--backend", "cpu", "--frames", "1", "--repeat", "1"])
        .env("ENTELECHY_PACKAGES", &installed)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // A preset inside it is placed by the package name, and the preset's
    // own `use` resolves from the package's real location.
    std::fs::write(
        elsewhere.join("preset.esn"),
        "use \"pixelfx.entpkg\";\nnode source: kernel { requires { field.write; } output result: Image; domain = vec2(8, 8); compute { return vec4(0.5, 0.5, 0.5, 1.0); } }\nnode look: graph \"pixelfx.entpkg/graphs/nes.esn\" { source <- source.result; cell = 4.0; }\noutput result <- look.result;\n",
    )
    .unwrap();
    let output = Command::new(entelechy)
        .arg("check")
        .arg(elsewhere.join("preset.esn"))
        .env("ENTELECHY_PACKAGES", &installed)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    assert!(stdout.contains("graph `preset` with 3 node(s)"), "{stdout}");
    // Where the command runs finds no package: from the directory that
    // holds it, without ENTELECHY_PACKAGES, the name is not found either,
    // and the check says where it looked and how to look further.
    let output = Command::new(entelechy)
        .arg("check")
        .arg("elsewhere/scene.esn")
        .current_dir(&dir)
        .env_remove("ENTELECHY_PACKAGES")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!output.status.success(), "{stdout}");
    assert!(
        stdout.contains("no package `pixelfx.entpkg` is found: not at `elsewhere/pixelfx.entpkg`, no package directory is set; add the directory that holds `pixelfx.entpkg` to `ENTELECHY_PACKAGES`"),
        "{stdout}"
    );
    // `package` checks what the package holds: a Kernel that does not
    // compile is a problem.
    let broken = dir.join("broken.entpkg");
    std::fs::create_dir_all(broken.join("kernels")).unwrap();
    std::fs::write(
        broken.join("package.esn"),
        "package broken@1 { language = 1; ir = 1; }\nuse \"kernels/\";\n",
    )
    .unwrap();
    std::fs::write(
        broken.join("kernels/k.ent"),
        "kernel broken.k@1 { requires { field.write; } output result: Rows<Int>; compute { return nope; } }\n",
    )
    .unwrap();
    let output = Command::new(entelechy)
        .arg("package")
        .arg(&broken)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "{stderr}");
    assert!(stderr.contains("`nope` is not defined"), "{stderr}");

    // A package needing a newer language is refused with both versions.
    let manifest = unpacked.join("package.esn");
    let text = std::fs::read_to_string(&manifest).unwrap();
    std::fs::write(&manifest, text.replace("language = 1", "language = 99")).unwrap();
    let output = Command::new(entelechy)
        .arg("check")
        .arg(elsewhere.join("scene.esn"))
        .env("ENTELECHY_PACKAGES", &installed)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!output.status.success(), "{stdout}");
    assert!(
        stdout.contains("package pixelfx@1 needs language 99, and this Entelechy has language 1"),
        "{stdout}"
    );
    let output = Command::new(entelechy).arg("version").output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.starts_with(&format!("entelechy = {}\n", env!("CARGO_PKG_VERSION")))
            && stdout.contains("entelechy.language = 1")
            && stdout.contains("entelechy.package = 1"),
        "{stdout}"
    );
}

#[test]
fn render_writes_a_png_per_frame_and_one_wav_of_every_block() {
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    let scenes =
        std::env::temp_dir().join(format!("entelechy-render-scenes-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scenes);
    std::fs::create_dir_all(&scenes).unwrap();
    // A tone of 800 stereo frames a block, by the clock, and its trace.
    let tone = scenes.join("sound.esn");
    std::fs::write(
        &tone,
        "node tone: kernel { requires { field.write; time.read; } output result: Rows<Vec2>; domain = 800;
    compute { let t = time() + float(element()) / 48000.0; let v = sin(t * 1382.3) * 0.4; return vec2(v, v * 0.5); } }
node scope: kernel { requires { field.read; field.write; time.read; } input block: Rows<Vec2> <- tone.result; output result: Image; domain = vec2(320, 240);
    compute { let s = at(block, int(coord().x * 2.5)); let on = abs(coord().y - 120.0 - s.x * 90.0) < 1.5; return vec4(if on { 1.0 } else { 0.0 }, fract(time()), 0.0, 1.0); } }
output view <- scope.result;
output sound <- tone.result;
",
    )
    .unwrap();
    // A count fed back off the frame, shown in a window-sized picture.
    let machine_scene = scenes.join("async.esn");
    std::fs::write(
        &machine_scene,
        "kernel count.step@1 { requires { field.read; field.write; } input previous: Rows<Int>; output result: Rows<Int>; compute { return at(previous, 0) + 1; } }
node zero: kernel { requires { field.write; } output result: Rows<Int>; domain = 1; compute { return 0; } }
node counter: count.step@1 { previous <~ counter.result from zero.result; async = true; }
node view: kernel { requires { field.read; field.write; input.read; } input count: Rows<Int> <- counter.result; output result: Image; domain = window();
    compute { return vec4(float(at(count, 0)) / 8.0, 0.0, 0.0, 1.0); } }
output result <- view.result;
",
    )
    .unwrap();
    let out = std::env::temp_dir().join(format!("entelechy-render-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    let output = Command::new(entelechy)
        .args(["render", "--frames", "3", "--out"])
        .arg(&out)
        .arg(&tone)
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
        let bytes = std::fs::read(entelechy::render::picture_path(&out, index)).unwrap();
        let picture = entelechy_formats::png::decode(&bytes).unwrap();
        assert_eq!((picture.width, picture.height), (320, 240));
    }
    assert!(!out.join("0003.png").exists());
    let sound = entelechy_formats::wav::decode(
        &std::fs::read(out.join(entelechy::render::SOUND_NAME)).unwrap(),
    )
    .unwrap();
    assert_eq!(sound.rate, 48_000);
    assert_eq!(
        sound.field.domain().count(),
        3 * 800,
        "three blocks of 800 frames"
    );
    // The same frames give the same bytes: the clock is the count.
    let first = std::fs::read(entelechy::render::picture_path(&out, 2)).unwrap();
    let again = std::env::temp_dir().join(format!("entelechy-render-again-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&again);
    assert!(
        Command::new(entelechy)
            .args(["render", "--frames", "3", "--out"])
            .arg(&again)
            .arg(&tone)
            .status()
            .unwrap()
            .success()
    );
    assert_eq!(
        std::fs::read(entelechy::render::picture_path(&again, 2)).unwrap(),
        first
    );
    // An async node is settled after every frame: each picture shows the
    // step before it, none skipped.
    let machine =
        std::env::temp_dir().join(format!("entelechy-render-async-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&machine);
    let output = Command::new(entelechy)
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
        .arg(&machine_scene)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(
        stderr.contains("rendered 2 picture(s) of 32x32"),
        "{stderr}"
    );
    for dir in [&out, &again, &machine, &scenes] {
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[test]
fn a_frames_input_brings_in_the_frame_of_the_moment_and_wraps() {
    let dir = std::env::temp_dir().join(format!("entelechy-frames-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("clip")).unwrap();
    for (index, color) in [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
        .iter()
        .enumerate()
    {
        let pixels = vec![*color; 4];
        std::fs::write(
            dir.join(format!("clip/{index}.png")),
            entelechy_formats::png::encode(2, 2, &pixels),
        )
        .unwrap();
    }
    std::fs::write(
        dir.join("scene.esn"),
        "input clip: Image = frames \"clip/\" at 1;\nnode view: kernel { requires { field.read; field.write; } input clip: Image <- clip; output result: Image; compute { return sample(clip); } }\noutput result <- view.result;\n",
    )
    .unwrap();
    let mut session = Session::open(
        &dir.join("scene.esn"),
        None,
        entelechy_session::DEFAULT_BUDGET_WORDS,
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
        entelechy_formats::png::encode(2, 2, &[[0.0, 0.0, 1.0, 1.0]; 4]),
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
    let dir = std::env::temp_dir().join(format!("entelechy-fill-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("scene.esn"),
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
        &dir.join("scene.esn"),
        None,
        entelechy_session::DEFAULT_BUDGET_WORDS,
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
        .set_param("level", entelechy_kernel::Value::Float(1.0))
        .unwrap();
    session
        .set_param("depth", entelechy_kernel::Value::Int(10))
        .unwrap();
    session.sync();
    let report = session.evaluate(&Context::default()).unwrap();
    assert_eq!(read(&mut session, report.output), vec![11.0; 4]);
    // A count of zero yields what the loop carries.
    session
        .set_param("depth", entelechy_kernel::Value::Int(0))
        .unwrap();
    session.sync();
    let report = session.evaluate(&Context::default()).unwrap();
    assert_eq!(read(&mut session, report.output), vec![1.0; 4]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_name_a_kernel_cannot_find_names_the_file_that_declares_it() {
    let dir = std::env::temp_dir().join(format!("entelechy-located-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("kernels")).unwrap();
    std::fs::write(
        dir.join("kernels/util.ent"),
        "fn wobble(x: Float) -> Float { return x * 1.5; }\nkernel util.nothing@1 { requires { field.write; } output result: Rows<Float>; compute { return 0.0; } }\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("kernels/main.ent"),
        "kernel main.k@1 { requires { field.write; } output result: Rows<Float>; compute { return wobble(2.0); } }\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "use \"kernels/main.ent\";\nnode k: main.k@1 { domain = 2; }\noutput result <- k.result;\n",
    )
    .unwrap();
    let mut session = Session::open(
        &dir.join("scene.esn"),
        None,
        entelechy_session::DEFAULT_BUDGET_WORDS,
    );
    let events = session.sync();
    let reported = rejected(&events).join("\n");
    assert!(reported.contains("`wobble` is not a builtin"), "{reported}");
    assert!(
        reported.contains("fn `wobble` is declared in util.ent"),
        "{reported}"
    );
    assert!(reported.contains("`use \"util.ent\";`"), "{reported}");
    // Two documents using each other are reported as a cycle.
    std::fs::write(
        dir.join("kernels/main.ent"),
        "use \"util.ent\";\nstruct Tint { r: Float; }\nkernel main.k@1 { requires { field.write; } output result: Rows<Float>; compute { return wobble(2.0); } }\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("kernels/util.ent"),
        "use \"main.ent\";\nfn wobble(x: Float) -> Float { return x * 1.5; }\nkernel util.nothing@1 { requires { field.write; } output result: Rows<Float>; compute { return 0.0; } }\n",
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
    let dir = std::env::temp_dir().join(format!("entelechy-bytes-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("blob.bin"), [79u8, 82, 76, 65, 0, 255, 128]).unwrap();
    std::fs::write(
        dir.join("scene.esn"),
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
        &dir.join("scene.esn"),
        None,
        entelechy_session::DEFAULT_BUDGET_WORDS,
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

/// A grid carried from frame to frame through a graph placed twice, each
/// placement a node placed three times; the scene's path in `dir`.
fn steps(dir: &Path) -> PathBuf {
    std::fs::write(
        dir.join("steps.ent"),
        "kernel steps.start@1 { requires { field.write; } output result: Grid<Int>; compute { return int(coord().x) + int(coord().y) * 8; } }
kernel steps.spread@1 { requires { field.read; field.write; } input cells: Grid<Int>; output result: Grid<Int>; compute { return (sample(cells) * 5 + 1) % 97; } }
kernel steps.view@1 { requires { field.read; field.write; } input cells: Grid<Int>; output result: Image; compute { let v = float(at(cells, coord() / 8.0)) / 96.0; return vec4(v, v, v, 1.0); } }
",
    )
    .unwrap();
    std::fs::write(
        dir.join("step.esn"),
        "use \"steps.ent\";
input cells: Grid<Int>;
node spread: steps.spread@1 { times = 3; carry cells <- result; cells <- cells; }
output result <- spread.result;
",
    )
    .unwrap();
    std::fs::write(
        dir.join("steps.esn"),
        "use \"steps.ent\";
node start: steps.start@1 { domain = vec2(8, 4); }
node steps: graph \"step.esn\" { times = 2; carry cells <- result; cells <~ steps.result from start.result; }
node view: steps.view@1 { cells <- steps.result; domain = vec2(64, 32); }
output result <- view.result;
output grid <- steps.result;
",
    )
    .unwrap();
    dir.join("steps.esn")
}

/// A placed graph's port follows the last dot, and a repeated node's name
/// is its last time's; any other name is reported as what it is.
#[test]
fn a_save_names_a_node_a_placed_graphs_node_or_an_output() {
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    let dir = scratch("saves");
    let scene = steps(&dir);
    let run = |target: &str, file: &str| {
        Command::new(entelechy)
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
    let output = run("steps/2.spread/3.result", "by-node.field");
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
    // A repeated node's name is its last time's, as the scene's own
    // `output grid <- steps.result;` means it.
    let output = run("steps.result", "by-name.field");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        std::fs::read(dir.join("by-output.field")).unwrap(),
        std::fs::read(dir.join("by-name.field")).unwrap()
    );
    // A placed graph's port that is none of its outputs names nothing.
    let output = run("steps/2.cells", "nothing.field");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        stderr.contains("is not a node: it is placed as"),
        "{stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `bench` reports the least of several runs; its `--json` output is read
/// back by `--compare`.
#[test]
fn bench_measures_frames_and_compares_with_what_it_wrote() {
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    let dir = scratch("bench");
    let json = dir.join("before.json");
    // A scene carried from frame to frame, so its nodes run each frame.
    let scene = steps(&dir);
    let measure = |extra: &[&str]| {
        Command::new(entelechy)
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
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    let dir = scratch("unloaded");
    let scene = dir.join("broken.esn");
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
        let output = Command::new(entelechy)
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
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
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
            .publish_source(&source, entelechy_kernel::Expect::Any)
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

/// Every `entelechy new` template is made, checked and replayed.
#[test]
fn what_new_writes_checks_and_replays() {
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    let root = scratch("new");
    for kind in ["picture", "sound", "game"] {
        let dir = root.join(kind);
        let made = Command::new(entelechy)
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
        assert!(dir.join("scene.esn").exists(), "{listed}");

        let checked = Command::new(entelechy)
            .arg("check")
            .arg(dir.join("scene.esn"))
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
        let tested = Command::new(entelechy)
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

        let again = Command::new(entelechy)
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
    let dir = std::env::temp_dir().join(format!("entelechy-budget-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // A grid of a million pixels: four million words, copied once more by
    // the second node.
    std::fs::write(
        dir.join("scene.esn"),
        "node fill: kernel { requires { field.write; } output result: Image; domain = vec2(1024, 1024); compute { return vec4(coord().x, coord().y, 0.0, 1.0); } }\n\
         node keep: kernel { requires { field.read; field.write; } input source: Image <- fill.result; output result: Image; domain = vec2(1024, 1024); compute { return sample(source) * 0.5; } }\n\
         output result <- keep.result;\n",
    )
    .unwrap();
    let scene = dir.join("scene.esn");

    // Two mebibytes hold neither.
    let mut small = Session::open(&scene, None, entelechy_session::budget_of_megabytes(2));
    small.sync();
    let refused = small.evaluate(&Context::default()).unwrap_err().to_string();
    assert!(
        refused.contains("resource budget exceeded") && refused.contains("available"),
        "{refused}"
    );

    // Sixty-four hold both, with the words the Kernel wrote.
    let mut enough = Session::open(&scene, None, entelechy_session::budget_of_megabytes(64));
    enough.sync();
    let report = enough.evaluate(&Context::default()).unwrap();
    let image = enough.runtime().host().image(report.output).unwrap();
    assert_eq!(image.width(), 1024);
    assert_eq!(image.pixels()[100][0], 50.0, "half of that pixel's own x");

    assert_eq!(entelechy_session::budget_of_megabytes(1), 256 * 1024);
    let _ = std::fs::remove_dir_all(&dir);
}

/// `times` resolves when the document is lowered.
#[test]
fn a_parents_param_reaches_a_placed_graphs_times() {
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    let dir = std::env::temp_dir().join(format!("entelechy-times-param-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("step.ent"),
        "kernel t.zero@1 { requires { field.write; } output result: Rows<Int>; compute { return 0; } }\nkernel t.inc@1 { requires { field.read; field.write; } input previous: Rows<Int>; output result: Rows<Int>; compute { return sample(previous) + 1; } }\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("chain.esn"),
        "use \"step.ent\";\nparam n: Int = 5;\ninput seed: Rows<Int>;\nnode step: t.inc@1 { times = n; carry previous <- result; previous <- seed; }\noutput result <- step.result;\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "use \"step.ent\";\nparam rounds: Int = 2;\nnode seed: t.zero@1 { domain = 4; }\nnode chain: graph \"chain.esn\" { seed <- seed.result; n = rounds; }\noutput result <- chain.result;\n",
    )
    .unwrap();
    let output = Command::new(entelechy)
        .arg("check")
        .arg(dir.join("scene.esn"))
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
    // File data and Kernel outputs are narrow, and a narrow output saved
    // as a Field file round-trips its bytes.
    let dir = scratch("narrow-save");
    let pixels: Vec<[f32; 4]> = (0..64)
        .map(|k| [k as f32 / 63.0, 0.5, 1.0 - k as f32 / 63.0, 1.0])
        .collect();
    std::fs::write(
        dir.join("albedo.png"),
        entelechy_formats::png::encode(8, 8, &pixels),
    )
    .unwrap();
    std::fs::write(dir.join("tiles.bin"), [0u8, 16, 32, 48]).unwrap();
    let scene = dir.join("packed.esn");
    std::fs::write(
        &scene,
        "input albedo: Grid<Vec4 as unorm8> = \"albedo.png\";
input tiles: Rows<Int as u8> = bytes \"tiles.bin\";
node height: kernel { requires { field.write; } output result: Grid<Float as unorm16>; domain = vec2(16, 9);
    compute { return 0.5 + 0.25 * sin(coord().x * 0.7) * cos(coord().y * 0.9); } }
node normal: kernel { requires { field.read; field.write; } input height: Grid<Float as unorm16> <- height.result; output result: Grid<Vec3 as snorm8>;
    compute { let c = icoord(); let dx = at(height, c + ivec2(1, 0)) - at(height, c - ivec2(1, 0)); return normalize(vec3(-dx * 6.0, 0.3, 1.0)); } }
node dark: Grid<Vec4 as f16> { domain = vec2(16, 9); }
node light: kernel { requires { field.read; field.write; } input albedo: Grid<Vec4 as unorm8> <- albedo; input normal: Grid<Vec3 as snorm8> <- normal.result;
    input tiles: Rows<Int as u8> <- tiles; input previous: Grid<Vec4 as f16> <~ light.result from dark.result; output result: Grid<Vec4 as f16>; domain = vec2(16, 9);
    compute { let a = at(albedo, coord() / 2.0); let tile = at(tiles, int(coord().x) % 4); let lit = a.rgb * max(sample(normal).z, 0.0) * (1.0 + float(tile) / 64.0) * 3.0;
        return mix(sample(previous), vec4(lit, 1.0), 0.25); } }
output result <- light.result;
",
    )
    .unwrap();
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
    assert_eq!(words(albedo), 8 * 8, "a word a pixel, not four");
    let tiles = session.external("tiles").unwrap();
    assert_eq!(words(tiles), 4 / 4, "a byte a tile");
    let node = |name: &str| {
        session
            .cached(&entelechy_graph::NodeId(name.into()), "result")
            .unwrap()
    };
    assert_eq!(words(node("height")), 16 * 9 * 2 / 4);
    assert_eq!(words(node("normal")), 16 * 9 * 3 / 4);
    assert_eq!(words(node("light")), 16 * 9 * 8 / 4);
    let normal = node("normal");

    let file = dir.join("normal.field");
    session.allow_write_to(&file).unwrap();
    session
        .save(&entelechy_graph::NodeId("normal".into()), "result", &file)
        .unwrap();
    let bytes = std::fs::read(&file).unwrap();
    let header = b"ENTELECHYFIELD 1\nGrid 16 9 1 f:snorm8,f:snorm8,f:snorm8\n";
    assert!(bytes.starts_with(header));
    assert_eq!(bytes.len(), header.len() + 16 * 9 * 3);
    let back = entelechy_formats::file::decode(&bytes).unwrap();
    let host = session.runtime().host();
    assert_eq!(&back, host.data(normal).unwrap());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_parameter_of_any_value_type_is_set_by_its_layout() {
    // A camera that is a Mat4 and a style that is a struct: a value of
    // their layout is taken in their shape; another layout is refused by
    // name.
    let dir = scratch("value-params");
    std::fs::write(
        dir.join("draw.ent"),
        "struct Style { ground: Vec3; edge: Vec3; width: Float; }
kernel draw.plane@1 {
    requires { field.write; }
    param lens: Mat4 = mat4(vec4(2.2, 0.0, 0.0, 0.0), vec4(0.0, 2.2, 0.0, 0.0), vec4(0.0, 0.0, 1.0, -1.0), vec4(0.0, 0.0, 0.0, 0.0));
    param style: Style = Style { ground: vec3(0.06, 0.07, 0.1), edge: vec3(1.0, 0.95, 0.8), width: 1.0 };
    output result: Image;
    compute {
        let p = lens * vec4(coord() / domain() - vec2(0.5, 0.5), 1.0, 1.0);
        let on = abs(p.x) < style.width * 0.1;
        let c = if on { style.edge } else { style.ground };
        return vec4(c.x, c.y, c.z, 1.0);
    }
}
",
    )
    .unwrap();
    let scene = dir.join("plane.esn");
    std::fs::write(
        &scene,
        "use \"draw.ent\";
param lens: Mat4 = mat4(vec4(2.2, 0.0, 0.0, 0.0), vec4(0.0, 2.2, 0.0, 0.0), vec4(0.0, 0.0, 1.0, -1.0), vec4(0.0, 0.0, 0.0, 0.0));
param style: Style = Style { ground: vec3(0.06, 0.07, 0.1), edge: vec3(1.0, 0.95, 0.8), width: 1.0 };
node plane: draw.plane@1 { domain = vec2(32, 24); lens = lens; style = style; }
output result <- plane.result;
",
    )
    .unwrap();
    let mut session = Session::open(&scene, None, 1 << 22);
    assert!(session.sync().contains(&Event::SceneLoaded));
    use entelechy_kernel::Value;
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
    let _ = std::fs::remove_dir_all(&dir);
}

/// The interpreter declining every module holding a `Double`: a device
/// without doubles, for the session to route around.
struct NoDoubles(entelechy_cpu::CpuBackend);

impl entelechy_kernel::Backend for NoDoubles {
    fn name(&self) -> &'static str {
        "device"
    }
    fn declines(&self, module: &entelechy_ir::Validated) -> Option<String> {
        module
            .holds(entelechy_ir::Type::Double)
            .then(|| "it holds a Double".to_owned())
    }
    fn prepare(
        &self,
        module: &entelechy_ir::Validated,
    ) -> Result<Box<dyn entelechy_kernel::Prepared>, String> {
        self.0.prepare(module)
    }
    fn execute(
        &self,
        prepared: &dyn entelechy_kernel::Prepared,
        module: &entelechy_ir::Validated,
        invocation: &entelechy_kernel::Invocation,
        scope: &mut entelechy_kernel::Scope<'_>,
    ) -> Result<Vec<entelechy_kernel::Handle>, entelechy_kernel::ExecutionError> {
        entelechy_kernel::Backend::execute(&self.0, prepared, module, invocation, scope)
    }
}

#[test]
fn a_node_off_the_main_backend_is_said_when_the_scene_loads() {
    let dir = scratch("routed");
    std::fs::write(
        dir.join("scene.esn"),
        "node wide: kernel { requires { field.write; } output result: Rows<Double>; domain = 4;
            compute { return double(element()) * 0.1; } }
node plain: kernel { requires { field.write; } output result: Rows<Float>; domain = 4;
            compute { return float(element()); } }
output result <- plain.result;
",
    )
    .unwrap();
    let runtime = entelechy_kernel::Runtime::with_backends(
        1 << 20,
        Box::new(NoDoubles(entelechy_cpu::CpuBackend)),
        Some(Box::new(entelechy_cpu::CpuBackend)),
    );
    let mut session = Session::open_with(&dir.join("scene.esn"), None, runtime);
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
        entelechy_session::Limits {
            words: 1 << 22,
            ..Default::default()
        },
    ) else {
        return;
    };
    let dir = scratch("device-failure");
    std::fs::write(
        dir.join("scene.esn"),
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
        entelechy_session::Limits {
            words: 1 << 22,
            ..Default::default()
        },
    )
    .unwrap();
    let mut sessions = [
        Session::open_with(&dir.join("scene.esn"), None, reference),
        Session::open_with(&dir.join("scene.esn"), None, device),
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
        session
            .set_param("k", entelechy_kernel::Value::Int(1))
            .unwrap();
        // Element 2 overflows each time: nothing from the failed run is
        // kept.
        answers.push(evaluate(session));
        answers.push(evaluate(session));
        session
            .set_param("k", entelechy_kernel::Value::Int(10))
            .unwrap();
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
            std::fs::write(dir.join("scene.esn"), text).unwrap();
            let limits = entelechy_session::Limits {
                words: 1 << 22,
                work,
                small: 0,
            };
            for backend in ["cpu", "jit", "gpu"] {
                let Ok(runtime) = Session::runtime_for(backend, limits) else {
                    continue;
                };
                let mut session = Session::open_with(&dir.join("scene.esn"), None, runtime);
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
        std::fs::write(dir.join("scene.esn"), text).unwrap();
        for backend in ["cpu", "jit", "gpu"] {
            for (fusion, pinned) in [(true, false), (false, false), (true, true)] {
                let limits = entelechy_session::Limits {
                    words: 1 << 22,
                    work,
                    small: 0,
                };
                let Ok(runtime) = Session::runtime_for(backend, limits) else {
                    continue;
                };
                let mut session = Session::open_with(&dir.join("scene.esn"), None, runtime);
                session.set_fusion(fusion);
                if pinned {
                    session.pin(entelechy_graph::NodeId("a".into()));
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
        std::fs::write(dir.join("scene.esn"), scene).unwrap();
        let mut answers = Vec::new();
        for backend in ["cpu", "jit", "gpu"] {
            let Ok(runtime) = Session::runtime_for(
                backend,
                entelechy_session::Limits {
                    words: 1 << 22,
                    ..Default::default()
                },
            ) else {
                continue;
            };
            let mut session = Session::open_with(&dir.join("scene.esn"), None, runtime);
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

/// A domain is charged before any allocation: past the budget or `Int`
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
            "holds more than 2147483647 elements",
        ),
        (
            "domain = vec2(4294967295, 4294967295);",
            "Grid<Int>",
            "holds more than 2147483647 elements",
        ),
    ];
    let dir = scratch("huge-domain");
    for backend in ["cpu", "jit", "gpu"] {
        let Ok(runtime) = Session::runtime_for(backend, Default::default()) else {
            continue;
        };
        std::fs::write(dir.join("scene.esn"), "node w: kernel { requires { field.write; } output result: Rows<Int>; domain = 4; compute { return 1; } }\noutput result <- w.result;").unwrap();
        let mut session = Session::open_with(&dir.join("scene.esn"), None, runtime);
        for (domain, ty, why) in scenes {
            std::fs::write(
                dir.join("scene.esn"),
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
        std::fs::write(dir.join("scene.esn"), "node w: kernel { requires { field.write; } output result: Rows<Int>; domain = 8; compute { return 2; } }\noutput result <- w.result;").unwrap();
        session.sync();
        assert!(session.evaluate(&Context::default()).is_ok(), "{backend}");
    }
}

/// A document's Kernels change together: when one of two Kernels sharing
/// a changed struct does not compile, neither publishes and the running
/// pair stays. Fixed, both land in the same sync, on the worker and
/// inline.
#[test]
fn a_documents_kernels_publish_together_or_not_at_all() {
    let dir = std::env::temp_dir().join(format!("entelechy-together-{}", std::process::id()));
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
    std::fs::write(dir.join("pair.ent"), pair("a", false)).unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "use \"pair.ent\";\nnode make: t.make@1 { domain = 3; }\nnode use: t.use@1 { p <- make.result; }\noutput result <- use.result;\n",
    )
    .unwrap();
    for background in [false, true] {
        let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
        session.set_background(background);
        std::fs::write(dir.join("pair.ent"), pair("a", false)).unwrap();
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
        std::fs::write(dir.join("pair.ent"), pair("b", true)).unwrap();
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

        std::fs::write(dir.join("pair.ent"), pair("b", false)).unwrap();
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
/// window opens) holds what `entelechy replay` holds.
#[test]
fn a_replay_played_as_a_window_plays_it_holds_as_it_does_headless() {
    // A probe of the pointer, named by the replay but read by no output,
    // and a picture the window sizes.
    let base = std::env::temp_dir().join(format!("entelechy-windowed-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    std::fs::write(
        base.join("pointer.esn"),
        "node probe: kernel { requires { field.write; input.read; } output result: Rows<Int>; domain = 2;
    compute { return if element() == 0 { int(pointer().x) } else if pointer_down() { 1 } else { 0 }; } }
node view: kernel { requires { field.read; field.write; input.read; } input probe: Rows<Int> <- probe.result; output result: Image; domain = window();
    compute { return if coord().x < float(at(probe, 0)) { vec4(1.0, 1.0, 1.0, 1.0) } else { vec4(0.0, 0.0, 0.0, 1.0) }; } }
output result <- view.result;
",
    )
    .unwrap();
    let text = "scene pointer.esn
window 40 20
frame 0.0
pointer 10 5 up
frame 0.02
expect probe.result[0] = 10
expect probe.result[1] = 0
expect 4 4 = 1 1 1 0.01
expect 20 4 = 0 0 0 0.01
pointer 30 5 down
frame 0.04
expect probe.result[0] = 30
expect probe.result[1] = 1
expect 20 4 = 1 1 1 0.01
";
    for name in ["pointer.replay"] {
        let replay = entelechy::replay::parse(text, &base).unwrap();
        let runtime = Session::runtime_for("jit", Default::default()).unwrap();
        let mut session = Session::open_with(&replay.scene, replay.kernels.as_deref(), runtime);
        session.sync();
        let size = replay.steps.iter().find_map(|step| match step {
            entelechy::replay::Step::Window { size, scale } => Some((*size, *scale)),
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
        let mut replayer = entelechy::replay::Replayer::start(&replay, &mut session);
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
        assert_eq!(outcome.checks, 7, "{name}");
    }
    let _ = std::fs::remove_dir_all(&base);
}

/// An undeclared param set before load stops the load with why; a held
/// value goes when its declaration goes.
#[test]
fn a_param_is_one_the_document_declares() {
    let dir = scratch("declared");
    let scene = dir.join("scene.esn");
    let text = "use \"kernels/\";
param level: Float = 0.5;
node seed: image.gradient@1 { domain = vec2(8, 8); }
node gray: image.grayscale@1 { source <- seed.result; amount = level; }
output result <- gray.result;
";
    std::fs::write(&scene, text).unwrap();
    // On the command line, a node's own value is not a param.
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    let output = Command::new(entelechy)
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
    let output = Command::new(entelechy)
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
        .set_param("level", entelechy_kernel::Value::Float(1.0))
        .unwrap();
    assert_eq!(
        session.params().get("level"),
        Some(&entelechy_kernel::Value::Float(1.0))
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
        .set_param("level", entelechy_kernel::Value::Float(1.0))
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
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    session.sync();
    session.pin(entelechy_graph::NodeId("show".into()));
    let report = session.evaluate(&Context::default()).unwrap();
    let out = dir.join("out.ppm");
    let refused = session.export(report.output, &out).unwrap_err().to_string();
    assert!(
        refused.contains("writing files is not allowed"),
        "{refused}"
    );
    let field = dir.join("show.field");
    let refused = session
        .save(&entelechy_graph::NodeId("show".into()), "result", &field)
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
        .save(&entelechy_graph::NodeId("show".into()), "result", &field)
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
    let scene = dir.join("scene.esn");
    let original = std::fs::read_to_string(&scene).unwrap();
    let mut runtime = Session::runtime_for("cpu", entelechy_session::Limits::default()).unwrap();
    runtime.set_grants(
        [
            entelechy_kernel::Capability::FieldRead,
            entelechy_kernel::Capability::FieldWrite,
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

/// A fault in a function a library declares is one problem, in the
/// library, however many Kernels meet it: its own Kernels, and a scene's
/// Kernel written in place that inherits the function.
#[test]
fn a_fault_in_a_shared_function_is_reported_once_where_it_is_written() {
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    let dir = std::env::temp_dir().join(format!("entelechy-shared-fault-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("kernels")).unwrap();
    let kernel = |name: &str| {
        format!(
            "kernel d.{name}@1 {{ requires {{ field.write; }} output result: Rows<Float>; compute {{ return 1.0; }} }}\n"
        )
    };
    std::fs::write(
        dir.join("kernels/lib.ent"),
        format!(
            "fn broken(t: Float) -> Float {{ return t * nope; }}\n{}{}{}",
            kernel("a"),
            kernel("b"),
            kernel("c")
        ),
    )
    .unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "use \"kernels/lib.ent\";\nnode a: d.a@1 { domain = 2; }\nnode v: kernel {\n    requires { field.write; }\n    domain = 2;\n    output result: Rows<Float>;\n    compute { return 5.0; }\n}\noutput result <- v.result;\n",
    )
    .unwrap();
    let output = Command::new(entelechy)
        .arg("check")
        .arg(dir.join("scene.esn"))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let faults: Vec<&str> = stdout
        .lines()
        .filter(|line| line.contains("`nope` is not defined"))
        .collect();
    assert_eq!(faults.len(), 1, "{stdout}");
    assert!(faults[0].contains("lib.ent:1:"), "{stdout}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A value computed in two Kernels, the second holding it equal to the
/// first's, passes a replay and fails it rounded apart: the decision is
/// made twice.
#[test]
fn a_value_two_kernels_each_compute_fails_a_replay_rounded_apart() {
    let dir = std::env::temp_dir().join(format!("entelechy-apart-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("twice.esn"),
        "node first: kernel {
    requires { field.write; }
    output result: Rows<Float>;
    compute { let x = 0.1 + float(element()) * 0.013; return x * 0.7 + 0.3; }
    domain = 4096;
}
node second: kernel {
    requires { field.read; field.write; }
    input first: Rows<Float> <- first.result;
    output result: Rows<Int>;
    compute { let x = 0.1 + float(element()) * 0.013; return if x * 0.7 + 0.3 == sample(first) { 0 } else { 1 }; }
    merge(a, b) { return a + b; }
}
output differ <- second.result;
",
    )
    .unwrap();
    let replay = dir.join("twice.replay");
    std::fs::write(
        &replay,
        "scene twice.esn\nframe 0\nexpect second.result[0] = 0\n",
    )
    .unwrap();
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    let plain = Command::new(entelechy)
        .arg("replay")
        .arg(&replay)
        .output()
        .unwrap();
    assert!(
        plain.status.success(),
        "{}",
        String::from_utf8_lossy(&plain.stdout)
    );
    let apart = Command::new(entelechy)
        .args(["replay", "--round-apart"])
        .arg(&replay)
        .output()
        .unwrap();
    let said = String::from_utf8_lossy(&apart.stdout);
    assert!(!apart.status.success(), "{said}");
    assert!(said.contains("rounded apart, way"), "{said}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A chain of runs that together hold more than the Host has room for
/// runs to its end: the outputs the frame has done with are let go to make
/// room, and what an expectation reads is kept.
#[test]
fn a_chain_longer_than_the_host_holds_lets_go_of_what_it_is_done_with() {
    let dir = std::env::temp_dir().join(format!("entelechy-room-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("link.esn"),
        "input x: Rows<Float>;
node up: kernel {
    requires { field.read; field.write; }
    input x: Rows<Float> <- x;
    output result: Rows<Float>;
    compute { return at(x, element() + 1) * 0.0 + at(x, element()) + 1.0; }
}
node on: kernel {
    requires { field.read; field.write; }
    input x: Rows<Float> <- up.result;
    output result: Rows<Float>;
    compute { return at(x, element() - 1) * 0.0 + at(x, element()); }
}
output result <- on.result;
",
    )
    .unwrap();
    // Forty links of a hundred thousand floats, sixteen megabytes, in a
    // Host of four.
    std::fs::write(
        dir.join("chain.esn"),
        "node seed: kernel {
    requires { field.write; }
    output result: Rows<Float>;
    compute { return float(element()); }
    domain = 100000;
}
node chain: graph \"link.esn\" {
    times = 40;
    carry x <- result;
    x <- seed.result;
}
node last: kernel {
    requires { field.read; field.write; }
    input x: Rows<Float> <- chain.result;
    output result: Rows<Float>;
    compute { return sample(x); }
}
output out <- last.result;
",
    )
    .unwrap();
    let replay = dir.join("chain.replay");
    std::fs::write(
        &replay,
        "scene chain.esn\nframe 0\nexpect last.result[5] = 45.0\nexpect seed.result[5] = 5.0\nframe 1\nexpect last.result[5] = 45.0\n",
    )
    .unwrap();
    let ran = Command::new(env!("CARGO_BIN_EXE_entelechy"))
        .args(["replay", "--memory", "4"])
        .arg(&replay)
        .output()
        .unwrap();
    let said = String::from_utf8_lossy(&ran.stdout);
    assert!(
        ran.status.success(),
        "{said}{}",
        String::from_utf8_lossy(&ran.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Room made while an `async` node's last output is read on the frame
/// keeps that output: a node passed without running is read as it is.
#[test]
fn room_made_keeps_an_async_nodes_last_output_the_frame_reads() {
    let dir = std::env::temp_dir().join(format!("entelechy-room-async-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("plus.ent"),
        "kernel a.plus@1 {
    requires { field.read; field.write; }
    input a: Rows<Float>;
    output result: Rows<Float>;
    compute { return sample(a) + 1.0; }
}
",
    )
    .unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "use \"plus.ent\";
node x: kernel {
    requires { field.write; time.read; }
    output result: Rows<Float>;
    domain = 16384;
    compute { return time(); }
}
node p: a.plus@1 { a <- x.result; async = true; }
node r: kernel {
    requires { field.read; field.write; time.read; }
    input a: Rows<Float> <- p.result;
    output result: Rows<Float>;
    domain = 110000;
    compute { return at(a, 0) + time(); }
}
node o: kernel {
    requires { field.read; field.write; }
    input a: Rows<Float> <- r.result;
    input b: Rows<Float> <- p.result;
    output result: Rows<Float>;
    domain = 1;
    compute { return at(a, 0) + at(b, 0) * 0.0; }
}
output result <- o.result;
",
    )
    .unwrap();
    let replay = dir.join("scene.replay");
    std::fs::write(
        &replay,
        "scene scene.esn\nframe 0.0\nexpect o.result[0] = 1.0\nframe 1.0\nframe 2.0\n",
    )
    .unwrap();
    let ran = Command::new(env!("CARGO_BIN_EXE_entelechy"))
        .args(["replay", "--memory", "1"])
        .arg(&replay)
        .output()
        .unwrap();
    let said = String::from_utf8_lossy(&ran.stdout);
    assert!(
        ran.status.success(),
        "{said}{}",
        String::from_utf8_lossy(&ran.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A requirement the parameters break is refused as the scene loads, and
/// stops an evaluation before anything runs when a param set later breaks
/// it; each says what it came to. As given, the scene runs.
#[test]
fn a_requirement_the_parameters_break_stops_the_evaluation() {
    let dir = std::env::temp_dir().join(format!("entelechy-require-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let scene = dir.join("scene.esn");
    std::fs::write(
        &scene,
        "param cells: Int = 64;
param bits: Int = 6;
require keys: cells <= 1 << bits;
node n: kernel {
    requires { field.write; }
    output result: Rows<Int>;
    compute { return element(); }
    domain = 1;
}
output result <- n.result;
",
    )
    .unwrap();
    let run = |extra: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_entelechy"))
            .arg("run")
            .arg(&scene)
            .args(extra)
            .output()
            .unwrap()
    };
    let ran = run(&[]);
    assert!(
        ran.status.success(),
        "{}",
        String::from_utf8_lossy(&ran.stderr)
    );
    let broken = run(&["--param", "bits=5"]);
    let said = String::from_utf8_lossy(&broken.stderr);
    assert!(!broken.status.success());
    assert!(
        said.contains("requirement `keys`")
            && said.contains("here 64 <= 32, with cells = 64, bits = 5"),
        "{said}"
    );
    assert!(said.contains(":3:9)"), "{said}");
    // Set while the scene runs, the evaluation after refuses it.
    let replay = dir.join("scene.replay");
    std::fs::write(&replay, "scene scene.esn\nframe 0\nparam bits 5\nframe 1\n").unwrap();
    let replayed = Command::new(env!("CARGO_BIN_EXE_entelechy"))
        .arg("replay")
        .arg(&replay)
        .output()
        .unwrap();
    let said = String::from_utf8_lossy(&replayed.stdout);
    assert!(!replayed.status.success());
    assert!(
        said.contains("frame at 1: the graph's requirement `keys`"),
        "{said}{}",
        String::from_utf8_lossy(&replayed.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A recording goes where `--out` says: a `.wav` file takes the sound
/// alone, with no picture written; a replay records the frames its steps
/// evaluate, into a directory or a `.wav` file, one replay at a time.
#[test]
fn a_render_and_a_replay_record_where_out_says() {
    let entelechy = env!("CARGO_BIN_EXE_entelechy");
    let dir = std::env::temp_dir().join(format!("entelechy-record-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "node tone: kernel { requires { field.write; time.read; } output result: Rows<Vec2>; domain = 800;
    compute { let t = time() + float(element()) / 48000.0; let v = sin(t * 1382.3) * 0.4; return vec2(v, v * 0.5); } }
node scope: kernel { requires { field.read; field.write; time.read; } input block: Rows<Vec2> <- tone.result; output result: Image; domain = vec2(32, 24);
    compute { return vec4(fract(time()), 0.0, 0.0, 1.0); } }
output view <- scope.result;
output sound <- tone.result;
",
    )
    .unwrap();
    let blocks = |path: &std::path::Path| {
        entelechy_formats::wav::decode(&std::fs::read(path).unwrap())
            .unwrap()
            .field
            .domain()
            .count()
            / 800
    };
    let run = |args: &[&str]| {
        let output = Command::new(entelechy)
            .args(args)
            .current_dir(&dir)
            .output()
            .unwrap();
        (
            output.status.success(),
            String::from_utf8_lossy(&output.stderr).into_owned()
                + &String::from_utf8_lossy(&output.stdout),
        )
    };

    // The sound alone.
    let (ok, said) = run(&["render", "scene.esn", "--frames", "4", "--out", "tone.wav"]);
    assert!(ok, "{said}");
    assert_eq!(blocks(&dir.join("tone.wav")), 4);
    assert!(!dir.join("frames").exists(), "{said}");
    assert!(!said.contains("picture"), "{said}");

    // A replay's frames, into a directory: a picture and a block each.
    std::fs::write(
        dir.join("play.replay"),
        "scene scene.esn\nframe 0.0 to 0.2 every 0.1\nexpect count tone.result = 800\n",
    )
    .unwrap();
    let (ok, said) = run(&["replay", "play.replay", "--out", "played"]);
    assert!(ok, "{said}");
    assert!(said.contains("ok (3 frames, 1 checks)"), "{said}");
    for index in 0..3 {
        let bytes =
            std::fs::read(entelechy::render::picture_path(&dir.join("played"), index)).unwrap();
        assert_eq!(entelechy_formats::png::decode(&bytes).unwrap().width, 32);
    }
    assert!(!dir.join("played/0003.png").exists());
    assert_eq!(
        blocks(&dir.join("played").join(entelechy::render::SOUND_NAME)),
        3
    );

    // Or the sound alone.
    let (ok, said) = run(&["replay", "play.replay", "--out", "played.wav"]);
    assert!(ok, "{said}");
    assert_eq!(blocks(&dir.join("played.wav")), 3);

    // One replay at a time.
    let (ok, said) = run(&["replay", "play.replay", "play.replay", "--out", "both"]);
    assert!(!ok);
    assert!(
        said.contains("--out records one run of one replay"),
        "{said}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
