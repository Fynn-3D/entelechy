//! The `orla` command.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use orla_kernel::Context;
use orla_session::{Event, Session};

/// Runs `orla-<tool>` with `args`: the one beside this `orla`, else the
/// one on `PATH`. Its exit status is `orla`'s.
fn beside(tool: &str, args: &[String]) -> ExitCode {
    let name = format!("orla-{tool}{}", std::env::consts::EXE_SUFFIX);
    let path = std::env::current_exe()
        .ok()
        .and_then(|exe| Some(exe.parent()?.join(&name)))
        .filter(|path| path.is_file())
        .unwrap_or_else(|| PathBuf::from(&name));
    match std::process::Command::new(&path).args(args).status() {
        Ok(status) => ExitCode::from(status.code().map_or(1, |code| code as u8)),
        Err(error) => {
            eprintln!("error: `orla {tool}` runs `{name}`, beside `orla` or on PATH: {error}");
            ExitCode::FAILURE
        }
    }
}

fn usage_error(message: &str) -> Result<(), String> {
    Err(format!("{message}\n{}", orla::commands::usage()))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(command) = args.first().and_then(|name| orla::commands::command(name)) {
        if command.passes {
            return beside(command.name, &args[1..]);
        }
        if let Err(message) = command.check(&args[1..]) {
            eprintln!("error: {message}");
            return ExitCode::FAILURE;
        }
    }
    let result = match args.first().map(String::as_str) {
        Some("check") => check(&args[1..]),
        Some("run") => run(&args[1..]),
        Some("render") => render(&args[1..]),
        Some("replay") => replay(&args[1..]),
        Some("serve") => serve(&args[1..]),
        Some("new") => new_scene(&args[1..]),
        Some("bench") => measure(&args[1..]),
        Some("graph") => graph(&args[1..]),
        Some("trace") => trace(&args[1..]),
        Some("help") => help(&args[1..]),
        Some("version") => {
            println!(
                "orla = {}\norla.api = {}\norla.language = {}\norla.ir = {}\norla.package = {}",
                env!("CARGO_PKG_VERSION"),
                orla_kernel::ABI_VERSION,
                orla_language::LANGUAGE_VERSION,
                orla_ir::IR_VERSION,
                orla_session::PACKAGE_VERSION
            );
            Ok(())
        }
        Some("pack") => match &args[1..] {
            [dir, rest @ ..] => {
                let dir = Path::new(dir);
                let out = rest.first().map(PathBuf::from).unwrap_or_else(|| {
                    let name = dir
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    PathBuf::from(if name.ends_with(".orlapkg") {
                        name
                    } else {
                        format!("{name}.orlapkg")
                    })
                });
                orla::package::pack(dir, &out).map(|count| {
                    eprintln!("packed {count} file(s) into {}", out.display());
                })
            }
            _ => usage_error("pack needs a package directory"),
        },
        Some("unpack") => match &args[1..] {
            [file, dir, ..] => orla::package::unpack(Path::new(file), Path::new(dir))
                .map(|count| eprintln!("unpacked {count} file(s) into {dir}")),
            _ => usage_error("unpack needs an .orlapkg file and a new directory"),
        },
        Some("package") => match &args[1..] {
            [dir, ..] => orla::package::describe(Path::new(dir)).map(|text| println!("{text}")),
            _ => usage_error("package needs a package directory"),
        },
        _ => {
            eprintln!("{}", orla::commands::usage());
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn check(args: &[String]) -> Result<(), String> {
    let show_ir = args.iter().any(|arg| arg == "--ir");
    let files: Vec<&String> = args.iter().filter(|arg| *arg != "--ir").collect();
    if files.is_empty() {
        return Err("check needs at least one document".into());
    }
    let packages = orla_session::Disk::with_packages(orla_session::environment_packages());
    let mut failures = 0;
    for file in files {
        let checked = orla_session::check_document(&packages, Path::new(file), show_ir);
        for (_, line) in &checked.lines {
            println!("{line}");
        }
        failures += checked.problems();
    }
    if failures > 0 {
        Err(format!("{failures} problem(s) found"))
    } else {
        Ok(())
    }
}

struct RunOptions {
    /// The session: scene, library, inputs, params, backend, fusion,
    /// limits and allowances.
    session: orla_session::SessionOptions,
    saves: Vec<(String, String, PathBuf)>,
    /// The file the scene's picture goes to.
    out: Option<PathBuf>,
    watch: bool,
    /// The window's size in pixels and scale factor, for `window()`.
    window: Option<([f32; 2], f32)>,
    /// Print each executed node's time.
    profile: bool,
}

/// `render`: the scene and its session flags, and how to render.
struct RenderOptions {
    session: orla_session::SessionOptions,
    render: orla::render::Options,
    /// Node outputs written after the last frame, as `run --save`.
    saves: Vec<(String, String, PathBuf)>,
}

/// `--save <node.port>=<file>`, or `<node>=<file>` for its `result`.
fn parse_save(binding: &str) -> Result<(String, String, PathBuf), String> {
    let (target, path) = binding
        .split_once('=')
        .ok_or_else(|| "--save needs <node.port>=<file>".to_string())?;
    let (node, port) = orla_graph::split_port(target)
        .map(|(node, port)| (node.0, port))
        .unwrap_or_else(|| (target.to_owned(), "result".to_owned()));
    let path = PathBuf::from(path);
    orla_formats::writable(&path).map_err(|why| format!("--save: {why}"))?;
    Ok((node, port, path))
}

fn parse_render(args: &[String]) -> Result<RenderOptions, String> {
    let mut options = RenderOptions {
        session: orla_session::SessionOptions::from_environment(),
        saves: Vec::new(),
        render: orla::render::Options {
            frames: 0,
            rate: 60.0,
            sound_rate: 48_000,
            out: PathBuf::from("frames"),
            window: None,
        },
    };
    let mut iter = args.iter().cloned();
    while let Some(arg) = iter.next() {
        if options.session.read(&arg, &mut || iter.next())? {
            continue;
        }
        let mut value = |flag: &str| iter.next().ok_or_else(|| format!("{flag} needs a value"));
        let number = |flag: &str, text: String| -> Result<f32, String> {
            text.parse::<f32>()
                .ok()
                .filter(|n| *n > 0.0)
                .ok_or_else(|| format!("{flag} needs a positive number"))
        };
        match arg.as_str() {
            "--frames" => {
                options.render.frames =
                    value("--frames")?
                        .parse::<u32>()
                        .ok()
                        .filter(|n| *n > 0)
                        .ok_or_else(|| "--frames needs a whole number of at least 1".to_string())?;
            }
            "--rate" => options.render.rate = number("--rate", value("--rate")?)?,
            "--sound-rate" => {
                options.render.sound_rate = number("--sound-rate", value("--sound-rate")?)? as u32;
            }
            "--out" => options.render.out = PathBuf::from(value("--out")?),
            "--window" => {
                options.render.window = Some(orla_session::parse_window(&value("--window")?)?);
            }
            "--save" => options.saves.push(parse_save(&value("--save")?)?),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    options.session.scene()?;
    if options.render.frames == 0 {
        return Err("render needs --frames <n>".into());
    }
    Ok(options)
}

fn render(args: &[String]) -> Result<(), String> {
    let options = parse_render(args)?;
    let mut session = options.session.open("jit")?;
    // A file named on the command line is the person's own allowance.
    for (node, _, path) in &options.saves {
        session.pin(orla_graph::NodeId(node.clone()));
        session.allow_write_to(path)?;
    }
    let events = session.sync();
    report_events(&events);
    loaded(&session, &events)?;
    let started = Instant::now();
    let total = options.render.frames;
    let summary = orla::render::render(&mut session, &options.render, |index| {
        if (index + 1) % 60 == 0 || index + 1 == total {
            eprintln!("frame {}/{total}", index + 1);
        }
    })?;
    recorded(
        "rendered",
        &orla::render::Target::of(&options.render.out),
        &summary,
        options.render.sound_rate,
        started,
    );
    report_events(&session.sync());
    for (node, port, path) in &options.saves {
        session
            .save(&orla_graph::NodeId(node.clone()), port, path)
            .map_err(|error| format!("cannot save {node}.{port}: {error}"))?;
        eprintln!(
            "saved {node}.{port} after the last frame -> {}",
            path.display()
        );
    }
    Ok(())
}

fn parse_run(args: &[String]) -> Result<RunOptions, String> {
    let mut options = RunOptions {
        session: orla_session::SessionOptions::from_environment(),
        saves: Vec::new(),
        out: None,
        watch: false,
        window: None,
        profile: false,
    };
    let mut iter = args.iter().cloned();
    while let Some(arg) = iter.next() {
        if options.session.read(&arg, &mut || iter.next())? {
            continue;
        }
        let mut value = |flag: &str| iter.next().ok_or_else(|| format!("{flag} needs a value"));
        match arg.as_str() {
            "--window" => options.window = Some(orla_session::parse_window(&value("--window")?)?),
            "--out" => {
                let out = PathBuf::from(value("--out")?);
                orla_formats::writable(&out).map_err(|why| format!("--out: {why}"))?;
                options.out = Some(out);
            }
            "--watch" => options.watch = true,
            "--profile" => options.profile = true,
            "--save" => options.saves.push(parse_save(&value("--save")?)?),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    options.session.scene()?;
    Ok(options)
}

fn run(args: &[String]) -> Result<(), String> {
    let options = parse_run(args)?;
    let mut session = options.session.open("jit")?;
    // A profile waits for the device after each node.
    session.set_measured(options.profile);
    // A file named on the command line is the person's own allowance.
    for (node, _, path) in &options.saves {
        session.pin(orla_graph::NodeId(node.clone()));
        session.allow_write_to(path)?;
    }
    if let Some(out) = &options.out {
        session.allow_write_to(out)?;
    }
    let events = session.sync();
    report_events(&events);
    // Under `--watch`, a scene that fails to load may be fixed later.
    if !options.watch {
        loaded(&session, &events)?;
    }
    let clock = Instant::now();
    let window = options.window;
    let mut healthy = evaluate_and_export(
        &mut session,
        options.out.as_deref(),
        &options.saves,
        0.0,
        window,
        options.profile,
    );
    let _ = &clock;
    if !options.watch {
        report_events(&session.sync());
        return if healthy {
            Ok(())
        } else {
            Err("the scene did not evaluate".into())
        };
    }
    let watched = session
        .watched()
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    eprintln!("watching {watched}; edit a document and save");
    loop {
        std::thread::sleep(Duration::from_millis(200));
        let events = session.sync();
        if events.is_empty() {
            continue;
        }
        report_events(&events);
        let changed = events.iter().any(|event| {
            matches!(
                event,
                Event::Published { .. } | Event::Retired { .. } | Event::SceneLoaded
            )
        });
        if changed || !healthy {
            healthy = evaluate_and_export(
                &mut session,
                options.out.as_deref(),
                &options.saves,
                clock.elapsed().as_secs_f32(),
                window,
                options.profile,
            );
        }
    }
}

/// `orla new`: a directory that runs.
fn new_scene(args: &[String]) -> Result<(), String> {
    let mut kind = orla::scaffold::Kind::Picture;
    let mut dir: Option<PathBuf> = None;
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        index += 1;
        match arg {
            "--kind" => {
                let name = args
                    .get(index)
                    .cloned()
                    .ok_or_else(|| "--kind needs a value".to_string())?;
                index += 1;
                kind = orla::scaffold::Kind::from_name(&name).ok_or_else(|| {
                    format!(
                        "`{name}` is not a kind; they are {}",
                        orla::scaffold::Kind::NAMES.join(", ")
                    )
                })?;
            }
            other if other.starts_with("--") => return Err(format!("unknown flag {other}")),
            other => {
                if dir.replace(PathBuf::from(other)).is_some() {
                    return Err("new takes exactly one directory".into());
                }
            }
        }
    }
    let dir = dir.ok_or_else(|| "new needs a directory".to_string())?;
    let name = dir
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "scene".to_owned());
    let written = orla::scaffold::new_scene(&dir, kind, &name)?;
    for path in &written {
        println!("{}", dir.join(path).display());
    }
    let scene = dir.join("scene.osn");
    let replay = written
        .iter()
        .find(|path| path.ends_with(".replay"))
        .map(|path| dir.join(path))
        .unwrap_or_else(|| dir.join("tests"));
    // Written as `orla` was run, so each line runs as it is.
    let orla = std::env::args()
        .next()
        .filter(|me| me.contains('/'))
        .unwrap_or_else(|| "orla".to_owned());
    eprintln!(
        "play it:  {orla} player {}\nrun it:   {orla} run {} --window 640x400 --out frame.png\ntest it:  {orla} replay {}",
        scene.display(),
        scene.display(),
        replay.display()
    );
    Ok(())
}

/// `orla bench`: the same frames, repeated, on one backend or several.
fn measure(args: &[String]) -> Result<(), String> {
    let mut options = orla_session::bench::Options {
        session: orla_session::SessionOptions::from_environment(),
        ..orla_session::bench::Options::default()
    };
    let mut backends: Vec<String> = Vec::new();
    let mut iter = args.iter().cloned();
    while let Some(arg) = iter.next() {
        let mut value = |flag: &str| iter.next().ok_or_else(|| format!("{flag} needs a value"));
        match arg.as_str() {
            // Repeatable: backends run one after another.
            "--backend" => {
                backends.push(value("--backend")?);
                continue;
            }
            "--frames" => {
                options.frames = value("--frames")?
                    .parse()
                    .map_err(|_| "--frames needs a count".to_string())?;
                continue;
            }
            "--repeat" => {
                options.repeat = value("--repeat")?
                    .parse()
                    .map_err(|_| "--repeat needs a count".to_string())?;
                continue;
            }
            "--rate" => {
                options.rate = value("--rate")?
                    .parse()
                    .map_err(|_| "--rate needs frames a second".to_string())?;
                continue;
            }
            "--nodes" => {
                options.nodes = true;
                continue;
            }
            "--json" => {
                options.json = Some(PathBuf::from(value("--json")?));
                continue;
            }
            "--compare" => {
                options.compare = Some(PathBuf::from(value("--compare")?));
                continue;
            }
            "--window" => {
                options.window = Some(orla_session::parse_window(&value("--window")?)?);
                continue;
            }
            _ => {}
        }
        if !options.session.read(&arg, &mut || iter.next())? {
            return Err(format!("unknown flag {arg}"));
        }
    }
    options.session.scene()?;
    // `--backend all`: every backend this build has, reference first.
    if backends.iter().any(|name| name == "all") {
        backends = ["cpu", "jit", "gpu"]
            .iter()
            .filter(|name| Session::runtime_for(name, orla_session::Limits::default()).is_ok())
            .map(|name| (*name).to_owned())
            .collect();
    }
    if !backends.is_empty() {
        options.backends = backends;
    }
    let before = match &options.compare {
        Some(path) => Some(orla_session::bench::read(path)?),
        None => None,
    };
    let measured = orla_session::bench::bench(&options)?;
    print!(
        "{}",
        orla_session::bench::table(&options, &measured, before.as_deref())
    );
    if let Some(path) = &options.json {
        std::fs::write(path, orla_session::bench::json(&options, &measured))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

fn graph(args: &[String]) -> Result<(), String> {
    // `run`'s session flags: the plan printed is the one `run` executes
    // with the same params and inputs.
    let mut options = orla_session::SessionOptions::from_environment();
    let mut iter = args.iter().cloned();
    while let Some(arg) = iter.next() {
        if !options.read(&arg, &mut || iter.next())? {
            return Err(format!("unknown flag {arg}"));
        }
    }
    options
        .scene()
        .map_err(|_| "graph needs a scene file".to_string())?;
    let mut session = options.open("cpu")?;
    let events = session.sync();
    for event in &events {
        if matches!(event, Event::Rejected { .. } | Event::SceneRejected(_)) {
            eprintln!("{event}");
        }
    }
    let graph = session
        .graph()
        .ok_or_else(|| "the scene did not load".to_string())?
        .clone();
    print!("{}", orla_scene::print(&graph));
    let kernels: Vec<String> = session
        .runtime()
        .kernels()
        .map(|version| format!("{} {}", version.id(), version.generation()))
        .collect();
    println!("\n// published: {}", kernels.join(", "));
    match session.plan() {
        Ok(plan) => {
            println!("// plan: {} group(s)", plan.len());
            // The nodes holding state, in the order they advance.
            let stateful: Vec<String> = plan
                .iter()
                .flat_map(|entry| std::iter::once(&entry.node).chain(&entry.absorbed))
                .filter(|node| graph.reads(node).is_some_and(|reads| reads.holds_state))
                .map(ToString::to_string)
                .collect();
            if !stateful.is_empty() {
                println!(
                    "// state, in the order it advances: {}",
                    stateful.join(" -> ")
                );
            }
            for entry in &plan {
                let absorbed = if entry.absorbed.is_empty() {
                    String::new()
                } else {
                    format!(
                        " + {}",
                        entry
                            .absorbed
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(" + ")
                    )
                };
                let domain = entry.domain.map(|d| format!(" {d}")).unwrap_or_default();
                let counted = if entry.counted.is_empty() {
                    String::new()
                } else {
                    format!(
                        "; counts {}",
                        entry
                            .counted
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                };
                println!(
                    "//   {}{absorbed}: {}{domain}{counted}",
                    entry.node, entry.kernel
                );
                // What each node of the group holding state reads of
                // state: what this evaluation made, or what the last
                // committed. A node holding none reads what is newest, as
                // a picture of it does.
                for node in std::iter::once(&entry.node).chain(&entry.absorbed) {
                    let Some(reads) = graph.reads(node).filter(|reads| reads.holds_state) else {
                        continue;
                    };
                    let names = |nodes: &[orla_graph::NodeId]| {
                        nodes
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(", ")
                    };
                    let mut said = vec!["holds state".to_owned()];
                    if !reads.new_state.is_empty() {
                        said.push(format!(
                            "reads the new state of {}",
                            names(&reads.new_state)
                        ));
                    }
                    if !reads.committed.is_empty() {
                        said.push(format!(
                            "reads the last committed {}",
                            names(&reads.committed)
                        ));
                    }
                    println!("//     {node}: {}", said.join("; "));
                }
            }
            Ok(())
        }
        Err(error) => Err(format!("cannot plan: {error}")),
    }
}

fn serve(args: &[String]) -> Result<(), String> {
    let mut options = orla_session::SessionOptions::from_environment();
    let mut listen = None;
    let mut window = None;
    let mut origins = Vec::new();
    let mut iter = args.iter().cloned();
    while let Some(arg) = iter.next() {
        if options.read(&arg, &mut || iter.next())? {
            continue;
        }
        let mut value = |flag: &str| iter.next().ok_or_else(|| format!("{flag} needs a value"));
        match arg.as_str() {
            "--window" => window = Some(orla_session::parse_window(&value("--window")?)?),
            "--listen" => listen = Some(value("--listen")?),
            "--allow-origin" => origins.push(value("--allow-origin")?),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    // Allowances come from whoever started the server, never from a
    // client.
    let mut session = options.open("jit")?;
    session.set_window(window);
    // Reads are held to the scene's documents' directories and the
    // allowances.
    session.restrict_reads()?;
    report_events(&session.sync());
    session.set_background(true);
    let (sink, requests) = std::sync::mpsc::channel();
    if let Some(address) = &listen {
        let origins = orla_protocol::websocket::Origins::with(origins);
        let bound =
            orla_protocol::listen(address, sink.clone(), origins).map_err(|e| e.to_string())?;
        eprintln!("listening on {bound}");
    }
    if orla_protocol::stdin_into(sink).is_err() {
        // Without a reader thread, requests are answered in turn here.
        for incoming in orla_protocol::stdin_lines() {
            if orla_protocol::answer(&mut session, incoming) {
                break;
            }
        }
        return Ok(());
    }
    // A `compose` or `publish` is answered once its change stands; other
    // requests are answered meanwhile. With every input closed, what is
    // owed is still answered before the process ends.
    let mut answers = orla_protocol::Answers::new();
    let mut open = true;
    loop {
        let incoming = if !open {
            if !answers.waiting() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
            None
        } else if answers.waiting() {
            match requests.recv_timeout(std::time::Duration::from_millis(1)) {
                Ok(incoming) => Some(incoming),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => None,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    open = false;
                    None
                }
            }
        } else {
            match requests.recv() {
                Ok(incoming) => Some(incoming),
                Err(_) => break,
            }
        };
        if let Some(incoming) = incoming
            && answers.answer(&mut session, incoming)
        {
            break;
        }
        if answers.poll(&mut session) {
            break;
        }
    }
    Ok(())
}

/// Says what a recording wrote, and where.
fn recorded(
    what: &str,
    target: &orla::render::Target,
    summary: &orla::render::Summary,
    sound_rate: u32,
    started: Instant,
) {
    if let (orla::render::Target::Directory(dir), Some((width, height))) = (target, summary.size) {
        eprintln!(
            "{what} {} picture(s) of {width}x{height} to {}/0000.png ... in {:?}",
            summary.pictures,
            dir.display(),
            started.elapsed()
        );
    }
    if summary.channels > 0 {
        eprintln!(
            "{what} {:.2} s of sound ({} channel(s) at {sound_rate} Hz) to {}",
            summary.sound_frames as f64 / f64::from(sound_rate),
            summary.channels,
            target.sound_path().display()
        );
    }
}

fn replay(args: &[String]) -> Result<(), String> {
    let mut backend = "jit".to_string();
    let mut limits = orla_session::Limits::default();
    let mut apart = false;
    let mut watch = false;
    let mut fusion = true;
    let mut files = Vec::new();
    let mut out: Option<PathBuf> = None;
    let mut sound_rate = 48_000u32;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--out" => {
                out = Some(PathBuf::from(
                    iter.next()
                        .ok_or_else(|| "--out needs a value".to_string())?,
                ));
            }
            "--sound-rate" => {
                sound_rate = iter
                    .next()
                    .and_then(|text| text.parse::<u32>().ok())
                    .filter(|n| *n > 0)
                    .ok_or_else(|| "--sound-rate needs a positive whole number".to_string())?;
            }
            "--backend" => {
                backend = iter
                    .next()
                    .cloned()
                    .ok_or_else(|| "--backend needs a value".to_string())?;
            }
            "--round-apart" => apart = true,
            "--no-fusion" => fusion = false,
            "--watch" => watch = true,
            flag if orla_session::Limits::takes(flag) => {
                let text = iter.next().ok_or_else(|| format!("{flag} needs a value"))?;
                limits.read(flag, text)?;
            }
            other if other.starts_with("--") => return Err(format!("unknown flag {other}")),
            other => files.push(PathBuf::from(other)),
        }
    }
    if files.is_empty() {
        return Err("replay needs at least one .replay file".into());
    }
    if out.is_some() && (files.len() > 1 || apart) {
        return Err("--out records one run of one replay".into());
    }
    if watch && (out.is_some() || apart) {
        return Err("--watch plays the replays plainly, each as its documents change".into());
    }
    // Their documents, each parsed once while its file is unchanged.
    let store = std::sync::Arc::new(orla_session::Disk::with_packages(
        orla_session::environment_packages(),
    ));
    // Many replays in one run share their backends, each Kernel prepared
    // once.
    // And each scene's replays share one session, started over for each.
    let mut resident = if out.is_some() {
        None
    } else {
        Some(orla::replay::Resident::new(
            orla_session::Warm::open(&backend, limits)?,
            std::sync::Arc::clone(&store) as std::sync::Arc<dyn orla_session::Store>,
            fusion,
        ))
    };
    let mut failed = 0;
    // Each replay played: its file, itself, the nodes it touched, and
    // whether it failed.
    let mut played: Vec<(
        PathBuf,
        orla::replay::Replay,
        std::collections::BTreeSet<String>,
        bool,
    )> = Vec::new();
    for file in &files {
        let text = std::fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
        let base = file.parent().unwrap_or_else(|| Path::new("."));
        let replay = orla::replay::parse(&text, base)
            .map_err(|error| format!("{}: {error}", file.display()))?;
        let store = std::sync::Arc::clone(&store);
        if apart && replay.alike {
            println!(
                "{}: left out rounded apart: it holds two forms alike",
                file.display()
            );
            continue;
        }
        let ran = if let (true, Some(resident)) = (apart, &mut resident) {
            rounded_apart(&replay, resident)
        } else if let Some(out) = &out {
            let runtime = Session::runtime_for(&backend, limits)?;
            let target = orla::render::Target::of(out);
            let mut recorder = orla::render::Recorder::new(target.clone(), sound_rate)?;
            let started = Instant::now();
            let mut each = |session: &mut Session, report: &orla_graph::Report| {
                recorder.frame(session, report)
            };
            let ran = orla::replay::run_each(&replay, store, runtime, fusion, Some(&mut each));
            let summary = recorder.finish()?;
            recorded("recorded", &target, &summary, sound_rate, started);
            ran
        } else if let Some(resident) = &mut resident {
            resident.play_as(Some(&file.display().to_string()), &replay, None)
        } else {
            let runtime = Session::runtime_for(&backend, limits)?;
            orla::replay::run_fused(&replay, store, runtime, fusion)
        };
        let touched = ran
            .as_ref()
            .map(|outcome| outcome.touched.clone())
            .unwrap_or_default();
        let passed = said_outcome(file, ran);
        if !passed {
            failed += 1;
        }
        played.push((file.clone(), replay, touched, !passed));
    }
    if watch && let Some(resident) = resident {
        return watch_replays(resident, played);
    }
    if failed > 0 {
        Err(format!("{failed} replay(s) failed"))
    } else {
        Ok(())
    }
}

/// Says how a replay went; whether it passed.
fn said_outcome(file: &Path, ran: Result<orla::replay::Outcome, String>) -> bool {
    match ran {
        Ok(outcome) if outcome.failures.is_empty() => {
            println!(
                "{}: ok ({} frames, {} checks)",
                file.display(),
                outcome.frames,
                outcome.checks
            );
            for line in outcome.lines() {
                println!("  {line}");
            }
            true
        }
        Ok(outcome) => {
            println!(
                "{}: {} of {} checks failed",
                file.display(),
                outcome.failures.len(),
                outcome.checks
            );
            for line in outcome.lines() {
                println!("  {line}");
            }
            false
        }
        Err(error) => {
            println!("{}: {error}", file.display());
            false
        }
    }
}

/// `orla replay --watch`: the replays played again as their documents,
/// inputs and files change, each scene's on its kept session. A change
/// plays only the replays it reaches (a node they touched runs a Kernel
/// published again, or reads an input read again; the graph rebuilt
/// otherwise, or the replay's own file changed: all of the scene's), the
/// ones that failed last first, each said as it ends.
fn watch_replays(
    mut resident: orla::replay::Resident,
    mut played: Vec<(
        PathBuf,
        orla::replay::Replay,
        std::collections::BTreeSet<String>,
        bool,
    )>,
) -> Result<(), String> {
    let stamp = |path: &Path| std::fs::metadata(path).and_then(|m| m.modified()).ok();
    let watched = |resident: &orla::replay::Resident,
                   played: &[(
        PathBuf,
        orla::replay::Replay,
        std::collections::BTreeSet<String>,
        bool,
    )]| {
        let mut all: Vec<PathBuf> = resident.watched();
        all.extend(played.iter().map(|(file, ..)| file.clone()));
        all.into_iter()
            .map(|path| {
                let at = stamp(&path);
                (path, at)
            })
            .collect::<Vec<_>>()
    };
    let mut seen = watched(&resident, &played);
    eprintln!(
        "watching {} file(s) of {} replay(s); edit and save",
        seen.len(),
        played.len()
    );
    loop {
        std::thread::sleep(Duration::from_millis(200));
        let now = watched(&resident, &played);
        if now == seen {
            continue;
        }
        let started = Instant::now();
        let changed_files: Vec<PathBuf> = now
            .iter()
            .filter(|entry| !seen.contains(entry))
            .map(|(path, _)| path.clone())
            .collect();
        let changes = resident.changed();
        for change in changes.values() {
            for said in &change.said {
                println!("{said}");
            }
        }
        // A replay's own file, changed: read again, and played.
        let mut again: Vec<usize> = Vec::new();
        for (index, (file, replay, touched, _)) in played.iter_mut().enumerate() {
            if changed_files.contains(file) {
                if let Ok(text) = std::fs::read_to_string(&*file)
                    && let Ok(parsed) =
                        orla::replay::parse(&text, file.parent().unwrap_or_else(|| Path::new(".")))
                {
                    *replay = parsed;
                }
                again.push(index);
            } else if resident.affects(&changes, &replay.scene, touched) {
                again.push(index);
            }
        }
        // The ones that failed last, first.
        again.sort_by_key(|&index| !played[index].3);
        let mut failed = 0;
        for &index in &again {
            let (file, replay, ..) = &played[index];
            let ran = resident.play_as(Some(&file.display().to_string()), replay, None);
            let touched = ran
                .as_ref()
                .map(|outcome| outcome.touched.clone())
                .unwrap_or_default();
            let passed = said_outcome(file, ran);
            if !passed {
                failed += 1;
            }
            played[index].2 = touched;
            played[index].3 = !passed;
        }
        eprintln!(
            "{} of {} replay(s) reached by the change, {failed} failed, in {:.1} s",
            again.len(),
            played.len(),
            started.elapsed().as_secs_f64()
        );
        seen = watched(&resident, &played);
    }
}

fn help(args: &[String]) -> Result<(), String> {
    let json = args.iter().any(|arg| arg == "--json");
    let Some(name) = args.iter().find(|arg| *arg != "--json") else {
        print!("{}", orla::help::full());
        return Ok(());
    };
    match orla::help::answer(name) {
        Ok(answer) if json => {
            println!("{answer}");
            Ok(())
        }
        Ok(answer) => {
            print!("{}", orla::help::text(&answer));
            Ok(())
        }
        Err(refusal) => {
            if json {
                println!("{refusal}");
            }
            Err(refusal["message"]
                .as_str()
                .unwrap_or("nothing named so")
                .to_owned())
        }
    }
}

/// `orla trace`: one element of a node, every binding printed.
fn trace(args: &[String]) -> Result<(), String> {
    let mut rest = Vec::new();
    let mut node = None;
    let mut at: Option<[u32; 3]> = None;
    let mut time = 0.0f32;
    let mut iter = args.iter().peekable();
    let number = |text: &str, flag: &str| -> Result<u32, String> {
        text.parse::<u32>()
            .map_err(|_| format!("{flag} takes whole numbers, not `{text}`"))
    };
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--node" => node = iter.next().cloned(),
            "--element" => {
                let text = iter.next().ok_or("--element needs a value")?;
                at = Some([number(text, "--element")?, 0, 0]);
            }
            "--at" => {
                let x = iter.next().ok_or("--at needs <x> <y> [<z>]")?;
                let y = iter.next().ok_or("--at needs <x> <y> [<z>]")?;
                let z = match iter.peek() {
                    Some(next) if !next.starts_with("--") && next.parse::<u32>().is_ok() => {
                        number(iter.next().expect("peeked"), "--at")?
                    }
                    _ => 0,
                };
                at = Some([number(x, "--at")?, number(y, "--at")?, z]);
            }
            "--time" => {
                let text = iter.next().ok_or("--time needs a value")?;
                time = text
                    .parse()
                    .map_err(|_| format!("--time takes seconds, not `{text}`"))?;
            }
            _ => rest.push(arg.clone()),
        }
    }
    let node = node.ok_or("trace needs --node <name>")?;
    let at = at.ok_or("trace needs --element <i> or --at <x> <y> [<z>]")?;
    let options = parse_run(&rest)?;
    let mut session = options.session.open("jit")?;
    let events = session.sync();
    report_events(&events);
    loaded(&session, &events)?;
    let (size, scale) = options.window.unwrap_or(([0.0, 0.0], 1.0));
    let context = Context {
        time,
        window: size,
        scale,
        ..Context::default()
    };
    let id = orla_graph::NodeId(node.clone());
    let orla_session::Traced {
        trace: traced,
        kernel,
        place: written_at,
    } = session.trace(&id, at, &context)?;
    let version = session
        .runtime()
        .kernel(&kernel)
        .ok_or_else(|| format!("`{kernel}` is not published"))?;
    let module = version.module().ok_or("a native operation has no trace")?;
    let scan = module.manifest.mode == orla_ir::Mode::Scan;
    let place = if scan {
        format!("step {}", traced.at[0])
    } else if traced.at[2] > 0 {
        format!(
            "element ({}, {}, {})",
            traced.at[0], traced.at[1], traced.at[2]
        )
    } else {
        format!("element ({}, {})", traced.at[0], traced.at[1])
    };
    println!("{node} ({kernel}), {place}, as the evaluation at time {time} runs it");
    for value in &traced.bindings {
        let shown = if value.ran {
            orla_protocol::words_text(module, value.ty, &value.words)
        } else {
            "- (not run)".to_owned()
        };
        println!(
            "  {}: {} = {shown}",
            value.name,
            orla_protocol::type_name(module, value.ty)
        );
    }
    let show = |value: &orla_session::TraceValue, indent: &str, arrow: &str| {
        let shown = if value.ran {
            orla_protocol::words_text(module, value.ty, &value.words)
        } else {
            "- (not run)".to_owned()
        };
        println!(
            "{indent}{arrow}{}: {} = {shown}",
            value.name,
            orla_protocol::type_name(module, value.ty)
        );
    };
    for (number, call) in traced.calls.iter().enumerate() {
        println!("  call {} #{}", call.function, number + 1);
        for value in &call.bindings {
            show(value, "    ", "");
        }
        for value in &call.results {
            show(value, "    ", "-> ");
        }
    }
    if traced.calls_truncated {
        println!(
            "  ... more calls than the {} recorded",
            orla_session::CALL_RECORDS
        );
    }
    for value in &traced.results {
        show(value, "  ", "-> ");
    }
    if let Some(fault) = &traced.fault {
        println!("  failed: {}", fault.message);
        if let Some(place) = &written_at {
            println!("  at {place}");
        }
        if let Some(function) = &fault.function {
            println!("  in `{function}`, as it failed");
            for value in &fault.bindings {
                show(value, "    ", "");
            }
        }
    }
    Ok(())
}

fn report_events(events: &[Event]) {
    for event in events {
        eprintln!("{event}");
    }
}

/// Whether the scene loaded; the error names why it did not.
fn loaded(session: &Session, events: &[Event]) -> Result<(), String> {
    if session.graph().is_some() {
        return Ok(());
    }
    let why: Vec<String> = events
        .iter()
        .filter_map(|event| match event {
            Event::SceneRejected(error) => Some(error.to_string()),
            _ => None,
        })
        .collect();
    Err(if why.is_empty() {
        "the scene did not load".into()
    } else {
        format!("the scene did not load: {}", why.join("; "))
    })
}

fn evaluate_and_export(
    session: &mut Session,
    out: Option<&Path>,
    saves: &[(String, String, PathBuf)],
    time: f32,
    window: Option<([f32; 2], f32)>,
    profile: bool,
) -> bool {
    let started = Instant::now();
    let (size, scale) = window.unwrap_or(([0.0, 0.0], 1.0));
    let context = Context {
        time,
        window: size,
        scale,
        ..Context::default()
    };
    match session.evaluate(&context) {
        Ok(report) => {
            // Wait for the device so the time below is the work.
            session.runtime().settle_device();
            let executed = report
                .executed
                .iter()
                .map(
                    |node| match report.fused.iter().find(|(consumer, _)| consumer == node) {
                        Some((_, absorbed)) => format!(
                            "{node}+{}",
                            absorbed
                                .iter()
                                .map(ToString::to_string)
                                .collect::<Vec<_>>()
                                .join("+")
                        ),
                        None => node.to_string(),
                    },
                )
                .collect::<Vec<_>>()
                .join(", ");
            if profile {
                // A profile of the first frame includes compilation and
                // allocation; `bench --nodes` measures a running scene.
                // The measurement floor
                // (`orla_kernel::Backend::measurement_floor`) comes off
                // every line.
                let floor = session.measurement_floor().as_secs_f64() * 1000.0;
                // Each node's placement, the reason when off the main
                // backend, and its Host/device traffic.
                for ((node, took), placed) in report.timings.iter().zip(&report.placements) {
                    let name = node.to_string();
                    let ms = (took.as_secs_f64() * 1000.0 - floor).max(0.0);
                    let time = orla_session::bench::NodeTime {
                        node: name.clone(),
                        ms,
                        backend: placed.backend.to_owned(),
                        routed: placed.routed.clone(),
                        uploaded: placed.uploaded,
                        downloaded: placed.downloaded,
                    };
                    eprintln!("  {name:<28} {ms:>9.3} ms  {}", time.placement_text());
                }
                let sum: f64 = report
                    .timings
                    .iter()
                    .map(|(_, took)| (took.as_secs_f64() * 1000.0 - floor).max(0.0))
                    .sum();
                let kind = if session.measures_the_device() {
                    "what the device's own clock said"
                } else {
                    "the call and the wait after it"
                };
                eprintln!(
                    "  {:<28} {sum:>9.3} ms  ({kind}; the first frame, which compiles and allocates, and a node runs alone under a profile — `bench --nodes` measures a scene already running)",
                    "(sum)"
                );
            }
            for (node, port, path) in saves {
                match session.save(&orla_graph::NodeId(node.clone()), port, path) {
                    Ok(()) => eprintln!("saved {node}.{port} -> {}", path.display()),
                    Err(error) => {
                        eprintln!("error: cannot save {node}.{port}: {error}");
                        return false;
                    }
                }
            }
            for (node, port, kernel) in &report.migrated {
                eprintln!("state of `{node}.{port}` carried into its new layout by {kernel}");
            }
            for (node, port, reason) in &report.reset {
                eprintln!("state of `{node}.{port}` restarted from its initial value: {reason}");
            }
            for refused in &report.refused {
                eprintln!("{refused}");
            }
            let names = |nodes: &[orla_graph::NodeId]| {
                nodes
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            if !report.landed.is_empty() {
                eprintln!("landed from off the frame: [{}]", names(&report.landed));
            }
            if !report.pending.is_empty() {
                eprintln!("still running off the frame: [{}]", names(&report.pending));
            }
            // The picture goes to `--out`; other outputs to `--save`.
            let evaluated = format!(
                "evaluated in {:?}: executed [{executed}], reused {}",
                started.elapsed(),
                report.reused.len()
            );
            let picture = session.picture_of(&report);
            let (Some(picture), Some(out)) = (picture, out) else {
                let why = match (picture, out) {
                    (None, Some(out)) => {
                        format!(
                            "; no output is a picture, so nothing went to {}",
                            out.display()
                        )
                    }
                    (Some(_), None) if saves.is_empty() => {
                        "; nothing written: name a file with `--out <file.png>`".to_owned()
                    }
                    _ => String::new(),
                };
                eprintln!("{evaluated}{why}");
                return true;
            };
            match session.export(picture, out) {
                Ok(()) => {
                    eprintln!("{evaluated} -> {}", out.display());
                    true
                }
                Err(error) => {
                    eprintln!("error: {error}");
                    false
                }
            }
        }
        Err(error) => {
            // Prefix the failing node's source location.
            let at = error
                .node()
                .and_then(|node| session.graph()?.node(node)?.at.clone())
                .map(|at| format!("{at}: "))
                .unwrap_or_default();
            eprintln!("{at}evaluation failed: {error}; last output is kept");
            // Where in the Kernel an element failed, traced again.
            if let Some(place) = session.failure_place(&error, &context) {
                eprintln!("  the failing operation is written at {place}");
            }
            false
        }
    }
}

/// A replay run once for each way of choosing the Kernels that round apart
/// ([`orla_session::apart`]), its failures named by the way that found them.
fn rounded_apart(
    replay: &orla::replay::Replay,
    resident: &mut orla::replay::Resident,
) -> Result<orla::replay::Outcome, String> {
    let mut all = orla::replay::Outcome::default();
    for split in 0..orla_session::apart::SPLITS {
        let outcome = resident.play(replay, Some(split))?;
        // Every way runs the same frames, checks and shows.
        all.frames = outcome.frames;
        all.checks = outcome.checks;
        all.shown = outcome.shown;
        all.refused = outcome.refused;
        all.failures.extend(
            outcome
                .failures
                .into_iter()
                .map(|failure| format!("rounded apart, way {}: {failure}", split + 1)),
        );
    }
    Ok(all)
}
