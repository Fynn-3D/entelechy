//! The `orla` command.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use orla_kernel::Context;
use orla_session::{Event, Session};

const USAGE: &str = "\
usage:
  orla check [--ir] <file.orla|file.osn>...
      Parse documents, compile their Kernels, and print manifests or diagnostics.
      --ir prints each Kernel's IR as a backend runs it: where its work is
      counted (`work spend`) and looked at (`work look`) included.
  orla run <scene.osn> [--library <dir|file>] [--input <name>=<file.ppm>]...
                       [--save <node.port>=<file.field|.png|.ppm|.pgm|.wav>]...
                       [--out <file.png|.ppm|.field>] [--watch] [--backend cpu|gpu|jit]
                       [--param <name>=<literal>]... [--no-fusion]
                       [--window <w>x<h>[@<scale>]] [--profile] [--memory <MB>] [--work <M>]
                       [--allow file.read|file.write=<dir>]... [--grant <capability,...>]
      Evaluate a scene; write its picture to --out and each --save, in the
      format each file's extension names, and nothing else. With --watch, keep running and
      republish Kernels as their files change; a rejected change keeps the
      running version. --window is the window a scene sized by `window()`
      would fill, in pixels, with its scale factor.
      --memory says how many mebibytes of Fields the Host may hold at once
      (2048 by default); a scene of millions of triangles wants more, and
      `run`, `render`, `replay`, `serve`, `graph`, `trace` and `bench` all
      take it. --work says how many million loop iterations one run of a
      Kernel may take, all its elements together (4000 by default), and
      the same commands take it.
      --profile prints how long each executed node took; it waits for the
      device after each one, so a GPU's numbers are the work, not the
      submission, and the frame is slower than it would be. What
      measuring a node costs before it does anything is taken off every
      line; a node runs alone under a profile, so the lines add up to
      more than a frame, and the sum says by how much.
      --allow file.write=<dir> lets the scene's outputs written `to` a file
      land in that directory and under it, once an evaluation that changed
      them commits; nothing is written without it. `render`, `serve` and
      the Player take it too. --allow file.read=<dir> lets a served
      scene's inputs read files there: under `serve`, where an agent
      composes the scene, a scene reads only beside its documents
      without it.
      --grant field.read,field.write,... names the capabilities a Kernel
      may be given, and no others (every one without it): a Kernel that
      requires another is refused when it is prepared, and what ran
      before goes on running. `render`, `serve`, `bench` and the Player
      take it too.
  orla render <scene.osn> --frames <n> [--rate <fps>] [--sound-rate <hz>]
                          [--out <dir>] [--library <dir|file>] [--input <name>=<file.ppm>]...
                          [--param <name>=<literal>]... [--window <w>x<h>[@<scale>]]
                          [--backend cpu|gpu|jit] [--no-fusion]
                          [--save <node.port>=<file.field|.png|.ppm|.pgm|.wav>]...
                          [--allow file.read|file.write=<dir>]... [--grant <capability,...>]
      Step the scene for n frames at the rate (60 by default) and write each
      frame's picture as <dir>/0000.png, 0001.png, ... and the sound output,
      block after block, as <dir>/sound.wav at the sound rate (48000 by
      default, what Kernels read as rate()). The clock is the frame count
      over the rate, so a render is the same every time; async nodes are
      waited for after every frame. ffmpeg makes a video of the directory.
      --save writes a node's output after the last frame, as run's does:
      how a state trained or settled over many frames is kept.
  orla trace <scene.osn> --node <name> (--element <i> | --at <x> <y> [<z>])
                         [--time <seconds>] [--param <name>=<literal>]...
                         [--window <w>x<h>[@<scale>]] [--library <dir|file>]
                         [--backend cpu|gpu|jit]
      Evaluate the scene, then run one element of a node's Kernel (or one
      step of a scan) on the interpreter and print every `let` of it with
      its value, and the result. Nothing is committed by the trace.
  orla help [<name>]
      List every builtin a Kernel may call with one line each, or show one
      builtin's signatures, what it does, and the capability it needs.
  orla version
      Print Orla's release and the versions of the ABI, the language, the IR
      and the package format, each its own.
  orla package <dir.orlapkg>
      Check a package directory's manifest and say what it holds.
  orla pack <dir.orlapkg> [<file.orlapkg>]
      Write a package directory as one archive file, its manifest checked.
  orla unpack <file.orlapkg> <new dir>
      Read an archive back into a directory, its manifest checked.
  orla replay <file.replay>... [--backend cpu|gpu|jit]
      Drive scenes through scripted frames and inputs and check pixels.
  orla new <dir> [--kind picture|sound|game]
      Write a scene that runs: a Kernel, the scene that places it, a replay
      that checks what it draws, and a README saying what to change first.
      `picture` is one Kernel over the window, `sound` a voice with a
      drawing of it, `game` a grid of state the arrow keys move.
  orla bench <scene.osn> [--frames <n>] [--repeat <n>] [--rate <fps>]
                         [--backend cpu|gpu|jit|all]... [--nodes]
                         [--window <w>x<h>[@<scale>]] [--library <dir|file>]
                         [--param <name>=<literal>]... [--input <name>=<file>]...
                         [--no-fusion] [--json <file>] [--compare <file>]
      Step the scene for --frames frames --repeat times on each backend and
      print the least of the runs, a millisecond a frame. The first frame
      of all is thrown away: it compiles and allocates. --nodes measures
      every node as well, waiting for the device after each one, native
      operations included; what measuring a node costs before it does
      anything is taken off every line, and the sum says how the parts
      come to more than the whole, since a node runs alone there and a
      frame overlaps them. --json
      writes the measurement, and --compare reads one back and prints what
      changed since, which is how a change is shown to cost nothing.
  orla graph <scene.osn> [--library <dir|file>] [--no-fusion]
      Print the flattened graph and what would execute, fused groups marked.
  orla serve <scene.osn> [--library <dir|file>] [--backend cpu|gpu|jit]
                         [--input <name>=<file.ppm>]... [--param <name>=<literal>]...
                         [--no-fusion] [--memory <MB>] [--work <M>]
                         [--listen <host:port>] [--window <w>x<h>[@<scale>]]
                         [--allow file.read|file.write=<dir>]... [--allow-origin <origin>]...
                         [--grant <capability,...>]
      Serve the creation protocol: JSON requests on stdin, one response per
      line on stdout, and on every connection to --listen, where a client's
      quit ends only its connection. A browser's page connects from this
      machine or from a file; a page elsewhere only from an origin
      --allow-origin names (`https://host:port`). --window is what evaluations that name
      no window see (the protocol's `window` op changes it). See
      docs/PROTOCOL.md.";

fn usage_error(message: &str) -> Result<(), String> {
    Err(format!("{message}\n{USAGE}"))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
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
            eprintln!("{USAGE}");
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
    let mut failures = 0;
    for file in files {
        let checked = orla_session::check_document(&orla_session::Disk, Path::new(file), show_ir);
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
    Ok((node, port, PathBuf::from(path)))
}

fn parse_render(args: &[String]) -> Result<RenderOptions, String> {
    let mut options = RenderOptions {
        session: orla_session::SessionOptions::default(),
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
    let out = options.render.out.display();
    if let Some((width, height)) = summary.size {
        eprintln!(
            "rendered {} picture(s) of {width}x{height} to {out}/0000.png ... in {:?}",
            summary.pictures,
            started.elapsed()
        );
    }
    if summary.channels > 0 {
        eprintln!(
            "rendered {:.2} s of sound ({} channel(s) at {} Hz) to {out}/{}",
            summary.sound_frames as f64 / options.render.sound_rate as f64,
            summary.channels,
            options.render.sound_rate,
            orla::render::SOUND_NAME
        );
    }
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
        session: orla_session::SessionOptions::default(),
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
            "--out" => options.out = Some(PathBuf::from(value("--out")?)),
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
    eprintln!(
        "play it:  orla-player {}\nrun it:   orla run {} --window 640x400 --out frame.ppm\ntest it:  orla replay {}",
        scene.display(),
        scene.display(),
        replay.display()
    );
    Ok(())
}

/// `orla bench`: the same frames, repeated, on one backend or several.
fn measure(args: &[String]) -> Result<(), String> {
    let mut options = orla_session::bench::Options::default();
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
    let mut options = orla_session::SessionOptions::default();
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
        .ok_or_else(|| "the scene did not load".to_string())?;
    print!("{}", orla_scene::print(graph));
    let kernels: Vec<String> = session
        .runtime()
        .kernels()
        .map(|version| format!("{} {}", version.id(), version.generation()))
        .collect();
    println!("\n// published: {}", kernels.join(", "));
    match session.plan() {
        Ok(plan) => {
            println!("// plan: {} group(s)", plan.len());
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
                println!("//   {}{absorbed}: {}{domain}", entry.node, entry.kernel);
            }
            Ok(())
        }
        Err(error) => Err(format!("cannot plan: {error}")),
    }
}

fn serve(args: &[String]) -> Result<(), String> {
    let mut options = orla_session::SessionOptions::default();
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
    // requests are answered meanwhile.
    let mut answers = orla_protocol::Answers::new();
    loop {
        let incoming = if answers.waiting() {
            match requests.recv_timeout(std::time::Duration::from_millis(1)) {
                Ok(incoming) => Some(incoming),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => None,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
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

fn replay(args: &[String]) -> Result<(), String> {
    let mut backend = "jit".to_string();
    let mut limits = orla_session::Limits::default();
    let mut files = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--backend" => {
                backend = iter
                    .next()
                    .cloned()
                    .ok_or_else(|| "--backend needs a value".to_string())?;
            }
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
    let mut failed = 0;
    for file in &files {
        let text = std::fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
        let base = file.parent().unwrap_or_else(|| Path::new("."));
        let replay = orla::replay::parse(&text, base)
            .map_err(|error| format!("{}: {error}", file.display()))?;
        let runtime = Session::runtime_for(&backend, limits)?;
        match orla::replay::run(&replay, runtime) {
            Ok(outcome) if outcome.failures.is_empty() => {
                println!(
                    "{}: ok ({} frames, {} checks)",
                    file.display(),
                    outcome.frames,
                    outcome.checks
                );
            }
            Ok(outcome) => {
                failed += 1;
                println!(
                    "{}: {} of {} checks failed",
                    file.display(),
                    outcome.failures.len(),
                    outcome.checks
                );
                for failure in &outcome.failures {
                    println!("  {failure}");
                }
            }
            Err(error) => {
                failed += 1;
                println!("{}: {error}", file.display());
            }
        }
    }
    if failed > 0 {
        Err(format!("{failed} replay(s) failed"))
    } else {
        Ok(())
    }
}

/// `orla help`: the builtins, all or one.
fn help(args: &[String]) -> Result<(), String> {
    match args.first() {
        None => {
            let builtins = orla_compiler::help::builtins();
            let width = builtins.iter().map(|b| b.name.len()).max().unwrap_or(0);
            let mut kind = "";
            for builtin in &builtins {
                if builtin.kind != kind {
                    kind = builtin.kind;
                    println!("{kind}");
                }
                println!("  {:<width$}  {}", builtin.name, builtin.doc);
            }
            let forms = orla_compiler::help::forms();
            let width = forms.iter().map(|f| f.name.len()).max().unwrap_or(0);
            let mut kind = "";
            for form in &forms {
                if form.kind != kind {
                    kind = form.kind;
                    println!("{kind}");
                }
                println!("  {:<width$}  {}", form.name, form.shape);
            }
            println!(
                "orla help <name> shows a builtin's signatures or a form's shape and meaning; docs/DOCUMENTS.md the language."
            );
            Ok(())
        }
        Some(name) => {
            if let Some(form) = orla_compiler::help::form_named(name) {
                println!("{} ({})", form.name, form.kind);
                println!("  {}", form.shape);
                println!("{}", form.doc);
                return Ok(());
            }
            let Some(builtin) = orla_compiler::help::builtin(name) else {
                let similar = orla_compiler::help::similar(name);
                return Err(if similar.is_empty() {
                    format!("nothing named `{name}`; `orla help` lists the builtins and the forms")
                } else {
                    format!(
                        "nothing named `{name}`; did you mean {}?",
                        similar.join(", ")
                    )
                });
            };
            println!("{} ({})", builtin.name, builtin.kind);
            for signature in &builtin.signatures {
                println!("  {signature}");
            }
            println!("{}", builtin.doc);
            if let Some(capability) = builtin.requires {
                println!("Needs `{capability}` in the Kernel's `requires`.");
            }
            if builtin.entry_only {
                println!(
                    "Only the entry (`compute`, `scan`, `scatter`) may call it; a function takes its value as an argument."
                );
            }
            Ok(())
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
    let (traced, kernel) = session.trace(&id, at, &context)?;
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
    println!("{node} ({kernel}), {place}, as the evaluation after time {time} runs it");
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
            false
        }
    }
}
