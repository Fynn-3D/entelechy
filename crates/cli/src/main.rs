//! The `orla` command.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use orla::session::{Event, Session};
use orla_runtime::Context;

const USAGE: &str = "\
usage:
  orla check [--ir] <file.orla|file.osn>...
      Parse documents, compile their Kernels, and print manifests or diagnostics.
  orla run <scene.osn> [--library <dir|file>] [--input <name>=<file.ppm>]...
                       [--save <node.port>=<file.field|.ppm|.pgm>]...
                       [--out <file.ppm>] [--watch] [--backend cpu|gpu]
                       [--param <name>=<literal>]... [--no-fusion]
      Evaluate a scene and write its output. With --watch, keep running and
      republish Kernels as their files change; a rejected change keeps the
      running version.
  orla replay <file.replay>... [--backend cpu|gpu]
      Drive scenes through scripted frames and inputs and check pixels.
  orla graph <scene.osn> [--library <dir|file>] [--no-fusion]
      Print the flattened graph and what would execute, fused groups marked.
  orla serve <scene.osn> [--library <dir|file>] [--backend cpu|gpu]
      Serve the creation protocol: JSON requests on stdin, one response per
      line on stdout. See crates/cli/src/protocol.rs for the operations.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("check") => check(&args[1..]),
        Some("run") => run(&args[1..]),
        Some("replay") => replay(&args[1..]),
        Some("serve") => serve(&args[1..]),
        Some("graph") => graph(&args[1..]),
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
        let text = std::fs::read_to_string(file).map_err(|e| format!("{file}: {e}"))?;
        let document = match orla_language::parse_document(&text) {
            Ok(document) => document,
            Err(error) => {
                failures += 1;
                println!("{file}:{error}");
                continue;
            }
        };
        let name = Path::new(file)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("scene");
        let base = Path::new(file).parent().unwrap_or_else(|| Path::new("."));
        let mut loader = |path: &Path| -> Result<String, String> {
            std::fs::read_to_string(path).map_err(|error| error.to_string())
        };
        let mut declare =
            |base: &Path, uses: &[orla_language::Use]| orla::session::declarations_for(base, uses);
        let lowered = match orla_graph::Graph::load_with(
            &document,
            &text,
            name,
            base,
            &mut loader,
            &mut declare,
        ) {
            Ok(lowered) => lowered,
            Err(error) => {
                failures += 1;
                println!("{file}:{error}");
                continue;
            }
        };
        let mut document = document;
        if let Ok((structs, types, functions)) =
            orla::session::declarations_for(base, &document.uses)
        {
            document.inherit(&structs, &types, &functions);
        }
        let mut kernels: Vec<(Option<orla_ir::KernelId>, &orla_language::Kernel)> = document
            .kernels
            .iter()
            .map(|kernel| (None, kernel))
            .collect();
        kernels.extend(
            lowered
                .inline
                .iter()
                .map(|inline| (Some(inline.id.clone()), &inline.kernel)),
        );
        for (id, kernel) in kernels {
            match orla_compiler::compile_kernel(kernel, id.as_ref()) {
                Ok(module) => {
                    print!("{file}: {}\n{}", module.manifest.id, module.manifest);
                    if show_ir {
                        println!("{module}");
                    }
                }
                Err(error) => {
                    failures += 1;
                    println!("{file}:{error}");
                }
            }
        }
        // The documents it uses, with their own declarations: a scene is
        // not checked until its libraries are.
        match orla::session::used_documents(base, &document.uses) {
            Ok(used) => {
                let mut compiled = 0;
                for path in &used {
                    let text = match std::fs::read_to_string(path) {
                        Ok(text) => text,
                        Err(error) => {
                            failures += 1;
                            println!("{}: {error}", path.display());
                            continue;
                        }
                    };
                    let Ok(mut used_document) = orla_language::parse_document(&text) else {
                        continue; // reported by `used_documents`
                    };
                    let dir = path.parent().unwrap_or_else(|| Path::new("."));
                    match orla::session::declarations_for(dir, &used_document.uses) {
                        Ok((structs, types, functions)) => {
                            used_document.inherit(&structs, &types, &functions)
                        }
                        Err(error) => {
                            failures += 1;
                            println!("{}: {error}", path.display());
                        }
                    }
                    for kernel in &used_document.kernels {
                        match orla_compiler::compile_kernel(kernel, None) {
                            Ok(module) => {
                                compiled += 1;
                                if show_ir {
                                    println!(
                                        "{}: {}\n{module}",
                                        path.display(),
                                        module.manifest.id
                                    );
                                }
                            }
                            Err(error) => {
                                failures += 1;
                                println!("{}:{error}", path.display());
                            }
                        }
                    }
                }
                if !used.is_empty() {
                    println!(
                        "{file}: {compiled} kernel(s) compiled from {} used document(s)",
                        used.len()
                    );
                }
            }
            Err(error) => {
                failures += 1;
                println!("{file}: {error}");
            }
        }
        if !document.nodes.is_empty() {
            match lowered.graph.order() {
                Ok(order) => println!(
                    "{file}: graph `{}` with {} node(s), {} use(s)",
                    lowered.graph.name,
                    order.len(),
                    document.uses.len()
                ),
                Err(error) => {
                    failures += 1;
                    println!("{file}: {error}");
                }
            }
        }
    }
    if failures > 0 {
        Err(format!("{failures} problem(s) found"))
    } else {
        Ok(())
    }
}

struct RunOptions {
    scene: PathBuf,
    kernels: Option<PathBuf>,
    inputs: Vec<(String, PathBuf)>,
    saves: Vec<(String, String, PathBuf)>,
    out: PathBuf,
    watch: bool,
    backend: String,
    params: Vec<(String, orla_runtime::Value)>,
    fusion: bool,
}

fn parse_run(args: &[String]) -> Result<RunOptions, String> {
    let mut options = RunOptions {
        scene: PathBuf::new(),
        kernels: None,
        inputs: Vec::new(),
        saves: Vec::new(),
        out: PathBuf::from("output.ppm"),
        watch: false,
        backend: "cpu".into(),
        params: Vec::new(),
        fusion: true,
    };
    let mut scene = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let mut value = |flag: &str| {
            iter.next()
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match arg.as_str() {
            "--library" | "--kernels" => options.kernels = Some(PathBuf::from(value("--library")?)),
            "--out" => options.out = PathBuf::from(value("--out")?),
            "--watch" => options.watch = true,
            "--no-fusion" => options.fusion = false,
            "--backend" => options.backend = value("--backend")?,
            "--param" => {
                let binding = value("--param")?;
                let (name, text) = binding
                    .split_once('=')
                    .ok_or_else(|| "--param needs <name>=<literal>".to_string())?;
                let expr = orla_language::parse_literal(text).map_err(|e| e.to_string())?;
                let literal = orla_compiler::literal(&expr).map_err(|e| e.to_string())?;
                options.params.push((name.to_owned(), literal));
            }
            "--input" => {
                let binding = value("--input")?;
                let (name, path) = binding
                    .split_once('=')
                    .ok_or_else(|| "--input needs <name>=<file.ppm>".to_string())?;
                options.inputs.push((name.to_owned(), PathBuf::from(path)));
            }
            "--save" => {
                let binding = value("--save")?;
                let (target, path) = binding
                    .split_once('=')
                    .ok_or_else(|| "--save needs <node.port>=<file>".to_string())?;
                let (node, port) = target.split_once('.').unwrap_or((target, "result"));
                options
                    .saves
                    .push((node.to_owned(), port.to_owned(), PathBuf::from(path)));
            }
            other if other.starts_with("--") => return Err(format!("unknown flag {other}")),
            other => {
                if scene.replace(PathBuf::from(other)).is_some() {
                    return Err("run takes exactly one scene file".into());
                }
            }
        }
    }
    options.scene = scene.ok_or_else(|| "run needs a scene file".to_string())?;
    Ok(options)
}

fn run(args: &[String]) -> Result<(), String> {
    let options = parse_run(args)?;
    let runtime = Session::runtime_for(&options.backend, orla::session::DEFAULT_BUDGET_PIXELS)?;
    let mut session = Session::open_with(&options.scene, options.kernels.as_deref(), runtime);
    for (name, path) in &options.inputs {
        session
            .bind_external(name, path)
            .map_err(|e| e.to_string())?;
    }
    for (name, value) in &options.params {
        session.set_param(name, *value);
    }
    session.set_fusion(options.fusion);
    for (node, _, _) in &options.saves {
        session.pin(orla_graph::NodeId(node.clone()));
    }
    let events = session.sync();
    report_events(&events);
    let clock = Instant::now();
    let mut healthy = evaluate_and_export(&mut session, &options.out, &options.saves, 0.0);
    let _ = &clock;
    if !options.watch {
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
                &options.out,
                &options.saves,
                clock.elapsed().as_secs_f32(),
            );
        }
    }
}

fn graph(args: &[String]) -> Result<(), String> {
    let mut library = None;
    let mut scene = None;
    let mut fusion = true;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--library" => {
                library = Some(PathBuf::from(
                    iter.next()
                        .ok_or_else(|| "--library needs a value".to_string())?,
                ));
            }
            "--no-fusion" => fusion = false,
            other if other.starts_with("--") => return Err(format!("unknown flag {other}")),
            other => {
                if scene.replace(PathBuf::from(other)).is_some() {
                    return Err("graph takes exactly one scene file".into());
                }
            }
        }
    }
    let scene = scene.ok_or_else(|| "graph needs a scene file".to_string())?;
    let mut session = Session::open(
        &scene,
        library.as_deref(),
        orla::session::DEFAULT_BUDGET_PIXELS,
    );
    session.set_fusion(fusion);
    let events = session.sync();
    for event in &events {
        if matches!(event, Event::Rejected { .. } | Event::SceneRejected(_)) {
            eprintln!("{event}");
        }
    }
    let graph = session
        .graph()
        .ok_or_else(|| "the scene did not load".to_string())?;
    print!("{graph}");
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
    let mut backend = "cpu".to_string();
    let mut library = None;
    let mut scene = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--backend" => {
                backend = iter
                    .next()
                    .cloned()
                    .ok_or_else(|| "--backend needs a value".to_string())?;
            }
            "--library" => {
                library = Some(PathBuf::from(
                    iter.next()
                        .ok_or_else(|| "--library needs a value".to_string())?,
                ));
            }
            other if other.starts_with("--") => return Err(format!("unknown flag {other}")),
            other => {
                if scene.replace(PathBuf::from(other)).is_some() {
                    return Err("serve takes exactly one scene file".into());
                }
            }
        }
    }
    let scene = scene.ok_or_else(|| "serve needs a scene file".to_string())?;
    let runtime = Session::runtime_for(&backend, orla::session::DEFAULT_BUDGET_PIXELS)?;
    let mut session = Session::open_with(&scene, library.as_deref(), runtime);
    report_events(&session.sync());
    session.set_background(true);
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    orla::protocol::serve(&mut session, stdin.lock(), stdout.lock()).map_err(|e| e.to_string())
}

fn replay(args: &[String]) -> Result<(), String> {
    let mut backend = "cpu".to_string();
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
        let runtime = Session::runtime_for(&backend, orla::session::DEFAULT_BUDGET_PIXELS)?;
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

fn report_events(events: &[Event]) {
    for event in events {
        eprintln!("{event}");
    }
}

fn evaluate_and_export(
    session: &mut Session,
    out: &Path,
    saves: &[(String, String, PathBuf)],
    time: f32,
) -> bool {
    let started = Instant::now();
    let context = Context {
        time,
        ..Context::default()
    };
    match session.evaluate(&context) {
        Ok(report) => {
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
            for (node, port, path) in saves {
                match session.save(&orla_graph::NodeId(node.clone()), port, path) {
                    Ok(()) => eprintln!("saved {node}.{port} -> {}", path.display()),
                    Err(error) => {
                        eprintln!("error: cannot save {node}.{port}: {error}");
                        return false;
                    }
                }
            }
            match session.export(report.output, out) {
                Ok(()) => {
                    eprintln!(
                        "evaluated in {:?}: executed [{executed}], reused {} -> {}",
                        started.elapsed(),
                        report.reused.len(),
                        out.display()
                    );
                    true
                }
                Err(error) => {
                    eprintln!("export failed: {error}");
                    false
                }
            }
        }
        Err(error) => {
            eprintln!("evaluation failed: {error}; last output is kept");
            false
        }
    }
}
