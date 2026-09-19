//! The `orla` command.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use orla::session::{Event, Session};
use orla_runtime::Context;

const USAGE: &str = "\
usage:
  orla check [--ir] <file.orla>...
      Compile Kernels and print their manifests, or diagnostics.
  orla run <scene.osn> [--kernels <dir>] [--input <name>=<file.ppm>]...
                       [--out <file.ppm>] [--watch] [--backend cpu|gpu]
      Evaluate a scene and write its output. With --watch, keep running and
      republish Kernels as their files change; a rejected change keeps the
      running version.
  orla replay <file.replay>... [--backend cpu|gpu]
      Drive scenes through scripted frames and inputs and check pixels.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("check") => check(&args[1..]),
        Some("run") => run(&args[1..]),
        Some("replay") => replay(&args[1..]),
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
        return Err("check needs at least one .orla file".into());
    }
    let mut failures = 0;
    for file in files {
        let source = std::fs::read_to_string(file).map_err(|e| format!("{file}: {e}"))?;
        match orla_compiler::compile(&source) {
            Ok(module) => {
                print!("{file}: ok\n{}", module.manifest);
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
    if failures > 0 {
        Err(format!("{failures} kernel(s) failed to compile"))
    } else {
        Ok(())
    }
}

struct RunOptions {
    scene: PathBuf,
    kernels: Option<PathBuf>,
    inputs: Vec<(String, PathBuf)>,
    out: PathBuf,
    watch: bool,
    backend: String,
}

fn parse_run(args: &[String]) -> Result<RunOptions, String> {
    let mut options = RunOptions {
        scene: PathBuf::new(),
        kernels: None,
        inputs: Vec::new(),
        out: PathBuf::from("output.ppm"),
        watch: false,
        backend: "cpu".into(),
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
            "--kernels" => options.kernels = Some(PathBuf::from(value("--kernels")?)),
            "--out" => options.out = PathBuf::from(value("--out")?),
            "--watch" => options.watch = true,
            "--backend" => options.backend = value("--backend")?,
            "--input" => {
                let binding = value("--input")?;
                let (name, path) = binding
                    .split_once('=')
                    .ok_or_else(|| "--input needs <name>=<file.ppm>".to_string())?;
                options.inputs.push((name.to_owned(), PathBuf::from(path)));
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
    let events = session.sync();
    report_events(&events);
    let clock = Instant::now();
    let mut healthy = evaluate_and_export(&mut session, &options.out, 0.0);
    let _ = &clock;
    if !options.watch {
        return if healthy {
            Ok(())
        } else {
            Err("the scene did not evaluate".into())
        };
    }
    eprintln!(
        "watching {} and {}; edit a .orla file and save",
        options.scene.display(),
        session.kernels_dir().display()
    );
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
            healthy =
                evaluate_and_export(&mut session, &options.out, clock.elapsed().as_secs_f32());
        }
    }
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

fn evaluate_and_export(session: &mut Session, out: &Path, time: f32) -> bool {
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
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
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
