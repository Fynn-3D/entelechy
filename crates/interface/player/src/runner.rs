//! `entelechy-runner`: standalone native runner for compiled Entelechy
//! applications (`Bundle` or `Standalone` envelope).
//!
//! An exported Windows `.exe` or Linux binary carries its [`Standalone`]
//! envelope in a 24-byte trailer at the end of the executable; an exported
//! macOS `.app` keeps the Mach-O runner untouched in `Contents/MacOS/` so its
//! linker signature holds, and places `app.entapp` in `Contents/Resources/`.
//! Passing a `.bundle` or `.entapp` path on the command line runs it directly.

#![cfg_attr(
    all(target_os = "windows", not(debug_assertions)),
    windows_subsystem = "windows"
)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use entelechy_player::gamepad::Gamepad;
use entelechy_player::window::{
    PointerMode, SystemClipboard, WindowConfig, WindowHooks, WindowRunner, clock,
};
use entelechy_session::bundle::{Bundle, Standalone, read_trailer};
use entelechy_session::play::hud::loading_card;
use entelechy_session::play::{Frame, Play};
use entelechy_session::{Limits, Session, Wait};
use winit::event_loop::ActiveEventLoop;

const USAGE: &str = "\
usage:
  entelechy-runner [<file.bundle|file.entapp>] [--backend gpu|jit|best]
                   [--window <w>x<h>] [--scale <n>] [--no-audio]
                   [--hold-pointer | --lock-pointer] [--save-dir <dir>]
                   [--frames <n>]
      Run a compiled Entelechy scene bundle as a standalone native application.
      With no file argument, reads the application appended to this executable
      or bundled in ../Resources/app.entapp (macOS .app). Persistent scene saves
      are stored in the user's application support folder (or --save-dir <dir>).
      --frames <n> steps n frames headlessly and exits.";

struct RunnerArgs {
    file: Option<PathBuf>,
    backend: Option<String>,
    window: Option<(u32, u32)>,
    scale: u32,
    no_audio: bool,
    pointer: Option<PointerMode>,
    save_dir: Option<PathBuf>,
    frames: Option<u32>,
}

fn parse_args(args: &[String]) -> Result<RunnerArgs, String> {
    let mut parsed = RunnerArgs {
        file: None,
        backend: None,
        window: None,
        scale: 2,
        no_audio: false,
        pointer: None,
        save_dir: None,
        frames: None,
    };
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--backend" => {
                let name = iter.next().ok_or("--backend needs a value")?;
                if !matches!(name.as_str(), "gpu" | "jit" | "cpu" | "best") {
                    return Err(format!("--backend is gpu, jit, cpu or best, not `{name}`"));
                }
                parsed.backend = Some(name.clone());
            }
            "--window" => {
                let text = iter.next().ok_or("--window needs <w>x<h>")?;
                parsed.window = Some(parse_wh(text)?);
            }
            "--scale" => {
                parsed.scale = iter
                    .next()
                    .ok_or("--scale needs a positive integer")?
                    .parse::<u32>()
                    .ok()
                    .filter(|n| *n > 0)
                    .ok_or_else(|| "--scale needs a positive integer".to_owned())?;
            }
            "--no-audio" => parsed.no_audio = true,
            "--hold-pointer" => parsed.pointer = Some(PointerMode::Hold),
            "--lock-pointer" => parsed.pointer = Some(PointerMode::Lock),
            "--save-dir" => {
                parsed.save_dir = Some(PathBuf::from(
                    iter.next().ok_or("--save-dir needs a directory")?,
                ));
            }
            "--frames" => {
                parsed.frames = Some(
                    iter.next()
                        .ok_or("--frames needs a count")?
                        .parse::<u32>()
                        .map_err(|_| "--frames needs a non-negative integer".to_owned())?,
                );
            }
            other if other.starts_with("--") => return Err(format!("unknown flag {other}")),
            other => {
                if parsed.file.is_some() {
                    return Err("name at most one .bundle or .entapp file".into());
                }
                parsed.file = Some(PathBuf::from(other));
            }
        }
    }
    Ok(parsed)
}

fn parse_wh(text: &str) -> Result<(u32, u32), String> {
    let parts: Vec<&str> = text.split('x').collect();
    let [w, h] = parts.as_slice() else {
        return Err(format!("window size `{text}` is not <w>x<h>"));
    };
    let w = w
        .parse::<u32>()
        .ok()
        .filter(|n| *n > 0)
        .ok_or_else(|| format!("window width `{w}` must be positive"))?;
    let h = h
        .parse::<u32>()
        .ok()
        .filter(|n| *n > 0)
        .ok_or_else(|| format!("window height `{h}` must be positive"))?;
    Ok((w, h))
}

fn load_file(path: &Path) -> Result<Standalone, String> {
    if let Some(app) = read_trailer(path)? {
        return Ok(app);
    }
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if Standalone::has_magic(&bytes) {
        return Standalone::decode(&bytes).map_err(|e| format!("{}: {e}", path.display()));
    }
    let title = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Entelechy")
        .to_owned();
    Ok(Standalone {
        title,
        backend: "best".into(),
        pointer: PointerMode::Free,
        window: None,
        bundle: bytes,
    })
}

fn locate_embedded() -> Result<Standalone, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    if let Some(app) = read_trailer(&exe)? {
        return Ok(app);
    }
    if let Some(dir) = exe.parent() {
        let resources = dir.join("../Resources");
        for candidate in [
            resources.join("app.entapp"),
            resources.join("scene.bundle"),
            dir.join("app.entapp"),
            dir.join("scene.bundle"),
        ] {
            if candidate.is_file() {
                return load_file(&candidate);
            }
        }
    }
    Err(format!(
        "no embedded application found in {}; pass a .bundle file or export an app with `entelechy export`",
        exe.display()
    ))
}

fn save_slug(title: &str) -> String {
    let mut out = String::new();
    let mut prev_sep = false;
    for ch in title.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_sep = false;
        } else if !out.is_empty() && !prev_sep {
            out.push('-');
            prev_sep = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() { "app".into() } else { out }
}

fn default_save_dir(title: &str) -> Option<PathBuf> {
    let slug = save_slug(title);
    if cfg!(target_os = "macos") {
        let home = std::env::var_os("HOME")?;
        return Some(
            PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("Entelechy")
                .join(slug)
                .join("saves"),
        );
    }
    if cfg!(target_os = "windows") {
        if let Some(appdata) = std::env::var_os("APPDATA") {
            return Some(
                PathBuf::from(appdata)
                    .join("Entelechy")
                    .join(slug)
                    .join("saves"),
            );
        }
        let profile = std::env::var_os("USERPROFILE")?;
        return Some(
            PathBuf::from(profile)
                .join("AppData")
                .join("Roaming")
                .join("Entelechy")
                .join(slug)
                .join("saves"),
        );
    }
    if let Some(xdg) = std::env::var_os("XDG_DATA_HOME") {
        return Some(
            PathBuf::from(xdg)
                .join("entelechy")
                .join(slug)
                .join("saves"),
        );
    }
    let home = std::env::var_os("HOME")?;
    Some(
        PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("entelechy")
            .join(slug)
            .join("saves"),
    )
}

struct AppHooks {
    title: String,
    badge: String,
    save_dir: Option<PathBuf>,
    warming: bool,
}

impl AppHooks {
    fn finish_warming(runner: &mut WindowRunner<Self>) {
        runner.hooks.warming = false;
        runner.play.set_stage_card(None);
        if !runner.session.save_slots().is_empty()
            && let Some(dir) = &runner.hooks.save_dir
            && let Err(why) = runner.session.allow_bundle_saves(dir)
        {
            eprintln!(
                "warning: cannot configure saves in {}: {why}",
                dir.display()
            );
        }
        runner.after_scene_switch();
        let context = runner.play.context(runner.view());
        if let Ok(report) = runner.session.evaluate(&context)
            && let Ok((width, height)) = runner.session.runtime().host().image_size(report.output)
        {
            runner.last_size = (width, height);
            runner.fit_initial_output(width, height);
        }
    }
}

impl WindowHooks for AppHooks {
    fn initial_title(&self) -> &str {
        &self.title
    }

    fn title(&self, _session: &Session) -> String {
        self.title.clone()
    }

    fn is_loading(&self) -> bool {
        self.warming
    }

    fn poll_pending(runner: &mut WindowRunner<Self>, _event_loop: &ActiveEventLoop) {
        if !runner.hooks.warming {
            return;
        }
        if !runner.session.scene_waiting() && runner.session.preparing_left() == 0 {
            Self::finish_warming(runner);
            return;
        }
        let card = loading_card(
            &runner.hooks.badge,
            &runner.hooks.title,
            &runner.session.progress(),
        );
        runner.play.set_stage_card(Some(card));
    }
}

fn main() -> ExitCode {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    if raw_args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let args = match parse_args(&raw_args) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("error: {message}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let app = match &args.file {
        Some(path) => load_file(path),
        None => locate_embedded(),
    };
    let app = match app {
        Ok(app) => app,
        Err(message) => {
            eprintln!("error: {message}");
            return ExitCode::FAILURE;
        }
    };
    let bundle = match Bundle::decode(&app.bundle) {
        Ok(bundle) => bundle,
        Err(error) => {
            eprintln!("error: invalid bundle: {error}");
            return ExitCode::FAILURE;
        }
    };
    let title = if app.title.is_empty() {
        bundle.graph.name.clone()
    } else {
        app.title.clone()
    };
    let chosen_backend = args
        .backend
        .as_deref()
        .or_else(|| (!app.backend.is_empty()).then_some(app.backend.as_str()))
        .unwrap_or("best");
    let headless = args.frames.is_some();
    let loaded = match Session::open_bundle(
        bundle,
        chosen_backend,
        Limits::default(),
        Wait::blocking(),
        !headless,
    ) {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };
    let mut session = loaded.session;
    let save_dir = args.save_dir.clone().or_else(|| default_save_dir(&title));
    let warming = !headless && (session.scene_waiting() || session.preparing_left() > 0);
    if !warming
        && !session.save_slots().is_empty()
        && let Some(dir) = &save_dir
        && let Err(why) = session.allow_bundle_saves(dir)
    {
        eprintln!(
            "warning: cannot configure saves in {}: {why}",
            dir.display()
        );
    }
    session.give_clipboard(Box::new(SystemClipboard));
    session.set_background(true);

    let mut play = Play::new(clock);
    play.set_standalone(true);
    let badge = format!("native · {}", session.runtime().backend_name());
    if warming {
        play.set_stage_card(Some(loading_card(&badge, &title, &session.progress())));
    }

    let window = args.window.or(app.window);
    if let Some(count) = args.frames {
        let (w, h) = window.unwrap_or((320, 180));
        let view = entelechy_session::play::View {
            window: [w as f32, h as f32],
            scale: 1.0,
            rate: 48_000.0,
            queued: 0,
        };
        let mut size = (0u32, 0u32);
        for step in 0..count.max(1) {
            match play.frame(&mut session, view) {
                Frame::Shown(shown) => {
                    if let Ok((pw, ph)) = session.runtime().host().image_size(shown.picture) {
                        size = (pw, ph);
                    }
                    if !play.confirm(&mut session) {
                        eprintln!("error: frame {step} failed: {}", play.status());
                        return ExitCode::FAILURE;
                    }
                }
                Frame::Failed => {
                    eprintln!("error: frame {step} failed: {}", play.status());
                    return ExitCode::FAILURE;
                }
                Frame::Wait => {}
            }
        }
        println!(
            "{}: ran {} frame(s) on {} ({}x{})",
            title,
            count.max(1),
            session.runtime().backend_name(),
            size.0,
            size.1
        );
        return ExitCode::SUCCESS;
    }

    let gamepad = Gamepad::open();
    let config = WindowConfig {
        scale: args.scale,
        requested: window,
        pointer_mode: args.pointer.unwrap_or(app.pointer),
        audio_allowed: !args.no_audio,
    };
    let mut runner = WindowRunner::new(
        session,
        play,
        loaded.display,
        gamepad,
        config,
        AppHooks {
            title,
            badge,
            save_dir,
            warming,
        },
    );
    if !runner.hooks.warming {
        let context = runner.play.context(runner.view());
        if let Ok(report) = runner.session.evaluate(&context)
            && let Ok((width, height)) = runner.session.runtime().host().image_size(report.output)
        {
            runner.last_size = (width, height);
        }
    }
    runner.run()
}
