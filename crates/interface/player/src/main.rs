//! The `entelechy-player` command: a window, sound card and pads over a live
//! session. Frames, key meanings and the frame line are
//! `entelechy_session::play`'s; the window loop is `entelechy_player::window`.

use std::process::ExitCode;
use std::sync::mpsc::{Receiver, channel};

use entelechy_player::gamepad::Gamepad;
use entelechy_player::window::{
    PointerMode, SystemClipboard, WindowConfig, WindowHooks, WindowRunner, clock,
};
use entelechy_session::Session;
use entelechy_session::play::Play;
use serde_json::Value as Json;
use winit::event_loop::ActiveEventLoop;

const USAGE: &str = "\
usage:
  entelechy-player <scene.esn> [--library <dir|file>] [--input <name>=<path>]...
              [--output <name>=<path>]... [--param <name>=<literal>]... [--scale <n>]
              [--folder <name,...> [--folder-allow <file.write,file.delete>]]
              [--replay <file.replay>] [--show-while-building]
              [--window <w>x<h>] [--backend gpu|cpu|jit|best] [--memory <MB>] [--work <M>]
              [--allow file.read|file.write|file.delete=<dir>]...
              [--allow clipboard.read|clipboard.write]... [--grant <capability,...>]
      Open a window showing the scene output once its Kernels have first
      prepared, the terminal showing them preparing as `run` does: a line of
      seeds, one for each Kernel. Kernel files are republished as
      they change, the disk looked at four times a second; a rejected change
      keeps the running version. The pointer
      and keys reach Kernels that declare `input.read`, and so do the
      window's size in pixels and its scale factor, which a node's `domain`
      may take as `window()`; --window opens the window at that logical size
      instead of the output's size times --scale. F5 resets state;
      F6 cycles the preview through the graph's nodes and back to the output;
      F10 shows a line of frame data over the picture, naming the port shown.
      F8 holds the scene's time still or lets it run on; F9 steps it a
      frame (1/60 s) on, F7 a frame back. Frames go on while it is held,
      so the scene still takes the keys and pointer: what it draws from
      its time stands still.
      Tab, Escape and every other key are the scene's.
      --param <name>=<literal> sets a parameter the scene declares, before
      its first frame.
      --show-while-building opens the window once what state and kept
      outputs read is built, what only shows drawn by the processor until
      the device has built it: a sooner window, slower frames meanwhile.
      --folder <name,...> lets Cmd+O (Ctrl+O elsewhere) choose a folder in the
      system's dialog for those inputs and outputs, each one that reads or
      writes a directory (files, checks, outline, to files): placed there at
      once, the folder allowed reads and what --folder-allow
      <file.write,file.delete> says, a folder chosen before given back.
      --input and --output read and write the scene's input or output at the
      path given in place of the one it names.
      --hold-pointer holds the pointer in the window once it is clicked:
      the system's cursor is locked and hidden, the scene's pointer moves
      with the mouse and stays inside the window (the scene draws it);
      the window losing the focus (Cmd+Tab) lets it go.
      --lock-pointer locks and hides the system's cursor once clicked and
      moves the scene's pointer by the mouse's motion without clamping to
      the window's edges, for a first-person view; Escape or losing focus
      lets it go.
      --memory says how many mebibytes of Fields the Host may hold at once,
      2048 by default; a scene of millions of triangles wants more.
      --work says how many million loop iterations one run of a Kernel may
      take, all its elements together, 4000 by default.
      --allow file.write=<dir> lets the scene's outputs written `to` a file
      land in that directory; nothing is written without it.
      --allow file.delete=<dir> lets an output `to files` remove and
      rename files there.
      --allow file.read=<dir> lets a served scene's inputs read files
      there; with --serve a scene reads only beside its documents without
      it.
      --allow clipboard.read lets an input `= clipboard` read the system's
      clipboard, --allow clipboard.write an output `to clipboard` put text
      there; without them the clipboard reads empty and nothing is put.
      --replay <file.replay> plays a replay in the window: its pointer,
      keys and window drive the scene at the replay's own times, its
      checks run as they come, and when it ends what it observed is
      printed and the window is yours. The replay names its scene, so
      `entelechy-player --replay <file.replay>` is enough.
      Close the window, or press Cmd+Q, to quit.
      With --serve, the creation protocol runs on stdin and stdout inside the
      window, and on every connection to --listen <host:port>: an agent's
      publish, compose, and param land in the frame a person watches, and
      its evaluate observes that person's input. A browser's page connects
      from this machine; a page elsewhere only from an origin
      --allow-origin <origin> names (`null` for a page opened from a file).";

struct Options {
    session: entelechy_session::SessionOptions,
    scale: u32,
    window: Option<(u32, u32)>,
    serve: bool,
    listen: Option<String>,
    origins: Vec<String>,
    no_audio: bool,
    show_while_building: bool,
    replay: Option<entelechy_session::replay::Replay>,
    pointer_mode: PointerMode,
    folder: Vec<String>,
    folder_allow: Vec<String>,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut options = Options {
        session: entelechy_session::SessionOptions::from_environment(),
        scale: 2,
        window: None,
        serve: false,
        listen: None,
        origins: Vec::new(),
        no_audio: false,
        show_while_building: false,
        replay: None,
        pointer_mode: PointerMode::Free,
        folder: Vec::new(),
        folder_allow: Vec::new(),
    };
    let mut iter = args.iter().cloned();
    while let Some(arg) = iter.next() {
        if options.session.read(&arg, &mut || iter.next())? {
            continue;
        }
        let mut value = |flag: &str| iter.next().ok_or_else(|| format!("{flag} needs a value"));
        match arg.as_str() {
            "--serve" => options.serve = true,
            "--listen" => {
                options.serve = true;
                options.listen = Some(value("--listen")?);
            }
            "--allow-origin" => options.origins.push(value("--allow-origin")?),
            "--no-audio" => options.no_audio = true,
            "--show-while-building" => options.show_while_building = true,
            "--hold-pointer" => options.pointer_mode = PointerMode::Hold,
            "--lock-pointer" => options.pointer_mode = PointerMode::Lock,
            "--folder" => options.folder.extend(listed(&value("--folder")?)),
            "--folder-allow" => {
                for grant in listed(&value("--folder-allow")?) {
                    if !["file.read", "file.write", "file.delete"].contains(&grant.as_str()) {
                        return Err(format!(
                            "--folder-allow takes file.read, file.write and file.delete, not `{grant}`"
                        ));
                    }
                    options.folder_allow.push(grant);
                }
            }
            "--replay" => {
                let path = std::path::PathBuf::from(value("--replay")?);
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                let base = path.parent().unwrap_or(std::path::Path::new("."));
                let replay = entelechy_session::replay::parse(&text, base)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                options.replay = Some(replay);
            }
            "--scale" => {
                options.scale = value("--scale")?
                    .parse::<u32>()
                    .ok()
                    .filter(|scale| *scale > 0)
                    .ok_or_else(|| "--scale needs a positive integer".to_string())?;
            }
            "--window" => {
                let ([w, h], _) = entelechy_session::parse_window(&value("--window")?)?;
                options.window = Some((w as u32, h as u32));
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if let Some(replay) = &options.replay {
        options
            .session
            .scene
            .get_or_insert_with(|| replay.scene.clone());
        if options.session.library.is_none() {
            options.session.library = replay.kernels.clone();
        }
        if options.window.is_none() {
            options.window = replay.steps.iter().find_map(|step| match step {
                entelechy_session::replay::Step::Window { size, scale } => {
                    Some(((size[0] / scale) as u32, (size[1] / scale) as u32))
                }
                _ => None,
            });
        }
    }
    options.session.scene()?;
    Ok(options)
}

fn listed(text: &str) -> Vec<String> {
    text.split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect()
}

struct DevHooks {
    requests: Option<Receiver<entelechy_protocol::Incoming>>,
    answers: entelechy_protocol::Answers,
    folder: Vec<String>,
    folder_allow: Vec<String>,
}

impl WindowHooks for DevHooks {
    fn initial_title(&self) -> &str {
        "Entelechy Player"
    }

    fn title(&self, session: &Session) -> String {
        let name = session
            .graph()
            .map(|graph| graph.name.clone())
            .unwrap_or_else(|| "no scene".into());
        let folder = session
            .chosen_folder()
            .and_then(|dir| dir.file_name())
            .map(|name| format!(" — {}", name.to_string_lossy()))
            .unwrap_or_default();
        format!("Entelechy Player: {name}{folder}")
    }

    fn poll_pending(runner: &mut WindowRunner<Self>, event_loop: &ActiveEventLoop) {
        let Some(requests) = &runner.hooks.requests else {
            return;
        };
        let pending: Vec<entelechy_protocol::Incoming> =
            std::iter::from_fn(|| requests.try_recv().ok()).collect();
        if !pending.is_empty() || runner.hooks.answers.is_waiting() {
            runner.confirm();
        }
        let mut answers = std::mem::take(&mut runner.hooks.answers);
        let quit = {
            let mut served = ServedRunner(runner);
            let mut q = answers.poll(&mut served);
            for incoming in pending {
                q |= answers.answer(&mut served, incoming);
            }
            q
        };
        runner.hooks.answers = answers;
        if quit {
            event_loop.exit();
        }
    }

    fn wants_folder(&self) -> bool {
        !self.folder.is_empty()
    }

    fn is_serving(&self) -> bool {
        self.requests.is_some()
    }

    fn choose_folder(&mut self, session: &mut Session) -> bool {
        let Some(dir) = rfd::FileDialog::new()
            .set_title("Choose a folder for the scene")
            .pick_folder()
        else {
            return false;
        };
        match session.choose_folder(&self.folder, &dir, &self.folder_allow) {
            Ok(()) => {
                eprintln!("folder: {}", dir.display());
                true
            }
            Err(why) => {
                eprintln!("folder {}: {why}", dir.display());
                false
            }
        }
    }
}

struct ServedRunner<'a>(&'a mut WindowRunner<DevHooks>);

impl entelechy_protocol::Served for ServedRunner<'_> {
    fn session(&mut self) -> &mut Session {
        &mut self.0.session
    }

    fn own(&mut self, incoming: &entelechy_protocol::Incoming) -> Option<(String, bool)> {
        let (id, request) = incoming.parsed.as_ref().ok()?;
        if request.get("op").and_then(Json::as_str) == Some("pause") {
            if let Err(error) = entelechy_protocol::ops::check(request) {
                return Some((entelechy_protocol::envelope(id, Err(error)), false));
            }
            if let Some(paused) = request.get("paused").and_then(Json::as_bool) {
                self.0.play.pause(paused);
            }
            if let Some(step) = request.get("step").and_then(Json::as_i64) {
                self.0.play.step(step.clamp(-100_000, 100_000) as i32);
            }
            self.0.say();
            let answer =
                serde_json::json!({ "paused": self.0.play.paused(), "time": self.0.play.time() });
            return Some((entelechy_protocol::envelope(id, Ok(answer)), false));
        }
        if request.get("op").and_then(Json::as_str) != Some("evaluate") {
            return None;
        }
        if entelechy_protocol::steps_asked(request) {
            let view = self.0.view();
            let window_ctx = self.0.play.context(view);
            let result =
                entelechy_protocol::step_for_window(&mut self.0.session, request, &window_ctx).map(
                    |(answer, advanced)| {
                        self.0.play.advance_clock(advanced);
                        let _ = self.0.frame();
                        answer
                    },
                );
            return Some((entelechy_protocol::envelope(id, result), false));
        }
        let measured = entelechy_protocol::measure_asked(request);
        self.0.session.set_measured(measured);
        let frame = self.0.frame();
        if measured {
            self.0.session.set_measured(false);
        }
        let ignored = entelechy_protocol::context_given(request);
        let result = match frame {
            Some(report) if self.0.confirm() => {
                let mut answer =
                    entelechy_protocol::evaluation_json(&self.0.session, &report, measured);
                if !ignored.is_empty() {
                    answer["ignored"] = serde_json::json!(ignored);
                    answer["note"] = serde_json::json!(
                        "the window's input and clock are the person's: the Player runs each evaluate under them and reads none of the fields named in `ignored`; `entelechy serve` takes them"
                    );
                }
                Ok(answer)
            }
            _ => Err(serde_json::json!({ "message": self.0.play.status() })),
        };
        Some((entelechy_protocol::envelope(id, result), false))
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let options = match parse(&args) {
        Ok(options) => options,
        Err(message) => {
            eprintln!("error: {message}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let backend = options.session.backend.as_deref().unwrap_or("best");
    let built = match Session::build(
        backend,
        options.session.limits,
        entelechy_gpu::Wait::blocking(),
    ) {
        Ok(built) => built,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };
    for passed in &built.passed_over {
        eprintln!("backend passed over: {passed}");
    }
    let (runtime, display) = (built.runtime, built.display);
    eprintln!("backend: {}", runtime.backend_name());
    let gamepad = Gamepad::open();
    eprintln!(
        "gamepad: {}",
        match &gamepad {
            Some(pad) if pad.connected() > 0 => "connected, mapped to arrows, Z, X, Enter",
            Some(_) => "none connected; a pad plugged in later maps to arrows, Z, X, Enter",
            None => "unavailable",
        }
    );
    let store = match &options.replay {
        Some(replay) => entelechy_session::replay::store_for(replay, options.session.store()),
        None => options.session.store(),
    };
    let opened = options
        .session
        .open_over_in(runtime, store)
        .and_then(|mut session| {
            if let Some(replay) = &options.replay {
                entelechy_session::replay::allow_on(replay, &mut session)?;
            }
            Ok(session)
        });
    let mut session = match opened {
        Ok(session) => session,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };
    session.give_clipboard(Box::new(SystemClipboard));
    if options.serve
        && let Err(error) = session.restrict_reads()
    {
        eprintln!("error: {error}");
        return ExitCode::FAILURE;
    }
    session.show_while_building(options.show_while_building && options.replay.is_none());
    let events = {
        let _watch = entelechy_session::watch::watch(session.progress());
        session.sync()
    };
    for event in events {
        eprintln!("{event}");
    }
    session.set_background(true);

    let mut requests = None;
    if options.serve {
        let (sender, receiver) = channel();
        if let Some(address) = &options.listen {
            let origins = entelechy_protocol::websocket::Origins::with(options.origins.clone());
            match entelechy_protocol::listen(address, sender.clone(), origins) {
                Ok(bound) => eprintln!("listening on {bound}"),
                Err(error) => {
                    eprintln!("error: cannot listen on {address}: {error}");
                    return ExitCode::FAILURE;
                }
            }
        }
        if let Err(error) = entelechy_protocol::stdin_into(sender) {
            eprintln!("error: cannot read standard input: {error}");
            return ExitCode::FAILURE;
        }
        requests = Some(receiver);
        eprintln!("serving the creation protocol on stdin/stdout");
    }

    let hooks = DevHooks {
        requests,
        answers: entelechy_protocol::Answers::new(),
        folder: options.folder,
        folder_allow: options.folder_allow,
    };
    let config = WindowConfig {
        scale: options.scale,
        requested: options.window,
        pointer_mode: options.pointer_mode,
        audio_allowed: !options.no_audio,
    };
    let mut runner = WindowRunner::new(session, Play::new(clock), display, gamepad, config, hooks);
    if let Some(replay) = &options.replay {
        runner.play.replay(replay, &mut runner.session);
        runner.say();
    }
    let context = runner.play.context(runner.view());
    let probed = (!runner.play.replaying()).then(|| runner.session.evaluate(&context));
    match probed {
        Some(Ok(report)) => {
            if let Ok((width, height)) = runner.session.runtime().host().image_size(report.output) {
                runner.last_size = (width, height);
            }
        }
        Some(Err(error)) => {
            let told = runner.session.told(&error);
            eprintln!("evaluation failed: {told}; the window opens when it succeeds");
            if let Some(place) = runner.session.failure_place(&error, &context) {
                eprintln!("  the failing operation is written at {place}");
            }
        }
        None => {}
    }
    runner.run()
}
