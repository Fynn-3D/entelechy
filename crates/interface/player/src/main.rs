//! The `entelechy-player` command: a window, sound card and pads over a live
//! session. Frames, key meanings and the frame line are
//! `entelechy_session::play`'s; the window loop is `entelechy_player::window`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use entelechy_kernel::Progress;
use entelechy_player::gamepad::Gamepad;
use entelechy_player::window::{
    PointerMode, SystemClipboard, WindowConfig, WindowHooks, WindowRunner, clock,
};
use entelechy_session::play::hud::{StageCard, loading_card};
use entelechy_session::play::{Play, View};
use entelechy_session::{Kept, Session, Store, Warm};
use serde_json::Value as Json;
use winit::event_loop::ActiveEventLoop;

const EMPTY_STAGE: &str = "\
node void: kernel {
    requires { field.write; }
    output result: Image;
    domain = vec2(1, 1);
    compute { return vec4(0.0, 0.0, 0.0, 1.0); }
}
output result <- void.result;
";

const USAGE: &str = "\
usage:
  entelechy-player [<scene.esn>] [--library <dir|file>] [--input <name>=<path>]...
              [--output <name>=<path>]... [--param <name>=<literal>]... [--scale <n>]
              [--folder <name,...> [--folder-allow <file.write,file.delete>]]
              [--replay <file.replay>] [--show-while-building]
              [--window <w>x<h>] [--backend gpu|cpu|jit|best] [--memory <MB>] [--work <M>]
              [--allow file.read|file.write|file.delete=<dir>]...
              [--allow clipboard.read|clipboard.write]... [--grant <capability,...>]
      Open a window showing the scene output once its Kernels have first
      prepared, showing Entelechy's Living Seed emblem and live build progress
      on the canvas while Kernels compile and build. Without <scene.esn> or
      --replay, open an empty resting stage; Cmd+O (Ctrl+O elsewhere, when
      --folder is not set) chooses a .esn scene or .replay file, and dropping
      a .esn, .replay, .ent or project folder onto the window switches to it
      in place over the same warm backend. Kernel files are republished as
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

enum Target {
    EmptyStage,
    Scene(PathBuf),
    Replay(entelechy_session::replay::Replay),
}

impl Target {
    fn display_title(&self) -> Option<String> {
        match self {
            Self::EmptyStage => None,
            Self::Scene(path) => {
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("scene");
                if matches!(stem, "scene" | "main")
                    && let Some(dir) = path.parent().and_then(|p| p.file_name())
                {
                    return Some(dir.to_string_lossy().into_owned());
                }
                Some(stem.to_owned())
            }
            Self::Replay(replay) => replay
                .scene
                .parent()
                .and_then(|p| p.file_name())
                .or_else(|| replay.scene.file_stem())
                .map(|s| s.to_string_lossy().into_owned()),
        }
    }
}

fn load_replay_file(path: &Path) -> Result<entelechy_session::replay::Replay, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let base = path.parent().unwrap_or(Path::new("."));
    entelechy_session::replay::parse(&text, base).map_err(|e| format!("{}: {e}", path.display()))
}

fn resolve_scene_in_dir(dir: &Path) -> Result<PathBuf, String> {
    for candidate in ["scene.esn", "main.esn"] {
        let path = dir.join(candidate);
        if path.is_file() {
            return Ok(path);
        }
    }
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut scenes: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("esn"))
        })
        .collect();
    scenes.sort();
    match scenes.len() {
        1 => Ok(scenes.remove(0)),
        0 => Err(format!("{}: no .esn scene found in folder", dir.display())),
        _ => Err(format!(
            "{}: multiple .esn files in folder; choose one directly",
            dir.display()
        )),
    }
}

fn resolve_path_target(path: &Path) -> Result<Target, String> {
    if path.is_dir() {
        return resolve_scene_in_dir(path).map(Target::Scene);
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default();
    if ext.eq_ignore_ascii_case("replay") {
        return load_replay_file(path).map(Target::Replay);
    }
    if ext.eq_ignore_ascii_case("esn") {
        return Ok(Target::Scene(path.to_path_buf()));
    }
    if ext.eq_ignore_ascii_case("ent") {
        let mut dir = path.parent();
        for _ in 0..3 {
            let Some(anc) = dir else { break };
            if anc
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".entpkg"))
            {
                break;
            }
            if let Ok(scene) = resolve_scene_in_dir(anc) {
                return Ok(Target::Scene(scene));
            }
            dir = anc.parent();
        }
    }
    Err(format!(
        "{}: expected a .esn scene, .replay file, or project folder",
        path.display()
    ))
}

fn has_entpkg(dir: &Path) -> bool {
    std::fs::read_dir(dir).ok().is_some_and(|entries| {
        entries.filter_map(Result::ok).any(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.ends_with(".entpkg"))
        })
    })
}

fn discover_packages(base: &[PathBuf], scene: Option<&Path>) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let mut seen = Vec::new();
    let mut push = |candidate: PathBuf| {
        let Ok(canon) = candidate.canonicalize() else {
            return;
        };
        if !seen.contains(&canon) {
            seen.push(canon);
            dirs.push(candidate);
        }
    };
    for dir in base {
        push(dir.clone());
    }
    if let Ok(exe) = std::env::current_exe() {
        for anc in exe.ancestors().skip(1).take(4) {
            let candidate = anc.join("packages");
            if has_entpkg(&candidate) {
                push(candidate);
            }
        }
    }
    if let Some(scene) = scene {
        let anchored = scene.canonicalize().unwrap_or_else(|_| scene.to_path_buf());
        for anc in anchored.ancestors().skip(1).take(6) {
            for rel in ["packages", "entelechy/packages"] {
                let candidate = anc.join(rel);
                if has_entpkg(&candidate) {
                    push(candidate);
                }
            }
        }
    }
    dirs
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
                let path = PathBuf::from(value("--replay")?);
                options.replay = Some(load_replay_file(&path)?);
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
    if options.replay.is_none()
        && let Some(given) = options.session.scene.clone()
        && (given.is_dir()
            || given
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("replay") || e.eq_ignore_ascii_case("ent")))
    {
        match resolve_path_target(&given)? {
            Target::Scene(scene) => options.session.scene = Some(scene),
            Target::Replay(replay) => {
                options.session.scene = None;
                options.replay = Some(replay);
            }
            Target::EmptyStage => {}
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
    Ok(options)
}

fn listed(text: &str) -> Vec<String> {
    text.split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect()
}

fn empty_stage_prompt() -> &'static str {
    if cfg!(target_os = "macos") {
        "cmd+o or drop a .esn / .replay"
    } else {
        "ctrl+o or drop a .esn / .replay"
    }
}

struct LoadedScene {
    session: Session,
    play: Play,
    from_cli: bool,
    initial_size: Option<(u32, u32)>,
    label: Option<PathBuf>,
}

struct PendingLoad {
    title: String,
    progress: Progress,
    fusing: Arc<AtomicBool>,
    go: Arc<AtomicBool>,
    aborted: Arc<AtomicBool>,
    receiver: Receiver<Result<LoadedScene, String>>,
}

impl Drop for PendingLoad {
    fn drop(&mut self) {
        self.aborted.store(true, Ordering::Release);
        self.go.store(true, Ordering::Release);
    }
}

struct StageEnv {
    warm: Warm,
    backend_name: &'static str,
    base_options: entelechy_session::SessionOptions,
    serve: bool,
    listen_bound: Option<String>,
    show_while_building: bool,
}

impl StageEnv {
    fn badge(&self) -> String {
        match &self.listen_bound {
            Some(bound) => format!("live · {bound}"),
            None => format!("native · {}", self.backend_name),
        }
    }

    fn empty_card(&self, error: Option<&str>) -> StageCard {
        StageCard {
            badge: self.badge(),
            title: None,
            progress: None,
            failed: error.is_some(),
            detail: error
                .map(str::to_owned)
                .unwrap_or_else(|| empty_stage_prompt().to_owned()),
        }
    }

    fn open_empty(&self, keep_hud: bool) -> Result<(Session, Play), String> {
        let mut session_options = self.base_options.clone();
        session_options.library = None;
        session_options.overrides = Default::default();
        session_options.params.clear();
        session_options.packages = discover_packages(&session_options.packages, None);
        let stage_path = PathBuf::from("__stage__/scene.esn");
        let kept = Arc::new(Kept::new(session_options.store()));
        kept.write(&stage_path, EMPTY_STAGE.as_bytes())
            .map_err(|e| e.to_string())?;
        session_options.scene = Some(stage_path);

        let mut session = session_options.open_over_in(self.warm.runtime(), kept)?;
        session.give_clipboard(Box::new(SystemClipboard));
        if self.serve {
            session.restrict_reads()?;
        }
        let _ = session.sync();
        session.set_background(true);

        let mut play = Play::new(clock);
        play.set_stage_card(Some(self.empty_card(None)));
        if keep_hud {
            play.set_hud(&mut session, true);
        }
        Ok((session, play))
    }

    /// Prepares a target scene or replay before syncing its Kernels, returning
    /// the unsynced `Session` (whose `progress()` tracks compilation) and any
    /// replay to start once synced.
    fn prepare_session(
        &self,
        target: Target,
        from_cli: bool,
    ) -> Result<(Session, Option<entelechy_session::replay::Replay>), String> {
        let mut session_options = self.base_options.clone();
        if !from_cli {
            session_options.library = None;
            session_options.overrides = Default::default();
            session_options.params.clear();
        }
        let mut replay_to_run = None;
        let store: Arc<dyn Store> = match target {
            Target::EmptyStage => unreachable!("empty stage uses open_empty"),
            Target::Scene(path) => {
                session_options.packages =
                    discover_packages(&session_options.packages, Some(&path));
                session_options.scene = Some(path);
                session_options.store()
            }
            Target::Replay(replay) => {
                if !from_cli || session_options.scene.is_none() {
                    session_options.scene = Some(replay.scene.clone());
                }
                if session_options.library.is_none() {
                    session_options.library = replay.kernels.clone();
                }
                session_options.packages =
                    discover_packages(&session_options.packages, session_options.scene.as_deref());
                let store = entelechy_session::replay::store_for(&replay, session_options.store());
                replay_to_run = Some(replay);
                store
            }
        };
        let mut session = session_options.open_over_in(self.warm.runtime(), store)?;
        if let Some(replay) = &replay_to_run {
            entelechy_session::replay::allow_on(replay, &mut session)?;
        }
        if self.serve {
            session.restrict_reads()?;
        }
        session.show_while_building(self.show_while_building && replay_to_run.is_none());
        Ok((session, replay_to_run))
    }

    fn spawn_load(
        &self,
        target: Target,
        from_cli: bool,
        keep_hud: bool,
        warm_view: View,
        label: Option<PathBuf>,
    ) -> Result<PendingLoad, String> {
        let title = target.display_title().unwrap_or_else(|| "scene".to_owned());
        let (mut session, replay_to_run) = self.prepare_session(target, from_cli)?;
        let progress = session.progress();
        let fusing = Arc::new(AtomicBool::new(false));
        let fusing_worker = Arc::clone(&fusing);
        let go = Arc::new(AtomicBool::new(false));
        let go_worker = Arc::clone(&go);
        let aborted = Arc::new(AtomicBool::new(false));
        let aborted_worker = Arc::clone(&aborted);
        let (sender, receiver) = channel();
        std::thread::Builder::new()
            .name("entelechy-load".into())
            .spawn(move || {
                let waited = Instant::now();
                while !go_worker.load(Ordering::Acquire)
                    && waited.elapsed() < Duration::from_millis(500)
                {
                    std::thread::sleep(Duration::from_millis(4));
                }
                if aborted_worker.load(Ordering::Acquire) {
                    return;
                }
                let events = {
                    let _watch = entelechy_session::watch::watch(session.progress());
                    session.sync()
                };
                for event in events {
                    eprintln!("{event}");
                }
                if aborted_worker.load(Ordering::Acquire) {
                    return;
                }

                // Pre-warm fused pointwise GPU pipelines and device input uploads
                // while the loading card is still shown, so swapping to the scene
                // never stalls on a blank screen when individual Kernels were
                // already cached in `Warm`.
                let mut initial_size = None;
                let mut play = Play::new(clock);
                if replay_to_run.is_none() && session.why_no_scene().is_none() {
                    fusing_worker.store(true, Ordering::Release);
                    let warm_ctx = play.context(warm_view);
                    let _ = session.set_overlap(true);
                    if let Ok(report) = session.evaluate(&warm_ctx)
                        && let Ok(size) = session.runtime().host().image_size(report.output)
                    {
                        initial_size = Some(size);
                    }
                    let _ = session.set_overlap(false);
                    session.reset_state();
                }
                if aborted_worker.load(Ordering::Acquire) {
                    return;
                }
                session.set_background(true);
                if keep_hud {
                    play.set_hud(&mut session, true);
                }
                if let Some(replay) = &replay_to_run {
                    play.replay(replay, &mut session);
                }
                let _ = sender.send(Ok(LoadedScene {
                    session,
                    play,
                    from_cli,
                    initial_size,
                    label,
                }));
            })
            .map_err(|e| format!("cannot start loader thread: {e}"))?;
        Ok(PendingLoad {
            title,
            progress,
            fusing,
            go,
            aborted,
            receiver,
        })
    }
}

struct DevHooks {
    stage: StageEnv,
    empty_stage: bool,
    loading: Option<PendingLoad>,
    requests: Option<Receiver<entelechy_protocol::Incoming>>,
    answers: entelechy_protocol::Answers,
    folder: Vec<String>,
    folder_allow: Vec<String>,
}

impl DevHooks {
    fn apply_loaded(runner: &mut WindowRunner<Self>, mut loaded: LoadedScene) {
        loaded.session.give_clipboard(Box::new(SystemClipboard));
        if let Some(path) = &loaded.label {
            loaded.play.say(format!("opened {}", path.display()));
        }
        runner.session = loaded.session;
        runner.play = loaded.play;
        runner.hooks.empty_stage = false;
        runner.hooks.loading = None;
        runner.hooks.answers = entelechy_protocol::Answers::new();

        runner.after_scene_switch();
        if let Some((width, height)) = loaded.initial_size {
            runner.last_size = (width, height);
            if loaded.from_cli {
                runner.fit_initial_output(width, height);
            }
        }
    }

    fn switch_to(
        &mut self,
        path: &Path,
        session: &mut Session,
        play: &mut Play,
        view: View,
    ) -> bool {
        let target = match resolve_path_target(path) {
            Ok(target) => target,
            Err(why) => {
                let message = format!("open failed: {why}");
                if self.empty_stage {
                    play.set_stage_card(Some(self.stage.empty_card(Some(&message))));
                }
                play.say(message);
                return false;
            }
        };
        let keep_hud = play.hud();
        let _ = play.confirm(session);
        let pending =
            match self
                .stage
                .spawn_load(target, false, keep_hud, view, Some(path.to_path_buf()))
            {
                Ok(pending) => pending,
                Err(why) => {
                    let message = format!("open {}: {why}", path.display());
                    if self.empty_stage {
                        play.set_stage_card(Some(self.stage.empty_card(Some(&message))));
                    }
                    play.say(message);
                    return false;
                }
            };

        if let Ok((stage_session, mut stage_play)) = self.stage.open_empty(keep_hud) {
            let badge = self.stage.badge();
            let card = loading_card(&badge, &pending.title, &pending.progress);
            stage_play.set_stage_card(Some(card));
            *session = stage_session;
            *play = stage_play;
        }
        self.empty_stage = false;
        self.loading = Some(pending);
        true
    }
}

impl WindowHooks for DevHooks {
    fn initial_title(&self) -> &str {
        "Entelechy Player"
    }

    fn title(&self, session: &Session) -> String {
        let listen = self
            .stage
            .listen_bound
            .as_deref()
            .map(|bound| format!(" — {bound}"))
            .unwrap_or_default();
        if let Some(loading) = &self.loading {
            return format!("Entelechy Player: {}{listen}", loading.title);
        }
        if self.empty_stage {
            return format!("Entelechy Player{listen}");
        }
        let name = session
            .graph()
            .map(|graph| graph.name.clone())
            .unwrap_or_else(|| "no scene".into());
        let folder = session
            .chosen_folder()
            .and_then(|dir| dir.file_name())
            .map(|name| format!(" — {}", name.to_string_lossy()))
            .unwrap_or_default();
        format!("Entelechy Player: {name}{folder}{listen}")
    }

    fn poll_pending(runner: &mut WindowRunner<Self>, event_loop: &ActiveEventLoop) {
        if let Some(loading) = &runner.hooks.loading {
            if runner.loading_presented {
                loading.go.store(true, Ordering::Release);
            }
            match loading.receiver.try_recv() {
                Ok(Ok(loaded)) => {
                    Self::apply_loaded(runner, loaded);
                }
                Ok(Err(why)) => {
                    runner.hooks.empty_stage = true;
                    runner.hooks.loading = None;
                    let card = runner.hooks.stage.empty_card(Some(&why));
                    runner.play.set_stage_card(Some(card));
                    runner.play.say(why);
                    runner.after_scene_switch();
                    return;
                }
                Err(_) => {
                    let badge = runner.hooks.stage.badge();
                    let card = if loading.fusing.load(Ordering::Acquire) {
                        StageCard {
                            badge,
                            title: Some(loading.title.clone()),
                            progress: Some(0.95),
                            failed: false,
                            detail: "building fused pipelines...".to_owned(),
                        }
                    } else {
                        loading_card(&badge, &loading.title, &loading.progress)
                    };
                    runner.play.set_stage_card(Some(card));
                    return;
                }
            }
        }

        let Some(requests) = &runner.hooks.requests else {
            return;
        };
        let pending: Vec<entelechy_protocol::Incoming> =
            std::iter::from_fn(|| requests.try_recv().ok()).collect();
        if !pending.is_empty() || runner.hooks.answers.is_waiting() {
            runner.confirm();
        }
        let composed = pending.iter().any(|incoming| {
            incoming
                .parsed
                .as_ref()
                .ok()
                .and_then(|(_, req)| req.get("op"))
                .and_then(Json::as_str)
                == Some("compose")
        });
        let had_pending = !pending.is_empty();
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
        if composed && runner.hooks.empty_stage && runner.session.graph().is_some() {
            runner.hooks.empty_stage = false;
            runner.play.set_stage_card(None);
            runner.update_title();
        }
        if had_pending {
            runner.play.wake();
        }
        if quit {
            event_loop.exit();
        }
    }

    fn wants_folder(&self) -> bool {
        !self.folder.is_empty()
    }

    fn wants_open_shortcut(&self) -> bool {
        self.folder.is_empty()
    }

    fn is_serving(&self) -> bool {
        self.requests.is_some()
    }

    fn is_loading(&self) -> bool {
        self.loading.is_some()
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

    fn open_shortcut(&mut self, session: &mut Session, play: &mut Play, view: View) -> bool {
        let Some(file) = rfd::FileDialog::new()
            .set_title("Open an Entelechy scene or replay")
            .add_filter("Entelechy (.esn, .replay, .ent)", &["esn", "replay", "ent"])
            .pick_file()
        else {
            return false;
        };
        self.switch_to(&file, session, play, view)
    }

    fn drop_path(
        &mut self,
        path: PathBuf,
        session: &mut Session,
        play: &mut Play,
        view: View,
    ) -> bool {
        self.switch_to(&path, session, play, view)
    }
}

struct ServedRunner<'a>(&'a mut WindowRunner<DevHooks>);

impl entelechy_protocol::Served for ServedRunner<'_> {
    fn session(&mut self) -> &mut Session {
        let view = self.0.view();
        self.0.session.set_window(Some((view.window, view.scale)));
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

    let mut requests = None;
    let mut listen_bound = None;
    if options.serve {
        let (sender, receiver) = channel();
        if let Some(address) = &options.listen {
            let origins = entelechy_protocol::websocket::Origins::with(options.origins.clone());
            match entelechy_protocol::listen(address, sender.clone(), origins) {
                Ok(bound) => {
                    eprintln!("listening on {bound}");
                    listen_bound = Some(bound.to_string());
                }
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

    let backend = options.session.backend.as_deref().unwrap_or("best");
    let (warm, display, passed_over) = match Warm::open_for_window(
        backend,
        options.session.limits,
        entelechy_gpu::Wait::blocking(),
    ) {
        Ok(opened) => opened,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };
    for passed in &passed_over {
        eprintln!("backend passed over: {passed}");
    }
    let backend_name = warm.runtime().backend_name();
    eprintln!("backend: {backend_name}");
    let gamepad = Gamepad::open();
    eprintln!(
        "gamepad: {}",
        match &gamepad {
            Some(pad) if pad.connected() > 0 => "connected, mapped to arrows, Z, X, Enter",
            Some(_) => "none connected; a pad plugged in later maps to arrows, Z, X, Enter",
            None => "unavailable",
        }
    );

    let initial_target = match (&options.replay, &options.session.scene) {
        (Some(replay), _) => Target::Replay(replay.clone()),
        (None, Some(scene)) => Target::Scene(scene.clone()),
        (None, None) => Target::EmptyStage,
    };
    let stage = StageEnv {
        warm,
        backend_name,
        base_options: options.session,
        serve: options.serve,
        listen_bound,
        show_while_building: options.show_while_building,
    };

    let (session, mut play) = match stage.open_empty(false) {
        Ok(opened) => opened,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };
    let mut empty_stage = true;
    let mut loading = None;

    if !matches!(initial_target, Target::EmptyStage) {
        let (w, h) = options
            .window
            .unwrap_or(entelechy_player::window::DEFAULT_WINDOW);
        let initial_view = View {
            window: [w as f32, h as f32],
            scale: 1.0,
            rate: 0.0,
            queued: 0,
        };
        let pending = match stage.spawn_load(initial_target, true, false, initial_view, None) {
            Ok(pending) => pending,
            Err(error) => {
                eprintln!("error: {error}");
                return ExitCode::FAILURE;
            }
        };
        let card = loading_card(&stage.badge(), &pending.title, &pending.progress);
        play.set_stage_card(Some(card));
        empty_stage = false;
        loading = Some(pending);
    }

    let hooks = DevHooks {
        stage,
        empty_stage,
        loading,
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
    let mut runner = WindowRunner::new(session, play, display, gamepad, config, hooks);
    runner.say();
    runner.run()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_zero_argument_empty_stage() {
        let options = parse(&[]).expect("zero args parse cleanly");
        assert!(options.session.scene.is_none());
        assert!(options.replay.is_none());
    }

    #[test]
    fn empty_stage_renders_centered_logo_card_and_rests() {
        let (warm, _, _) = Warm::open_for_window(
            "jit",
            entelechy_session::Limits::default(),
            entelechy_gpu::Wait::blocking(),
        )
        .unwrap();
        let stage = StageEnv {
            warm,
            backend_name: "jit",
            base_options: entelechy_session::SessionOptions::default(),
            serve: true,
            listen_bound: Some("127.0.0.1:7788".into()),
            show_while_building: false,
        };
        let (mut session, mut play) = stage.open_empty(false).unwrap();
        let view = entelechy_session::play::View {
            window: [960.0, 540.0],
            scale: 2.0,
            rate: 0.0,
            queued: 0,
        };
        assert!(matches!(
            play.frame(&mut session, view),
            entelechy_session::play::Frame::Wait
        ));
        let panel = play.panel().expect("centered Entelechy card is rendered");
        assert!(panel.centered);
        assert!(
            play.resting(),
            "empty stage settles to 0 Hz resting quiescence immediately"
        );
        assert!(
            session.why_no_scene().is_none() && session.sync_lines().is_empty(),
            "restricted reads keep the virtual stage directory readable across syncs"
        );
    }

    #[test]
    fn loading_card_lights_tiers_as_progress_advances() {
        let progress = Progress::default();
        let card0 = loading_card("native · gpu", "jukebox", &progress);
        assert_eq!(card0.progress, Some(0.08));
        let rendered = entelechy_session::play::hud::stage_card(&card0, 2);
        assert!(rendered.centered);
        assert_eq!(rendered.width, 348 * 2);
        assert!(
            rendered.pixels.contains(&0x00FF_B000),
            "Solar Seed rendered"
        );
    }

    #[test]
    fn discovers_packages_without_environment_variable() {
        let example = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../examples/text/scene.esn")
            .canonicalize()
            .unwrap();
        let found = discover_packages(&[], Some(&example));
        assert!(
            found.iter().any(|dir| dir.join("text.entpkg").is_dir()),
            "auto-discovers repo packages without ENTELECHY_PACKAGES: {found:?}"
        );

        let kernel = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../examples/text/kernels")
            .read_dir()
            .ok()
            .and_then(|mut d| d.find_map(Result::ok))
            .map(|e| e.path());
        if let Some(ent) = kernel {
            assert!(matches!(resolve_path_target(&ent), Ok(Target::Scene(_))));
        }
    }
}
