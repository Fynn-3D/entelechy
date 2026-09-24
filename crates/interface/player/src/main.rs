//! The `orla-player` command: a window over a live session.

use std::num::NonZeroU32;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use orla_gpu::present::{Display, Overlay, Presenter};
use orla_player::audio::Audio;
use orla_player::gamepad::Gamepad;
use orla_player::hud::{self, Panel, Stats};
use orla_player::{Placement, key_code, to_output};
use orla_runtime::{Context, InputState};
use orla_session::Session;
use serde_json::Value as Json;
use std::sync::mpsc::{Receiver, channel};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const USAGE: &str = "\
usage:
  orla-player <scene.osn> [--library <dir|file>] [--input <name>=<file.ppm>]... [--scale <n>]
              [--replay <file.replay>]
              [--window <w>x<h>] [--backend gpu|cpu|jit] [--memory <MB>] [--work <M>]
              [--allow file.read|file.write=<dir>]... [--grant <capability,...>]
      Open a window showing the scene output. Kernel files are republished as
      they change; a rejected change keeps the running version. The pointer
      and keys reach Kernels that declare `input.read`, and so do the
      window's size in pixels and its scale factor, which a node's `domain`
      may take as `window()`; --window opens the window at that logical size
      instead of the output's size times --scale. F5 resets state;
      Tab cycles the preview through the graph's nodes and back to the output;
      F10 shows a line of frame data over the picture.
      --memory says how many mebibytes of Fields the Host may hold at once,
      2048 by default; a scene of millions of triangles wants more.
      --work says how many million loop iterations one run of a Kernel may
      take, all its elements together, 4000 by default.
      --allow file.write=<dir> lets the scene's outputs written `to` a file
      land in that directory; nothing is written without it.
      --allow file.read=<dir> lets a served scene's inputs read files
      there; with --serve a scene reads only beside its documents without
      it.
      --replay <file.replay> plays a replay in the window: its pointer,
      keys and window drive the scene at the replay's own times, its
      checks run as they come, and when it ends what it observed is
      printed and the window is yours. The replay names its scene, so
      `orla-player --replay <file.replay>` is enough.
      Press Escape or close the window to quit.
      With --serve, the creation protocol runs on stdin and stdout inside the
      window, and on every connection to --listen <host:port>: an agent's
      publish, compose, and param land in the frame a person watches, and
      its evaluate observes that person's input. A browser's page connects
      from this machine or from a file; a page elsewhere only from an
      origin --allow-origin <origin> names.";

/// What a replay says for the frame about to be drawn.
enum Replayed {
    /// Its next frame's time has not come.
    Wait,
    /// Its next frame, under this context.
    Frame(Context),
    /// No replay is playing.
    Not,
}

/// The window `window()` reads before one opens, when none was asked for.
const DEFAULT_WINDOW: (u32, u32) = (960, 540);

struct Options {
    /// The session: scene, library, inputs, params, backend, limits and
    /// allowances.
    session: orla_session::SessionOptions,
    scale: u32,
    /// The window's logical size, when given; else the output's times `scale`.
    window: Option<(u32, u32)>,
    serve: bool,
    listen: Option<String>,
    origins: Vec<String>,
    /// Leave the sound card closed even when the scene has a sound output.
    no_audio: bool,
    /// A replay the window plays: its inputs drive the scene, frame by
    /// frame at its own times, until it ends and the person takes over.
    replay: Option<orla_session::replay::Replay>,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut options = Options {
        session: orla_session::SessionOptions::default(),
        scale: 2,
        window: None,
        serve: false,
        listen: None,
        origins: Vec::new(),
        no_audio: false,
        replay: None,
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
            "--replay" => {
                let path = std::path::PathBuf::from(value("--replay")?);
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                let base = path.parent().unwrap_or(std::path::Path::new("."));
                let replay = orla_session::replay::parse(&text, base)
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
                let ([w, h], _) = orla_session::parse_window(&value("--window")?)?;
                options.window = Some((w as u32, h as u32));
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    // A replay names its scene, its library and its window; what the
    // command line says first stands.
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
                orla_session::replay::Step::Window { size, scale } => {
                    Some(((size[0] / scale) as u32, (size[1] / scale) as u32))
                }
                _ => None,
            });
        }
    }
    options.session.scene()?;
    Ok(options)
}

struct Player {
    session: Session,
    scale: u32,
    /// The logical size the window opens at, when asked for; also what
    /// `window()` reads before the window exists.
    requested: Option<(u32, u32)>,
    clock: Instant,
    window: Option<Arc<Window>>,
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
    /// The device the backend computes on, when it has one to show: a
    /// window presents what it left resident there without a readback.
    display: Option<Display>,
    /// The GPU presenter, when the backend is the GPU and the surface could
    /// be configured; then frames are shown without any readback.
    presenter: Option<Presenter>,
    downloads: u64,
    last_frame: Vec<[f32; 4]>,
    last_size: (u32, u32),
    placement: Placement,
    input: InputState,
    cursor: (f32, f32),
    status: String,
    /// What a run of failures has already said, so a hand on the
    /// window's corner does not fill the terminal.
    reported: orla_player::Reporting,
    /// How the last frame reached the window, for the F10 line.
    path: &'static str,
    /// Protocol requests arriving from stdin or a socket, when serving.
    requests: Option<Receiver<orla_protocol::Incoming>>,
    /// The node whose image is shown instead of the graph output.
    preview: Option<orla_graph::NodeId>,
    /// The sound card, opened when a scene has a sound-shaped output.
    audio: Option<Audio>,
    /// The gamepads, when the system offers them.
    gamepad: Option<Gamepad>,
    audio_allowed: bool,
    audio_failed: bool,
    /// The F10 line: on or off, what it measures, and its pixels for this
    /// frame when on.
    hud: bool,
    stats: Stats,
    panel: Option<Panel>,
    /// A replay being played, and when its first frame was shown.
    replaying: Option<(orla_session::replay::Replayer, Option<Instant>)>,
}

impl Player {
    fn context(&self) -> Context {
        // The window's physical size and scale factor; before it opens, the
        // requested size at scale one, or a plain default, so a scene sized
        // by the window has something to fill on its first evaluation.
        let (window, scale) = match &self.window {
            Some(window) => {
                let size = window.inner_size();
                (
                    [size.width as f32, size.height as f32],
                    window.scale_factor() as f32,
                )
            }
            None => {
                let (w, h) = self.requested.unwrap_or(DEFAULT_WINDOW);
                ([w as f32, h as f32], 1.0)
            }
        };
        Context {
            time: self.clock.elapsed().as_secs_f32(),
            rate: self.audio.as_ref().map_or(0.0, |audio| audio.rate() as f32),
            input: self.input,
            window,
            scale,
        }
    }

    /// Whether the sound device is still playing the last block: then the
    /// device sets the pace and the next evaluation waits for it.
    fn device_is_ahead(&self) -> bool {
        self.audio.as_ref().is_some_and(|audio| !audio.has_room())
    }

    fn frame(&mut self) -> Option<orla_graph::Report> {
        if let Some(gamepad) = &mut self.gamepad {
            gamepad.poll(&mut self.input);
        }
        let events = self.session.sync();
        for event in &events {
            eprintln!("{event}");
        }
        // A replay, while it plays, is the input: its frame, at its time,
        // under its pointer, keys and window; the person's hands wait.
        let replayed = match self.replay_frame() {
            Replayed::Wait => return None,
            Replayed::Frame(context) => Some(context),
            Replayed::Not => None,
        };
        self.stats.begin();
        let started = Instant::now();
        let context = replayed.unwrap_or_else(|| self.context());
        self.input.wheel = [0.0, 0.0];
        match self.session.evaluate(&context) {
            Ok(report) => {
                if replayed.is_some()
                    && let Some((replayer, _)) = &mut self.replaying
                {
                    replayer.evaluated(report.clone());
                }
                let elapsed = started.elapsed();
                // The window claims the first output that is a grid of
                // pixels; the sound card the first that is rows of one or
                // two floats. Both go by structure, never by name.
                let displayed = self.displayed_output(&report);
                let target = self
                    .preview
                    .as_ref()
                    .and_then(|node| self.session.cached(node, "result"))
                    .unwrap_or(displayed);
                self.panel = self.hud.then(|| self.panel_for(&report, target, &context));
                // With the line on, the frame waits here for the
                // device to finish what evaluating gave it, so what it
                // took is a number of its own rather than a share of
                // `present`, where it would look like the display's
                // refresh. Asking costs the overlap between this
                // frame's device work and the next frame's own; only
                // the line asks.
                let device = (self.hud && self.session.runtime().has_device()).then(|| {
                    let waiting = Instant::now();
                    self.session.runtime().settle_device();
                    waiting.elapsed()
                });
                let presenting = Instant::now();
                self.path = self.show(target);
                self.stats.record(elapsed, device, presenting.elapsed());
                self.play(&report);
                if self.reported.worked() {
                    eprintln!("evaluating again");
                }
                self.status = "running".into();
                Some(report)
            }
            Err(error) => {
                let message = format!("evaluation failed: {error}");
                // One failure a frame, each naming the size the window
                // was that frame, is one failure: a hand on the corner
                // should not fill the terminal. What is printed again is
                // a failure of another shape, or this one after a frame
                // that worked.
                if self.reported.failed(&message) {
                    eprintln!("{message}; last frame is kept");
                }
                self.status = message;
                self.blit();
                None
            }
        }
    }

    /// The replay's next frame, once its time has come; `Wait` before,
    /// and `Not` when no replay is playing. Past its last frame the
    /// replay's remaining checks run, what it observed is said, and the
    /// person has the window.
    fn replay_frame(&mut self) -> Replayed {
        let Some((replayer, since)) = &mut self.replaying else {
            return Replayed::Not;
        };
        let clock = since
            .get_or_insert_with(Instant::now)
            .elapsed()
            .as_secs_f32();
        if replayer.next_time().is_some_and(|time| time > clock) {
            return Replayed::Wait;
        }
        match replayer.next_frame(&mut self.session) {
            Ok(Some(frame)) => {
                self.status = format!("replaying, {:.2} s", frame.context.time);
                Replayed::Frame(frame.context)
            }
            Ok(None) => {
                let outcome = &replayer.outcome;
                let said = if outcome.failures.is_empty() {
                    format!(
                        "replay: ok ({} frames, {} checks)",
                        outcome.frames, outcome.checks
                    )
                } else {
                    format!(
                        "replay: {} of {} checks failed\n  {}",
                        outcome.failures.len(),
                        outcome.checks,
                        outcome.failures.join("\n  ")
                    )
                };
                eprintln!("{said}; the window is yours");
                self.replaying = None;
                Replayed::Not
            }
            Err(error) => {
                eprintln!("replay stopped: {error}; the window is yours");
                self.replaying = None;
                Replayed::Not
            }
        }
    }

    /// Answers protocol requests that arrived since the last frame. An
    /// `evaluate` runs one window frame under the person's input and
    /// reports it; everything else goes through the shared handler.
    fn serve_pending(&mut self, event_loop: &ActiveEventLoop) {
        let Some(requests) = &self.requests else {
            return;
        };
        let pending: Vec<orla_protocol::Incoming> =
            std::iter::from_fn(|| requests.try_recv().ok()).collect();
        for incoming in pending {
            let (response, quit) = match orla_protocol::parse_request(&incoming.line) {
                Ok((id, request))
                    if request.get("op").and_then(Json::as_str) == Some("evaluate") =>
                {
                    let result = match self.frame() {
                        Some(report) => Ok(orla_protocol::report_json(&self.session, &report)),
                        None => Err(serde_json::json!({ "message": self.status })),
                    };
                    (orla_protocol::envelope(&id, result), false)
                }
                _ => orla_protocol::respond_line(&mut self.session, &incoming.line),
            };
            incoming.reply.send(response);
            // A quit on standard input closes the window; on a socket it
            // closes that connection, which an empty line tells its writer.
            if quit && !incoming.local {
                incoming.reply.send(String::new());
            }
            if quit && incoming.local {
                event_loop.exit();
            }
        }
    }

    /// Moves the preview to the next node, or back to the graph output
    /// after the last one. Previewed nodes are pinned so fusion keeps them
    /// materialized.
    fn cycle_preview(&mut self) {
        let nodes: Vec<orla_graph::NodeId> = self
            .session
            .graph()
            .map(|graph| graph.nodes.iter().map(|node| node.id.clone()).collect())
            .unwrap_or_default();
        if let Some(current) = self.preview.take() {
            self.session.unpin(&current);
            let position = nodes.iter().position(|node| *node == current);
            if let Some(next) = position.and_then(|p| nodes.get(p + 1)) {
                self.preview = Some(next.clone());
            }
        } else {
            self.preview = nodes.first().cloned();
        }
        if let Some(node) = &self.preview {
            self.session.pin(node.clone());
        }
        let view = self
            .preview
            .as_ref()
            .map_or("output".to_string(), ToString::to_string);
        eprintln!("previewing {view}");
    }

    /// The F10 line for this frame: the smoothed times, then the backend,
    /// the window, the shown image, what ran, and what waits off the frame.
    fn panel_for(
        &self,
        report: &orla_graph::Report,
        shown: orla_runtime::Handle,
        context: &Context,
    ) -> Panel {
        let host = self.session.runtime().host();
        let (width, height) = host.image_size(shown).unwrap_or((0, 0));
        let fused: usize = report
            .fused
            .iter()
            .map(|(_, absorbed)| absorbed.len())
            .sum();
        let view = match &self.preview {
            Some(node) => format!("view {node}"),
            None => "output".to_string(),
        };
        let sound = match &self.audio {
            Some(audio) => format!(" | sound {}", audio.describe()),
            None => String::new(),
        };
        // The nodes that left the main backend this frame, and what the
        // trip carried: what a device session pays that a profile would
        // otherwise have to be asked for.
        let routed: Vec<&orla_graph::Placed> = report
            .placements
            .iter()
            .filter(|placed| placed.routed.is_some())
            .collect();
        let off = if routed.is_empty() {
            String::new()
        } else {
            let carried: u64 = routed.iter().map(|p| p.uploaded + p.downloaded).sum();
            let carried = carried
                + report
                    .placements
                    .iter()
                    .filter(|placed| placed.routed.is_none())
                    .map(|p| p.uploaded)
                    .sum::<u64>();
            format!(
                " | off {} {} ({})",
                routed[0].backend,
                routed.len(),
                orla_session::bench::bytes_text(carried)
            )
        };
        let rest = format!(
            "{}{off} | window {}x{} x{:.0} | {view} {width}x{height} | ran {} reused {} fused {fused} | async {} | {} | downloads {}{sound} | t {:.1} s",
            self.session.runtime().backend_name(),
            context.window[0],
            context.window[1],
            context.scale,
            report.executed.len(),
            report.reused.len(),
            report.pending.len(),
            self.path,
            self.downloads,
            context.time,
        );
        hud::panel(&self.stats.line(&rest), context.scale.round() as u32)
    }

    /// Shows an output image: directly from the device when it is resident
    /// and a presenter exists, otherwise by downloading it and blitting.
    /// Returns a short description of the path taken.
    fn show(&mut self, output: orla_runtime::Handle) -> &'static str {
        if let (Some(window), Some(presenter)) = (&self.window, &mut self.presenter) {
            let host = self.session.runtime().host();
            let resident = host.resident(output).ok().flatten();
            // Only a grid of float pixels is drawn by the device as it lies;
            // a picture kept narrow, or anything else, downloads (widened)
            // and is drawn or refused below with a clear message.
            let presentable = host
                .schema(output)
                .is_ok_and(|s| s.is_pixel() && s.is_wide())
                && host.domain(output).is_ok_and(|d| d.depth() == 1);
            if let (true, Some(device), Ok((width, height))) =
                (presentable, resident, host.image_size(output))
            {
                let size = window.inner_size();
                if let Some(display) = &self.display {
                    let overlay = self.panel.as_ref().map(|panel| Overlay {
                        pixels: &panel.pixels,
                        width: panel.width,
                        height: panel.height,
                    });
                    match display.present(
                        presenter,
                        device,
                        width,
                        height,
                        size.width,
                        size.height,
                        overlay,
                    ) {
                        Ok(Some(placement)) => {
                            self.last_size = (width, height);
                            self.placement = placement;
                            return "presented on the device";
                        }
                        Ok(None) => {
                            self.last_size = (width, height);
                            return "window not ready";
                        }
                        Err(error) => eprintln!("present failed: {error}; downloading instead"),
                    }
                }
            }
        }
        if let Ok(image) = self.session.runtime_mut().image(output) {
            self.downloads += 1;
            self.last_size = (image.width(), image.height());
            self.last_frame.clear();
            self.last_frame.extend_from_slice(image.pixels());
        }
        self.blit();
        "downloaded and blitted"
    }

    /// The output the window shows: the first grid of pixels, or the
    /// graph's first output when there is none.
    fn displayed_output(&self, report: &orla_graph::Report) -> orla_runtime::Handle {
        let host = self.session.runtime().host();
        report
            .outputs
            .iter()
            .map(|(_, handle)| *handle)
            .find(|handle| {
                host.schema(*handle).is_ok_and(|s| s.is_pixel())
                    && host.domain(*handle).is_ok_and(|d| d.depth() == 1)
            })
            .unwrap_or(report.output)
    }

    /// Hands the first sound-shaped output to the sound card: rows of one
    /// float (mono) or two (stereo). Opens the device on first need.
    fn play(&mut self, report: &orla_graph::Report) {
        if !self.audio_allowed || self.audio_failed {
            return;
        }
        let sound = {
            let host = self.session.runtime().host();
            report
                .outputs
                .iter()
                .map(|(_, handle)| *handle)
                .find(|handle| {
                    host.domain(*handle)
                        .is_ok_and(|d| d.shape() == orla_runtime::Shape::Rows)
                        && host.schema(*handle).is_ok_and(|s| {
                            (1..=2).contains(&s.value_words())
                                && s.scalars()
                                    .iter()
                                    .all(|c| *c == orla_runtime::Scalar::Float)
                        })
                })
        };
        let Some(handle) = sound else {
            return;
        };
        let (frames, words_per_frame) = {
            let host = self.session.runtime().host();
            (
                host.domain(handle).map(|d| d.count()).unwrap_or(0),
                host.schema(handle).map(|s| s.value_words()).unwrap_or(0),
            )
        };
        if self.audio.is_none() {
            match Audio::open(frames) {
                Ok(audio) => {
                    eprintln!("audio: {}", audio.describe());
                    self.audio = Some(audio);
                }
                Err(error) => {
                    eprintln!("audio unavailable: {error}");
                    self.audio_failed = true;
                    return;
                }
            }
        }
        // Materializing is the observation boundary: on the GPU this is
        // one small download per frame, on the frame thread, never in the
        // device callback.
        if self.session.runtime_mut().materialize(handle).is_err() {
            return;
        }
        let words: Vec<u32> = match self.session.runtime().host().data(handle) {
            Ok(data) => data.values().to_vec(),
            Err(_) => return,
        };
        if let Some(audio) = &mut self.audio {
            audio.push(&words, words_per_frame);
        }
    }

    fn update_title(&self) {
        if let Some(window) = &self.window {
            let name = self
                .session
                .graph()
                .map(|graph| graph.name.clone())
                .unwrap_or_else(|| "no scene".into());
            window.set_title(&format!("Orla Player: {name}"));
        }
    }

    fn blit(&mut self) {
        if self.presenter.is_some() && self.last_frame.is_empty() {
            return;
        }
        let (Some(window), Some(surface)) = (&self.window, &mut self.surface) else {
            return;
        };
        let size = window.inner_size();
        let (Some(width), Some(height)) =
            (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
        else {
            return;
        };
        if surface.resize(width, height).is_err() {
            return;
        }
        let Ok(mut buffer) = surface.buffer_mut() else {
            return;
        };
        self.placement = orla_player::present(
            &self.last_frame,
            self.last_size.0,
            self.last_size.1,
            size.width,
            size.height,
            &mut buffer,
        );
        if let Some(panel) = &self.panel {
            hud::stamp(panel, &mut buffer, size.width, size.height);
        }
        let _ = buffer.present();
    }

    fn update_pointer(&mut self) {
        self.input.pointer = to_output(
            &self.placement,
            self.cursor.0,
            self.cursor.1,
            self.last_size.0,
            self.last_size.1,
        );
    }
}

impl ApplicationHandler for Player {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let (width, height) = if self.last_size == (0, 0) {
            (256, 256)
        } else {
            self.last_size
        };
        let (logical_width, logical_height) = self
            .requested
            .unwrap_or((width * self.scale, height * self.scale));
        let attributes = Window::default_attributes()
            .with_title("Orla Player")
            .with_inner_size(LogicalSize::new(logical_width, logical_height));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                eprintln!("cannot create a window: {error}");
                event_loop.exit();
                return;
            }
        };
        let size = window.inner_size();
        if let Some(display) = &self.display {
            match display.presenter(window.clone(), size.width, size.height) {
                Ok(presenter) => {
                    self.presenter = Some(presenter);
                    eprintln!("presenting on the device");
                }
                Err(error) => eprintln!("device presentation unavailable ({error}); blitting"),
            }
        }
        if self.presenter.is_none() {
            let context = softbuffer::Context::new(window.clone());
            let surface =
                context.and_then(|context| softbuffer::Surface::new(&context, window.clone()));
            match surface {
                Ok(surface) => self.surface = Some(surface),
                Err(error) => {
                    eprintln!("cannot present to the window: {error}");
                    event_loop.exit();
                    return;
                }
            }
        }
        self.window = Some(window);
        self.update_title();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = (position.x as f32, position.y as f32);
                self.update_pointer();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let held = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => self.input.pointer_down = held,
                    MouseButton::Right => self.input.pointer_right = held,
                    MouseButton::Middle => self.input.pointer_middle = held,
                    _ => {}
                }
                self.update_pointer();
            }
            // The wheel accumulates until the next evaluation reads it.
            WindowEvent::MouseWheel { delta, .. } => {
                let (x, y) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (x, y),
                    MouseScrollDelta::PixelDelta(p) => (p.x as f32 / 40.0, p.y as f32 / 40.0),
                };
                self.input.wheel[0] += x;
                self.input.wheel[1] += y;
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state,
                        repeat: false,
                        ..
                    },
                ..
            } => {
                let down = state == ElementState::Pressed;
                match logical_key {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::F5) => {
                        if down {
                            self.session.reset_state();
                            eprintln!("state reset");
                        }
                    }
                    Key::Named(NamedKey::Tab) => {
                        if down {
                            self.cycle_preview();
                        }
                    }
                    Key::Named(NamedKey::F10) => {
                        if down {
                            self.hud = !self.hud;
                            self.panel = None;
                        }
                    }
                    Key::Named(named) => {
                        if let Some(code) = key_code(&format!("{named:?}")) {
                            self.input.set_key(code, down);
                        }
                    }
                    Key::Character(text) => {
                        if let Some(code) = key_code(&text) {
                            self.input.set_key(code, down);
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => {
                // With sound, the device paces evaluation: a block is
                // produced when the ring can take it, not when the window
                // redraws, so nothing is dropped and nothing races ahead.
                if self.device_is_ahead() {
                    std::thread::sleep(Duration::from_millis(1));
                } else {
                    self.frame();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.serve_pending(event_loop);
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let options = match parse(&args) {
        Ok(options) => options,
        Err(message) => {
            eprintln!("error: {message}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let budget = options.session.limits;
    let runtime = match options.session.backend.as_deref() {
        Some(name) => Session::runtime_and_display(name, budget),
        None => Session::runtime_and_display("gpu", budget).or_else(|error| {
            eprintln!("gpu backend unavailable ({error}); using the jit");
            Session::runtime_and_display("jit", budget).or_else(|error| {
                eprintln!("jit backend unavailable ({error}); using the interpreter");
                Session::runtime_and_display("cpu", budget)
            })
        }),
    };
    let (runtime, display) = match runtime {
        Ok(built) => built,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("backend: {}", runtime.backend_name());
    let gamepad = Gamepad::open();
    eprintln!(
        "gamepad: {}",
        match &gamepad {
            Some(pad) if pad.connected() => "connected, mapped to arrows, Z, X, Enter",
            Some(_) => "none connected; a pad plugged in later maps to arrows, Z, X, Enter",
            None => "unavailable",
        }
    );
    let mut session = match options.session.open_over(runtime) {
        Ok(session) => session,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };
    // An agent composes a served scene and reads back what it holds:
    // what it reads is held to its documents' directories and what the
    // person allowed.
    if options.serve {
        session.restrict_reads();
    }
    for event in session.sync() {
        eprintln!("{event}");
    }
    // The first load published inline; replacements prepare off the frame.
    session.set_background(true);
    let mut player = Player {
        session,
        scale: options.scale,
        requested: options.window,
        clock: Instant::now(),
        window: None,
        surface: None,
        display,
        presenter: None,
        downloads: 0,
        last_frame: Vec::new(),
        last_size: (0, 0),
        placement: Placement::default(),
        input: InputState::default(),
        cursor: (0.0, 0.0),
        status: "starting".into(),
        reported: orla_player::Reporting::default(),
        path: "",
        requests: None,
        preview: None,
        audio: None,
        gamepad,
        audio_allowed: !options.no_audio,
        audio_failed: false,
        hud: false,
        stats: Stats::default(),
        panel: None,
        replaying: None,
    };
    if let Some(replay) = &options.replay {
        let replayer = orla_session::replay::Replayer::start(replay, &mut player.session);
        player.replaying = Some((replayer, None));
        eprintln!("playing the replay: its inputs drive the window until it ends");
    }
    if options.serve {
        let (sender, receiver) = channel();
        if let Some(address) = &options.listen {
            let origins = orla_protocol::websocket::Origins::with(options.origins.clone());
            match orla_protocol::listen(address, sender.clone(), origins) {
                Ok(bound) => eprintln!("listening on {bound}"),
                Err(error) => {
                    eprintln!("error: cannot listen on {address}: {error}");
                    return ExitCode::FAILURE;
                }
            }
        }
        orla_protocol::stdin_into(sender);
        player.requests = Some(receiver);
        eprintln!("serving the creation protocol on stdin/stdout");
    }
    // Evaluate once before opening the window so it can take the output size.
    let context = player.context();
    match player.session.evaluate(&context) {
        Ok(report) => {
            if let Ok((width, height)) = player.session.runtime().host().image_size(report.output) {
                player.last_size = (width, height);
            }
        }
        Err(error) => {
            eprintln!("evaluation failed: {error}; the window opens when it succeeds");
        }
    }
    let event_loop = match EventLoop::new() {
        Ok(event_loop) => event_loop,
        Err(error) => {
            eprintln!("error: cannot start the event loop: {error}");
            return ExitCode::FAILURE;
        }
    };
    event_loop.set_control_flow(ControlFlow::Poll);
    match event_loop.run_app(&mut player) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
