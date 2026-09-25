//! The `orla-player` command: a window over a live session. The frame
//! itself, what the keys mean and what the line of frame data says are
//! the session's play (`orla_session::play`), which every interface that
//! plays a scene shares; this is the window, the sound card and the pads
//! it hands its person's hands from, and shows the picture on.

use std::num::NonZeroU32;
use std::process::ExitCode;
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use orla_gpu::present::{Display, Overlay, Presenter};
use orla_player::audio::Audio;
use orla_player::gamepad::Gamepad;
use orla_session::Session;
use orla_session::play::{self, Asked, Frame, Placement, Play, View, hud};
use serde_json::Value as Json;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::Key;
use winit::window::{Window, WindowId};

const USAGE: &str = "\
usage:
  orla-player <scene.osn> [--library <dir|file>] [--input <name>=<file.ppm>]... [--scale <n>]
              [--replay <file.replay>]
              [--window <w>x<h>] [--backend gpu|cpu|jit|best] [--memory <MB>] [--work <M>]
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

/// The window's clock: seconds since the Player started.
fn clock() -> f64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

struct Player {
    session: Session,
    play: Play,
    scale: u32,
    /// The logical size the window opens at, when asked for; also what
    /// `window()` reads before the window exists.
    requested: Option<(u32, u32)>,
    window: Option<Arc<Window>>,
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
    /// The device the backend computes on, when it has one to show: a
    /// window presents what it left resident there without a readback.
    display: Option<Display>,
    /// The GPU presenter, when the backend is the GPU and the surface could
    /// be configured; then frames are shown without any readback.
    presenter: Option<Presenter>,
    last_frame: Vec<[f32; 4]>,
    last_size: (u32, u32),
    /// Protocol requests arriving from stdin or a socket, when serving.
    requests: Option<Receiver<orla_protocol::Incoming>>,
    /// The sound card, opened when a scene has a sound-shaped output.
    audio: Option<Audio>,
    /// The gamepads, when the system offers them.
    gamepad: Option<Gamepad>,
    audio_allowed: bool,
    audio_failed: bool,
}

impl Player {
    /// What the window says about itself: its physical size and scale
    /// factor; before it opens, the requested size at scale one, or a
    /// plain default, so a scene sized by the window has something to
    /// fill on its first evaluation. And the sound card's rate.
    fn view(&self) -> View {
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
        View {
            window,
            scale,
            rate: self.audio.as_ref().map_or(0.0, |audio| audio.rate() as f32),
        }
    }

    /// Whether the sound device is still playing the last block: then the
    /// device sets the pace and the next evaluation waits for it.
    fn device_is_ahead(&self) -> bool {
        self.audio.as_ref().is_some_and(|audio| !audio.has_room())
    }

    /// Prints what the play has to say.
    fn say(&mut self) {
        for line in self.play.said() {
            eprintln!("{line}");
        }
    }

    fn frame(&mut self) -> Option<orla_graph::Report> {
        if let Some(gamepad) = &mut self.gamepad {
            gamepad.poll(&mut self.play);
        }
        let view = self.view();
        let frame = self.play.frame(&mut self.session, view);
        self.say();
        match frame {
            Frame::Wait => None,
            Frame::Failed => {
                let placement = self.blit();
                self.play.landed(placement, self.last_size);
                None
            }
            Frame::Shown(shown) => {
                let presenting = clock();
                let (placement, path, downloaded) = self.show(shown.picture);
                self.play.placed(
                    placement,
                    self.last_size,
                    path,
                    downloaded,
                    clock() - presenting,
                );
                self.sound(shown.sound);
                Some(shown.report)
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
                        None => Err(serde_json::json!({ "message": self.play.status() })),
                    };
                    (orla_protocol::envelope(&id, result), false)
                }
                _ => orla_protocol::respond_from(&mut self.session, &incoming),
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

    /// Shows a picture: directly from the device when it is resident
    /// and a presenter exists, otherwise by downloading it and blitting.
    /// Answers where it landed, how, and whether that took a download.
    fn show(&mut self, output: orla_runtime::Handle) -> (Option<Placement>, &'static str, bool) {
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
                    let overlay = self.play.panel().map(|panel| Overlay {
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
                            return (Some(placement), "presented on the device", false);
                        }
                        Ok(None) => {
                            self.last_size = (width, height);
                            return (None, "window not ready", false);
                        }
                        Err(error) => eprintln!("present failed: {error}; downloading instead"),
                    }
                }
            }
        }
        let downloaded = match self.session.picture(output) {
            Ok(image) => {
                self.last_size = (image.width(), image.height());
                self.last_frame.clear();
                self.last_frame.extend_from_slice(image.pixels());
                true
            }
            Err(_) => false,
        };
        (self.blit(), "downloaded and blitted", downloaded)
    }

    /// Hands the frame's sound output to the sound card, opening it on
    /// first need.
    fn sound(&mut self, sound: Option<orla_runtime::Handle>) {
        if !self.audio_allowed || self.audio_failed {
            return;
        }
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
                    self.play.sound_opened(audio.describe());
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
        let words: Vec<u32> = match self.session.field(handle) {
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

    /// Draws the last downloaded picture, and the line of frame data
    /// over it, into the window: where it landed, when it was drawn.
    fn blit(&mut self) -> Option<Placement> {
        if self.presenter.is_some() && self.last_frame.is_empty() {
            return None;
        }
        let (Some(window), Some(surface)) = (&self.window, &mut self.surface) else {
            return None;
        };
        let size = window.inner_size();
        let (Some(width), Some(height)) =
            (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
        else {
            return None;
        };
        if surface.resize(width, height).is_err() {
            return None;
        }
        let Ok(mut buffer) = surface.buffer_mut() else {
            return None;
        };
        let placement = play::present(
            &self.last_frame,
            self.last_size.0,
            self.last_size.1,
            size.width,
            size.height,
            &mut buffer,
        );
        if let Some(panel) = self.play.panel() {
            hud::stamp(panel, &mut buffer, size.width, size.height);
        }
        let _ = buffer.present();
        Some(placement)
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
                self.play.pointer(position.x as f32, position.y as f32);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let held = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => self.play.button(0, held),
                    MouseButton::Right => self.play.button(1, held),
                    MouseButton::Middle => self.play.button(2, held),
                    _ => {}
                }
            }
            // The wheel accumulates until the next evaluation reads it.
            WindowEvent::MouseWheel { delta, .. } => {
                let (x, y) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (x, y),
                    MouseScrollDelta::PixelDelta(p) => (p.x as f32 / 40.0, p.y as f32 / 40.0),
                };
                self.play.wheel(x, y);
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
                let name = match logical_key {
                    Key::Named(named) => format!("{named:?}"),
                    Key::Character(text) => text.to_string(),
                    _ => return,
                };
                let asked = self.play.key(&mut self.session, &name, down);
                self.say();
                if asked == Some(Asked::Stop) {
                    event_loop.exit();
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
    let backend = options.session.backend.as_deref().unwrap_or("best");
    let built = match Session::build(backend, options.session.limits, orla_gpu::Wait::blocking()) {
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
        play: Play::new(clock),
        scale: options.scale,
        requested: options.window,
        window: None,
        surface: None,
        display,
        presenter: None,
        last_frame: Vec::new(),
        last_size: (0, 0),
        requests: None,
        audio: None,
        gamepad,
        audio_allowed: !options.no_audio,
        audio_failed: false,
    };
    if let Some(replay) = &options.replay {
        player.play.replay(replay, &mut player.session);
        player.say();
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
        if let Err(error) = orla_protocol::stdin_into(sender) {
            eprintln!("error: cannot read standard input: {error}");
            return ExitCode::FAILURE;
        }
        player.requests = Some(receiver);
        eprintln!("serving the creation protocol on stdin/stdout");
    }
    // Evaluate once before opening the window so it can take the output
    // size. Not under a replay: its window is the replay's, and a frame
    // the replay did not ask for would move the world it starts from.
    let context = player.play.context(player.view());
    let probed = (!player.play.replaying()).then(|| player.session.evaluate(&context));
    match probed {
        Some(Ok(report)) => {
            if let Ok((width, height)) = player.session.runtime().host().image_size(report.output) {
                player.last_size = (width, height);
            }
        }
        Some(Err(error)) => {
            eprintln!("evaluation failed: {error}; the window opens when it succeeds");
        }
        None => {}
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
