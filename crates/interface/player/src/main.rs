//! The `entelechy-player` command: a window, sound card and pads over a live
//! session. Frames, key meanings and the frame line are
//! `entelechy_session::play`'s.

use std::num::NonZeroU32;
use std::process::ExitCode;
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use entelechy_gpu::present::{Display, Overlay, Presenter};
use entelechy_player::audio::Audio;
use entelechy_player::gamepad::Gamepad;
use entelechy_session::Session;
use entelechy_session::play::{self, Frame, Placement, Play, View, hud};
use serde_json::Value as Json;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{
    DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::Key;
use winit::window::{CursorGrabMode, Window, WindowId};

const USAGE: &str = "\
usage:
  entelechy-player <scene.esn> [--library <dir|file>] [--input <name>=<file.ppm>]... [--scale <n>]
              [--replay <file.replay>]
              [--window <w>x<h>] [--backend gpu|cpu|jit|best] [--memory <MB>] [--work <M>]
              [--allow file.read|file.write|file.delete=<dir>]... [--grant <capability,...>]
      Open a window showing the scene output. Kernel files are republished as
      they change; a rejected change keeps the running version. The pointer
      and keys reach Kernels that declare `input.read`, and so do the
      window's size in pixels and its scale factor, which a node's `domain`
      may take as `window()`; --window opens the window at that logical size
      instead of the output's size times --scale. F5 resets state;
      F6 cycles the preview through the graph's nodes and back to the output;
      F10 shows a line of frame data over the picture, naming the port shown.
      Tab, Escape and every other key are the scene's.
      --hold-pointer holds the pointer in the window once it is clicked:
      the system's cursor is locked and hidden, the scene's pointer moves
      with the mouse and stays inside the window (the scene draws it);
      the window losing the focus (Cmd+Tab) lets it go.
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

/// The window `window()` reads before one opens, when none was asked for.
const DEFAULT_WINDOW: (u32, u32) = (960, 540);

struct Options {
    /// The session: scene, library, inputs, params, backend, limits and
    /// allowances.
    session: entelechy_session::SessionOptions,
    scale: u32,
    /// The window's logical size, when given; else the output's times `scale`.
    window: Option<(u32, u32)>,
    serve: bool,
    listen: Option<String>,
    origins: Vec<String>,
    /// Leave the sound card closed even when the scene has a sound output.
    no_audio: bool,
    /// A replay driving the scene frame by frame at its own times; the
    /// person takes over when it ends.
    replay: Option<entelechy_session::replay::Replay>,
    /// Hold the pointer in the window once it is clicked.
    hold_pointer: bool,
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
        replay: None,
        hold_pointer: false,
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
            "--hold-pointer" => options.hold_pointer = true,
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
    // A replay supplies scene, library and window the command line
    // leaves unset.
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
    /// The backend's device, when it has one to present from.
    display: Option<Display>,
    /// The GPU presenter, when the backend is the GPU and the surface is
    /// configured; it shows frames without readback.
    presenter: Option<Presenter>,
    last_frame: Option<entelechy_kernel::Image>,
    last_size: (u32, u32),
    /// Protocol requests arriving from stdin or a socket, when serving.
    requests: Option<Receiver<entelechy_protocol::Incoming>>,
    /// The answers owed, including a `compose` or `publish` waiting for
    /// its change to stand while frames continue.
    answers: entelechy_protocol::Answers,
    /// The sound card, opened when a scene has a sound-shaped output.
    audio: Option<Audio>,
    /// The gamepads, when the system offers them.
    gamepad: Option<Gamepad>,
    audio_allowed: bool,
    audio_failed: bool,
    /// The sound of the frame in flight, fetched to be heard next frame
    /// ([`Session::overlaps`]).
    sound_behind: Option<entelechy_kernel::Fetch>,
    /// Whether the pointer is held once the window is clicked, where the
    /// system's cursor last was, and the held pointer while it is held
    /// ([`entelechy_player::pointer`]).
    hold_pointer: bool,
    cursor: (f64, f64),
    held: Option<(f64, f64)>,
}

impl entelechy_protocol::Served for Player {
    fn session(&mut self) -> &mut Session {
        &mut self.session
    }

    /// `evaluate` runs one window frame under the person's input;
    /// measured, each of its nodes waits for the device.
    fn own(&mut self, incoming: &entelechy_protocol::Incoming) -> Option<(String, bool)> {
        let (id, request) = entelechy_protocol::parse_request(&incoming.line).ok()?;
        if request.get("op").and_then(Json::as_str) != Some("evaluate") {
            return None;
        }
        let measured = entelechy_protocol::measure_asked(&request);
        self.session.set_measured(measured);
        let frame = self.frame();
        if measured {
            self.session.set_measured(false);
        }
        // The frame answered is confirmed: what it committed holds. Its
        // input and clock are the person's; context the request gave is
        // named as not read, so a caller is not misled.
        let ignored = entelechy_protocol::context_given(&request);
        let result = match frame {
            Some(report) if self.confirm() => {
                let mut answer =
                    entelechy_protocol::evaluation_json(&self.session, &report, measured);
                if !ignored.is_empty() {
                    answer["ignored"] = serde_json::json!(ignored);
                    answer["note"] = serde_json::json!(
                        "the window's input and clock are the person's: the Player runs each evaluate under them and reads none of the fields named in `ignored`; `entelechy serve` takes them"
                    );
                }
                Ok(answer)
            }
            _ => Err(serde_json::json!({ "message": self.play.status() })),
        };
        Some((entelechy_protocol::envelope(&id, result), false))
    }
}

impl Player {
    /// The window's physical size and scale factor (before it opens, the
    /// requested size at scale one, or the default), and the sound rate.
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

    /// Whether the sound device is still playing the last block; it then
    /// paces the next evaluation.
    fn device_is_ahead(&self) -> bool {
        self.audio.as_ref().is_some_and(|audio| !audio.has_room())
    }

    /// Prints what the play has to say.
    fn say(&mut self) {
        for line in self.play.said() {
            eprintln!("{line}");
        }
    }

    /// Confirms the frame in flight; one found failed is taken back, and
    /// its sound is not heard.
    fn confirm(&mut self) -> bool {
        let confirmed = self.play.confirm(&mut self.session);
        if !confirmed {
            self.sound_behind = None;
        }
        self.say();
        confirmed
    }

    fn frame(&mut self) -> Option<entelechy_graph::Report> {
        if let Some(gamepad) = &mut self.gamepad {
            gamepad.poll(&mut self.play);
        }
        let view = self.view();
        let frame = self.play.frame(&mut self.session, view);
        self.say();
        match frame {
            Frame::Wait => None,
            Frame::Failed => {
                // A failed frame takes back the one before: its sound
                // is not heard.
                self.sound_behind = None;
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

    /// Answers protocol requests arrived since the last frame; `evaluate`
    /// runs a window frame, the rest go to the shared handler.
    fn serve_pending(&mut self, event_loop: &ActiveEventLoop) {
        let Some(requests) = &self.requests else {
            return;
        };
        let pending: Vec<entelechy_protocol::Incoming> =
            std::iter::from_fn(|| requests.try_recv().ok()).collect();
        // A request sees what the last frame left, confirmed.
        if !pending.is_empty() || self.answers.is_waiting() {
            self.confirm();
        }
        // Waiting answers first, then new requests; a quit on standard
        // input closes the window.
        let mut answers = std::mem::take(&mut self.answers);
        let mut quit = answers.poll(self);
        for incoming in pending {
            quit |= answers.answer(self, incoming);
        }
        self.answers = answers;
        if quit {
            event_loop.exit();
        }
    }

    /// Shows a picture from the device when resident and presentable,
    /// else by download and blit. Returns placement, path, and whether it
    /// downloaded.
    fn show(
        &mut self,
        output: entelechy_kernel::Handle,
    ) -> (Option<Placement>, &'static str, bool) {
        if let (Some(window), Some(presenter)) = (&self.window, &mut self.presenter) {
            // What the device cannot show as it lies downloads (widened)
            // and is drawn or refused below.
            if let Some((device, width, height)) = self.session.on_device(output) {
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
                self.last_frame = Some(image);
                true
            }
            Err(_) => false,
        };
        (self.blit(), "downloaded and blitted", downloaded)
    }

    /// Hands the frame's sound output to the sound card, opening it on
    /// first use.
    fn sound(&mut self, sound: Option<entelechy_kernel::Handle>) {
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
        // Materializing is the observation boundary: on the frame thread,
        // never in the device callback. Overlapping, the sound heard is
        // the frame before's, confirmed by now and fetched behind its work
        // alone: this frame's is fetched for the next.
        let words: Vec<u32> = if self.session.overlaps() {
            let Ok(fetch) = self.session.fetch(handle) else {
                return;
            };
            match self
                .sound_behind
                .replace(fetch)
                .map(entelechy_kernel::Fetch::take)
            {
                Some(Ok(data)) => data.values().to_vec(),
                _ => return,
            }
        } else {
            match self.session.field(handle) {
                Ok(data) => data.values().to_vec(),
                Err(_) => return,
            }
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
            window.set_title(&format!("Entelechy Player: {name}"));
        }
    }

    /// Draws the last downloaded picture and the frame line over it;
    /// returns where it landed.
    fn blit(&mut self) -> Option<Placement> {
        if self.presenter.is_some() && self.last_frame.is_none() {
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
        let pixels = self
            .last_frame
            .as_ref()
            .map_or(&[][..], entelechy_kernel::Image::pixels);
        let placement = play::present(
            pixels,
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

impl Player {
    /// The pointer held: the system's cursor locked (or, where it cannot
    /// be, confined) and hidden, the held pointer where it was.
    fn hold(&mut self) {
        let Some(window) = &self.window else { return };
        let locked = window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
        if locked.is_ok() {
            window.set_cursor_visible(false);
            self.held = Some(self.cursor);
        }
    }

    /// The pointer let go: the system's cursor free and shown where the
    /// held one was.
    fn let_go(&mut self) {
        let (Some(at), Some(window)) = (self.held.take(), &self.window) else {
            return;
        };
        let _ = window.set_cursor_grab(CursorGrabMode::None);
        let _ = window.set_cursor_position(winit::dpi::PhysicalPosition::new(at.0, at.1));
        window.set_cursor_visible(true);
        self.cursor = at;
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
            .with_title("Entelechy Player")
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
                self.cursor = (position.x, position.y);
                if self.held.is_none() {
                    self.play.pointer(position.x as f32, position.y as f32);
                }
            }
            // Out of the focus, a held pointer is let go.
            WindowEvent::Focused(false) => self.let_go(),
            WindowEvent::MouseInput { state, button, .. } => {
                let held = state == ElementState::Pressed;
                if held && self.hold_pointer && self.held.is_none() {
                    self.hold();
                }
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
                self.play.key(&mut self.session, &name, down);
                self.say();
            }
            WindowEvent::RedrawRequested => {
                // With sound, a block is produced when the ring can take
                // it: the device paces evaluation.
                if self.device_is_ahead() {
                    std::thread::sleep(Duration::from_millis(1));
                } else {
                    self.frame();
                }
            }
            _ => {}
        }
    }

    /// A held pointer moves by the mouse's own motion, inside the window.
    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        let (Some(at), Some(window)) = (self.held, &self.window) else {
            return;
        };
        if let DeviceEvent::MouseMotion { delta } = event {
            let size = window.inner_size();
            let next = entelechy_player::pointer::held_step(
                at,
                delta,
                window.scale_factor(),
                (size.width, size.height),
            );
            self.held = Some(next);
            self.play.pointer(next.0 as f32, next.1 as f32);
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
    let mut session = match options.session.open_over(runtime) {
        Ok(session) => session,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };
    // A served scene's reads are held to its documents' directories and
    // the person's allowances, fixed before the scene is first read.
    if options.serve
        && let Err(error) = session.restrict_reads()
    {
        eprintln!("error: {error}");
        return ExitCode::FAILURE;
    }
    for event in session.sync() {
        eprintln!("{event}");
    }
    // The first load published inline; replacements prepare off the frame.
    session.set_background(true);
    // The window's frames overlap the device's: the next is made while
    // the device draws this one.
    if let Err(error) = session.set_overlap(true) {
        eprintln!("evaluation failed: {error}");
    }
    let mut player = Player {
        session,
        play: Play::new(clock),
        scale: options.scale,
        requested: options.window,
        window: None,
        surface: None,
        display,
        presenter: None,
        last_frame: None,
        last_size: (0, 0),
        requests: None,
        answers: entelechy_protocol::Answers::new(),
        audio: None,
        gamepad,
        audio_allowed: !options.no_audio,
        audio_failed: false,
        sound_behind: None,
        hold_pointer: options.hold_pointer,
        cursor: (0.0, 0.0),
        held: None,
    };
    if let Some(replay) = &options.replay {
        player.play.replay(replay, &mut player.session);
        player.say();
    }
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
        player.requests = Some(receiver);
        eprintln!("serving the creation protocol on stdin/stdout");
    }
    // Evaluate once so the window opens at the output size; a replay
    // owns its window and every frame, so it skips this.
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
