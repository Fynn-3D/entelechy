//! The `orla-player` command: a window over a live session.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use orla::session::Session;
use orla_gpu::GpuBackend;
use orla_gpu::present::Presenter;
use orla_player::{Placement, key_code, to_output};
use orla_runtime::{Context, InputState};
use serde_json::Value as Json;
use std::sync::mpsc::{Receiver, channel};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const USAGE: &str = "\
usage:
  orla-player <scene.osn> [--library <dir|file>] [--input <name>=<file.ppm>]... [--scale <n>]
              [--backend gpu|cpu]
      Open a window showing the scene output. Kernel files are republished as
      they change; a rejected change keeps the running version. The pointer
      and keys reach Kernels that declare `input.read`. F5 resets state.
      Press Escape or close the window to quit.
      With --serve, the creation protocol runs on stdin and stdout inside the
      window: an agent's publish, compose, and param land in the frame a
      person watches, and its evaluate observes that person's input.";

struct Options {
    scene: PathBuf,
    kernels: Option<PathBuf>,
    inputs: Vec<(String, PathBuf)>,
    scale: u32,
    backend: Option<String>,
    serve: bool,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut options = Options {
        scene: PathBuf::new(),
        kernels: None,
        inputs: Vec::new(),
        scale: 2,
        backend: None,
        serve: false,
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
            "--backend" => options.backend = Some(value("--backend")?),
            "--serve" => options.serve = true,
            "--scale" => {
                options.scale = value("--scale")?
                    .parse::<u32>()
                    .ok()
                    .filter(|scale| *scale > 0)
                    .ok_or_else(|| "--scale needs a positive integer".to_string())?;
            }
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
                    return Err("exactly one scene file is expected".into());
                }
            }
        }
    }
    options.scene = scene.ok_or_else(|| "a scene file is required".to_string())?;
    Ok(options)
}

struct Player {
    session: Session,
    scale: u32,
    clock: Instant,
    window: Option<Arc<Window>>,
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
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
    last_title: Instant,
    /// Protocol requests arriving from stdin, when serving.
    requests: Option<Receiver<String>>,
}

impl Player {
    fn context(&self) -> Context {
        Context {
            time: self.clock.elapsed().as_secs_f32(),
            input: self.input,
        }
    }

    fn frame(&mut self) -> Option<orla_graph::Report> {
        let events = self.session.sync();
        for event in &events {
            eprintln!("{event}");
        }
        let started = Instant::now();
        let context = self.context();
        match self.session.evaluate(&context) {
            Ok(report) => {
                let elapsed = started.elapsed();
                let shown = self.show(report.output);
                if self.last_title.elapsed() > Duration::from_millis(250) {
                    self.status = format!(
                        "{:.1} ms, {} executed, {} reused, {}, {} downloads",
                        elapsed.as_secs_f64() * 1000.0,
                        report.executed.len(),
                        report.reused.len(),
                        shown,
                        self.downloads
                    );
                    self.last_title = Instant::now();
                    self.update_title();
                }
                Some(report)
            }
            Err(error) => {
                let message = format!("evaluation failed: {error}");
                if self.status != message {
                    eprintln!("{message}; last frame is kept");
                    self.status = message;
                    self.update_title();
                }
                self.blit();
                None
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
        let pending: Vec<String> = std::iter::from_fn(|| requests.try_recv().ok()).collect();
        for line in pending {
            let (response, quit) = match orla::protocol::parse_request(&line) {
                Ok((id, request))
                    if request.get("op").and_then(Json::as_str) == Some("evaluate") =>
                {
                    let result = match self.frame() {
                        Some(report) => Ok(orla::protocol::report_json(&self.session, &report)),
                        None => Err(serde_json::json!({ "message": self.status })),
                    };
                    (orla::protocol::envelope(&id, result), false)
                }
                _ => orla::protocol::respond_line(&mut self.session, &line),
            };
            println!("{response}");
            if quit {
                event_loop.exit();
            }
        }
    }

    /// Shows an output image: directly from the device when it is resident
    /// and a presenter exists, otherwise by downloading it and blitting.
    /// Returns a short description of the path taken.
    fn show(&mut self, output: orla_runtime::ImageHandle) -> &'static str {
        if let (Some(window), Some(presenter)) = (&self.window, &mut self.presenter) {
            let host = self.session.runtime().host();
            let resident = host.resident(output).ok().flatten();
            if let (Some(device), Ok((width, height))) = (resident, host.image_size(output)) {
                let size = window.inner_size();
                let backend = self.session.runtime().backend().as_any();
                if let Some(gpu) = backend.downcast_ref::<GpuBackend>() {
                    match gpu.present(presenter, device, width, height, size.width, size.height) {
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

    fn update_title(&self) {
        if let Some(window) = &self.window {
            let name = self
                .session
                .graph()
                .map(|graph| graph.name.clone())
                .unwrap_or_else(|| "no scene".into());
            window.set_title(&format!("Orla Player: {name} ({})", self.status));
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
        let attributes = Window::default_attributes()
            .with_title("Orla Player")
            .with_inner_size(LogicalSize::new(width * self.scale, height * self.scale));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                eprintln!("cannot create a window: {error}");
                event_loop.exit();
                return;
            }
        };
        let size = window.inner_size();
        let backend = self.session.runtime().backend().as_any();
        if let Some(gpu) = backend.downcast_ref::<GpuBackend>() {
            match gpu.presenter(window.clone(), size.width, size.height) {
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
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.input.pointer_down = state == ElementState::Pressed;
                self.update_pointer();
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
                self.frame();
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
    let budget = orla::session::DEFAULT_BUDGET_PIXELS;
    let runtime = match options.backend.as_deref() {
        Some(name) => Session::runtime_for(name, budget),
        None => Session::runtime_for("gpu", budget).or_else(|error| {
            eprintln!("gpu backend unavailable ({error}); using cpu");
            Session::runtime_for("cpu", budget)
        }),
    };
    let runtime = match runtime {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("backend: {}", runtime.backend_name());
    let mut session = Session::open_with(&options.scene, options.kernels.as_deref(), runtime);
    for (name, path) in &options.inputs {
        if let Err(error) = session.bind_external(name, path) {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    }
    for event in session.sync() {
        eprintln!("{event}");
    }
    let mut player = Player {
        session,
        scale: options.scale,
        clock: Instant::now(),
        window: None,
        surface: None,
        presenter: None,
        downloads: 0,
        last_frame: Vec::new(),
        last_size: (0, 0),
        placement: Placement::default(),
        input: InputState::default(),
        cursor: (0.0, 0.0),
        status: "starting".into(),
        last_title: Instant::now(),
        requests: None,
    };
    if options.serve {
        let (sender, receiver) = channel();
        std::thread::spawn(move || {
            use std::io::BufRead;
            let stdin = std::io::stdin();
            for line in stdin.lock().lines().map_while(Result::ok) {
                if !line.trim().is_empty() && sender.send(line).is_err() {
                    break;
                }
            }
        });
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
