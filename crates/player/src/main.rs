//! The `orla-player` command: a window over a live session.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::process::ExitCode;
use std::rc::Rc;
use std::time::{Duration, Instant};

use orla::session::{Event, Session};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const USAGE: &str = "\
usage:
  orla-player <scene.osn> [--kernels <dir>] [--input <name>=<file.ppm>]... [--scale <n>]
      Open a window showing the scene output. Kernel files are republished as
      they change; a rejected change keeps the running version. Press Escape
      or close the window to quit.";

struct Options {
    scene: PathBuf,
    kernels: Option<PathBuf>,
    inputs: Vec<(String, PathBuf)>,
    scale: u32,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut options = Options {
        scene: PathBuf::new(),
        kernels: None,
        inputs: Vec::new(),
        scale: 2,
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
    window: Option<Rc<Window>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
    last_frame: Vec<[f32; 4]>,
    last_size: (u32, u32),
    status: String,
    last_title: Instant,
}

impl Player {
    fn frame(&mut self) {
        let events = self.session.sync();
        for event in &events {
            eprintln!("{event}");
        }
        let started = Instant::now();
        match self.session.evaluate(self.clock.elapsed().as_secs_f32()) {
            Ok(report) => {
                let host = self.session.runtime().host();
                if let Ok(image) = host.image(report.output) {
                    self.last_size = (image.width(), image.height());
                    self.last_frame.clear();
                    self.last_frame.extend_from_slice(image.pixels());
                }
                let elapsed = started.elapsed();
                if self.last_title.elapsed() > Duration::from_millis(250) {
                    self.status = format!(
                        "{:.1} ms, {} executed, {} reused",
                        elapsed.as_secs_f64() * 1000.0,
                        report.executed.len(),
                        report.reused.len()
                    );
                    self.last_title = Instant::now();
                    self.update_title();
                }
            }
            Err(error) => {
                let message = format!("evaluation failed: {error}");
                if self.status != message {
                    eprintln!("{message}; last frame is kept");
                    self.status = message;
                    self.update_title();
                }
            }
        }
        self.blit(events.is_empty());
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

    fn blit(&mut self, _quiet: bool) {
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
        orla_player::present(
            &self.last_frame,
            self.last_size.0,
            self.last_size.1,
            size.width,
            size.height,
            &mut buffer,
        );
        let _ = buffer.present();
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
            Ok(window) => Rc::new(window),
            Err(error) => {
                eprintln!("cannot create a window: {error}");
                event_loop.exit();
                return;
            }
        };
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
        self.window = Some(window);
        self.update_title();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::RedrawRequested => self.frame(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
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
    let mut session = Session::open(
        &options.scene,
        options.kernels.as_deref(),
        orla::session::DEFAULT_BUDGET_PIXELS,
    );
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
        last_frame: Vec::new(),
        last_size: (0, 0),
        status: "starting".into(),
        last_title: Instant::now(),
    };
    // Evaluate once before opening the window so it can take the output size.
    match player.session.evaluate(0.0) {
        Ok(report) => {
            if let Ok(image) = player.session.runtime().host().image(report.output) {
                player.last_size = (image.width(), image.height());
                player.last_frame = image.pixels().to_vec();
            }
        }
        Err(error) => {
            eprintln!("evaluation failed: {error}; the window opens when it succeeds");
            let _ = Event::SceneLoaded;
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
