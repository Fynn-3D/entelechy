//! Shared native window loop (`winit` + `wgpu` presenter + `softbuffer`
//! fallback + `cpal` audio + `gilrs` gamepads) for both `entelechy-player`
//! and standalone exported applications (`entelechy-runner`).

use std::num::NonZeroU32;
use std::process::ExitCode;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use entelechy_gpu::present::{Display, Overlay, Presenter};
use entelechy_session::Session;
pub use entelechy_session::bundle::PointerMode;
use entelechy_session::play::{self, Frame, Placement, Play, View, hud};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{
    DeviceEvent, DeviceId, ElementState, Ime, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, ModifiersState};
use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
use winit::window::{CursorGrabMode, CursorIcon, CustomCursor, Window, WindowId};

use crate::audio::Audio;
use crate::gamepad::Gamepad;

/// The window `window()` reads before one opens, when none was asked for.
pub const DEFAULT_WINDOW: (u32, u32) = (960, 540);

/// The system's clipboard, opened for each read and write.
pub struct SystemClipboard;

impl entelechy_session::Clipboard for SystemClipboard {
    fn text(&mut self) -> Option<String> {
        arboard::Clipboard::new().ok()?.get_text().ok()
    }

    fn put(&mut self, text: &str) -> Result<(), String> {
        arboard::Clipboard::new()
            .and_then(|mut clipboard| clipboard.set_text(text.to_owned()))
            .map_err(|e| format!("the clipboard: {e}"))
    }
}

/// Seconds since the process first asked for the window clock.
pub fn clock() -> f64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// Window presentation options shared by the Player and the standalone Runner.
pub struct WindowConfig {
    /// Pixel scale multiplier when no explicit logical `requested` size is set.
    pub scale: u32,
    /// Requested logical window size, when given.
    pub requested: Option<(u32, u32)>,
    /// Pointer capture mode when the window is clicked.
    pub pointer_mode: PointerMode,
    /// Whether opening the system audio output is permitted.
    pub audio_allowed: bool,
}

/// Host callbacks differentiating the developer `entelechy-player` from a
/// standalone `entelechy-runner` application.
pub trait WindowHooks: Sized {
    /// Initial window title before the session names one.
    fn initial_title(&self) -> &str;

    /// Current window title for the active session.
    fn title(&self, session: &Session) -> String;

    /// Polls background protocol requests between frames, when serving.
    fn poll_pending(_runner: &mut WindowRunner<Self>, _event_loop: &ActiveEventLoop) {}

    /// Whether Cmd+O / Ctrl+O opens a folder chooser.
    fn wants_folder(&self) -> bool {
        false
    }

    /// Opens the folder chooser and applies it to `session`; returns true if updated.
    fn choose_folder(&mut self, _session: &mut Session) -> bool {
        false
    }
}

/// Live window state driving `Session` + `Play` over `winit`.
pub struct WindowRunner<H> {
    /// Running Entelechy session.
    pub session: Session,
    /// Shared device-free play state.
    pub play: Play,
    /// Front-specific hooks (`entelechy-player` protocol/folder or standalone title).
    pub hooks: H,
    scale: u32,
    requested: Option<(u32, u32)>,
    window: Option<Arc<Window>>,
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
    display: Option<Display>,
    presenter: Option<Presenter>,
    last_frame: Option<entelechy_kernel::Image>,
    /// Last output picture dimensions in pixels.
    pub last_size: (u32, u32),
    audio: Option<Audio>,
    gamepad: Option<Gamepad>,
    audio_allowed: bool,
    audio_failed: bool,
    pointer_look: Option<play::PointerLook>,
    pointer_mode: PointerMode,
    cursor: (f64, f64),
    held: Option<(f64, f64)>,
    swallowed_button: Option<MouseButton>,
    modifiers: ModifiersState,
    composing: bool,
}

impl<H: WindowHooks> WindowRunner<H> {
    /// Creates a new window runner around `session`, `play`, and `hooks`.
    pub fn new(
        session: Session,
        play: Play,
        display: Option<Display>,
        gamepad: Option<Gamepad>,
        config: WindowConfig,
        hooks: H,
    ) -> Self {
        Self {
            session,
            play,
            hooks,
            scale: config.scale,
            requested: config.requested,
            window: None,
            surface: None,
            display,
            presenter: None,
            last_frame: None,
            last_size: (0, 0),
            audio: None,
            gamepad,
            audio_allowed: config.audio_allowed,
            audio_failed: false,
            pointer_look: None,
            pointer_mode: config.pointer_mode,
            cursor: (0.0, 0.0),
            held: None,
            swallowed_button: None,
            modifiers: ModifiersState::empty(),
            composing: false,
        }
    }

    /// Current physical window dimensions, scale factor, and audio queue state.
    pub fn view(&self) -> View {
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
            queued: self.audio.as_ref().map_or(0, Audio::queued),
        }
    }

    fn device_is_ahead(&self) -> bool {
        self.audio.as_ref().is_some_and(|audio| !audio.has_room())
    }

    /// Prints queued `Play` status lines to standard error.
    pub fn say(&mut self) {
        for line in self.play.said() {
            eprintln!("{line}");
        }
    }

    /// Confirms the frame in flight; returns whether it committed cleanly.
    pub fn confirm(&mut self) -> bool {
        let confirmed = self.play.confirm(&mut self.session);
        self.say();
        confirmed
    }

    /// Evaluates and presents one window frame.
    pub fn frame(&mut self) -> Option<entelechy_graph::Report> {
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
                if let Some(look) = self.play.pointer_look(&mut self.session, &shown.report) {
                    self.pointer_look = Some(look);
                }
                Some(shown.report)
            }
        }
    }

    /// Refreshes the window's title from [`WindowHooks::title`].
    pub fn update_title(&self) {
        if let Some(window) = &self.window {
            window.set_title(&self.hooks.title(&self.session));
        }
    }

    fn show(
        &mut self,
        output: entelechy_kernel::Handle,
    ) -> (Option<Placement>, &'static str, bool) {
        if let (Some(window), Some(presenter)) = (&self.window, &mut self.presenter)
            && let Some((device, width, height)) = self.session.on_device(output)
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
        let Some(words) = self.play.heard(&mut self.session, handle) else {
            return;
        };
        if let Some(audio) = &mut self.audio {
            audio.push(&words, words_per_frame);
        }
    }

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

    fn wear(&mut self, event_loop: &ActiveEventLoop) {
        let (Some(window), Some(look)) = (&self.window, self.pointer_look.take()) else {
            return;
        };
        match look {
            play::PointerLook::System => window.set_cursor(CursorIcon::Default),
            play::PointerLook::Picture {
                width,
                height,
                rgba,
                hotspot,
            } => {
                let side = |n: u32| n as u16;
                match CustomCursor::from_rgba(
                    rgba,
                    side(width),
                    side(height),
                    side(hotspot.0),
                    side(hotspot.1),
                ) {
                    Ok(source) => window.set_cursor(event_loop.create_custom_cursor(source)),
                    Err(error) => eprintln!("pointer refused by the system: {error}"),
                }
            }
        }
    }

    fn hold(&mut self) {
        let Some(window) = &self.window else { return };
        let locked = window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
        if locked.is_ok() {
            window.set_cursor_visible(false);
            if self.pointer_mode == PointerMode::Lock {
                window.set_ime_allowed(false);
                self.composing = false;
            }
            self.held = Some(self.cursor);
        }
    }

    fn let_go(&mut self) {
        let (Some(at), Some(window)) = (self.held.take(), &self.window) else {
            return;
        };
        let _ = window.set_cursor_grab(CursorGrabMode::None);
        let _ = window.set_cursor_position(winit::dpi::PhysicalPosition::new(at.0, at.1));
        window.set_cursor_visible(true);
        if self.pointer_mode == PointerMode::Lock {
            window.set_ime_allowed(true);
        }
        self.cursor = at;
    }

    /// Runs the native `winit` event loop until the window closes.
    pub fn run(mut self) -> ExitCode {
        let event_loop = match EventLoop::new() {
            Ok(event_loop) => event_loop,
            Err(error) => {
                eprintln!("error: cannot start the event loop: {error}");
                return ExitCode::FAILURE;
            }
        };
        event_loop.set_control_flow(ControlFlow::Poll);
        match event_loop.run_app(&mut self) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::FAILURE
            }
        }
    }
}

impl<H: WindowHooks> ApplicationHandler for WindowRunner<H> {
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
            .with_title(self.hooks.initial_title())
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
        window.set_ime_allowed(true);
        self.window = Some(window);
        self.update_title();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                let _ = self.confirm();
                event_loop.exit();
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = (position.x, position.y);
                if self.held.is_none() && self.pointer_mode != PointerMode::Lock {
                    self.play.pointer(position.x as f32, position.y as f32);
                }
            }
            WindowEvent::Focused(false) => {
                self.swallowed_button = None;
                self.let_go();
                self.play.release_all();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let held = state == ElementState::Pressed;
                if held && self.pointer_mode != PointerMode::Free && self.held.is_none() {
                    self.hold();
                    if self.pointer_mode == PointerMode::Lock && !self.play.replaying() {
                        self.swallowed_button = Some(button);
                        return;
                    }
                }
                if !held && self.swallowed_button == Some(button) {
                    self.swallowed_button = None;
                    return;
                }
                match button {
                    MouseButton::Left => self.play.button(0, held),
                    MouseButton::Right => self.play.button(1, held),
                    MouseButton::Middle => self.play.button(2, held),
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let (x, y) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (x, y),
                    MouseScrollDelta::PixelDelta(p) => (p.x as f32 / 40.0, p.y as f32 / 40.0),
                };
                self.play.wheel(x, y);
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
                let m = self.modifiers;
                self.play
                    .modifiers(m.shift_key(), m.control_key(), m.alt_key(), m.super_key());
            }
            WindowEvent::Ime(Ime::Preedit(text, _)) => self.composing = !text.is_empty(),
            WindowEvent::Ime(Ime::Commit(text)) => {
                self.composing = false;
                self.play.typed(&text);
            }
            WindowEvent::Ime(Ime::Disabled) => self.composing = false,
            WindowEvent::KeyboardInput { event, .. } => {
                let logical_key = event.key_without_modifiers();
                let KeyEvent {
                    state,
                    repeat,
                    text,
                    ..
                } = event;
                let down = state == ElementState::Pressed;
                let shortcut = self.modifiers.control_key() || self.modifiers.super_key();
                let opener = if cfg!(target_os = "macos") {
                    self.modifiers.super_key()
                } else {
                    self.modifiers.control_key()
                };
                if self.hooks.wants_folder()
                    && opener
                    && matches!(&logical_key, Key::Character(c) if c.eq_ignore_ascii_case("o"))
                {
                    if down && !repeat && self.hooks.choose_folder(&mut self.session) {
                        self.update_title();
                    }
                    return;
                }
                if down
                    && !self.composing
                    && !shortcut
                    && let Some(text) = &text
                {
                    self.play.typed(text);
                }
                if repeat {
                    return;
                }
                let name = match logical_key {
                    Key::Named(named) => format!("{named:?}"),
                    Key::Character(text) => text.to_string(),
                    _ => return,
                };
                if down
                    && name == "Escape"
                    && self.pointer_mode == PointerMode::Lock
                    && self.held.is_some()
                {
                    self.swallowed_button = None;
                    self.let_go();
                    self.play.release_all();
                    return;
                }
                self.play.key(&mut self.session, &name, down);
                self.say();
            }
            WindowEvent::RedrawRequested => {
                if self.device_is_ahead() {
                    std::thread::sleep(Duration::from_millis(1));
                } else {
                    self.frame();
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        let (Some(at), Some(window)) = (self.held, &self.window) else {
            return;
        };
        if let DeviceEvent::MouseMotion { delta } = event {
            let scale = window.scale_factor();
            if self.pointer_mode == PointerMode::Lock {
                self.play
                    .move_pointer((delta.0 * scale) as f32, (delta.1 * scale) as f32);
            } else {
                let size = window.inner_size();
                let next = crate::pointer::held_step(at, delta, scale, (size.width, size.height));
                self.held = Some(next);
                self.play.pointer(next.0 as f32, next.1 as f32);
            }
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.confirm();
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        H::poll_pending(self, event_loop);
        self.wear(event_loop);
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}
