//! The Player's frame, device-free. Every interface that plays a scene
//! hands its keys, pointer, wheel and pads here, asks for a frame, and
//! shows the picture and plays the sound it names. This module owns the
//! meaning of F5, Tab and F10, of F7, F8 and F9 (the scene's clock held
//! still and stepped a frame at a time), the pad
//! mapping, the picture and sound choice, replay driving and the
//! frame-data line.

pub mod hud;
pub mod pad;

use entelechy_graph::{NodeId, Report};
use entelechy_kernel::{Context, Handle, InputState};

pub use entelechy_rules::placement::{Placement, placement};
pub use pad::{Pad, PadAxis, PadButton};

use crate::Session;
#[cfg(feature = "documents")]
use crate::replay::{Replay, Replayer};

/// Deduplicates a run of failures: the first frame of a failure is
/// reported, repeats that differ only in numbers are quiet, and the
/// first working frame after them is reported.
#[derive(Default)]
pub struct Reporting {
    shape: String,
}

impl Reporting {
    /// Whether this failure is new: its shape differs from the current.
    pub fn failed(&mut self, message: &str) -> bool {
        let shape = failure_shape(message);
        if self.shape == shape {
            return false;
        }
        self.shape = shape;
        true
    }

    /// Whether a working frame ends a run of failures.
    pub fn worked(&mut self) -> bool {
        let was_failing = !self.shape.is_empty();
        self.shape.clear();
        was_failing
    }
}

/// A failure's message with each number replaced by `#`.
fn failure_shape(message: &str) -> String {
    let mut shape = String::with_capacity(message.len());
    let mut inside = false;
    for character in message.chars() {
        let digit =
            character.is_ascii_digit() || (inside && (character == '.' || character == ','));
        if digit {
            if !inside {
                shape.push('#');
                inside = true;
            }
        } else {
            inside = false;
            shape.push(character);
        }
    }
    shape
}

/// Maps a window position to output pixel coordinates; positions in the
/// letterbox clamp to the nearest edge.
pub fn to_output(
    placement: &Placement,
    window_x: f32,
    window_y: f32,
    width: u32,
    height: u32,
) -> [f32; 2] {
    if placement.width == 0 || placement.height == 0 {
        return [0.0, 0.0];
    }
    let x = (window_x - placement.x as f32) * width as f32 / placement.width as f32;
    let y = (window_y - placement.y as f32) * height as f32 / placement.height as f32;
    [
        x.clamp(0.0, width.saturating_sub(1) as f32),
        y.clamp(0.0, height.saturating_sub(1) as f32),
    ]
}

/// CPU presentation: converts linear float pixels to `0x00RRGGBB`,
/// scales them nearest-neighbor into the target frame at their aspect
/// ratio, letterboxes with black, and returns the placement.
pub fn present(
    pixels: &[[f32; 4]],
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
    frame: &mut [u32],
) -> Placement {
    frame.fill(0);
    let placed = placement(width, height, target_width, target_height);
    if placed.width == 0 || placed.height == 0 {
        return placed;
    }
    for ty in 0..placed.height {
        let sy = (ty as u64 * height as u64 / placed.height as u64) as usize;
        let row = &pixels[sy * width as usize..(sy + 1) * width as usize];
        let target_row = (placed.y + ty) as usize * target_width as usize;
        for tx in 0..placed.width {
            let sx = (tx as u64 * width as u64 / placed.width as u64) as usize;
            let [r, g, b, _] = row[sx];
            let channel = |v: f32| u32::from(entelechy_formats::byte(v));
            frame[target_row + (placed.x + tx) as usize] =
                (channel(r) << 16) | (channel(g) << 8) | channel(b);
        }
    }
    placed
}

/// The browser's code for a key, by the key's name or the character it
/// typed (`key()` reads these; its help lists them): a letter or a digit
/// by its ASCII uppercase, a character typed with shift by the key under
/// it on a US keyboard, or a listed named key.
pub fn key_code(name: &str) -> Option<i32> {
    let mut chars = name.chars();
    if let (Some(c), None) = (chars.next(), chars.next()) {
        let upper = c.to_ascii_uppercase();
        return match upper {
            'A'..='Z' | '0'..='9' => Some(upper as i32),
            ' ' => Some(32),
            ')' => Some(48),
            '!' => Some(49),
            '@' => Some(50),
            '#' => Some(51),
            '$' => Some(52),
            '%' => Some(53),
            '^' => Some(54),
            '&' => Some(55),
            '*' => Some(56),
            '(' => Some(57),
            ';' | ':' => Some(186),
            '=' | '+' => Some(187),
            ',' | '<' => Some(188),
            '-' | '_' => Some(189),
            '.' | '>' => Some(190),
            '/' | '?' => Some(191),
            '`' | '~' => Some(192),
            '[' | '{' => Some(219),
            '\\' | '|' => Some(220),
            ']' | '}' => Some(221),
            '\'' | '"' => Some(222),
            _ => None,
        };
    }
    Some(match name {
        "Space" => 32,
        "Enter" => 13,
        "Escape" => 27,
        "Backspace" => 8,
        "Tab" => 9,
        "Shift" => 16,
        "Control" => 17,
        "Alt" => 18,
        "PageUp" => 33,
        "PageDown" => 34,
        "End" => 35,
        "Home" => 36,
        "ArrowLeft" => 37,
        "ArrowUp" => 38,
        "ArrowRight" => 39,
        "ArrowDown" => 40,
        "Delete" => 46,
        "Super" | "Meta" => 91,
        _ => return None,
    })
}

/// The showing device's state this frame.
#[derive(Clone, Copy, Debug)]
pub struct View {
    /// Window size in physical pixels; zero before a window exists.
    pub window: [f32; 2],
    /// Scale factor.
    pub scale: f32,
    /// Sound sample rate; zero without a sound device.
    pub rate: f32,
    /// The samples (per channel) the sound device holds and has not yet
    /// played (`queued()`); zero without one.
    pub queued: u32,
}

/// A frame to show.
pub struct Shown {
    /// The evaluation's report.
    pub report: Report,
    /// The picture: the graph's, or the node F6 previews.
    pub picture: Handle,
    /// The sound output, if any.
    pub sound: Option<Handle>,
}

/// What the pointer looks like over the window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PointerLook {
    /// The system's own pointer.
    System,
    /// The picture the scene gives `to pointer`: its size, its pixels as
    /// 8-bit RGBA from the top row down, and the point it points with
    /// (what it gives `to pointer_tip`, else its middle).
    Picture {
        /// Pixels across.
        width: u32,
        /// Pixels down.
        height: u32,
        /// Four bytes a pixel, not premultiplied.
        rgba: Vec<u8>,
        /// The pixel it points with.
        hotspot: (u32, u32),
    },
}

impl PointerLook {
    /// The pointer `image` is, pointing with `hotspot` (pixels across and
    /// down from its top left, held inside it; its middle when `None`), or
    /// why it cannot be one.
    pub fn of(image: &entelechy_kernel::Image, hotspot: Option<[f32; 2]>) -> Result<Self, String> {
        let (width, height) = (image.width(), image.height());
        if width > crate::POINTER_MOST || height > crate::POINTER_MOST {
            return Err(format!(
                "the picture `to pointer` is {width}x{height} pixels; a pointer is at most {most}x{most}",
                most = crate::POINTER_MOST
            ));
        }
        let rgba = image
            .pixels()
            .iter()
            .flat_map(|pixel| pixel.map(entelechy_formats::byte))
            .collect();
        let within = |at: f32, side: u32| {
            let most = side.saturating_sub(1) as f32;
            if at.is_finite() {
                at.floor().clamp(0.0, most) as u32
            } else {
                side / 2
            }
        };
        let hotspot = match hotspot {
            Some([x, y]) => (within(x, width), within(y, height)),
            None => (width / 2, height / 2),
        };
        Ok(Self::Picture {
            width,
            height,
            rgba,
            hotspot,
        })
    }
}

/// The pixel the first row of a hotspot output names, as two numbers
/// ([`Session::hotspot_of`]).
fn hotspot_in(data: &entelechy_kernel::FieldData) -> Option<[f32; 2]> {
    let words = data.storage();
    let scalars = data.schema().scalars();
    let number = |at: usize| match scalars.get(at)? {
        entelechy_kernel::Scalar::Float => Some(f32::from_bits(*words.get(at)?)),
        entelechy_kernel::Scalar::Int => Some(*words.get(at)? as i32 as f32),
        _ => None,
    };
    Some([number(0)?, number(1)?])
}

/// A frame's result.
pub enum Frame {
    /// A replay's next frame is not due; nothing was evaluated.
    Wait,
    /// A frame to show.
    Shown(Box<Shown>),
    /// The evaluation failed; the last picture stays, and
    /// [`Play::status`] says why.
    Failed,
}

/// A scene being played: input, preview, frame-data line, replay, and
/// the last picture placement, which maps the pointer.
pub struct Play {
    /// The host's clock, in seconds.
    clock: fn() -> f64,
    began: f64,
    /// The scene's time held still at this, while paused; frames go on
    /// under it, so the scene still takes the person's input.
    held: Option<f64>,
    input: InputState,
    pad: Pad,
    /// Presses since the last frame, and releases a frame has not seen
    /// pressed yet: every press lasts at least one frame.
    pressed: Vec<Held>,
    late: Vec<Held>,
    /// Characters typed and not yet read by a frame.
    typing: String,
    /// Pad buttons let go of, and when: a button pressed again within
    /// [`PAD_SETTLE`] was held all along.
    pad_released: Vec<(PadButton, f64)>,
    /// The pointer in window pixels.
    cursor: (f32, f32),
    placement: Placement,
    shown_size: (u32, u32),
    /// The node shown instead of the graph's picture.
    preview: Option<NodeId>,
    hud: bool,
    stats: hud::Stats,
    panel: Option<hud::Panel>,
    reported: Reporting,
    /// A replay and when its first frame was shown.
    #[cfg(feature = "documents")]
    replaying: Option<(Replayer, Option<f64>)>,
    /// How the last picture reached the window.
    path: String,
    downloads: u64,
    /// The open sound device's description.
    sound: Option<String>,
    status: String,
    /// Lines for the device to print or show.
    said: Vec<String>,
    /// When the disk was last looked at, `None` before the first frame.
    synced: Option<f64>,
    /// Whether a frame has been shown: before the first, a scene still
    /// waiting for its Kernels (prepared in turns, in a page) is waited for.
    shown: bool,
    /// The sound fetched behind the frame in flight, heard the frame after
    /// ([`Play::heard`]); a frame found failed lets it go unheard.
    sound_behind: Option<entelechy_kernel::Fetch>,
    /// The pointer's picture and hotspot last asked for, and the ones
    /// fetched behind the frame in flight ([`Play::pointer_look`]).
    pointer_wanted: Option<(Handle, Option<Handle>)>,
    pointer_behind: Option<(entelechy_kernel::Fetch, Option<entelechy_kernel::Fetch>)>,
    /// The look an interface was last told, and why the picture asked for
    /// cannot be one, said once.
    pointer_shown: PointerLook,
    pointer_refused: Option<String>,
}

/// How long a pad button's release waits, in seconds: a press within it
/// is the contacts chattering, the button still held.
const PAD_SETTLE: f64 = 0.04;

/// How often the disk is looked at, in seconds: changes to documents and
/// input files are found within it, and frames between do not stat them.
const SYNC_EVERY: f64 = 0.25;

/// How far one step moves the scene's held time, in seconds: a frame at
/// sixty a second.
pub const STEP: f64 = 1.0 / 60.0;

impl Play {
    /// A play timed by `clock`, the host's clock in seconds.
    pub fn new(clock: fn() -> f64) -> Self {
        Self {
            clock,
            began: clock(),
            held: None,
            input: InputState::default(),
            pad: Pad::default(),
            pressed: Vec::new(),
            late: Vec::new(),
            typing: String::new(),
            pad_released: Vec::new(),
            cursor: (0.0, 0.0),
            placement: Placement::default(),
            shown_size: (0, 0),
            preview: None,
            hud: false,
            stats: hud::Stats::default(),
            panel: None,
            reported: Reporting::default(),
            #[cfg(feature = "documents")]
            replaying: None,
            path: String::new(),
            downloads: 0,
            sound: None,
            status: "starting".into(),
            said: Vec::new(),
            synced: None,
            shown: false,
            sound_behind: None,
            pointer_wanted: None,
            pointer_behind: None,
            pointer_shown: PointerLook::System,
            pointer_refused: None,
        }
    }

    /// Plays `replay`: its inputs drive the scene at its times until it
    /// ends; then the person does.
    #[cfg(feature = "documents")]
    pub fn replay(&mut self, replay: &Replay, session: &mut Session) {
        self.replaying = Some((Replayer::start(replay, session), None));
        self.said
            .push("playing the replay: its inputs drive the scene until it ends".into());
    }

    /// Whether a replay is playing.
    pub fn replaying(&self) -> bool {
        #[cfg(feature = "documents")]
        {
            self.replaying.is_some()
        }
        #[cfg(not(feature = "documents"))]
        {
            false
        }
    }

    /// The scene's time: seconds since the play began, less the time it
    /// was paused; while paused, the time it is held at.
    pub fn time(&self) -> f64 {
        self.held.unwrap_or_else(|| (self.clock)() - self.began)
    }

    /// Whether the scene's time is held still.
    pub fn paused(&self) -> bool {
        self.held.is_some()
    }

    /// Holds the scene's time still, or lets it run on from where it was
    /// held. Frames go on while it is held: what the scene draws from its
    /// time stands still, and it still takes the person's input; a scene
    /// that steps every frame without reading its time steps on.
    pub fn pause(&mut self, paused: bool) {
        match (paused, self.held) {
            (true, None) => {
                let at = self.time();
                self.held = Some(at);
                self.said.push(format!("paused at {at:.3} s"));
            }
            (false, Some(at)) => {
                self.began = (self.clock)() - at;
                self.held = None;
                self.said.push(format!("playing on from {at:.3} s"));
            }
            _ => {}
        }
    }

    /// Moves the held time `frames` steps of [`STEP`] (back when below
    /// zero, never before the start), pausing first if it was running.
    pub fn step(&mut self, frames: i32) {
        let at = (self.time() + f64::from(frames) * STEP).max(0.0);
        self.held = Some(at);
        self.said.push(format!("paused at {at:.3} s"));
    }

    /// The context of the person's frame on `view`.
    pub fn context(&self, view: View) -> Context {
        Context {
            time: self.time(),
            rate: view.rate,
            queued: view.queued,
            input: self.input,
            window: view.window,
            scale: view.scale,
        }
    }

    /// A key by name (`a`, `ArrowLeft`, `F5`). F5 resets state, F6
    /// moves the preview, F10 toggles the frame-data line; F8 holds the
    /// scene's time still or lets it run, F9 steps it a frame on and F7 a
    /// frame back (while a replay plays, its inputs keep the time). Every
    /// other key goes to the scene, Tab and Escape among them.
    pub fn key(&mut self, session: &mut Session, name: &str, down: bool) {
        match name {
            "F7" | "F8" | "F9" => {
                if down && !self.replaying() {
                    match name {
                        "F8" => self.pause(!self.paused()),
                        "F9" => self.step(1),
                        _ => self.step(-1),
                    }
                }
                return;
            }
            _ => {}
        }
        match name {
            "F5" => {
                if down {
                    session.reset_state();
                    self.said.push("state reset".into());
                }
            }
            "F6" => {
                if down {
                    self.cycle_preview(session);
                }
            }
            "F10" => {
                if down {
                    self.hud = !self.hud;
                    self.panel = None;
                    // Its device time is clocked only while the line shows.
                    session.set_clocked(self.hud);
                }
            }
            _ => {
                if let Some(code) = key_code(name) {
                    self.hold(Held::Key(code), down);
                }
            }
        }
    }

    /// A press or release; a release before any frame saw the press
    /// takes effect after the next frame.
    fn hold(&mut self, held: Held, down: bool) {
        if down {
            self.late.retain(|late| *late != held);
            if !self.pressed.contains(&held) {
                self.pressed.push(held);
            }
            self.set(held, true);
        } else if self.pressed.contains(&held) {
            if !self.late.contains(&held) {
                self.late.push(held);
            }
        } else {
            self.set(held, false);
        }
    }

    fn set(&mut self, held: Held, down: bool) {
        match held {
            Held::Key(code) => self.input.set_key(code, down),
            Held::Button(0) => self.input.pointer_down = down,
            Held::Button(1) => self.input.pointer_right = down,
            Held::Button(2) => self.input.pointer_middle = down,
            Held::Button(_) => {}
        }
    }

    /// Every key let go: the window lost the focus, and the releases of
    /// the keys held go to whatever has it.
    pub fn release_keys(&mut self) {
        let keys: Vec<Held> = self
            .pressed
            .iter()
            .copied()
            .filter(|held| matches!(held, Held::Key(_)))
            .collect();
        self.pressed.retain(|held| !matches!(held, Held::Key(_)));
        self.late.retain(|held| !matches!(held, Held::Key(_)));
        for held in keys {
            self.set(held, false);
        }
        for code in 0..256 {
            self.input.set_key(code, false);
        }
    }

    /// The modifier keys the system says are held: Shift, Control, Alt
    /// and Command (or Windows) that are not are let go, whichever
    /// window their release went to.
    pub fn modifiers(&mut self, shift: bool, control: bool, alt: bool, command: bool) {
        for (code, held) in [(16, shift), (17, control), (18, alt), (91, command)] {
            if !held {
                self.pressed.retain(|h| *h != Held::Key(code));
                self.late.retain(|h| *h != Held::Key(code));
                self.input.set_key(code, false);
            }
        }
    }

    /// The pointer at `(x, y)` in window pixels.
    pub fn pointer(&mut self, x: f32, y: f32) {
        self.cursor = (x, y);
        self.update_pointer();
    }

    /// A pointer button, 0 left, 1 right, 2 middle.
    pub fn button(&mut self, which: u8, down: bool) {
        self.hold(Held::Button(which), down);
        self.update_pointer();
    }

    /// Characters typed (a key's text, or what an input method commits),
    /// what `typed()` reads: the next frame takes as many as it holds,
    /// the frames after the rest. Control characters (Enter, Tab,
    /// Backspace) and the private codes some systems give arrows and
    /// function keys are keys, which `key()` reads, not characters.
    pub fn typed(&mut self, text: &str) {
        self.typing.extend(
            text.chars()
                .filter(|c| !c.is_control() && !('\u{F700}'..='\u{F8FF}').contains(c)),
        );
    }

    /// Wheel lines, y positive away from the person; accumulated until
    /// the next frame.
    pub fn wheel(&mut self, x: f32, y: f32) {
        self.input.wheel[0] += x;
        self.input.wheel[1] += y;
    }

    /// A pad's button. A pad's contacts chatter: a release is held back
    /// [`PAD_SETTLE`], and a press within it was the button held.
    pub fn pad_button(&mut self, button: PadButton, down: bool) {
        self.pad_released
            .retain(|(released, _)| *released != button);
        if down {
            self.pad.button(&mut self.input, button, true);
        } else {
            self.pad_released.push((button, (self.clock)()));
        }
    }

    /// Lets go of the pad buttons released [`PAD_SETTLE`] ago and not
    /// pressed since.
    fn settle_pad(&mut self) {
        let now = (self.clock)();
        let (gone, held): (Vec<_>, Vec<_>) = std::mem::take(&mut self.pad_released)
            .into_iter()
            .partition(|(_, at)| now - at >= PAD_SETTLE);
        self.pad_released = held;
        for (button, _) in gone {
            self.pad.button(&mut self.input, button, false);
        }
    }

    /// A pad's axis.
    pub fn pad_axis(&mut self, axis: PadAxis, value: f32) {
        self.pad.axis(&mut self.input, axis, value);
    }

    /// How many pads are connected, what `pads()` reads.
    pub fn pads_connected(&mut self, count: u32) {
        self.input.pads = count;
    }

    /// Records the opened sound device's description.
    pub fn sound_opened(&mut self, description: impl Into<String>) {
        self.sound = Some(description.into());
    }

    /// One frame: syncs disk changes (every [`SYNC_EVERY`] seconds, every
    /// frame while a replay plays), takes a due replay frame, evaluates,
    /// and names the picture and sound.
    pub fn frame(&mut self, session: &mut Session, view: View) -> Frame {
        let begun = (self.clock)();
        let waited_before = session.device_waited();
        // Kernels waiting their turn (a page) are prepared by syncs: every
        // frame then, so the turns go on.
        let due = self.replaying()
            || session.preparing_left() > 0
            || self.synced.is_none_or(|at| begun - at >= SYNC_EVERY);
        if due {
            self.synced = Some(begun);
            self.said.extend(session.sync_lines());
        }
        // A first load still waiting for its Kernels has nothing to run
        // yet; once a frame is shown, a replacement waits behind the
        // version running.
        if !self.shown && session.scene_waiting() {
            return Frame::Wait;
        }
        // A window's frames overlap the device's: the next is made while
        // the device draws this one.
        if !session.overlaps()
            && let Err(error) = session.set_overlap(true)
        {
            self.failed(&session.told(&error), || None);
            return Frame::Failed;
        }
        // While a replay plays, it is the whole input.
        let replayed = match self.replay_frame(session) {
            Replayed::Wait => return Frame::Wait,
            Replayed::Frame(context) => Some(*context),
            Replayed::Not => None,
        };
        self.stats.begin(begun);
        self.settle_pad();
        let mut context = replayed.unwrap_or_else(|| self.context(view));
        if replayed.is_none() {
            self.typing = context.input.type_text(&self.typing).to_owned();
        }
        self.input.wheel = [0.0, 0.0];
        // Presses are seen; late releases apply from the next frame.
        self.pressed.clear();
        for held in std::mem::take(&mut self.late) {
            self.set(held, false);
        }
        // A replay reads what each frame left, so its frames are confirmed.
        let evaluated = session.evaluate(&context).and_then(|report| {
            if replayed.is_some() {
                session.confirm()?;
            }
            Ok(report)
        });
        match evaluated {
            Ok(report) => {
                #[cfg(feature = "documents")]
                if replayed.is_some()
                    && let Some((replayer, _)) = &mut self.replaying
                {
                    replayer.evaluated(report.clone());
                } else {
                    // A replay says its refusals with its outcome.
                    self.said
                        .extend(report.refused.iter().map(ToString::to_string));
                }
                #[cfg(not(feature = "documents"))]
                self.said
                    .extend(report.refused.iter().map(ToString::to_string));
                // A preview let go cold is made again to be shown.
                if let Some(node) = &self.preview {
                    let _ = session.warm(node, "result");
                }
                let picture = self
                    .preview
                    .as_ref()
                    .and_then(|node| session.cached(node, "result"))
                    .unwrap_or_else(|| displayed(session, &report));
                let sound = sound(session, &report);
                self.panel = self
                    .hud
                    .then(|| self.panel_for(session, &report, picture, &context));
                // The frame's own work on the processor: all of it but its
                // waits for the device, which are the device's time, not
                // its. The device's own time is its clock's, read for a
                // frame it has done (`set_clocked`).
                let waited = session
                    .device_waited()
                    .saturating_sub(waited_before)
                    .as_secs_f64();
                let working = (self.clock)() - begun - waited;
                self.stats.waited(
                    std::time::Duration::from_secs_f64(waited.max(0.0)),
                    &report.waits,
                );
                self.stats.record(
                    std::time::Duration::from_secs_f64(working.max(0.0)),
                    if self.hud {
                        session.frame_device_time()
                    } else {
                        None
                    },
                );
                if self.hud
                    && let Some(times) = session.device_times()
                {
                    self.stats.told(&times);
                }
                if self.reported.worked() {
                    self.said.push("evaluating again".into());
                }
                self.status = "running".into();
                self.shown = true;
                Frame::Shown(Box::new(Shown {
                    report,
                    picture,
                    sound,
                }))
            }
            Err(error) => {
                // Where in a Kernel it failed, traced again only when the
                // failure is news: a scene failing every frame is traced once.
                let told = session.told(&error);
                #[cfg(feature = "documents")]
                self.failed(&told, || {
                    session
                        .failure_place(&error, &context)
                        .map(|place| place.to_string())
                });
                #[cfg(not(feature = "documents"))]
                self.failed(&told, || None);
                Frame::Failed
            }
        }
    }

    /// Confirms the frame in flight before something other than the next
    /// frame reads the session (a request); a failed one is said as a
    /// failed frame is. Whether it was confirmed.
    pub fn confirm(&mut self, session: &mut Session) -> bool {
        match session.confirm() {
            Ok(()) => true,
            Err(error) => {
                self.failed(&session.told(&error), || None);
                false
            }
        }
    }

    /// The sound to hear now, as words, `sound` being the frame's sound
    /// output ([`Shown::sound`]). Overlapping, it is the frame before's,
    /// confirmed by now and fetched behind its own work alone, while this
    /// frame's is fetched for the next: hearing never waits for the frame
    /// just made. Otherwise this frame's, brought back now.
    pub fn heard(
        &mut self,
        session: &mut Session,
        sound: entelechy_kernel::Handle,
    ) -> Option<Vec<u32>> {
        if session.overlaps() {
            let fetch = session.fetch(sound).ok()?;
            return match self
                .sound_behind
                .replace(fetch)
                .map(entelechy_kernel::Fetch::take)
            {
                Some(Ok(data)) => Some(data.values().to_vec()),
                _ => None,
            };
        }
        session.field(sound).ok().map(|data| data.values().to_vec())
    }

    /// The pointer's look when it differs from the one an interface was
    /// last told, `report` being the frame's ([`Shown::report`]); `None`
    /// when it is the same. The scene's picture and hotspot are read only
    /// when one of their Fields is a new one: overlapping, they are fetched
    /// behind the frame's own work and looked at the frame after, as the
    /// sound is heard.
    pub fn pointer_look(&mut self, session: &mut Session, report: &Report) -> Option<PointerLook> {
        let wanted = session
            .pointer_of(report)
            .map(|picture| (picture, session.hotspot_of(report)));
        let fetched = self.pointer_behind.take().and_then(|(picture, hotspot)| {
            let picture = picture.take().ok()?;
            let hotspot = hotspot.and_then(|fetch| fetch.take().ok());
            Some((picture, hotspot))
        });
        let mut seen = match (wanted, fetched) {
            (None, _) => Some(PointerLook::System),
            (Some(_), Some((picture, hotspot))) => self.looked(
                picture.to_image().ok(),
                hotspot.as_ref().and_then(hotspot_in),
            ),
            (Some(_), None) => None,
        };
        if let Some((picture, hotspot)) = wanted
            && wanted != self.pointer_wanted
        {
            if session.overlaps() {
                self.pointer_behind = session
                    .fetch(picture)
                    .ok()
                    .map(|fetch| (fetch, hotspot.and_then(|handle| session.fetch(handle).ok())));
            } else {
                let point = hotspot
                    .and_then(|handle| session.field(handle).ok())
                    .and_then(hotspot_in);
                seen = self.looked(session.picture(picture).ok(), point);
            }
        }
        self.pointer_wanted = wanted;
        let seen = seen?;
        if seen == self.pointer_shown {
            return None;
        }
        self.pointer_shown = seen.clone();
        Some(seen)
    }

    /// The look a fetched picture gives the pointer, pointing with
    /// `hotspot`; one that cannot be one is said once and leaves the
    /// system's.
    fn looked(
        &mut self,
        image: Option<entelechy_kernel::Image>,
        hotspot: Option<[f32; 2]>,
    ) -> Option<PointerLook> {
        match PointerLook::of(&image?, hotspot) {
            Ok(look) => {
                self.pointer_refused = None;
                Some(look)
            }
            Err(why) => {
                if self.pointer_refused.as_ref() != Some(&why) {
                    self.said.push(why.clone());
                    self.pointer_refused = Some(why);
                }
                Some(PointerLook::System)
            }
        }
    }

    /// Says a failed evaluation once, and keeps it as the status. The
    /// sound and the pointer fetched behind the frame before are not
    /// taken: that frame is taken back with them.
    fn failed(&mut self, told: &str, place: impl FnOnce() -> Option<String>) {
        self.sound_behind = None;
        self.pointer_behind = None;
        self.pointer_wanted = None;
        let message = format!("evaluation failed: {told}");
        if self.reported.failed(&message) {
            self.said.push(format!("{message}; last frame is kept"));
            if let Some(place) = place() {
                self.said
                    .push(format!("  the failing operation is written at {place}"));
            }
        }
        self.status = message;
    }

    /// Records how the device showed the picture: placement (`None`:
    /// unchanged), size, path (`presented on the device`, `downloaded
    /// and blitted`), whether it downloaded, and seconds taken.
    pub fn placed(
        &mut self,
        placement: Option<Placement>,
        size: (u32, u32),
        path: &str,
        downloaded: bool,
        presenting: f64,
    ) {
        self.landed(placement, size);
        path.clone_into(&mut self.path);
        if downloaded {
            self.downloads += 1;
        }
        self.stats
            .presented(std::time::Duration::from_secs_f64(presenting.max(0.0)));
    }

    /// Records the picture's placement (`None`: unchanged) and size,
    /// which map the pointer.
    pub fn landed(&mut self, placement: Option<Placement>, size: (u32, u32)) {
        if let Some(placement) = placement {
            self.placement = placement;
        }
        self.shown_size = size;
        self.update_pointer();
    }

    /// The frame-data line, when F10 shows it.
    pub fn panel(&self) -> Option<&hud::Panel> {
        self.panel.as_ref()
    }

    /// `running`, a replay's time, or the failure that kept the last
    /// picture.
    pub fn status(&self) -> &str {
        &self.status
    }

    /// Adds a line from the interface, such as a skipped backend.
    pub fn say(&mut self, line: String) {
        self.said.push(line);
    }

    /// Lines since the last call: events, failures, the preview, a
    /// replay's outcome.
    pub fn said(&mut self) -> Vec<String> {
        std::mem::take(&mut self.said)
    }

    fn update_pointer(&mut self) {
        self.input.pointer = to_output(
            &self.placement,
            self.cursor.0,
            self.cursor.1,
            self.shown_size.0,
            self.shown_size.1,
        );
    }

    /// What the window shows, as the line of frame data names it: the
    /// previewed node's port, or the scene output `shown` is.
    pub fn showing(&self, report: &Report, shown: Handle) -> String {
        match &self.preview {
            Some(node) => format!("view {node}.result"),
            None => match report.outputs.iter().find(|(_, handle)| *handle == shown) {
                Some((name, _)) => format!("output {name}"),
                None => "output".to_string(),
            },
        }
    }

    /// Moves the preview to the next node, or to the graph output after
    /// the last. A previewed node is pinned, so fusion keeps it.
    fn cycle_preview(&mut self, session: &mut Session) {
        let nodes: Vec<NodeId> = session
            .graph()
            .map(|graph| graph.nodes.iter().map(|node| node.id.clone()).collect())
            .unwrap_or_default();
        if let Some(current) = self.preview.take() {
            session.unpin(&current);
            let position = nodes.iter().position(|node| *node == current);
            if let Some(next) = position.and_then(|p| nodes.get(p + 1)) {
                self.preview = Some(next.clone());
            }
        } else {
            self.preview = nodes.first().cloned();
        }
        if let Some(node) = &self.preview {
            session.pin(node.clone());
        }
        let view = self
            .preview
            .as_ref()
            .map_or("output".to_string(), ToString::to_string);
        self.said.push(format!("previewing {view}"));
    }

    /// The replay's next due frame. After the last, the outcome is said
    /// and the person has the scene.
    #[cfg(feature = "documents")]
    fn replay_frame(&mut self, session: &mut Session) -> Replayed {
        let now = (self.clock)();
        let Some((replayer, since)) = &mut self.replaying else {
            return Replayed::Not;
        };
        let clock = now - *since.get_or_insert(now);
        if replayer.next_time().is_some_and(|time| time > clock) {
            return Replayed::Wait;
        }
        match replayer.next_frame(session) {
            Ok(Some(frame)) => {
                self.status = format!("replaying, {:.2} s", frame.context.time);
                Replayed::Frame(Box::new(frame.context))
            }
            Ok(None) => {
                let outcome = &replayer.outcome;
                let head = if outcome.failures.is_empty() {
                    format!(
                        "replay: ok ({} frames, {} checks)",
                        outcome.frames, outcome.checks
                    )
                } else {
                    format!(
                        "replay: {} of {} checks failed",
                        outcome.failures.len(),
                        outcome.checks
                    )
                };
                let said = std::iter::once(head)
                    .chain(outcome.lines())
                    .collect::<Vec<_>>()
                    .join("\n  ");
                self.said.push(format!("{said}; the scene is yours"));
                self.replaying = None;
                Replayed::Not
            }
            Err(error) => {
                self.said
                    .push(format!("replay stopped: {error}; the scene is yours"));
                self.replaying = None;
                Replayed::Not
            }
        }
    }

    #[cfg(not(feature = "documents"))]
    fn replay_frame(&mut self, _session: &mut Session) -> Replayed {
        Replayed::Not
    }

    /// The F10 panel: groups laid left to right as the window has room,
    /// each number in a fixed width, made again four times a second
    /// (`hud::Stats::held`), the panel never narrowing: the smoothed
    /// times; backend, window and the image shown; what ran; the Host's
    /// memory and what a frame let go to make room; presenting; then the
    /// nodes that kept the device busiest.
    fn panel_for(
        &mut self,
        session: &Session,
        report: &Report,
        shown: Handle,
        context: &Context,
    ) -> hud::Panel {
        let host = session.runtime().host();
        let (width, height) = host.image_size(shown).unwrap_or((0, 0));
        let fused: usize = report
            .fused
            .iter()
            .map(|(_, absorbed)| absorbed.len())
            .sum();
        let view = self.showing(report, shown);
        let sound = match &self.sound {
            Some(sound) => format!(" | sound {sound}"),
            None => String::new(),
        };
        // Nodes routed off the main backend, and the bytes carried.
        let routed: Vec<&entelechy_graph::Placed> = report
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
                " | off {} {:>3} ({})",
                routed[0].backend,
                routed.len(),
                crate::bench::bytes_text(carried)
            )
        };
        let mib = |bytes: u64| format!("{:>6.1} MiB", bytes as f64 / (1024.0 * 1024.0));
        let used = host.used_words() as u64 * 4;
        let budget = host.budget_words() as u64 * 4;
        let share = if budget > 0 { used * 100 / budget } else { 0 };
        let groups = [
            format!(
                "{}{off} | window {}x{} x{:.0} | {view} {width}x{height}",
                session.runtime().backend_name(),
                context.window[0],
                context.window[1],
                context.scale,
            ),
            format!(
                "ran {:>4} reused {:>4} fused {fused:>3} | async {:>2}",
                report.executed.len(),
                report.reused.len(),
                report.pending.len(),
            ),
            format!(
                "mem {} / {} ({share:>3}%){}",
                mib(used),
                mib(budget),
                if report.let_go > 0 {
                    format!(" let go {:>3} past the budget", report.let_go)
                } else {
                    String::new()
                }
            ),
            format!("{} | downloads {:>5}{sound}", self.path, self.downloads),
            format!(
                "t {:>8.1} s{}",
                context.time,
                if self.paused() { " paused" } else { "" }
            ),
        ];
        // As many groups to a line as the window has room for.
        let scale = context.scale.round().max(1.0) as u32;
        let columns = ((context.window[0] * context.scale) as u32 / (6 * scale)).max(40) as usize;
        let said = self.stats.held(|stats| {
            let mut first = vec![stats.times()];
            first.extend(groups);
            let mut said = hud::flow(&first, columns);
            said.extend(hud::flow(&stats.waiting(8), columns));
            said.extend(hud::flow(&stats.busiest(12), columns));
            said
        });
        hud::lines_at_least(&said, scale, self.stats.widest())
    }
}

/// A button (0 left, 1 right, 2 middle) or a key, by its code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Held {
    Button(u8),
    Key(i32),
}

/// A replay's answer for the next frame.
#[cfg_attr(not(feature = "documents"), allow(dead_code))]
enum Replayed {
    Wait,
    Frame(Box<Context>),
    Not,
}

/// The scene's picture ([`Session::picture_of`]), else the graph's
/// first output.
fn displayed(session: &Session, report: &Report) -> Handle {
    session.picture_of(report).unwrap_or(report.output)
}

/// The output a sound device plays ([`Session::sound_of`]).
fn sound(session: &Session, report: &Report) -> Option<Handle> {
    session.sound_of(report).map(|(handle, _)| handle)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NOW: AtomicU64 = AtomicU64::new(0);

    fn clock() -> f64 {
        f64::from_bits(NOW.load(Ordering::SeqCst))
    }

    fn at(seconds: f64) {
        NOW.store(seconds.to_bits(), Ordering::SeqCst);
    }

    /// A button's contacts chattering, released and pressed again within
    /// the settling time, is the button held; a release that settles lets
    /// go of it.
    #[test]
    fn a_chattering_pad_button_is_held_once() {
        use super::{PAD_SETTLE, PadButton, Play};
        let select = entelechy_kernel::pad::SELECT;
        at(10.0);
        let mut play = Play::new(clock);
        play.pad_button(PadButton::Select, true);
        play.settle_pad();
        assert!(play.input.button(select));
        at(10.016);
        play.pad_button(PadButton::Select, false);
        at(10.020);
        play.settle_pad();
        assert!(play.input.button(select), "a release not yet settled");
        play.pad_button(PadButton::Select, true);
        at(10.024);
        play.pad_button(PadButton::Select, false);
        at(10.024 + PAD_SETTLE * 0.5);
        play.settle_pad();
        assert!(
            play.input.button(select),
            "pressed again within the settling time"
        );
        at(10.024 + PAD_SETTLE + 0.001);
        play.settle_pad();
        assert!(!play.input.button(select), "a release settled lets go");
    }

    /// Failures differing only in numbers are one run, reported once.
    #[test]
    fn a_run_of_one_failure_is_printed_once() {
        let side = |v: &str| {
            format!(
                "evaluation failed: node `canvas`: `domain` resolves to window() / 2, whose sides must be positive whole numbers, found {v}"
            )
        };
        let mut reporting = super::Reporting::default();
        assert!(reporting.failed(&side("953.5")), "the first one is news");
        for size in ["945.5", "939.5", "937.5", "933.5"] {
            assert!(!reporting.failed(&side(size)), "{size} is the same failure");
        }
        // Another node, or another fault, is another failure.
        assert!(reporting.failed(&side("933.5").replace("canvas", "heat")));
        assert!(reporting.failed("evaluation failed: node `bodies`: integer overflow"));

        // A working frame ends the run once; the failure is then new.
        assert!(reporting.worked());
        assert!(!reporting.worked(), "nothing was failing");
        assert!(reporting.failed(&side("953.5")));
    }

    use super::*;

    #[test]
    fn letterboxes_and_scales_without_distortion() {
        let pixels = vec![[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]];
        let mut frame = vec![0xFFFF_FFFF; 8 * 2];
        let placed = present(&pixels, 2, 1, 8, 2, &mut frame);
        assert_eq!(
            placed,
            Placement {
                x: 2,
                y: 0,
                width: 4,
                height: 2
            }
        );
        assert_eq!(frame[0], 0, "left letterbox column is black");
        assert_eq!(frame[2], 0x00FF_0000);
        assert_eq!(frame[4], 0x0000_00FF);
        assert_eq!(frame[10], 0x00FF_0000, "second row is a scaled copy");
        let mut frame = vec![0xFFFF_FFFF; 4 * 4];
        present(&pixels, 2, 1, 4, 4, &mut frame);
        assert_eq!(frame[0], 0, "top letterbox row is black");
        assert_eq!(frame[4], 0x00FF_0000);
        assert_eq!(frame[7], 0x0000_00FF);
        assert_eq!(frame[12], 0, "bottom letterbox row is black");
    }

    #[test]
    fn window_positions_map_to_output_pixels() {
        let placed = Placement {
            x: 2,
            y: 0,
            width: 4,
            height: 2,
        };
        assert_eq!(to_output(&placed, 2.0, 0.0, 2, 1), [0.0, 0.0]);
        assert_eq!(to_output(&placed, 4.0, 1.0, 2, 1), [1.0, 0.0]);
        assert_eq!(
            to_output(&placed, 0.0, 5.0, 2, 1),
            [0.0, 0.0],
            "letterbox clamps"
        );
        assert_eq!(key_code("a"), Some(65));
        assert_eq!(key_code("ArrowLeft"), Some(37));
        assert_eq!(key_code("F5"), None);
        // A character typed with shift is the key under it.
        assert_eq!(key_code(";"), Some(186));
        assert_eq!(key_code(":"), Some(186));
        assert_eq!(key_code("("), Some(57));
        assert_eq!(key_code("{"), key_code("["));
        assert_eq!(key_code("\""), Some(222));
        assert_eq!(key_code("\\"), Some(220));
        assert_eq!(key_code("Delete"), Some(46));
        assert_eq!(key_code("Super"), Some(91));
    }
}
