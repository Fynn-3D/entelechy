//! The Player's frame, with no window: what every interface that plays
//! a scene does the same way, whatever its devices. A window, a page's
//! canvas and anything else that shows a scene hand their person's keys,
//! pointer, wheel and pads here, ask for a frame, show the picture it
//! names on their own device and play its sound on theirs, and print
//! what it says. What F5, Tab and F10 mean, how a pad is a keyboard,
//! which output is the picture and which the sound, how a replay drives
//! the scene, and what the line of frame data says are decided once,
//! here.

pub mod hud;
pub mod pad;

use orla_graph::{NodeId, Report};
use orla_runtime::{Context, Handle, InputState};

pub use orla_gpu::present::{Placement, placement};
pub use pad::{Pad, PadAxis, PadButton};

use crate::Session;
use crate::replay::{Replay, Replayer};

/// What a run of failures says on the way past.
///
/// A scene that cannot evaluate says so once a frame. Dragging a
/// window's corner fails every frame, each one naming the size the
/// window was, so the terminal fills with the same failure wearing
/// different numbers. [`Reporting`] prints a failure's first frame,
/// stays quiet while it keeps happening, and says when evaluating comes
/// back.
#[derive(Default)]
pub struct Reporting {
    shape: String,
}

impl Reporting {
    /// Whether this failure is worth printing: a new one, or one of
    /// another shape than the one still happening.
    pub fn failed(&mut self, message: &str) -> bool {
        let shape = failure_shape(message);
        if self.shape == shape {
            return false;
        }
        self.shape = shape;
        true
    }

    /// Whether a frame that worked ends a run of failures, and so is
    /// worth a line of its own.
    pub fn worked(&mut self) -> bool {
        let was_failing = !self.shape.is_empty();
        self.shape.clear();
        was_failing
    }
}

/// A failure's shape, with the numbers in it taken out: two failures
/// that differ only in a size, a count or an element are one failure.
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

/// Converts linear float pixels to `0x00RRGGBB` and scales them with
/// nearest-neighbour sampling into a `target_width x target_height` frame,
/// preserving aspect ratio and letterboxing with black. Returns where the
/// image was placed. This is the CPU path; the GPU backend presents
/// resident images without it.
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
            let channel = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u32;
            frame[target_row + (placed.x + tx) as usize] =
                (channel(r) << 16) | (channel(g) << 8) | channel(b);
        }
    }
    placed
}

/// Browser-style key code for a key name as winit reports it: single
/// characters by ASCII uppercase, plus the named keys Kernels commonly use.
pub fn key_code(name: &str) -> Option<i32> {
    let mut chars = name.chars();
    if let (Some(c), None) = (chars.next(), chars.next()) {
        let upper = c.to_ascii_uppercase();
        return match upper {
            'A'..='Z' | '0'..='9' => Some(upper as i32),
            ' ' => Some(32),
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
        "ArrowLeft" => 37,
        "ArrowUp" => 38,
        "ArrowRight" => 39,
        "ArrowDown" => 40,
        _ => return None,
    })
}

/// What the device showing the scene says about itself this frame.
#[derive(Clone, Copy, Debug)]
pub struct View {
    /// The window's size in physical pixels; zero by zero before there
    /// is one.
    pub window: [f32; 2],
    /// Its scale factor.
    pub scale: f32,
    /// The sound device's sample rate, zero without one.
    pub rate: f32,
}

/// A frame the play made: the evaluation's report, the picture to show
/// (the graph's picture, or the node Tab previews) and the sound output,
/// when the scene has one.
pub struct Shown {
    /// What the evaluation did.
    pub report: Report,
    /// The Field to show.
    pub picture: Handle,
    /// The first output that is rows of one or two floats.
    pub sound: Option<Handle>,
}

/// What a frame came to.
pub enum Frame {
    /// A replay's next frame's time has not come; nothing was evaluated.
    Wait,
    /// A frame to show.
    Shown(Box<Shown>),
    /// The evaluation failed; the last picture stays, and
    /// [`Play::status`] says why.
    Failed,
}

/// What a key asks of the device beyond the scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Asked {
    /// Escape: stop playing (a window closes).
    Stop,
}

/// A scene played: the person's input, the preview, the line of frame
/// data, the replay being played, and what the device last did with a
/// picture, which is how the pointer is found in it.
pub struct Play {
    /// The host's clock, in seconds from any start.
    clock: fn() -> f64,
    /// When this play began, by that clock.
    began: f64,
    input: InputState,
    pad: Pad,
    /// The pointer in window pixels, as the device last said.
    cursor: (f32, f32),
    /// Where the picture landed in the window, and its size.
    placement: Placement,
    shown_size: (u32, u32),
    /// The node shown instead of the graph's picture.
    preview: Option<NodeId>,
    hud: bool,
    stats: hud::Stats,
    panel: Option<hud::Panel>,
    reported: Reporting,
    /// A replay being played, and when its first frame was shown.
    replaying: Option<(Replayer, Option<f64>)>,
    /// How the last picture reached the window, as the device said.
    path: String,
    downloads: u64,
    /// The sound device, as the device describes it, when one is open.
    sound: Option<String>,
    status: String,
    /// What is to be said: lines for the device to print or show.
    said: Vec<String>,
}

impl Play {
    /// A play whose time is `clock`'s: the host's clock, in seconds.
    pub fn new(clock: fn() -> f64) -> Self {
        Self {
            clock,
            began: clock(),
            input: InputState::default(),
            pad: Pad::default(),
            cursor: (0.0, 0.0),
            placement: Placement::default(),
            shown_size: (0, 0),
            preview: None,
            hud: false,
            stats: hud::Stats::default(),
            panel: None,
            reported: Reporting::default(),
            replaying: None,
            path: String::new(),
            downloads: 0,
            sound: None,
            status: "starting".into(),
            said: Vec::new(),
        }
    }

    /// Plays `replay` over `session`: its inputs drive the scene at its
    /// own times until it ends, and then the person does.
    pub fn replay(&mut self, replay: &Replay, session: &mut Session) {
        self.replaying = Some((Replayer::start(replay, session), None));
        self.said
            .push("playing the replay: its inputs drive the scene until it ends".into());
    }

    /// Whether a replay is playing.
    pub fn replaying(&self) -> bool {
        self.replaying.is_some()
    }

    /// Seconds since the play began.
    pub fn time(&self) -> f32 {
        ((self.clock)() - self.began) as f32
    }

    /// The context of the person's frame on `view`.
    pub fn context(&self, view: View) -> Context {
        Context {
            time: self.time(),
            rate: view.rate,
            input: self.input,
            window: view.window,
            scale: view.scale,
        }
    }

    /// A key, by its name as the device reports it (`a`, `ArrowLeft`,
    /// `F5`), pressed or released. F5 resets the scene's state, Tab moves
    /// the preview, F10 shows or hides the line of frame data, and Escape
    /// asks the device to stop; every other key goes to the scene.
    pub fn key(&mut self, session: &mut Session, name: &str, down: bool) -> Option<Asked> {
        match name {
            "Escape" => return Some(Asked::Stop),
            "F5" => {
                if down {
                    session.reset_state();
                    self.said.push("state reset".into());
                }
            }
            "Tab" => {
                if down {
                    self.cycle_preview(session);
                }
            }
            "F10" => {
                if down {
                    self.hud = !self.hud;
                    self.panel = None;
                }
            }
            _ => {
                if let Some(code) = key_code(name) {
                    self.input.set_key(code, down);
                }
            }
        }
        None
    }

    /// The pointer at `(x, y)` in window pixels.
    pub fn pointer(&mut self, x: f32, y: f32) {
        self.cursor = (x, y);
        self.update_pointer();
    }

    /// A pointer button, 0 left, 1 right, 2 middle.
    pub fn button(&mut self, which: u8, down: bool) {
        match which {
            0 => self.input.pointer_down = down,
            1 => self.input.pointer_right = down,
            2 => self.input.pointer_middle = down,
            _ => {}
        }
        self.update_pointer();
    }

    /// The wheel turned by `(x, y)` lines, y positive away from the
    /// person; it adds up until the next frame reads it.
    pub fn wheel(&mut self, x: f32, y: f32) {
        self.input.wheel[0] += x;
        self.input.wheel[1] += y;
    }

    /// A pad's button.
    pub fn pad_button(&mut self, button: PadButton, down: bool) {
        self.pad.button(&mut self.input, button, down);
    }

    /// A pad's axis.
    pub fn pad_axis(&mut self, axis: PadAxis, value: f32) {
        self.pad.axis(&mut self.input, axis, value);
    }

    /// The sound device, as the device describes it, once it is open.
    pub fn sound_opened(&mut self, description: impl Into<String>) {
        self.sound = Some(description.into());
    }

    /// One frame: what changed on disk is read, a replay's frame is
    /// taken when its time has come, the scene is evaluated, and the
    /// picture and the sound are named for the device.
    pub fn frame(&mut self, session: &mut Session, view: View) -> Frame {
        for event in session.sync() {
            self.said.push(event.to_string());
        }
        // A replay, while it plays, is the input: its frame, at its time,
        // under its pointer, keys and window; the person's hands wait.
        let replayed = match self.replay_frame(session) {
            Replayed::Wait => return Frame::Wait,
            Replayed::Frame(context) => Some(context),
            Replayed::Not => None,
        };
        self.stats.begin((self.clock)());
        let started = (self.clock)();
        let context = replayed.unwrap_or_else(|| self.context(view));
        self.input.wheel = [0.0, 0.0];
        match session.evaluate(&context) {
            Ok(report) => {
                if replayed.is_some()
                    && let Some((replayer, _)) = &mut self.replaying
                {
                    replayer.evaluated(report.clone());
                }
                let evaluated = (self.clock)() - started;
                // The picture is the first output that is a grid of
                // pixels, the sound the first that is rows of one or two
                // floats: both go by structure, never by name.
                let picture = self
                    .preview
                    .as_ref()
                    .and_then(|node| session.cached(node, "result"))
                    .unwrap_or_else(|| displayed(session, &report));
                let sound = sound(session, &report);
                self.panel = self
                    .hud
                    .then(|| self.panel_for(session, &report, picture, &context));
                // With the line on, the frame waits for the device to
                // finish what evaluating gave it, so what it took is a
                // number of its own rather than a share of presenting.
                let device = (self.hud && session.runtime().has_device()).then(|| {
                    let waiting = (self.clock)();
                    session.runtime().settle_device();
                    (self.clock)() - waiting
                });
                self.stats.record(
                    std::time::Duration::from_secs_f64(evaluated.max(0.0)),
                    device.map(|d| std::time::Duration::from_secs_f64(d.max(0.0))),
                );
                if self.reported.worked() {
                    self.said.push("evaluating again".into());
                }
                self.status = "running".into();
                Frame::Shown(Box::new(Shown {
                    report,
                    picture,
                    sound,
                }))
            }
            Err(error) => {
                let message = format!("evaluation failed: {error}");
                // One failure a frame, each naming the size the window
                // was, is one failure.
                if self.reported.failed(&message) {
                    self.said.push(format!("{message}; last frame is kept"));
                }
                self.status = message;
                Frame::Failed
            }
        }
    }

    /// What the device did with the frame's picture: where in the window
    /// it landed (`None`: where it was), its size, how it got there (`presented on the device`,
    /// `downloaded and blitted`), whether that took a download, and how
    /// long it took, in seconds. The pointer is found through where it
    /// landed.
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

    /// Where the picture landed in the window, when the device knows
    /// (`None`: where it was), and its size: how the pointer is found in
    /// it.
    pub fn landed(&mut self, placement: Option<Placement>, size: (u32, u32)) {
        if let Some(placement) = placement {
            self.placement = placement;
        }
        self.shown_size = size;
        self.update_pointer();
    }

    /// The line of frame data, when F10 has it shown.
    pub fn panel(&self) -> Option<&hud::Panel> {
        self.panel.as_ref()
    }

    /// Why the last frame is what it is: `running`, a replay's time, or
    /// the failure that kept the last picture.
    pub fn status(&self) -> &str {
        &self.status
    }

    /// What there is to say since the last call: events, failures, the
    /// preview, a replay's outcome. A device prints them, or shows them.
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

    /// Moves the preview to the next node, or back to the graph output
    /// after the last one. Previewed nodes are pinned so fusion keeps them
    /// materialized.
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

    /// The replay's next frame, once its time has come. Past its last
    /// frame what it observed is said, and the person has the scene.
    fn replay_frame(&mut self, session: &mut Session) -> Replayed {
        let now = (self.clock)();
        let Some((replayer, since)) = &mut self.replaying else {
            return Replayed::Not;
        };
        let clock = (now - *since.get_or_insert(now)) as f32;
        if replayer.next_time().is_some_and(|time| time > clock) {
            return Replayed::Wait;
        }
        match replayer.next_frame(session) {
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

    /// The F10 line for this frame: the smoothed times, then the backend,
    /// the window, the shown image, what ran, and what waits off the frame.
    fn panel_for(
        &self,
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
        let view = match &self.preview {
            Some(node) => format!("view {node}"),
            None => "output".to_string(),
        };
        let sound = match &self.sound {
            Some(sound) => format!(" | sound {sound}"),
            None => String::new(),
        };
        // The nodes that left the main backend this frame, and what the
        // trip carried.
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
                crate::bench::bytes_text(carried)
            )
        };
        let rest = format!(
            "{}{off} | window {}x{} x{:.0} | {view} {width}x{height} | ran {} reused {} fused {fused} | async {} | {} | downloads {}{sound} | t {:.1} s",
            session.runtime().backend_name(),
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
}

/// What a replay says for the frame about to be made.
enum Replayed {
    Wait,
    Frame(Context),
    Not,
}

/// The output a device shows: the first grid of pixels, or the graph's
/// first output when there is none.
fn displayed(session: &Session, report: &Report) -> Handle {
    let host = session.runtime().host();
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

/// The output a sound device plays: the first that is rows of one float
/// (mono) or two (stereo).
fn sound(session: &Session, report: &Report) -> Option<Handle> {
    let host = session.runtime().host();
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
}

#[cfg(test)]
mod tests {
    /// A hand on the window's corner: one failure a frame, each naming
    /// the size the window was.
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

        // A frame that works ends the run, and says so once; the same
        // failure after it is news again.
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
    }
}
