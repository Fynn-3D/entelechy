//! Declarative replay: drive a scene through a scripted sequence of frames
//! and inputs and check pixels, so an application's tests are data rather
//! than Rust.
//!
//! ```text
//! scene ../scene.osn        # required; paths are relative to the replay file
//! library ../kernels/       # optional extra Kernels beyond the scene's `use`
//! frame 0.0                 # evaluate at this session time
//! pointer 20 20 down        # move the pointer and press; `up` releases
//! pointer up                # release without moving
//! button right down         # a pointer button: left, right, middle; `up` releases
//! wheel 0 -1                # what the wheel scrolls before the next frame only
//! window 240 135            # the window's size in pixels, for `window()`,
//! window 240 135 0.25       # and its scale factor (one when not given)
//! key 67 down               # press a key by code; `up` releases
//! reset                     # forget committed state
//! param exposure 1.5        # override a graph parameter (a literal)
//! expect 11 10 = 1.0 1.0 1.0        # r g b of the last output, tolerance 0.01
//! expect 11 10 = 1.0 1.0 1.0 0.05   # explicit tolerance
//! expect step.result[0].score = 1   # a field of an element of a node's output:
//!                                   # ints and bools exact, floats and vectors
//!                                   # with an optional trailing tolerance
//! expect fall.result[0].speed > 2.5 # or on one side of a bound, every
//!                                   # component: `<`, `<=`, `>`, `>=`, strict
//!                                   # (no tolerance), numbers only
//! expect count ground.result = 6889 # how many elements a node's output holds,
//!                                   # which is what a `domain` came out as
//! expect placed ring main           # the node runs on the session's main
//! expect placed filter ordered      # backend, or off it (a scan on a GPU
//!                                   # session): the same line holds on every
//!                                   # backend where nothing is routed
//! ```
//!
//! An element expectation names the node's output as the Kernel declared
//! it, so it reads as a statement about the application's state rather
//! than about the picture. A count expectation names the output alone:
//! it is how a node whose size a Kernel decided is written down. Nodes
//! named either way are pinned before the first frame, so fusion never
//! hides them.
//!
//! A frame evaluates only what its expectations need: the named nodes,
//! the state, and the picture only when a pixel expectation follows the
//! frame. A replay that asserts on state never renders, and costs what
//! the state costs.

use std::fmt;
use std::path::{Path, PathBuf};

use orla_graph::Report;
use orla_runtime::{Context, Handle, InputState, Runtime};

use crate::Session;

/// One scripted step.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    /// Evaluate at a session time.
    Frame(f32),
    /// Wait for every `async` node's run off the frame to land, so the
    /// next frame reads it.
    Settle,
    /// Move the pointer and set the button.
    Pointer {
        /// New position, or keep the current one.
        position: Option<[f32; 2]>,
        /// Button state.
        down: bool,
    },
    /// Press or release a key code.
    Key {
        /// Key code.
        code: i32,
        /// Pressed or released.
        down: bool,
    },
    /// Press or release a pointer button by number: 0 left, 1 right, 2 middle.
    Button {
        /// Button number.
        code: i32,
        /// Pressed or released.
        down: bool,
    },
    /// What the wheel scrolls for the next frame.
    Wheel([f32; 2]),
    /// Where a pad's axis pair stands: 0 the left stick, 1 the right, 2
    /// the two triggers.
    Stick {
        /// Which pair.
        which: i32,
        /// Its two values.
        at: [f32; 2],
    },
    /// The window's size in pixels and its scale factor from here on.
    Window {
        /// Width and height in pixels.
        size: [f32; 2],
        /// Physical pixels per logical pixel.
        scale: f32,
    },
    /// Forget committed state.
    Reset,
    /// Forget one port's committed state; the rest keeps its own.
    ResetPort {
        /// The producer.
        node: String,
        /// Its port.
        port: String,
    },
    /// Override a graph parameter.
    Param {
        /// Parameter name.
        name: String,
        /// The value.
        value: orla_runtime::Value,
    },
    /// Check a field of an element of a node's output after the last frame.
    ExpectElement {
        /// The node.
        node: String,
        /// Its output port.
        port: String,
        /// Element index.
        index: u32,
        /// Struct fields to follow, outermost first; empty for the element
        /// itself.
        path: Vec<String>,
        /// How the value is held to what is written: equal (within a
        /// tolerance for floats), or on one side of it.
        compare: Compare,
        /// Expected components as written.
        values: Vec<String>,
        /// Line number, for messages.
        line: u32,
    },
    /// Check how many elements a node's output holds after the last
    /// frame: what a node whose `domain` a Kernel decided came out as.
    ExpectCount {
        /// The node.
        node: String,
        /// Its output port.
        port: String,
        /// Expected element count.
        count: usize,
        /// Line number, for messages.
        line: u32,
    },
    /// Check where a node runs: on the session's main backend, or off
    /// it on the ordered one. A session without an ordered backend runs
    /// everything on the main one, so `ordered` holds only where a second
    /// backend exists (a GPU session); a replay run on the CPU or the JIT
    /// checks it as `main`.
    ExpectPlaced {
        /// The node.
        node: String,
        /// Whether it should run off the main backend.
        routed: bool,
        /// Line number, for messages.
        line: u32,
    },
    /// Check a pixel of the last frame.
    Expect {
        /// Column.
        x: u32,
        /// Row.
        y: u32,
        /// Expected red, green, blue.
        rgb: [f32; 3],
        /// Allowed absolute difference per channel.
        tolerance: f32,
        /// Line number, for messages.
        line: u32,
    },
}

/// A parsed replay file.
#[derive(Clone, Debug, PartialEq)]
pub struct Replay {
    /// Scene path, resolved.
    pub scene: PathBuf,
    /// Kernel directory, resolved, if given.
    pub kernels: Option<PathBuf>,
    /// Steps in order.
    pub steps: Vec<Step>,
}

/// A replay file does not parse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayError {
    /// 1-based line.
    pub line: u32,
    /// What is wrong.
    pub message: String,
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ReplayError {}

/// Parses replay text; relative paths resolve against `base`.
pub fn parse(text: &str, base: &Path) -> Result<Replay, ReplayError> {
    let mut scene = None;
    let mut kernels = None;
    let mut steps = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line = index as u32 + 1;
        let error = |message: String| ReplayError { line, message };
        let content = raw.split('#').next().unwrap_or("").trim();
        if content.is_empty() {
            continue;
        }
        let words: Vec<&str> = content.split_whitespace().collect();
        let number = |text: &str, what: &str| {
            text.parse::<f32>()
                .map_err(|_| error(format!("{what} `{text}` is not a number")))
        };
        match words.as_slice() {
            ["scene", path] => scene = Some(base.join(path)),
            ["library" | "kernels", path] => kernels = Some(base.join(path)),
            ["frame", time] => steps.push(Step::Frame(number(time, "time")?)),
            ["settle"] => steps.push(Step::Settle),
            ["pointer", state @ ("down" | "up")] => steps.push(Step::Pointer {
                position: None,
                down: *state == "down",
            }),
            ["pointer", x, y, rest @ ..] => {
                let down = match rest {
                    [] | ["up"] => false,
                    ["down"] => true,
                    _ => return Err(error("pointer ends with `down` or `up`".into())),
                };
                steps.push(Step::Pointer {
                    position: Some([number(x, "x")?, number(y, "y")?]),
                    down,
                });
            }
            ["button", name, state @ ("down" | "up")] => steps.push(Step::Button {
                code: match *name {
                    "left" => 0,
                    "right" => 1,
                    "middle" => 2,
                    "south" => orla_runtime::pad::SOUTH,
                    "east" => orla_runtime::pad::EAST,
                    "west" => orla_runtime::pad::WEST,
                    "north" => orla_runtime::pad::NORTH,
                    "l1" => orla_runtime::pad::LEFT_SHOULDER,
                    "r1" => orla_runtime::pad::RIGHT_SHOULDER,
                    "l3" => orla_runtime::pad::LEFT_THUMB,
                    "r3" => orla_runtime::pad::RIGHT_THUMB,
                    "start" => orla_runtime::pad::START,
                    "select" => orla_runtime::pad::SELECT,
                    _ => {
                        return Err(error(
                            "button is `left`, `right`, `middle`, or a pad's `south`, `east`, \
                             `west`, `north`, `l1`, `r1`, `l3`, `r3`, `start`, `select`"
                                .into(),
                        ));
                    }
                },
                down: *state == "down",
            }),
            ["wheel", x, y] => steps.push(Step::Wheel([number(x, "x")?, number(y, "y")?])),
            ["stick", which, x, y] => steps.push(Step::Stick {
                which: match *which {
                    "left" => 0,
                    "right" => 1,
                    "triggers" => 2,
                    _ => return Err(error("stick is `left`, `right`, or `triggers`".into())),
                },
                at: [number(x, "x")?, number(y, "y")?],
            }),
            ["window", w, h, rest @ ..] if rest.len() <= 1 => {
                let scale = match rest {
                    [s] => number(s, "scale")?,
                    _ => 1.0,
                };
                let size = [number(w, "width")?, number(h, "height")?];
                if size[0] <= 0.0 || size[1] <= 0.0 || scale <= 0.0 {
                    return Err(error(
                        "window takes a positive width, height, and scale".into(),
                    ));
                }
                steps.push(Step::Window { size, scale });
            }
            ["key", code, state @ ("down" | "up")] => steps.push(Step::Key {
                code: code
                    .parse::<i32>()
                    .map_err(|_| error(format!("key code `{code}` is not an integer")))?,
                down: *state == "down",
            }),
            ["reset"] => steps.push(Step::Reset),
            ["reset", target] => {
                let (node, port) = orla_graph::split_port(target)
                    .ok_or_else(|| error("`reset` takes nothing or `node.port`".to_owned()))?;
                steps.push(Step::ResetPort { node: node.0, port });
            }
            ["param", name, value @ ..] if !value.is_empty() => {
                let text = value.join(" ");
                let expr = orla_language::parse_literal(&text)
                    .map_err(|e| error(format!("param `{name}`: {e}")))?;
                let value = orla_compiler::literal(&expr)
                    .map_err(|e| error(format!("param `{name}`: {e}")))?;
                steps.push(Step::Param {
                    name: name.to_string(),
                    value,
                });
            }
            ["expect", "count", target, "=", count] => {
                let (node, port) = orla_graph::split_port(target)
                    .ok_or_else(|| error("`expect count` takes `node.port`".to_owned()))?;
                steps.push(Step::ExpectCount {
                    node: node.0,
                    port,
                    count: count.parse::<usize>().map_err(|_| {
                        error(format!("`{count}` is not a whole number of elements"))
                    })?,
                    line,
                });
            }
            ["expect", "placed", node, place] => {
                let routed = match *place {
                    "main" => false,
                    "ordered" => true,
                    other => {
                        return Err(error(format!(
                            "`expect placed` takes `main` or `ordered`, found `{other}`"
                        )));
                    }
                };
                steps.push(Step::ExpectPlaced {
                    node: node.to_string(),
                    routed,
                    line,
                });
            }
            ["expect", target, sign, values @ ..]
                if target.contains('[') && !values.is_empty() && Compare::read(sign).is_some() =>
            {
                let (node, port, index, path) = crate::parse_target(target).map_err(error)?;
                steps.push(Step::ExpectElement {
                    node,
                    port,
                    index,
                    path,
                    compare: Compare::read(sign).expect("matched"),
                    values: values.iter().map(|v| (*v).to_owned()).collect(),
                    line,
                });
            }
            ["expect", x, y, "=", r, g, b, rest @ ..] => {
                let tolerance = match rest {
                    [] => 0.01,
                    [tolerance] => number(tolerance, "tolerance")?,
                    _ => return Err(error("expect takes r g b and an optional tolerance".into())),
                };
                let pixel = |text: &str| {
                    text.parse::<u32>()
                        .map_err(|_| error(format!("pixel coordinate `{text}` is not an integer")))
                };
                steps.push(Step::Expect {
                    x: pixel(x)?,
                    y: pixel(y)?,
                    rgb: [number(r, "r")?, number(g, "g")?, number(b, "b")?],
                    tolerance,
                    line,
                });
            }
            _ => return Err(error(format!("cannot read `{content}`"))),
        }
    }
    Ok(Replay {
        scene: scene.ok_or(ReplayError {
            line: 1,
            message: "replay must name its scene: `scene <path>`".into(),
        })?,
        kernels,
        steps,
    })
}

/// What a replay run observed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outcome {
    /// Frames evaluated.
    pub frames: usize,
    /// Expectations checked.
    pub checks: usize,
    /// Expectations that failed, as messages.
    pub failures: Vec<String>,
}

/// Runs a replay on a fresh session over `runtime`: every frame
/// evaluates what its expectations need, and nothing else.
pub fn run(replay: &Replay, runtime: Runtime) -> Result<Outcome, String> {
    let mut session = Session::open_with(&replay.scene, replay.kernels.as_deref(), runtime);
    let mut rejected = Vec::new();
    for event in session.sync() {
        if let crate::Event::Rejected { .. } | crate::Event::SceneRejected(_) = &event {
            rejected.push(event.to_string());
        }
    }
    if !rejected.is_empty() {
        return Err(rejected.join("\n"));
    }
    let mut replayer = Replayer::start(replay, &mut session);
    while let Some(frame) = replayer.next_frame(&mut session)? {
        let report = session
            .evaluate_for(&frame.context, &frame.demand)
            .map_err(|error| format!("frame at {}: {error}", frame.context.time))?;
        replayer.evaluated(report);
    }
    Ok(replayer.outcome)
}

/// A frame a replay asks for: the context it evaluates under, and what it
/// must evaluate, the nodes its expectations name, and the picture when a
/// pixel expectation follows it. A caller that shows the picture anyway
/// (a window) evaluates everything.
pub struct Frame {
    /// The time, the input and the window of this frame.
    pub context: Context,
    /// What the expectations need.
    pub demand: std::collections::BTreeSet<orla_graph::NodeId>,
}

/// A replay played a frame at a time over a session its caller owns: the
/// steps between two frames are applied (inputs, the window, params,
/// expectations of the frame before), and the next frame is handed back
/// for the caller to evaluate however it evaluates. `orla replay` runs one
/// to the end at once; a window plays one a frame at a time, and shows it.
pub struct Replayer {
    steps: Vec<Step>,
    next: usize,
    input: InputState,
    window: ([f32; 2], f32),
    last: Option<Report>,
    named: std::collections::BTreeSet<orla_graph::NodeId>,
    producers: Vec<orla_graph::NodeId>,
    /// What the replay has observed so far.
    pub outcome: Outcome,
}

impl Replayer {
    /// Starts `replay` over `session`, which has read its scene: the nodes
    /// the expectations name are pinned, so fusion never hides them.
    pub fn start(replay: &Replay, session: &mut Session) -> Self {
        let named: std::collections::BTreeSet<orla_graph::NodeId> = replay
            .steps
            .iter()
            .filter_map(|step| match step {
                Step::ExpectElement { node, .. } | Step::ExpectCount { node, .. } => {
                    Some(orla_graph::NodeId(node.clone()))
                }
                _ => None,
            })
            .collect();
        for node in &named {
            session.pin(node.clone());
        }
        let producers = session
            .graph()
            .map(|graph| {
                graph
                    .outputs
                    .iter()
                    .filter_map(|(_, source)| match source {
                        orla_graph::Source::Node { node, .. } => Some(node.clone()),
                        orla_graph::Source::External(_) => None,
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self {
            steps: replay.steps.clone(),
            next: 0,
            input: InputState::default(),
            window: ([0.0, 0.0], 1.0),
            last: None,
            named,
            producers,
            outcome: Outcome::default(),
        }
    }

    /// The time of the next frame, when there is one.
    pub fn next_time(&self) -> Option<f32> {
        self.steps[self.next..].iter().find_map(|step| match step {
            Step::Frame(time) => Some(*time),
            _ => None,
        })
    }

    /// Whether every step has been applied.
    pub fn finished(&self) -> bool {
        self.next >= self.steps.len()
    }

    /// Applies the steps up to the next frame and hands that frame back;
    /// `None` once the replay is over, every step applied. Takes the last
    /// frame's report first ([`Replayer::evaluated`]) for the
    /// expectations that follow it.
    pub fn next_frame(&mut self, session: &mut Session) -> Result<Option<Frame>, String> {
        while self.next < self.steps.len() {
            let position = self.next;
            self.next += 1;
            if let Step::Frame(time) = &self.steps[position] {
                let context = Context {
                    time: *time,
                    rate: 0.0,
                    input: self.input,
                    window: self.window.0,
                    scale: self.window.1,
                };
                // The wheel is what scrolled before this frame only.
                self.input.wheel = [0.0, 0.0];
                // The picture is demanded only when a pixel expectation
                // follows this frame, before the next.
                let mut demand = self.named.clone();
                let pixels_follow = self.steps[position + 1..]
                    .iter()
                    .take_while(|step| !matches!(step, Step::Frame(_)))
                    .any(|step| matches!(step, Step::Expect { .. }));
                if pixels_follow {
                    demand.extend(self.producers.iter().cloned());
                }
                return Ok(Some(Frame { context, demand }));
            }
            let step = self.steps[position].clone();
            self.apply(&step, session)?;
        }
        Ok(None)
    }

    /// Takes the report of the frame [`Replayer::next_frame`] handed back.
    pub fn evaluated(&mut self, report: Report) {
        self.outcome.frames += 1;
        self.last = Some(report);
    }

    /// One step that is not a frame.
    fn apply(&mut self, step: &Step, session: &mut Session) -> Result<(), String> {
        let input = &mut self.input;
        let outcome = &mut self.outcome;
        let last = &self.last;
        match step {
            Step::Frame(_) => {}
            Step::Window { size, scale } => self.window = (*size, *scale),
            Step::Settle => {
                session
                    .settle()
                    .map_err(|error| format!("settle: {error}"))?;
            }
            Step::Pointer { position, down } => {
                if let Some(position) = position {
                    input.pointer = *position;
                }
                input.pointer_down = *down;
            }
            Step::Key { code, down } => input.set_key(*code, *down),
            Step::Button { code, down } => input.set_button(*code, *down),
            Step::Stick { which, at } => {
                if let Some(pair) = input.sticks.get_mut(*which as usize) {
                    *pair = *at;
                }
            }
            Step::Wheel(delta) => input.wheel = *delta,
            Step::Reset => session.reset_state(),
            Step::ResetPort { node, port } => {
                session
                    .reset_port(&orla_graph::NodeId(node.clone()), port)
                    .map_err(|error| format!("reset {node}.{port}: {error}"))?;
            }
            Step::Param { name, value } => {
                session.set_param(name, value.clone())?;
                let _ = session.sync();
            }
            Step::ExpectElement {
                node,
                port,
                index,
                path,
                compare,
                values,
                line,
            } => {
                if last.is_none() {
                    return Err(format!("line {line}: expect before any frame"));
                }
                outcome.checks += 1;
                let target = format!(
                    "{node}.{port}[{index}]{}",
                    path.iter().map(|p| format!(".{p}")).collect::<String>()
                );
                match check_element(session, node, port, *index, path, *compare, values) {
                    Ok(None) => {}
                    Ok(Some(mismatch)) => outcome
                        .failures
                        .push(format!("line {line}: {target} {mismatch}")),
                    Err(message) => return Err(format!("line {line}: {target}: {message}")),
                }
            }
            Step::ExpectCount {
                node,
                port,
                count,
                line,
            } => {
                if last.is_none() {
                    return Err(format!("line {line}: expect before any frame"));
                }
                outcome.checks += 1;
                match session.element_count(&orla_graph::NodeId(node.clone()), port) {
                    Ok(actual) if actual == *count => {}
                    Ok(actual) => outcome.failures.push(format!(
                        "line {line}: {node}.{port} holds {actual} element(s), expected {count}"
                    )),
                    Err(message) => {
                        return Err(format!("line {line}: count {node}.{port}: {message}"));
                    }
                }
            }
            Step::ExpectPlaced { node, routed, line } => {
                outcome.checks += 1;
                let placements = session.placements();
                let Some((_, placement)) = placements.iter().find(|(id, _)| id == node) else {
                    return Err(format!(
                        "line {line}: no node `{node}` with a published Kernel"
                    ));
                };
                // Where there is no second backend, nothing can leave the
                // main one: `ordered` is a claim about a session that has one.
                let has_ordered = session.runtime().ordered_backend_name().is_some();
                let expected = *routed && has_ordered;
                if placement.routed.is_some() != expected {
                    outcome.failures.push(match &placement.routed {
                        Some(why) => format!(
                            "line {line}: {node} runs on {}, off the main backend ({why})",
                            placement.backend
                        ),
                        None => format!(
                            "line {line}: {node} runs on the main backend ({})",
                            placement.backend
                        ),
                    });
                }
            }
            Step::Expect {
                x,
                y,
                rgb,
                tolerance,
                line,
            } => {
                let output: Handle = last
                    .as_ref()
                    .map(|report| report.output)
                    .ok_or_else(|| format!("line {line}: expect before any frame"))?;
                let actual = session
                    .read_pixel(output, *x, *y)
                    .map_err(|error| format!("line {line}: {error}"))?;
                outcome.checks += 1;
                let off = actual
                    .iter()
                    .zip(rgb)
                    .any(|(a, e)| (a - e).abs() > *tolerance);
                if off {
                    outcome.failures.push(format!(
                        "line {line}: pixel ({x}, {y}) is {:.3} {:.3} {:.3}, expected {:.3} {:.3} {:.3}",
                        actual[0], actual[1], actual[2], rgb[0], rgb[1], rgb[2]
                    ));
                }
            }
        }
        Ok(())
    }
}

/// How an element expectation holds a value to what is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compare {
    /// `=`: equal, a float within the tolerance.
    Equal,
    /// `<`: every component below.
    Below,
    /// `<=`: every component at most.
    AtMost,
    /// `>`: every component above.
    Above,
    /// `>=`: every component at least.
    AtLeast,
}

impl Compare {
    /// The comparison a sign writes, if it writes one.
    fn read(sign: &str) -> Option<Self> {
        Some(match sign {
            "=" => Self::Equal,
            "<" => Self::Below,
            "<=" => Self::AtMost,
            ">" => Self::Above,
            ">=" => Self::AtLeast,
            _ => return None,
        })
    }

    fn sign(self) -> &'static str {
        match self {
            Self::Equal => "=",
            Self::Below => "<",
            Self::AtMost => "<=",
            Self::Above => ">",
            Self::AtLeast => ">=",
        }
    }

    /// Whether `actual` stands where this says of `expected`.
    fn holds(self, actual: f64, expected: f64) -> bool {
        match self {
            Self::Equal => actual == expected,
            Self::Below => actual < expected,
            Self::AtMost => actual <= expected,
            Self::Above => actual > expected,
            Self::AtLeast => actual >= expected,
        }
    }
}

/// Reads one element field of a node's output and compares it with the
/// written values. `Ok(None)` means it matched; `Ok(Some(text))` says how
/// it did not; `Err` means the expectation could not be read at all.
fn check_element(
    session: &mut Session,
    node: &str,
    port: &str,
    index: u32,
    path: &[String],
    compare: Compare,
    values: &[String],
) -> Result<Option<String>, String> {
    use orla_ir::{Number, Type};
    let node_id = orla_graph::NodeId(node.to_owned());
    let (ty, words) = session.element_field(&node_id, port, index, path)?;
    if matches!(ty, Type::Struct(_)) {
        return Err("name a field of the struct; whole structs are not compared".into());
    }
    let numbers = orla_ir::template(ty, &[])
        .rebuilt(&mut words.iter().copied())
        .numbers();
    let components = numbers.len();
    let exact = numbers
        .iter()
        .all(|n| matches!(n, Number::Int(_) | Number::Long(_) | Number::Bool(_)));
    if compare != Compare::Equal && numbers.iter().any(|n| matches!(n, Number::Bool(_))) {
        return Err(format!(
            "a {ty} is equal or not: `{}` takes numbers",
            compare.sign()
        ));
    }
    let tolerance = match values.len() {
        n if n == components => 0.01,
        // A bound is a bound: only `=` takes a tolerance.
        n if n == components + 1 && !exact && compare == Compare::Equal => values[components]
            .parse::<f64>()
            .map_err(|_| format!("tolerance `{}` is not a number", values[components]))?,
        _ => {
            return Err(format!(
                "a {ty} takes {components} value(s){}",
                if exact || compare != Compare::Equal {
                    ""
                } else {
                    " and an optional tolerance"
                }
            ));
        }
    };
    let actual: Vec<String> = numbers
        .iter()
        .map(|n| match n {
            Number::Float(v) => format!("{v}"),
            Number::Double(v) => format!("{v}"),
            Number::Int(v) => v.to_string(),
            Number::Long(v) => v.to_string(),
            Number::Bool(v) => v.to_string(),
        })
        .collect();
    let near = |expected: &str, v: f64| {
        expected.parse::<f64>().is_ok_and(|e| match compare {
            Compare::Equal => (e - v).abs() <= tolerance,
            other => other.holds(v, e),
        })
    };
    // Integers compare exactly, as integers, whichever the comparison.
    let whole = |expected: &str, v: i64| {
        expected.parse::<i64>().is_ok_and(|e| match compare {
            Compare::Equal => v == e,
            Compare::Below => v < e,
            Compare::AtMost => v <= e,
            Compare::Above => v > e,
            Compare::AtLeast => v >= e,
        })
    };
    let matches = numbers.iter().zip(values).all(|(n, expected)| match n {
        Number::Int(v) => whole(expected, i64::from(*v)),
        Number::Long(v) => whole(expected, *v),
        Number::Bool(v) => match expected.as_str() {
            "true" => *v,
            "false" => !*v,
            _ => false,
        },
        Number::Float(v) => near(expected, f64::from(*v)),
        Number::Double(v) => near(expected, *v),
    });
    Ok((!matches).then(|| match compare {
        Compare::Equal => format!(
            "is {}, expected {}",
            actual.join(" "),
            values[..components].join(" ")
        ),
        other => format!(
            "is {}, expected {} {}",
            actual.join(" "),
            other.sign(),
            values[..components].join(" ")
        ),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_element_is_held_equal_or_to_one_side() {
        let replay = parse(
            "scene s.osn\nexpect a.result[0] = 1\nexpect a.result[0] < 2\nexpect a.result[1].v >= 0.5 0.5 0.5\nexpect a.result[2] > -1\nexpect a.result[3] <= 0\n",
            Path::new("."),
        )
        .unwrap();
        let compares: Vec<Compare> = replay
            .steps
            .iter()
            .map(|step| match step {
                Step::ExpectElement { compare, .. } => *compare,
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(
            compares,
            [
                Compare::Equal,
                Compare::Below,
                Compare::AtLeast,
                Compare::Above,
                Compare::AtMost
            ]
        );
        // Not a comparison it knows, and the line says so.
        let error = parse("scene s.osn\nexpect a.result[0] ~ 1\n", Path::new(".")).unwrap_err();
        assert!(error.message.contains("cannot read"), "{error}");
        assert!(Compare::Below.holds(1.0, 2.0) && !Compare::Below.holds(2.0, 2.0));
        assert!(Compare::AtLeast.holds(2.0, 2.0) && !Compare::Above.holds(2.0, 2.0));
    }

    #[test]
    fn parses_every_step() {
        let replay = parse(
            "scene s.osn\nkernels k\nframe 0.5\npointer 1 2 down\npointer up\nkey 65 down\nreset\nexpect 1 2 = 0.1 0.2 0.3\nexpect 1 2 = 0 0 0 0.5\nbutton right down\nwheel 0 -1\n",
            Path::new("/base"),
        )
        .unwrap();
        assert_eq!(replay.scene, Path::new("/base/s.osn"));
        assert_eq!(replay.kernels, Some(PathBuf::from("/base/k")));
        let _ = "library";
        assert_eq!(replay.steps.len(), 9);
        assert!(matches!(replay.steps[6], Step::Expect { tolerance, .. } if tolerance == 0.5));
        assert!(matches!(
            replay.steps[7],
            Step::Button {
                code: 1,
                down: true
            }
        ));
        assert!(matches!(replay.steps[8], Step::Wheel([x, y]) if x == 0.0 && y == -1.0));
        let error = parse("scene s\nbutton fourth down\n", Path::new(".")).unwrap_err();
        assert!(error.message.contains("left"), "{error}");
        let replay = parse(
            "scene s.osn\nexpect step.result[0].score = 1\nexpect mix.result[3] = 0.5 0.25 0.02\n",
            Path::new("."),
        )
        .unwrap();
        assert!(matches!(
            &replay.steps[0],
            Step::ExpectElement { node, port, index: 0, path, values, .. }
                if node == "step" && port == "result" && path == &["score"] && values == &["1"]
        ));
        assert!(
            matches!(&replay.steps[1], Step::ExpectElement { path, values, .. } if path.is_empty() && values.len() == 3)
        );
        let replay = parse(
            "scene s\nexpect placed ring main\nexpect placed filter ordered\n",
            Path::new("."),
        )
        .unwrap();
        assert!(matches!(
            &replay.steps[1],
            Step::ExpectPlaced { node, routed: true, .. } if node == "filter"
        ));
        let error = parse("scene s\nexpect placed ring gpu\n", Path::new(".")).unwrap_err();
        assert!(error.message.contains("`main` or `ordered`"), "{error}");
        let error = parse("frame 1\n", Path::new(".")).unwrap_err();
        assert!(error.message.contains("scene"), "{error}");
        let error = parse("scene s\nexpect 1 = 2\n", Path::new(".")).unwrap_err();
        assert_eq!(error.line, 2);
    }
}
