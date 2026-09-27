//! Declarative replay: an application's tests as data. A replay drives a
//! scene through scripted frames and inputs and checks what they produce.
//! Every form a line takes is in `docs/USAGE.md`, *Replay tests*, which
//! [`parse`] is held to.

use std::fmt;
use std::path::{Path, PathBuf};

use orla_graph::Report;
use orla_kernel::{Context, Handle, InputState, Runtime};

use crate::Session;

/// One form a replay line takes, as `help` and `docs/USAGE.md` show it.
#[derive(Clone, Debug)]
pub struct Form {
    /// The line's first word.
    pub keyword: &'static str,
    /// Its shape: `<...>` a value, `[...]` optional, `|` a choice.
    pub shape: &'static str,
    /// What it does.
    pub doc: &'static str,
    /// Lines written in it, each one the parser reads.
    pub examples: &'static [&'static str],
}

/// Every form a replay line takes, in the order a replay uses them.
pub fn forms() -> Vec<Form> {
    let form = |keyword, shape, doc, examples| Form {
        keyword,
        shape,
        doc,
        examples,
    };
    vec![
        form(
            "scene",
            "scene <path>",
            "the scene to test, relative to the replay file; required",
            &["scene ../scene.osn"],
        ),
        form(
            "library",
            "library <path>",
            "Kernels beyond the scene's `use`",
            &["library ../kernels/"],
        ),
        form(
            "frame",
            "frame <seconds>",
            "evaluates one frame at this session time",
            &["frame 0.0"],
        ),
        form(
            "settle",
            "settle",
            "waits for every `async` run to land",
            &["settle"],
        ),
        form(
            "pointer",
            "pointer [<x> <y>] [down | up]",
            "moves the pointer, in the output's pixels from its top left (what `pointer()` reads), and presses or releases it",
            &["pointer 20 20 down", "pointer up"],
        ),
        form(
            "button",
            "button <name> down | up",
            "left, right, middle, or a pad's south, east, west, north, l1, r1, l3, r3, start, select",
            &["button right down", "button south up"],
        ),
        form(
            "stick",
            "stick left | right | triggers <x> <y>",
            "a pad's axis pair",
            &["stick left 0.5 -1"],
        ),
        form(
            "wheel",
            "wheel <x> <y>",
            "the scroll for the next frame only",
            &["wheel 0 -1"],
        ),
        form(
            "window",
            "window <w> <h> [<scale>]",
            "the window in pixels, for `window()`, and its scale factor; one when not given",
            &["window 240 135", "window 240 135 0.25"],
        ),
        form(
            "key",
            "key <code> down | up",
            "a key by the browser's code: letters and digits by uppercase ASCII, space 32, enter 13, arrows 37 to 40",
            &["key 67 down"],
        ),
        form(
            "reset",
            "reset [<node.port>]",
            "forgets every node's committed state, or one port's",
            &["reset", "reset step.result"],
        ),
        form(
            "param",
            "param <name> <literal>",
            "sets a graph parameter",
            &["param exposure 1.5"],
        ),
        form(
            "expect",
            "expect <x> <y> = <r> <g> <b> [<tolerance>]",
            "the picture's color at x y from its top left, each channel 0 to 1 as the Kernel wrote it; tolerance 0.01 when not given",
            &[
                "expect 11 10 = 1.0 1.0 1.0",
                "expect 11 10 = 1.0 1.0 1.0 0.05",
            ],
        ),
        form(
            "expect",
            "expect <node.port[i][.field]> = <value>... [<tolerance>]",
            "an element of a node's output, or a field of it (a struct's element is compared a field at a time): ints and bools exactly, floats and vectors within the tolerance; `i` counts along the Field, x fastest, then y, then z",
            &["expect step.result[0].score = 1"],
        ),
        form(
            "expect",
            "expect <node.port[i][.field]> < | <= | > | >= <number>...",
            "every component of it past a bound",
            &["expect fall.result[0].speed > 2.5"],
        ),
        form(
            "expect",
            "expect count <node.port> = <n>",
            "how many elements a node's output holds",
            &["expect count ground.result = 6889"],
        ),
        form(
            "expect",
            "expect placed <node> main | ordered",
            "where a node runs: the session's main backend, or ordered work off it; a session with one backend reads both as main",
            &["expect placed ring main", "expect placed filter ordered"],
        ),
    ]
}

/// The forms as `docs/USAGE.md` shows them.
pub fn forms_table() -> String {
    let mut out = String::from("| form | what it does | example |\n| --- | --- | --- |\n");
    for form in forms() {
        let examples = form
            .examples
            .iter()
            .map(|example| format!("`{example}`"))
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!(
            "| `{}` | {} | {examples} |\n",
            form.shape.replace('|', "\\|"),
            form.doc
        ));
    }
    out
}

/// One scripted step.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    /// Evaluate at a session time.
    Frame(f32),
    /// Wait for every `async` node's run to land.
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
    /// A pad's axis pair: 0 left stick, 1 right stick, 2 triggers.
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
    /// Forget one port's committed state.
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
        value: orla_kernel::Value,
    },
    /// Check a field of an element of a node's output after the last frame.
    ExpectElement {
        /// The node.
        node: String,
        /// Its output port.
        port: String,
        /// Element index.
        index: u32,
        /// Struct fields, outermost first; empty for the element itself.
        path: Vec<String>,
        /// Equal (floats within a tolerance), or a bound.
        compare: Compare,
        /// Expected components as written.
        values: Vec<String>,
        /// Line number, for messages.
        line: u32,
    },
    /// Check how many elements a node's output holds after the last
    /// frame.
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
    /// Check where a node runs: the main backend, or the ordered one.
    /// Without an ordered backend (CPU, JIT), `ordered` checks as `main`.
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
                    "south" => orla_kernel::pad::SOUTH,
                    "east" => orla_kernel::pad::EAST,
                    "west" => orla_kernel::pad::WEST,
                    "north" => orla_kernel::pad::NORTH,
                    "l1" => orla_kernel::pad::LEFT_SHOULDER,
                    "r1" => orla_kernel::pad::RIGHT_SHOULDER,
                    "l3" => orla_kernel::pad::LEFT_THUMB,
                    "r3" => orla_kernel::pad::RIGHT_THUMB,
                    "start" => orla_kernel::pad::START,
                    "select" => orla_kernel::pad::SELECT,
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
            [word, ..] => {
                let all = forms();
                let shapes: Vec<&str> = all
                    .iter()
                    .filter(|form| form.keyword == *word)
                    .map(|form| form.shape)
                    .collect();
                return Err(error(if shapes.is_empty() {
                    let mut keywords: Vec<&str> = all.iter().map(|form| form.keyword).collect();
                    keywords.dedup();
                    format!(
                        "`{word}` is no step; the steps are {}{}",
                        keywords.join(", "),
                        orla_language::did_you_mean(word, keywords.iter().copied())
                    )
                } else {
                    format!(
                        "cannot read `{content}`; `{word}` is written {}",
                        shapes.join(", or ")
                    )
                }));
            }
            [] => unreachable!("an empty line is skipped"),
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

/// Runs a replay on a fresh session over `runtime`, each frame
/// evaluating only what its expectations need.
pub fn run(replay: &Replay, runtime: Runtime) -> Result<Outcome, String> {
    run_in(replay, std::sync::Arc::new(crate::Disk::default()), runtime)
}

/// [`run`], reading documents from `store`.
pub fn run_in(
    replay: &Replay,
    store: std::sync::Arc<dyn crate::Store>,
    runtime: Runtime,
) -> Result<Outcome, String> {
    let mut session = Session::open_in(store, &replay.scene, replay.kernels.as_deref(), runtime);
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

/// A frame a replay asks for: its context and what it must evaluate. A
/// caller that shows the picture (a window) evaluates everything.
pub struct Frame {
    /// The time, input and window of this frame.
    pub context: Context,
    /// The named nodes, and the picture when a pixel expectation follows.
    pub demand: std::collections::BTreeSet<orla_graph::NodeId>,
}

/// A replay played a frame at a time over a caller's session: the steps
/// between frames are applied, and the next frame is handed back for the
/// caller to evaluate.
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
    /// Starts `replay` over a synced `session`, pinning the nodes the
    /// expectations name.
    pub fn start(replay: &Replay, session: &mut Session) -> Self {
        // What an expectation names, by the node that produces it: a
        // placed graph's output names the node inside that gives it.
        let named: std::collections::BTreeSet<orla_graph::NodeId> = replay
            .steps
            .iter()
            .filter_map(|step| match step {
                Step::ExpectElement { node, port, .. } | Step::ExpectCount { node, port, .. } => {
                    Some(session.producer(&orla_graph::NodeId(node.clone()), port).0)
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

    /// Applies the steps up to the next frame and returns it; `None` once
    /// every step is applied. Expectations read the report given to
    /// [`Replayer::evaluated`].
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
                // The wheel scrolls for one frame.
                self.input.wheel = [0.0, 0.0];
                // Demand the picture when a pixel expectation follows.
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

    /// Applies one non-frame step.
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
                // `ordered` expects routing only where a second backend exists.
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

/// Compares one element field of a node's output with the written values:
/// `Ok(None)` matched, `Ok(Some(text))` the mismatch, `Err` unreadable.
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
        // Only `=` takes a tolerance.
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
    // Integers compare exactly under every comparison.
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

    /// Every form's examples parse, together they cover every step, and
    /// `docs/USAGE.md` shows the forms; `ORLA_BLESS=1` writes them again.
    #[test]
    fn the_forms_parse_cover_every_step_and_are_shown() {
        let lines: Vec<&str> = forms()
            .iter()
            .flat_map(|form| form.examples.iter().copied())
            .collect();
        let replay = parse(&lines.join("\n"), Path::new(".")).unwrap();
        assert!(replay.kernels.is_some(), "`library` is a form");
        let mut seen = [false; 15];
        for step in &replay.steps {
            seen[match step {
                Step::Frame(_) => 0,
                Step::Settle => 1,
                Step::Pointer { .. } => 2,
                Step::Key { .. } => 3,
                Step::Button { .. } => 4,
                Step::Wheel(_) => 5,
                Step::Stick { .. } => 6,
                Step::Window { .. } => 7,
                Step::Reset => 8,
                Step::ResetPort { .. } => 9,
                Step::Param { .. } => 10,
                Step::ExpectElement { .. } => 11,
                Step::ExpectCount { .. } => 12,
                Step::ExpectPlaced { .. } => 13,
                Step::Expect { .. } => 14,
            }] = true;
        }
        assert!(
            seen.iter().all(|seen| *seen),
            "a step no form shows: {seen:?}"
        );
        for form in forms() {
            for example in form.examples {
                assert!(example.starts_with(form.keyword), "{example}");
            }
        }
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../docs/USAGE.md");
        let text = std::fs::read_to_string(&path).unwrap();
        let start = text
            .find("\n| form |")
            .expect("USAGE.md has the replay forms")
            + 1;
        let end = start + text[start..].find("\n\n").expect("the table ends") + 1;
        let table = forms_table();
        if text[start..end] != table {
            if std::env::var_os("ORLA_BLESS").is_some() {
                std::fs::write(&path, format!("{}{table}{}", &text[..start], &text[end..]))
                    .unwrap();
            } else {
                panic!(
                    "docs/USAGE.md's replay forms are not the parser's; write them again with ORLA_BLESS=1 cargo test -p orla-session --lib\n{table}"
                );
            }
        }
    }

    /// A line no form reads names the steps, or how its keyword is written.
    #[test]
    fn a_line_no_form_reads_says_what_is() {
        let error = parse("scene s.osn\nfram 0", Path::new(".")).unwrap_err();
        assert!(
            error.message.contains("did you mean `frame`"),
            "{}",
            error.message
        );
        let error = parse("scene s.osn\nstick up 1 2", Path::new(".")).unwrap_err();
        assert!(
            error.message.contains("stick is `left`"),
            "{}",
            error.message
        );
        let error = parse("scene s.osn\nwheel 1", Path::new(".")).unwrap_err();
        assert!(
            error.message.contains("`wheel` is written wheel <x> <y>"),
            "{}",
            error.message
        );
    }

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
        // An unknown comparison is refused.
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
