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
//! expect count ground.result = 6889 # how many elements a node's output holds,
//!                                   # which is what a `domain` came out as
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

use orla_session::Session;

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
                let (node, port) = target
                    .split_once('.')
                    .ok_or_else(|| error("`expect count` takes `node.port`".to_owned()))?;
                steps.push(Step::ExpectCount {
                    node: node.to_owned(),
                    port: port.to_owned(),
                    count: count.parse::<usize>().map_err(|_| {
                        error(format!("`{count}` is not a whole number of elements"))
                    })?,
                    line,
                });
            }
            ["expect", target, "=", values @ ..] if target.contains('[') && !values.is_empty() => {
                let (node, port, index, path) =
                    orla_session::parse_target(target).map_err(error)?;
                steps.push(Step::ExpectElement {
                    node,
                    port,
                    index,
                    path,
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

/// Runs a replay on a fresh session over `runtime`.
pub fn run(replay: &Replay, runtime: Runtime) -> Result<Outcome, String> {
    let mut session = Session::open_with(&replay.scene, replay.kernels.as_deref(), runtime);
    let mut rejected = Vec::new();
    for event in session.sync() {
        if let orla_session::Event::Rejected { .. } | orla_session::Event::SceneRejected(_) = &event
        {
            rejected.push(event.to_string());
        }
    }
    if !rejected.is_empty() {
        return Err(rejected.join("\n"));
    }
    for step in &replay.steps {
        if let Step::ExpectElement { node, .. } | Step::ExpectCount { node, .. } = step {
            session.pin(orla_graph::NodeId(node.clone()));
        }
    }
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
    let producers: Vec<orla_graph::NodeId> = session
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
    let mut input = InputState::default();
    let mut window = ([0.0, 0.0], 1.0);
    let mut last: Option<Report> = None;
    let mut outcome = Outcome::default();
    for (position, step) in replay.steps.iter().enumerate() {
        match step {
            Step::Window { size, scale } => window = (*size, *scale),
            Step::Frame(time) => {
                let context = Context {
                    time: *time,
                    rate: 0.0,
                    input,
                    window: window.0,
                    scale: window.1,
                };
                // The picture is demanded only when a pixel expectation
                // follows this frame, before the next.
                let mut demand = named.clone();
                let pixels_follow = replay.steps[position + 1..]
                    .iter()
                    .take_while(|step| !matches!(step, Step::Frame(_)))
                    .any(|step| matches!(step, Step::Expect { .. }));
                if pixels_follow {
                    demand.extend(producers.iter().cloned());
                }
                let report = session
                    .evaluate_for(&context, &demand)
                    .map_err(|error| format!("frame at {time}: {error}"))?;
                outcome.frames += 1;
                last = Some(report);
                input.wheel = [0.0, 0.0];
            }
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
            Step::Param { name, value } => {
                session.set_param(name, value.clone())?;
                let _ = session.sync();
            }
            Step::ExpectElement {
                node,
                port,
                index,
                path,
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
                match check_element(&mut session, node, port, *index, path, values) {
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
    }
    Ok(outcome)
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
    let tolerance = match values.len() {
        n if n == components => 0.01,
        n if n == components + 1 && !exact => values[components]
            .parse::<f64>()
            .map_err(|_| format!("tolerance `{}` is not a number", values[components]))?,
        _ => {
            return Err(format!(
                "a {ty} takes {components} value(s){}",
                if exact {
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
        expected
            .parse::<f64>()
            .is_ok_and(|e| (e - v).abs() <= tolerance)
    };
    let matches = numbers.iter().zip(values).all(|(n, expected)| match n {
        Number::Int(v) => expected.parse::<i64>().is_ok_and(|e| e == i64::from(*v)),
        Number::Long(v) => expected.parse::<i64>().is_ok_and(|e| e == *v),
        Number::Bool(v) => match expected.as_str() {
            "true" => *v,
            "false" => !*v,
            _ => false,
        },
        Number::Float(v) => near(expected, f64::from(*v)),
        Number::Double(v) => near(expected, *v),
    });
    Ok((!matches).then(|| {
        format!(
            "is {}, expected {}",
            actual.join(" "),
            values[..components].join(" ")
        )
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let error = parse("frame 1\n", Path::new(".")).unwrap_err();
        assert!(error.message.contains("scene"), "{error}");
        let error = parse("scene s\nexpect 1 = 2\n", Path::new(".")).unwrap_err();
        assert_eq!(error.line, 2);
    }
}
