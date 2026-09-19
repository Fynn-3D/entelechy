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
//! key 67 down               # press a key by code; `up` releases
//! reset                     # forget committed state
//! param exposure 1.5        # override a graph parameter (a literal)
//! expect 11 10 = 1.0 1.0 1.0        # r g b of the last output, tolerance 0.01
//! expect 11 10 = 1.0 1.0 1.0 0.05   # explicit tolerance
//! expect step.result[0].score = 1   # a field of an element of a node's output:
//!                                   # ints and bools exact, floats and vectors
//!                                   # with an optional trailing tolerance
//! ```
//!
//! An element expectation names the node's output as the Kernel declared
//! it, so it reads as a statement about the application's state rather
//! than about the picture. Nodes named that way are pinned before the
//! first frame, so fusion never hides them.

use std::fmt;
use std::path::{Path, PathBuf};

use orla_graph::Report;
use orla_runtime::{Context, Handle, InputState, Runtime};

use crate::session::Session;

/// One scripted step.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    /// Evaluate at a session time.
    Frame(f32),
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
                    _ => return Err(error("button is `left`, `right`, or `middle`".into())),
                },
                down: *state == "down",
            }),
            ["wheel", x, y] => steps.push(Step::Wheel([number(x, "x")?, number(y, "y")?])),
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
            ["expect", target, "=", values @ ..] if target.contains('[') && !values.is_empty() => {
                let (head, rest) = target
                    .split_once('[')
                    .ok_or_else(|| error("expect an element as `node.port[index]`".into()))?;
                let (index, tail) = rest
                    .split_once(']')
                    .ok_or_else(|| error("expect an element as `node.port[index]`".into()))?;
                let (node, port) = head
                    .rsplit_once('.')
                    .ok_or_else(|| error("expect names `node.port`".into()))?;
                let index = index
                    .parse::<u32>()
                    .map_err(|_| error(format!("element index `{index}` is not an integer")))?;
                let path: Vec<String> = tail
                    .split('.')
                    .filter(|part| !part.is_empty())
                    .map(str::to_owned)
                    .collect();
                steps.push(Step::ExpectElement {
                    node: node.to_owned(),
                    port: port.to_owned(),
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
        if let crate::session::Event::Rejected { .. } | crate::session::Event::SceneRejected(_) =
            &event
        {
            rejected.push(event.to_string());
        }
    }
    if !rejected.is_empty() {
        return Err(rejected.join("\n"));
    }
    for step in &replay.steps {
        if let Step::ExpectElement { node, .. } = step {
            session.pin(orla_graph::NodeId(node.clone()));
        }
    }
    let mut input = InputState::default();
    let mut last: Option<Report> = None;
    let mut outcome = Outcome::default();
    for step in &replay.steps {
        match step {
            Step::Frame(time) => {
                let context = Context {
                    time: *time,
                    rate: 0.0,
                    input,
                };
                let report = session
                    .evaluate(&context)
                    .map_err(|error| format!("frame at {time}: {error}"))?;
                outcome.frames += 1;
                last = Some(report);
                input.wheel = [0.0, 0.0];
            }
            Step::Pointer { position, down } => {
                if let Some(position) = position {
                    input.pointer = *position;
                }
                input.pointer_down = *down;
            }
            Step::Key { code, down } => input.set_key(*code, *down),
            Step::Button { code, down } => match code {
                0 => input.pointer_down = *down,
                1 => input.pointer_right = *down,
                _ => input.pointer_middle = *down,
            },
            Step::Wheel(delta) => input.wheel = *delta,
            Step::Reset => session.reset_state(),
            Step::Param { name, value } => session.set_param(name, *value),
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
    use orla_ir::Type;
    let node_id = orla_graph::NodeId(node.to_owned());
    let kernel = session
        .graph()
        .and_then(|graph| graph.nodes.iter().find(|n| n.id == node_id))
        .map(|n| n.kernel.clone())
        .ok_or_else(|| format!("no node `{node}` in the scene"))?;
    let version = session
        .runtime()
        .kernel(&kernel)
        .ok_or_else(|| format!("`{kernel}` is not published"))?;
    let output = version
        .manifest()
        .outputs
        .iter()
        .find(|o| o.name == port)
        .ok_or_else(|| format!("`{kernel}` has no output `{port}`"))?;
    let structs: Vec<orla_ir::StructDef> = version
        .module()
        .map(|m| m.structs.clone())
        .unwrap_or_default();
    // Follow the field path through the struct table to a word offset
    // and a type.
    let mut ty = output.element();
    let mut offset = 0usize;
    let words_of = |ty: Type| orla_ir::schema_of(ty, &structs).map_or(1, |s| s.words());
    for field in path {
        let Type::Struct(id) = ty else {
            return Err(format!("`{field}` names a field, but the value is a {ty}"));
        };
        let def = &structs[id as usize];
        let mut found = None;
        for f in &def.fields {
            if &f.name == field {
                found = Some(f.ty);
                break;
            }
            offset += words_of(f.ty);
        }
        ty = found.ok_or_else(|| format!("struct `{}` has no field `{field}`", def.name))?;
    }
    let handle = session
        .cached(&node_id, port)
        .ok_or_else(|| "the node has no materialized output; is it wired?".to_string())?;
    session
        .runtime_mut()
        .materialize(handle)
        .map_err(|e| e.to_string())?;
    let element = session
        .runtime()
        .host()
        .element(handle, index)
        .map_err(|e| e.to_string())?;
    let words = &element[offset..offset + words_of(ty)];
    let components = match ty {
        Type::Float | Type::Int | Type::Bool => 1,
        Type::Vec2 => 2,
        Type::Vec3 => 3,
        Type::Vec4 => 4,
        Type::Struct(_) => {
            return Err("name a field of the struct; whole structs are not compared".into());
        }
    };
    let exact = matches!(ty, Type::Int | Type::Bool);
    let tolerance = match values.len() {
        n if n == components => 0.01,
        n if n == components + 1 && !exact => values[components]
            .parse::<f32>()
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
    let actual: Vec<String> = words
        .iter()
        .map(|w| match ty {
            Type::Int => (*w as i32).to_string(),
            Type::Bool => (*w != 0).to_string(),
            _ => format!("{}", f32::from_bits(*w)),
        })
        .collect();
    let matches = words.iter().zip(values).all(|(w, expected)| match ty {
        Type::Int => expected.parse::<i32>().is_ok_and(|e| e == *w as i32),
        Type::Bool => match expected.as_str() {
            "true" => *w != 0,
            "false" => *w == 0,
            _ => false,
        },
        _ => expected
            .parse::<f32>()
            .is_ok_and(|e| (e - f32::from_bits(*w)).abs() <= tolerance),
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
