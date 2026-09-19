//! Declarative replay: drive a scene through a scripted sequence of frames
//! and inputs and check pixels, so an application's tests are data rather
//! than Rust.
//!
//! ```text
//! scene ../scene.osn        # required; paths are relative to the replay file
//! kernels ../kernels        # optional; defaults to <scene dir>/kernels
//! frame 0.0                 # evaluate at this session time
//! pointer 20 20 down        # move the pointer and press; `up` releases
//! pointer up                # release without moving
//! key 67 down               # press a key by code; `up` releases
//! reset                     # forget committed state
//! expect 11 10 = 1.0 1.0 1.0        # r g b of the last output, tolerance 0.01
//! expect 11 10 = 1.0 1.0 1.0 0.05   # explicit tolerance
//! ```

use std::fmt;
use std::path::{Path, PathBuf};

use orla_graph::Report;
use orla_runtime::{Context, ImageHandle, InputState, Runtime};

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
    /// Forget committed state.
    Reset,
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
            ["kernels", path] => kernels = Some(base.join(path)),
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
            ["key", code, state @ ("down" | "up")] => steps.push(Step::Key {
                code: code
                    .parse::<i32>()
                    .map_err(|_| error(format!("key code `{code}` is not an integer")))?,
                down: *state == "down",
            }),
            ["reset"] => steps.push(Step::Reset),
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
    let mut input = InputState::default();
    let mut last: Option<Report> = None;
    let mut outcome = Outcome::default();
    for step in &replay.steps {
        match step {
            Step::Frame(time) => {
                let context = Context { time: *time, input };
                let report = session
                    .evaluate(&context)
                    .map_err(|error| format!("frame at {time}: {error}"))?;
                outcome.frames += 1;
                last = Some(report);
            }
            Step::Pointer { position, down } => {
                if let Some(position) = position {
                    input.pointer = *position;
                }
                input.pointer_down = *down;
            }
            Step::Key { code, down } => input.set_key(*code, *down),
            Step::Reset => session.reset_state(),
            Step::Expect {
                x,
                y,
                rgb,
                tolerance,
                line,
            } => {
                let output: ImageHandle = last
                    .as_ref()
                    .map(|report| report.output)
                    .ok_or_else(|| format!("line {line}: expect before any frame"))?;
                let actual = session
                    .runtime()
                    .host()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_step() {
        let replay = parse(
            "scene s.osn\nkernels k\nframe 0.5\npointer 1 2 down\npointer up\nkey 65 down\nreset\nexpect 1 2 = 0.1 0.2 0.3\nexpect 1 2 = 0 0 0 0.5\n",
            Path::new("/base"),
        )
        .unwrap();
        assert_eq!(replay.scene, Path::new("/base/s.osn"));
        assert_eq!(replay.kernels, Some(PathBuf::from("/base/k")));
        assert_eq!(replay.steps.len(), 7);
        assert!(matches!(replay.steps[6], Step::Expect { tolerance, .. } if tolerance == 0.5));
        let error = parse("frame 1\n", Path::new(".")).unwrap_err();
        assert!(error.message.contains("scene"), "{error}");
        let error = parse("scene s\nexpect 1 = 2\n", Path::new(".")).unwrap_err();
        assert_eq!(error.line, 2);
    }
}
