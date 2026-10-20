//! Declarative replay: an application's tests as data. A replay drives a
//! scene through scripted frames and inputs and checks what they produce.
//! Every form a line takes is in `docs/USAGE.md`, *Replay tests*, which
//! [`parse`] is held to.

use std::fmt;
use std::path::{Path, PathBuf};

use entelechy_graph::Report;
use entelechy_kernel::{Context, Handle, InputState, Runtime};

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
            &["scene ../scene.esn"],
        ),
        form(
            "library",
            "library <path>",
            "Kernels beyond the scene's `use`",
            &["library ../kernels/"],
        ),
        form(
            "allow",
            "allow file.write <dir> | allow file.delete <dir>",
            "lets the scene's outputs write (or remove and rename) files under the directory, relative to the replay file, as `--allow` does; what they write is kept for the run and never reaches the disk, so every run starts from the same files",
            &["allow file.write ../laws/"],
        ),
        form(
            "alike",
            "alike",
            "its checks hold two forms of one computation equal word for word, which they are only where every Kernel rounds alike, so `--round-apart` leaves the replay out",
            &["alike"],
        ),
        form(
            "frame",
            "frame <seconds> [to <seconds> every <seconds>]",
            "evaluates one frame at this session time; with `to` and `every`, a frame at each step from the first time to the last, both included",
            &["frame 0.0", "frame 0.1 to 2.0 every 0.1"],
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
            "pads",
            "pads <count>",
            "how many pads are connected from here on, what `pads()` reads",
            &["pads 1"],
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
            "type",
            "type \"<text>\"",
            "characters typed, what `typed()` reads on the next frame (32 a frame, the rest on the frames after); `\\\"`, `\\\\`, `\\n` and `\\t` inside the quotes, `#` among them no comment",
            &["type \"你好, Quill\""],
        ),
        form(
            "clipboard",
            "clipboard \"<text>\"",
            "the clipboard holding this text from here on, as a person copying elsewhere leaves it: what an input `= clipboard` reads; the same escapes as `type`",
            &["clipboard \"from elsewhere\""],
        ),
        form(
            "key",
            "key <code> down | up",
            "a key by the browser's code, as `key()` reads it: letters and digits by uppercase ASCII, space 32, enter 13, arrows 37 to 40, the rest in `help key`",
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
            "an element of a node's output, or a field of it (a struct's element is compared a field at a time; an array's element is named by its index, `.cells.2`): ints and bools exactly, floats and vectors within the tolerance; `i` counts along the Field, x fastest, then y, then z",
            &[
                "expect step.result[0].score = 1",
                "expect step.result[0].cells.2 = 0",
            ],
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
        form(
            "show",
            "show <node.port[i][.field]>",
            "prints an element of a node's output after the last frame, or a field of it, a struct field by field; it checks nothing, and the run reports what it shows",
            &["show step.result[0]", "show step.result[0].score"],
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
    /// Evaluate at a session time, in seconds.
    Frame(f64),
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
    /// Characters typed, for the next frames to read with `typed()`, as
    /// many a frame as one holds.
    Type(String),
    /// The clipboard's text from here on.
    Clipboard(String),
    /// A pad's axis pair: 0 left stick, 1 right stick, 2 triggers.
    Stick {
        /// Which pair.
        which: i32,
        /// Its two values.
        at: [f32; 2],
    },
    /// How many pads are connected from here on.
    Pads(u32),
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
        value: entelechy_kernel::Value,
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
    /// Print an element of a node's output after the last frame, or a
    /// field of it.
    Show {
        /// The node.
        node: String,
        /// Its output port.
        port: String,
        /// Element index.
        index: u32,
        /// Struct fields, outermost first; empty for the element itself.
        path: Vec<String>,
        /// Line number, for the report.
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
    /// The directories its scene's outputs may write files in, or remove
    /// and rename them in, resolved ([`store_for`]).
    pub allowed: Vec<crate::Allowance>,
    /// Whether its checks hold two forms of one computation equal word
    /// for word: `--round-apart` leaves it out.
    pub alike: bool,
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

/// The most frames one `frame ... to ... every` line evaluates.
const MAX_FRAMES: u64 = 1 << 20;

/// The text of a `type` or `clipboard` line after its keyword: one
/// quoted string, with `\"`, `\\`, `\n` and `\t`, then nothing but a
/// comment.
fn quoted_text(rest: &str, keyword: &str) -> Result<String, String> {
    let shape = format!("`{keyword}` takes one quoted text: {keyword} \"<text>\"");
    let mut chars = rest.trim_start().chars();
    if chars.next() != Some('"') {
        return Err(shape.clone());
    }
    let mut text = String::new();
    loop {
        match chars.next() {
            None => return Err(shape.clone()),
            Some('"') => break,
            Some('\\') => text.push(match chars.next() {
                Some('"') => '"',
                Some('\\') => '\\',
                Some('n') => '\n',
                Some('t') => '\t',
                other => {
                    return Err(format!(
                        "`{keyword}` knows `\\\"`, `\\\\`, `\\n` and `\\t`, not `\\{}`",
                        other.map(String::from).unwrap_or_default()
                    ));
                }
            }),
            Some(c) => text.push(c),
        }
    }
    let after = chars.as_str().trim();
    if !after.is_empty() && !after.starts_with('#') {
        return Err(shape.clone());
    }
    Ok(text)
}

/// Parses replay text; relative paths resolve against `base`.
pub fn parse(text: &str, base: &Path) -> Result<Replay, ReplayError> {
    let mut scene = None;
    let mut kernels = None;
    let mut steps = Vec::new();
    let mut alike = false;
    let mut allowed = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line = index as u32 + 1;
        let error = |message: String| ReplayError { line, message };
        // Typed and copied text is quoted, a `#` in it no comment.
        if let Some(rest) = raw.trim_start().strip_prefix("type ") {
            steps.push(Step::Type(quoted_text(rest, "type").map_err(error)?));
            continue;
        }
        if let Some(rest) = raw.trim_start().strip_prefix("clipboard ") {
            steps.push(Step::Clipboard(
                quoted_text(rest, "clipboard").map_err(error)?,
            ));
            continue;
        }
        let content = raw.split('#').next().unwrap_or("").trim();
        if content.is_empty() {
            continue;
        }
        let words: Vec<&str> = content.split_whitespace().collect();
        let number = |text: &str, what: &str| {
            text.parse::<f32>()
                .map_err(|_| error(format!("{what} `{text}` is not a number")))
        };
        let wide = |text: &str, what: &str| {
            text.parse::<f64>()
                .map_err(|_| error(format!("{what} `{text}` is not a number")))
        };
        match words.as_slice() {
            ["scene", path] => scene = Some(base.join(path)),
            ["library" | "kernels", path] => kernels = Some(base.join(path)),
            ["frame", time] => steps.push(Step::Frame(wide(time, "time")?)),
            ["frame", from, "to", last, "every", step] => {
                let (from, last, step) = (
                    wide(from, "time")?,
                    wide(last, "time")?,
                    wide(step, "step")?,
                );
                let forward = step > 0.0 && last >= from && last.is_finite();
                if !forward {
                    return Err(error(format!(
                        "frames run forward: `every` must be above 0 and `to` at or after the first time, not {from} to {last} every {step}"
                    )));
                }
                // Each time from the first, so steps never drift.
                let count = ((last - from) / step + 1e-3).floor() as u64 + 1;
                if count > MAX_FRAMES {
                    return Err(error(format!(
                        "{count} frames in one line; a line holds at most {MAX_FRAMES}"
                    )));
                }
                steps.extend((0..count).map(|i| Step::Frame(from + i as f64 * step)));
            }
            ["settle"] => steps.push(Step::Settle),
            ["alike"] => alike = true,
            ["allow", "file.write", dir] => {
                allowed.push(crate::Allowance::FileWrite(base.join(dir)));
            }
            ["allow", "file.delete", dir] => {
                allowed.push(crate::Allowance::FileDelete(base.join(dir)));
            }
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
                    "south" => entelechy_kernel::pad::SOUTH,
                    "east" => entelechy_kernel::pad::EAST,
                    "west" => entelechy_kernel::pad::WEST,
                    "north" => entelechy_kernel::pad::NORTH,
                    "l1" => entelechy_kernel::pad::LEFT_SHOULDER,
                    "r1" => entelechy_kernel::pad::RIGHT_SHOULDER,
                    "l3" => entelechy_kernel::pad::LEFT_THUMB,
                    "r3" => entelechy_kernel::pad::RIGHT_THUMB,
                    "start" => entelechy_kernel::pad::START,
                    "select" => entelechy_kernel::pad::SELECT,
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
            ["pads", count] => {
                steps.push(Step::Pads(count.parse::<u32>().map_err(|_| {
                    error(format!("pads `{count}` is not a count of pads"))
                })?))
            }
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
                let (node, port) = entelechy_graph::split_port(target)
                    .ok_or_else(|| error("`reset` takes nothing or `node.port`".to_owned()))?;
                steps.push(Step::ResetPort {
                    node: node.0.to_string(),
                    port,
                });
            }
            ["param", name, value @ ..] if !value.is_empty() => {
                let text = value.join(" ");
                let expr = entelechy_language::parse_literal(&text)
                    .map_err(|e| error(format!("param `{name}`: {e}")))?;
                let value = entelechy_compiler::literal(&expr)
                    .map_err(|e| error(format!("param `{name}`: {e}")))?;
                steps.push(Step::Param {
                    name: name.to_string(),
                    value,
                });
            }
            ["expect", "count", target, "=", count] => {
                let (node, port) = entelechy_graph::split_port(target)
                    .ok_or_else(|| error("`expect count` takes `node.port`".to_owned()))?;
                steps.push(Step::ExpectCount {
                    node: node.0.to_string(),
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
            ["show", target] if target.contains('[') => {
                let (node, port, index, path) = crate::parse_target(target).map_err(error)?;
                steps.push(Step::Show {
                    node,
                    port,
                    index,
                    path,
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
                        entelechy_language::did_you_mean(word, keywords.iter().copied())
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
        allowed,
        alike,
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
    /// What `show` steps printed, in order.
    pub shown: Vec<String>,
    /// Groups the backend refused to run fused, each said once.
    pub refused: Vec<String>,
    /// The nodes its outcome depends on: what its frames asked for, and
    /// every node those read, now or as committed state, by the graph,
    /// whether a frame ran them or spared them as unchanged.
    pub touched: std::collections::BTreeSet<String>,
}

impl Outcome {
    /// The report of a run: the groups refused fused, the shown values,
    /// then the failures, each a line under the replay's own.
    pub fn lines(&self) -> Vec<String> {
        self.refused
            .iter()
            .chain(&self.shown)
            .chain(&self.failures)
            .cloned()
            .collect()
    }
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
    run_fused(replay, store, runtime, true)
}

/// [`run_in`], fusing pointwise producers into their consumers or not.
pub fn run_fused(
    replay: &Replay,
    store: std::sync::Arc<dyn crate::Store>,
    runtime: Runtime,
    fusion: bool,
) -> Result<Outcome, String> {
    run_each(replay, store, runtime, fusion, None)
}

/// What a replay run hands each frame it evaluates, once evaluated: the
/// session and the frame's report. A run that hands its frames on
/// evaluates the scene's outputs every frame.
pub type Each<'a> = dyn FnMut(&mut Session, &Report) -> Result<(), String> + 'a;

/// [`run_fused`], handing each evaluated frame to `each`.
pub fn run_each(
    replay: &Replay,
    store: std::sync::Arc<dyn crate::Store>,
    runtime: Runtime,
    fusion: bool,
    each: Option<&mut Each<'_>>,
) -> Result<Outcome, String> {
    let store = store_for(replay, store);
    let mut session = Session::open_in(store, &replay.scene, replay.kernels.as_deref(), runtime);
    session.set_fusion(fusion);
    let events = session.sync();
    allow_on(replay, &mut session)?;
    played_on(replay, &mut session, events, each)
}

/// The store a replay's session reads and writes: `store`, or, where the
/// replay allows its outputs to write, `store` with every write kept
/// beside it ([`crate::Kept`]): a run changes nothing on disk, and every
/// run starts from the same files.
pub fn store_for(
    replay: &Replay,
    store: std::sync::Arc<dyn crate::Store>,
) -> std::sync::Arc<dyn crate::Store> {
    if replay.allowed.is_empty() {
        store
    } else {
        std::sync::Arc::new(crate::Kept::new(store))
    }
}

/// Gives `session` what `replay` allows ([`Replay::allowed`]).
pub fn allow_on(replay: &Replay, session: &mut Session) -> Result<(), String> {
    for allowance in &replay.allowed {
        match allowance {
            crate::Allowance::FileWrite(dir) => session.allow_writes(dir)?,
            crate::Allowance::FileDelete(dir) => session.allow_deletes(dir)?,
            _ => {}
        }
    }
    Ok(())
}

/// Whether the session's documents loaded: the rejections said, if any.
fn loaded(events: Vec<crate::Event>) -> Result<(), String> {
    let rejected: Vec<String> = events
        .into_iter()
        .filter(|event| {
            matches!(
                event,
                crate::Event::Rejected { .. } | crate::Event::SceneRejected(_)
            )
        })
        .map(|event| event.to_string())
        .collect();
    if rejected.is_empty() {
        Ok(())
    } else {
        Err(rejected.join("\n"))
    }
}

/// A replay played on a session that has just read its documents.
fn played_on(
    replay: &Replay,
    session: &mut Session,
    events: Vec<crate::Event>,
    mut each: Option<&mut Each<'_>>,
) -> Result<Outcome, String> {
    loaded(events)?;
    let session = &mut *session;
    let mut replayer = Replayer::start(replay, session);
    let mut asked: std::collections::BTreeSet<entelechy_graph::NodeId> =
        std::collections::BTreeSet::new();
    while let Some(frame) = replayer.next_frame(session)? {
        // A scene allowed to write reads before each frame what it wrote
        // after the last: a law it wrote is in its place a frame later.
        if !replay.allowed.is_empty() {
            let _ = session.sync();
        }
        let at = frame.context.time;
        asked.extend(frame.demand.iter().cloned());
        let mut demand = frame.demand;
        if each.is_some() {
            demand.extend(replayer.producers.iter().cloned());
        }
        let report = session
            .evaluate_for(&frame.context, &demand)
            .map_err(|error| format!("frame at {at}: {error}"))?;
        if let Some(each) = &mut each {
            each(session, &report).map_err(|error| format!("frame at {at}: {error}"))?;
        }
        replayer.evaluated(report);
    }
    if let Some(graph) = session.graph() {
        replayer.outcome.touched = reaching(graph, &asked);
    }
    Ok(replayer.outcome)
}

/// The nodes `asked` and every node they read, now or as committed state
/// (with what feeds the state first), through the graph.
fn reaching(
    graph: &entelechy_graph::Graph,
    asked: &std::collections::BTreeSet<entelechy_graph::NodeId>,
) -> std::collections::BTreeSet<String> {
    let mut seen = std::collections::BTreeSet::new();
    let mut open: Vec<entelechy_graph::NodeId> = asked.iter().cloned().collect();
    while let Some(id) = open.pop() {
        if !seen.insert(id.0.to_string()) {
            continue;
        }
        let Some(node) = graph.node(&id) else {
            continue;
        };
        for binding in node.inputs.values() {
            let sources = match binding {
                entelechy_graph::Binding::Now(source) => vec![source],
                entelechy_graph::Binding::Previous {
                    source, initial, ..
                } => vec![source, initial],
            };
            for source in sources {
                if let entelechy_graph::Source::Node { node, .. } = source {
                    open.push(node.clone());
                }
            }
        }
    }
    seen
}

/// What changed in a scene's documents since its session last read them.
#[derive(Clone, Debug, Default)]
pub struct Changed {
    /// The graph itself changed, or a document was refused: every replay
    /// of the scene plays again.
    pub whole: bool,
    /// The Kernels published again or retired.
    pub kernels: std::collections::BTreeSet<String>,
    /// The inputs read again.
    pub inputs: std::collections::BTreeSet<String>,
    /// What the sync said, to show.
    pub said: Vec<String>,
}

/// Replays played one after another, each scene's on one session kept
/// for them all: a replay starts the world over
/// ([`Session::start_over`]), so what it sees is what a fresh session
/// would show it, while the Kernels stay compiled and prepared, the
/// documents parsed, and the outputs no state feeds (a world baked from
/// its seed) are spared. A document changed between replays is read
/// again as a running session reads it. Rounded apart, each way keeps a
/// session of its own.
pub struct Resident {
    warm: crate::Warm,
    store: std::sync::Arc<dyn crate::Store>,
    fusion: bool,
    sessions: std::collections::HashMap<
        (std::path::PathBuf, Option<std::path::PathBuf>, Option<u32>),
        Session,
    >,
}

impl Resident {
    /// Over backends opened once and documents read through `store`.
    pub fn new(warm: crate::Warm, store: std::sync::Arc<dyn crate::Store>, fusion: bool) -> Self {
        Self {
            warm,
            store,
            fusion,
            sessions: std::collections::HashMap::new(),
        }
    }

    /// Plays `replay` on its scene's session, rounding apart in `split`
    /// when one is given ([`crate::apart`]).
    pub fn play(&mut self, replay: &Replay, split: Option<u32>) -> Result<Outcome, String> {
        self.play_as(None, replay, split)
    }

    /// [`Resident::play`], remembering the run as `name`: played again
    /// under that name, each frame takes from the same frame of this run
    /// what nothing it reads has changed ([`Session::recall_as`]).
    pub fn play_as(
        &mut self,
        name: Option<&str>,
        replay: &Replay,
        split: Option<u32>,
    ) -> Result<Outcome, String> {
        let runtime = |warm: &crate::Warm| match split {
            Some(split) => warm.runtime_apart(split),
            None => warm.runtime(),
        };
        // What a replay that writes wrote is its run's alone: it plays on
        // a session of its own, from the files as they are.
        if !replay.allowed.is_empty() {
            let store = store_for(replay, std::sync::Arc::clone(&self.store));
            let mut session = Session::open_in(
                store,
                &replay.scene,
                replay.kernels.as_deref(),
                runtime(&self.warm),
            );
            session.set_fusion(self.fusion);
            let events = session.sync();
            allow_on(replay, &mut session)?;
            return played_on(replay, &mut session, events, None);
        }
        let key = (replay.scene.clone(), replay.kernels.clone(), split);
        let events = match self.sessions.get_mut(&key) {
            Some(session) => session.start_over(),
            None => {
                let mut session = Session::open_in(
                    std::sync::Arc::clone(&self.store),
                    &replay.scene,
                    replay.kernels.as_deref(),
                    runtime(&self.warm),
                );
                session.set_fusion(self.fusion);
                let events = session.sync();
                self.sessions.insert(key.clone(), session);
                events
            }
        };
        let session = self.sessions.get_mut(&key).expect("kept above");
        session.recall_as(name.map(str::to_owned));
        let played = played_on(replay, session, events, None);
        session.recall_as(None);
        if played.is_err() {
            // A failed run leaves the session as no replay should find it.
            self.sessions.remove(&key);
        }
        played
    }

    /// Reads every kept session's documents again, saying per scene what
    /// changed ([`Changed`]). A graph rebuilt the same is no change of its
    /// own: what changed is what its Kernels and inputs say.
    pub fn changed(&mut self) -> std::collections::HashMap<std::path::PathBuf, Changed> {
        let mut changed: std::collections::HashMap<std::path::PathBuf, Changed> =
            std::collections::HashMap::new();
        for ((scene, _, _), session) in &mut self.sessions {
            let before = session.graph().cloned();
            let events = session.sync();
            let entry = changed.entry(scene.clone()).or_default();
            for event in &events {
                match event {
                    crate::Event::Published { id, .. } | crate::Event::Retired { id, .. } => {
                        entry.kernels.insert(id.to_string());
                    }
                    crate::Event::Loaded { name, .. }
                    | crate::Event::Text { name, .. }
                    | crate::Event::Absent { name, .. } => {
                        entry.inputs.insert(name.clone());
                    }
                    crate::Event::SceneLoaded => {
                        if session.graph() != before.as_ref() {
                            entry.whole = true;
                        }
                    }
                    crate::Event::Rejected { .. } | crate::Event::SceneRejected(_) => {
                        entry.whole = true;
                        entry.said.push(event.to_string());
                    }
                    _ => {}
                }
            }
        }
        changed
    }

    /// Whether what changed reaches a replay of `scene` that touched
    /// `touched`: a node it touched runs a Kernel published again or
    /// retired, or reads an input read again.
    pub fn affects(
        &self,
        changed: &std::collections::HashMap<std::path::PathBuf, Changed>,
        scene: &std::path::Path,
        touched: &std::collections::BTreeSet<String>,
    ) -> bool {
        let Some(change) = changed.get(scene) else {
            return false;
        };
        if change.whole {
            return true;
        }
        if change.kernels.is_empty() && change.inputs.is_empty() {
            return false;
        }
        let Some(graph) = self
            .sessions
            .iter()
            .find(|((kept, _, _), _)| kept == scene)
            .and_then(|(_, session)| session.graph())
        else {
            return true;
        };
        touched.iter().any(|name| {
            let Some(node) = graph.node(&entelechy_graph::NodeId(name.clone().into())) else {
                return true;
            };
            change.kernels.contains(&node.kernel.to_string())
                || node.inputs.values().any(|binding| {
                    let sources = match binding {
                        entelechy_graph::Binding::Now(source) => vec![source],
                        entelechy_graph::Binding::Previous {
                            source, initial, ..
                        } => vec![source, initial],
                    };
                    sources.into_iter().any(|source| {
                        matches!(source, entelechy_graph::Source::External(input) if change.inputs.contains(input))
                    })
                })
        })
    }

    /// The documents and files every kept session reads, to watch.
    pub fn watched(&self) -> Vec<std::path::PathBuf> {
        let mut all: Vec<std::path::PathBuf> =
            self.sessions.values().flat_map(Session::watched).collect();
        all.sort();
        all.dedup();
        all
    }
}

/// A frame a replay asks for: its context and what it must evaluate. A
/// caller that shows the picture (a window) evaluates everything.
pub struct Frame {
    /// The time, input and window of this frame.
    pub context: Context,
    /// The named nodes, and the picture when a pixel expectation follows.
    pub demand: std::collections::BTreeSet<entelechy_graph::NodeId>,
}

/// A replay played a frame at a time over a caller's session: the steps
/// between frames are applied, and the next frame is handed back for the
/// caller to evaluate.
pub struct Replayer {
    steps: Vec<Step>,
    next: usize,
    input: InputState,
    /// Characters typed and not yet read by a frame.
    typing: String,
    window: ([f32; 2], f32),
    last: Option<Report>,
    named: std::collections::BTreeSet<entelechy_graph::NodeId>,
    producers: Vec<entelechy_graph::NodeId>,
    /// What the replay has observed so far.
    pub outcome: Outcome,
}

impl Replayer {
    /// Starts `replay` over a synced `session`, pinning the nodes the
    /// expectations name.
    pub fn start(replay: &Replay, session: &mut Session) -> Self {
        // What an expectation names, by the node that produces it: a
        // placed graph's output names the node inside that gives it.
        let named: std::collections::BTreeSet<entelechy_graph::NodeId> = replay
            .steps
            .iter()
            .filter_map(|step| match step {
                Step::ExpectElement { node, port, .. }
                | Step::ExpectCount { node, port, .. }
                | Step::Show { node, port, .. } => Some(
                    session
                        .producer(&entelechy_graph::NodeId(node.clone().into()), port)
                        .0,
                ),
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
                        entelechy_graph::Source::Node { node, .. } => Some(node.clone()),
                        entelechy_graph::Source::External(_) => None,
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self {
            steps: replay.steps.clone(),
            next: 0,
            input: InputState::default(),
            typing: String::new(),
            window: ([0.0, 0.0], 1.0),
            last: None,
            named,
            producers,
            outcome: Outcome::default(),
        }
    }

    /// The time of the next frame, when there is one.
    pub fn next_time(&self) -> Option<f64> {
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
                // The characters typed, as many as a frame holds.
                let mut input = self.input;
                self.typing = input.type_text(&self.typing).to_owned();
                let context = Context {
                    time: *time,
                    // A replay has no sound device: nothing is queued.
                    rate: 0.0,
                    queued: 0,
                    input,
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
        self.outcome
            .refused
            .extend(report.refused.iter().map(ToString::to_string));
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
            Step::Type(text) => self.typing.push_str(text),
            Step::Clipboard(text) => session
                .place_clipboard(text)
                .map_err(|error| format!("clipboard: {error}"))?,
            Step::Pads(count) => input.pads = *count,
            Step::Reset => session.reset_state(),
            Step::ResetPort { node, port } => {
                session
                    .reset_port(&entelechy_graph::NodeId(node.clone().into()), port)
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
            Step::Show {
                node,
                port,
                index,
                path,
                line,
            } => {
                if last.is_none() {
                    return Err(format!("line {line}: show before any frame"));
                }
                let target = format!(
                    "{node}.{port}[{index}]{}",
                    path.iter().map(|p| format!(".{p}")).collect::<String>()
                );
                let value = shown(session, node, port, *index, path)
                    .map_err(|message| format!("line {line}: {target}: {message}"))?;
                outcome
                    .shown
                    .push(format!("line {line}: {target} is {value}"));
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
                match session.element_count(&entelechy_graph::NodeId(node.clone().into()), port) {
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
                // Where it ran in the last frame, which a run's size may
                // decide; where it is prepared to run when it has not run.
                let ran = last.as_ref().and_then(|report| {
                    report
                        .placements
                        .iter()
                        .find(|placed| placed.node.0 == *node)
                        .map(|placed| entelechy_kernel::Placement {
                            backend: placed.backend,
                            routed: placed.routed.clone(),
                        })
                });
                let placements = session.placements();
                let prepared = placements
                    .iter()
                    .find(|(id, _)| id == node)
                    .map(|(_, placement)| placement.clone());
                let Some(placement) = ran.or(prepared) else {
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

/// A number as a replay writes it.
fn written(n: &entelechy_ir::Number) -> String {
    use entelechy_ir::Number;
    match n {
        Number::Float(v) => format!("{v}"),
        Number::Double(v) => format!("{v}"),
        Number::Int(v) => v.to_string(),
        Number::Long(v) => v.to_string(),
        Number::Bool(v) => v.to_string(),
    }
}

/// An element field of a node's output as `show` prints it: its numbers,
/// or a struct's fields in braces, each named.
fn shown(
    session: &mut Session,
    node: &str,
    port: &str,
    index: u32,
    path: &[String],
) -> Result<String, String> {
    let node_id = entelechy_graph::NodeId(node.to_owned().into());
    let (ty, words) = session.element_field(&node_id, port, index, path)?;
    if matches!(ty, entelechy_ir::Type::Struct(_)) {
        let fields = session.element_fields(&node_id, port, path)?;
        let mut parts = Vec::with_capacity(fields.len());
        for field in fields {
            let mut deeper = path.to_vec();
            deeper.push(field.clone());
            parts.push(format!(
                "{field}: {}",
                shown(session, node, port, index, &deeper)?
            ));
        }
        return Ok(format!("{{ {} }}", parts.join(", ")));
    }
    let numbers = entelechy_ir::template(ty, &[])
        .rebuilt(&mut words.iter().copied())
        .numbers();
    Ok(numbers.iter().map(written).collect::<Vec<_>>().join(" "))
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
    use entelechy_ir::{Number, Type};
    let node_id = entelechy_graph::NodeId(node.to_owned().into());
    let (ty, words) = session.element_field(&node_id, port, index, path)?;
    if matches!(ty, Type::Struct(_)) {
        return Err("name a field of the struct; whole structs are not compared".into());
    }
    let numbers = entelechy_ir::template(ty, &[])
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
    let actual: Vec<String> = numbers.iter().map(written).collect();
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
    /// `docs/USAGE.md` shows the forms; `ENTELECHY_BLESS=1` writes them again.
    #[test]
    fn the_forms_parse_cover_every_step_and_are_shown() {
        let lines: Vec<&str> = forms()
            .iter()
            .flat_map(|form| form.examples.iter().copied())
            .collect();
        let replay = parse(&lines.join("\n"), Path::new(".")).unwrap();
        assert!(replay.kernels.is_some(), "`library` is a form");
        let mut seen = [false; 19];
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
                Step::Pads(_) => 15,
                Step::Show { .. } => 16,
                Step::Type(_) => 17,
                Step::Clipboard(_) => 18,
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
            if std::env::var_os("ENTELECHY_BLESS").is_some() {
                std::fs::write(&path, format!("{}{table}{}", &text[..start], &text[end..]))
                    .unwrap();
            } else {
                panic!(
                    "docs/USAGE.md's replay forms are not the parser's; write them again with ENTELECHY_BLESS=1 cargo test -p entelechy-session --lib\n{table}"
                );
            }
        }
    }

    /// A range of frames is a frame at each step, both ends included, each
    /// time counted from the first; a range that runs backward is refused.
    #[test]
    fn a_range_of_frames_is_a_frame_at_each_step() {
        let replay = parse("scene s.esn\nframe 0.1 to 1.0 every 0.1\n", Path::new(".")).unwrap();
        let times: Vec<f64> = replay
            .steps
            .iter()
            .map(|step| match step {
                Step::Frame(time) => *time,
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(times.len(), 10);
        for (i, time) in (1..=10).zip(&times) {
            assert!((time - f64::from(i) / 10.0).abs() < 1e-12, "{times:?}");
        }
        let error = parse("scene s.esn\nframe 1 to 0 every 0.1\n", Path::new(".")).unwrap_err();
        assert!(
            error.message.contains("frames run forward"),
            "{}",
            error.message
        );
        let error = parse("scene s.esn\nframe 0 to 1 every 0\n", Path::new(".")).unwrap_err();
        assert!(
            error.message.contains("frames run forward"),
            "{}",
            error.message
        );
    }

    /// A line no form reads names the steps, or how its keyword is written.
    #[test]
    fn a_line_no_form_reads_says_what_is() {
        let error = parse("scene s.esn\nfram 0", Path::new(".")).unwrap_err();
        assert!(
            error.message.contains("did you mean `frame`"),
            "{}",
            error.message
        );
        let error = parse("scene s.esn\nstick up 1 2", Path::new(".")).unwrap_err();
        assert!(
            error.message.contains("stick is `left`"),
            "{}",
            error.message
        );
        let error = parse("scene s.esn\nwheel 1", Path::new(".")).unwrap_err();
        assert!(
            error.message.contains("`wheel` is written wheel <x> <y>"),
            "{}",
            error.message
        );
    }

    #[test]
    fn an_element_is_held_equal_or_to_one_side() {
        let replay = parse(
            "scene s.esn\nexpect a.result[0] = 1\nexpect a.result[0] < 2\nexpect a.result[1].v >= 0.5 0.5 0.5\nexpect a.result[2] > -1\nexpect a.result[3] <= 0\n",
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
        let error = parse("scene s.esn\nexpect a.result[0] ~ 1\n", Path::new(".")).unwrap_err();
        assert!(error.message.contains("cannot read"), "{error}");
        assert!(Compare::Below.holds(1.0, 2.0) && !Compare::Below.holds(2.0, 2.0));
        assert!(Compare::AtLeast.holds(2.0, 2.0) && !Compare::Above.holds(2.0, 2.0));
    }

    #[test]
    fn parses_every_step() {
        let replay = parse(
            "scene s.esn\nkernels k\nframe 0.5\npointer 1 2 down\npointer up\nkey 65 down\nreset\nexpect 1 2 = 0.1 0.2 0.3\nexpect 1 2 = 0 0 0 0.5\nbutton right down\nwheel 0 -1\n",
            Path::new("/base"),
        )
        .unwrap();
        assert_eq!(replay.scene, Path::new("/base/s.esn"));
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
            "scene s.esn\nexpect step.result[0].score = 1\nexpect mix.result[3] = 0.5 0.25 0.02\n",
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
