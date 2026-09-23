//! Measuring a scene the same way twice. `run --profile` says where one
//! frame went; this steps a scene for a count of frames, several times,
//! on one backend or on all of them, and reports the least of the runs,
//! which is the measurement least disturbed by everything else on the
//! machine.
//!
//! The clock is the frame count over the rate, as `render`'s is, so the
//! work measured is the same work every time. A run's first frame is
//! thrown away: it compiles the Kernels, allocates the Fields, and warms
//! the caches, and measuring it would measure the machine's morning.
//!
//! A measurement written with `--json` can be read back with `--compare`
//! on a later build, which is how a change is shown to cost nothing.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use orla_runtime::Context;

use crate::Session;

/// How to measure.
#[derive(Clone, Debug)]
pub struct Options {
    /// The scene.
    pub scene: PathBuf,
    /// A library directory or file beside the scene's own.
    pub kernels: Option<PathBuf>,
    /// Backends to measure, in order.
    pub backends: Vec<String>,
    /// Frames in a run, after the warm-up frame.
    pub frames: u32,
    /// Runs of that many frames.
    pub repeat: u32,
    /// Frames per second the clock advances by.
    pub rate: f32,
    /// The window a scene sized by `window()` fills, and its scale.
    pub window: Option<([f32; 2], f32)>,
    /// Graph parameters to override.
    pub params: Vec<(String, orla_runtime::Value)>,
    /// External Fields to bind.
    pub inputs: Vec<(String, PathBuf)>,
    /// Whether fusion is on.
    pub fusion: bool,
    /// Words the Host may hold at once.
    pub memory: usize,
    /// Whether to measure each node as well as the frame.
    pub nodes: bool,
    /// Where to write the measurement as JSON.
    pub json: Option<PathBuf>,
    /// A measurement to compare against.
    pub compare: Option<PathBuf>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            scene: PathBuf::new(),
            kernels: None,
            backends: vec!["jit".to_owned()],
            frames: 30,
            repeat: 5,
            rate: 60.0,
            window: None,
            params: Vec::new(),
            inputs: Vec::new(),
            fusion: true,
            memory: crate::DEFAULT_BUDGET_WORDS,
            nodes: false,
            json: None,
            compare: None,
        }
    }
}

/// What one backend measured.
#[derive(Clone, Debug, PartialEq)]
pub struct Measured {
    /// The backend measured.
    pub backend: String,
    /// Milliseconds a frame in the best run.
    pub best: f64,
    /// Milliseconds a frame in the middle run, which says how much the
    /// machine wandered while measuring.
    pub median: f64,
    /// Milliseconds a frame each node took, when asked for; a node that
    /// did not run in a frame is counted over the frames all the same,
    /// as the frame's cost is what is being divided.
    pub nodes: Vec<(String, f64)>,
    /// Whether any node ran at all after the warm-up frame. A scene
    /// that does not change with the clock reuses everything, and what
    /// is measured is the check that nothing changed.
    pub ran: bool,
    /// Whether the node numbers are what the device's own clock said
    /// about the work. Where a driver takes the timestamps and resolves
    /// nothing, they are the call and its wait instead, less the floor
    /// below, and a reader has to be told which.
    pub device_clock: bool,
    /// What measuring one node cost before the node did anything, in
    /// milliseconds, already taken off every number above.
    pub floor: f64,
}

/// Measures every backend the options name.
pub fn bench(options: &Options) -> Result<Vec<Measured>, String> {
    let mut measured = Vec::new();
    for backend in &options.backends {
        measured.push(one(options, backend)?);
    }
    Ok(measured)
}

/// Measures one backend: a session of its own, then the measurement.
fn one(options: &Options, backend: &str) -> Result<Measured, String> {
    let mut session = open(options, backend)?;
    let (size, scale) = options.window.unwrap_or(([0.0, 0.0], 1.0));
    let base = Context {
        window: size,
        scale,
        ..Context::default()
    };
    measure(
        &mut session,
        backend,
        options.frames,
        options.repeat,
        options.rate,
        base,
        options.nodes,
    )
}

/// Measures the session as it stands: a warm-up frame, then `repeat`
/// runs of `frames` frames, and, when asked, one more run that times
/// every node. The clock advances by the frame count over `rate` from
/// the context given, as `advance` advances it, so what the session is
/// left holding is the last frame's.
pub fn measure(
    session: &mut Session,
    backend: &str,
    frames: u32,
    repeat: u32,
    rate: f32,
    base: Context,
    nodes: bool,
) -> Result<Measured, String> {
    let options = Options {
        frames,
        repeat,
        rate,
        nodes,
        ..Options::default()
    };
    let options = &options;
    let context = |frame: u32| Context {
        time: base.time + frame as f32 / rate,
        ..base
    };
    // The warm-up frame compiles and allocates; its time is the
    // machine's morning, not the scene's.
    session
        .evaluate(&context(0))
        .map_err(|e| format!("{backend}: {e}"))?;
    let mut runs: Vec<f64> = Vec::new();
    let mut frame = 1;
    let mut ran = false;
    for _ in 0..options.repeat.max(1) {
        let started = Instant::now();
        for _ in 0..options.frames.max(1) {
            let report = session
                .evaluate(&context(frame))
                .map_err(|e| format!("{backend}: {e}"))?;
            ran |= !report.executed.is_empty();
            frame += 1;
            // The device is asked to run dry inside the measurement, as
            // a Player's frame ends by presenting: a dispatch is
            // submitted and not waited for, so without this a GPU's
            // frame would be the time it took to ask for one.
            session.runtime().settle_device();
        }
        let took = started.elapsed().as_secs_f64() * 1000.0;
        runs.push(took / f64::from(options.frames.max(1)));
    }
    runs.sort_by(f64::total_cmp);
    let run = if options.nodes {
        node_times(session, options, &context, &mut frame, backend)?
    } else {
        NodeRun {
            nodes: Vec::new(),
            device_clock: false,
            floor: 0.0,
        }
    };
    Ok(Measured {
        backend: backend.to_owned(),
        best: runs[0],
        median: runs[runs.len() / 2],
        nodes: run.nodes,
        ran,
        device_clock: run.device_clock,
        floor: run.floor,
    })
}

/// What one measured run of the nodes came to: each node's
/// milliseconds a frame, whether those are the device's own clock or
/// the call and its wait, and what measuring one node cost before it
/// did anything.
struct NodeRun {
    nodes: Vec<(String, f64)>,
    device_clock: bool,
    floor: f64,
}

/// One more run with the device waited for after every node, which is
/// what makes a GPU's numbers the work rather than the submission, and
/// which is why it is not the run the frame time comes from.
fn node_times(
    session: &mut Session,
    options: &Options,
    context: &impl Fn(u32) -> Context,
    frame: &mut u32,
    backend: &str,
) -> Result<NodeRun, String> {
    session.set_measured(true);
    let mut totals: BTreeMap<String, f64> = BTreeMap::new();
    let frames = options.frames.max(1);
    for _ in 0..frames {
        let report = session
            .evaluate(&context(*frame))
            .map_err(|e| format!("{backend}: {e}"))?;
        *frame += 1;
        for (node, took) in &report.timings {
            *totals.entry(node.0.clone()).or_default() += took.as_secs_f64() * 1000.0;
        }
    }
    session.set_measured(false);
    // What measuring a node costs before the node does anything comes
    // off every node: one submission and one wait, the same for all of
    // them, which is a tenth of a millisecond on a device and nothing
    // at all where the caller's clock is the whole measurement. Asked
    // after the run, because a backend only knows once it has tried.
    let floor = session.measurement_floor().as_secs_f64() * 1000.0;
    // A node placed several times, or a graph placed several times, is
    // one line: `steps/*.spread/*` and how many there were. Sixty-four
    // lines of a tenth of a millisecond say less than their sum does.
    let mut grouped: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    for (node, total) in totals {
        let entry = grouped.entry(pattern(&node)).or_default();
        entry.0 += (total / f64::from(frames) - floor).max(0.0);
        entry.1 += 1;
    }
    let mut nodes: Vec<(String, f64)> = grouped
        .into_iter()
        .map(|(node, (took, count))| {
            let name = if count > 1 {
                format!("{node} ({count})")
            } else {
                node
            };
            (name, took)
        })
        .collect();
    nodes.sort_by(|a, b| b.1.total_cmp(&a.1));
    Ok(NodeRun {
        nodes,
        device_clock: session.measures_the_device(),
        floor,
    })
}

/// What a node's name is without its repetitions: `steps/3.spread/6`
/// is one of the `steps/*.spread/*`.
fn pattern(node: &str) -> String {
    node.split('.')
        .map(|part| match part.split_once('/') {
            Some((name, count)) if count.chars().all(|c| c.is_ascii_digit()) => {
                format!("{name}/*")
            }
            _ => part.to_owned(),
        })
        .collect::<Vec<_>>()
        .join(".")
}

fn open(options: &Options, backend: &str) -> Result<Session, String> {
    let runtime = Session::runtime_for(backend, options.memory)?;
    let mut session = Session::open_with(&options.scene, options.kernels.as_deref(), runtime);
    for (name, path) in &options.inputs {
        session
            .bind_external(name, path)
            .map_err(|e| e.to_string())?;
    }
    for (name, value) in &options.params {
        session.set_param(name, *value);
    }
    session.set_fusion(options.fusion);
    let events = session.sync();
    let rejected: Vec<String> = events
        .iter()
        .filter_map(|event| match event {
            crate::Event::Rejected { path, error } => Some(format!("{}:{error}", path.display())),
            _ => None,
        })
        .collect();
    if !rejected.is_empty() {
        return Err(rejected.join("\n"));
    }
    Ok(session)
}

/// The measurement as a table: a line a backend, the fastest first
/// backend as the one the others are compared against, and, when nodes
/// were measured, what each cost.
pub fn table(options: &Options, measured: &[Measured], before: Option<&[Measured]>) -> String {
    let mut out = String::new();
    let window = match options.window {
        Some(([w, h], scale)) => format!(", window {w}x{h}@{scale}"),
        None => String::new(),
    };
    out.push_str(&format!(
        "{}: {} frame(s) a run, {} run(s){window}\n\n",
        options.scene.display(),
        options.frames.max(1),
        options.repeat.max(1)
    ));
    let first = measured.first().map(|m| m.best).unwrap_or(1.0);
    let against = measured
        .first()
        .map(|m| m.backend.clone())
        .unwrap_or_default();
    out.push_str(&format!(
        "backend   frame ms    median   vs {against:<8}   was\n"
    ));
    for entry in measured {
        let against = if entry.best > 0.0 {
            format!("{:.2}x", first / entry.best)
        } else {
            "-".to_owned()
        };
        let was = before
            .and_then(|before| before.iter().find(|b| b.backend == entry.backend))
            .map(|b| {
                let change = (entry.best - b.best) / b.best * 100.0;
                format!("{:8.2} ms {change:+.1}%", b.best)
            })
            .unwrap_or_default();
        out.push_str(&format!(
            "{:<9} {:8.2}  {:8.2}  {:>9}    {was}\n",
            entry.backend, entry.best, entry.median, against
        ));
    }
    if measured.iter().any(|entry| !entry.ran) {
        out.push_str(
            "\nnothing ran after the first frame: this scene does not change with the clock, so what is measured is the finding of that out\n",
        );
    }
    for entry in measured {
        if entry.nodes.is_empty() {
            continue;
        }
        let how = if entry.device_clock {
            "what the device's own clock said each one took".to_owned()
        } else if entry.floor > 0.0 {
            format!(
                "the call and the wait after it, less the {:.3} ms measuring one node costs before it does anything (this driver takes the timestamps and resolves nothing)",
                entry.floor
            )
        } else {
            "the call, which is the whole of the work on this backend".to_owned()
        };
        out.push_str(&format!(
            "\n{}, ms a frame a node — {how}:\n",
            entry.backend
        ));
        for (node, took) in &entry.nodes {
            out.push_str(&format!("  {node:<28} {took:8.3}\n"));
        }
        // What the parts come to against the whole. A node measured
        // alone waits for its own work, so the setting up of the next
        // one no longer hides behind it: the difference is what a frame
        // overlaps, and naming it is the difference between a profile
        // that can be read and one that quietly does not add up.
        let sum: f64 = entry.nodes.iter().map(|(_, took)| took).sum();
        out.push_str(&format!(
            "  {:<28} {sum:8.3} against a frame of {:.3}; a node runs alone here, and what a frame overlaps it does not\n",
            "(sum)", entry.best
        ));
    }
    out
}

/// The measurement as JSON, for `--compare` on a later build.
pub fn json(options: &Options, measured: &[Measured]) -> String {
    let backends: Vec<String> = measured
        .iter()
        .map(|entry| {
            let nodes: Vec<String> = entry
                .nodes
                .iter()
                .map(|(node, took)| format!("      {{\"node\": {node:?}, \"ms\": {took:.4}}}"))
                .collect();
            format!(
                "    {{\"backend\": {:?}, \"best_ms\": {:.4}, \"median_ms\": {:.4}, \"nodes\": [\n{}\n    ]}}",
                entry.backend,
                entry.best,
                entry.median,
                nodes.join(",\n")
            )
        })
        .collect();
    format!(
        "{{\n  \"scene\": {:?},\n  \"frames\": {},\n  \"repeat\": {},\n  \"backends\": [\n{}\n  ]\n}}\n",
        options.scene.display().to_string(),
        options.frames.max(1),
        options.repeat.max(1),
        backends.join(",\n")
    )
}

/// Reads back what `--json` wrote. Only the numbers a comparison needs
/// are read, and a file that is not one is a failure rather than a
/// silent zero.
pub fn read(path: &Path) -> Result<Vec<Measured>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut measured = Vec::new();
    for part in text.split("{\"backend\":").skip(1) {
        let backend = part
            .split('"')
            .nth(1)
            .ok_or_else(|| format!("{}: no backend name", path.display()))?;
        let number = |key: &str| -> Result<f64, String> {
            part.split(&format!("\"{key}\":"))
                .nth(1)
                .and_then(|rest| rest.split(&[',', '}'][..]).next())
                .and_then(|text| text.trim().parse::<f64>().ok())
                .ok_or_else(|| format!("{}: no `{key}` for {backend}", path.display()))
        };
        measured.push(Measured {
            backend: backend.to_owned(),
            best: number("best_ms")?,
            median: number("median_ms")?,
            nodes: Vec::new(),
            ran: true,
            device_clock: false,
            floor: 0.0,
        });
    }
    if measured.is_empty() {
        return Err(format!("{}: no measurements", path.display()));
    }
    Ok(measured)
}
