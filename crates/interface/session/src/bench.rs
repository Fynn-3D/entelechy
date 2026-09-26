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

use orla_kernel::Context;

use crate::Session;

/// How to measure.
#[derive(Clone, Debug)]
pub struct Options {
    /// The session each backend opens: the scene, its library, inputs,
    /// params, fusion and limits. Its backend is not read; `backends`
    /// says which.
    pub session: crate::SessionOptions,
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
            session: crate::SessionOptions::default(),
            backends: vec!["jit".to_owned()],
            frames: 30,
            repeat: 5,
            rate: 60.0,
            window: None,
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
    /// What each node took a frame, when asked for; a node that did not
    /// run in a frame is counted over the frames all the same, as the
    /// frame's cost is what is being divided.
    pub nodes: Vec<NodeTime>,
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
    nodes: Vec<NodeTime>,
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
    // A node's time, where it ran, and its traffic, over every frame.
    #[derive(Default)]
    struct Total {
        ms: f64,
        backend: String,
        routed: Option<String>,
        uploaded: u64,
        downloaded: u64,
    }
    let mut totals: BTreeMap<String, Total> = BTreeMap::new();
    let frames = options.frames.max(1);
    for _ in 0..frames {
        let report = session
            .evaluate(&context(*frame))
            .map_err(|e| format!("{backend}: {e}"))?;
        *frame += 1;
        for ((node, took), placed) in report.timings.iter().zip(&report.placements) {
            let total = totals.entry(node.0.clone()).or_default();
            total.ms += took.as_secs_f64() * 1000.0;
            total.backend = placed.backend.to_owned();
            total.routed.clone_from(&placed.routed);
            total.uploaded += placed.uploaded;
            total.downloaded += placed.downloaded;
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
    let mut grouped: BTreeMap<String, (NodeTime, usize)> = BTreeMap::new();
    let per_frame = u64::from(frames);
    for (node, total) in totals {
        let (entry, count) = grouped.entry(pattern(&node)).or_insert_with(|| {
            (
                NodeTime {
                    node: String::new(),
                    ms: 0.0,
                    backend: total.backend.clone(),
                    routed: total.routed.clone(),
                    uploaded: 0,
                    downloaded: 0,
                },
                0,
            )
        });
        entry.ms += (total.ms / f64::from(frames) - floor).max(0.0);
        entry.uploaded += total.uploaded / per_frame;
        entry.downloaded += total.downloaded / per_frame;
        *count += 1;
    }
    let mut nodes: Vec<NodeTime> = grouped
        .into_iter()
        .map(|(node, (mut time, count))| {
            time.node = if count > 1 {
                format!("{node} ({count})")
            } else {
                node
            };
            time
        })
        .collect();
    nodes.sort_by(|a, b| b.ms.total_cmp(&a.ms));
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
    let mut session = options.session.open(backend)?;
    let events = session.sync();
    let rejected: Vec<String> = events
        .iter()
        .filter_map(|event| match event {
            crate::Event::Rejected { path, error } => Some(format!("{}:{error}", path.display())),
            crate::Event::SceneRejected(error) => Some(format!("scene rejected: {error}")),
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
        options
            .session
            .scene
            .as_deref()
            .unwrap_or(Path::new(""))
            .display(),
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
        for time in &entry.nodes {
            out.push_str(&format!(
                "  {:<28} {:8.3}  {}\n",
                time.node,
                time.ms,
                time.placement_text()
            ));
        }
        // What the parts come to against the whole. A node measured
        // alone waits for its own work, so the setting up of the next
        // one no longer hides behind it: the difference is what a frame
        // overlaps, and naming it is the difference between a profile
        // that can be read and one that quietly does not add up.
        let sum: f64 = entry.nodes.iter().map(|time| time.ms).sum();
        out.push_str(&format!(
            "  {:<28} {sum:8.3} against a frame of {:.3}; a node runs alone here, and what a frame overlaps it does not\n",
            "(sum)", entry.best
        ));
    }
    out
}

/// What one node, or the nodes one pattern names, took a frame, and
/// where it ran.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeTime {
    /// The node, or its pattern and how many it stands for.
    pub node: String,
    /// Milliseconds a frame.
    pub ms: f64,
    /// The backend that ran it.
    pub backend: String,
    /// Why the main backend did not, when it did not.
    pub routed: Option<String>,
    /// Bytes a frame its runs brought to a device from Host memory.
    pub uploaded: u64,
    /// Bytes a frame its runs brought back to Host memory.
    pub downloaded: u64,
}

impl NodeTime {
    /// Where it ran and what that carried, as a profile line ends: the
    /// backend, why it is not the main one, and the traffic.
    pub fn placement_text(&self) -> String {
        let mut out = self.backend.clone();
        if let Some(why) = &self.routed {
            out.push_str(&format!(" ({why})"));
        }
        if self.uploaded > 0 {
            out.push_str(&format!(", {} up", bytes_text(self.uploaded)));
        }
        if self.downloaded > 0 {
            out.push_str(&format!(", {} down", bytes_text(self.downloaded)));
        }
        out
    }
}

/// A count of bytes as a person reads it: `512 B`, `1.5 KiB`, `3.2 MiB`.
pub fn bytes_text(bytes: u64) -> String {
    let b = bytes as f64;
    if bytes < 1024 {
        format!("{bytes} B")
    } else if b < 1024.0 * 1024.0 {
        format!("{:.1} KiB", b / 1024.0)
    } else {
        format!("{:.1} MiB", b / (1024.0 * 1024.0))
    }
}

/// The measurement as JSON, for `--compare` on a later build.
pub fn json(options: &Options, measured: &[Measured]) -> String {
    let backends: Vec<String> = measured
        .iter()
        .map(|entry| {
            let nodes: Vec<String> = entry
                .nodes
                .iter()
                .map(|time| {
                    format!(
                        "      {{\"node\": {:?}, \"ms\": {:.4}, \"backend\": {:?}}}",
                        time.node, time.ms, time.backend
                    )
                })
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
        options
            .session
            .scene
            .as_deref()
            .unwrap_or(Path::new(""))
            .display()
            .to_string(),
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
