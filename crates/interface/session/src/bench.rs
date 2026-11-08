//! Repeatable scene measurement: steps a scene for a frame count,
//! several runs, on each named backend, and reports the best run.
//!
//! The clock is frame count over rate, as `render`'s is, so each run
//! does the same work. A warm-up frame (compiling, allocating) precedes
//! the runs. `--json` output reads back with `--compare`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use entelechy_kernel::Context;

use crate::Session;

/// How to measure.
#[derive(Clone, Debug)]
pub struct Options {
    /// The session each backend opens; `backends` overrides its backend.
    pub session: crate::SessionOptions,
    /// Backends to measure, in order.
    pub backends: Vec<String>,
    /// Frames in a run, after the warm-up frame.
    pub frames: u32,
    /// Runs of that many frames.
    pub repeat: u32,
    /// Frames per second the clock advances by.
    pub rate: f32,
    /// Window size and scale for `window()`.
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
    /// Milliseconds a frame in the median run.
    pub median: f64,
    /// Milliseconds the first frame took, which the runs leave out: the
    /// work done once, compiling, building, filling what is kept.
    pub first: f64,
    /// Milliseconds a frame of the best run's the frame's thread waited
    /// for the device; the rest of the frame is its own work.
    pub wait: f64,
    /// Each node that kept the frame's thread waiting for the device in
    /// the best run, milliseconds a frame, longest first.
    pub waits: Vec<(String, f64)>,
    /// Milliseconds a frame of the best run waited for the device that no
    /// node did: at the frame's end, confirming the frame before, writing.
    pub wait_after: f64,
    /// With nodes, the device's busy time a frame by its own clock, where
    /// it keeps one.
    pub gpu: Option<f64>,
    /// Each node's milliseconds a frame, when asked for, averaged over
    /// all frames.
    pub nodes: Vec<NodeTime>,
    /// Whether any node ran after the warm-up frame.
    pub ran: bool,
    /// Whether node times come from the device's clock; otherwise they
    /// are the call and its wait, less `floor`.
    pub device_clock: bool,
    /// Milliseconds one empty node measurement costs, already subtracted
    /// from every node time.
    pub floor: f64,
    /// Groups the backend refused to run fused, which ran apart.
    pub refused: Vec<String>,
}

/// Measures every backend the options name.
pub fn bench(options: &Options) -> Result<Vec<Measured>, String> {
    let mut measured = Vec::new();
    for backend in &options.backends {
        measured.push(one(options, backend)?);
    }
    Ok(measured)
}

/// Measures one backend in its own session.
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

/// Measures the session: a warm-up frame, `repeat` runs of `frames`
/// frames, and, with `nodes`, one run timing every node. Time advances
/// from `base` by frame count over `rate`, as `advance` does; the
/// session is left at the last frame.
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
        time: base.time + f64::from(frame) / f64::from(rate),
        ..base
    };
    // The first frame compiles and allocates, left out of the runs and
    // said apart.
    let warm_started = Instant::now();
    let warm = session
        .evaluate(&context(0))
        .map_err(|e| format!("{backend}: {e}"))?;
    session.runtime().settle_device();
    let first = warm_started.elapsed().as_secs_f64() * 1000.0;
    let mut refused: Vec<String> = warm.refused.iter().map(ToString::to_string).collect();
    // Each run: its milliseconds a frame, of those its waits, and its
    // waits node by node.
    let mut runs: Vec<(f64, f64, BTreeMap<String, f64>)> = Vec::new();
    let mut frame = 1;
    let mut ran = false;
    // The frames overlap the device's as a window's do: the next is made
    // while the device runs this one.
    let overlapped = session.overlaps();
    session
        .set_overlap(true)
        .map_err(|e| format!("{backend}: {e}"))?;
    let ms = |duration: std::time::Duration| duration.as_secs_f64() * 1000.0;
    for _ in 0..options.repeat.max(1) {
        let started = Instant::now();
        let waited_before = session.device_waited();
        let mut waits: BTreeMap<String, f64> = BTreeMap::new();
        for _ in 0..options.frames.max(1) {
            let report = session
                .evaluate(&context(frame))
                .map_err(|e| format!("{backend}: {e}"))?;
            ran |= !report.executed.is_empty();
            refused.extend(report.refused.iter().map(ToString::to_string));
            for (node, spent) in &report.waits {
                *waits.entry(pattern(&node.0)).or_default() += ms(*spent);
            }
            frame += 1;
        }
        // The last frame's work is the run's too.
        session.confirm().map_err(|e| format!("{backend}: {e}"))?;
        session.runtime().settle_device();
        let took = started.elapsed().as_secs_f64() * 1000.0;
        let waited = ms(session.device_waited().saturating_sub(waited_before));
        let frames = f64::from(options.frames.max(1));
        runs.push((took / frames, waited / frames, waits));
    }
    session
        .set_overlap(overlapped)
        .map_err(|e| format!("{backend}: {e}"))?;
    runs.sort_by(|a, b| a.0.total_cmp(&b.0));
    // What a frame waited for, node by node and at its end: the best
    // run's, as its own wait is.
    let frames = f64::from(options.frames.max(1));
    let median = runs[runs.len() / 2].0;
    let (best, wait, best_waits) = runs.swap_remove(0);
    let mut waits: Vec<(String, f64)> = best_waits
        .into_iter()
        .map(|(node, sum)| (node, sum / frames))
        .filter(|(_, ms)| *ms >= 0.005)
        .collect();
    waits.sort_by(|a, b| b.1.total_cmp(&a.1));
    let by_nodes: f64 = waits.iter().map(|(_, ms)| ms).sum();
    let wait_after = (wait - by_nodes).max(0.0);
    let run = if options.nodes {
        node_times(session, options, &context, &mut frame, backend)?
    } else {
        NodeRun {
            nodes: Vec::new(),
            device_clock: false,
            floor: 0.0,
            gpu: None,
        }
    };
    Ok(Measured {
        backend: backend.to_owned(),
        best,
        median,
        first,
        wait,
        waits,
        wait_after,
        gpu: run.gpu,
        nodes: run.nodes,
        ran,
        device_clock: run.device_clock,
        floor: run.floor,
        refused,
    })
}

/// A node-timing run's result; fields as in [`Measured`].
struct NodeRun {
    nodes: Vec<NodeTime>,
    device_clock: bool,
    floor: f64,
    gpu: Option<f64>,
}

/// A separate run timing every node. Where the device keeps a clock the
/// frames run as a window's do, overlapped and clocked: each node's
/// device time is what the clock says its runs took, and its call is its
/// time on the frame's thread, nothing waiting for either. Elsewhere it
/// waits for the device after every node ([`measured_node_times`]).
fn node_times(
    session: &mut Session,
    options: &Options,
    context: &impl Fn(u32) -> Context,
    frame: &mut u32,
    backend: &str,
) -> Result<NodeRun, String> {
    if !session.runtime().has_device() {
        return measured_node_times(session, options, context, frame, backend);
    }
    let overlapped = session.overlaps();
    session
        .set_overlap(true)
        .map_err(|e| format!("{backend}: {e}"))?;
    session.set_clocked(true);
    let mut calls = Profile::default();
    let mut device: BTreeMap<String, f64> = BTreeMap::new();
    let mut busy = 0.0;
    let mut told: BTreeSet<u64> = BTreeSet::new();
    let mut hear = |session: &Session| {
        if let Some(times) = session.device_times()
            && told.insert(times.frame)
        {
            busy += times.busy.as_secs_f64() * 1000.0;
            for (node, spent) in &times.nodes {
                *device.entry(pattern(&node.0)).or_default() += spent.as_secs_f64() * 1000.0;
            }
        }
    };
    let frames = options.frames.max(1);
    let evaluated = (|| {
        for _ in 0..frames {
            let report = session
                .evaluate(&context(*frame))
                .map_err(|e| format!("{backend}: {e}"))?;
            *frame += 1;
            calls.add(&report);
            hear(session);
        }
        session.confirm().map_err(|e| format!("{backend}: {e}"))?;
        session.runtime().settle_device();
        hear(session);
        Ok::<(), String>(())
    })();
    session.set_clocked(false);
    session
        .set_overlap(overlapped)
        .map_err(|e| format!("{backend}: {e}"))?;
    evaluated?;
    // A device that tells no frame done keeps no clock to read.
    if told.is_empty() {
        return measured_node_times(session, options, context, frame, backend);
    }
    let heard = told.len() as f64;
    let mut nodes = calls.nodes(0.0);
    for time in &mut nodes {
        time.call = time.ms;
        let name = time.node.split(" (").next().unwrap_or(&time.node);
        if let Some(spent) = device.get(name) {
            time.ms = spent / heard;
        }
    }
    nodes.sort_by(|a, b| b.ms.total_cmp(&a.ms));
    Ok(NodeRun {
        nodes,
        device_clock: true,
        floor: 0.0,
        gpu: Some(busy / heard),
    })
}

/// A run that waits for the device after every node, so a node's time
/// is its call and the wait after it: where the device keeps no clock.
fn measured_node_times(
    session: &mut Session,
    options: &Options,
    context: &impl Fn(u32) -> Context,
    frame: &mut u32,
    backend: &str,
) -> Result<NodeRun, String> {
    session.set_measured(true);
    let mut profile = Profile::default();
    let frames = options.frames.max(1);
    for _ in 0..frames {
        let report = session
            .evaluate(&context(*frame))
            .map_err(|e| format!("{backend}: {e}"))?;
        *frame += 1;
        profile.add(&report);
    }
    session.set_measured(false);
    // The per-node measurement floor comes off every node; the backend
    // knows it only after the run.
    let floor = session.measurement_floor().as_secs_f64() * 1000.0;
    let nodes = profile.nodes(floor);
    Ok(NodeRun {
        nodes,
        device_clock: session.measures_the_device(),
        floor,
        gpu: None,
    })
}

/// Each node's time, placement and traffic over frames timed node by node
/// (a session [`Session::set_measured`]): what `bench --nodes` reports,
/// and `render` and `replay` under `--profile`.
#[derive(Default)]
pub struct Profile {
    frames: u32,
    totals: BTreeMap<String, Total>,
}

/// A node's time, placement and traffic over all frames.
#[derive(Default)]
struct Total {
    ms: f64,
    backend: String,
    routed: Option<entelechy_kernel::Routed>,
    uploaded: u64,
    downloaded: u64,
}

impl Profile {
    /// Adds a frame's report: each node it ran.
    pub fn add(&mut self, report: &entelechy_graph::Report) {
        self.frames += 1;
        for ((node, took), placed) in report.timings.iter().zip(&report.placements) {
            let total = self.totals.entry(node.0.to_string()).or_default();
            total.ms += took.as_secs_f64() * 1000.0;
            total.backend = placed.backend.to_owned();
            total.routed.clone_from(&placed.routed);
            total.uploaded += placed.uploaded;
            total.downloaded += placed.downloaded;
        }
    }

    /// How many frames were added.
    pub fn frames(&self) -> u32 {
        self.frames
    }

    /// Each node's milliseconds a frame (over every frame added, a frame
    /// that reused it counting none), less `floor` (what measuring one
    /// node costs before it does anything); repeated placements grouped
    /// into one line, `steps/*.spread/*` and their count; the busiest
    /// first.
    pub fn nodes(&self, floor: f64) -> Vec<NodeTime> {
        let frames = self.frames.max(1);
        let mut grouped: BTreeMap<String, (NodeTime, usize)> = BTreeMap::new();
        let per_frame = u64::from(frames);
        for (node, total) in &self.totals {
            let (entry, count) = grouped.entry(pattern(node)).or_insert_with(|| {
                (
                    NodeTime {
                        node: String::new(),
                        ms: 0.0,
                        call: 0.0,
                        backend: total.backend.clone(),
                        routed: total.routed.as_deref().map(str::to_owned),
                        uploaded: 0,
                        downloaded: 0,
                    },
                    0,
                )
            });
            entry.ms += (total.ms / f64::from(frames) - floor).max(0.0);
            entry.call = entry.ms;
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
        nodes
    }

    /// The profile as lines: a heading saying how the nodes were timed
    /// (`device_clock`, `floor`), each node's line, their sum.
    pub fn lines(&self, floor: f64, device_clock: bool) -> Vec<String> {
        let nodes = self.nodes(floor);
        let how = if device_clock {
            "what the device's own clock said each one took".to_owned()
        } else if floor > 0.0 {
            format!(
                "the call and the wait after it, less the {floor:.3} ms measuring one node costs before it does anything"
            )
        } else {
            "the call, which is the whole of the work on this backend".to_owned()
        };
        let mut lines = vec![format!(
            "profile of {} frame(s) after the first, ms a frame a node — {how}:",
            self.frames
        )];
        for time in &nodes {
            lines.push(format!(
                "  {:<28} {:8.3}  {}",
                time.node,
                time.ms,
                time.placement_text()
            ));
        }
        let sum: f64 = nodes.iter().map(|time| time.ms).sum();
        lines.push(format!(
            "  {:<28} {sum:8.3}  (a node runs alone under a profile, so the sum is more than a frame)",
            "(sum)"
        ));
        lines
    }
}

/// A node name with repetition indices as `*`: `steps/3.spread/6` is
/// `steps/*.spread/*`.
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

/// The measurement as a table: a line per backend, compared against the
/// first, then node costs when measured.
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
    for entry in measured {
        out.push_str(&format!(
            "{}: the first frame took {:.1} ms, left out above: what is done once, compiling and filling what is kept\n",
            entry.backend, entry.first
        ));
    }
    for entry in measured {
        if entry.wait <= 0.0 && entry.waits.is_empty() {
            continue;
        }
        out.push_str(&format!(
            "{}: a frame of {:.2} ms is {:.2} ms of the frame's own work and {:.2} ms waiting for the device\n",
            entry.backend,
            entry.best,
            (entry.best - entry.wait).max(0.0),
            entry.wait
        ));
        let mut said: Vec<String> = entry
            .waits
            .iter()
            .map(|(node, ms)| format!("{node} {ms:.2}"))
            .collect();
        said.push(format!("the frame's end {:.2}", entry.wait_after));
        out.push_str(&format!(
            "{}: waited for the device, ms a frame: {}\n",
            entry.backend,
            said.join(" | ")
        ));
    }
    for entry in measured {
        for refused in &entry.refused {
            out.push_str(&format!("\n{}: {refused}\n", entry.backend));
        }
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
        let clocked = entry.device_clock && entry.gpu.is_some();
        let how = if clocked {
            "on the device by its own clock, beside the call on the frame's thread, the frames run as a window's with nothing waiting for either".to_owned()
        } else if entry.device_clock {
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
            let call = if clocked && (time.call - time.ms).abs() > 0.0005 {
                format!("  call {:6.3}", time.call)
            } else {
                String::new()
            };
            out.push_str(&format!(
                "  {:<28} {:8.3}{call}  {}\n",
                time.node,
                time.ms,
                time.placement_text()
            ));
        }
        let sum: f64 = entry.nodes.iter().map(|time| time.ms).sum();
        // What ran on the device: the nodes routed off it ran elsewhere.
        let on_device: f64 = entry
            .nodes
            .iter()
            .filter(|time| time.routed.is_none())
            .map(|time| time.ms)
            .sum();
        match entry.gpu {
            // The device's busy time a frame against the nodes': the rest
            // ran between nodes (a frame's sound, its presenting copy).
            Some(gpu) if clocked => {
                let calls: f64 = entry.nodes.iter().map(|time| time.call).sum();
                out.push_str(&format!(
                    "  {:<28} {on_device:8.3}  on the device, of its {gpu:.3} a frame, the rest between nodes; the calls {calls:.3} on the frame's thread\n",
                    "(device)"
                ));
            }
            // The node sum against the frame; the gap is what a frame
            // overlaps and isolated node timing does not.
            _ => out.push_str(&format!(
                "  {:<28} {sum:8.3} against a frame of {:.3}; a node runs alone here, and what a frame overlaps it does not\n",
                "(sum)", entry.best
            )),
        }
    }
    out
}

/// A node's (or node pattern's) time a frame and placement.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeTime {
    /// The node, or its pattern and count.
    pub node: String,
    /// Milliseconds a frame: on its device by the device's clock, or on
    /// the processor.
    pub ms: f64,
    /// Milliseconds a frame on the frame's thread, giving its runs to the
    /// device or running them; the same as `ms` for a processor's node.
    pub call: f64,
    /// The backend that ran it.
    pub backend: String,
    /// Why it was routed off the main backend, if it was.
    pub routed: Option<String>,
    /// Bytes a frame uploaded from Host memory to a device.
    pub uploaded: u64,
    /// Bytes a frame downloaded back to Host memory.
    pub downloaded: u64,
}

impl NodeTime {
    /// A profile line's tail: backend, routing reason, traffic.
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

/// Bytes for reading: `512 B`, `1.5 KiB`, `3.2 MiB`.
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
                "    {{\"backend\": {:?}, \"best_ms\": {:.4}, \"median_ms\": {:.4}, \"first_ms\": {:.4}, \"wait_ms\": {:.4}, \"nodes\": [\n{}\n    ]}}",
                entry.backend,
                entry.best,
                entry.median,
                entry.first,
                entry.wait,
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

/// Reads back the numbers a comparison needs from `--json` output; a
/// malformed file is an error.
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
            first: number("first_ms").unwrap_or(0.0),
            wait: number("wait_ms").unwrap_or(0.0),
            waits: Vec::new(),
            wait_after: 0.0,
            gpu: None,
            nodes: Vec::new(),
            ran: true,
            device_clock: false,
            floor: 0.0,
            refused: Vec::new(),
        });
    }
    if measured.is_empty() {
        return Err(format!("{}: no measurements", path.display()));
    }
    Ok(measured)
}
