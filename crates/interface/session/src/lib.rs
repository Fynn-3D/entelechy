//! A live session over one `.osn` document and the libraries it uses.
//!
//! The document is the graph: the session parses it, publishes its inline
//! Kernels, follows its `use` items to `.orla` libraries, and builds the
//! Graph. On every [`Session::sync`] it re-reads what changed on disk and
//! republishes only the Kernels whose text changed, keeping a running
//! version whenever a replacement is rejected. Source resolution lives here,
//! not in the runtimes.
//!
//! A session is four parts, each owning its own state and reaching the
//! others only through what it is handed: the [scene](scene) reads the
//! documents and lowers the Graph; the [ledger](kernels) knows which
//! document published which Kernel and publishes through the runtime it
//! is given; the [inputs](inputs) own the Fields a scene's inputs hold in
//! the Host; and [source resolution](sources) holds nothing at all. The
//! session keeps the runtime and the evaluator, and says in what order
//! the parts sync.
//!
//! This crate is what every interface drives: the command line, the
//! Player, the C ABI, the importer and the exporter each open a
//! [`Session`], and none of them depends on another. The wire an agent
//! speaks to it is `orla-protocol`, and the file formats it reads and
//! writes are `orla-formats`; neither is the session's own.

#![warn(missing_docs)]

pub mod bench;
mod inputs;
mod kernels;
mod observe;
mod scene;
mod sources;
mod stamp;

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use orla_formats::ppm;
use orla_graph::{EvaluateError, Evaluator, Graph, Report};
use orla_runtime::{Backend, Context, Expect, Generation, Handle, KernelId, Runtime, Value};

pub use observe::parse_target;
pub use sources::{
    Declarations, PACKAGE_VERSION, declarations_for, declarations_within, package_manifest,
    read_document, used_documents,
};

/// What `trace` returns: the reference interpreter's record of one
/// element, which only the interpreter keeps, so an interface reads it
/// here rather than reaching for that backend itself.
pub use orla_cpu::{CALL_RECORDS, Trace, TraceValue};

/// Parses a window given on a command line: `<w>x<h>` or `<w>x<h>@<scale>`,
/// returning the size in pixels and the scale factor (one when absent).
pub fn parse_window(text: &str) -> Result<([f32; 2], f32), String> {
    let (size, scale) = match text.split_once('@') {
        Some((size, scale)) => (
            size,
            scale
                .parse::<f32>()
                .ok()
                .filter(|s| *s > 0.0)
                .ok_or_else(|| "--window's scale must be a positive number".to_string())?,
        ),
        None => (text, 1.0),
    };
    let (w, h) = size
        .split_once('x')
        .ok_or_else(|| "--window needs <w>x<h> or <w>x<h>@<scale>".to_string())?;
    let side = |text: &str| {
        text.parse::<u32>()
            .ok()
            .filter(|n| *n > 0)
            .map(|n| n as f32)
            .ok_or_else(|| "--window's sides must be positive integers".to_string())
    };
    Ok(([side(w)?, side(h)?], scale))
}

/// How many words a session's Host may hold at once unless it is told
/// otherwise: two gibibytes of them. A budget is a net and not a
/// policy — it stops a scene that has run away before the machine
/// starts swapping — so the default is generous enough for a scene
/// with millions of triangles in it and `--memory` states the policy
/// where a scene wants more.
pub const DEFAULT_BUDGET_WORDS: usize = 512 * 1024 * 1024;

/// The words in a number of mebibytes, which is how `--memory` asks.
pub fn budget_of_megabytes(megabytes: usize) -> usize {
    megabytes.saturating_mul(1024 * 1024 / 4)
}

/// What a session's Host may hold and spend: the words of Fields it
/// holds at once (`--memory`), and the loop iterations one run of a
/// Kernel may take, all its elements together (`--work`). Both are nets
/// and not policies, and a scene that wants more says so.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Words the Host may hold at once.
    pub words: usize,
    /// Loop iterations one run may take.
    pub work: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            words: DEFAULT_BUDGET_WORDS,
            work: orla_runtime::DEFAULT_WORK,
        }
    }
}

impl Limits {
    /// Whether `flag` sets a limit: `--memory` or `--work`.
    pub fn takes(flag: &str) -> bool {
        matches!(flag, "--memory" | "--work")
    }

    /// Sets the limit `flag` names from its value: `--memory <MB>` in
    /// whole mebibytes, `--work <M>` in millions of iterations.
    pub fn read(&mut self, flag: &str, text: &str) -> Result<(), String> {
        let whole = |unit: &str| -> Result<u64, String> {
            text.parse::<u64>().ok().filter(|n| *n > 0).ok_or_else(|| {
                format!("{flag} takes a whole number of {unit}, at least one, found `{text}`")
            })
        };
        match flag {
            "--memory" => self.words = budget_of_megabytes(whole("mebibytes")? as usize),
            "--work" => {
                let millions = whole("millions of iterations")?;
                let most = orla_runtime::MAX_WORK / 1_000_000;
                if millions > most {
                    return Err(format!(
                        "--work is at most {most} million iterations, found {millions}"
                    ));
                }
                self.work = millions * 1_000_000;
            }
            _ => return Err(format!("{flag} sets no limit")),
        }
        Ok(())
    }
}

/// Something the session observed while synchronizing with the filesystem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// A changed Kernel compiled and went to the preparation worker; the
    /// running version stays until a later sync publishes the replacement.
    Preparing {
        /// Document that defines it.
        path: PathBuf,
        /// Kernel id.
        id: KernelId,
    },
    /// A Kernel was compiled and published.
    Published {
        /// Document that defines it.
        path: PathBuf,
        /// Kernel id.
        id: KernelId,
        /// New generation.
        generation: Generation,
    },
    /// A Kernel changed but was not accepted; the previous version stays.
    Rejected {
        /// Document that defines it.
        path: PathBuf,
        /// Diagnostic.
        error: String,
    },
    /// A Kernel disappeared from its document.
    Retired {
        /// Document that defined it.
        path: PathBuf,
        /// Kernel id that left the registry.
        id: KernelId,
    },
    /// The scene document was re-read and its graph rebuilt.
    SceneLoaded,
    /// An input's text was written into the Host as rows of its
    /// characters' codes, at first or after it changed.
    Text {
        /// The input.
        name: String,
        /// How many characters.
        chars: usize,
    },
    /// An input's file was read into the Host, at first or after it changed.
    Loaded {
        /// The input.
        name: String,
        /// The file.
        path: PathBuf,
        /// What came in: shape, extent, layout, and what the file adds
        /// (a sound's rate).
        detail: String,
    },
    /// The scene document changed but does not parse; the previous graph stays.
    SceneRejected(String),
    /// A node will run off the session's main backend, and why: what it
    /// costs a frame to take its inputs there and its outputs back is
    /// the scene's to know before it measures anything. Said once, when
    /// the node's placement changes.
    Routed {
        /// The node.
        node: String,
        /// The backend that runs it.
        backend: &'static str,
        /// Why the main backend does not.
        why: String,
    },
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preparing { path, id } => write!(f, "preparing {id} from {}", path.display()),
            Self::Published {
                path,
                id,
                generation,
            } => write!(f, "published {id} {generation} from {}", path.display()),
            Self::Rejected { path, error } => write!(f, "rejected {}:{error}", path.display()),
            Self::Retired { path, id } => write!(f, "retired {id} ({})", path.display()),
            Self::SceneLoaded => f.write_str("scene loaded"),
            Self::Loaded { name, path, detail } => {
                write!(f, "loaded {name} from {}: {detail}", path.display())
            }
            Self::Text { name, chars } => write!(f, "text {name}: {chars} characters"),
            Self::SceneRejected(error) => write!(f, "scene rejected: {error}"),
            Self::Routed { node, backend, why } => {
                write!(f, "node `{node}` runs on {backend} ({why})")
            }
        }
    }
}

/// Session-level failures.
#[derive(Debug)]
pub enum SessionError {
    /// A file could not be read or written.
    Io(PathBuf, std::io::Error),
    /// An external image could not be imported or the output exported.
    Image(PathBuf, ppm::PpmError),
    /// No scene has been loaded successfully.
    NoScene,
    /// The Host rejected an import.
    Host(orla_runtime::HostError),
    /// A resident image could not be brought back for observation.
    Observe(String),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(path, error) => write!(f, "{}: {error}", path.display()),
            Self::Image(path, error) => write!(f, "{}: {error}", path.display()),
            Self::NoScene => f.write_str("no scene has been loaded"),
            Self::Host(error) => write!(f, "{error}"),
            Self::Observe(message) => write!(f, "cannot observe image: {message}"),
        }
    }
}

impl std::error::Error for SessionError {}

/// A scene, the libraries it uses, and the runtime state built from them.
pub struct Session {
    scene: scene::Scene,
    kernels: kernels::Kernels,
    inputs: inputs::Inputs,
    runtime: Runtime,
    evaluator: Evaluator,
    /// Where each node was last said to run, so a change is said once.
    placed: BTreeMap<String, Option<String>>,
    /// The window a headless session reports when an evaluation names
    /// none: its size in pixels and its scale factor.
    window: Option<([f32; 2], f32)>,
}

/// The error an evaluation returns when no scene has loaded.
fn no_scene() -> EvaluateError {
    EvaluateError::Graph(orla_graph::GraphError::UnknownOutput(
        orla_graph::Source::External("<no scene>".into()),
    ))
}

impl Session {
    /// Creates a CPU-backed session; call [`Session::sync`] to load files.
    /// `library` is an extra directory or file of Kernels to use.
    pub fn open(scene: &Path, library: Option<&Path>, budget_words: usize) -> Self {
        Self::open_with(
            scene,
            library,
            Runtime::with_backend(budget_words, Box::new(orla_cpu::CpuBackend)),
        )
    }

    /// Creates a session over a runtime built by the caller, for choosing
    /// the backend.
    pub fn open_with(scene: &Path, library: Option<&Path>, runtime: Runtime) -> Self {
        Self {
            scene: scene::Scene::new(scene, library),
            kernels: kernels::Kernels::default(),
            inputs: inputs::Inputs::default(),
            runtime,
            evaluator: Evaluator::new(),
            placed: BTreeMap::new(),
            window: None,
        }
    }

    /// The runtime, for inspection.
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Builds a runtime over a named backend: `cpu`, or `gpu` when a device
    /// is available.
    pub fn runtime_for(backend: &str, limits: Limits) -> Result<Runtime, String> {
        let mut runtime = Self::backend_runtime(backend, limits.words)?;
        runtime.host_mut().set_work(limits.work);
        Ok(runtime)
    }

    fn backend_runtime(backend: &str, budget_words: usize) -> Result<Runtime, String> {
        match backend {
            "cpu" => Ok(Runtime::with_backend(
                budget_words,
                Box::new(orla_cpu::CpuBackend),
            )),
            "gpu" => {
                // A GPU walks a scan on one thread; the JIT takes the
                // scans, the GPU everything else.
                let gpu = orla_gpu::GpuBackend::new().map_err(|error| error.to_string())?;
                let backend: Box<dyn Backend> = Box::new(gpu);
                let ordered: Option<Box<dyn Backend>> = orla_jit::JitBackend::new()
                    .ok()
                    .map(|jit| Box::new(jit) as Box<dyn Backend>);
                Ok(Runtime::with_backends(budget_words, backend, ordered))
            }
            "jit" => {
                let jit = orla_jit::JitBackend::new()?;
                let backend: Box<dyn Backend> = Box::new(jit);
                Ok(Runtime::with_backend(budget_words, backend))
            }
            other => Err(format!("unknown backend `{other}`; use cpu, gpu, or jit")),
        }
    }

    /// Mutable runtime access for interfaces that need to observe images.
    pub fn runtime_mut(&mut self) -> &mut Runtime {
        &mut self.runtime
    }

    /// The current graph, if the scene loaded.
    pub fn graph(&self) -> Option<&Graph> {
        self.scene.graph()
    }

    /// The scene document.
    pub fn root(&self) -> &Path {
        self.scene.root()
    }

    /// Overrides a graph parameter for later evaluations, in the shape of
    /// its declared type: `[[1, 0], [0, 1]]` for a `Mat2`, an integer for a
    /// float. A value of another layout is refused. Unknown names are kept
    /// as given and take effect if the graph later declares them.
    pub fn set_param(&mut self, name: &str, value: Value) -> Result<(), String> {
        self.scene.set_param(name, value)
    }

    /// Current graph parameter overrides.
    pub fn params(&self) -> &BTreeMap<String, Value> {
        self.scene.params()
    }

    /// Every document the session currently watches.
    pub fn watched(&self) -> Vec<PathBuf> {
        self.kernels
            .documents()
            .into_iter()
            .chain(self.inputs.watched())
            .collect()
    }

    /// Imports a PPM file as the external image `name`.
    pub fn bind_external(&mut self, name: &str, path: &Path) -> Result<(), SessionError> {
        let image = ppm::read(path).map_err(|e| SessionError::Image(path.to_path_buf(), e))?;
        let host = self.runtime.host_mut();
        let handle = host.adopt_image(image).map_err(SessionError::Host)?;
        self.inputs.hold(host, name, handle);
        Ok(())
    }

    /// Reconciles the runtime with the files on disk and reports what
    /// changed: replacements the worker finished publish first, then the
    /// documents are read, then the inputs, and last where each node runs.
    pub fn sync(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        self.kernels.collect_pending(&mut self.runtime, &mut events);
        self.scene
            .sync(&mut self.kernels, &mut self.runtime, &mut events);
        self.inputs
            .sync(self.scene.graph(), self.runtime.host_mut(), &mut events);
        self.sync_placements(&mut events);
        events.append(&mut self.inputs.take_frame_events());
        events
    }

    /// Where every node of the scene runs: the backend, and why it is not
    /// the main one when it is not. A node whose Kernel is not published
    /// is left out.
    pub fn placements(&self) -> Vec<(String, orla_runtime::Placement)> {
        let Some(graph) = self.scene.graph() else {
            return Vec::new();
        };
        graph
            .nodes
            .iter()
            .filter_map(|node| {
                let version = self.runtime.kernel(&node.kernel)?;
                Some((node.id.to_string(), self.runtime.placement(&version)))
            })
            .collect()
    }

    /// Says which nodes run off the main backend, once each time that
    /// changes.
    fn sync_placements(&mut self, events: &mut Vec<Event>) {
        let now = self.placements();
        for (node, placement) in &now {
            let before = self.placed.get(node);
            if before == Some(&placement.routed) {
                continue;
            }
            if let Some(why) = &placement.routed {
                events.push(Event::Routed {
                    node: node.clone(),
                    backend: placement.backend,
                    why: why.clone(),
                });
            }
        }
        self.placed = now
            .into_iter()
            .map(|(node, placement)| (node, placement.routed))
            .collect();
    }

    /// Prepares replacements of running Kernels and fused groups on a
    /// worker thread, so a republish never stalls the caller; a
    /// replacement publishes on the first sync after it is ready.
    pub fn set_background(&mut self, enabled: bool) {
        self.runtime.set_background(enabled);
    }

    /// Whether replacements prepare on a worker thread.
    pub fn background(&self) -> bool {
        self.runtime.background()
    }

    /// Publishes Kernel source that did not come from a document, for the
    /// creation protocol. The source holds exactly one Kernel. Ownership
    /// stays with documents: a document that later defines the id
    /// republishes it from the file.
    pub fn publish_source(
        &mut self,
        source: &str,
        expect: Expect,
    ) -> Result<std::sync::Arc<orla_runtime::KernelVersion>, String> {
        let candidate = self.prepare_source(source)?;
        self.runtime
            .publish(candidate, expect)
            .map_err(|e| e.to_string())
    }

    /// Prepares `source`, a document holding one Kernel, without
    /// publishing it. The Kernel sees the declarations the source makes
    /// or `use`s, and the scene's shared ones after those, so a Kernel
    /// from a file republishes through the protocol as it is.
    pub fn prepare_source(&self, source: &str) -> Result<orla_runtime::Candidate, String> {
        self.scene
            .prepare_source(source, &self.kernels, &self.runtime)
    }

    /// Retires a published Kernel; returns whether one was registered.
    pub fn retire_kernel(&mut self, id: &KernelId) -> bool {
        self.kernels.retire_kernel(id, &mut self.runtime)
    }

    /// The Field an input of the scene holds now: a file's, a text's, a
    /// frame's, or what the caller supplied.
    pub fn external(&self, name: &str) -> Option<Handle> {
        self.inputs.handles().get(name).copied()
    }

    /// The cached image of a node's output port, if it was evaluated.
    pub fn cached(&self, node: &orla_graph::NodeId, port: &str) -> Option<Handle> {
        self.evaluator.cached(node, port)
    }

    /// Waits for every `async` node's run off the frame and takes it in;
    /// the next evaluation reads what landed. See
    /// [`orla_graph::Evaluator::settle`].
    pub fn settle(&mut self) -> Result<Vec<orla_graph::NodeId>, EvaluateError> {
        self.evaluator.settle(&mut self.runtime)
    }

    /// Whether any `async` node's run is still going.
    pub fn has_running(&self) -> bool {
        self.evaluator.has_running()
    }

    /// Evaluates the current graph under `context`. Unchanged nodes are
    /// reused; state commits only when the whole evaluation succeeds.
    pub fn evaluate(&mut self, context: &Context) -> Result<Report, EvaluateError> {
        self.inputs
            .select_frames(self.runtime.host_mut(), context.time);
        let graph = self.scene.graph().ok_or_else(no_scene)?;
        self.evaluator.evaluate(
            graph,
            &mut self.runtime,
            self.inputs.handles(),
            context,
            self.scene.params(),
        )
    }

    /// Evaluates only what `demand` needs; see
    /// [`orla_graph::Evaluator::evaluate_for`].
    pub fn evaluate_for(
        &mut self,
        context: &Context,
        demand: &std::collections::BTreeSet<orla_graph::NodeId>,
    ) -> Result<Report, EvaluateError> {
        self.inputs
            .select_frames(self.runtime.host_mut(), context.time);
        let graph = self.scene.graph().ok_or_else(no_scene)?;
        self.evaluator.evaluate_for(
            graph,
            &mut self.runtime,
            self.inputs.handles(),
            context,
            self.scene.params(),
            demand,
        )
    }

    /// Replaces the committed state of `node.port` with `words`; see
    /// [`orla_graph::Evaluator::write`].
    pub fn write(
        &mut self,
        node: &orla_graph::NodeId,
        port: &str,
        words: Vec<u32>,
    ) -> Result<(), EvaluateError> {
        let graph = self.scene.graph().ok_or_else(no_scene)?;
        self.evaluator
            .write(graph, &mut self.runtime, node, port, words)
    }

    /// Forgets committed state; `<~` bindings restart from their initial
    /// sources on the next evaluation.
    pub fn reset_state(&mut self) {
        self.evaluator.reset_state(&mut self.runtime);
    }

    /// Sets the window a headless session reports when an evaluation names
    /// none: the size in pixels and the scale factor; `None` for no window.
    pub fn set_window(&mut self, window: Option<([f32; 2], f32)>) {
        self.window = window;
    }

    /// The window a headless session reports, if one was set.
    pub fn window(&self) -> Option<([f32; 2], f32)> {
        self.window
    }

    /// Makes the backend wait for its device after every node, so a
    /// profile measures the work rather than the submission; on the GPU
    /// that is the difference between a dispatch and a picture. The CPU
    /// and the JIT are measured as they are.
    pub fn set_measured(&mut self, on: bool) {
        if let Some(gpu) = self
            .runtime
            .backend()
            .as_any()
            .downcast_ref::<orla_gpu::GpuBackend>()
        {
            gpu.set_measured(on);
        }
    }

    /// What measuring one node costs before the node does anything, so
    /// a profile can take it off what it prints; see
    /// [`orla_runtime::Backend::measurement_floor`]. Ask after a
    /// measured run: a backend only knows once it has tried.
    pub fn measurement_floor(&self) -> std::time::Duration {
        self.runtime().measurement_floor()
    }

    /// Whether a profile's numbers are what the device's own clock said
    /// about the work, rather than what the call and its wait took.
    pub fn measures_the_device(&self) -> bool {
        self.runtime()
            .backend()
            .as_any()
            .downcast_ref::<orla_gpu::GpuBackend>()
            .is_some_and(|gpu| gpu.device_clock_answered())
    }

    /// Enables or disables fusion of pointwise chains.
    pub fn set_fusion(&mut self, enabled: bool) {
        self.evaluator.set_fusion(enabled);
    }

    /// Whether fusion is enabled.
    pub fn fusion(&self) -> bool {
        self.evaluator.fusion()
    }

    /// Keeps a node materialized so its image can be read.
    pub fn pin(&mut self, node: orla_graph::NodeId) {
        self.evaluator.pin(node);
    }

    /// What would execute right now, in order, with fusion applied.
    pub fn plan(&mut self) -> Result<Vec<orla_graph::PlanEntry>, EvaluateError> {
        let graph = self.scene.graph().ok_or_else(no_scene)?;
        self.evaluator.plan_summary(
            graph,
            &mut self.runtime,
            self.inputs.handles(),
            self.scene.params(),
        )
    }

    /// Lets a node be fused away again.
    pub fn unpin(&mut self, node: &orla_graph::NodeId) {
        self.evaluator.unpin(node);
    }
}
