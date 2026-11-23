//! The session: its runtime, evaluator and backing scene (either live
//! source documents or a compiled [`crate::bundle::Bundle`]).

use std::collections::{BTreeMap, BTreeSet};
#[cfg(feature = "documents")]
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "documents")]
use entelechy_documents::{Event, inputs, kernels, scene, writes};
use entelechy_graph::{EvaluateError, Evaluator, Graph, Report};
use entelechy_kernel::{Backend, Context, Handle, KernelId, Runtime, Value};

use crate::{Limits, SessionError};

mod bundled;
#[cfg(feature = "documents")]
mod docs;
#[cfg(feature = "documents")]
mod inspect;
#[cfg(feature = "documents")]
mod time;

#[cfg(any(feature = "documents", target_os = "ios"))]
pub(super) const BEST: &[&str] = &["gpu", "jit", "cpu"];
#[cfg(not(any(feature = "documents", target_os = "ios")))]
pub(super) const BEST: &[&str] = &["gpu", "jit"];

/// A scene, the Kernels it uses, and the runtime state built from them.
pub struct Session {
    pub(super) backing: Backing,
    /// The inputs whose Host copies were let go once on a device
    /// ([`Session::let_go_of_input_copies`]).
    pub(crate) copies_let_go: BTreeSet<Handle>,
    pub(super) runtime: Runtime,
    pub(super) evaluator: Evaluator,
    /// The Kernels the graph's nodes use, by the version last seen
    /// ([`Session::finish_used`]).
    used: BTreeMap<KernelId, entelechy_kernel::Generation>,
    /// The headless window: size in pixels and scale factor.
    window: Option<([f32; 2], f32)>,
    /// The names asked to stay materialized ([`Session::pin`]), resolved
    /// against the running graph before each evaluation.
    pub(super) pins: Vec<entelechy_graph::NodeId>,
    /// Each clocked frame's nodes' runs, until the device tells the frame.
    node_clock: crate::clocked::NodeClock,
    /// The clipboard: the session's own, or the system's given.
    pub(super) board: crate::clipboard::Board,
    /// The Kernels an input `= progress` reads: the session's own, or
    /// another's given.
    watched: crate::preparing::Watched,
    /// The context the last frame was evaluated under, which outputs let
    /// go cold are made again under when read ([`Session::warm`]).
    pub(super) last_context: Context,
    /// Why the last evaluation failed, until one succeeds.
    pub(super) failing: Option<String>,
    /// Whether what only shows is shown from the processor while the
    /// device's compiler builds it ([`Session::show_while_building`]).
    shown_while_building: bool,
    /// The Kernels handed to threads of their own to be built there
    /// ([`Session::finish_used`]).
    handed: BTreeSet<KernelId>,
}

pub(super) enum Backing {
    #[cfg(feature = "documents")]
    Documents(Box<Docs>),
    Bundled(Box<bundled::BundledScene>),
}

#[cfg(feature = "documents")]
pub(super) struct Docs {
    pub(super) scene: scene::Scene,
    pub(super) kernels: kernels::Kernels,
    pub(super) inputs: inputs::Inputs,
    pub(super) reads: Arc<std::sync::Mutex<entelechy_documents::reads::Reads>>,
    pub(super) writes: writes::Writes,
    pub(super) nothing_saved: bool,
    pub(super) placed: BTreeMap<String, Option<entelechy_kernel::Routed>>,
    pub(super) idle: BTreeMap<PathBuf, Vec<KernelId>>,
    pub(super) unrun: BTreeMap<String, String>,
    pub(super) shaped: Option<(Arc<Graph>, u64)>,
    pub(super) history: entelechy_documents::history::History,
    pub(super) origin: crate::Origin,
    pub(super) claim: Claim,
    pub(super) later: Vec<Event>,
    pub(super) resized: bool,
    pub(super) bound: BTreeMap<String, PathBuf>,
    pub(super) chosen: Option<(PathBuf, Vec<crate::Allowance>)>,
}

/// A request's claim on the scene it wrote, from asked to landed.
#[cfg(feature = "documents")]
#[derive(Default)]
pub(crate) enum Claim {
    #[default]
    None,
    Asked,
    Waiting(crate::Origin, Vec<Event>),
    Landed(Vec<Event>),
}

impl Backing {
    #[cfg(feature = "documents")]
    pub(super) fn docs(&self) -> &Docs {
        match self {
            Self::Documents(docs) => docs,
            Self::Bundled(_) => panic!("document operation called on a bundled session"),
        }
    }

    #[cfg(feature = "documents")]
    pub(super) fn docs_mut(&mut self) -> &mut Docs {
        match self {
            Self::Documents(docs) => docs,
            Self::Bundled(_) => panic!("document operation called on a bundled session"),
        }
    }

    pub(super) fn bundled_mut(&mut self) -> Option<&mut bundled::BundledScene> {
        match self {
            #[cfg(feature = "documents")]
            Self::Documents(_) => None,
            Self::Bundled(b) => Some(b),
        }
    }

    pub(super) fn graph(&self) -> Option<&Graph> {
        match self {
            #[cfg(feature = "documents")]
            Self::Documents(docs) => docs.scene.graph(),
            Self::Bundled(b) => b.graph.as_deref(),
        }
    }

    pub(super) fn shared_graph(&self) -> Option<&Arc<Graph>> {
        match self {
            #[cfg(feature = "documents")]
            Self::Documents(docs) => docs.scene.shared_graph(),
            Self::Bundled(b) => b.graph.as_ref(),
        }
    }

    pub(super) fn handles(&self) -> &BTreeMap<String, Handle> {
        match self {
            #[cfg(feature = "documents")]
            Self::Documents(docs) => docs.inputs.handles(),
            Self::Bundled(b) => &b.handles,
        }
    }

    pub(super) fn params(&self) -> &BTreeMap<String, Value> {
        match self {
            #[cfg(feature = "documents")]
            Self::Documents(docs) => docs.scene.params(),
            Self::Bundled(b) => &b.params,
        }
    }

    fn is_presented(&self, output: &str) -> bool {
        match self {
            #[cfg(feature = "documents")]
            Self::Documents(docs) => docs
                .scene
                .files()
                .presented
                .iter()
                .any(|p| p.output == output),
            Self::Bundled(b) => b.presented.iter().any(|(out, _)| out == output),
        }
    }

    fn presented_output(&self, device: &str) -> Option<&str> {
        match self {
            #[cfg(feature = "documents")]
            Self::Documents(docs) => docs
                .scene
                .files()
                .presented
                .iter()
                .find(|p| p.device == device)
                .map(|p| p.output.as_str()),
            Self::Bundled(b) => b
                .presented
                .iter()
                .find(|(_, dev)| dev == device)
                .map(|(out, _)| out.as_str()),
        }
    }
}

/// The error an evaluation returns when no scene has loaded.
pub(super) fn no_scene() -> EvaluateError {
    EvaluateError::Graph(entelechy_graph::GraphError::UnknownOutput(
        entelechy_graph::Source::External("<no scene>".into()),
    ))
}

impl crate::Warm {
    /// Opens `backend` (`cpu`, `gpu` or `jit`) once, for many sessions.
    pub fn open(backend: &str, limits: Limits) -> Result<Self, String> {
        let (main, ordered) = Session::backends(backend, entelechy_gpu::Wait::blocking())?;
        Ok(Self::over(main, ordered, limits))
    }
}

/// A main backend and the one for ordered work beside it.
pub(crate) type Backends = (Box<dyn Backend>, Option<Box<dyn Backend>>);
type Display = entelechy_gpu::present::Display;
/// Backends opened, and the main one's display when it has one.
type Opened = (Box<dyn Backend>, Option<Box<dyn Backend>>, Option<Display>);

/// How long a sync prepares Kernels where no thread can start, at least
/// one each.
pub(super) const TURN: std::time::Duration = std::time::Duration::from_millis(12);

impl Session {
    pub(super) fn from_backing(runtime: Runtime, backing: Backing) -> Self {
        Self {
            backing,
            copies_let_go: BTreeSet::new(),
            runtime,
            evaluator: Evaluator::new(),
            used: BTreeMap::new(),
            window: None,
            pins: Vec::new(),
            node_clock: crate::clocked::NodeClock::default(),
            board: crate::clipboard::Board::default(),
            watched: crate::preparing::Watched::default(),
            last_context: Context::default(),
            failing: None,
            shown_while_building: false,
            handed: BTreeSet::new(),
        }
    }

    /// The runtime, for inspection.
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Builds a runtime over a named backend ([`Session::build`]),
    /// blocking on the device.
    pub fn runtime_for(backend: &str, limits: Limits) -> Result<Runtime, String> {
        Self::build(backend, limits, entelechy_gpu::Wait::blocking()).map(|built| built.runtime)
    }

    /// Builds a runtime over `cpu`, `gpu`, `jit`, or `best` (the first
    /// available backend that opens). The device is waited for as `wait`
    /// says.
    pub fn build(
        backend: &str,
        limits: Limits,
        wait: entelechy_gpu::Wait,
    ) -> Result<crate::Built, String> {
        let mut passed_over = Vec::new();
        let one = [backend];
        let named: &[&str] = match backend {
            "best" => BEST,
            _ => &one,
        };
        for name in named {
            match Self::backend_runtime(name, limits.words, wait) {
                Ok((mut runtime, display)) => {
                    runtime.host_mut().set_work(limits.work);
                    runtime.set_small(limits.small);
                    return Ok(crate::Built {
                        runtime,
                        display,
                        passed_over,
                    });
                }
                Err(why) if named.len() > 1 => passed_over.push(format!("{name}: {why}")),
                Err(why) => return Err(why),
            }
        }
        Err(format!("no backend opens: {}", passed_over.join("; ")))
    }

    fn backend_runtime(
        backend: &str,
        budget_words: usize,
        wait: entelechy_gpu::Wait,
    ) -> Result<(Runtime, Option<Display>), String> {
        let (main, ordered, display) = Self::opened(backend, wait)?;
        Ok((Runtime::with_backends(budget_words, main, ordered), display))
    }

    /// The backends `backend` names, opened: the main one and the one for
    /// ordered work (the JIT beside a GPU).
    pub(crate) fn backends(backend: &str, wait: entelechy_gpu::Wait) -> Result<Backends, String> {
        Self::opened(backend, wait).map(|(main, ordered, _)| (main, ordered))
    }

    fn opened(backend: &str, wait: entelechy_gpu::Wait) -> Result<Opened, String> {
        match backend {
            #[cfg(any(feature = "documents", target_os = "ios"))]
            "cpu" => Ok((Box::new(entelechy_cpu::CpuBackend), None, None)),
            #[cfg(not(any(feature = "documents", target_os = "ios")))]
            "cpu" => Err(
                "the interpreter backend is not included in this runtime; use gpu, jit, or best"
                    .to_owned(),
            ),
            "gpu" => {
                let gpu = entelechy_gpu::GpuBackend::with_wait(wait)
                    .map_err(|error| error.to_string())?;
                let display = gpu.display();
                Ok((Box::new(gpu), Self::ordered_beside_gpu(), Some(display)))
            }
            "jit" => Ok((Self::jit()?, None, None)),
            other => Err(format!(
                "unknown backend `{other}`; use cpu, gpu, jit, or best"
            )),
        }
    }

    /// The JIT, where the machine or a WebAssembly host runs its code.
    #[cfg(not(target_os = "ios"))]
    fn jit() -> Result<Box<dyn Backend>, String> {
        entelechy_jit::JitBackend::new().map(|jit| Box::new(jit) as Box<dyn Backend>)
    }

    /// iOS forbids mapping executable pages for runtime-generated code.
    #[cfg(target_os = "ios")]
    fn jit() -> Result<Box<dyn Backend>, String> {
        Err("the JIT is unavailable on iOS: executable memory cannot be mapped".into())
    }

    /// The processor backend kept beside a GPU for small or ordered runs.
    fn ordered_beside_gpu() -> Option<Box<dyn Backend>> {
        if let Ok(jit) = Self::jit() {
            return Some(jit);
        }
        #[cfg(any(feature = "documents", target_os = "ios"))]
        {
            Some(Box::new(entelechy_cpu::CpuBackend))
        }
        #[cfg(not(any(feature = "documents", target_os = "ios")))]
        None
    }

    /// The current graph, if the scene loaded.
    pub fn graph(&self) -> Option<&Graph> {
        self.backing.graph()
    }

    /// Current graph parameter overrides.
    pub fn params(&self) -> &BTreeMap<String, Value> {
        self.backing.params()
    }

    /// Names the run the next evaluations are, from its first frame.
    pub fn recall_as(&mut self, name: Option<String>) {
        self.evaluator.recall_as(name);
    }

    /// The bytes remembered runs hold.
    pub fn recalled_bytes(&self) -> usize {
        self.evaluator.recalled_bytes()
    }

    /// The system's clipboard for the scene's `= clipboard` inputs and `to
    /// clipboard` outputs, in the session's own one's stead.
    pub fn give_clipboard(&mut self, device: Box<dyn crate::Clipboard>) {
        self.board.give(device);
    }

    /// The Kernels `progress` watches, for the scene's `= progress`
    /// inputs, in the session's own stead: another scene's, preparing.
    pub fn watch(&mut self, progress: entelechy_kernel::Progress) {
        self.watched.give(progress);
    }

    /// Where each node with a published Kernel runs, and why when off the
    /// main backend.
    pub fn placements(&self) -> Vec<(String, entelechy_kernel::Placement)> {
        let Some(graph) = self.graph() else {
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

    pub(super) fn finish_used(&mut self) {
        if self.runtime.prepares_in_turns() && self.runtime.prepare_queued() > 0 {
            return;
        }
        let Some(shared) = self.backing.shared_graph().cloned() else {
            return;
        };
        let few = self.evaluator.runs_few(
            &shared,
            &self.runtime,
            self.backing.handles(),
            self.backing.params(),
            &self.last_context,
        );
        let graph: &Graph = &shared;
        let mut wanted: BTreeMap<entelechy_ir::KernelId, bool> = BTreeMap::new();
        for node in &graph.nodes {
            *wanted.entry(node.kernel.clone()).or_default() |= !few.contains(&node.id);
        }
        let mut now = BTreeMap::new();
        let mut first = Vec::new();
        for node in &graph.nodes {
            let Some(version) = self.runtime.kernel(&node.kernel) else {
                continue;
            };
            let id = version.id().clone();
            if now.insert(id.clone(), version.generation()).is_none()
                && !self.used.contains_key(&id)
                && wanted.get(&node.kernel).copied().unwrap_or(true)
            {
                first.push(version);
            }
        }
        let shown = if self.shown_while_building && !self.runtime.prepares_in_turns() {
            match &self.backing {
                #[cfg(feature = "documents")]
                Backing::Documents(docs) => shown_kernels(
                    graph,
                    docs.scene
                        .files()
                        .presented
                        .iter()
                        .map(|p| (p.output.as_str(), p.device.as_str())),
                ),
                Backing::Bundled(b) => shown_kernels(
                    graph,
                    b.presented
                        .iter()
                        .map(|(out, dev)| (out.as_str(), dev.as_str())),
                ),
            }
        } else {
            BTreeSet::new()
        };
        let (later, now_first): (Vec<_>, Vec<_>) = first
            .into_iter()
            .partition(|version| shown.contains(version.id()));
        let preparer = self.runtime.preparer();
        if !now_first.is_empty() {
            preparer.expect(now_first.iter().map(|version| version.id()));
            finish_each(&now_first, |version| preparer.finish(version));
        }
        let handed_now: Vec<_> = later.iter().map(|version| version.id().clone()).collect();
        if !later.is_empty() {
            preparer.expect(later.iter().map(|version| version.id()));
            let started = std::thread::Builder::new()
                .name("entelechy-finish".into())
                .spawn(move || {
                    finish_each(&later, |version| preparer.finish(version));
                });
            if started.is_ok() {
                self.handed.extend(handed_now);
            }
        }
        self.runtime
            .set_shown_while_building(shown.intersection(&self.handed).cloned().collect());
        self.used = now;
    }

    /// Shows what only shows from the processor while the device's
    /// compiler builds it.
    pub fn show_while_building(&mut self, on: bool) {
        self.shown_while_building = on;
    }

    /// Prepares replacements of Kernels and fused groups on a worker
    /// thread.
    pub fn set_background(&mut self, enabled: bool) {
        self.runtime.set_background(enabled);
        #[cfg(feature = "documents")]
        if let Backing::Documents(docs) = &mut self.backing {
            docs.inputs.set_background(enabled);
        }
    }

    /// Whether replacements prepare on a worker thread.
    pub fn background(&self) -> bool {
        self.runtime.background()
    }

    /// Prepares Kernels a few each sync instead of on workers, as where no
    /// thread can start (a page).
    pub fn prepare_in_turns(&mut self) {
        self.runtime.prepare_in_turns();
        #[cfg(feature = "documents")]
        if let Backing::Documents(docs) = &mut self.backing {
            docs.inputs.set_background(true);
        }
    }

    /// One wait for a preparation to finish.
    pub fn wait_a_turn(&mut self) {
        if self.runtime.prepares_in_turns() {
            self.runtime.prepare_turn(usize::MAX);
        } else {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// What preparing the scene's Kernels is doing, readable from any
    /// thread while a sync prepares them ([`entelechy_kernel::Progress`]).
    pub fn progress(&self) -> entelechy_kernel::Progress {
        self.runtime.progress()
    }

    /// The Kernels still waiting their turn to prepare where no thread can
    /// start, and one more while what they prepare for has yet to land; zero
    /// elsewhere.
    pub fn preparing_left(&self) -> usize {
        let landing = self.runtime.prepares_in_turns() && self.scene_waiting();
        self.runtime.prepare_queued() + usize::from(landing)
    }

    /// Whether the scene last read waits for its Kernels before it
    /// replaces the running one.
    pub fn scene_waiting(&self) -> bool {
        match &self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(docs) => docs.scene.waiting() || docs.kernels.preparing(),
            Backing::Bundled(b) => b.pending.is_some(),
        }
    }

    /// The Field a scene input holds now.
    pub fn external(&self, name: &str) -> Option<Handle> {
        self.backing.handles().get(name).copied()
    }

    /// The node and port that produce what `node.port` names.
    pub fn producer(
        &self,
        node: &entelechy_graph::NodeId,
        port: &str,
    ) -> (entelechy_graph::NodeId, String) {
        self.graph()
            .and_then(|graph| graph.producer(node, port))
            .unwrap_or_else(|| (node.clone(), port.to_owned()))
    }

    /// Makes again what `node.port` names when its outputs were let go
    /// cold ([`entelechy_graph::Evaluator::is_cold`]), under the last
    /// frame's context, committing nothing.
    pub fn warm(
        &mut self,
        node: &entelechy_graph::NodeId,
        port: &str,
    ) -> Result<(), EvaluateError> {
        let (node, _) = self.producer(node, port);
        if !self.evaluator.is_cold(&node) {
            return Ok(());
        }
        let graph = self.backing.graph().ok_or_else(no_scene)?;
        let context = self.last_context;
        self.evaluator.warm_for_reading(
            graph,
            &mut self.runtime,
            self.backing.handles(),
            &context,
            self.backing.params(),
            &node,
        )
    }

    /// The nodes whose outputs were let go cold
    /// ([`entelechy_graph::Evaluator::is_cold`]).
    pub fn cold_nodes(&self) -> Vec<entelechy_graph::NodeId> {
        self.evaluator.cold_nodes()
    }

    /// The Host, to let go of what it keeps ([`crate::memory`]).
    pub(crate) fn host_mut(&mut self) -> &mut entelechy_kernel::Host {
        self.runtime.host_mut()
    }

    /// The scene's inputs, by name, and the Fields they are now.
    pub(crate) fn input_handles(&self) -> &BTreeMap<String, Handle> {
        self.backing.handles()
    }

    /// The Fields fields nodes hold of the files in, by node and path.
    pub(crate) fn chunk_handles(&self) -> Vec<(String, Handle)> {
        match &self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(docs) => docs.inputs.chunk_handles().collect(),
            Backing::Bundled(_) => Vec::new(),
        }
    }

    /// Every Field the evaluator holds ([`entelechy_graph::Evaluator::holdings`]).
    pub(crate) fn holdings(&self) -> Vec<entelechy_graph::Holding> {
        self.evaluator.holdings()
    }

    /// The Field last evaluated for what `node.port` names, if any.
    pub fn cached(&self, node: &entelechy_graph::NodeId, port: &str) -> Option<Handle> {
        let (node, port) = self.producer(node, port);
        self.evaluator.cached(&node, &port)
    }

    /// The node `node` runs fused into, if any
    /// ([`entelechy_graph::Evaluator::fused_into`]).
    pub fn fused_into(&self, node: &entelechy_graph::NodeId) -> Option<entelechy_graph::NodeId> {
        let (node, _) = self.producer(node, "result");
        self.evaluator.fused_into(&node).cloned()
    }

    /// The state ports (`node.port`) whose initial source is a node running
    /// `kernel`.
    pub fn started_by(&self, kernel: &KernelId) -> Vec<String> {
        let Some(graph) = self.graph() else {
            return Vec::new();
        };
        let runs = |node: &entelechy_graph::NodeId| {
            graph
                .nodes
                .iter()
                .any(|n| &n.id == node && &n.kernel == kernel)
        };
        let mut out = Vec::new();
        for node in &graph.nodes {
            for (port, binding) in &node.inputs {
                if let entelechy_graph::Binding::Previous {
                    initial: entelechy_graph::Source::Node { node: from, .. },
                    ..
                } = binding
                    && runs(from)
                {
                    out.push(format!("{}.{port}", node.id));
                }
            }
        }
        out
    }

    /// Waits for every `async` node's run and takes it in; see
    /// [`entelechy_graph::Evaluator::settle`].
    pub fn settle(&mut self) -> Result<Vec<entelechy_graph::NodeId>, EvaluateError> {
        self.evaluator.settle(&mut self.runtime)
    }

    /// Whether any `async` node's run is still going.
    pub fn has_running(&self) -> bool {
        self.evaluator.has_running()
    }

    pub(super) fn one_frame_in_flight(&self) {
        if self.runtime.has_device() {
            self.runtime.settle_device();
        }
    }

    /// Whether later evaluations overlap the device's work on the frame
    /// before; see [`entelechy_graph::Evaluator::set_overlap`].
    pub fn set_overlap(&mut self, on: bool) -> Result<(), EvaluateError> {
        let set = self.evaluator.set_overlap(on, &mut self.runtime);
        self.tally();
        set
    }

    /// Whether evaluations overlap ([`Session::set_overlap`]).
    pub fn overlaps(&self) -> bool {
        self.evaluator.overlaps()
    }

    /// Whether frames are clocked on the device as they run; see
    /// [`entelechy_kernel::Runtime::set_clocked`].
    pub fn set_clocked(&mut self, on: bool) {
        self.runtime.set_clocked(on);
        if !on {
            self.node_clock.clear();
        }
    }

    /// The device's busy time for the latest clocked frame it has done.
    pub fn frame_device_time(&self) -> Option<std::time::Duration> {
        self.runtime.frame_device_time()
    }

    /// The latest clocked frame the device has done, node by node.
    pub fn device_times(&self) -> Option<crate::DeviceTimes> {
        self.node_clock.times(&self.runtime.clocked_frame()?)
    }

    /// How long callers have been blocked on the device's work in all.
    pub fn device_waited(&self) -> std::time::Duration {
        self.runtime.device_waited()
    }

    /// Confirms the frame in flight, if any; see
    /// [`entelechy_graph::Evaluator::confirm`].
    pub fn confirm(&mut self) -> Result<(), EvaluateError> {
        let confirmed = self.evaluator.confirm(&mut self.runtime);
        self.tally();
        #[cfg(feature = "documents")]
        self.write_confirmed();
        confirmed
    }

    fn tally(&mut self) {
        let taken = self.evaluator.take_taken_back();
        #[cfg(feature = "documents")]
        if let Backing::Documents(docs) = &mut self.backing {
            docs.history.taken_back(taken);
        }
        #[cfg(not(feature = "documents"))]
        let _ = taken;
    }

    /// Evaluates the current graph under `context`.
    pub fn evaluate(&mut self, context: &Context) -> Result<Report, EvaluateError> {
        #[cfg(feature = "documents")]
        self.place_by_rows();
        if !self.evaluator.overlaps() {
            self.one_frame_in_flight();
        }
        #[cfg(feature = "documents")]
        if let Backing::Documents(docs) = &mut self.backing {
            let store = Arc::clone(docs.scene.store());
            docs.inputs
                .select_frames(store.as_ref(), self.runtime.host_mut(), context.time);
        }
        self.hold_progress();
        self.repin();
        self.last_context = *context;
        #[cfg(feature = "documents")]
        self.await_written();
        let graph = self.backing.shared_graph().cloned().ok_or_else(no_scene)?;
        let evaluated = self.evaluator.evaluate(
            &graph,
            &mut self.runtime,
            self.backing.handles(),
            context,
            self.backing.params(),
        );
        let report = match evaluated {
            Ok(report) => report,
            Err(error) => {
                self.tally();
                let error = self.with_refusal(error);
                self.failing = Some(self.told(&error));
                return Err(error);
            }
        };
        self.failing = None;
        #[cfg(feature = "documents")]
        if let Backing::Documents(docs) = &mut self.backing {
            docs.history.ran();
        }
        self.feed_fonts();
        #[cfg(feature = "documents")]
        {
            self.feed_file_nodes();
            self.feed_field_nodes();
        }
        self.let_go_of_input_copies();
        if let Some(frame) = self.runtime.end_frame() {
            self.node_clock.ended(frame, report.clocked.clone());
        }
        self.written_after(&report);
        #[cfg(feature = "documents")]
        self.follow_rows();
        Ok(report)
    }

    pub(super) fn written_after(&mut self, report: &Report) {
        match &mut self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(docs) => {
                docs.writes
                    .ask(&docs.scene.files().writes, &report.outputs, &self.runtime);
                let to_clipboard: Vec<String> = docs
                    .scene
                    .files()
                    .presented
                    .iter()
                    .filter(|p| p.device == crate::devices::CLIPBOARD)
                    .map(|p| p.output.clone())
                    .collect();
                self.write_confirmed();
                self.board.after(
                    to_clipboard.iter().map(String::as_str),
                    report,
                    &mut self.runtime,
                );
            }
            Backing::Bundled(b) => {
                let to_clipboard: Vec<String> = b
                    .presented
                    .iter()
                    .filter(|(_, dev)| dev == crate::devices::CLIPBOARD)
                    .map(|(out, _)| out.clone())
                    .collect();
                self.board.after(
                    to_clipboard.iter().map(String::as_str),
                    report,
                    &mut self.runtime,
                );
            }
        }
    }

    /// A failed evaluation as it is said: a node that ran others fused
    /// into it names them, as the fault may be any of theirs.
    pub fn told(&self, error: &EvaluateError) -> String {
        let said = error.to_string();
        let EvaluateError::Execution { node, .. } = error else {
            return said;
        };
        let parts = self.evaluator.fused_parts(node);
        if parts.is_empty() {
            return said;
        }
        let parts: Vec<String> = parts.iter().map(|part| format!("`{part}`")).collect();
        format!(
            "{said} (it ran {} fused into it: the fault may be any of theirs)",
            parts.join(", ")
        )
    }

    pub(super) fn with_refusal(&self, error: EvaluateError) -> EvaluateError {
        #[cfg(feature = "documents")]
        if let Backing::Documents(docs) = &self.backing
            && let EvaluateError::UnknownKernel {
                node,
                kernel,
                near,
                refused: None,
            } = error
        {
            let refused = docs.kernels.refused(&kernel).map(str::to_owned);
            return EvaluateError::UnknownKernel {
                node,
                kernel,
                near,
                refused,
            };
        }
        error
    }

    pub(super) fn hold_progress(&mut self) {
        let own = self.runtime.progress();
        match &mut self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(docs) => {
                let progresses = docs.scene.files().fed_by(crate::devices::PROGRESS);
                let already = progresses.iter().all(|name| docs.inputs.holds(name));
                let inputs = &mut docs.inputs;
                let said = self.watched.sync(
                    &progresses,
                    already,
                    &own,
                    self.runtime.host_mut(),
                    |host, name, h| inputs.hold(host, name, h),
                );
                docs.later
                    .extend(said.into_iter().map(|why| Event::Rejected {
                        path: PathBuf::from(crate::devices::PROGRESS),
                        error: crate::Diagnostic::new(why),
                    }));
            }
            Backing::Bundled(b) => {
                let progresses: Vec<String> = b
                    .feeds
                    .iter()
                    .filter(|(_, dev)| dev == crate::devices::PROGRESS)
                    .map(|(inp, _)| inp.clone())
                    .collect();
                let already = progresses.iter().all(|name| b.handles.contains_key(name));
                let handles = &mut b.handles;
                let said = self.watched.sync(
                    &progresses,
                    already,
                    &own,
                    self.runtime.host_mut(),
                    |host, name, h| bundled::hold_in(handles, host, name, h),
                );
                b.said
                    .extend(said.into_iter().map(|why| format!("progress: {why}")));
            }
        }
    }

    /// Forgets committed state; `<~` bindings restart from their initial
    /// sources on the next evaluation.
    pub fn reset_state(&mut self) {
        self.evaluator.reset_state(&mut self.runtime);
        #[cfg(feature = "documents")]
        if let Backing::Documents(docs) = &mut self.backing {
            docs.history
                .changed(crate::Change::Reset, docs.origin.clone());
        }
    }

    /// Sets the window a headless session reports when an evaluation names
    /// none: size in pixels and scale factor.
    pub fn set_window(&mut self, window: Option<([f32; 2], f32)>) {
        self.window = window;
    }

    /// The window a headless session reports, if one was set.
    pub fn window(&self) -> Option<([f32; 2], f32)> {
        self.window
    }

    /// The context the last frame was evaluated under.
    pub fn last_context(&self) -> Context {
        self.last_context
    }

    /// Makes the backend wait for its device after every node, so a
    /// profile measures the work, not the submission.
    pub fn set_measured(&mut self, on: bool) {
        self.runtime.backend().set_measured(on);
    }

    /// The fixed cost of measuring one node.
    pub fn measurement_floor(&self) -> std::time::Duration {
        self.runtime().measurement_floor()
    }

    /// Whether a profile's numbers come from the device's own clock.
    pub fn measures_the_device(&self) -> bool {
        self.runtime().backend().measures_the_device()
    }

    /// Enables or disables fusion of pointwise chains.
    pub fn set_fusion(&mut self, enabled: bool) {
        self.evaluator.set_fusion(enabled);
    }

    /// Whether fusion is enabled.
    pub fn fusion(&self) -> bool {
        self.evaluator.fusion()
    }

    /// Keeps what `node` names materialized so it can be read.
    pub fn pin(&mut self, node: entelechy_graph::NodeId) {
        if !self.pins.contains(&node) {
            self.pins.push(node);
        }
        self.repin();
    }

    /// What would execute right now, in order, with fusion applied.
    pub fn plan(&mut self) -> Result<Vec<entelechy_graph::PlanEntry>, EvaluateError> {
        self.repin();
        let graph = self.backing.shared_graph().cloned().ok_or_else(no_scene)?;
        let mut context = self.last_context;
        if context.window == [0.0, 0.0]
            && let Some((size, scale)) = self.window
        {
            context.window = size;
            context.scale = scale;
        }
        self.evaluator.plan_summary(
            &graph,
            &mut self.runtime,
            self.backing.handles(),
            self.backing.params(),
            &context,
        )
    }

    /// Lets fusion have what `node` names back.
    pub fn unpin(&mut self, node: &entelechy_graph::NodeId) {
        self.pins.retain(|pinned| pinned != node);
        for inner in self.pinned_by(node) {
            self.evaluator.unpin(&inner);
        }
    }

    pub(super) fn pinned_by(&self, name: &entelechy_graph::NodeId) -> Vec<entelechy_graph::NodeId> {
        let graph = self.graph();
        let last = graph
            .and_then(|graph| {
                graph
                    .repeated
                    .iter()
                    .find(|(repeated, _)| *repeated == name.0)
            })
            .map_or_else(|| name.clone(), |(_, last)| last.clone());
        let name = &last;
        let placed = graph.and_then(|graph| graph.placed.iter().find(|(id, _)| *id == name.0));
        match placed {
            Some((_, outputs)) => outputs
                .iter()
                .filter_map(|(_, source)| match source {
                    entelechy_graph::Source::Node { node, .. } => Some(node.clone()),
                    entelechy_graph::Source::External(_) => None,
                })
                .collect(),
            None => vec![name.clone()],
        }
    }

    pub(super) fn repin(&mut self) {
        for name in self.pins.clone() {
            for node in self.pinned_by(&name) {
                self.evaluator.pin(node);
            }
        }
        for node in self.fed_sources() {
            self.evaluator.pin(node);
        }
        #[cfg(feature = "documents")]
        {
            for node in self.written_sources() {
                self.evaluator.pin(node);
            }
            if let Backing::Documents(docs) = &self.backing {
                let rows = docs.scene.times_rows();
                for (row, _) in rows {
                    let source = entelechy_graph::NodeId(row.source.0.as_str().into());
                    for node in self.pinned_by(&source) {
                        self.evaluator.pin(node);
                    }
                }
            }
        }
    }

    pub(super) fn fed_sources(&self) -> Vec<entelechy_graph::NodeId> {
        match &self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(docs) => {
                let files = docs.scene.files();
                files
                    .fonts
                    .iter()
                    .map(|font| &font.codes)
                    .chain(files.file_nodes.iter().map(|node| &node.paths))
                    .chain(files.field_nodes.iter().map(|node| &node.paths))
                    .filter_map(|source| match source {
                        entelechy_graph::Source::Node { node, port } => {
                            Some(self.producer(node, port).0)
                        }
                        entelechy_graph::Source::External(_) => None,
                    })
                    .collect()
            }
            Backing::Bundled(b) => b
                .fonts
                .iter()
                .filter_map(|font| match &font.atlas.codes {
                    entelechy_graph::Source::Node { node, port } => {
                        Some(self.producer(node, port).0)
                    }
                    entelechy_graph::Source::External(_) => None,
                })
                .collect(),
        }
    }

    pub(super) fn feed_fonts(&mut self) {
        match &self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(_) => self.feed_document_fonts(),
            Backing::Bundled(_) => self.feed_bundled_fonts(),
        }
    }

    /// A Field's words taken later, not waiting for the work submitted
    /// after this call; see [`entelechy_kernel::Runtime::fetch`].
    pub fn fetch(&self, handle: Handle) -> Result<entelechy_kernel::Fetch, SessionError> {
        self.runtime
            .fetch(handle)
            .map_err(|e| SessionError::Observe(e.to_string()))
    }

    /// A Field's data as the Host holds it, brought back from the device
    /// first.
    pub fn field(&mut self, handle: Handle) -> Result<&entelechy_kernel::FieldData, SessionError> {
        self.runtime
            .materialize(handle)
            .map_err(|e| SessionError::Observe(e.to_string()))?;
        self.runtime.host().data(handle).map_err(SessionError::Host)
    }

    /// The output of `report` every interface shows: the first picture
    /// ([`entelechy_kernel::Schema::is_picture`]) the scene gives no
    /// device.
    pub fn picture_of(&self, report: &Report) -> Option<Handle> {
        report
            .outputs
            .iter()
            .filter(|(name, _)| !self.backing.is_presented(name))
            .map(|(_, handle)| *handle)
            .find(|handle| self.is_picture(*handle))
    }

    /// Whether the Field `handle` is a picture
    /// ([`entelechy_kernel::Schema::is_picture`]).
    pub fn is_picture(&self, handle: Handle) -> bool {
        let host = self.runtime.host();
        match (host.schema(handle), host.domain(handle)) {
            (Ok(schema), Ok(domain)) => schema.is_picture(domain),
            _ => false,
        }
    }

    /// The Field of the scene output `name` (the first when `None`) as
    /// the last evaluation left it; the one place outputs are read by
    /// name.
    pub fn output(&self, name: Option<&str>) -> Result<Handle, String> {
        let graph = self.graph().ok_or("no scene is loaded")?;
        let (name, source) = match name {
            Some(name) => graph
                .outputs
                .iter()
                .find(|(output, _)| output == name)
                .ok_or_else(|| {
                    #[cfg(feature = "documents")]
                    let hint = entelechy_language::did_you_mean(
                        name,
                        graph.outputs.iter().map(|(output, _)| output.as_str()),
                    );
                    #[cfg(not(feature = "documents"))]
                    let hint = "";
                    format!("no output named `{name}`{hint}")
                })?,
            None => graph.primary_output().ok_or("the graph has no output")?,
        };
        match source {
            entelechy_graph::Source::Node { node, port } => self
                .cached(node, port)
                .ok_or_else(|| "evaluate before reading an output".to_owned()),
            entelechy_graph::Source::External(_) => {
                Err(format!("output `{name}` is an external image"))
            }
        }
    }

    /// Where the device holds `handle` and its size, when it is wide
    /// float pixels one deep, shown as it lies. Anything else is shown
    /// through [`Session::picture`].
    pub fn on_device(
        &self,
        handle: Handle,
    ) -> Option<(&(dyn std::any::Any + Send + Sync), u32, u32)> {
        let host = self.runtime.host();
        let as_it_lies = host
            .schema(handle)
            .is_ok_and(|s| s.is_pixel() && s.is_wide())
            && host.domain(handle).is_ok_and(|d| d.depth() == 1);
        if !as_it_lies {
            return None;
        }
        let device = host.resident(handle).ok().flatten()?;
        let (width, height) = host.image_size(handle).ok()?;
        Some((device, width, height))
    }

    /// The Field of `report` the scene gives device `device`, when it
    /// gives one and the Field is of the structure the device takes.
    pub fn presented(&self, report: &Report, device: &str) -> Option<Handle> {
        let output = self.backing.presented_output(device)?;
        let handle = report
            .outputs
            .iter()
            .find(|(name, _)| name == output)
            .map(|(_, handle)| *handle)?;
        let host = self.runtime.host();
        let (schema, domain) = (host.schema(handle).ok()?, host.domain(handle).ok()?);
        crate::devices::fits_output(device, domain.shape(), schema).then_some(handle)
    }

    /// The output of `report` every interface shows as its pointer: the
    /// picture the scene gives `to pointer`.
    pub fn pointer_of(&self, report: &Report) -> Option<Handle> {
        self.presented(report, crate::devices::POINTER)
    }

    /// The output of `report` that says where the pointer points within its
    /// picture: the rows the scene gives `to pointer_tip`.
    pub fn hotspot_of(&self, report: &Report) -> Option<Handle> {
        self.presented(report, crate::devices::POINTER_TIP)
            .filter(|handle| {
                self.runtime
                    .host()
                    .domain(*handle)
                    .is_ok_and(|d| d.count() > 0)
            })
    }

    /// The output of `report` every interface plays, and its channels: the
    /// rows the scene gives `to sound`.
    pub fn sound_of(&self, report: &Report) -> Option<(Handle, usize)> {
        let handle = self.presented(report, crate::devices::SOUND)?;
        let channels = self.runtime.host().schema(handle).ok()?.value_words();
        Some((handle, channels))
    }

    /// A Field of pixels as an image, brought back from the device first.
    pub fn picture(&mut self, handle: Handle) -> Result<entelechy_kernel::Image, SessionError> {
        self.runtime
            .image(handle)
            .map_err(|e| SessionError::Observe(e.to_string()))
    }
}

fn finish_each<T: Sync>(items: &[T], f: impl Fn(&T) + Sync) {
    #[cfg(feature = "documents")]
    {
        entelechy_documents::spread::map(items, f);
    }
    #[cfg(not(feature = "documents"))]
    for item in items {
        f(item);
    }
}

fn shown_kernels<'a>(
    graph: &Graph,
    devices: impl Iterator<Item = (&'a str, &'a str)>,
) -> BTreeSet<KernelId> {
    let mut presented: BTreeSet<String> = graph
        .outputs
        .iter()
        .take(1)
        .map(|(name, _)| name.clone())
        .collect();
    presented.extend(
        devices
            .filter(|(_, device)| crate::devices::every_frame(device))
            .map(|(output, _)| output.to_owned()),
    );
    let shown = graph.shown_only(&presented);
    let mut kernels: BTreeMap<&KernelId, bool> = BTreeMap::new();
    for node in &graph.nodes {
        *kernels.entry(&node.kernel).or_insert(true) &= shown.contains(&node.id);
    }
    kernels
        .into_iter()
        .filter(|(_, only)| *only)
        .map(|(id, _)| id.clone())
        .collect()
}
