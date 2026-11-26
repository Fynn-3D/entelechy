//! The session: its runtime, evaluator and backing scene (either live
//! source documents or a compiled [`crate::bundle::Bundle`]).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
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
mod saves;
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
    /// Whether at least one evaluation or warp tick has succeeded since the
    /// scene opened or state was last reset.
    pub(super) has_evaluated: bool,
    /// The most recent successful evaluation report, for live diagnostics.
    pub(super) last_report: Option<Report>,
    /// Why the last evaluation failed, until one succeeds.
    pub(super) failing: Option<String>,
    /// Whether what only shows is shown from the processor while the
    /// device's compiler builds it ([`Session::show_while_building`]).
    shown_while_building: bool,
    /// The Kernels handed to threads of their own to be built there
    /// ([`Session::finish_used`]).
    handed: BTreeSet<KernelId>,
    /// Kernels queued to finish in turns where no thread can start (a page),
    /// and how many at the front have already been warmed ahead.
    finishing: VecDeque<Arc<entelechy_kernel::KernelVersion>>,
    warmed: usize,
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

    /// Opens `backend` (`cpu`, `gpu`, `jit`, or `best`) once for a live window
    /// that may switch scenes across sessions while sharing one device display
    /// and compiled Kernel cache.
    pub fn open_for_window(
        backend: &str,
        limits: Limits,
        wait: entelechy_gpu::Wait,
    ) -> Result<(Self, Option<Display>, Vec<String>), String> {
        let mut passed_over = Vec::new();
        let one = [backend];
        let named: &[&str] = match backend {
            "best" => BEST,
            _ => &one,
        };
        for name in named {
            match Session::opened(name, wait) {
                Ok((main, ordered, display)) => {
                    return Ok((Self::over(main, ordered, limits), display, passed_over));
                }
                Err(why) if named.len() > 1 => passed_over.push(format!("{name}: {why}")),
                Err(why) => return Err(why),
            }
        }
        Err(format!("no backend opens: {}", passed_over.join("; ")))
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
/// How many Kernels at most warm their device pipelines concurrently ahead
/// of [`Session::finish_turn`] in a page, keeping the browser's GPU process
/// fed without saturating Windows DXC / Dawn worker threads past Chrome's
/// GPU watchdog.
const WARM_AHEAD: usize = 2;

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
            has_evaluated: false,
            last_report: None,
            failing: None,
            shown_while_building: false,
            handed: BTreeSet::new(),
            finishing: VecDeque::new(),
            warmed: 0,
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
        let handles = self.backing.handles();
        let output_seeds = shared
            .outputs
            .iter()
            .filter_map(|(_, source)| match source {
                entelechy_graph::Source::Node { node, .. } => Some(node.clone()),
                entelechy_graph::Source::External(_) => None,
            });
        let fallback_seeds = shared
            .fallbacks
            .iter()
            .filter(|(name, _)| !handles.contains_key(name))
            .filter_map(|(_, source)| match source {
                entelechy_graph::Source::Node { node, .. } => Some(node.clone()),
                entelechy_graph::Source::External(_) => None,
            });
        let reached = shared.read_through(
            self.pins
                .iter()
                .cloned()
                .chain(output_seeds)
                .chain(fallback_seeds),
        );
        let mut warm_context = self.last_context;
        if warm_context.window[0] <= 0.0 || warm_context.window[1] <= 0.0 {
            warm_context.window = [1280.0, 720.0];
            if warm_context.scale <= 0.0 {
                warm_context.scale = 1.0;
            }
        }
        let few = self.evaluator.runs_few(
            &shared,
            &self.runtime,
            handles,
            self.backing.params(),
            &warm_context,
        );
        let graph: &Graph = &shared;
        let mut wanted: BTreeMap<entelechy_ir::KernelId, bool> = BTreeMap::new();
        for node in &graph.nodes {
            if !reached.contains(&node.id) {
                continue;
            }
            *wanted.entry(node.kernel.clone()).or_default() |= !few.contains(&node.id);
        }
        let preparer = self.runtime.preparer();
        let mut now = BTreeMap::new();
        let mut first = Vec::new();
        for node in &graph.nodes {
            if !wanted.get(&node.kernel).copied().unwrap_or(false) {
                continue;
            }
            let Some(version) = self.runtime.kernel(&node.kernel) else {
                continue;
            };
            let id = version.id().clone();
            if now.insert(id.clone(), version.generation()).is_none()
                && !self.used.contains_key(&id)
            {
                first.push(version);
            }
        }
        if self.runtime.prepares_in_turns() {
            if !first.is_empty() {
                preparer.expect(first.iter().map(|version| version.id()));
                self.finishing.extend(first);
                self.warm_finishing(&preparer);
            }
            self.used = now;
            return;
        }
        let shown = if self.shown_while_building {
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

    fn warm_finishing(&mut self, preparer: &entelechy_kernel::Preparer) {
        let heavy = self.finishing.iter().take(WARM_AHEAD).any(|version| {
            version
                .module()
                .is_some_and(|m| m.typical_work() > 8_000 || m.functions.len() > 20)
        });
        let limit = if heavy { 1 } else { WARM_AHEAD };
        let target = self.finishing.len().min(limit);
        while self.warmed < target {
            preparer.warm(&self.finishing[self.warmed]);
            self.warmed += 1;
        }
    }

    pub(super) fn finish_turn(&mut self) {
        if self.finishing.is_empty() {
            self.warmed = 0;
            return;
        }
        let preparer = self.runtime.preparer();
        self.warm_finishing(&preparer);
        let began = entelechy_kernel::clock::now();
        while let Some(version) = self.finishing.pop_front() {
            self.warmed = self.warmed.saturating_sub(1);
            preparer.finish(&version);
            self.warm_finishing(&preparer);
            if self.finishing.is_empty() || entelechy_kernel::clock::since(began) >= TURN {
                break;
            }
        }
        if self.finishing.is_empty() {
            self.warmed = 0;
        }
    }

    pub(super) fn finish_all_turns(&mut self) {
        if self.finishing.is_empty() {
            self.warmed = 0;
            return;
        }
        let preparer = self.runtime.preparer();
        self.warm_finishing(&preparer);
        while let Some(version) = self.finishing.pop_front() {
            self.warmed = self.warmed.saturating_sub(1);
            preparer.finish(&version);
            self.warm_finishing(&preparer);
        }
        self.warmed = 0;
    }

    /// Whether this session was opened from a compiled [`crate::bundle::Bundle`]
    /// rather than source documents.
    pub fn is_bundled(&self) -> bool {
        matches!(self.backing, Backing::Bundled(_))
    }

    /// Whether the loaded scene requires a real compute GPU (`wgpu`) to run
    /// interactively without locking up a single-threaded browser main thread:
    /// true for 3D worlds using `Volume` fields, `Group` workgroups, heavy
    /// raymarching/SDF Kernels, or large multi-pass hardware rasterization.
    pub fn needs_compute_gpu(&self) -> bool {
        match &self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(_) => {}
            Backing::Bundled(b) if b.needs_compute_gpu => return true,
            Backing::Bundled(_) => {}
        }
        let Some(graph) = self.graph() else {
            return false;
        };
        let handles = self.backing.handles();
        let output_seeds = graph.outputs.iter().filter_map(|(_, source)| match source {
            entelechy_graph::Source::Node { node, .. } => Some(node.clone()),
            entelechy_graph::Source::External(_) => None,
        });
        let fallback_seeds = graph
            .fallbacks
            .iter()
            .filter(|(name, _)| !handles.contains_key(name))
            .filter_map(|(_, source)| match source {
                entelechy_graph::Source::Node { node, .. } => Some(node.clone()),
                entelechy_graph::Source::External(_) => None,
            });
        let reached = graph.read_through(
            self.pins
                .iter()
                .cloned()
                .chain(output_seeds)
                .chain(fallback_seeds),
        );
        let mut seen_kernels = BTreeSet::new();
        let mut has_native_raster = false;
        let mut has_non_map = false;
        let mut sum_typical = 0u64;
        for node in &graph.nodes {
            if !reached.contains(&node.id) {
                continue;
            }
            let Some(version) = self.runtime.kernel(&node.kernel) else {
                continue;
            };
            if version.native().is_some() {
                has_native_raster = true;
                continue;
            }
            let manifest = version.manifest();
            if matches!(manifest.mode, entelechy_ir::Mode::Group { .. }) {
                return true;
            }
            if !matches!(
                manifest.mode,
                entelechy_ir::Mode::Map | entelechy_ir::Mode::Scan
            ) {
                has_non_map = true;
            }
            if manifest
                .inputs
                .iter()
                .chain(&manifest.outputs)
                .any(|port| port.shape() == entelechy_abi::Shape::Volume)
            {
                return true;
            }
            if seen_kernels.insert(version.id().clone())
                && let Some(module) = version.module()
            {
                let typ = module.typical_work();
                if typ > 6_000 {
                    return true;
                }
                sum_typical = sum_typical.saturating_add(typ);
            }
        }
        sum_typical > 20_000 || (has_native_raster && (reached.len() > 24 || has_non_map))
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
            self.finish_all_turns();
        } else {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// What preparing the scene's Kernels is doing, readable from any
    /// thread while a sync prepares them ([`entelechy_kernel::Progress`]).
    pub fn progress(&self) -> entelechy_kernel::Progress {
        self.runtime.progress()
    }

    /// The Kernels still waiting their turn to prepare or finish where no
    /// thread can start, and one more while what they prepare for has yet
    /// to land; zero elsewhere.
    pub fn preparing_left(&self) -> usize {
        let waiting = match &self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(docs) => docs.scene.waiting() || docs.kernels.preparing(),
            Backing::Bundled(b) => b.pending.is_some(),
        };
        let landing = self.runtime.prepares_in_turns() && waiting;
        self.runtime.prepare_queued() + self.finishing.len() + usize::from(landing)
    }

    /// Whether the scene last read waits for its Kernels before it
    /// replaces the running one.
    pub fn scene_waiting(&self) -> bool {
        !self.finishing.is_empty()
            || match &self.backing {
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

    /// Whether the session has no background work in flight: no `async`
    /// node running, no fused Kernel preparing, and no scene waiting to
    /// prepare or land.
    pub fn is_quiescent(&self) -> bool {
        !self.evaluator.has_running()
            && !self.evaluator.has_pending_fusions()
            && self.preparing_left() == 0
            && !self.scene_waiting()
    }

    /// Whether any node reached by a frame (`demand` `None`) or by a state
    /// step (`demand` `Some`) reads the clock (`time.read`) or a timed
    /// image sequence.
    pub fn reads_clock_for(&self, demand: Option<&BTreeSet<entelechy_graph::NodeId>>) -> bool {
        #[cfg(feature = "documents")]
        if let Backing::Documents(docs) = &self.backing
            && !docs.scene.files().sequences.is_empty()
        {
            return true;
        }
        let Some(graph) = self.graph() else {
            return false;
        };
        let handles = self.backing.handles();
        let fallbacks = graph
            .fallbacks
            .iter()
            .filter(|(name, _)| !handles.contains_key(name))
            .filter_map(|(_, source)| match source {
                entelechy_graph::Source::Node { node, .. } => Some(node.clone()),
                entelechy_graph::Source::External(_) => None,
            });
        let reached = match demand {
            None => graph.read_through(self.pins.iter().cloned().chain(fallbacks)),
            Some(extra) => {
                graph.read_state_awake(extra.iter().cloned().chain(fallbacks), &BTreeSet::new())
            }
        };
        let at = graph.index();
        reached.iter().any(|id| {
            at.get(id)
                .and_then(|idx| self.runtime.kernel(&graph.nodes[*idx].kernel))
                .is_some_and(|version| {
                    version
                        .manifest()
                        .requires
                        .contains(entelechy_abi::Capability::TimeRead)
                })
        })
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
        self.write_bundled_confirmed();
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

    pub(super) fn before_tick(&mut self, time: f64, force_wait: bool) {
        #[cfg(not(feature = "documents"))]
        let _ = time;
        #[cfg(feature = "documents")]
        self.place_by_rows();
        if force_wait || !self.evaluator.overlaps() {
            self.one_frame_in_flight();
        }
        #[cfg(feature = "documents")]
        if let Backing::Documents(docs) = &mut self.backing {
            let store = Arc::clone(docs.scene.store());
            docs.inputs
                .select_frames(store.as_ref(), self.runtime.host_mut(), time);
        }
        self.hold_progress();
        self.repin();
        self.await_written();
    }

    pub(super) fn after_tick_committed(&mut self) {
        self.failing = None;
        self.has_evaluated = true;
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
    }

    /// Evaluates the current graph under `context`.
    pub fn evaluate(&mut self, context: &Context) -> Result<Report, EvaluateError> {
        self.before_tick(context.time, false);
        self.last_context = *context;
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
        self.after_tick_committed();
        if let Some(frame) = self.runtime.end_frame() {
            self.node_clock.ended(frame, report.clocked.clone());
        }
        self.written_after(&report);
        #[cfg(feature = "documents")]
        self.follow_rows();
        self.last_report = Some(report.clone());
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
                self.ask_bundled_saves(&report.outputs);
                self.write_bundled_confirmed();
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
        self.has_evaluated = false;
        self.last_report = None;
        #[cfg(feature = "documents")]
        if let Backing::Documents(docs) = &mut self.backing {
            docs.history
                .changed(crate::Change::Reset, docs.origin.clone());
        }
    }

    /// The most recent successful evaluation report, if any.
    pub fn last_report(&self) -> Option<&Report> {
        self.last_report.as_ref()
    }

    /// Why the last evaluation failed, until a subsequent evaluation succeeds.
    pub fn failing(&self) -> Option<&str> {
        self.failing.as_deref()
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
        for node in self.written_sources() {
            self.evaluator.pin(node);
        }
        #[cfg(feature = "documents")]
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

    pub(super) fn await_written(&mut self) {
        match &self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(docs) => {
                let writes = &docs.scene.files().writes;
                self.evaluator
                    .await_outputs(writes.iter().map(|write| write.output.as_str()));
            }
            Backing::Bundled(b) => {
                self.evaluator
                    .await_outputs(b.saves.iter().map(|save| save.output.as_str()));
            }
        }
    }

    pub(super) fn written_sources(&self) -> Vec<entelechy_graph::NodeId> {
        let Some(graph) = self.graph() else {
            return Vec::new();
        };
        let resolve =
            |output_name: &str| match graph.outputs.iter().find(|(name, _)| name == output_name)? {
                (_, entelechy_graph::Source::Node { node, port }) => {
                    Some(self.producer(node, port).0)
                }
                (_, entelechy_graph::Source::External(_)) => None,
            };
        match &self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(docs) => docs
                .scene
                .files()
                .writes
                .iter()
                .filter_map(|write| resolve(&write.output))
                .collect(),
            Backing::Bundled(b) => b
                .saves
                .iter()
                .filter_map(|save| resolve(&save.output))
                .collect(),
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

    /// Resolves `target` (`node` or `node.port`, including placed subgraphs
    /// and repeated nodes) to the producing `(node, port)` in the loaded
    /// graph.
    pub fn resolve_port(&self, target: &str) -> Result<(entelechy_graph::NodeId, String), String> {
        let graph = self
            .graph()
            .ok_or_else(|| "no scene is loaded".to_owned())?;
        let whole = entelechy_graph::NodeId(target.to_owned().into());
        if let Some(found) = graph.producer(&whole, "result") {
            return Ok(found);
        }
        if let Some((node, port)) = entelechy_graph::split_port(target)
            && let Some(found) = graph.producer(&node, &port)
        {
            return Ok(found);
        }
        #[cfg(feature = "documents")]
        let hint = {
            let known: Vec<&str> = graph.nodes.iter().map(|n| n.id.0.as_str()).collect();
            let near = entelechy_language::nearest(target, known.iter().copied());
            entelechy_language::suggestion(near)
        };
        #[cfg(not(feature = "documents"))]
        let hint = String::new();
        Err(format!(
            "`{target}` is not a node or `node.port` of this scene{hint}"
        ))
    }

    /// Reads the first scalar number of element 0 of `handle` as `f64`.
    fn first_scalar(&mut self, handle: Handle, label: &str) -> Result<f64, String> {
        let data = self.field(handle).map_err(|e| format!("`{label}`: {e}"))?;
        if data.domain().count() == 0 {
            return Err(format!("`{label}` has 0 elements"));
        }
        let schema = data.schema();
        let per = schema.value_words().max(1);
        let words = data.values();
        let numbers = entelechy_kernel::Number::of_element(schema, &words[..per]);
        let Some(first) = numbers.first().copied() else {
            return Err(format!("`{label}` holds no scalar number"));
        };
        Ok(match first {
            entelechy_kernel::Number::Float(v) => f64::from(v),
            entelechy_kernel::Number::Double(v) => v,
            entelechy_kernel::Number::Int(v) => f64::from(v),
            entelechy_kernel::Number::Long(v) => v as f64,
            entelechy_kernel::Number::Bool(v) => f64::from(u8::from(v)),
        })
    }

    /// Steps causal `<~` state forward up to `steps` ticks separated by `dt`
    /// seconds without evaluating unasked viewport or audio outputs, stopping
    /// early when `until` becomes non-zero or non-finite, or once a clockless
    /// scene settles to a fixed point.
    #[allow(clippy::too_many_arguments)]
    pub fn step_state(
        &mut self,
        base_context: &Context,
        time_given: bool,
        steps: u64,
        dt: f64,
        nodes: &[entelechy_graph::NodeId],
        until: Option<&str>,
    ) -> Result<StepOutcome, StepError> {
        const MAX_STEPS: u64 = 100_000;
        if steps == 0 || steps > MAX_STEPS {
            return Err(StepError::Message(format!(
                "`steps` must be between 1 and {MAX_STEPS}, got {steps}"
            )));
        }
        if !dt.is_finite() || dt < 0.0 {
            return Err(StepError::Message(format!(
                "`dt` must be a non-negative finite number of seconds, got {dt}"
            )));
        }
        self.confirm().map_err(|error| StepError::Evaluate {
            step: 0,
            context: Box::new(self.last_context),
            error: Box::new(error),
        })?;
        self.one_frame_in_flight();

        let until_target = match until {
            Some(target) => Some((
                target,
                self.resolve_port(target).map_err(StepError::Message)?,
            )),
            None => None,
        };
        let mut requested_producers = Vec::with_capacity(nodes.len());
        for node in nodes {
            let (producer, _) = self
                .resolve_port(node.0.as_str())
                .map_err(StepError::Message)?;
            requested_producers.push(producer);
        }

        let mut temporary_pins = Vec::new();
        for candidate in requested_producers
            .iter()
            .chain(until_target.as_ref().map(|(_, (n, _))| n))
        {
            if !self.pins.contains(candidate) && !temporary_pins.contains(candidate) {
                temporary_pins.push(candidate.clone());
                self.pins.push(candidate.clone());
            }
        }
        self.repin();

        let result =
            (|| -> Result<StepOutcome, StepError> {
                let mut demand: BTreeSet<entelechy_graph::NodeId> =
                    requested_producers.into_iter().collect();
                if let Some((_, (ref node, _))) = until_target {
                    demand.insert(node.clone());
                }
                demand.extend(self.fed_sources());
                demand.extend(self.written_sources());

                let start_time = if time_given {
                    base_context.time
                } else if self.has_evaluated {
                    self.last_context.time + dt
                } else {
                    0.0
                };
                let clock_moves = dt != 0.0 && self.reads_clock_for(Some(&demand));
                let began = entelechy_kernel::clock::now();
                let mut active = 0u64;
                let mut last_report: Option<entelechy_graph::StateStep> = None;
                let mut final_time = start_time;
                let mut stopped = StepStop::Steps;
                let mut until_value = None;
                let mut ran_steps = 0u64;

                let mut held_input = entelechy_abi::InputState::default();
                held_input.pointer = base_context.input.pointer;
                held_input.sticks = base_context.input.sticks;
                held_input.pads = base_context.input.pads;
                for button in 0..entelechy_abi::BUTTONS {
                    if base_context.input.button(button) {
                        held_input.set_button(button, true);
                    }
                }
                for code in 0..256 {
                    if base_context.input.key(code) {
                        held_input.set_key(code, true);
                    }
                }

                for index in 0..steps {
                    let mut step_context = *base_context;
                    step_context.time = start_time + (index as f64) * dt;
                    if index > 0 {
                        step_context.input = held_input;
                    }
                    self.before_tick(step_context.time, false);
                    let graph = self.backing.shared_graph().cloned().ok_or_else(|| {
                        StepError::Evaluate {
                            step: index + 1,
                            context: Box::new(step_context),
                            error: Box::new(no_scene()),
                        }
                    })?;
                    let step = match self.evaluator.evaluate_state(
                        &graph,
                        &mut self.runtime,
                        self.backing.handles(),
                        &step_context,
                        self.backing.params(),
                        &demand,
                    ) {
                        Ok(step) => step,
                        Err(error) => {
                            self.tally();
                            let error = self.with_refusal(error);
                            self.failing = Some(self.told(&error));
                            return Err(StepError::Evaluate {
                                step: index + 1,
                                context: Box::new(step_context),
                                error: Box::new(error),
                            });
                        }
                    };
                    self.last_context = step_context;
                    self.after_tick_committed();
                    final_time = step_context.time;
                    ran_steps = index + 1;
                    if !step.committed.is_empty() {
                        active += 1;
                    }
                    #[cfg(feature = "documents")]
                    self.write_confirmed();
                    self.write_bundled_confirmed();
                    #[cfg(feature = "documents")]
                    self.follow_rows();

                    if let Some((label, (ref u_node, ref u_port))) = until_target {
                        let handle = self.evaluator.cached(u_node, u_port).ok_or_else(|| {
                            StepError::Message(format!(
                                "`{label}` produced no value on step {}",
                                index + 1
                            ))
                        })?;
                        let value = self
                            .first_scalar(handle, label)
                            .map_err(StepError::Message)?;
                        until_value = Some(value);
                        if !value.is_finite() {
                            stopped = StepStop::NonFinite;
                            last_report = Some(step);
                            break;
                        }
                        if value != 0.0 {
                            stopped = StepStop::Until;
                            last_report = Some(step);
                            break;
                        }
                    }

                    let settled_now = index > 0
                        && !clock_moves
                        && step.executed.is_empty()
                        && step.committed.is_empty()
                        && step.pending.is_empty()
                        && !self.has_running()
                        && !match &self.backing {
                            #[cfg(feature = "documents")]
                            Backing::Documents(docs) => docs.resized,
                            Backing::Bundled(_) => false,
                        };
                    last_report = Some(step);
                    if settled_now {
                        stopped = StepStop::Settled;
                        break;
                    }
                }

                Ok(StepOutcome {
                    steps: ran_steps,
                    active,
                    time: final_time,
                    elapsed: entelechy_kernel::clock::since(began),
                    stopped,
                    until_value,
                    last: last_report.expect("steps >= 1"),
                })
            })();

        for pin in &temporary_pins {
            self.unpin(pin);
        }
        result
    }
}

/// Why a [`Session::step_state`] stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepStop {
    /// Ran all requested `steps`.
    Steps,
    /// The `until` target's first scalar became non-zero.
    Until,
    /// The `until` target's first scalar became `NaN` or infinite.
    NonFinite,
    /// Reached a fixed point where no state, async run, or clock-reading
    /// node can change further.
    Settled,
}

impl StepStop {
    /// Wire name returned by a multi-step `evaluate`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Steps => "steps",
            Self::Until => "until",
            Self::NonFinite => "nonfinite",
            Self::Settled => "settled",
        }
    }
}

/// Outcome of stepping causal state across many ticks ([`Session::step_state`]).
#[derive(Clone, Debug)]
pub struct StepOutcome {
    /// How many ticks ran.
    pub steps: u64,
    /// How many ticks advanced at least one `<~` state port.
    pub active: u64,
    /// Causal time of the final tick that ran, in seconds.
    pub time: f64,
    /// Wall-clock duration spent stepping.
    pub elapsed: std::time::Duration,
    /// Why stepping stopped.
    pub stopped: StepStop,
    /// Value of `until`'s first scalar on the stopping tick, when `until`
    /// was given.
    pub until_value: Option<f64>,
    /// Summary of the final tick that ran.
    pub last: entelechy_graph::StateStep,
}

/// Why a [`Session::step_state`] could not complete.
#[derive(Debug)]
pub enum StepError {
    /// Invalid target, step count, or probe read failure.
    Message(String),
    /// Graph evaluation failed on `step` (1-based) under `context`.
    Evaluate {
        /// 1-based tick index that faulted.
        step: u64,
        /// Context of the faulting tick.
        context: Box<Context>,
        /// Underlying evaluation error.
        error: Box<EvaluateError>,
    },
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
