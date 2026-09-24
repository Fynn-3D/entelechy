//! The session itself: the four parts, the runtime they share, and the
//! order they sync in. Its fields are private to this module, so a part
//! handed a session still reaches only what the session hands it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use orla_formats::ppm;
use orla_graph::{EvaluateError, Evaluator, Graph, Report};
use orla_runtime::{Backend, Context, Expect, Handle, KernelId, Runtime, Value};

use crate::{Event, Limits, SessionError, Trace, inputs, kernels, observe, scene, writes};

/// A scene, the libraries it uses, and the runtime state built from them.
pub struct Session {
    scene: scene::Scene,
    kernels: kernels::Kernels,
    inputs: inputs::Inputs,
    writes: writes::Writes,
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
            writes: writes::Writes::default(),
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
        Self::runtime_and_display(backend, limits).map(|(runtime, _)| runtime)
    }

    /// The same, with the display a window presents what the backend
    /// leaves resident through, when the backend has a device to show.
    /// It is taken before the backend goes into the runtime, which knows
    /// only a `Backend` from then on.
    pub fn runtime_and_display(
        backend: &str,
        limits: Limits,
    ) -> Result<(Runtime, Option<orla_gpu::present::Display>), String> {
        let (mut runtime, display) = Self::backend_runtime(backend, limits.words)?;
        runtime.host_mut().set_work(limits.work);
        Ok((runtime, display))
    }

    fn backend_runtime(
        backend: &str,
        budget_words: usize,
    ) -> Result<(Runtime, Option<orla_gpu::present::Display>), String> {
        match backend {
            "cpu" => Ok((
                Runtime::with_backend(budget_words, Box::new(orla_cpu::CpuBackend)),
                None,
            )),
            "gpu" => {
                // A GPU walks a scan on one thread; the JIT takes the
                // scans, the GPU everything else.
                let gpu = orla_gpu::GpuBackend::new().map_err(|error| error.to_string())?;
                let display = gpu.display();
                let backend: Box<dyn Backend> = Box::new(gpu);
                let ordered: Option<Box<dyn Backend>> = orla_jit::JitBackend::new()
                    .ok()
                    .map(|jit| Box::new(jit) as Box<dyn Backend>);
                Ok((
                    Runtime::with_backends(budget_words, backend, ordered),
                    Some(display),
                ))
            }
            "jit" => {
                let jit = orla_jit::JitBackend::new()?;
                let backend: Box<dyn Backend> = Box::new(jit);
                Ok((Runtime::with_backend(budget_words, backend), None))
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
        let documents = self.scene.read_from();
        self.inputs.sync(
            self.scene.graph(),
            self.scene.files(),
            &documents,
            self.runtime.host_mut(),
            &mut events,
        );
        self.sync_placements(&mut events);
        events.append(&mut self.inputs.take_frame_events());
        events.append(&mut self.writes.take_events());
        events
    }

    /// Allows the scene's outputs to be written to files in `dir` and
    /// under it (`output name <- source to "file";`). Nothing is written
    /// anywhere else, and nothing at all until something is allowed.
    pub fn allow_writes(&mut self, dir: &Path) -> Result<(), String> {
        self.writes.allow(dir)
    }

    /// Holds what the scene's inputs read to the directories the scene
    /// names, its own, the graphs' it places and the libraries and
    /// packages it uses, and the ones [`Session::allow_reads`] allowed. For a session an
    /// agent composes and reads back from (`serve`, the Player's
    /// `--serve`): a scene a person runs reads what it names.
    pub fn restrict_reads(&mut self) {
        self.inputs.restrict_reads();
    }

    /// Allows the scene's inputs to be read from files in `dir` and under
    /// it, where reads are restricted ([`Session::restrict_reads`]).
    pub fn allow_reads(&mut self, dir: &Path) -> Result<(), String> {
        self.inputs.allow_reads(dir)
    }

    /// Grants what a command line's `--allow` said.
    pub fn allow(&mut self, allowance: &crate::Allowance) -> Result<(), String> {
        match allowance {
            crate::Allowance::FileRead(dir) => self.allow_reads(dir),
            crate::Allowance::FileWrite(dir) => self.allow_writes(dir),
        }
    }

    /// Whether `text`, as the scene document, would be taken whole: it
    /// lowers, the documents it places load, and every Kernel it defines
    /// itself compiles. Nothing is written, published or run: what a
    /// caller asks before it replaces the document, so a rejected text
    /// never reaches the disk.
    pub fn check_scene(&self, text: &str) -> Result<(), crate::Diagnostic> {
        self.scene.check(text)
    }

    /// Where a write to `path` lands, once it is known to be inside a
    /// directory [`Session::allow_writes`] allowed: what a caller acting
    /// for someone other than the person running the session, an agent
    /// over the protocol, asks before it writes anywhere.
    pub fn allowed_write(&self, path: &Path) -> Result<std::path::PathBuf, String> {
        self.writes.place(path)
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
    ) -> Result<std::sync::Arc<orla_runtime::KernelVersion>, crate::Diagnostic> {
        let candidate = self.prepare_source(source)?;
        self.runtime
            .publish(candidate, expect)
            .map_err(|e| crate::Diagnostic::new(e.to_string()))
    }

    /// Prepares `source`, a document holding one Kernel, without
    /// publishing it. The Kernel sees the declarations the source makes
    /// or `use`s, and the scene's shared ones after those, so a Kernel
    /// from a file republishes through the protocol as it is.
    pub fn prepare_source(
        &self,
        source: &str,
    ) -> Result<orla_runtime::Candidate, crate::Diagnostic> {
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
        let report = self.evaluator.evaluate(
            graph,
            &mut self.runtime,
            self.inputs.handles(),
            context,
            self.scene.params(),
        )?;
        // Committed: what the evaluation changed may now leave the session.
        self.writes
            .after(&self.scene.files().writes, &report, &mut self.runtime);
        Ok(report)
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
        self.runtime.backend().set_measured(on);
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

    // What follows observes: each hands [`observe`] the parts it reads.

    /// Writes a node's output as a file: `.field` for any Field, `.png` or
    /// `.ppm` for a grid of pixels (a `Grid<Float>` as gray), `.pgm` for a
    /// `Grid<Float>`, `.wav` at 48 kHz for rows of one or two floats, `.mid`
    /// for rows of notes. The node must have been evaluated and kept
    /// materialized (pinned).
    pub fn save(
        &mut self,
        node: &orla_graph::NodeId,
        port: &str,
        path: &Path,
    ) -> Result<(), String> {
        observe::save(
            self.scene.graph(),
            &self.evaluator,
            &mut self.runtime,
            node,
            port,
            path,
        )
    }

    /// How many elements a node's output holds: what a node whose
    /// `domain` a Kernel decided came out as.
    pub fn element_count(&self, node: &orla_graph::NodeId, port: &str) -> Result<usize, String> {
        observe::element_count(&self.evaluator, &self.runtime, node, port)
    }

    /// Runs one element of `node`'s Kernel on the interpreter and returns
    /// every binding of it: element `at` of a map, step `at[0]` of a scan.
    /// The scene is evaluated first under `context`, with the node's
    /// sources pinned so their Fields exist, and the node's invocation is
    /// what an evaluation would give it now; nothing is committed by the
    /// trace itself. Pins the trace added are removed again.
    pub fn trace(
        &mut self,
        node: &orla_graph::NodeId,
        at: [u32; 3],
        context: &Context,
    ) -> Result<(Trace, KernelId), String> {
        let (target, version) = observe::trace_target(self.scene.graph(), &self.runtime, node)?;
        // The sources the node reads must have Fields: pin them for this
        // evaluation, then let fusion have them back.
        let pinned: Vec<orla_graph::NodeId> = observe::trace_sources(&target)
            .into_iter()
            .filter(|source| !self.evaluator.pins().contains(source))
            .collect();
        for source in &pinned {
            self.evaluator.pin(source.clone());
        }
        let traced = self
            .evaluate(context)
            .map_err(|error| error.to_string())
            .and_then(|_| {
                let graph = self.scene.graph().ok_or("no scene is loaded")?;
                observe::trace_element(
                    graph,
                    &self.evaluator,
                    &mut self.runtime,
                    self.inputs.handles(),
                    self.scene.params(),
                    self.kernels.unoptimized(&target.kernel),
                    (&target, &version),
                    context,
                    at,
                )
            });
        for source in pinned {
            self.evaluator.unpin(&source);
        }
        traced.map(|trace| (trace, target.kernel.clone()))
    }

    /// One element field of a node's output: the value's type and its
    /// words. `node.port[index].field.field` is how a replay and the
    /// protocol both name it; [`parse_target`] reads that spelling.
    pub fn element_field(
        &mut self,
        node: &orla_graph::NodeId,
        port: &str,
        index: u32,
        path: &[String],
    ) -> Result<(orla_ir::Type, Vec<u32>), String> {
        observe::element_field(
            self.scene.graph(),
            &self.evaluator,
            &mut self.runtime,
            (node, port),
            index,
            path,
        )
    }

    /// Writes a cached image to a PPM file, materializing it if it was
    /// resident on the device.
    pub fn export(&mut self, image: Handle, path: &Path) -> Result<(), SessionError> {
        observe::export(&mut self.runtime, image, path)
    }

    /// Reads one pixel of a cached image, materializing it if needed.
    pub fn read_pixel(
        &mut self,
        image: Handle,
        x: u32,
        y: u32,
    ) -> Result<orla_runtime::Pixel, SessionError> {
        observe::read_pixel(&mut self.runtime, image, x, y)
    }
}
