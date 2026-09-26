//! The session itself: the four parts, the runtime they share, and the
//! order they sync in. Its fields are private to this module, so a part
//! handed a session still reaches only what the session hands it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use orla_formats::ppm;
use orla_graph::{EvaluateError, Evaluator, Graph, Report};
use orla_runtime::{Backend, Context, Expect, Handle, KernelId, Runtime, Value};

use crate::store::Store;
use crate::{Event, Limits, SessionError, Trace, inputs, kernels, observe, scene, writes};

mod time;

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
    /// What changed the world, and what was refused.
    history: crate::history::History,
    /// Where what the session is asked to do comes from now.
    origin: crate::Origin,
    /// Whether the next scene the session loads is the change the
    /// request being answered made ([`Session::claim_scene`]), rather
    /// than one found on disk.
    claimed: bool,
    /// The origin of a claimed scene still waiting for its Kernels, which
    /// it carries when it lands.
    claim_waiting: Option<crate::Origin>,
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
        Self::open_in(Arc::new(crate::Disk), scene, library, runtime)
    }

    /// Creates a session whose documents are in `store`: the disk, or
    /// the files a caller with none (a page) hands in
    /// ([`crate::Memory`]).
    pub fn open_in(
        store: Arc<dyn Store>,
        scene: &Path,
        library: Option<&Path>,
        runtime: Runtime,
    ) -> Self {
        Self {
            scene: scene::Scene::new(store, scene, library),
            kernels: kernels::Kernels::default(),
            inputs: inputs::Inputs::default(),
            writes: writes::Writes::default(),
            runtime,
            evaluator: Evaluator::new(),
            placed: BTreeMap::new(),
            window: None,
            history: crate::history::History::default(),
            origin: crate::Origin::Local,
            claimed: false,
            claim_waiting: None,
        }
    }

    /// Where the session's documents are, for what its caller reads and
    /// writes on its behalf.
    pub fn store(&self) -> &dyn Store {
        self.scene.store().as_ref()
    }

    /// The runtime, for inspection.
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Builds a runtime over a named backend ([`Session::build`]), the
    /// device, if any, waited for by blocking.
    pub fn runtime_for(backend: &str, limits: Limits) -> Result<Runtime, String> {
        Self::build(backend, limits, orla_gpu::Wait::blocking()).map(|built| built.runtime)
    }

    /// Builds a runtime over the backend named: `cpu`, `gpu`, `jit`, or
    /// `best`, the GPU where there is a device, else the JIT, else the
    /// interpreter, which is what an interface that names none runs on.
    /// The device, if any, is waited for as `wait` says: a page, which
    /// cannot block, gives its host turns until the device answers.
    pub fn build(
        backend: &str,
        limits: Limits,
        wait: orla_gpu::Wait,
    ) -> Result<crate::Built, String> {
        let mut passed_over = Vec::new();
        let one = [backend];
        let named: &[&str] = match backend {
            "best" => &["gpu", "jit", "cpu"],
            _ => &one,
        };
        for name in named {
            match Self::backend_runtime(name, limits.words, wait) {
                Ok((mut runtime, display)) => {
                    runtime.host_mut().set_work(limits.work);
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

    /// Opens `scene` in `store` on `backend` and reads it, its reads held
    /// from the start when `hold_reads` ([`Session::restrict_reads`]).
    /// With `best`, the first of the GPU, the JIT and the interpreter on
    /// which a runtime opens and the scene loads: a scene a device will
    /// not take (a shader its driver refuses) runs on the next, and each
    /// passed over is said with why. Refused, it says why the scene did
    /// not load on the last tried.
    pub fn open_loaded(
        store: Arc<dyn Store>,
        scene: &Path,
        library: Option<&Path>,
        backend: &str,
        limits: Limits,
        wait: orla_gpu::Wait,
        hold_reads: bool,
    ) -> Result<crate::Loaded, String> {
        let one = [backend];
        let named: &[&str] = match backend {
            "best" => &["gpu", "jit", "cpu"],
            _ => &one,
        };
        let mut passed_over = Vec::new();
        let mut last = String::new();
        for name in named {
            let built = match Self::build(name, limits, wait) {
                Ok(built) => built,
                Err(why) if named.len() > 1 => {
                    passed_over.push(format!("{name}: {why}"));
                    continue;
                }
                Err(why) => return Err(why),
            };
            let mut session = Self::open_in(Arc::clone(&store), scene, library, built.runtime);
            if hold_reads {
                session.restrict_reads()?;
            }
            let events = session.sync();
            if session.graph().is_some() {
                return Ok(crate::Loaded {
                    session,
                    display: built.display,
                    events,
                    passed_over,
                });
            }
            last = events
                .iter()
                .find_map(|event| match event {
                    Event::SceneRejected(error) => Some(error.to_string()),
                    Event::Rejected { path, error } => Some(format!("{}: {error}", path.display())),
                    _ => None,
                })
                .unwrap_or_else(|| format!("{} did not load", scene.display()));
            passed_over.push(format!("{name}: the scene did not load: {last}"));
        }
        Err(if last.is_empty() {
            format!("no backend opens: {}", passed_over.join("; "))
        } else {
            last
        })
    }

    fn backend_runtime(
        backend: &str,
        budget_words: usize,
        wait: orla_gpu::Wait,
    ) -> Result<(Runtime, Option<orla_gpu::present::Display>), String> {
        match backend {
            "cpu" => Ok((
                Runtime::with_backend(budget_words, Box::new(orla_cpu::CpuBackend)),
                None,
            )),
            "gpu" => {
                // A GPU walks a scan on one thread; the JIT takes the
                // scans, the GPU everything else.
                let gpu =
                    orla_gpu::GpuBackend::with_wait(wait).map_err(|error| error.to_string())?;
                let display = gpu.display();
                let backend: Box<dyn Backend> = Box::new(gpu);
                let ordered = Self::jit().ok();
                Ok((
                    Runtime::with_backends(budget_words, backend, ordered),
                    Some(display),
                ))
            }
            "jit" => Ok((Runtime::with_backend(budget_words, Self::jit()?), None)),
            other => Err(format!(
                "unknown backend `{other}`; use cpu, gpu, jit, or best"
            )),
        }
    }

    /// The JIT, where there is somewhere to run the code it makes: the
    /// machine itself, or where Orla is WebAssembly, a host that loads it.
    fn jit() -> Result<Box<dyn Backend>, String> {
        orla_jit::JitBackend::new().map(|jit| Box::new(jit) as Box<dyn Backend>)
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
        self.scene.set_param(name, value)?;
        self.history.changed(
            crate::Change::Param {
                name: name.to_owned(),
            },
            self.origin.clone(),
        );
        Ok(())
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

    /// Reconciles the runtime with the files in its store and reports what
    /// changed: replacements the worker finished publish first, then the
    /// documents are read, then the inputs, and last where each node runs.
    pub fn sync(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        let store = Arc::clone(self.scene.store());
        let inputs = &self.inputs;
        let readable = |path: &Path| inputs.readable(store.as_ref(), path);
        self.scene
            .sync(&mut self.kernels, &mut self.runtime, &mut events, &readable);
        let documents = self.scene.read_from();
        self.inputs.sync(
            self.scene.store().as_ref(),
            self.scene.graph(),
            self.scene.files(),
            &documents,
            self.runtime.host_mut(),
            &mut events,
        );
        self.sync_placements(&mut events);
        events.append(&mut self.inputs.take_frame_events());
        events.append(&mut self.writes.take_events());
        self.remember(&events);
        events
    }

    /// Allows the scene's outputs to be written to files in `dir` and
    /// under it (`output name <- source to "file";`). Nothing is written
    /// anywhere else, and nothing at all until something is allowed.
    pub fn allow_writes(&mut self, dir: &Path) -> Result<(), String> {
        self.writes.allow(self.scene.store().as_ref(), dir)
    }

    /// Holds what the scene's inputs read to the directories the scene
    /// names, its own, the graphs' it places and the libraries and
    /// packages it uses, and the ones [`Session::allow_reads`] allowed:
    /// for a session an agent composes and reads back from (`serve`, the
    /// Player's `--serve`, a program's `orla_request`), where a scene a
    /// person runs reads what it names. Decided when the session is
    /// opened, before its first sync, or refused: the directories are
    /// those of the scene as it first loads, and nothing after changes
    /// them.
    pub fn restrict_reads(&mut self) -> Result<(), String> {
        self.inputs.restrict_reads()
    }

    /// Allows the scene's inputs to be read from files in `dir` and under
    /// it, where reads are restricted ([`Session::restrict_reads`]).
    pub fn allow_reads(&mut self, dir: &Path) -> Result<(), String> {
        self.inputs.allow_reads(self.scene.store().as_ref(), dir)
    }

    /// Allows a write to the one file `path`: a file the person running
    /// the session named themselves, as a command line's `--save` or
    /// `--out` does. Nothing else under its directory is allowed by it.
    pub fn allow_write_to(&mut self, path: &Path) -> Result<(), String> {
        self.writes.allow_file(self.scene.store().as_ref(), path)
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
    /// itself compiles and requires only what the session grants. Nothing
    /// is written, published or run: what a
    /// caller asks before it replaces the document, so a rejected text
    /// never reaches the disk.
    pub fn check_scene(&self, text: &str) -> Result<(), crate::Diagnostic> {
        let granted = |module: &orla_ir::Validated| {
            self.runtime
                .granted(module)
                .map_err(|why| crate::Diagnostic::new(why.to_string()))
        };
        let readable = |path: &Path| self.allowed_read(path);
        self.scene.check(text, &readable, &granted)
    }

    /// Whether `path` may be read on the session's behalf: anywhere,
    /// unless reads are restricted ([`Session::restrict_reads`]), and then
    /// where the scene's documents are, in a package, or where allowed.
    pub fn allowed_read(&self, path: &Path) -> Result<(), String> {
        self.inputs.readable(self.scene.store().as_ref(), path)
    }

    /// Where a write to `path` lands, once it is known to be inside a
    /// directory [`Session::allow_writes`] allowed: what a caller acting
    /// for someone other than the person running the session, an agent
    /// over the protocol, asks before it writes anywhere.
    pub fn allowed_write(&self, path: &Path) -> Result<std::path::PathBuf, String> {
        self.writes.place(self.scene.store().as_ref(), path)
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
        let published = self.prepare_source(source).and_then(|candidate| {
            self.runtime
                .publish(candidate, expect)
                .map_err(|e| crate::Diagnostic::new(e.to_string()))
        });
        let origin = self.origin.clone();
        match &published {
            Ok(version) => {
                self.history
                    .published(version.id(), version.generation(), source, origin)
            }
            Err(error) => {
                // The id the source names, when it names one it can.
                let subject = orla_language::parse_document(source)
                    .ok()
                    .and_then(|document| document.kernels.first().and_then(|k| k.id.clone()))
                    .map_or_else(|| "a Kernel".to_owned(), |id| id.value);
                self.history
                    .refused(subject, error.to_string(), Some(source), origin);
            }
        }
        published
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
        let retired = self.kernels.retire_kernel(id, &mut self.runtime);
        if retired {
            self.history.retired(id, self.origin.clone());
        }
        retired
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
        self.inputs.select_frames(
            self.scene.store().as_ref(),
            self.runtime.host_mut(),
            context.time,
        );
        let graph = self.scene.graph().ok_or_else(no_scene)?;
        let report = self.evaluator.evaluate(
            graph,
            &mut self.runtime,
            self.inputs.handles(),
            context,
            self.scene.params(),
        )?;
        // Committed: what the evaluation changed may now leave the session.
        self.writes.after(
            self.scene.store().as_ref(),
            &self.scene.files().writes,
            &report,
            &mut self.runtime,
        );
        self.history.ran();
        Ok(report)
    }

    /// Evaluates only what `demand` needs; see
    /// [`orla_graph::Evaluator::evaluate_for`].
    pub fn evaluate_for(
        &mut self,
        context: &Context,
        demand: &std::collections::BTreeSet<orla_graph::NodeId>,
    ) -> Result<Report, EvaluateError> {
        self.inputs.select_frames(
            self.scene.store().as_ref(),
            self.runtime.host_mut(),
            context.time,
        );
        let graph = self.scene.graph().ok_or_else(no_scene)?;
        let report = self.evaluator.evaluate_for(
            graph,
            &mut self.runtime,
            self.inputs.handles(),
            context,
            self.scene.params(),
            demand,
        )?;
        self.history.ran();
        Ok(report)
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
            .write(graph, &mut self.runtime, node, port, words)?;
        self.history.changed(
            crate::Change::Written {
                node: node.to_string(),
                port: port.to_owned(),
            },
            self.origin.clone(),
        );
        Ok(())
    }

    /// Forgets committed state; `<~` bindings restart from their initial
    /// sources on the next evaluation.
    pub fn reset_state(&mut self) {
        self.evaluator.reset_state(&mut self.runtime);
        self.history
            .changed(crate::Change::Reset, self.origin.clone());
    }

    /// Forgets the committed state of `node.port` alone; its `<~` readers
    /// restart from their initial sources on the next evaluation, and
    /// every other port keeps its state. The port must be state.
    pub fn reset_port(
        &mut self,
        node: &orla_graph::NodeId,
        port: &str,
    ) -> Result<(), EvaluateError> {
        let graph = self.scene.graph().ok_or_else(no_scene)?;
        self.evaluator
            .reset_port(graph, &mut self.runtime, node, port)?;
        self.history.changed(
            crate::Change::ResetPort {
                node: node.to_string(),
                port: port.to_owned(),
            },
            self.origin.clone(),
        );
        Ok(())
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
    /// materialized (pinned). The file must be one the session may write
    /// ([`Session::allowed_write`]), whoever asks.
    pub fn save(
        &mut self,
        node: &orla_graph::NodeId,
        port: &str,
        path: &Path,
    ) -> Result<(), String> {
        let path = self.allowed_write(path)?;
        observe::save(
            self.scene.store().as_ref(),
            self.scene.graph(),
            &self.evaluator,
            &mut self.runtime,
            node,
            port,
            &path,
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
    /// resident on the device. The file must be one the session may
    /// write ([`Session::allowed_write`]), whoever asks.
    pub fn export(&mut self, image: Handle, path: &Path) -> Result<(), SessionError> {
        let path = self.allowed_write(path).map_err(SessionError::Observe)?;
        observe::export(self.scene.store().as_ref(), &mut self.runtime, image, &path)
    }

    /// A Field's data as the Host holds it, brought back from the device
    /// first when one holds it: what a front reads to answer or show,
    /// asking the session rather than reaching into its runtime.
    pub fn field(&mut self, handle: Handle) -> Result<&orla_runtime::FieldData, SessionError> {
        self.runtime
            .materialize(handle)
            .map_err(|e| SessionError::Observe(e.to_string()))?;
        self.runtime.host().data(handle).map_err(SessionError::Host)
    }

    /// The output of `report` a scene shows: the first that is a picture,
    /// a grid of pixels one deep. Every interface that shows a scene
    /// shows this one.
    pub fn picture_of(&self, report: &orla_graph::Report) -> Option<Handle> {
        let host = self.runtime.host();
        report
            .outputs
            .iter()
            .map(|(_, handle)| *handle)
            .find(|handle| {
                host.schema(*handle).is_ok_and(|s| s.is_pixel())
                    && host
                        .domain(*handle)
                        .is_ok_and(|d| d.shape() == orla_runtime::Shape::Grid && d.depth() == 1)
            })
    }

    /// The output of `report` a scene sounds, and its channels: the first
    /// that is rows of one float (mono) or two (stereo). Every interface
    /// that plays a scene's sound plays this one.
    pub fn sound_of(&self, report: &orla_graph::Report) -> Option<(Handle, usize)> {
        let host = self.runtime.host();
        report.outputs.iter().find_map(|(_, handle)| {
            let rows = host
                .domain(*handle)
                .is_ok_and(|d| d.shape() == orla_runtime::Shape::Rows);
            let schema = host.schema(*handle).ok()?;
            let floats = schema
                .scalars()
                .iter()
                .all(|c| *c == orla_runtime::Scalar::Float);
            let channels = schema.value_words();
            (rows && floats && (1..=2).contains(&channels)).then_some((*handle, channels))
        })
    }

    /// A Field of pixels as an image, brought back from the device first
    /// when one holds it.
    pub fn picture(&mut self, handle: Handle) -> Result<orla_runtime::Image, SessionError> {
        self.runtime
            .image(handle)
            .map_err(|e| SessionError::Observe(e.to_string()))
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
