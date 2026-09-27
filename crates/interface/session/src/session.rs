//! The session: the four parts, their shared runtime, and their sync
//! order. Its fields are private to this module; a part reaches only what
//! the session hands it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use orla_formats::ppm;
use orla_graph::{EvaluateError, Evaluator, Graph, Report};
use orla_kernel::{Backend, Context, Expect, Handle, KernelId, Runtime, Value};

use crate::store::Store;
use crate::{
    Event, Limits, Publishing, SessionError, Trace, inputs, kernels, observe, scene, writes,
};

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
    /// The headless window: size in pixels and scale factor.
    window: Option<([f32; 2], f32)>,
    /// What changed the world, and what was refused.
    history: crate::history::History,
    /// Where the current request comes from.
    origin: crate::Origin,
    /// The request whose change the scene being read is
    /// ([`Session::claim_scene`]).
    claim: Claim,
    /// The names asked to stay materialized ([`Session::pin`]), resolved
    /// against the running graph before each evaluation.
    pins: Vec<orla_graph::NodeId>,
}

/// A request's claim on the scene it wrote, from asked to landed. A scene
/// read in its place (a document changed on disk) ends it.
#[derive(Default)]
pub(crate) enum Claim {
    /// No request's.
    #[default]
    None,
    /// The next scene read is the change of the request being answered.
    Asked,
    /// The request's scene waits for its Kernels.
    Waiting(crate::Origin),
    /// The request's scene landed on a later sync, with that sync's
    /// events ([`Session::take_landing`]).
    Landed(Vec<Event>),
}

/// The error an evaluation returns when no scene has loaded.
fn no_scene() -> EvaluateError {
    EvaluateError::Graph(orla_graph::GraphError::UnknownOutput(
        orla_graph::Source::External("<no scene>".into()),
    ))
}

impl Session {
    /// Creates a CPU-backed session; [`Session::sync`] loads files.
    /// `library` is an extra directory or file of Kernels.
    pub fn open(scene: &Path, library: Option<&Path>, budget_words: usize) -> Self {
        Self::open_with(
            scene,
            library,
            Runtime::with_backend(budget_words, Box::new(orla_cpu::CpuBackend)),
        )
    }

    /// Creates a session over a runtime the caller built.
    pub fn open_with(scene: &Path, library: Option<&Path>, runtime: Runtime) -> Self {
        Self::open_in(Arc::new(crate::Disk::default()), scene, library, runtime)
    }

    /// Creates a session whose documents are in `store`: the disk, or
    /// files handed in ([`crate::Memory`]).
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
            claim: Claim::None,
            pins: Vec::new(),
        }
    }

    /// Where the session's documents are. Callers acting for others go
    /// through [`Session::read_file`], [`Session::write_file`] and
    /// [`Session::write_scene`], which check allowances first.
    pub(crate) fn store(&self) -> &dyn Store {
        self.scene.store().as_ref()
    }

    /// The runtime, for inspection.
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Builds a runtime over a named backend ([`Session::build`]),
    /// blocking on the device.
    pub fn runtime_for(backend: &str, limits: Limits) -> Result<Runtime, String> {
        Self::build(backend, limits, orla_gpu::Wait::blocking()).map(|built| built.runtime)
    }

    /// Builds a runtime over `cpu`, `gpu`, `jit`, or `best` (the first of
    /// GPU, JIT, interpreter that opens). The device is waited for as
    /// `wait` says.
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

    /// Opens `scene` in `store` on `backend` and reads it, reads
    /// restricted from the start when `hold_reads`
    /// ([`Session::restrict_reads`]). With `best`, the first of GPU, JIT,
    /// interpreter on which the runtime opens and the scene loads; each
    /// passed over is listed with why. The error is why the scene failed
    /// on the last tried.
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
                // The JIT runs scans; the GPU runs everything else.
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

    /// The JIT, where the machine or a WebAssembly host runs its code.
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

    /// The type of the param `name` the scene declares, as written; an
    /// undeclared one is refused, saying where it lives instead. `None`
    /// before a scene loads, which holds any name.
    pub fn param_type(&self, name: &str) -> Result<Option<String>, String> {
        self.scene
            .declared_param(name)
            .map(|param| param.map(|param| param.ty.to_string()))
    }

    /// Overrides a graph parameter for later evaluations, in the shape of
    /// its declared type (`[[1, 0], [0, 1]]` for a `Mat2`); another layout
    /// is refused. Unknown names are kept for a later declaration.
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

    /// Reconciles the runtime with its store and reports what changed, in
    /// order: finished replacements, documents, inputs, placements.
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
    /// under it (`output name <- source to "file";`). Outputs are written
    /// only where allowed.
    pub fn allow_writes(&mut self, dir: &Path) -> Result<(), String> {
        self.writes.allow(self.scene.store().as_ref(), dir)
    }

    /// Restricts input reads to the scene's own directories (its own, its
    /// placed graphs', its libraries' and packages') and those
    /// [`Session::allow_reads`] allowed; for sessions an agent drives
    /// (`serve`, `--serve`, `orla_request`). Refused after the first
    /// sync; the directories are fixed as the scene first loads.
    pub fn restrict_reads(&mut self) -> Result<(), String> {
        self.inputs.restrict_reads()
    }

    /// Allows the scene's inputs to be read from files in `dir` and under
    /// it, where reads are restricted ([`Session::restrict_reads`]).
    pub fn allow_reads(&mut self, dir: &Path) -> Result<(), String> {
        self.inputs.allow_reads(self.scene.store().as_ref(), dir)
    }

    /// Allows a write to the one file `path`, named by the person running
    /// the session (`--save`, `--out`). Its directory stays closed.
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
    /// lowers, its placed documents load, and each Kernel it defines
    /// compiles within the session's grants. Side-effect free; callers ask
    /// before replacing the document.
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

    /// Where a write to `path` lands, if inside an allowed directory
    /// ([`Session::allow_writes`]). Callers acting for an agent ask this
    /// before every write.
    pub fn allowed_write(&self, path: &Path) -> Result<std::path::PathBuf, String> {
        self.writes.place(self.scene.store().as_ref(), path)
    }

    /// The scene document's text as it stands in the store.
    pub fn scene_text(&self) -> std::io::Result<String> {
        self.store().read_text(self.scene.root())
    }

    /// Reads `path` where the session may read ([`Session::allowed_read`]).
    pub fn read_file(&self, path: &Path) -> Result<String, String> {
        self.allowed_read(path)?;
        self.store()
            .read_text(path)
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Writes `bytes` to `path`, whole, where the session may write
    /// ([`Session::allowed_write`]); answers where they landed.
    pub fn write_file(&self, path: &Path, bytes: &[u8]) -> Result<std::path::PathBuf, String> {
        let target = self.allowed_write(path)?;
        self.store()
            .write(&target, bytes)
            .map_err(|e| format!("{}: {e}", target.display()))?;
        Ok(target)
    }

    /// Replaces the scene document with `text`, whole, once
    /// [`Session::check_scene`] accepts it; a refusal leaves the document
    /// as it is and is recorded as an attempt
    /// ([`Session::attempts_since`]). The outer error is the store's; the
    /// inner, the refusal.
    pub fn write_scene(
        &mut self,
        text: &str,
    ) -> Result<Result<(), crate::Diagnostic>, SessionError> {
        if let Err(why) = self.check_scene(text) {
            let subject = self.scene.root().display().to_string();
            self.history
                .refused(subject, why.to_string(), Some(text), self.origin.clone());
            return Ok(Err(why));
        }
        let root = self.scene.root();
        self.store()
            .write(root, text.as_bytes())
            .map_err(|e| SessionError::Io(root.to_path_buf(), e))?;
        Ok(Ok(()))
    }

    /// Where each node with a published Kernel runs, and why when off the
    /// main backend.
    pub fn placements(&self) -> Vec<(String, orla_kernel::Placement)> {
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

    /// Reports nodes off the main backend, once per change.
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

    /// Prepares replacements of Kernels and fused groups on a worker
    /// thread; each publishes on the first sync after it is ready.
    pub fn set_background(&mut self, enabled: bool) {
        self.runtime.set_background(enabled);
    }

    /// Whether replacements prepare on a worker thread.
    pub fn background(&self) -> bool {
        self.runtime.background()
    }

    /// Whether the scene last read waits for its Kernels before it
    /// replaces the running one.
    pub fn scene_waiting(&self) -> bool {
        self.scene.waiting()
    }

    /// The events of the later sync on which a request's claimed scene
    /// landed, taken once. Empty when it landed on the request's own sync.
    pub fn take_landing(&mut self) -> Option<Vec<Event>> {
        match std::mem::take(&mut self.claim) {
            Claim::Landed(events) => Some(events),
            other => {
                self.claim = other;
                None
            }
        }
    }

    /// Starts [`Session::publish_source`] without waiting: compiles (or
    /// refuses) now, prepares where replacements prepare
    /// ([`Session::set_background`]). [`Session::publishing`] finishes it.
    pub fn publish_later(
        &mut self,
        source: &str,
        expect: Expect,
    ) -> Result<Publishing, crate::Diagnostic> {
        match self.scene.compile_source(source, &self.kernels) {
            Ok(module) => Ok(Publishing {
                ticket: self.runtime.prepare_later(module, source.to_owned()),
                expect,
                source: source.to_owned(),
            }),
            Err(error) => {
                self.refused_source(source, &error);
                Err(error)
            }
        }
    }

    /// Finishes [`Session::publish_later`] once prepared: published under
    /// its expectation, or refused; `None` while preparing.
    pub fn publishing(
        &mut self,
        publishing: &Publishing,
    ) -> Option<Result<std::sync::Arc<orla_kernel::KernelVersion>, crate::Diagnostic>> {
        let prepared = self.runtime.poll(publishing.ticket)?;
        let published = prepared
            .map_err(|e| crate::Diagnostic::new(e.to_string()))
            .and_then(|candidate| {
                self.runtime
                    .publish(candidate, publishing.expect)
                    .map_err(|e| crate::Diagnostic::new(e.to_string()))
            });
        match &published {
            Ok(version) => self.history.published(
                version.id(),
                version.generation(),
                &publishing.source,
                self.origin.clone(),
            ),
            Err(error) => self.refused_source(&publishing.source, error),
        }
        Some(published)
    }

    /// Records a refused Kernel source, by its id when it parses.
    fn refused_source(&mut self, source: &str, error: &crate::Diagnostic) {
        let subject = orla_language::parse_document(source)
            .ok()
            .and_then(|document| document.kernels.first().and_then(|k| k.id.clone()))
            .map_or_else(|| "a Kernel".to_owned(), |id| id.value);
        self.history.refused(
            subject,
            error.to_string(),
            Some(source),
            self.origin.clone(),
        );
    }

    /// Publishes a source holding exactly one Kernel, outside any
    /// document. Documents own ids: one that later defines the id
    /// republishes it.
    pub fn publish_source(
        &mut self,
        source: &str,
        expect: Expect,
    ) -> Result<std::sync::Arc<orla_kernel::KernelVersion>, crate::Diagnostic> {
        let publishing = self.publish_later(source, expect)?;
        loop {
            if let Some(published) = self.publishing(&publishing) {
                return published;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// Prepares `source`, one Kernel, without publishing it. It sees its
    /// own and `use`d declarations, then the scene's shared ones.
    pub fn prepare_source(
        &self,
        source: &str,
    ) -> Result<orla_kernel::Candidate, crate::Diagnostic> {
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

    /// The Field a scene input holds now.
    pub fn external(&self, name: &str) -> Option<Handle> {
        self.inputs.handles().get(name).copied()
    }

    /// The node and port that produce what `node.port` names: itself, or
    /// for a placed graph's output or a scene output, the node that gives
    /// it ([`orla_graph::Graph::producer`]).
    pub fn producer(&self, node: &orla_graph::NodeId, port: &str) -> (orla_graph::NodeId, String) {
        self.scene
            .graph()
            .and_then(|graph| graph.producer(node, port))
            .unwrap_or_else(|| (node.clone(), port.to_owned()))
    }

    /// The Field last evaluated for what `node.port` names, if any.
    pub fn cached(&self, node: &orla_graph::NodeId, port: &str) -> Option<Handle> {
        let (node, port) = self.producer(node, port);
        self.evaluator.cached(&node, &port)
    }

    /// The node `node` runs fused into, if any
    /// ([`orla_graph::Evaluator::fused_into`]).
    pub fn fused_into(&self, node: &orla_graph::NodeId) -> Option<orla_graph::NodeId> {
        let (node, _) = self.producer(node, "result");
        self.evaluator.fused_into(&node).cloned()
    }

    /// The declarations `kernel` shares with its document's other Kernels,
    /// as that document writes them (its `struct`s, `type`s and `fn`s),
    /// and the document; none for a Kernel no document publishes.
    pub fn shared_with(&self, kernel: &KernelId) -> Option<(PathBuf, String)> {
        use orla_language::StatementKind;
        let path = self.kernels.owner(kernel)?.to_path_buf();
        let text = self.store().read_text(&path).ok()?;
        let outline = orla_language::outline(&text).ok()?;
        let shared: Vec<&str> = outline
            .statements
            .iter()
            .filter(|s| {
                matches!(
                    s.kind,
                    StatementKind::Struct | StatementKind::Type | StatementKind::Fn
                )
            })
            .filter_map(|s| text.get(s.range.clone()))
            .collect();
        Some((path, shared.join("\n\n")))
    }

    /// The state ports (`node.port`) whose initial source is a node running
    /// `kernel`: they keep their committed state when it changes, and start
    /// from it again only when reset.
    pub fn started_by(&self, kernel: &KernelId) -> Vec<String> {
        let Some(graph) = self.scene.graph() else {
            return Vec::new();
        };
        let runs = |node: &orla_graph::NodeId| {
            graph
                .nodes
                .iter()
                .any(|n| &n.id == node && &n.kernel == kernel)
        };
        let mut out = Vec::new();
        for node in &graph.nodes {
            for (port, binding) in &node.inputs {
                if let orla_graph::Binding::Previous {
                    initial: orla_graph::Source::Node { node: from, .. },
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

    /// Why `node`'s `port` has no value to read: not a node, placed as
    /// several, folded into its reader, or not run since it was placed.
    pub fn why_no_value(&self, node: &orla_graph::NodeId, port: &str) -> String {
        let (node, port) = self.producer(node, port);
        observe::why_no_value(self.scene.graph(), &self.evaluator, &node, &port)
    }

    /// Waits for every `async` node's run and takes it in; see
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
        self.repin();
        let graph = self.scene.graph().ok_or_else(no_scene)?;
        let report = self.evaluator.evaluate(
            graph,
            &mut self.runtime,
            self.inputs.handles(),
            context,
            self.scene.params(),
        )?;
        // Committed: changed outputs may now be written.
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
        self.repin();
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
        self.repin();
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

    /// Forgets the committed state of the state port `node.port` alone;
    /// its `<~` readers restart on the next evaluation.
    pub fn reset_port(
        &mut self,
        node: &orla_graph::NodeId,
        port: &str,
    ) -> Result<(), EvaluateError> {
        self.repin();
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
    /// none: size in pixels and scale factor.
    pub fn set_window(&mut self, window: Option<([f32; 2], f32)>) {
        self.window = window;
    }

    /// The window a headless session reports, if one was set.
    pub fn window(&self) -> Option<([f32; 2], f32)> {
        self.window
    }

    /// Makes the backend wait for its device after every node, so a
    /// profile measures the work, not the submission.
    pub fn set_measured(&mut self, on: bool) {
        self.runtime.backend().set_measured(on);
    }

    /// The fixed cost of measuring one node; see
    /// [`orla_kernel::Backend::measurement_floor`]. Known after a
    /// measured run.
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

    /// Keeps what `node` names materialized so it can be read: a node, or
    /// a placed graph by the nodes that give its outputs. The name holds
    /// across a scene read again.
    pub fn pin(&mut self, node: orla_graph::NodeId) {
        if !self.pins.contains(&node) {
            self.pins.push(node);
        }
        self.repin();
    }

    /// What would execute right now, in order, with fusion applied.
    pub fn plan(&mut self) -> Result<Vec<orla_graph::PlanEntry>, EvaluateError> {
        self.repin();
        let graph = self.scene.graph().ok_or_else(no_scene)?;
        self.evaluator.plan_summary(
            graph,
            &mut self.runtime,
            self.inputs.handles(),
            self.scene.params(),
        )
    }

    /// Lets fusion have what `node` names back.
    pub fn unpin(&mut self, node: &orla_graph::NodeId) {
        self.pins.retain(|pinned| pinned != node);
        for inner in self.pinned_by(node) {
            self.evaluator.unpin(&inner);
        }
    }

    /// The nodes `name` pins in the running graph: itself, or a placed
    /// graph's output nodes.
    fn pinned_by(&self, name: &orla_graph::NodeId) -> Vec<orla_graph::NodeId> {
        let placed = self
            .scene
            .graph()
            .and_then(|graph| graph.placed.iter().find(|(id, _)| *id == name.0));
        match placed {
            Some((_, outputs)) => outputs
                .iter()
                .filter_map(|(_, source)| match source {
                    orla_graph::Source::Node { node, .. } => Some(node.clone()),
                    orla_graph::Source::External(_) => None,
                })
                .collect(),
            None => vec![name.clone()],
        }
    }

    /// Pins what every pinned name names in the running graph.
    fn repin(&mut self) {
        for name in self.pins.clone() {
            for node in self.pinned_by(&name) {
                self.evaluator.pin(node);
            }
        }
    }

    // What follows observes: each hands [`observe`] the parts it reads.

    /// Writes an evaluated, pinned node's output as a file: `.field` for
    /// any Field, `.png`/`.ppm` for pixels (a `Grid<Float>` as gray),
    /// `.pgm` for a `Grid<Float>`, `.wav` at 48 kHz for rows of one or two
    /// floats, `.mid` for rows of notes. Only where
    /// [`Session::allowed_write`] allows, whoever asks.
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

    /// How many elements a node's output holds.
    pub fn element_count(&self, node: &orla_graph::NodeId, port: &str) -> Result<usize, String> {
        let (node, port) = self.producer(node, port);
        observe::element_count(&self.evaluator, &self.runtime, &node, &port)
    }

    /// Runs one element of `node`'s Kernel on the interpreter and returns
    /// every binding: element `at` of a map, step `at[0]` of a scan. The
    /// scene is evaluated first under `context` with the node's sources
    /// pinned; the trace commits nothing and removes the pins it added.
    pub fn trace(
        &mut self,
        node: &orla_graph::NodeId,
        at: [u32; 3],
        context: &Context,
    ) -> Result<(Trace, KernelId), String> {
        let (target, version) = observe::trace_target(self.scene.graph(), &self.runtime, node)?;
        // Pin the node's sources for this evaluation so they have Fields.
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

    /// One element field of a node's output: its type and words. Named
    /// `node.port[index].field.field` ([`crate::observe::parse_target`]).
    pub fn element_field(
        &mut self,
        node: &orla_graph::NodeId,
        port: &str,
        index: u32,
        path: &[String],
    ) -> Result<(orla_ir::Type, Vec<u32>), String> {
        let (node, port) = self.producer(node, port);
        observe::element_field(
            self.scene.graph(),
            &self.evaluator,
            &mut self.runtime,
            (&node, &port),
            index,
            path,
        )
    }

    /// Writes a Field to a file in the format its extension names, as
    /// [`Session::save`] does. Only where [`Session::allowed_write`]
    /// allows, whoever asks.
    pub fn export(&mut self, handle: Handle, path: &Path) -> Result<(), SessionError> {
        let path = self.allowed_write(path).map_err(SessionError::Observe)?;
        observe::write(
            self.scene.store().as_ref(),
            &mut self.runtime,
            handle,
            &path,
        )
        .map_err(|why| match why {
            observe::Written::Io(path, e) => SessionError::Io(path, e),
            observe::Written::Observe(why) => SessionError::Observe(why),
            other => SessionError::Unwritable(path.clone(), other.to_string()),
        })
    }

    /// A Field's data as the Host holds it, brought back from the device
    /// first.
    pub fn field(&mut self, handle: Handle) -> Result<&orla_kernel::FieldData, SessionError> {
        self.runtime
            .materialize(handle)
            .map_err(|e| SessionError::Observe(e.to_string()))?;
        self.runtime.host().data(handle).map_err(SessionError::Host)
    }

    /// The output of `report` every interface shows: the first picture
    /// ([`orla_kernel::Schema::is_picture`]).
    pub fn picture_of(&self, report: &orla_graph::Report) -> Option<Handle> {
        report
            .outputs
            .iter()
            .map(|(_, handle)| *handle)
            .find(|handle| self.is_picture(*handle))
    }

    /// Whether the Field `handle` is a picture
    /// ([`orla_kernel::Schema::is_picture`]).
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
                    let known = graph.outputs.iter().map(|(output, _)| output.as_str());
                    format!(
                        "no output named `{name}`{}",
                        orla_language::did_you_mean(name, known)
                    )
                })?,
            None => graph.primary_output().ok_or("the graph has no output")?,
        };
        match source {
            orla_graph::Source::Node { node, port } => self
                .cached(node, port)
                .ok_or_else(|| "evaluate before reading an output".to_owned()),
            orla_graph::Source::External(_) => Err(format!("output `{name}` is an external image")),
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

    /// The output of `report` every interface plays, and its channels:
    /// the first rows of one float (mono) or two (stereo).
    pub fn sound_of(&self, report: &orla_graph::Report) -> Option<(Handle, usize)> {
        let host = self.runtime.host();
        report.outputs.iter().find_map(|(_, handle)| {
            let rows = host
                .domain(*handle)
                .is_ok_and(|d| d.shape() == orla_kernel::Shape::Rows);
            let schema = host.schema(*handle).ok()?;
            let floats = schema
                .scalars()
                .iter()
                .all(|c| *c == orla_kernel::Scalar::Float);
            let channels = schema.value_words();
            (rows && floats && (1..=2).contains(&channels)).then_some((*handle, channels))
        })
    }

    /// A Field of pixels as an image, brought back from the device first.
    pub fn picture(&mut self, handle: Handle) -> Result<orla_kernel::Image, SessionError> {
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
    ) -> Result<orla_kernel::Pixel, SessionError> {
        observe::read_pixel(&mut self.runtime, image, x, y)
    }
}
