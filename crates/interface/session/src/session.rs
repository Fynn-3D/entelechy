//! The session: the four parts, their shared runtime, and their sync
//! order. Its fields are private to this module; a part reaches only what
//! the session hands it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use entelechy_formats::ppm;
use entelechy_graph::{EvaluateError, Evaluator, Graph, Report};
use entelechy_kernel::{Backend, Context, Expect, Handle, KernelId, Runtime, Value};

use crate::store::Store;
use crate::{
    Event, Limits, Publishing, SessionError, SourcePlace, Traced, inputs, kernels, observe, scene,
    writes,
};

mod time;

/// A scene, the libraries it uses, and the runtime state built from them.
pub struct Session {
    scene: scene::Scene,
    kernels: kernels::Kernels,
    inputs: inputs::Inputs,
    /// What the session may read; its store reads only there.
    reads: Arc<std::sync::Mutex<crate::reads::Reads>>,
    writes: writes::Writes,
    runtime: Runtime,
    evaluator: Evaluator,
    /// Where each node was last said to run, so a change is said once.
    placed: BTreeMap<String, Option<String>>,
    /// The Kernels last said to be published and used by no node, by
    /// document ([`Event::Idle`]).
    idle: BTreeMap<PathBuf, Vec<KernelId>>,
    /// What was last said of nodes whose default domain leaves elements
    /// unrun ([`Event::Unrun`]), by node.
    unrun: BTreeMap<String, String>,
    /// The Kernels the graph's nodes use, by the version last seen
    /// ([`Session::finish_used`]).
    used: BTreeMap<entelechy_kernel::KernelId, entelechy_kernel::Generation>,
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
    pins: Vec<entelechy_graph::NodeId>,
    /// Each clocked frame's nodes' runs, until the device tells the frame.
    node_clock: crate::clocked::NodeClock,
    /// The externals bound to files by the caller's overrides, as bound.
    bound: BTreeMap<String, PathBuf>,
    /// The folder last chosen ([`Session::choose_folder`]) and what it was
    /// allowed.
    chosen: Option<(PathBuf, Vec<crate::Allowance>)>,
    /// The clipboard: the session's own, or the system's given.
    board: crate::clipboard::Board,
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
    /// The request's scene waits for its Kernels, with the events of the
    /// syncs since it was asked.
    Waiting(crate::Origin, Vec<Event>),
    /// The request's scene landed on a later sync, with that sync's
    /// events ([`Session::take_landing`]).
    Landed(Vec<Event>),
}

/// The error an evaluation returns when no scene has loaded.
fn no_scene() -> EvaluateError {
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

impl Session {
    /// Creates a CPU-backed session; [`Session::sync`] loads files.
    /// `library` is an extra directory or file of Kernels.
    pub fn open(scene: &Path, library: Option<&Path>, budget_words: usize) -> Self {
        Self::open_with(
            scene,
            library,
            Runtime::with_backend(budget_words, Box::new(entelechy_cpu::CpuBackend)),
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
        let reads = Arc::new(std::sync::Mutex::new(crate::reads::Reads::default()));
        let store: Arc<dyn Store> = Arc::new(crate::reads::Held::new(store, Arc::clone(&reads)));
        Self {
            scene: scene::Scene::new(store, scene, library),
            kernels: kernels::Kernels::default(),
            inputs: inputs::Inputs::default(),
            reads,
            writes: writes::Writes::default(),
            runtime,
            evaluator: Evaluator::new(),
            placed: BTreeMap::new(),
            idle: BTreeMap::new(),
            unrun: BTreeMap::new(),
            used: BTreeMap::new(),
            window: None,
            history: crate::history::History::default(),
            origin: crate::Origin::Local,
            claim: Claim::None,
            pins: Vec::new(),
            node_clock: crate::clocked::NodeClock::default(),
            bound: BTreeMap::new(),
            chosen: None,
            board: crate::clipboard::Board::default(),
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
        Self::build(backend, limits, entelechy_gpu::Wait::blocking()).map(|built| built.runtime)
    }

    /// Builds a runtime over `cpu`, `gpu`, `jit`, or `best` (the first of
    /// GPU, JIT, interpreter that opens). The device is waited for as
    /// `wait` says.
    pub fn build(
        backend: &str,
        limits: Limits,
        wait: entelechy_gpu::Wait,
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
        wait: entelechy_gpu::Wait,
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
        wait: entelechy_gpu::Wait,
    ) -> Result<(Runtime, Option<entelechy_gpu::present::Display>), String> {
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
            "cpu" => Ok((Box::new(entelechy_cpu::CpuBackend), None, None)),
            "gpu" => {
                // The JIT runs scans; the GPU runs everything else.
                let gpu = entelechy_gpu::GpuBackend::with_wait(wait)
                    .map_err(|error| error.to_string())?;
                let display = gpu.display();
                Ok((Box::new(gpu), Self::jit().ok(), Some(display)))
            }
            "jit" => Ok((Self::jit()?, None, None)),
            other => Err(format!(
                "unknown backend `{other}`; use cpu, gpu, jit, or best"
            )),
        }
    }

    /// The JIT, where the machine or a WebAssembly host runs its code.
    fn jit() -> Result<Box<dyn Backend>, String> {
        entelechy_jit::JitBackend::new().map(|jit| Box::new(jit) as Box<dyn Backend>)
    }

    /// The current graph, if the scene loaded.
    pub fn graph(&self) -> Option<&Graph> {
        self.scene.graph()
    }

    /// The scene's font nodes, every level's: each the externals its
    /// glyphs, outlines and metrics are in, and where its codes come from.
    pub fn fonts(&self) -> &[entelechy_scene::FontNode] {
        &self.scene.files().fonts
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

    /// Names the run the next evaluations are, from its first frame: each
    /// frame takes from the same frame of the last run of that name what
    /// nothing it reads has changed, and is remembered for the next
    /// ([`entelechy_graph::Evaluator::recall_as`]). `None` remembers nothing.
    pub fn recall_as(&mut self, name: Option<String>) {
        self.evaluator.recall_as(name);
    }

    /// The bytes remembered runs hold.
    pub fn recalled_bytes(&self) -> usize {
        self.evaluator.recalled_bytes()
    }

    /// Starts the world over as the scene opened it, the Kernels and the
    /// documents kept: every `async` run landed, every pin dropped, every
    /// port's committed state forgotten, every parameter back at its
    /// default, and the documents read again. What a replay needs before
    /// it plays on a session another played on: the session's Kernels,
    /// prepared and compiled, and the outputs no state feeds, stay as
    /// they were.
    pub fn start_over(&mut self) -> Vec<Event> {
        let _ = self.settle();
        for pin in std::mem::take(&mut self.pins) {
            for inner in self.pinned_by(&pin) {
                self.evaluator.unpin(&inner);
            }
        }
        self.reset_state();
        let overridden: Vec<String> = self.scene.params().keys().cloned().collect();
        self.scene.clear_params();
        for name in overridden {
            self.history
                .changed(crate::Change::Param { name }, self.origin.clone());
        }
        self.sync()
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
        // Every document walks the files it reaches; a path is looked at
        // once a sync.
        store.sync_begins();
        let reads = Arc::clone(&self.reads);
        let readable = |path: &Path| crate::reads::lock(&reads).check(store.as_ref(), path);
        self.scene
            .sync(&mut self.kernels, &mut self.runtime, &mut events, &readable);
        // Externals the caller's overrides bind to files, bound when new.
        for (name, path) in self.scene.bound().to_vec() {
            if self.bound.get(&name) == Some(&path) {
                continue;
            }
            match self.bind_external(&name, &path) {
                Ok(()) => {
                    events.push(Event::Loaded {
                        name: name.clone(),
                        path: path.clone(),
                        detail: "bound by the caller".into(),
                    });
                    self.bound.insert(name, path);
                }
                Err(error) => events.push(Event::Rejected {
                    path: path.clone(),
                    error: crate::Diagnostic::new(error.to_string()),
                }),
            }
        }
        // The first-loaded scene's own documents and input files, worked
        // out only while reads wait for them.
        let read_from: Vec<std::path::PathBuf> = if crate::reads::lock(&reads).wants_directories() {
            let files = self.scene.files();
            self.scene
                .read_from()
                .into_iter()
                .chain(
                    files
                        .assets
                        .iter()
                        .filter_map(|asset| asset.path.parent().map(Path::to_path_buf)),
                )
                .chain(files.sequences.iter().map(|(_, dir, ..)| dir.clone()))
                .chain(files.checks.iter().map(|(_, path)| path.clone()))
                .chain(files.files.iter().map(|(_, dir)| dir.clone()))
                .chain(files.outlines.iter().map(|(_, path)| path.clone()))
                .chain(
                    files
                        .fonts
                        .iter()
                        .filter_map(|font| font.path.parent().map(Path::to_path_buf)),
                )
                .collect()
        } else {
            Vec::new()
        };
        crate::reads::lock(&reads).settle(store.as_ref(), &read_from);
        let tables = crate::inputs::table_elements(self.scene.graph(), &self.runtime);
        let declared = crate::inputs::Declared {
            graph: self.scene.graph(),
            files: self.scene.files(),
            tables: &tables,
        };
        self.inputs.sync(
            store.as_ref(),
            &declared,
            &readable,
            self.runtime.host_mut(),
            &mut events,
        );
        let clipboards = self.scene.files().clipboards.clone();
        self.board.sync(
            &clipboards,
            &mut self.inputs,
            self.runtime.host_mut(),
            &mut events,
        );
        self.finish_used();
        self.sync_placements(&mut events);
        self.sync_shape(&mut events);
        events.append(&mut self.inputs.take_frame_events());
        events.append(&mut self.writes.take_events());
        self.remember(&events);
        store.sync_ends();
        events
    }

    /// Allows the scene's outputs to be written to files in `dir` and
    /// under it (`output name <- source to "file";`). Outputs are written
    /// only where allowed.
    pub fn allow_writes(&mut self, dir: &Path) -> Result<(), String> {
        self.writes.allow(self.scene.store().as_ref(), dir)
    }

    /// Allows an output `to files` to remove and rename files in `dir` and
    /// under it; writing them still needs [`Session::allow_writes`].
    pub fn allow_deletes(&mut self, dir: &Path) -> Result<(), String> {
        self.writes.allow_deletes(self.scene.store().as_ref(), dir)
    }

    /// Restricts input reads to the scene's own directories (its own, its
    /// placed graphs', its libraries' and packages') and those
    /// [`Session::allow_reads`] allowed; for sessions an agent drives
    /// (`serve`, `--serve`, `entelechy_request`). Refused after the first
    /// sync; the directories are fixed as the scene first loads.
    pub fn restrict_reads(&mut self) -> Result<(), String> {
        crate::reads::lock(&self.reads).restrict()
    }

    /// Allows the scene's inputs to be read from files in `dir` and under
    /// it, where reads are restricted ([`Session::restrict_reads`]).
    pub fn allow_reads(&mut self, dir: &Path) -> Result<(), String> {
        crate::reads::lock(&self.reads).allow(self.store(), dir)
    }

    /// Allows a write to the one file `path`, named by the person running
    /// the session (`--save`, `--out`). Its directory stays closed.
    pub fn allow_write_to(&mut self, path: &Path) -> Result<(), String> {
        self.writes.allow_file(self.scene.store().as_ref(), path)
    }

    /// Places the scene's inputs and outputs where `overrides` say: the
    /// running scene's at once (read again at the next sync), every scene
    /// read after. A name the scene does not read or write from a file is
    /// refused, and nothing changes.
    pub fn place(&mut self, overrides: crate::overrides::Overrides) -> Result<(), String> {
        self.scene.place(overrides)
    }

    /// Where the caller places the scene's inputs and outputs.
    pub fn overrides(&self) -> &crate::overrides::Overrides {
        self.scene.overrides()
    }

    /// A folder the person chose for the scene: each of `names`, an input
    /// or an output that reads or writes a directory (`files`, `checks`,
    /// `outline`, `to files`), placed there, and the folder given each of
    /// `grants` (`file.read`, `file.write`, `file.delete`), reads always.
    /// A folder chosen before gives back what it was given. Refused,
    /// nothing changes.
    pub fn choose_folder(
        &mut self,
        names: &[String],
        dir: &Path,
        grants: &[String],
    ) -> Result<(), String> {
        if self.scene.graph().is_none() {
            return Err("no scene runs to take a folder".into());
        }
        if !self.store().is_dir(dir) {
            return Err(format!("{} is not a directory", dir.display()));
        }
        let mut overrides = self.scene.overrides().clone();
        for name in names {
            match crate::overrides::kind_of(self.scene.written(), name) {
                Some(crate::overrides::Kind::Directory) => {}
                Some(crate::overrides::Kind::File) => {
                    return Err(format!(
                        "`{name}` reads or writes a file, not a directory: a folder cannot be chosen for it"
                    ));
                }
                None => {
                    return Err(format!(
                        "the scene reads and writes nothing named `{name}` from files"
                    ));
                }
            }
            if self
                .scene
                .written()
                .writes
                .iter()
                .any(|w| &w.output == name)
            {
                overrides.outputs.insert(name.clone(), dir.to_path_buf());
            } else {
                overrides.inputs.insert(name.clone(), dir.to_path_buf());
            }
        }
        let allowances = std::iter::once("file.read")
            .chain(grants.iter().map(String::as_str))
            .map(|grant| crate::parse_allow(&format!("{grant}={}", dir.display())))
            .collect::<Result<Vec<_>, _>>()?;
        self.scene.place(overrides)?;
        if let Some((before, _)) = self.chosen.take() {
            self.writes.revoke(self.scene.store().as_ref(), &before);
            crate::reads::lock(&self.reads).revoke(self.scene.store().as_ref(), &before);
        }
        for allowance in &allowances {
            self.allow(allowance)?;
        }
        self.chosen = Some((dir.to_path_buf(), allowances));
        Ok(())
    }

    /// The folder last chosen ([`Session::choose_folder`]).
    pub fn chosen_folder(&self) -> Option<&Path> {
        self.chosen.as_ref().map(|(dir, _)| dir.as_path())
    }

    /// The system's clipboard for the scene's `= clipboard` inputs and `to
    /// clipboard` outputs, in the session's own one's stead; reading and
    /// writing it then need `--allow clipboard.read` and
    /// `--allow clipboard.write`.
    pub fn give_clipboard(&mut self, device: Box<dyn crate::Clipboard>) {
        self.board.give(device);
    }

    /// Puts `text` on the clipboard, as a person copying elsewhere would,
    /// and has the scene's `= clipboard` inputs hold it at once.
    pub fn place_clipboard(&mut self, text: &str) -> Result<(), String> {
        self.board.place(text)?;
        let clipboards = self.scene.files().clipboards.clone();
        let mut events = Vec::new();
        self.board.sync(
            &clipboards,
            &mut self.inputs,
            self.runtime.host_mut(),
            &mut events,
        );
        self.remember(&events);
        Ok(())
    }

    /// Grants what a command line's `--allow` said.
    pub fn allow(&mut self, allowance: &crate::Allowance) -> Result<(), String> {
        match allowance {
            crate::Allowance::FileRead(dir) => self.allow_reads(dir),
            crate::Allowance::FileWrite(dir) => self.allow_writes(dir),
            crate::Allowance::FileDelete(dir) => self.allow_deletes(dir),
            crate::Allowance::ClipboardRead => {
                self.board.allow_read();
                Ok(())
            }
            crate::Allowance::ClipboardWrite => {
                self.board.allow_write();
                Ok(())
            }
        }
    }

    /// Whether `text`, as the scene document, would be taken whole: it
    /// lowers, its placed documents load, and each Kernel it defines
    /// compiles within the session's grants. Side-effect free; callers ask
    /// before replacing the document.
    pub fn check_scene(&self, text: &str) -> Result<(), crate::Diagnostic> {
        let granted = |module: &entelechy_ir::Validated| {
            self.runtime
                .granted(module)
                .map_err(|why| crate::Diagnostic::new(why.to_string()))
        };
        let readable = |path: &Path| self.allowed_read(path);
        self.scene.check(text, &readable, &granted)
    }

    /// Whether `path` may be read on the session's behalf: anywhere,
    /// unless reads are restricted ([`Session::restrict_reads`]), and then
    /// where the scene's documents are or use, in a package directory, or
    /// where allowed.
    pub fn allowed_read(&self, path: &Path) -> Result<(), String> {
        crate::reads::lock(&self.reads).check(self.store(), path)
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

    /// Where `path`, spelled from the scene's directory or whole, lands as
    /// a document a compose may write: a `.ent`, `.esn` or `.replay` file
    /// under the scene's directory, judged where it lands after its links, and in
    /// no package directory. A package is the person's, kept as found.
    pub fn composable(&self, path: &Path) -> Result<PathBuf, String> {
        let store = self.store();
        let dir = self.scene.root().parent().unwrap_or_else(|| Path::new("."));
        let home = store.canonical(dir).unwrap_or_else(|_| store.whole(dir));
        let wanted = store.whole(&dir.join(path));
        if !matches!(
            wanted.extension().and_then(|e| e.to_str()),
            Some("ent" | "esn" | "replay")
        ) {
            return Err(format!(
                "`{}` is not a document: a compose writes `.ent`, `.esn` and `.replay` files",
                path.display()
            ));
        }
        // Where it lands: its nearest directory that exists, after links,
        // and the rest of its path, made new.
        let mut existing = wanted.clone();
        let mut rest = Vec::new();
        while store.kind(&existing).is_none() {
            let (Some(up), Some(name)) = (existing.parent(), existing.file_name()) else {
                break;
            };
            rest.push(name.to_owned());
            existing = up.to_path_buf();
        }
        let mut landed = store.canonical(&existing).unwrap_or(existing);
        for name in rest.iter().rev() {
            landed.push(name);
        }
        if !landed.starts_with(&home) {
            return Err(format!(
                "`{}` is outside the scene's directory `{}`: a compose writes the scene's documents there and nowhere else",
                path.display(),
                home.display()
            ));
        }
        for packages in store.packages() {
            let packages = store
                .canonical(packages)
                .unwrap_or_else(|_| store.whole(packages));
            if landed.starts_with(&packages) {
                return Err(format!(
                    "`{}` is in the package directory `{}`: a package is kept as it was found, not composed",
                    path.display(),
                    packages.display()
                ));
            }
        }
        Ok(landed)
    }

    /// The document a compose names by `path` ([`Session::composable`]):
    /// where it lands, and its text, none when it is new.
    pub fn composed_document(&self, path: &Path) -> Result<(PathBuf, Option<String>), String> {
        let landed = self.composable(path)?;
        if self.store().kind(&landed).is_none() {
            return Ok((landed, None));
        }
        self.allowed_read(&landed)?;
        let text = self
            .store()
            .read_text(&landed)
            .map_err(|e| format!("{}: {e}", landed.display()))?;
        Ok((landed, Some(text)))
    }

    /// Replaces documents of the scene, each whole, all or none: each where
    /// a compose may write ([`Session::composable`]), each the scene uses or
    /// places once all are written (a new one is used by the same change)
    /// or a replay that parses and names a scene there,
    /// and the scene with all of them checked as `entelechy check` checks
    /// it, their Kernels compiled and granted, before any reaches the
    /// store. The scene is written last. A refusal names the document at
    /// fault, leaves every document as it was, and is recorded as an
    /// attempt. The outer error is the store's.
    pub fn write_documents(
        &mut self,
        changes: &[(PathBuf, String)],
    ) -> Result<Result<(), crate::Refusal>, SessionError> {
        let checked = self.check_documents(changes);
        let landed = match checked {
            Ok(landed) => landed,
            Err(refusal) => {
                let text = changes
                    .iter()
                    .find(|(path, _)| {
                        self.composable(path).ok().as_deref() == Some(refusal.path.as_path())
                    })
                    .map(|(_, text)| text.as_str());
                self.history.refused(
                    refusal.path.display().to_string(),
                    refusal.diagnostic.to_string(),
                    text,
                    self.origin.clone(),
                );
                return Ok(Err(refusal));
            }
        };
        let root = self
            .store()
            .canonical(self.scene.root())
            .unwrap_or_else(|_| self.scene.root().to_path_buf());
        // The documents the scene reaches first, so the scene, read
        // again, finds them all.
        let (scene, others): (Vec<_>, Vec<_>) =
            landed.into_iter().partition(|(path, _)| *path == root);
        for (path, text) in others.iter().chain(&scene) {
            self.store()
                .write(path, text.as_bytes())
                .map_err(|e| SessionError::Io(path.clone(), e))?;
        }
        // What was written is remembered, each document's text, so any of
        // them can be read back and composed again.
        let written: Vec<(PathBuf, String)> = others.into_iter().chain(scene).collect();
        self.history.documents(&written, self.origin.clone());
        Ok(Ok(()))
    }

    fn check_documents(
        &self,
        changes: &[(PathBuf, String)],
    ) -> Result<Vec<(PathBuf, String)>, crate::Refusal> {
        let store = self.store();
        let refused = |path: PathBuf, message: String| crate::Refusal {
            path,
            diagnostic: crate::Diagnostic::new(message),
        };
        let mut landed: Vec<(PathBuf, String)> = Vec::new();
        for (path, text) in changes {
            let at = self
                .composable(path)
                .map_err(|why| refused(path.clone(), why))?;
            if landed.iter().any(|(seen, _)| *seen == at) {
                return Err(refused(
                    at,
                    "this document is named twice in one compose".into(),
                ));
            }
            landed.push((at, text.clone()));
        }
        let root = store
            .canonical(self.scene.root())
            .unwrap_or_else(|_| self.scene.root().to_path_buf());
        let overlay = crate::store::Overlay::new(store, &landed);
        let root_text = match landed.iter().find(|(path, _)| *path == root) {
            Some((_, text)) => text.clone(),
            None => (&overlay as &dyn Store)
                .read_text(&root)
                .map_err(|e| refused(root.clone(), e.to_string()))?,
        };
        // A replay written parses, and the scene and Kernels it names are
        // there once all is written; it names a scene, no scene names it.
        let is_replay = |path: &Path| path.extension().is_some_and(|e| e == "replay");
        for (path, text) in landed.iter().filter(|(path, _)| is_replay(path)) {
            let base = path.parent().unwrap_or_else(|| Path::new("."));
            let replay = crate::replay::parse(text, base).map_err(|error| crate::Refusal {
                path: path.clone(),
                diagnostic: crate::Diagnostic {
                    message: error.message.clone(),
                    at: Some((error.line, 1)),
                },
            })?;
            let scene_written = overlay.kind(&overlay.whole(&replay.scene))
                == Some(crate::store::Kind::File)
                && replay.scene.extension().is_some_and(|e| e == "esn");
            if !scene_written {
                return Err(refused(
                    path.clone(),
                    format!(
                        "the scene it names, `{}`, is not a scene document there",
                        replay.scene.display()
                    ),
                ));
            }
            if let Some(kernels) = &replay.kernels
                && overlay.kind(&overlay.whole(kernels)).is_none()
            {
                return Err(refused(
                    path.clone(),
                    format!(
                        "the Kernels it names, `{}`, are not there",
                        kernels.display()
                    ),
                ));
            }
        }
        // Each document written is one the scene, as written, uses or places;
        // a reach that cannot be worked out is the whole check's to say.
        let reached = reached_documents(&overlay, &root, &root_text);
        for (path, _) in &landed {
            if let Some(reached) = &reached
                && *path != root
                && !is_replay(path)
                && !reached.contains(path)
            {
                return Err(refused(
                    path.clone(),
                    format!(
                        "`{}` is written but the scene neither uses nor places it: a document a compose adds is used in the same compose",
                        path.display()
                    ),
                ));
            }
        }
        // The scene checked whole, as `entelechy check` checks it.
        let checked = crate::check::check_document(&overlay, &root, false);
        if let Some((_, line)) = checked.lines.iter().find(|(problem, _)| *problem) {
            return Err(placed(&overlay, line, &root, &landed));
        }
        let readable = |path: &Path| self.allowed_read(path);
        let granted = |module: &entelechy_ir::Validated| {
            self.runtime
                .granted(module)
                .map_err(|why| crate::Diagnostic::new(why.to_string()))
        };
        self.scene
            .check_in(&overlay, &root_text, &readable, &granted)
            .map_err(|diagnostic| crate::Refusal {
                path: root.clone(),
                diagnostic,
            })?;
        // A written Kernel document's Kernels are granted as the scene's are.
        for (path, _) in landed.iter().filter(|(path, _)| *path != root) {
            if path.extension().and_then(|e| e.to_str()) != Some("ent") {
                continue;
            }
            let mut document = (&overlay as &dyn Store)
                .document(path)
                .map(|document| (*document).clone())
                .map_err(|why| refused(path.clone(), why))?;
            let dir = path.parent().unwrap_or_else(|| Path::new("."));
            let (structs, types, functions) = crate::sources::declarations_for(
                &overlay,
                dir,
                &document.uses,
                entelechy_language::Own::of(&document),
            )
            .map_err(|why| refused(path.clone(), why))?;
            document.resolve((&structs, &types, &functions));
            for kernel in &document.kernels {
                let id = crate::sources::named_id(kernel).map_err(|diagnostic| crate::Refusal {
                    path: path.clone(),
                    diagnostic,
                })?;
                let module =
                    entelechy_compiler::compile_kernel(kernel, Some(&id)).map_err(|error| {
                        crate::Refusal {
                            path: path.clone(),
                            diagnostic: crate::Diagnostic::from(&error),
                        }
                    })?;
                granted(&module).map_err(|diagnostic| crate::Refusal {
                    path: path.clone(),
                    diagnostic,
                })?;
            }
        }
        Ok(landed)
    }

    /// Where each node with a published Kernel runs, and why when off the
    /// main backend.
    pub fn placements(&self) -> Vec<(String, entelechy_kernel::Placement)> {
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

    /// Finishes on the main backend, on every core at once, the Kernels
    /// the graph's nodes use for the first time
    /// ([`entelechy_kernel::Preparer::finish`]), so the frame after waits for
    /// none of them in turn. A replacement was finished in preparing it
    /// where the version it replaced ran there ([`Runtime::prepare_later`]),
    /// and a Kernel no node uses is left for a run that needs it.
    fn finish_used(&mut self) {
        let Some(graph) = self.scene.graph() else {
            return;
        };
        let mut now = BTreeMap::new();
        let mut first = Vec::new();
        for node in &graph.nodes {
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
        if !first.is_empty() {
            let preparer = self.runtime.preparer();
            crate::spread::map(&first, |version| preparer.finish(version));
        }
        self.used = now;
    }

    /// Says what shapes the scene's Kernels as a device's compiler meets
    /// them: each Kernel this sync published that is large
    /// ([`Event::Large`]), and, once per change, the Kernels a document of
    /// the scene publishes and no node uses ([`Event::Idle`]); a
    /// package's are its verbs, used or not.
    fn sync_shape(&mut self, events: &mut Vec<Event>) {
        let large: Vec<Event> = events
            .iter()
            .filter_map(|event| match event {
                Event::Published { path, id, .. } => {
                    let version = self.runtime.kernel(id)?;
                    let why = crate::large(version.module()?)?;
                    Some(Event::Large {
                        path: path.clone(),
                        id: id.clone(),
                        why,
                    })
                }
                _ => None,
            })
            .collect();
        events.extend(large);
        let Some(graph) = self.scene.graph() else {
            return;
        };
        let published = self.kernels.documents().into_iter().map(|path| {
            let ids = self.kernels.ids_of(&path);
            (path, ids)
        });
        let idle: BTreeMap<PathBuf, Vec<KernelId>> =
            crate::check::idle(self.scene.store().as_ref(), graph, published)
                .into_iter()
                .collect();
        for (path, ids) in &idle {
            if self.idle.get(path) != Some(ids) {
                events.push(Event::Idle {
                    path: path.clone(),
                    ids: ids.clone(),
                });
            }
        }
        self.idle = idle;
        let unrun: BTreeMap<String, String> = graph
            .nodes
            .iter()
            .filter_map(|node| {
                let version = self.runtime.kernel(&node.kernel)?;
                let module = version.module()?;
                let why = crate::check::defaulted_domain(
                    node,
                    &module.manifest,
                    crate::check::first_counted_against(module),
                )?;
                Some((node.id.to_string(), why))
            })
            .collect();
        for (node, why) in &unrun {
            if self.unrun.get(node) != Some(why) {
                events.push(Event::Unrun {
                    node: node.clone(),
                    why: why.clone(),
                });
            }
        }
        self.unrun = unrun;
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
    /// replaces the running one, for the documents it uses, compiling off
    /// the frames, before it is read again, or for a Kernel they changed,
    /// still preparing, before it runs.
    pub fn scene_waiting(&self) -> bool {
        self.scene.waiting() || self.kernels.preparing()
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
    ) -> Option<Result<std::sync::Arc<entelechy_kernel::KernelVersion>, crate::Diagnostic>> {
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
        let subject = entelechy_language::parse_document(source)
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
    ) -> Result<std::sync::Arc<entelechy_kernel::KernelVersion>, crate::Diagnostic> {
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
    ) -> Result<entelechy_kernel::Candidate, crate::Diagnostic> {
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
    /// it ([`entelechy_graph::Graph::producer`]).
    pub fn producer(
        &self,
        node: &entelechy_graph::NodeId,
        port: &str,
    ) -> (entelechy_graph::NodeId, String) {
        self.scene
            .graph()
            .and_then(|graph| graph.producer(node, port))
            .unwrap_or_else(|| (node.clone(), port.to_owned()))
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

    /// The declarations `kernel` shares with its document's other Kernels,
    /// as that document writes them (its `struct`s, `type`s and `fn`s),
    /// and the document; none for a Kernel no document publishes.
    pub fn shared_with(&self, kernel: &KernelId) -> Option<(PathBuf, String)> {
        use entelechy_language::StatementKind;
        let path = self.kernels.owner(kernel)?.to_path_buf();
        let text = self.store().read_text(&path).ok()?;
        let outline = entelechy_language::outline(&text).ok()?;
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

    /// Why `node`'s `port` has no value to read: not a node, placed as
    /// several, folded into its reader, or not run since it was placed.
    pub fn why_no_value(&self, node: &entelechy_graph::NodeId, port: &str) -> String {
        let (node, port) = self.producer(node, port);
        observe::why_no_value(self.scene.graph(), &self.evaluator, &node, &port)
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

    /// Waits for the device to finish the last evaluation's work before
    /// the next is submitted: one frame is in flight at most, so a caller
    /// evaluating faster than the device (an agent, a replay) holds one
    /// frame's device memory, not every frame it has queued.
    fn one_frame_in_flight(&self) {
        if self.runtime.has_device() {
            self.runtime.settle_device();
        }
    }

    /// Whether later evaluations overlap the device's work on the frame
    /// before; see [`entelechy_graph::Evaluator::set_overlap`]. Turning it off
    /// confirms the frame in flight.
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
    /// [`entelechy_kernel::Runtime::set_clocked`]. A clocked frame waits for
    /// nothing more than an unclocked one.
    pub fn set_clocked(&mut self, on: bool) {
        self.runtime.set_clocked(on);
        if !on {
            self.node_clock.clear();
        }
    }

    /// The device's busy time for the latest clocked frame it has done; see
    /// [`entelechy_kernel::Runtime::frame_device_time`].
    pub fn frame_device_time(&self) -> Option<std::time::Duration> {
        self.runtime.frame_device_time()
    }

    /// The latest clocked frame the device has done, node by node: what
    /// each node's runs kept it busy, by its own clock. Nothing waits; a
    /// frame is told some frames after it ran.
    pub fn device_times(&self) -> Option<crate::DeviceTimes> {
        self.node_clock.times(&self.runtime.clocked_frame()?)
    }

    /// How long callers have been blocked on the device's work in all;
    /// see [`entelechy_kernel::Runtime::device_waited`].
    pub fn device_waited(&self) -> std::time::Duration {
        self.runtime.device_waited()
    }

    /// Confirms the frame in flight, if any; see
    /// [`entelechy_graph::Evaluator::confirm`].
    pub fn confirm(&mut self) -> Result<(), EvaluateError> {
        let confirmed = self.evaluator.confirm(&mut self.runtime);
        self.tally();
        confirmed
    }

    /// Takes the frames the evaluator took back off the world's count.
    fn tally(&mut self) {
        self.history.taken_back(self.evaluator.take_taken_back());
    }

    /// Evaluates the current graph under `context`. Unchanged nodes are
    /// reused; state commits only when the whole evaluation succeeds.
    /// Overlapping, the frame before is confirmed here, and this one when
    /// it changed an output written to a file.
    pub fn evaluate(&mut self, context: &Context) -> Result<Report, EvaluateError> {
        // Overlapping bounds the frames in flight itself.
        if !self.evaluator.overlaps() {
            self.one_frame_in_flight();
        }
        self.inputs.select_frames(
            self.scene.store().as_ref(),
            self.runtime.host_mut(),
            context.time,
        );
        self.repin();
        let graph = self.scene.graph().ok_or_else(no_scene)?;
        let evaluated = self.evaluator.evaluate(
            graph,
            &mut self.runtime,
            self.inputs.handles(),
            context,
            self.scene.params(),
        );
        let report = match evaluated {
            Ok(report) => report,
            Err(error) => {
                self.tally();
                return Err(self.with_refusal(error));
            }
        };
        self.history.ran();
        self.feed_fonts();
        // The frame's runs are all given: a clocked frame ends here, its
        // nodes' runs kept until the device tells it.
        if let Some(frame) = self.runtime.end_frame() {
            self.node_clock.ended(frame, report.clocked.clone());
        }
        // Committed: changed outputs may now be written, once confirmed.
        if self.writes.due(&self.scene.files().writes, &report) {
            self.confirm()?;
        }
        self.writes.after(
            self.scene.store().as_ref(),
            &self.scene.files().writes,
            &report,
            &mut self.runtime,
        );
        self.board
            .after(&self.scene.files().writes, &report, &mut self.runtime);
        Ok(report)
    }

    /// A Kernel not published, told why its document refused it.
    fn with_refusal(&self, error: EvaluateError) -> EvaluateError {
        match error {
            EvaluateError::UnknownKernel {
                node,
                kernel,
                near,
                refused: None,
            } => {
                let refused = self.kernels.refused(&kernel).map(str::to_owned);
                EvaluateError::UnknownKernel {
                    node,
                    kernel,
                    near,
                    refused,
                }
            }
            other => other,
        }
    }

    /// Evaluates only what `demand` needs; see
    /// [`entelechy_graph::Evaluator::evaluate_for`].
    pub fn evaluate_for(
        &mut self,
        context: &Context,
        demand: &std::collections::BTreeSet<entelechy_graph::NodeId>,
    ) -> Result<Report, EvaluateError> {
        self.one_frame_in_flight();
        self.inputs.select_frames(
            self.scene.store().as_ref(),
            self.runtime.host_mut(),
            context.time,
        );
        self.repin();
        // The font nodes' characters are read after every frame, whatever
        // it asks for.
        let mut demand = demand.clone();
        demand.extend(self.font_codes());
        let demand = &demand;
        let graph = self.scene.graph().ok_or_else(no_scene)?;
        let report = self
            .evaluator
            .evaluate_for(
                graph,
                &mut self.runtime,
                self.inputs.handles(),
                context,
                self.scene.params(),
                demand,
            )
            .map_err(|error| self.with_refusal(error))?;
        self.history.ran();
        self.feed_fonts();
        Ok(report)
    }

    /// Replaces the committed state of `node.port` with `words`; see
    /// [`entelechy_graph::Evaluator::write`].
    pub fn write(
        &mut self,
        node: &entelechy_graph::NodeId,
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
        node: &entelechy_graph::NodeId,
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
    /// [`entelechy_kernel::Backend::measurement_floor`]. Known after a
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
    pub fn pin(&mut self, node: entelechy_graph::NodeId) {
        if !self.pins.contains(&node) {
            self.pins.push(node);
        }
        self.repin();
    }

    /// What would execute right now, in order, with fusion applied.
    pub fn plan(&mut self) -> Result<Vec<entelechy_graph::PlanEntry>, EvaluateError> {
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
    pub fn unpin(&mut self, node: &entelechy_graph::NodeId) {
        self.pins.retain(|pinned| pinned != node);
        for inner in self.pinned_by(node) {
            self.evaluator.unpin(&inner);
        }
    }

    /// The nodes `name` pins in the running graph: itself, or a placed
    /// graph's output nodes; a name placed `times = N` its last time's.
    fn pinned_by(&self, name: &entelechy_graph::NodeId) -> Vec<entelechy_graph::NodeId> {
        let graph = self.scene.graph();
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

    /// Pins what every pinned name names in the running graph.
    fn repin(&mut self) {
        for name in self.pins.clone() {
            for node in self.pinned_by(&name) {
                self.evaluator.pin(node);
            }
        }
        // What a font node reads its characters from stays materialized.
        for node in self.font_codes() {
            self.evaluator.pin(node);
        }
    }

    /// The nodes that produce the font nodes' characters.
    fn font_codes(&self) -> Vec<entelechy_graph::NodeId> {
        self.scene
            .files()
            .fonts
            .iter()
            .filter_map(|font| match &font.codes {
                entelechy_graph::Source::Node { node, port } => Some(self.producer(node, port).0),
                entelechy_graph::Source::External(_) => None,
            })
            .collect()
    }

    /// Gives each font node the characters its `codes` held when last
    /// evaluated, reading the Field only when it is another than it last
    /// read; the node's glyphs follow for the next evaluation.
    fn feed_fonts(&mut self) {
        let fonts = self.scene.files().fonts.clone();
        for font in &fonts {
            let handle = match &font.codes {
                entelechy_graph::Source::Node { node, port } => self.cached(node, port),
                entelechy_graph::Source::External(name) => self.inputs.handles().get(name).copied(),
            };
            let Some(handle) = handle else {
                continue;
            };
            if self.inputs.font_saw(font, handle) {
                continue;
            }
            let codes = self
                .runtime
                .materialize(handle)
                .map_err(|e| e.to_string())
                .and_then(|()| self.runtime.host().data(handle).map_err(|e| e.to_string()))
                .and_then(|data| {
                    if data.schema().scalars() == [entelechy_kernel::Scalar::Int] {
                        Ok(data.storage().iter().map(|&w| w as i32).collect())
                    } else {
                        Err("its `codes` are a `Rows<Int>` of characters".to_owned())
                    }
                });
            self.inputs
                .give_codes(font, handle, codes, self.runtime.host_mut());
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
        node: &entelechy_graph::NodeId,
        port: &str,
        path: &Path,
    ) -> Result<(), String> {
        let path = self.allowed_write(path)?;
        self.confirm().map_err(|error| error.to_string())?;
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
    pub fn element_count(
        &self,
        node: &entelechy_graph::NodeId,
        port: &str,
    ) -> Result<usize, String> {
        let (node, port) = self.producer(node, port);
        observe::element_count(&self.evaluator, &self.runtime, &node, &port)
    }

    /// Runs one element of `node`'s Kernel on the interpreter and returns
    /// every binding: element `at` of a map, step `at[0]` of a scan. What
    /// the node reads is evaluated first under `context` with its sources
    /// pinned, the node itself not run, so a node that fails is traced up
    /// to its failure. The trace commits nothing and removes the pins it
    /// added.
    pub fn trace(
        &mut self,
        node: &entelechy_graph::NodeId,
        at: [u32; 3],
        context: &Context,
    ) -> Result<Traced, String> {
        let (target, version) = observe::trace_target(self.scene.graph(), &self.runtime, node)?;
        // Pin the node's sources for this evaluation so they have Fields.
        let pinned: Vec<entelechy_graph::NodeId> = observe::trace_sources(&target)
            .into_iter()
            .filter(|source| !self.evaluator.pins().contains(source))
            .collect();
        for source in &pinned {
            self.evaluator.pin(source.clone());
        }
        // A trial: the node's sources are computed from the state last
        // committed, and nothing is committed, written or recorded.
        self.inputs.select_frames(
            self.scene.store().as_ref(),
            self.runtime.host_mut(),
            context.time,
        );
        self.repin();
        let readers = std::collections::BTreeSet::from([target.id.clone()]);
        let tried = match self.scene.graph() {
            Some(graph) => self.evaluator.evaluate_trial_reading(
                graph,
                &mut self.runtime,
                self.inputs.handles(),
                context,
                self.scene.params(),
                &readers,
            ),
            None => Err(no_scene()),
        };
        let written = self.kernels.unoptimized(&target.kernel);
        let traced = tried.map_err(|error| error.to_string()).and_then(|_| {
            let graph = self.scene.graph().ok_or("no scene is loaded")?;
            observe::trace_element(
                graph,
                &self.evaluator,
                &mut self.runtime,
                self.inputs.handles(),
                self.scene.params(),
                written.as_ref().map(|written| &written.module),
                (&target, &version),
                context,
                at,
            )
        });
        for source in pinned {
            self.evaluator.unpin(&source);
        }
        traced.map(|trace| Traced {
            place: trace
                .fault
                .as_ref()
                .zip(written.as_ref())
                .and_then(|(fault, written)| written.place_of(fault)),
            trace,
            kernel: target.kernel.clone(),
        })
    }

    /// Where in its source an evaluation's failure at one element is
    /// written, found by tracing that element again; `None` for any other
    /// failure, or when the trace does not fail there (a group of fused
    /// nodes failing in one it absorbed) or no document owns the Kernel.
    pub fn failure_place(
        &mut self,
        error: &EvaluateError,
        context: &Context,
    ) -> Option<SourcePlace> {
        let EvaluateError::Execution { node, error } = error else {
            return None;
        };
        let (x, y) = match error {
            entelechy_kernel::ExecutionError::Arithmetic { x, y, .. }
            | entelechy_kernel::ExecutionError::Exhausted { x, y, .. } => (*x, *y),
            _ => return None,
        };
        // A volume's row counts its layers through: past the first layer
        // the element is outside `[x, y, 0]` and nothing is placed.
        self.trace(node, [x, y, 0], context).ok()?.place
    }

    /// One element field of a node's output: its type and words. Named
    /// `node.port[index].field.field` ([`crate::observe::parse_target`]).
    pub fn element_field(
        &mut self,
        node: &entelechy_graph::NodeId,
        port: &str,
        index: u32,
        path: &[String],
    ) -> Result<(entelechy_ir::Type, Vec<u32>), String> {
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

    /// The name of each number in an element of what `node.port` names,
    /// in the order a read gives them: `hp`, `pos.x`; one number alone
    /// is `""`.
    pub fn element_names(
        &self,
        node: &entelechy_graph::NodeId,
        port: &str,
    ) -> Result<Vec<String>, String> {
        let (node, port) = self.producer(node, port);
        observe::element_names(self.scene.graph(), &self.runtime, (&node, &port))
    }

    /// The fields of the struct `path` names in `node.port`'s element, in
    /// order; none for a value that is not a struct.
    pub fn element_fields(
        &mut self,
        node: &entelechy_graph::NodeId,
        port: &str,
        path: &[String],
    ) -> Result<Vec<String>, String> {
        let (node, port) = self.producer(node, port);
        observe::element_fields(self.scene.graph(), &self.runtime, (&node, &port), path)
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
    /// ([`entelechy_kernel::Schema::is_picture`]).
    pub fn picture_of(&self, report: &entelechy_graph::Report) -> Option<Handle> {
        report
            .outputs
            .iter()
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
                    let known = graph.outputs.iter().map(|(output, _)| output.as_str());
                    format!(
                        "no output named `{name}`{}",
                        entelechy_language::did_you_mean(name, known)
                    )
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

    /// The output of `report` every interface plays, and its channels:
    /// the first rows of one float (mono) or two (stereo).
    pub fn sound_of(&self, report: &entelechy_graph::Report) -> Option<(Handle, usize)> {
        let host = self.runtime.host();
        report.outputs.iter().find_map(|(_, handle)| {
            let rows = host
                .domain(*handle)
                .is_ok_and(|d| d.shape() == entelechy_kernel::Shape::Rows);
            let schema = host.schema(*handle).ok()?;
            let floats = schema
                .scalars()
                .iter()
                .all(|c| *c == entelechy_kernel::Scalar::Float);
            let channels = schema.value_words();
            (rows && floats && (1..=2).contains(&channels)).then_some((*handle, channels))
        })
    }

    /// A Field of pixels as an image, brought back from the device first.
    pub fn picture(&mut self, handle: Handle) -> Result<entelechy_kernel::Image, SessionError> {
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
    ) -> Result<entelechy_kernel::Pixel, SessionError> {
        observe::read_pixel(&mut self.runtime, image, x, y)
    }
}

/// The documents the scene at `root`, its text `text`, uses or places,
/// and what those use and place in turn, each by its one name; none when
/// a document does not parse or a use does not resolve.
fn reached_documents(
    store: &dyn Store,
    root: &Path,
    text: &str,
) -> Option<std::collections::BTreeSet<PathBuf>> {
    let mut reached = std::collections::BTreeSet::new();
    let mut graphs = vec![(root.to_path_buf(), text.to_owned())];
    let one = |path: &Path| store.canonical(path).unwrap_or_else(|_| store.whole(path));
    while let Some((path, text)) = graphs.pop() {
        if !reached.insert(one(&path)) {
            continue;
        }
        let document = entelechy_language::parse_document(&text).ok()?;
        let base = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let used = crate::sources::used_documents(store, &base, &document.uses).ok()?;
        reached.extend(used.iter().map(|used| one(used)));
        for node in &document.nodes {
            if let entelechy_language::NodeKind::SubGraph(placed) = &node.kind {
                let at = crate::sources::file_at(store, &base.join(&placed.value));
                if let Ok(text) = store.read_text(&at) {
                    graphs.push((at, text));
                }
            }
        }
    }
    Some(reached)
}

/// A problem line of a check (`path:line:column: message`, or `path:
/// message`) as a refusal of the document it names: a place a graph names
/// by its name (`scene:39:6`, as a node says where it is written) is the
/// graph document of that name among those written, else the scene; the
/// scene too when the line names none.
fn placed(
    store: &dyn Store,
    line: &str,
    root: &Path,
    written: &[(PathBuf, String)],
) -> crate::Refusal {
    let document = |named: &str| {
        let path = store.whole(Path::new(named));
        if store.kind(&path) == Some(crate::store::Kind::File) {
            return store.canonical(&path).unwrap_or(path);
        }
        written
            .iter()
            .map(|(path, _)| path)
            .find(|path| {
                path.extension().is_some_and(|e| e == "esn")
                    && path.file_stem().is_some_and(|stem| stem == named)
            })
            .cloned()
            .unwrap_or_else(|| root.to_path_buf())
    };
    let bytes = line.as_bytes();
    let mut at = None;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b':' {
            let rest = &line[i + 1..];
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !digits.is_empty() && rest[digits.len()..].starts_with(':') {
                let after = &rest[digits.len() + 1..];
                let column: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
                if !column.is_empty() && after[column.len()..].starts_with(':') {
                    at = Some((
                        i,
                        digits.parse().unwrap_or(1),
                        column.parse().unwrap_or(1),
                        after[column.len() + 1..].trim().to_owned(),
                    ));
                    break;
                }
            }
            if rest.starts_with(' ') && at.is_none() && i > 0 {
                let path = store.whole(Path::new(&line[..i]));
                if store.kind(&path).is_some() {
                    return crate::Refusal {
                        path: store.canonical(&path).unwrap_or(path),
                        diagnostic: crate::Diagnostic::new(rest.trim()),
                    };
                }
            }
        }
        i += 1;
    }
    match at {
        Some((end, line_no, column, message)) => crate::Refusal {
            path: document(&line[..end]),
            diagnostic: crate::Diagnostic {
                message,
                at: Some((line_no, column)),
            },
        },
        None => crate::Refusal {
            path: root.to_path_buf(),
            diagnostic: crate::Diagnostic::new(line),
        },
    }
}
