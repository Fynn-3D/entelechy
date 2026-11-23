//! Document-backed session operations: opening over a [`Store`], syncing
//! `.esn` and `.ent` sources, input/output allowances, live composition and
//! Kernel publishing.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use entelechy_documents::store::Store;
use entelechy_documents::{Event, inputs, kernels, scene, writes};
use entelechy_formats::ppm;
use entelechy_kernel::{Expect, KernelId, Runtime};

use super::{BEST, Backing, Claim, Docs, Session, TURN};
use crate::{Limits, Publishing, SessionError};

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
        let reads = Arc::new(std::sync::Mutex::new(
            entelechy_documents::reads::Reads::default(),
        ));
        let store: Arc<dyn Store> = Arc::new(entelechy_documents::reads::Held::new(
            store,
            Arc::clone(&reads),
        ));
        Self::from_backing(
            runtime,
            Backing::Documents(Box::new(Docs {
                scene: scene::Scene::new(store, scene, library),
                kernels: kernels::Kernels::default(),
                inputs: inputs::Inputs::default(),
                reads,
                writes: writes::Writes::default(),
                nothing_saved: false,
                placed: BTreeMap::new(),
                idle: BTreeMap::new(),
                unrun: BTreeMap::new(),
                shaped: None,
                history: entelechy_documents::history::History::default(),
                origin: crate::Origin::Local,
                claim: Claim::None,
                later: Vec::new(),
                resized: false,
                bound: BTreeMap::new(),
                chosen: None,
            })),
        )
    }

    /// Where the session's documents are. Callers acting for others go
    /// through [`Session::read_file`], [`Session::write_file`] and
    /// [`Session::write_scene`], which check allowances first.
    pub(crate) fn store(&self) -> &dyn Store {
        self.backing.docs().scene.store().as_ref()
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
            "best" => BEST,
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
            // In a page (no thread can start) the Kernels prepare in turns
            // from the first load: it is opened waiting for them, and the
            // frames prepare them a few at a time instead of one long sync.
            let turns = cfg!(all(target_arch = "wasm32", target_os = "unknown"));
            if turns {
                session.set_background(true);
            }
            let events = session.sync_a_turn();
            let refused = events
                .iter()
                .any(|event| matches!(event, Event::SceneRejected(_) | Event::Rejected { .. }));
            if session.graph().is_some() || (turns && session.scene_waiting() && !refused) {
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

    /// The scene's font nodes, every level's: each the externals its
    /// glyphs, outlines and metrics are in, and where its codes come from.
    pub fn fonts(&self) -> &[entelechy_scene::FontNode] {
        &self.backing.docs().scene.files().fonts
    }

    /// The outputs the scene gives Host devices (`to sound`, `to pointer`,
    /// ...), one a device.
    pub fn devices(&self) -> &[entelechy_scene::Present] {
        &self.backing.docs().scene.files().presented
    }

    /// The scene's files nodes, at every level.
    pub fn file_nodes(&self) -> &[entelechy_scene::FilesNode] {
        &self.backing.docs().scene.files().file_nodes
    }

    /// The scene's fields nodes, at every level.
    pub fn field_nodes(&self) -> &[entelechy_scene::FieldsNode] {
        &self.backing.docs().scene.files().field_nodes
    }

    /// The scene document.
    pub fn root(&self) -> &Path {
        self.backing.docs().scene.root()
    }

    /// The type of the param `name` the scene declares, as written; an
    /// undeclared one is refused, saying where it lives instead. `None`
    /// before a scene loads, which holds any name.
    pub fn param_type(&mut self, name: &str) -> Result<Option<String>, String> {
        self.settle_lowering();
        self.backing
            .docs()
            .scene
            .declared_param(name)
            .map(|param| param.map(|param| param.ty.to_string()))
    }

    /// Reads on a scene being lowered on another thread, waiting for it:
    /// a param is judged by the newest scene read, whatever the thread's
    /// timing. What it says is said by the next sync.
    pub(super) fn settle_lowering(&mut self) {
        let mut events = Vec::new();
        let docs = self.backing.docs_mut();
        docs.scene
            .land_lowering(&mut docs.kernels, &mut self.runtime, &mut events);
        docs.later.extend(events);
    }

    /// Starts the world over as the scene opened it, the Kernels and the
    /// documents kept: every `async` run landed, every pin dropped, every
    /// port's committed state forgotten, every parameter back at its
    /// default, what the frames gave the font and files nodes forgotten
    /// (their codes, their paths and the files read for them), and the
    /// documents read again. What a replay needs before
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
        {
            let docs = self.backing.docs_mut();
            docs.inputs
                .start_fed_over(docs.scene.files(), self.runtime.host_mut());
        }
        self.settle_lowering();
        {
            let docs = self.backing.docs_mut();
            let overridden: Vec<String> = docs.scene.params().keys().cloned().collect();
            docs.scene.clear_params();
            for name in overridden {
                docs.history
                    .changed(crate::Change::Param { name }, docs.origin.clone());
            }
        }
        self.sync()
    }

    /// Every document the session currently watches.
    pub fn watched(&self) -> Vec<PathBuf> {
        let docs = self.backing.docs();
        docs.kernels
            .documents()
            .into_iter()
            .chain(docs.inputs.watched())
            .collect()
    }

    /// Imports a PPM file as the external image `name`.
    pub fn bind_external(&mut self, name: &str, path: &Path) -> Result<(), SessionError> {
        let image = ppm::read(path).map_err(|e| SessionError::Image(path.to_path_buf(), e))?;
        let host = self.runtime.host_mut();
        let handle = host.adopt_image(image).map_err(SessionError::Host)?;
        self.backing.docs_mut().inputs.hold(host, name, handle);
        Ok(())
    }

    /// Reconciles the runtime with its store and reports what changed, in
    /// order: finished replacements, documents, inputs, placements. A
    /// changed scene is read and said by this sync.
    pub fn sync(&mut self) -> Vec<Event> {
        let mut events = self.sync_apart(false);
        // Asked for, a sync lands what it read: in turns (a page), what
        // waits is prepared now. Frames and loading take a turn at a time
        // ([`Session::sync_a_turn`]).
        if self.runtime.prepares_in_turns() && self.preparing_left() > 0 {
            self.runtime.prepare_turn(usize::MAX);
            events.extend(self.sync_apart(false));
            self.finish_all_turns();
        }
        events
    }

    /// A sync that, where Kernels prepare in turns (a page), prepares only
    /// a turn's few: what loading a scene, frame by frame, gives the page
    /// between. Elsewhere it is [`Session::sync`].
    pub fn sync_a_turn(&mut self) -> Vec<Event> {
        self.sync_apart(false)
    }

    /// [`Session::sync`] for a frame: where replacements prepare off the
    /// frames, a changed scene is lowered on another thread while the
    /// running one goes on, and said by the sync it lands on.
    pub fn sync_on_frame(&mut self) -> Vec<Event> {
        self.sync_apart(true)
    }

    pub(super) fn sync_apart(&mut self, apart: bool) -> Vec<Event> {
        let mut events = std::mem::take(&mut self.backing.docs_mut().later);
        // Where no thread can start (a page), the Kernels waiting are
        // prepared for a moment each sync ([`TURN`]), so the page and its
        // device get their turns between them; elsewhere the workers do.
        let began = entelechy_kernel::clock::now();
        while self.runtime.prepare_turn(1) > 0 && entelechy_kernel::clock::since(began) < TURN {}
        let store = Arc::clone(self.backing.docs().scene.store());
        // Every document walks the files it reaches; a path is looked at
        // once a sync.
        store.sync_begins();
        let reads = Arc::clone(&self.backing.docs().reads);
        let readable =
            |path: &Path| entelechy_documents::reads::lock(&reads).check(store.as_ref(), path);
        {
            let docs = self.backing.docs_mut();
            docs.scene.sync(
                &mut docs.kernels,
                &mut self.runtime,
                &mut events,
                &readable,
                apart,
            );
        }
        // Externals the caller's overrides bind to files, bound when new.
        let bound_now = self.backing.docs().scene.bound().to_vec();
        for (name, path) in bound_now {
            if self.backing.docs().bound.get(&name) == Some(&path) {
                continue;
            }
            match self.bind_external(&name, &path) {
                Ok(()) => {
                    events.push(Event::Loaded {
                        name: name.clone(),
                        path: path.clone(),
                        detail: "bound by the caller".into(),
                    });
                    self.backing.docs_mut().bound.insert(name, path);
                }
                Err(error) => events.push(Event::Rejected {
                    path: path.clone(),
                    error: crate::Diagnostic::new(error.to_string()),
                }),
            }
        }
        // The first-loaded scene's own documents and input files, worked
        // out only while reads wait for them. A first load whose Kernels
        // still prepare in turns (a page) has no scene running yet: it reads
        // freely until it lands, and is fixed then.
        let landing = self.backing.docs().scene.graph().is_none() && self.scene_waiting();
        let read_from: Vec<PathBuf> =
            if entelechy_documents::reads::lock(&reads).wants_directories() && !landing {
                let docs = self.backing.docs();
                let files = docs.scene.files();
                docs.scene
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
                    .chain(files.file_nodes.iter().map(|node| node.dir.clone()))
                    .chain(files.field_nodes.iter().map(|node| node.dir.clone()))
                    .collect()
            } else {
                Vec::new()
            };
        if !landing {
            entelechy_documents::reads::lock(&reads).settle(store.as_ref(), &read_from);
        }
        let on_device = {
            let docs = self.backing.docs_mut();
            let tables =
                entelechy_documents::inputs::table_elements(docs.scene.graph(), &self.runtime);
            let unread: Vec<PathBuf> = if docs.nothing_saved {
                docs.scene
                    .files()
                    .writes
                    .iter()
                    .filter(|write| docs.writes.author(store.as_ref(), &write.path).is_none())
                    .map(|write| write.path.clone())
                    .collect()
            } else {
                Vec::new()
            };
            let declared = entelechy_documents::inputs::Declared {
                graph: docs.scene.graph(),
                files: docs.scene.files(),
                tables: &tables,
                unread: &unread,
            };
            docs.inputs
                .sync(&store, &declared, &readable, &mut self.runtime, &mut events)
        };
        self.copies_let_go.extend(on_device);
        {
            let docs = self.backing.docs_mut();
            let clipboards = docs.scene.files().fed_by(crate::devices::CLIPBOARD);
            let already = clipboards.iter().all(|name| docs.inputs.holds(name));
            let inputs = &mut docs.inputs;
            let said = self.board.sync(
                &clipboards,
                already,
                self.runtime.host_mut(),
                |host, name, h| {
                    inputs.hold(host, name, h);
                },
            );
            events.extend(said.into_iter().map(|why| Event::Rejected {
                path: PathBuf::from(crate::devices::CLIPBOARD),
                error: crate::Diagnostic::new(why),
            }));
        }
        self.finish_used();
        if self.runtime.prepares_in_turns() && entelechy_kernel::clock::since(began) < TURN {
            self.finish_turn();
        }
        self.sync_placements(&mut events);
        self.sync_shape(&mut events);
        {
            let docs = self.backing.docs_mut();
            events.append(&mut docs.inputs.take_frame_events());
            events.append(&mut docs.writes.take_events());
        }
        self.remember(&events);
        store.sync_ends();
        events
    }

    /// Reads what the scene saved for itself on disk as if it were not
    /// there, until this run writes it: a replay starts from the same
    /// files whoever has played the scene before it.
    pub fn start_from_nothing_saved(&mut self) {
        self.backing.docs_mut().nothing_saved = true;
    }

    /// Allows the scene's outputs to be written to files in `dir` and
    /// under it (`output name <- source to "file";`). Outputs are written
    /// only where allowed.
    pub fn allow_writes(&mut self, dir: &Path) -> Result<(), String> {
        let docs = self.backing.docs_mut();
        docs.writes.allow(docs.scene.store().as_ref(), dir)
    }

    /// Allows an output `to files` to remove and rename files in `dir` and
    /// under it; writing them still needs [`Session::allow_writes`].
    pub fn allow_deletes(&mut self, dir: &Path) -> Result<(), String> {
        let docs = self.backing.docs_mut();
        docs.writes.allow_deletes(docs.scene.store().as_ref(), dir)
    }

    /// Restricts input reads to the scene's own directories (its own, its
    /// placed graphs', its libraries' and packages') and those
    /// [`Session::allow_reads`] allowed; for sessions an agent drives
    /// (`serve`, `--serve`, `entelechy_request`). Refused after the first
    /// sync; the directories are fixed as the scene first loads.
    pub fn restrict_reads(&mut self) -> Result<(), String> {
        entelechy_documents::reads::lock(&self.backing.docs().reads).restrict()
    }

    /// Allows the scene's inputs to be read from files in `dir` and under
    /// it, where reads are restricted ([`Session::restrict_reads`]).
    pub fn allow_reads(&mut self, dir: &Path) -> Result<(), String> {
        entelechy_documents::reads::lock(&self.backing.docs().reads).allow(self.store(), dir)
    }

    /// Allows a write to the one file `path`, named by the person running
    /// the session (`--save`, `--out`). Its directory stays closed.
    pub fn allow_write_to(&mut self, path: &Path) -> Result<(), String> {
        let docs = self.backing.docs_mut();
        docs.writes.allow_file(docs.scene.store().as_ref(), path)
    }

    /// Places the scene's inputs and outputs where `overrides` say: the
    /// running scene's at once (read again at the next sync), every scene
    /// read after. A name the scene does not read or write from a file is
    /// refused, and nothing changes.
    pub fn place(
        &mut self,
        overrides: entelechy_documents::overrides::Overrides,
    ) -> Result<(), String> {
        self.backing.docs_mut().scene.place(overrides)
    }

    /// Where the caller places the scene's inputs and outputs.
    pub fn overrides(&self) -> &entelechy_documents::overrides::Overrides {
        self.backing.docs().scene.overrides()
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
        if self.backing.docs().scene.graph().is_none() {
            return Err("no scene runs to take a folder".into());
        }
        if !self.store().is_dir(dir) {
            return Err(format!("{} is not a directory", dir.display()));
        }
        let mut overrides = self.backing.docs().scene.overrides().clone();
        for name in names {
            match entelechy_documents::overrides::kind_of(self.backing.docs().scene.written(), name)
            {
                Some(entelechy_documents::overrides::Kind::Directory) => {}
                Some(entelechy_documents::overrides::Kind::File) => {
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
                .backing
                .docs()
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
        self.backing.docs_mut().scene.place(overrides)?;
        if let Some((before, _)) = self.backing.docs_mut().chosen.take() {
            let docs = self.backing.docs_mut();
            let store = Arc::clone(docs.scene.store());
            docs.writes.revoke(store.as_ref(), &before);
            entelechy_documents::reads::lock(&docs.reads).revoke(store.as_ref(), &before);
        }
        for allowance in &allowances {
            self.allow(allowance)?;
        }
        self.backing.docs_mut().chosen = Some((dir.to_path_buf(), allowances));
        Ok(())
    }

    /// The folder last chosen ([`Session::choose_folder`]).
    pub fn chosen_folder(&self) -> Option<&Path> {
        self.backing
            .docs()
            .chosen
            .as_ref()
            .map(|(dir, _)| dir.as_path())
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
        self.backing.docs().scene.check(text, &readable, &granted)
    }

    /// Whether `path` may be read on the session's behalf: anywhere,
    /// unless reads are restricted ([`Session::restrict_reads`]), and then
    /// where the scene's documents are or use, in a package directory, or
    /// where allowed.
    pub fn allowed_read(&self, path: &Path) -> Result<(), String> {
        entelechy_documents::reads::lock(&self.backing.docs().reads).check(self.store(), path)
    }

    /// Where a write to `path` lands, if inside an allowed directory
    /// ([`Session::allow_writes`]). Callers acting for an agent ask this
    /// before every write.
    pub fn allowed_write(&self, path: &Path) -> Result<PathBuf, String> {
        let docs = self.backing.docs();
        docs.writes.place(docs.scene.store().as_ref(), path)
    }

    /// The scene document's text as it stands in the store.
    pub fn scene_text(&self) -> std::io::Result<String> {
        self.store().read_text(self.backing.docs().scene.root())
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
    pub fn write_file(&self, path: &Path, bytes: &[u8]) -> Result<PathBuf, String> {
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
            let docs = self.backing.docs_mut();
            let subject = docs.scene.root().display().to_string();
            docs.history
                .refused(subject, why.to_string(), Some(text), docs.origin.clone());
            return Ok(Err(why));
        }
        let root = self.backing.docs().scene.root().to_path_buf();
        self.store()
            .write(&root, text.as_bytes())
            .map_err(|e| SessionError::Io(root, e))?;
        Ok(Ok(()))
    }

    /// Where `path`, spelled from the scene's directory or whole, lands as
    /// a document a compose may write: a `.ent`, `.esn` or `.replay` file
    /// under the scene's directory, judged where it lands after its links, and in
    /// no package directory. A package is the person's, kept as found.
    pub fn composable(&self, path: &Path) -> Result<PathBuf, String> {
        let store = self.store();
        let dir = self
            .backing
            .docs()
            .scene
            .root()
            .parent()
            .unwrap_or_else(|| Path::new("."));
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
                let docs = self.backing.docs_mut();
                docs.history.refused(
                    refusal.path.display().to_string(),
                    refusal.diagnostic.to_string(),
                    text,
                    docs.origin.clone(),
                );
                return Ok(Err(refusal));
            }
        };
        let root = self
            .store()
            .canonical(self.backing.docs().scene.root())
            .unwrap_or_else(|_| self.backing.docs().scene.root().to_path_buf());
        let (scene, others): (Vec<_>, Vec<_>) =
            landed.into_iter().partition(|(path, _)| *path == root);
        for (path, text) in others.iter().chain(&scene) {
            self.store()
                .write(path, text.as_bytes())
                .map_err(|e| SessionError::Io(path.clone(), e))?;
        }
        let written: Vec<(PathBuf, String)> = others.into_iter().chain(scene).collect();
        let docs = self.backing.docs_mut();
        docs.history.documents(&written, docs.origin.clone());
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
            .canonical(self.backing.docs().scene.root())
            .unwrap_or_else(|_| self.backing.docs().scene.root().to_path_buf());
        let overlay = entelechy_documents::store::Overlay::new(store, &landed);
        let root_text = match landed.iter().find(|(path, _)| *path == root) {
            Some((_, text)) => text.clone(),
            None => (&overlay as &dyn Store)
                .read_text(&root)
                .map_err(|e| refused(root.clone(), e.to_string()))?,
        };
        let is_replay = |path: &Path| path.extension().is_some_and(|e| e == "replay");
        for (path, text) in landed.iter().filter(|(path, _)| is_replay(path)) {
            let base = path.parent().unwrap_or_else(|| Path::new("."));
            let replay = crate::replay::parse(text, base).map_err(|error| crate::Refusal {
                path: path.clone(),
                diagnostic: crate::Diagnostic {
                    message: error.message.clone(),
                    at: Some((error.line, 1)),
                    fix: None,
                },
            })?;
            let scene_written = overlay.kind(&overlay.whole(&replay.scene))
                == Some(entelechy_documents::store::Kind::File)
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
        let checked = entelechy_documents::check::check_document(&overlay, &root, false);
        if let Some((_, line)) = checked.lines.iter().find(|(problem, _)| *problem) {
            let mut refusal = placed(&overlay, line, &root, &landed);
            refusal.diagnostic.fix = checked
                .fixes
                .iter()
                .find(|(said, _)| said == line)
                .map(|(_, fix)| Box::new(mended_in(&overlay, fix, &landed)));
            return Err(refusal);
        }
        let readable = |path: &Path| self.allowed_read(path);
        let granted = |module: &entelechy_ir::Validated| {
            self.runtime
                .granted(module)
                .map_err(|why| crate::Diagnostic::new(why.to_string()))
        };
        self.backing
            .docs()
            .scene
            .check_in(&overlay, &root_text, &readable, &granted)
            .map_err(|diagnostic| crate::Refusal {
                path: root.clone(),
                diagnostic,
            })?;
        for (path, _) in landed.iter().filter(|(path, _)| *path != root) {
            if path.extension().and_then(|e| e.to_str()) != Some("ent") {
                continue;
            }
            let mut document = (&overlay as &dyn Store)
                .document(path)
                .map(|document| (*document).clone())
                .map_err(|why| refused(path.clone(), why))?;
            let dir = path.parent().unwrap_or_else(|| Path::new("."));
            let entelechy_language::Declarations {
                structs,
                types,
                functions,
                qualified,
            } = entelechy_documents::sources::declarations_for(
                &overlay,
                dir,
                &document.uses,
                entelechy_language::Own::of(&document),
            )
            .map_err(|why| refused(path.clone(), why))?;
            document.unqualify(&qualified);
            document.resolve((&structs, &types, &functions));
            for kernel in &document.kernels {
                let id = entelechy_documents::sources::named_id(kernel).map_err(|diagnostic| {
                    crate::Refusal {
                        path: path.clone(),
                        diagnostic,
                    }
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
        let Some(graph) = self.backing.docs().scene.shared_graph().cloned() else {
            return;
        };
        let revision = self.runtime.revision();
        if self
            .backing
            .docs()
            .shaped
            .as_ref()
            .is_some_and(|(known, at)| Arc::ptr_eq(known, &graph) && *at == revision)
        {
            return;
        }
        let docs = self.backing.docs_mut();
        let published = docs.kernels.documents().into_iter().map(|path| {
            let ids = docs.kernels.ids_of(&path);
            (path, ids)
        });
        let idle: BTreeMap<PathBuf, Vec<KernelId>> =
            entelechy_documents::check::idle(docs.scene.store().as_ref(), &graph, published)
                .into_iter()
                .collect();
        for (path, ids) in &idle {
            if docs.idle.get(path) != Some(ids) {
                events.push(Event::Idle {
                    path: path.clone(),
                    ids: ids.clone(),
                });
            }
        }
        docs.idle = idle;
        let unrun: BTreeMap<String, String> = graph
            .nodes
            .iter()
            .filter_map(|node| {
                let version = self.runtime.kernel(&node.kernel)?;
                let module = version.module()?;
                let why = entelechy_documents::check::defaulted_domain(
                    node,
                    &module.manifest,
                    entelechy_documents::check::first_counted_against(module),
                )?;
                Some((node.id.to_string(), why))
            })
            .collect();
        for (node, why) in &unrun {
            if docs.unrun.get(node) != Some(why) {
                events.push(Event::Unrun {
                    node: node.clone(),
                    why: why.clone(),
                });
            }
        }
        docs.unrun = unrun;
        docs.shaped = Some((graph, revision));
    }

    fn sync_placements(&mut self, events: &mut Vec<Event>) {
        let now = self.placements();
        let docs = self.backing.docs_mut();
        for (node, placement) in &now {
            let before = docs.placed.get(node);
            if before == Some(&placement.routed) {
                continue;
            }
            if let Some(why) = &placement.routed {
                events.push(Event::Routed {
                    node: node.clone(),
                    backend: placement.backend,
                    why: why.to_string(),
                });
            }
        }
        docs.placed = now
            .into_iter()
            .map(|(node, placement)| (node, placement.routed))
            .collect();
    }

    /// The events of the later sync on which a request's claimed scene
    /// landed, taken once. Empty when it landed on the request's own sync.
    pub fn take_landing(&mut self) -> Option<Vec<Event>> {
        let docs = self.backing.docs_mut();
        match std::mem::take(&mut docs.claim) {
            Claim::Landed(events) => Some(events),
            other => {
                docs.claim = other;
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
        let compiled = {
            let docs = self.backing.docs();
            docs.scene.compile_source(source, &docs.kernels)
        };
        match compiled {
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
    ) -> Option<Result<Arc<entelechy_kernel::KernelVersion>, crate::Diagnostic>> {
        let prepared = self.runtime.poll(publishing.ticket)?;
        let published = prepared
            .map_err(|e| crate::Diagnostic::new(e.to_string()))
            .and_then(|candidate| {
                self.runtime
                    .publish(candidate, publishing.expect)
                    .map_err(|e| crate::Diagnostic::new(e.to_string()))
            });
        match &published {
            Ok(version) => {
                let docs = self.backing.docs_mut();
                docs.history.published(
                    version.id(),
                    version.generation(),
                    &publishing.source,
                    docs.origin.clone(),
                );
            }
            Err(error) => self.refused_source(&publishing.source, error),
        }
        Some(published)
    }

    fn refused_source(&mut self, source: &str, error: &crate::Diagnostic) {
        let subject = entelechy_language::parse_document(source)
            .ok()
            .and_then(|document| document.kernels.first().and_then(|k| k.id.clone()))
            .map_or_else(|| "a Kernel".to_owned(), |id| id.value);
        let docs = self.backing.docs_mut();
        docs.history.refused(
            subject,
            error.to_string(),
            Some(source),
            docs.origin.clone(),
        );
    }

    /// Publishes a source holding exactly one Kernel, outside any
    /// document. Documents own ids: one that later defines the id
    /// republishes it.
    pub fn publish_source(
        &mut self,
        source: &str,
        expect: Expect,
    ) -> Result<Arc<entelechy_kernel::KernelVersion>, crate::Diagnostic> {
        let publishing = self.publish_later(source, expect)?;
        loop {
            if let Some(published) = self.publishing(&publishing) {
                return published;
            }
            self.wait_a_turn();
        }
    }

    /// Prepares `source`, one Kernel, without publishing it. It sees its
    /// own and `use`d declarations, then the scene's shared ones.
    pub fn prepare_source(
        &self,
        source: &str,
    ) -> Result<entelechy_kernel::Candidate, crate::Diagnostic> {
        let docs = self.backing.docs();
        docs.scene
            .prepare_source(source, &docs.kernels, &self.runtime)
    }

    /// Retires a published Kernel; returns whether one was registered.
    pub fn retire_kernel(&mut self, id: &KernelId) -> bool {
        let docs = self.backing.docs_mut();
        let retired = docs.kernels.retire_kernel(id, &mut self.runtime);
        if retired {
            docs.history.retired(id, docs.origin.clone());
        }
        retired
    }
}

fn reached_documents(store: &dyn Store, root: &Path, text: &str) -> Option<BTreeSet<PathBuf>> {
    let mut reached = BTreeSet::new();
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
        let used =
            entelechy_documents::sources::used_documents(store, &base, &document.uses).ok()?;
        reached.extend(used.iter().map(|used| one(used)));
        for node in &document.nodes {
            if let entelechy_language::NodeKind::SubGraph(placed) = &node.kind {
                let at = entelechy_documents::sources::file_at(store, &base.join(&placed.value));
                if let Ok(text) = store.read_text(&at) {
                    graphs.push((at, text));
                }
            }
        }
    }
    Some(reached)
}

fn mended_in(store: &dyn Store, fix: &crate::Fix, landed: &[(PathBuf, String)]) -> crate::Fix {
    let at = store
        .canonical(&fix.document)
        .unwrap_or_else(|_| fix.document.clone());
    let text = landed
        .iter()
        .find(|(path, _)| *path == at)
        .and_then(|(_, text)| {
            let edit = entelechy_scene::edit::Edit {
                target: entelechy_scene::edit::Target::Statement {
                    kind: entelechy_scene::edit::StatementKind::Use,
                    name: fix.using.clone(),
                },
                text: Some(fix.item()),
            };
            entelechy_scene::edit::edit(text, &[edit]).ok()
        });
    crate::Fix {
        text,
        ..fix.clone()
    }
}

fn placed(
    store: &dyn Store,
    line: &str,
    root: &Path,
    written: &[(PathBuf, String)],
) -> crate::Refusal {
    let document = |named: &str| {
        let path = store.whole(Path::new(named));
        if store.kind(&path) == Some(entelechy_documents::store::Kind::File) {
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
                fix: None,
            },
        },
        None => crate::Refusal {
            path: root.to_path_buf(),
            diagnostic: crate::Diagnostic::new(line),
        },
    }
}
