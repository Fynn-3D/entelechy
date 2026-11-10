//! The scene from its documents: the root `.esn`, its sub-graphs and
//! libraries, their shared declarations, the lowered Graph, and caller
//! params. The scene reads documents; the ledger ([`Kernels`])
//! publishes their Kernels.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use entelechy_graph::{Graph, GraphParam};
use entelechy_ir::KernelId;
use entelechy_kernel::{Runtime, Value};

use crate::kernels::{Batch, Change, Defined, Kernels};
use crate::sources::{
    Expanded, declarations_for, declarations_within, functions_checked, is_library,
    locate_document, named_id, used_documents,
};
use crate::store::{Stamp, Store};
use crate::{Diagnostic, Event};

/// A scene document and what was built from it.
pub struct Scene {
    store: Arc<dyn Store>,
    root: PathBuf,
    /// Libraries the caller adds to the document's uses.
    extra_uses: Vec<PathBuf>,
    /// The root's stamp at its last lowering; `None` lowers it again.
    stamp: Option<Stamp>,
    /// Resolved `use` targets of the root and its sub-graphs.
    uses: Vec<(PathBuf, bool)>,
    /// Placed sub-graph documents and their loaded stamps; a change to
    /// any rebuilds the graph.
    sub_graphs: BTreeMap<PathBuf, Option<Stamp>>,
    /// Each library read, with the documents it uses at the stamps it was
    /// read with; a change to any reads it again.
    depends: BTreeMap<PathBuf, Vec<(PathBuf, Option<Stamp>)>>,
    /// Every document `depends` names as used, each once, with its stamp
    /// when a sync last found every one of them as `depends` has it:
    /// while they stamp the same and `depends` is as it was, no library
    /// uses a changed document, and a sync asks of each used document
    /// once, not once for every library that uses it.
    used_settled: Option<Vec<(PathBuf, Option<Stamp>)>>,
    /// Shared declarations from `use`s and the root's top level; a
    /// Kernel published through the protocol inherits them.
    structs: Vec<entelechy_language::StructDecl>,
    types: Vec<entelechy_language::TypeAlias>,
    functions: Vec<entelechy_language::FnDecl>,
    graph: Option<Arc<Graph>>,
    /// The lowered document's file inputs and outputs, as the caller's
    /// overrides place them; and as the document writes them.
    files: entelechy_scene::Files,
    written: entelechy_scene::Files,
    /// Where the caller places the scene's inputs and outputs.
    overrides: crate::overrides::Overrides,
    /// Externals the caller binds to files, from its overrides.
    bound: Vec<(String, PathBuf)>,
    /// The root's text as last lowered: the structure that runs.
    text: Option<String>,
    /// Caller overrides of graph parameters.
    params: BTreeMap<String, Value>,
    /// A lowered change whose Kernels are being prepared: it replaces
    /// the running scene when they publish, and is dropped if one is
    /// refused.
    staged: Option<Staged>,
    /// Whether a newer read superseded a staged scene
    /// ([`Scene::take_superseded`]).
    superseded: bool,
    /// What the libraries' `use`s last reached, kept while nothing it
    /// looked at changed.
    expanded: Option<Expanded>,
    /// What the last refused read of the root said, while no scene runs.
    refusal: Option<Vec<Event>>,
    /// Library documents being read and their Kernels compiled on another
    /// thread ([`Scene::sync_libraries`]).
    compiling: Option<Compiling>,
    /// The root being lowered on another thread, a scene already running
    /// meanwhile ([`Scene::read_root`]).
    lowering: Option<Lowering>,
    /// What the root's lowerings named within a sync, shared by them: a
    /// sync that reads its libraries lowers the root again
    /// ([`crate::sources::Naming`]).
    named: crate::sources::Named,
}

/// The running scene, kept to be put back whole ([`Scene::put_back`]).
pub struct Kept {
    stamp: Option<Stamp>,
    uses: Vec<(PathBuf, bool)>,
    sub_graphs: BTreeMap<PathBuf, Option<Stamp>>,
    declared: Declared,
    graph: Option<Arc<Graph>>,
    files: entelechy_scene::Files,
    written: entelechy_scene::Files,
    bound: Vec<(String, PathBuf)>,
    text: Option<String>,
    params: BTreeMap<String, Value>,
}

/// A lowered scene waiting for its Kernels.
struct Staged {
    graph: Graph,
    text: String,
    files: entelechy_scene::Files,
    written: entelechy_scene::Files,
    bound: Vec<(String, PathBuf)>,
    uses: Vec<(PathBuf, bool)>,
    sub_graphs: BTreeMap<PathBuf, Option<Stamp>>,
    /// Graph documents only the running scene places; their Kernels
    /// retire once this one runs.
    gone: Vec<PathBuf>,
    /// The running scene's declarations, restored if this is refused.
    declared: Declared,
}

type Declared = (
    Vec<entelechy_language::StructDecl>,
    Vec<entelechy_language::TypeAlias>,
    Vec<entelechy_language::FnDecl>,
);

/// What a sync read of one document: its stamp, what it said, and its
/// Kernels, none when it was refused.
/// The documents a document uses, each with its stamp when read.
type Depends = Option<Vec<(PathBuf, Option<Stamp>)>>;

struct Read {
    path: PathBuf,
    current: Option<Stamp>,
    said: Vec<Event>,
    defined: Option<Vec<Defined>>,
}

impl Scene {
    /// A scene read through `store`, its documents under `root`, with
    /// `library`, when given, added to the document's uses.
    pub fn new(store: Arc<dyn Store>, root: &Path, library: Option<&Path>) -> Self {
        Self {
            store,
            root: root.to_path_buf(),
            extra_uses: library.map(Path::to_path_buf).into_iter().collect(),
            stamp: None,
            uses: Vec::new(),
            sub_graphs: BTreeMap::new(),
            depends: BTreeMap::new(),
            used_settled: None,
            structs: Vec::new(),
            types: Vec::new(),
            functions: Vec::new(),
            graph: None,
            files: entelechy_scene::Files::default(),
            written: entelechy_scene::Files::default(),
            overrides: crate::overrides::Overrides::default(),
            bound: Vec::new(),
            text: None,
            params: BTreeMap::new(),
            staged: None,
            superseded: false,
            expanded: None,
            refusal: None,
            compiling: None,
            lowering: None,
            named: Default::default(),
        }
    }

    /// The directories the scene reads: the root's, its placed graphs',
    /// each used directory, each used file's parent, and the parent of
    /// every document those use in turn, with the package it lies in.
    pub fn read_from(&self) -> Vec<PathBuf> {
        let beside = |path: &Path| {
            path.parent()
                .filter(|dir| !dir.as_os_str().is_empty())
                .unwrap_or(Path::new("."))
                .to_path_buf()
        };
        let used = self
            .uses
            .iter()
            .map(|(path, _)| path)
            .chain(&self.extra_uses)
            .map(|path| {
                if self.store.is_dir(path) {
                    path.clone()
                } else {
                    beside(path)
                }
            });
        let store = self.store.as_ref();
        let reached = self.depends.values().flatten().flat_map(|(path, _)| {
            std::iter::once(beside(path)).chain(crate::sources::package_holding(store, path))
        });
        std::iter::once(beside(&self.root))
            .chain(self.sub_graphs.keys().map(|path| beside(path)))
            .chain(used)
            .chain(reached)
            .collect()
    }

    /// Where its documents are read from.
    pub fn store(&self) -> &Arc<dyn Store> {
        &self.store
    }

    /// The directory its documents are under.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The root's text the running graph was lowered from.
    pub fn text(&self) -> Option<&str> {
        self.text.as_deref()
    }

    /// The graph documents the running scene places, with the stamp each
    /// was read at.
    pub fn placed(&self) -> impl Iterator<Item = (&Path, Option<Stamp>)> {
        self.sub_graphs
            .iter()
            .map(|(path, stamp)| (path.as_path(), *stamp))
    }

    /// The scene's file inputs and outputs, where the caller places them.
    pub fn files(&self) -> &entelechy_scene::Files {
        &self.files
    }

    /// The scene's file inputs and outputs as its documents write them.
    pub fn written(&self) -> &entelechy_scene::Files {
        &self.written
    }

    /// Externals the caller binds to files ([`crate::overrides::Overrides`]).
    pub fn bound(&self) -> &[(String, PathBuf)] {
        &self.bound
    }

    /// The parameters and bindings set from outside the document.
    pub fn overrides(&self) -> &crate::overrides::Overrides {
        &self.overrides
    }

    /// Places the scene's inputs and outputs as `overrides` say, the
    /// running scene's at once and every scene read after; refused,
    /// nothing changes. Before a scene is read, kept for its first read.
    pub fn place(&mut self, overrides: crate::overrides::Overrides) -> Result<(), String> {
        if let Some(graph) = &self.graph {
            let mut files = self.written.clone();
            self.bound = overrides.apply(&mut files, &graph.externals)?;
            self.files = files;
        }
        self.overrides = overrides;
        Ok(())
    }

    /// The running graph, once a scene has loaded.
    pub fn graph(&self) -> Option<&Graph> {
        self.graph.as_deref()
    }

    /// The running graph, shared: the evaluator knows it unchanged while it
    /// holds it.
    pub fn shared_graph(&self) -> Option<&Arc<Graph>> {
        self.graph.as_ref()
    }

    /// The graph's parameters as they now stand.
    pub fn params(&self) -> &BTreeMap<String, Value> {
        &self.params
    }

    /// The newest document read (staged, else running), so an answer
    /// about params is independent of worker timing.
    fn newest(&self) -> Option<&Graph> {
        self.staged
            .as_ref()
            .map(|staged| &staged.graph)
            .or(self.graph.as_deref())
    }

    /// The param `name` a loaded scene declares; one it does not is
    /// refused, saying why. `None` before a scene loads, which holds any.
    pub fn declared_param(&self, name: &str) -> Result<Option<&GraphParam>, String> {
        let Some(graph) = self.newest() else {
            return Ok(None);
        };
        match graph.params.iter().find(|p| p.name == name) {
            Some(param) => Ok(Some(param)),
            None => Err(undeclared(graph, name)),
        }
    }

    /// Overrides a graph parameter; see `Session::set_param`.
    pub fn set_param(&mut self, name: &str, value: Value) -> Result<(), String> {
        let value = match self.declared_param(name)? {
            Some(param) => value.shaped_like(&param.default).ok_or_else(|| {
                format!(
                    "param `{name}` is of type {}; `{value}` is not one",
                    param.ty
                )
            })?,
            None => value,
        };
        let sizes = self
            .newest()
            .is_some_and(|graph| graph.times_params.iter().any(|p| p == name));
        let previous = self.params.insert(name.to_owned(), value.clone());
        // A param that sizes the graph (a node's `times`) relowers the
        // document on the next sync.
        if previous != Some(value) && sizes {
            self.stamp = None;
        }
        Ok(())
    }

    /// Drops every parameter override, each back at the scene's default; a
    /// param that sizes the graph relowers the document on the next sync.
    pub fn clear_params(&mut self) {
        let sizes = self.newest().is_some_and(|graph| {
            graph
                .times_params
                .iter()
                .any(|param| self.params.contains_key(param))
        });
        self.params.clear();
        if sizes {
            self.stamp = None;
        }
    }

    /// Reads every changed document the root reaches and hands their
    /// Kernels to `kernels`. A changed library rereads the root, whose
    /// Kernels inherit its declarations. `apart` lets a changed root be
    /// lowered on another thread, said by a later sync; otherwise it is
    /// lowered here, and one lowered apart is waited for.
    pub fn sync(
        &mut self,
        kernels: &mut Kernels,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        apart: bool,
    ) {
        let from = events.len();
        // The worker's finished changes publish first.
        let outcome = kernels.collect(runtime, events);
        self.land(outcome, kernels, runtime, events);
        // The libraries the running scene uses first: what changed there
        // either publishes now, or is handed to another thread before the
        // scene is read, which then waits for it.
        if self.sync_libraries(kernels, runtime, events) {
            self.stamp = None;
        }
        self.sync_root(kernels, runtime, events, readable, apart);
        if self.sync_libraries(kernels, runtime, events) {
            self.stamp = None;
            self.sync_root(kernels, runtime, events, readable, apart);
        }
        // A scene that loaded, however many times this sync: each node
        // bound to a Kernel a document it reaches only through another
        // writes, said with the `use` that publishes it.
        if events[from..]
            .iter()
            .any(|event| matches!(event, Event::SceneLoaded))
        {
            self.say_unpublished(kernels, events);
        }
        self.named = Default::default();
    }

    /// Says each node of the running scene bound to a Kernel written in a
    /// document the scene reaches only through another.
    fn say_unpublished(&self, kernels: &Kernels, events: &mut Vec<Event>) {
        let Some(graph) = &self.graph else { return };
        let own: Vec<entelechy_ir::KernelId> = std::iter::once(self.root.clone())
            .chain(self.sub_graphs.keys().cloned())
            .flat_map(|path| kernels.ids_of(&path))
            .collect();
        for (node, kernel, why) in
            crate::sources::unpublished(self.store.as_ref(), &self.root, graph, &self.uses, &own)
        {
            let Some((why, fix)) = why else { continue };
            events.push(Event::Rejected {
                path: self.root.clone(),
                error: Diagnostic {
                    fix: Some(Box::new(fix)),
                    ..Diagnostic::new(format!(
                        "node `{node}` binds `{kernel}`, which the scene does not publish; {why}"
                    ))
                },
            });
        }
    }

    fn document_name(&self) -> String {
        self.root
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("scene")
            .to_owned()
    }

    fn base(&self) -> PathBuf {
        self.root
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    }

    /// Parses and lowers the root's text with its placed documents;
    /// publishes nothing.
    fn lower(&self, text: &str, readable: &dyn Fn(&Path) -> Result<(), String>) -> Lowered {
        self.lower_in(self.store.as_ref(), text, readable)
    }

    /// [`Scene::lower`], its documents read from `store`.
    fn lower_in(
        &self,
        store: &dyn Store,
        text: &str,
        readable: &dyn Fn(&Path) -> Result<(), String>,
    ) -> Lowered {
        let document = self.parse_readable(text, readable)?;
        self.lowering_of(document, text).lower(store)
    }

    /// The root's text parsed, every document it uses or places having
    /// passed `readable` before any is read.
    fn parse_readable(
        &self,
        text: &str,
        readable: &dyn Fn(&Path) -> Result<(), String>,
    ) -> Result<entelechy_language::Document, Diagnostic> {
        let document =
            entelechy_language::parse_document(text).map_err(|e| Diagnostic::from(&e))?;
        let base = self.base();
        let placed = document.nodes.iter().filter_map(|node| match &node.kind {
            entelechy_language::NodeKind::SubGraph(path) => Some(path),
            _ => None,
        });
        for path in document.uses.iter().map(|item| &item.path).chain(placed) {
            readable(&base.join(&path.value))
                .map_err(|why| Diagnostic::new(format!("`{}` is not read: {why}", path.value)))?;
        }
        Ok(document)
    }

    /// What lowering the parsed root needs, apart from the scene.
    fn lowering_of(&self, document: entelechy_language::Document, text: &str) -> ToLower {
        ToLower {
            document,
            text: text.to_owned(),
            name: self.document_name(),
            base: self.base(),
            params: self.params.clone(),
            first: self.graph.is_none(),
            store: Arc::clone(&self.store),
            named: Arc::clone(&self.named),
        }
    }
}

/// A parsed root and what lowering it reads: on this thread or another.
struct ToLower {
    document: entelechy_language::Document,
    text: String,
    name: String,
    base: PathBuf,
    params: BTreeMap<String, Value>,
    first: bool,
    store: Arc<dyn Store>,
    named: crate::sources::Named,
}

impl ToLower {
    /// The root lowered: its graph ordered, its params known.
    fn lower(self, store: &dyn Store) -> Lowered {
        let Self {
            document,
            text,
            name,
            base,
            params,
            first,
            named,
            ..
        } = self;
        let mut loader = |path: &Path| locate_document(store, path);
        let naming = crate::sources::Naming::sharing(store, named);
        let mut declare =
            |base: &Path, uses: &[entelechy_language::Use], own: entelechy_language::Own<'_>| {
                naming.declare(base, uses, own)
            };
        let mut lowered = entelechy_scene::load_with_params(
            &document,
            &text,
            &name,
            &base,
            &mut loader,
            &mut declare,
            &params,
        )
        .map_err(|e| Diagnostic::from(&e))?;
        // Each read where it is: through a package's name, inside it.
        crate::sources::placed_reads(store, &mut lowered.files);
        // A graph with a missing source or a cycle is refused here,
        // while the running scene stays.
        lowered
            .graph
            .order()
            .map_err(|e| graph_diagnostic(&document, &e))?;
        // A held param the first scene does not declare is refused.
        if first
            && let Some(name) = params
                .keys()
                .find(|name| !lowered.graph.params.iter().any(|p| &p.name == *name))
        {
            return Err(Diagnostic::new(undeclared(&lowered.graph, name)));
        }
        // What the root's own reach names, named with its places'.
        let named = naming
            .keyed(
                &base,
                &document.uses,
                entelechy_language::Own::of(&document),
            )
            .map(|(declarations, _)| declarations);
        Ok((document, lowered, named))
    }
}

impl Scene {
    /// Whether `text` as the root lowers to a complete graph and all its
    /// own Kernels compile and are granted. Writes, publishes and runs
    /// nothing; see `Session::check_scene`.
    pub fn check(
        &self,
        text: &str,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        granted: &dyn Fn(&entelechy_ir::Validated) -> Result<(), Diagnostic>,
    ) -> Result<(), Diagnostic> {
        self.check_in(self.store.as_ref(), text, readable, granted)
    }

    /// [`Scene::check`], its documents read from `store`.
    pub fn check_in(
        &self,
        store: &dyn Store,
        text: &str,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        granted: &dyn Fn(&entelechy_ir::Validated) -> Result<(), Diagnostic>,
    ) -> Result<(), Diagnostic> {
        let (mut document, lowered, named) = self.lower_in(store, text, readable)?;
        let entelechy_language::Declarations {
            structs,
            types,
            functions,
            qualified,
        } = named.map_err(Diagnostic::new)?;
        document.unqualify(&qualified);
        functions_checked(&self.root, &document, (&structs, &types, &functions))?;
        document.resolve((&structs, &types, &functions));
        let root_key = PathBuf::from(self.document_name());
        let inline = lowered
            .inline
            .iter()
            .filter(|inline| inline.document == root_key)
            .map(|inline| (Ok(inline.id.clone()), &inline.kernel));
        let top = document
            .kernels
            .iter()
            .map(|kernel| (named_id(kernel), kernel));
        for (id, kernel) in top.chain(inline) {
            let id = id?;
            let module = entelechy_compiler::compile_kernel(kernel, Some(&id))
                .map_err(|error| Diagnostic::from(&error))?;
            granted(&module)?;
        }
        Ok(())
    }

    /// Reads the root when it or what it reaches changed. With no scene
    /// running it reads the root every sync, so a library fixed under an
    /// unchanged root is found; a read of the same root refused for the
    /// same reasons says nothing it has not said.
    fn sync_root(
        &mut self,
        kernels: &mut Kernels,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        apart: bool,
    ) {
        // The scene is read once the library documents being compiled
        // land, so it never names a Kernel they are about to publish.
        if self.compiling.is_some() {
            return;
        }
        let from = events.len();
        let unchanged = self.stamp.is_some() && self.store.stamp(&self.root) == self.stamp;
        self.read_root(kernels, runtime, events, readable, apart);
        let refused = events[from..]
            .iter()
            .any(|event| matches!(event, Event::SceneRejected(_)));
        if !refused || self.graph.is_some() || self.staged.is_some() {
            self.refusal = None;
            return;
        }
        if unchanged && self.refusal.as_deref() == Some(&events[from..]) {
            events.truncate(from);
            return;
        }
        self.refusal = Some(events[from..].to_vec());
    }

    fn read_root(
        &mut self,
        kernels: &mut Kernels,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        apart: bool,
    ) {
        // A root lowered on another thread reads on once it lands, unless
        // the root changed since: then it is read again. A sync that is
        // not apart waits for it.
        if !apart && let Some(lowering) = &mut self.lowering {
            lowering.wait();
        }
        let landed = match &mut self.lowering {
            None => None,
            Some(lowering) => match lowering.landed() {
                Some(read) => Some(read),
                None if lowering.lost => {
                    self.lowering = None;
                    self.stamp = None;
                    return;
                }
                None => return,
            },
        };
        let (text, read) = match landed {
            Some(read) => {
                let lowering = self.lowering.take().expect("a lowering landed");
                // What it read must still be so: the root, the graphs it
                // placed, and no library change since asking it read again.
                let still = self.stamp == lowering.stamp
                    && self.store.stamp(&self.root) == lowering.stamp
                    && lowering
                        .placed
                        .iter()
                        .all(|(path, stamp)| self.store.stamp(path) == *stamp);
                if !still {
                    self.stamp = None;
                    return;
                }
                (lowering.text, read)
            }
            None => match self.read_root_text(kernels, runtime, events, readable, apart) {
                Some(read) => read,
                None => return,
            },
        };
        let (document, lowered, named) = match read {
            Ok(read) => read,
            Err(error) => {
                events.push(Event::SceneRejected(error));
                return;
            }
        };
        self.read_lowered(text, document, lowered, named, kernels, runtime, events);
    }

    /// Reads the root's text when it or a graph it places changed, and
    /// lowers it: here, or, on a sync apart where replacements prepare off
    /// the frames and a scene runs meanwhile, on another thread (`None`
    /// until it lands).
    fn read_root_text(
        &mut self,
        kernels: &mut Kernels,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        apart: bool,
    ) -> Option<(String, Lowered)> {
        let current = self.store.stamp(&self.root);
        // The newest read: staged, else running.
        let read = self
            .staged
            .as_ref()
            .map_or(&self.sub_graphs, |staged| &staged.sub_graphs);
        let sub_graph_changed = read
            .iter()
            .any(|(path, loaded)| self.store.stamp(path) != *loaded);
        let loaded = self.graph.is_some() || self.staged.is_some();
        if current == self.stamp && loaded && !sub_graph_changed {
            return None;
        }
        self.stamp = current;
        // A newer change supersedes a staged one.
        if let Some(staged) = self.staged.take() {
            (self.structs, self.types, self.functions) = staged.declared;
            kernels.supersede_scene(runtime, events);
            self.superseded = true;
        }
        let text = match self.store.read_text(&self.root) {
            Ok(text) => text,
            Err(error) => {
                events.push(Event::SceneRejected(Diagnostic::new(error.to_string())));
                return None;
            }
        };
        if apart && runtime.background() && self.graph.is_some() {
            let document = match self.parse_readable(&text, readable) {
                Ok(document) => document,
                Err(error) => return Some((text, Err(error))),
            };
            match Lowering::start(self.lowering_of(document, &text), current, text) {
                Ok(lowering) => {
                    self.lowering = Some(lowering);
                    return None;
                }
                Err(text) => {
                    let read = self.lower(&text, readable);
                    return Some((text, read));
                }
            }
        }
        let read = self.lower(&text, readable);
        Some((text, read))
    }

    /// Reads on from the root's text lowered, with what its reach names:
    /// its Kernels staged, and the scene staged to replace the running one
    /// once they publish.
    #[allow(clippy::too_many_arguments)]
    fn read_lowered(
        &mut self,
        text: String,
        document: entelechy_language::Document,
        mut lowered: entelechy_scene::Lowered,
        named: Result<entelechy_language::Declarations, String>,
        kernels: &mut Kernels,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) {
        let current = self.stamp;
        let (nodes, links) = shape(&lowered.graph);
        runtime.progress().shape(nodes, links);
        // The inputs and outputs where the caller places them.
        let written = lowered.files.clone();
        let bound = match self
            .overrides
            .apply(&mut lowered.files, &lowered.graph.externals)
        {
            Ok(bound) => bound,
            Err(error) => {
                events.push(Event::SceneRejected(Diagnostic::new(error)));
                return;
            }
        };
        // An output goes to a device the Host's table knows that takes one
        // (its structure checked when its Field is there, `entelechy check`
        // before); an input is fed by one that gives the type it declares.
        let refused = lowered
            .files
            .presented
            .iter()
            .find_map(|present| {
                let why = match crate::devices::named(&present.device) {
                    Err(why) => why,
                    Ok(device) if device.takes.is_none() => {
                        format!("`{}` feeds an input and takes no output", present.device)
                    }
                    Ok(_) => return None,
                };
                Some((present.at, format!("output `{}`: {why}", present.output)))
            })
            .or_else(|| {
                lowered.files.feeds.iter().find_map(|feed| {
                    crate::devices::feed_fault(&feed.device, &feed.ty)
                        .map(|why| (feed.at, format!("input `{}`: {why}", feed.input)))
                })
            });
        if let Some((at, why)) = refused {
            events.push(Event::SceneRejected(Diagnostic::at(at, why)));
            return;
        }
        // Events from here on judge this update's own documents.
        let said_from = events.len();
        let declared = (
            self.structs.clone(),
            self.types.clone(),
            self.functions.clone(),
        );
        let name = self.document_name();
        // Kernels by owning document: the root owns its top-level and
        // node-inline Kernels; a sub-graph owns its inline ones.
        let mut by_document: BTreeMap<PathBuf, Vec<Defined>> = BTreeMap::new();
        let root = self.root.clone();
        let root_key = PathBuf::from(&name);
        let mut document = document;
        match named {
            Ok(entelechy_language::Declarations {
                structs,
                types,
                functions,
                qualified,
            }) => {
                document.unqualify(&qualified);
                if let Err(error) =
                    functions_checked(&root, &document, (&structs, &types, &functions))
                {
                    events.push(Event::SceneRejected(error));
                    return;
                }
                self.structs = structs.iter().chain(&document.structs).cloned().collect();
                self.types = types.iter().chain(&document.types).cloned().collect();
                self.functions = functions
                    .iter()
                    .chain(&document.functions)
                    .cloned()
                    .collect();
                document.resolve((&structs, &types, &functions));
            }
            Err(error) => events.push(Event::Rejected {
                path: root.clone(),
                error: Diagnostic::new(error),
            }),
        }
        for kernel in &document.kernels {
            match named_id(kernel) {
                Ok(id) => by_document.entry(root.clone()).or_default().push(Defined {
                    id,
                    text: text[entelechy_language::doc::with_doc(&text, kernel.range.clone())]
                        .to_owned(),
                    kernel: kernel.clone(),
                }),
                Err(message) => events.push(Event::Rejected {
                    path: root.clone(),
                    error: message,
                }),
            }
        }
        for inline in &lowered.inline {
            let owner = if inline.document == root_key {
                root.clone()
            } else {
                inline.document.clone()
            };
            by_document.entry(owner).or_default().push(Defined {
                id: inline.id.clone(),
                text: inline.text.clone(),
                kernel: inline.kernel.clone(),
            });
        }
        let mut sub_graphs = BTreeMap::new();
        for path in &lowered.documents {
            sub_graphs.insert(path.clone(), self.store.stamp(path));
            by_document.entry(path.clone()).or_default();
        }
        by_document.entry(root.clone()).or_default();
        // Graph documents the new scene drops.
        let gone: Vec<PathBuf> = self
            .sub_graphs
            .keys()
            .filter(|path| !sub_graphs.contains_key(*path))
            .cloned()
            .collect();
        let mut batches: Vec<(PathBuf, Batch)> = Vec::new();
        for (path, defined) in by_document {
            let current = if path == root {
                current
            } else {
                self.store.stamp(&path)
            };
            let staged = kernels.stage_document(
                self.store.as_ref(),
                &path,
                current,
                defined,
                &root,
                runtime,
                events,
            );
            batches.extend(staged.map(|batch| (path, batch)));
        }
        // The root and its placed graphs publish whole or not at all; a
        // refusal leaves every Kernel and the running scene in place.
        let refused = events[said_from..].iter().any(|event| {
            matches!(event, Event::Rejected { path, .. }
                if *path == root || sub_graphs.contains_key(path))
        });
        if refused {
            (self.structs, self.types, self.functions) = declared;
            let change = Change {
                documents: batches,
                scene: true,
            };
            kernels.refuse(change, runtime, events);
            events.push(Event::SceneRejected(Diagnostic::new(
                "the scene's own Kernels were refused, so none of them is published and it does not replace the scene running",
            )));
            return;
        }
        // The graph replaces the running scene once all its Kernels
        // publish together.
        self.staged = Some(Staged {
            graph: lowered.graph,
            text,
            files: lowered.files,
            written,
            bound,
            uses: lowered.uses,
            sub_graphs,
            gone,
            declared,
        });
        let change = Change {
            documents: batches,
            scene: true,
        };
        let outcome = kernels.make(change, runtime, events);
        self.land(outcome, kernels, runtime, events);
    }

    /// Applies the ledger's `outcome` for the staged scene: published
    /// replaces the running scene, refused drops it, `None` waits.
    fn land(
        &mut self,
        outcome: Option<bool>,
        kernels: &mut Kernels,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) {
        if self.staged.is_none() {
            return;
        }
        match outcome {
            None => {}
            Some(false) => {
                let staged = self.staged.take().expect("a scene is staged");
                (self.structs, self.types, self.functions) = staged.declared;
                events.push(Event::SceneRejected(Diagnostic::new(
                    "a Kernel of the scene was refused where it was prepared, so none of them is published and it does not replace the scene running",
                )));
            }
            Some(true) => {
                let staged = self.staged.take().expect("a scene is staged");
                // Dropped documents retire their Kernels only now, when
                // the scene that ran them is replaced.
                for path in staged.gone {
                    kernels.retire_document(&path, runtime, events);
                }
                self.sub_graphs = staged.sub_graphs;
                self.uses = staged.uses;
                // Values of params the document does not declare drop.
                let graph = staged.graph;
                self.params
                    .retain(|name, _| graph.params.iter().any(|p| &p.name == name));
                self.graph = Some(Arc::new(graph));
                self.text = Some(staged.text);
                self.files = staged.files;
                self.written = staged.written;
                self.bound = staged.bound;
                events.push(Event::SceneLoaded);
            }
        }
    }

    /// A copy of the running scene, for a refused change to restore
    /// from memory.
    pub fn keep(&self) -> Kept {
        Kept {
            stamp: self.stamp,
            uses: self.uses.clone(),
            sub_graphs: self.sub_graphs.clone(),
            declared: (
                self.structs.clone(),
                self.types.clone(),
                self.functions.clone(),
            ),
            graph: self.graph.clone(),
            files: self.files.clone(),
            written: self.written.clone(),
            bound: self.bound.clone(),
            text: self.text.clone(),
            params: self.params.clone(),
        }
    }

    /// Restores the scene [`Scene::keep`] kept, the root read at
    /// `stamp`, regardless of the text on disk; drops any staged scene.
    pub fn put_back(&mut self, kept: Kept, stamp: Option<Stamp>) {
        self.staged = None;
        self.stamp = stamp.or(kept.stamp);
        self.uses = kept.uses;
        self.sub_graphs = kept.sub_graphs;
        (self.structs, self.types, self.functions) = kept.declared;
        self.graph = kept.graph;
        self.files = kept.files;
        self.written = kept.written;
        self.bound = kept.bound;
        self.text = kept.text;
        self.params = kept.params;
    }

    /// Whether a staged scene was superseded since the last call.
    pub fn take_superseded(&mut self) -> bool {
        std::mem::take(&mut self.superseded)
    }

    /// Whether a staged scene waits for its Kernels, or documents it uses
    /// compile off the frames before it is read again.
    pub fn waiting(&self) -> bool {
        self.staged.is_some() || self.compiling.is_some() || self.lowering.is_some()
    }

    /// Whether the running or staged scene places `path` as a graph.
    fn places(&self, path: &Path) -> bool {
        self.sub_graphs.contains_key(path)
            || self
                .staged
                .as_ref()
                .is_some_and(|staged| staged.sub_graphs.contains_key(path))
    }

    /// Waits until every prepared change, the scene's included, has
    /// published or been refused.
    pub fn wait(&mut self, kernels: &mut Kernels, runtime: &mut Runtime, events: &mut Vec<Event>) {
        let outcome = kernels.wait(runtime, events);
        self.land(outcome, kernels, runtime, events);
    }

    /// Whether library documents are being read and compiled on another
    /// thread, to publish at a later sync.
    pub fn compiling(&self) -> bool {
        self.compiling.is_some() || self.lowering.is_some()
    }

    /// Waits for a root being lowered on another thread and reads it on,
    /// so what is judged by the newest scene read does not hang on the
    /// thread's timing.
    pub fn land_lowering(
        &mut self,
        kernels: &mut Kernels,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) {
        if let Some(lowering) = &mut self.lowering {
            lowering.wait();
            // A landed lowering is read on without reading the root again.
            self.read_root(kernels, runtime, events, &|_| Ok(()), false);
        }
    }

    /// Waits until the library documents being compiled, and the root
    /// being lowered, are done, for the next sync to publish.
    pub fn wait_compiled(&mut self) {
        if let Some(compiling) = &mut self.compiling {
            compiling.wait();
        }
        if let Some(lowering) = &mut self.lowering {
            lowering.wait();
        }
    }

    /// Loads the libraries the root uses; returns whether any changed.
    fn sync_libraries(
        &mut self,
        kernels: &mut Kernels,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) -> bool {
        let mut changed = false;
        let uses: Vec<(PathBuf, bool)> = self
            .uses
            .iter()
            .cloned()
            .chain(self.extra_uses.iter().map(|path| (path.clone(), false)))
            .collect();
        let store = self.store.as_ref();
        let expanded = match self.expanded.take() {
            Some(expanded) if expanded.holds(store, &uses) => expanded,
            _ => Expanded::of(store, uses),
        };
        // Each file once, in the order of the paths, as the expansion
        // left them: all but the root and what the scene places.
        let wanted: Vec<PathBuf> = expanded
            .distinct
            .iter()
            .filter(|path| **path != self.root && !self.places(path))
            .cloned()
            .collect();
        self.expanded = Some(expanded);
        let stale: Vec<PathBuf> = kernels
            .documents()
            .into_iter()
            .filter(|path| {
                *path != self.root && wanted.binary_search(path).is_err() && !self.places(path)
            })
            .collect();
        for path in stale {
            kernels.retire_document(&path, runtime, events);
        }
        // Each document's stamp, read once this sync however many
        // documents use it; the paths are whole, so their bytes name them.
        let store = Arc::clone(&self.store);
        let mut stamps: std::collections::HashMap<std::ffi::OsString, Option<Stamp>> =
            std::collections::HashMap::new();
        let mut stamp_of = |path: &Path| {
            if let Some(stamp) = stamps.get(path.as_os_str()) {
                return *stamp;
            }
            let stamp = store.stamp(path);
            stamps.insert(path.as_os_str().to_owned(), stamp);
            stamp
        };
        // A compile handed to another thread lands at the first sync after
        // it is done; until then nothing more is read.
        let mut read: Vec<Read> = Vec::new();
        let landed = self.compiling.as_mut().and_then(Compiling::landed);
        if landed.is_some() || self.compiling.as_ref().is_some_and(|c| c.lost) {
            self.compiling = None;
        }
        if let Some(compiled) = landed {
            for (document, depends) in compiled.reads {
                if wanted.binary_search(&document.path).is_err() {
                    continue;
                }
                self.used_settled = None;
                self.depends.remove(&document.path);
                if let Some(used) = depends {
                    self.depends.insert(document.path.clone(), used);
                }
                read.push(document);
            }
            kernels.take_made(compiled.made);
            changed = true;
        } else if self.compiling.is_none() {
            let mut to_read: Vec<(PathBuf, Option<Stamp>)> = Vec::new();
            // The libraries that use a changed document. While every used
            // document stamps as the last sync found it there are none;
            // else each library's are looked at, and found all unchanged
            // they are kept, each once, to be asked of alone from here on.
            let settled = self
                .used_settled
                .as_ref()
                .is_some_and(|used| used.iter().all(|(path, stamp)| stamp_of(path) == *stamp));
            let mut moved: BTreeSet<&Path> = BTreeSet::new();
            if !settled {
                for (path, used) in &self.depends {
                    if !used.iter().all(|(used, stamp)| stamp_of(used) == *stamp) {
                        moved.insert(path.as_path());
                    }
                }
                self.used_settled = moved.is_empty().then(|| {
                    let mut used: Vec<(PathBuf, Option<Stamp>)> =
                        self.depends.values().flatten().cloned().collect();
                    used.sort_by(|a, b| a.0.cmp(&b.0));
                    used.dedup_by(|a, b| a.0 == b.0);
                    used
                });
            }
            for path in wanted {
                let current = stamp_of(&path);
                // Read again when it changed, or any document it uses did.
                if kernels.read_at(&path, current) && !moved.contains(path.as_path()) {
                    continue;
                }
                to_read.push((path, current));
            }
            // Where replacements prepare off the frames, so are documents
            // read before read again and their Kernels compiled; one read
            // for the first time is read now, as a Kernel nothing runs yet
            // prepares now.
            if !to_read.is_empty()
                && runtime.background()
                && to_read.iter().all(|(path, _)| kernels.has_read(path))
            {
                let ledger =
                    kernels.ledger(to_read.iter().map(|(path, _)| path.as_path()), runtime);
                match Compiling::start(Arc::clone(&self.store), to_read, ledger) {
                    Ok(compiling) => {
                        self.compiling = Some(compiling);
                        return false;
                    }
                    Err(back) => to_read = back,
                }
            }
            changed |= !to_read.is_empty();
            let store = self.store.as_ref();
            let reads = crate::spread::map(&to_read, |(path, current)| {
                read_library(store, path.clone(), *current)
            });
            for (document, depends) in reads {
                self.used_settled = None;
                self.depends.remove(&document.path);
                if let Some(used) = depends {
                    self.depends.insert(document.path.clone(), used);
                }
                read.push(document);
            }
            // Every new or changed Kernel compiled and prepared at once; then
            // each document published in order, with what it said, as if
            // each were read, compiled and prepared in turn.
            let primed: Vec<(&Path, &[Defined])> = read
                .iter()
                .filter_map(|read| Some((read.path.as_path(), read.defined.as_deref()?)))
                .collect();
            kernels.prime(self.store.as_ref(), &primed, runtime);
        }
        for Read {
            path,
            current,
            said,
            defined,
        } in read
        {
            events.extend(said);
            let Some(defined) = defined else {
                kernels.mark_read(path, current);
                continue;
            };
            let root = self.root.clone();
            let outcome = kernels.publish_document(
                self.store.as_ref(),
                &path,
                current,
                defined,
                &root,
                runtime,
                events,
            );
            self.land(outcome, kernels, runtime, events);
        }
        kernels.unprime();
        changed
    }

    /// Prepares a one-Kernel document unpublished; see
    /// `Session::prepare_source`.
    pub fn prepare_source(
        &self,
        source: &str,
        kernels: &Kernels,
        runtime: &Runtime,
    ) -> Result<entelechy_kernel::Candidate, Diagnostic> {
        let module = self.compile_source(source, kernels)?;
        runtime
            .prepare_module(module, source)
            .map_err(|e| Diagnostic::new(e.to_string()))
    }

    /// Compiles a one-Kernel document as [`Scene::prepare_source`] does.
    pub fn compile_source(
        &self,
        source: &str,
        kernels: &Kernels,
    ) -> Result<entelechy_kernel::Validated, Diagnostic> {
        let kernel = self.resolve_source(source)?;
        entelechy_compiler::compile_kernel(&kernel, None).map_err(|e| {
            kernels.located(
                self.store.as_ref(),
                Diagnostic::from(&e),
                &self.root,
                &self.root,
            )
        })
    }

    /// `source`, one Kernel, as [`Scene::compile_source`] reads it: its own
    /// and `use`d declarations, then the scene's shared ones.
    pub fn resolve_source(&self, source: &str) -> Result<entelechy_language::Kernel, Diagnostic> {
        let mut document =
            entelechy_language::parse_document(source).map_err(|e| Diagnostic::from(&e))?;
        if document.kernels.len() != 1 {
            return Err(Diagnostic::new(format!(
                "expected one kernel, found {}",
                document.kernels.len()
            )));
        }
        let base = self.root.parent().unwrap_or_else(|| Path::new("."));
        let entelechy_language::Declarations {
            mut structs,
            mut types,
            mut functions,
            qualified,
        } = if document.uses.is_empty() {
            entelechy_language::Declarations::default()
        } else {
            declarations_for(
                self.store.as_ref(),
                base,
                &document.uses,
                entelechy_language::Own::of(&document),
            )
            .map_err(Diagnostic::new)?
        };
        document.unqualify(&qualified);
        // Shared declarations fill in names the source does not declare;
        // the source's own win, so a Kernel may redefine a shared struct
        // and change a state's layout while it runs.
        for shared in &self.structs {
            let declared = |d: &entelechy_language::StructDecl| d.name.value == shared.name.value;
            if !document.structs.iter().any(declared)
                && !structs.iter().any(declared)
                && !document.kernels[0].structs.iter().any(declared)
            {
                structs.push(shared.clone());
            }
        }
        for shared in &self.types {
            let declared = |d: &entelechy_language::TypeAlias| d.name.value == shared.name.value;
            if !document.types.iter().any(declared)
                && !types.iter().any(declared)
                && !document.kernels[0].types.iter().any(declared)
            {
                types.push(shared.clone());
            }
        }
        for shared in &self.functions {
            let declared = |d: &entelechy_language::FnDecl| d.name.value == shared.name.value;
            if !document.functions.iter().any(declared)
                && !functions.iter().any(declared)
                && !document.kernels[0].functions.iter().any(declared)
            {
                functions.push(shared.clone());
            }
        }
        document.resolve((&structs, &types, &functions));
        Ok(document.kernels.swap_remove(0))
    }
}

/// A graph's refusal at the place `document` wrote the source it names,
/// when it wrote it; a cycle, or a source in a placed graph, has none.
pub(crate) fn graph_diagnostic(
    document: &entelechy_language::Document,
    error: &entelechy_graph::GraphError,
) -> Diagnostic {
    use entelechy_graph::{GraphError, Source};
    let named = match error {
        GraphError::UnknownSource { source, .. } | GraphError::UnknownOutput(source) => {
            Some(source)
        }
        _ => None,
    };
    let span = named.and_then(|source| {
        document
            .source_refs()
            .into_iter()
            .find(|written| match source {
                Source::Node { node, port } => {
                    written.node.as_deref() == Some(node.0.as_str()) && &written.name == port
                }
                Source::External(name) => written.node.is_none() && &written.name == name,
            })
    });
    match span {
        Some(written) => Diagnostic::at(written.span, error.to_string()),
        None => Diagnostic::new(error.to_string()),
    }
}

/// Why the undeclared `name` is refused: the nearest declared param, or
/// the nodes that set it themselves.
fn undeclared(graph: &Graph, name: &str) -> String {
    let mut why = format!("`{name}` is not a param the scene declares");
    let near = entelechy_language::nearest(name, graph.params.iter().map(|p| p.name.as_str()));
    if near.is_some() {
        why.push_str(&entelechy_language::suggestion(near));
        return why;
    }
    let setters: Vec<&str> = graph
        .nodes
        .iter()
        .filter(|node| node.params.contains_key(name))
        .map(|node| node.id.0.as_str())
        .collect();
    let who = match setters.as_slice() {
        [] => return why,
        [one] => format!("node `{one}` sets its own"),
        many => format!(
            "nodes {} set their own",
            many.iter()
                .map(|node| format!("`{node}`"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    why.push_str(&format!(
        "; {who} `{name}`, which is the document's: change it there (over the protocol, `compose` with `edits`, node `{}`, member `{name}`), or declare a `param` there for the node to name",
        setters[0]
    ));
    why
}

/// One library document read: parsed, its declarations gathered and
/// checked, its Kernels found; and the documents it uses with their
/// stamps, when it got as far as gathering them. It reads the store and
/// nothing else, so it runs on any thread.
fn read_library(store: &dyn Store, path: PathBuf, current: Option<Stamp>) -> (Read, Depends) {
    let mut depends = None;
    let mut said = Vec::new();
    let text = match store.read_text(&path) {
        Ok(text) => text,
        Err(error) => {
            said.push(Event::Rejected {
                path: path.clone(),
                error: Diagnostic::new(error.to_string()),
            });
            return (
                Read {
                    path,
                    current,
                    said,
                    defined: None,
                },
                depends,
            );
        }
    };
    let document = match entelechy_language::parse_document(&text) {
        Ok(document) => document,
        Err(error) => {
            said.push(Event::Rejected {
                path: path.clone(),
                error: Diagnostic::from(&error),
            });
            return (
                Read {
                    path,
                    current,
                    said,
                    defined: None,
                },
                depends,
            );
        }
    };
    if !is_library(&document) {
        said.push(Event::Rejected {
            path: path.clone(),
            error: Diagnostic {
                message: "a used document must hold kernels only".into(),
                at: Some((1, 1)),
                fix: None,
            },
        });
        return (
            Read {
                path,
                current,
                said,
                defined: None,
            },
            depends,
        );
    }
    let dir = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let mut document = document;
    match declarations_within(
        store,
        Some(&path),
        &dir,
        &document.uses,
        entelechy_language::Own::of(&document),
    ) {
        Ok(entelechy_language::Declarations {
            structs,
            types,
            functions,
            qualified,
        }) => {
            document.unqualify(&qualified);
            let used = used_documents(store, &dir, &document.uses)
                .unwrap_or_default()
                .into_iter()
                .map(|used| {
                    let stamp = store.stamp(&used);
                    (used, stamp)
                })
                .collect();
            depends = Some(used);
            if let Err(error) = functions_checked(&path, &document, (&structs, &types, &functions))
            {
                said.push(Event::Rejected {
                    path: path.clone(),
                    error,
                });
                return (
                    Read {
                        path,
                        current,
                        said,
                        defined: None,
                    },
                    depends,
                );
            }
            document.resolve((&structs, &types, &functions));
        }
        Err(error) => {
            said.push(Event::Rejected {
                path: path.clone(),
                error: Diagnostic::new(error),
            });
            return (
                Read {
                    path,
                    current,
                    said,
                    defined: None,
                },
                depends,
            );
        }
    };
    let mut defined = Vec::new();
    for kernel in &document.kernels {
        match named_id(kernel) {
            Ok(id) => defined.push(Defined {
                id,
                text: text[entelechy_language::doc::with_doc(&text, kernel.range.clone())]
                    .to_owned(),
                kernel: kernel.clone(),
            }),
            Err(message) => said.push(Event::Rejected {
                path: path.clone(),
                error: message,
            }),
        }
    }
    (
        Read {
            path,
            current,
            said,
            defined: Some(defined),
        },
        depends,
    )
}

/// The root lowered on another thread ([`ToLower`]), for a later sync to
/// read on with.
struct Lowering {
    done: std::sync::mpsc::Receiver<(Lowered, Placed)>,
    ready: Option<Lowered>,
    /// Whether its thread ended without an answer (it panicked): the root
    /// is read again.
    lost: bool,
    /// The root's stamp and text it lowered, and the stamps of the graphs
    /// it placed as it lowered them.
    stamp: Option<Stamp>,
    text: String,
    placed: Placed,
}

/// The root parsed and lowered, or why not.
type Lowered = Result<
    (
        entelechy_language::Document,
        entelechy_scene::Lowered,
        Result<entelechy_language::Declarations, String>,
    ),
    Diagnostic,
>;
/// The graphs a lowering placed, each with its stamp as it was read.
type Placed = Vec<(PathBuf, Option<Stamp>)>;

impl Lowering {
    /// Starts lowering on a thread, or gives the text back where no thread
    /// can start.
    fn start(to_lower: ToLower, stamp: Option<Stamp>, text: String) -> Result<Self, String> {
        let (send, done) = std::sync::mpsc::channel();
        let started = std::thread::Builder::new()
            .name("entelechy-lower".into())
            // Room for the language's nesting bound (entelechy_language::DEEP_STACK).
            .stack_size(256 << 20)
            .spawn(move || {
                let store = Arc::clone(&to_lower.store);
                let lowered = to_lower.lower(store.as_ref());
                let placed = match &lowered {
                    Ok((_, lowered, _)) => lowered
                        .documents
                        .iter()
                        .map(|path| (path.clone(), store.stamp(path)))
                        .collect(),
                    Err(_) => Vec::new(),
                };
                let _ = send.send((lowered, placed));
            });
        match started {
            Ok(_) => Ok(Self {
                done,
                ready: None,
                lost: false,
                stamp,
                text,
                placed: Vec::new(),
            }),
            Err(_) => Err(text),
        }
    }

    /// The root lowered, once done.
    fn landed(&mut self) -> Option<Lowered> {
        if let Some(ready) = self.ready.take() {
            return Some(ready);
        }
        match self.done.try_recv() {
            Ok((lowered, placed)) => {
                self.placed = placed;
                Some(lowered)
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.lost = true;
                None
            }
        }
    }

    /// Waits until it is done.
    fn wait(&mut self) {
        if self.ready.is_none() {
            match self.done.recv() {
                Ok((lowered, placed)) => {
                    self.placed = placed;
                    self.ready = Some(lowered);
                }
                Err(_) => self.lost = true,
            }
        }
    }
}

/// The graph's nodes, each by its Kernel, and each node with each it
/// reads from, for the picture of a scene preparing
/// ([`entelechy_kernel::Progress::shape`]).
fn shape(graph: &Graph) -> (Vec<KernelId>, Vec<(usize, usize)>) {
    let place: BTreeMap<&entelechy_graph::NodeId, usize> = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(i, node)| (&node.id, i))
        .collect();
    let mut links = Vec::new();
    for (i, node) in graph.nodes.iter().enumerate() {
        for binding in node.inputs.values() {
            let source = match binding {
                entelechy_graph::Binding::Now(source) => source,
                entelechy_graph::Binding::Previous { source, .. } => source,
            };
            if let entelechy_graph::Source::Node { node: from, .. } = source
                && let Some(&from) = place.get(from)
            {
                links.push((i, from));
            }
        }
    }
    let nodes = graph.nodes.iter().map(|node| node.kernel.clone()).collect();
    (nodes, links)
}

/// Library documents read and their Kernels primed on another thread
/// ([`read_library`], [`crate::kernels::Ledger::prime`]), for the sync
/// after to publish.
struct Compiling {
    done: std::sync::mpsc::Receiver<Compiled>,
    ready: Option<Compiled>,
    /// Whether its thread ended without an answer (it panicked): its
    /// documents are read again.
    lost: bool,
}

/// What a [`Compiling`] made: each document read, with what it uses, in
/// order, and its Kernels primed.
struct Compiled {
    reads: Vec<(Read, Depends)>,
    made: crate::kernels::Made,
}

impl Compiling {
    /// Starts reading `to_read` on a thread, or gives it back where no
    /// thread can start.
    fn start(
        store: Arc<dyn Store>,
        to_read: Vec<(PathBuf, Option<Stamp>)>,
        ledger: crate::kernels::Ledger,
    ) -> Result<Self, Vec<(PathBuf, Option<Stamp>)>> {
        let (send, done) = std::sync::mpsc::channel();
        let shared = Arc::new(std::sync::Mutex::new(Some(to_read)));
        let theirs = Arc::clone(&shared);
        let started = std::thread::Builder::new()
            .name("entelechy-compile".into())
            // Room for the language's nesting bound
            // (entelechy_language::DEEP_STACK), which the compiler recurses through.
            .stack_size(256 << 20)
            .spawn(move || {
                let Some(to_read) = theirs
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take()
                else {
                    return;
                };
                let store = store.as_ref();
                let reads = crate::spread::map(&to_read, |(path, current)| {
                    read_library(store, path.clone(), *current)
                });
                let documents: Vec<(&Path, &[Defined])> = reads
                    .iter()
                    .filter_map(|(read, _)| Some((read.path.as_path(), read.defined.as_deref()?)))
                    .collect();
                let made = ledger.prime(store, &documents);
                let _ = send.send(Compiled { reads, made });
            });
        match started {
            Ok(_) => Ok(Self {
                done,
                ready: None,
                lost: false,
            }),
            Err(_) => Err(shared
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
                .unwrap_or_default()),
        }
    }

    /// What it made, once done.
    fn landed(&mut self) -> Option<Compiled> {
        if let Some(ready) = self.ready.take() {
            return Some(ready);
        }
        match self.done.try_recv() {
            Ok(compiled) => Some(compiled),
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.lost = true;
                None
            }
        }
    }

    /// Waits until it is done.
    fn wait(&mut self) {
        if self.ready.is_none() {
            match self.done.recv() {
                Ok(compiled) => self.ready = Some(compiled),
                Err(_) => self.lost = true,
            }
        }
    }
}
