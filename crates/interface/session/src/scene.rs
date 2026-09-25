//! The scene as its documents say it: the root `.osn`, the sub-graphs
//! it places, the libraries it uses, the declarations they share, the
//! Graph lowered from them, and the parameters the caller set. Reading
//! the documents is the scene's; publishing the Kernels they define is
//! the ledger's, which the scene hands them to.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use orla_graph::Graph;
use orla_runtime::{Runtime, Value};

use crate::kernels::{Defined, Kernels};
use crate::sources::{
    declarations_for, declarations_key, declarations_within, expand_use, is_library, named_id,
    read_document,
};
use crate::store::{Stamp, Store};
use crate::{Diagnostic, Event};

/// A scene document and what was built from it.
pub(crate) struct Scene {
    /// Where the documents are.
    store: Arc<dyn Store>,
    root: PathBuf,
    /// Libraries the caller adds to what the document uses.
    extra_uses: Vec<PathBuf>,
    /// The root's stamp when it was last lowered; `None` lowers it again.
    stamp: Option<Stamp>,
    /// `use` targets from the last good parse of the root and its
    /// sub-graphs, resolved.
    uses: Vec<(PathBuf, bool)>,
    /// Sub-graph documents placed by the root, with the stamps they were
    /// loaded at; a change to any of them rebuilds the graph.
    sub_graphs: BTreeMap<PathBuf, Option<Stamp>>,
    /// The scene's shared declarations, from its `use`s and its own top
    /// level, which a Kernel published through the protocol inherits.
    structs: Vec<orla_language::StructDecl>,
    types: Vec<orla_language::TypeAlias>,
    functions: Vec<orla_language::FnDecl>,
    graph: Option<Graph>,
    /// What the lowered document reads from and writes to files.
    files: orla_scene::Files,
    /// The root's text as last lowered: the structure that runs.
    text: Option<String>,
    /// Caller overrides of graph parameters.
    params: BTreeMap<String, Value>,
}

impl Scene {
    pub(crate) fn new(store: Arc<dyn Store>, root: &Path, library: Option<&Path>) -> Self {
        Self {
            store,
            root: root.to_path_buf(),
            extra_uses: library.map(Path::to_path_buf).into_iter().collect(),
            stamp: None,
            uses: Vec::new(),
            sub_graphs: BTreeMap::new(),
            structs: Vec::new(),
            types: Vec::new(),
            functions: Vec::new(),
            graph: None,
            files: orla_scene::Files::default(),
            text: None,
            params: BTreeMap::new(),
        }
    }

    /// The directories the scene reads from by what it says: the root's
    /// and those of the graphs it places, and what it `use`s, a
    /// library's or a package's directory as it is, a file's beside it.
    pub(crate) fn read_from(&self) -> Vec<PathBuf> {
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
        std::iter::once(beside(&self.root))
            .chain(self.sub_graphs.keys().map(|path| beside(path)))
            .chain(used)
            .collect()
    }

    pub(crate) fn store(&self) -> &Arc<dyn Store> {
        &self.store
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    /// The root document's text as last lowered: the structure the
    /// graph that runs was built from.
    pub(crate) fn text(&self) -> Option<&str> {
        self.text.as_deref()
    }

    /// What the scene reads from and writes to files.
    pub(crate) fn files(&self) -> &orla_scene::Files {
        &self.files
    }

    pub(crate) fn graph(&self) -> Option<&Graph> {
        self.graph.as_ref()
    }

    pub(crate) fn params(&self) -> &BTreeMap<String, Value> {
        &self.params
    }

    /// Overrides a graph parameter; see [`crate::Session::set_param`].
    pub(crate) fn set_param(&mut self, name: &str, value: Value) -> Result<(), String> {
        let declared = self
            .graph
            .as_ref()
            .and_then(|graph| graph.params.iter().find(|p| p.name == name));
        let value = match (declared, &self.graph) {
            (Some(param), _) => value
                .shaped_like(&param.default)
                .ok_or_else(|| format!("param `{name}` is a {}; `{value}` is not one", param.ty))?,
            // A loaded scene takes from outside only the params it
            // declares: a node's own value is the document's, and changes
            // with it. Before a scene is loaded (a command line's params)
            // the value is held for the scene that will declare it.
            (None, Some(graph)) => return Err(undeclared(graph, name)),
            (None, None) => value,
        };
        let previous = self.params.insert(name.to_owned(), value.clone());
        // A parameter that sizes the graph (a node's `times`) changes its
        // structure: the document is lowered again on the next sync.
        if previous != Some(value)
            && self
                .graph
                .as_ref()
                .is_some_and(|graph| graph.times_params.iter().any(|p| p == name))
        {
            self.stamp = None;
        }
        Ok(())
    }

    /// Reads the root and every document it reaches that changed,
    /// handing their Kernels to `kernels` to publish. A library that
    /// changed may have changed a declaration the root's Kernels
    /// inherit, so the root is read again after the libraries.
    pub(crate) fn sync(
        &mut self,
        kernels: &mut Kernels,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) {
        self.sync_root(kernels, runtime, events);
        if self.sync_libraries(kernels, runtime, events) {
            self.stamp = None;
            self.sync_root(kernels, runtime, events);
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

    /// The root's text read and lowered, the documents it places
    /// loaded: what a sync publishes from, nothing published yet.
    fn lower(
        &self,
        text: &str,
    ) -> Result<(orla_language::Document, orla_scene::Lowered), Diagnostic> {
        let document = orla_language::parse_document(text).map_err(|e| Diagnostic::from(&e))?;
        let store = self.store.as_ref();
        let mut loader = |path: &Path| read_document(store, path);
        let mut declare =
            |base: &Path, uses: &[orla_language::Use]| declarations_for(store, base, uses);
        let lowered = orla_scene::load_with_params(
            &document,
            text,
            &self.document_name(),
            &self.base(),
            &mut loader,
            &mut declare,
            &self.params,
        )
        .map_err(|e| Diagnostic::from(&e))?;
        // A graph whose nodes read what is not there, or read each other
        // round, runs no frame: it is refused here, where the scene it
        // would replace is still running, not found at the next frame.
        lowered
            .graph
            .order()
            .map_err(|e| Diagnostic::new(e.to_string()))?;
        // Params set before any scene was loaded (a command line's) were
        // held for the scene to declare; one it does not declare would
        // reach nothing, and is refused rather than left unsaid.
        if self.graph.is_none()
            && let Some(name) = self
                .params
                .keys()
                .find(|name| !lowered.graph.params.iter().any(|p| &p.name == *name))
        {
            return Err(Diagnostic::new(undeclared(&lowered.graph, name)));
        }
        Ok((document, lowered))
    }

    /// Whether `text`, as the root, would be taken whole: it lowers to a
    /// graph whose every source is there, and every Kernel it defines
    /// itself compiles. Nothing is written,
    /// published or run; see [`crate::Session::check_scene`].
    pub(crate) fn check(&self, text: &str) -> Result<(), Diagnostic> {
        let (mut document, lowered) = self.lower(text)?;
        let (structs, types, functions) =
            declarations_for(self.store.as_ref(), &self.base(), &document.uses)
                .map_err(Diagnostic::new)?;
        document.inherit(&structs, &types, &functions);
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
            orla_compiler::compile_kernel(kernel, Some(&id))
                .map_err(|error| Diagnostic::from(&error))?;
        }
        Ok(())
    }

    fn sync_root(&mut self, kernels: &mut Kernels, runtime: &mut Runtime, events: &mut Vec<Event>) {
        let current = self.store.stamp(&self.root);
        let sub_graph_changed = self
            .sub_graphs
            .iter()
            .any(|(path, loaded)| self.store.stamp(path) != *loaded);
        if current == self.stamp && self.graph.is_some() && !sub_graph_changed {
            return;
        }
        self.stamp = current;
        let text = match self.store.read_text(&self.root) {
            Ok(text) => text,
            Err(error) => {
                events.push(Event::SceneRejected(Diagnostic::new(error.to_string())));
                return;
            }
        };
        let (document, lowered) = match self.lower(&text) {
            Ok(read) => read,
            Err(error) => {
                events.push(Event::SceneRejected(error));
                return;
            }
        };
        let name = self.document_name();
        let base = self.base();
        // Kernels defined by the root: its top-level ones and the inline
        // ones of its own nodes. Inline Kernels of sub-graphs belong to
        // their documents.
        let mut by_document: BTreeMap<PathBuf, Vec<Defined>> = BTreeMap::new();
        let root = self.root.clone();
        let root_key = PathBuf::from(&name);
        // The root's own top-level Kernels see the shared declarations too.
        let mut document = document;
        let mut declarations = String::new();
        match declarations_for(self.store.as_ref(), &base, &document.uses) {
            Ok((structs, types, functions)) => {
                declarations = declarations_key(&structs, &types, &functions);
                self.structs = structs.iter().chain(&document.structs).cloned().collect();
                self.types = types.iter().chain(&document.types).cloned().collect();
                self.functions = functions
                    .iter()
                    .chain(&document.functions)
                    .cloned()
                    .collect();
                document.inherit(&structs, &types, &functions);
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
                    text: text[kernel.range.clone()].to_owned(),
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
        // Documents that were graph documents before and are not any more
        // release their kernels.
        let gone: Vec<PathBuf> = self
            .sub_graphs
            .keys()
            .filter(|path| !sub_graphs.contains_key(*path))
            .cloned()
            .collect();
        for path in gone {
            kernels.retire_document(&path, runtime, events);
        }
        for (path, defined) in by_document {
            let current = if path == root {
                current
            } else {
                self.store.stamp(&path)
            };
            let key = if path == root {
                declarations.clone()
            } else {
                // A sub-graph document's inline Kernels inherit through
                // its own `use` items, resolved during lowering.
                let dir = path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .to_path_buf();
                self.store
                    .read_text(&path)
                    .ok()
                    .and_then(|text| orla_language::parse_document(&text).ok())
                    .and_then(|doc| declarations_for(self.store.as_ref(), &dir, &doc.uses).ok())
                    .map(|(s, t, f)| declarations_key(&s, &t, &f))
                    .unwrap_or_default()
            };
            kernels.publish_document(
                self.store.as_ref(),
                &path,
                current,
                defined,
                &key,
                &root,
                runtime,
                events,
            );
        }
        self.sub_graphs = sub_graphs;
        self.uses = lowered.uses;
        // A value held for a param the document no longer declares
        // reaches nothing now, and is let go with the declaration.
        let graph = lowered.graph;
        self.params
            .retain(|name, _| graph.params.iter().any(|p| &p.name == name));
        self.graph = Some(graph);
        self.text = Some(text);
        self.files = lowered.files;
        events.push(Event::SceneLoaded);
    }

    /// Loads the libraries the root uses; returns whether any changed.
    fn sync_libraries(
        &mut self,
        kernels: &mut Kernels,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) -> bool {
        let mut changed = false;
        let mut wanted: Vec<PathBuf> = Vec::new();
        for (path, trailing_slash) in &self.uses {
            wanted
                .extend(expand_use(self.store.as_ref(), path, *trailing_slash).unwrap_or_default());
        }
        for path in &self.extra_uses {
            wanted.extend(expand_use(self.store.as_ref(), path, false).unwrap_or_default());
        }
        wanted.retain(|path| *path != self.root && !self.sub_graphs.contains_key(path));
        wanted.sort();
        wanted.dedup();
        let stale: Vec<PathBuf> = kernels
            .documents()
            .into_iter()
            .filter(|path| {
                *path != self.root && !wanted.contains(path) && !self.sub_graphs.contains_key(path)
            })
            .collect();
        for path in stale {
            kernels.retire_document(&path, runtime, events);
        }
        for path in wanted {
            let current = self.store.stamp(&path);
            if kernels.read_at(&path, current) {
                continue;
            }
            changed = true;
            let text = match self.store.read_text(&path) {
                Ok(text) => text,
                Err(error) => {
                    events.push(Event::Rejected {
                        path: path.clone(),
                        error: Diagnostic::new(error.to_string()),
                    });
                    kernels.mark_read(path, current);
                    continue;
                }
            };
            let document = match orla_language::parse_document(&text) {
                Ok(document) => document,
                Err(error) => {
                    events.push(Event::Rejected {
                        path: path.clone(),
                        error: Diagnostic::from(&error),
                    });
                    kernels.mark_read(path, current);
                    continue;
                }
            };
            if !is_library(&document) {
                events.push(Event::Rejected {
                    path: path.clone(),
                    error: Diagnostic {
                        message: "a used document must hold kernels only".into(),
                        at: Some((1, 1)),
                    },
                });
                kernels.mark_read(path, current);
                continue;
            }
            // A library's own `use` items bring in shared declarations.
            let dir = path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf();
            let mut document = document;
            let declarations =
                match declarations_within(self.store.as_ref(), Some(&path), &dir, &document.uses) {
                    Ok((structs, types, functions)) => {
                        document.inherit(&structs, &types, &functions);
                        declarations_key(&structs, &types, &functions)
                    }
                    Err(error) => {
                        events.push(Event::Rejected {
                            path: path.clone(),
                            error: Diagnostic::new(error),
                        });
                        kernels.mark_read(path, current);
                        continue;
                    }
                };
            let mut defined = Vec::new();
            for kernel in &document.kernels {
                match named_id(kernel) {
                    Ok(id) => defined.push(Defined {
                        id,
                        text: text[kernel.range.clone()].to_owned(),
                        kernel: kernel.clone(),
                    }),
                    Err(message) => events.push(Event::Rejected {
                        path: path.clone(),
                        error: message,
                    }),
                }
            }
            let root = self.root.clone();
            kernels.publish_document(
                self.store.as_ref(),
                &path,
                current,
                defined,
                &declarations,
                &root,
                runtime,
                events,
            );
        }
        changed
    }

    /// Prepares `source`, a document holding one Kernel, without
    /// publishing it; see [`crate::Session::prepare_source`].
    pub(crate) fn prepare_source(
        &self,
        source: &str,
        kernels: &Kernels,
        runtime: &Runtime,
    ) -> Result<orla_runtime::Candidate, Diagnostic> {
        let mut document =
            orla_language::parse_document(source).map_err(|e| Diagnostic::from(&e))?;
        if document.kernels.len() != 1 {
            return Err(Diagnostic::new(format!(
                "expected one kernel, found {}",
                document.kernels.len()
            )));
        }
        let base = self.root.parent().unwrap_or_else(|| Path::new("."));
        let (mut structs, mut types, mut functions) = if document.uses.is_empty() {
            (Vec::new(), Vec::new(), Vec::new())
        } else {
            declarations_for(self.store.as_ref(), base, &document.uses).map_err(Diagnostic::new)?
        };
        // The scene's shared declarations come after the source's own,
        // and a name the source declares, at its top or inside its
        // Kernel, is the source's: a Kernel may redefine a shared struct,
        // which is how a state's layout changes while it runs.
        for shared in &self.structs {
            let declared = |d: &orla_language::StructDecl| d.name.value == shared.name.value;
            if !document.structs.iter().any(declared)
                && !structs.iter().any(declared)
                && !document.kernels[0].structs.iter().any(declared)
            {
                structs.push(shared.clone());
            }
        }
        for shared in &self.types {
            let declared = |d: &orla_language::TypeAlias| d.name.value == shared.name.value;
            if !document.types.iter().any(declared)
                && !types.iter().any(declared)
                && !document.kernels[0].types.iter().any(declared)
            {
                types.push(shared.clone());
            }
        }
        for shared in &self.functions {
            let declared = |d: &orla_language::FnDecl| d.name.value == shared.name.value;
            if !document.functions.iter().any(declared)
                && !functions.iter().any(declared)
                && !document.kernels[0].functions.iter().any(declared)
            {
                functions.push(shared.clone());
            }
        }
        document.inherit(&structs, &types, &functions);
        let module = orla_compiler::compile_kernel(&document.kernels[0], None).map_err(|e| {
            kernels.located(
                self.store.as_ref(),
                Diagnostic::from(&e),
                &self.root,
                &self.root,
            )
        })?;
        runtime
            .prepare_module(module, source)
            .map_err(|e| Diagnostic::new(e.to_string()))
    }
}

/// Why `name` is refused as a param of `graph`, which does not declare
/// it: a declared param named nearly so, or the nodes that set a value of
/// that name themselves, which change as the document does.
fn undeclared(graph: &Graph, name: &str) -> String {
    let mut why = format!("`{name}` is not a param the scene declares");
    let near = orla_language::nearest(name, graph.params.iter().map(|p| p.name.as_str()));
    if let Some(near) = near {
        why.push_str(&format!("; did you mean `{near}`?"));
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
