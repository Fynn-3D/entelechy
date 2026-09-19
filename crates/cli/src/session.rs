//! A live session over one `.osn` document and the libraries it uses.
//!
//! The document is the graph: the session parses it, publishes its inline
//! Kernels, follows its `use` items to `.orla` libraries, and builds the
//! Graph. On every [`Session::sync`] it re-reads what changed on disk and
//! republishes only the Kernels whose text changed, keeping a running
//! version whenever a replacement is rejected. Source resolution lives here,
//! not in the runtimes.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use orla_graph::{EvaluateError, Evaluator, Graph, Report};
use orla_language::{Document, Kernel};
use orla_runtime::{Backend, Context, Expect, Generation, ImageHandle, KernelId, Runtime};

use crate::ppm;

/// Default pixel budget for a session's Host.
pub const DEFAULT_BUDGET_PIXELS: usize = 64 * 1024 * 1024;

/// Something the session observed while synchronizing with the filesystem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
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
    /// The scene document changed but does not parse; the previous graph stays.
    SceneRejected(String),
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Published {
                path,
                id,
                generation,
            } => write!(f, "published {id} {generation} from {}", path.display()),
            Self::Rejected { path, error } => write!(f, "rejected {}:{error}", path.display()),
            Self::Retired { path, id } => write!(f, "retired {id} ({})", path.display()),
            Self::SceneLoaded => f.write_str("scene loaded"),
            Self::SceneRejected(error) => write!(f, "scene rejected: {error}"),
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
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(path, error) => write!(f, "{}: {error}", path.display()),
            Self::Image(path, error) => write!(f, "{}: {error}", path.display()),
            Self::NoScene => f.write_str("no scene has been loaded"),
            Self::Host(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for SessionError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stamp {
    modified: Option<SystemTime>,
    len: u64,
}

fn stamp(path: &Path) -> Option<Stamp> {
    let metadata = std::fs::metadata(path).ok()?;
    Some(Stamp {
        modified: metadata.modified().ok(),
        len: metadata.len(),
    })
}

/// A `use` target: a file, or every `.orla` directly inside a directory.
fn expand_use(path: &Path, trailing_slash: bool) -> Vec<PathBuf> {
    if trailing_slash || path.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(path)
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .map(|entry| entry.path())
                    .filter(|path| path.extension().is_some_and(|ext| ext == "orla"))
                    .collect()
            })
            .unwrap_or_default();
        files.sort();
        files
    } else {
        vec![path.to_path_buf()]
    }
}

/// Kernels a document defines: id, source text, and the parsed Kernel.
struct Defined<'a> {
    id: KernelId,
    text: String,
    kernel: &'a Kernel,
}

/// A scene, the libraries it uses, and the runtime state built from them.
pub struct Session {
    root: PathBuf,
    extra_uses: Vec<PathBuf>,
    runtime: Runtime,
    evaluator: Evaluator,
    graph: Option<Graph>,
    root_stamp: Option<Stamp>,
    /// `use` targets from the last good parse of the root, resolved.
    root_uses: Vec<(PathBuf, bool)>,
    /// Kernel text by id for every document that published Kernels; the
    /// root document is one of them.
    documents: BTreeMap<PathBuf, DocumentState>,
    owners: BTreeMap<KernelId, PathBuf>,
    externals: BTreeMap<String, ImageHandle>,
}

#[derive(Default)]
struct DocumentState {
    stamp: Option<Stamp>,
    kernels: BTreeMap<KernelId, String>,
}

impl Session {
    /// Creates a CPU-backed session; call [`Session::sync`] to load files.
    /// `library` is an extra directory or file of Kernels to use.
    pub fn open(scene: &Path, library: Option<&Path>, budget_pixels: usize) -> Self {
        Self::open_with(scene, library, Runtime::new(budget_pixels))
    }

    /// Creates a session over a runtime built by the caller, for choosing
    /// the backend.
    pub fn open_with(scene: &Path, library: Option<&Path>, runtime: Runtime) -> Self {
        Self {
            root: scene.to_path_buf(),
            extra_uses: library.map(Path::to_path_buf).into_iter().collect(),
            runtime,
            evaluator: Evaluator::new(),
            graph: None,
            root_stamp: None,
            root_uses: Vec::new(),
            documents: BTreeMap::new(),
            owners: BTreeMap::new(),
            externals: BTreeMap::new(),
        }
    }

    /// The runtime, for inspection.
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Builds a runtime over a named backend: `cpu`, or `gpu` when a device
    /// is available.
    pub fn runtime_for(backend: &str, budget_pixels: usize) -> Result<Runtime, String> {
        match backend {
            "cpu" => Ok(Runtime::new(budget_pixels)),
            "gpu" => {
                let gpu = orla_gpu::GpuBackend::new().map_err(|error| error.to_string())?;
                let backend: Box<dyn Backend> = Box::new(gpu);
                Ok(Runtime::with_backend(budget_pixels, backend))
            }
            other => Err(format!("unknown backend `{other}`; use cpu or gpu")),
        }
    }

    /// The current graph, if the scene loaded.
    pub fn graph(&self) -> Option<&Graph> {
        self.graph.as_ref()
    }

    /// The scene document.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Every document the session currently watches.
    pub fn watched(&self) -> Vec<PathBuf> {
        self.documents.keys().cloned().collect()
    }

    /// Imports a PPM file as the external image `name`.
    pub fn bind_external(&mut self, name: &str, path: &Path) -> Result<(), SessionError> {
        let image = ppm::read(path).map_err(|e| SessionError::Image(path.to_path_buf(), e))?;
        let handle = self
            .runtime
            .host_mut()
            .adopt_image(image)
            .map_err(SessionError::Host)?;
        if let Some(previous) = self.externals.insert(name.to_owned(), handle) {
            let _ = self.runtime.host_mut().release_image(previous);
        }
        Ok(())
    }

    /// Reconciles the runtime with the files on disk and reports what changed.
    pub fn sync(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        self.sync_root(&mut events);
        self.sync_libraries(&mut events);
        events
    }

    fn document_name(&self) -> String {
        self.root
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("scene")
            .to_owned()
    }

    fn sync_root(&mut self, events: &mut Vec<Event>) {
        let current = stamp(&self.root);
        if current == self.root_stamp && self.graph.is_some() {
            return;
        }
        self.root_stamp = current;
        let text = match std::fs::read_to_string(&self.root) {
            Ok(text) => text,
            Err(error) => {
                events.push(Event::SceneRejected(error.to_string()));
                return;
            }
        };
        let document = match orla_language::parse_document(&text) {
            Ok(document) => document,
            Err(error) => {
                events.push(Event::SceneRejected(error.to_string()));
                return;
            }
        };
        let name = self.document_name();
        let lowered = match Graph::from_document(&document, &name) {
            Ok(lowered) => lowered,
            Err(error) => {
                events.push(Event::SceneRejected(error.to_string()));
                return;
            }
        };
        let mut defined = Vec::new();
        for kernel in &document.kernels {
            match named_id(kernel) {
                Ok(id) => defined.push(Defined {
                    id,
                    text: text[kernel.range.clone()].to_owned(),
                    kernel,
                }),
                Err(message) => events.push(Event::Rejected {
                    path: self.root.clone(),
                    error: message,
                }),
            }
        }
        for inline in &lowered.inline {
            defined.push(Defined {
                id: inline.id.clone(),
                text: text[inline.kernel.range.clone()].to_owned(),
                kernel: inline.kernel,
            });
        }
        let root = self.root.clone();
        self.publish_document(&root, current, defined, events);
        let base = self.root.parent().unwrap_or_else(|| Path::new("."));
        self.root_uses = document
            .uses
            .iter()
            .map(|item| (base.join(&item.path.value), item.path.value.ends_with('/')))
            .collect();
        self.graph = Some(lowered.graph);
        events.push(Event::SceneLoaded);
    }

    fn sync_libraries(&mut self, events: &mut Vec<Event>) {
        let mut wanted: Vec<PathBuf> = Vec::new();
        for (path, trailing_slash) in &self.root_uses {
            wanted.extend(expand_use(path, *trailing_slash));
        }
        for path in &self.extra_uses {
            wanted.extend(expand_use(path, false));
        }
        wanted.retain(|path| *path != self.root);
        wanted.dedup();
        let stale: Vec<PathBuf> = self
            .documents
            .keys()
            .filter(|path| **path != self.root && !wanted.contains(path))
            .cloned()
            .collect();
        for path in stale {
            self.retire_document(&path, events);
        }
        for path in wanted {
            let current = stamp(&path);
            if self
                .documents
                .get(&path)
                .is_some_and(|state| state.stamp == current)
            {
                continue;
            }
            let text = match std::fs::read_to_string(&path) {
                Ok(text) => text,
                Err(error) => {
                    events.push(Event::Rejected {
                        path: path.clone(),
                        error: format!(" {error}"),
                    });
                    self.documents.entry(path).or_default().stamp = current;
                    continue;
                }
            };
            let document = match orla_language::parse_document(&text) {
                Ok(document) => document,
                Err(error) => {
                    events.push(Event::Rejected {
                        path: path.clone(),
                        error: error.to_string(),
                    });
                    self.documents.entry(path).or_default().stamp = current;
                    continue;
                }
            };
            if !is_library(&document) {
                events.push(Event::Rejected {
                    path: path.clone(),
                    error: "1:1: a used document must hold kernels only".into(),
                });
                self.documents.entry(path).or_default().stamp = current;
                continue;
            }
            let mut defined = Vec::new();
            for kernel in &document.kernels {
                match named_id(kernel) {
                    Ok(id) => defined.push(Defined {
                        id,
                        text: text[kernel.range.clone()].to_owned(),
                        kernel,
                    }),
                    Err(message) => events.push(Event::Rejected {
                        path: path.clone(),
                        error: message,
                    }),
                }
            }
            self.publish_document(&path, current, defined, events);
        }
    }

    /// Publishes the Kernels a document now defines: retires the ones that
    /// left, keeps the unchanged, republishes the changed, and reports each.
    fn publish_document(
        &mut self,
        path: &Path,
        current: Option<Stamp>,
        defined: Vec<Defined<'_>>,
        events: &mut Vec<Event>,
    ) {
        let previous = self
            .documents
            .remove(path)
            .map(|state| state.kernels)
            .unwrap_or_default();
        let mut kernels = BTreeMap::new();
        for id in previous.keys() {
            if !defined.iter().any(|d| &d.id == id) {
                self.retire(path, id, events);
            }
        }
        for Defined { id, text, kernel } in defined {
            if let Some(owner) = self.owners.get(&id)
                && owner != path
            {
                events.push(Event::Rejected {
                    path: path.to_path_buf(),
                    error: format!(
                        " {id} is already published by {}; kernel ids are unique",
                        owner.display()
                    ),
                });
                continue;
            }
            if previous.get(&id) == Some(&text) && self.runtime.kernel(&id).is_some() {
                kernels.insert(id, text);
                continue;
            }
            match self.runtime.prepare_kernel(kernel, Some(&id), &text) {
                Ok(candidate) => {
                    let expect = match self.runtime.kernel(&id) {
                        Some(version) => Expect::Generation(version.generation()),
                        None => Expect::Absent,
                    };
                    match self.runtime.publish(candidate, expect) {
                        Ok(version) => {
                            events.push(Event::Published {
                                path: path.to_path_buf(),
                                id: id.clone(),
                                generation: version.generation(),
                            });
                            self.owners.insert(id.clone(), path.to_path_buf());
                        }
                        Err(error) => events.push(Event::Rejected {
                            path: path.to_path_buf(),
                            error: format!(" {error}"),
                        }),
                    }
                }
                Err(error) => events.push(Event::Rejected {
                    path: path.to_path_buf(),
                    error: error.to_string(),
                }),
            }
            kernels.insert(id, text);
        }
        self.documents.insert(
            path.to_path_buf(),
            DocumentState {
                stamp: current,
                kernels,
            },
        );
    }

    fn retire_document(&mut self, path: &Path, events: &mut Vec<Event>) {
        if let Some(state) = self.documents.remove(path) {
            for id in state.kernels.keys() {
                self.retire(path, id, events);
            }
        }
    }

    fn retire(&mut self, path: &Path, id: &KernelId, events: &mut Vec<Event>) {
        if self.owners.get(id).is_some_and(|owner| owner == path) {
            self.owners.remove(id);
            if self.runtime.retire(id).is_some() {
                events.push(Event::Retired {
                    path: path.to_path_buf(),
                    id: id.clone(),
                });
            }
        }
    }

    /// Evaluates the current graph under `context`. Unchanged nodes are
    /// reused; state commits only when the whole evaluation succeeds.
    pub fn evaluate(&mut self, context: &Context) -> Result<Report, EvaluateError> {
        let graph = self.graph.as_ref().ok_or_else(|| {
            EvaluateError::Graph(orla_graph::GraphError::UnknownOutput(
                orla_graph::Source::External("<no scene>".into()),
            ))
        })?;
        self.evaluator
            .evaluate(graph, &mut self.runtime, &self.externals, context)
    }

    /// Forgets committed state; `<~` bindings restart from their initial
    /// sources on the next evaluation.
    pub fn reset_state(&mut self) {
        self.evaluator.reset_state(&mut self.runtime);
    }

    /// Writes a cached image to a PPM file.
    pub fn export(&self, image: ImageHandle, path: &Path) -> Result<(), SessionError> {
        let image = self
            .runtime
            .host()
            .image(image)
            .map_err(SessionError::Host)?;
        ppm::write(path, image).map_err(|e| SessionError::Image(path.to_path_buf(), e))
    }
}

fn named_id(kernel: &Kernel) -> Result<KernelId, String> {
    match &kernel.id {
        Some(id) => id
            .value
            .parse::<KernelId>()
            .map_err(|error| format!("{}: {error}", id.span)),
        None => Err("1:1: a kernel outside a node needs an id".into()),
    }
}

fn is_library(document: &Document) -> bool {
    document.nodes.is_empty()
        && document.inputs.is_empty()
        && document.outputs.is_empty()
        && document.uses.is_empty()
}
