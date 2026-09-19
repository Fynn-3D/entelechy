//! A live session over a scene file and a directory of Kernel sources.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use orla_graph::{EvaluateError, Evaluator, Graph, Report};
use orla_runtime::{Backend, Expect, Generation, ImageHandle, KernelId, Runtime};

use crate::ppm;

/// Default pixel budget for a session's Host.
pub const DEFAULT_BUDGET_PIXELS: usize = 64 * 1024 * 1024;

/// Something the session observed while synchronizing with the filesystem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// A Kernel file was compiled and published.
    Published {
        /// Source path.
        path: PathBuf,
        /// Kernel id.
        id: KernelId,
        /// New generation.
        generation: Generation,
    },
    /// A Kernel file changed but did not compile; the previous version stays.
    Rejected {
        /// Source path.
        path: PathBuf,
        /// Diagnostic.
        error: String,
    },
    /// A Kernel file disappeared or now defines a different id.
    Retired {
        /// Source path.
        path: PathBuf,
        /// Kernel id that left the registry.
        id: KernelId,
    },
    /// The scene file was re-read.
    SceneLoaded,
    /// The scene file changed but does not parse; the previous graph stays.
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
            Self::Rejected { path, error } => {
                write!(f, "rejected {}: {error}", path.display())
            }
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

#[derive(Debug)]
struct Tracked {
    stamp: Stamp,
    id: Option<KernelId>,
}

/// A scene, its Kernel directory, and the runtime state built from them.
pub struct Session {
    scene_path: PathBuf,
    kernels_dir: PathBuf,
    runtime: Runtime,
    evaluator: Evaluator,
    graph: Option<Graph>,
    scene_stamp: Option<Stamp>,
    kernels: BTreeMap<PathBuf, Tracked>,
    externals: BTreeMap<String, ImageHandle>,
}

impl Session {
    /// Creates a CPU-backed session; call [`Session::sync`] to load files.
    pub fn open(scene: &Path, kernels_dir: Option<&Path>, budget_pixels: usize) -> Self {
        Self::open_with(scene, kernels_dir, Runtime::new(budget_pixels))
    }

    /// Creates a session over a runtime built by the caller, for choosing
    /// the backend.
    pub fn open_with(scene: &Path, kernels_dir: Option<&Path>, runtime: Runtime) -> Self {
        let kernels_dir = kernels_dir.map(Path::to_path_buf).unwrap_or_else(|| {
            scene
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("kernels")
        });
        Self {
            scene_path: scene.to_path_buf(),
            kernels_dir,
            runtime,
            evaluator: Evaluator::new(),
            graph: None,
            scene_stamp: None,
            kernels: BTreeMap::new(),
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

    /// The Kernel directory being watched.
    pub fn kernels_dir(&self) -> &Path {
        &self.kernels_dir
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
        self.sync_kernels(&mut events);
        self.sync_scene(&mut events);
        events
    }

    fn sync_kernels(&mut self, events: &mut Vec<Event>) {
        let mut present: Vec<PathBuf> = std::fs::read_dir(&self.kernels_dir)
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .map(|entry| entry.path())
                    .filter(|path| path.extension().is_some_and(|ext| ext == "orla"))
                    .collect()
            })
            .unwrap_or_default();
        present.sort();
        let removed: Vec<PathBuf> = self
            .kernels
            .keys()
            .filter(|path| !present.contains(path))
            .cloned()
            .collect();
        for path in removed {
            if let Some(Tracked { id: Some(id), .. }) = self.kernels.remove(&path) {
                self.retire(&path, &id, events);
            }
        }
        for path in present {
            let Some(current) = stamp(&path) else {
                continue;
            };
            if self
                .kernels
                .get(&path)
                .is_some_and(|tracked| tracked.stamp == current)
            {
                continue;
            }
            let previous_id = self
                .kernels
                .get(&path)
                .and_then(|tracked| tracked.id.clone());
            let published = match std::fs::read_to_string(&path) {
                Ok(source) => self.publish(&path, &source, previous_id.as_ref(), events),
                Err(error) => {
                    events.push(Event::Rejected {
                        path: path.clone(),
                        error: error.to_string(),
                    });
                    None
                }
            };
            let id = match (published, previous_id) {
                (Some(id), Some(previous)) if id != previous => {
                    self.retire(&path, &previous, events);
                    Some(id)
                }
                (Some(id), _) => Some(id),
                (None, previous) => previous,
            };
            self.kernels.insert(path, Tracked { stamp: current, id });
        }
    }

    fn publish(
        &mut self,
        path: &Path,
        source: &str,
        previous: Option<&KernelId>,
        events: &mut Vec<Event>,
    ) -> Option<KernelId> {
        let candidate = match self.runtime.prepare(source) {
            Ok(candidate) => candidate,
            Err(error) => {
                events.push(Event::Rejected {
                    path: path.to_path_buf(),
                    error: error.to_string(),
                });
                return None;
            }
        };
        let id = candidate.id().clone();
        let expect = match self.runtime.kernel(&id) {
            Some(current) if previous == Some(&id) => Expect::Generation(current.generation()),
            Some(current) => {
                let owner = self
                    .kernels
                    .iter()
                    .find(|(_, tracked)| tracked.id.as_ref() == Some(&id))
                    .map(|(path, _)| path.display().to_string());
                events.push(Event::Rejected {
                    path: path.to_path_buf(),
                    error: format!(
                        "{id} is already published as {} by {}",
                        current.generation(),
                        owner.unwrap_or_else(|| "another source".into())
                    ),
                });
                return None;
            }
            None => Expect::Absent,
        };
        match self.runtime.publish(candidate, expect) {
            Ok(version) => {
                events.push(Event::Published {
                    path: path.to_path_buf(),
                    id: id.clone(),
                    generation: version.generation(),
                });
                Some(id)
            }
            Err(error) => {
                events.push(Event::Rejected {
                    path: path.to_path_buf(),
                    error: error.to_string(),
                });
                None
            }
        }
    }

    fn retire(&mut self, path: &Path, id: &KernelId, events: &mut Vec<Event>) {
        if self.runtime.retire(id).is_some() {
            events.push(Event::Retired {
                path: path.to_path_buf(),
                id: id.clone(),
            });
        }
    }

    fn sync_scene(&mut self, events: &mut Vec<Event>) {
        let current = stamp(&self.scene_path);
        if current == self.scene_stamp && self.graph.is_some() {
            return;
        }
        self.scene_stamp = current;
        match std::fs::read_to_string(&self.scene_path) {
            Ok(text) => match Graph::parse(&text) {
                Ok(graph) => {
                    self.graph = Some(graph);
                    events.push(Event::SceneLoaded);
                }
                Err(error) => events.push(Event::SceneRejected(error.to_string())),
            },
            Err(error) => events.push(Event::SceneRejected(error.to_string())),
        }
    }

    /// Evaluates the current graph at session time `time` (seconds).
    /// Unchanged nodes are reused.
    pub fn evaluate(&mut self, time: f32) -> Result<Report, EvaluateError> {
        let graph = self.graph.as_ref().ok_or_else(|| {
            EvaluateError::Graph(orla_graph::GraphError::UnknownOutput(orla_graph::NodeId(
                "<no scene>".into(),
            )))
        })?;
        self.evaluator
            .evaluate(graph, &mut self.runtime, &self.externals, time)
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
