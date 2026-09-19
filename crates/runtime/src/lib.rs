#![warn(missing_docs)]
//! Kernel Runtime (L4).
//!
//! The runtime owns the Host, one backend, and the registry of published
//! Kernel versions. A version is immutable once published. Replacing a Kernel
//! is a two-step act: [`Runtime::prepare`] compiles, validates, and lets the
//! backend build whatever it needs (a shader pipeline, say) into a
//! [`Candidate`] without touching the registry; [`Runtime::publish`] swaps it
//! in atomically, and only if the caller's expectation about the current
//! generation still holds. A failed preparation leaves the running Kernel
//! untouched. Callers that hold an `Arc<KernelVersion>` keep executing the
//! version they pinned, whatever the registry does afterwards.
//!
//! Preparation is the only slow step (a device compiles a pipeline), so it
//! can leave the calling thread: [`Runtime::prepare_later`] hands a module
//! to a worker and returns a [`Ticket`]; [`Runtime::poll`] collects the
//! [`Candidate`] when it is done. Whatever was running keeps running in the
//! meantime. Without a worker the same calls prepare inline, so tools that
//! want determinism get it by never enabling one.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

pub use orla_abi::{
    Context, Domain, Host, HostError, Image, ImageHandle, InputState, Invocation, Pixel, Scope,
    Shape, Value,
};
pub use orla_compiler::CompileError;
pub use orla_cpu::ExecutionError;
pub use orla_ir::{KernelId, Manifest, Validated};

/// Backend-owned state built from a module at preparation time.
pub trait Prepared: Any + Send + Sync {
    /// For downcasting by the backend that built it.
    fn as_any(&self) -> &dyn Any;
}

/// An execution backend. `prepare` does all expensive work so that
/// publishing is a pointer swap.
pub trait Backend: Send + Sync {
    /// Short backend name for diagnostics, e.g. `cpu`.
    fn name(&self) -> &'static str;

    /// For interfaces that know a specific backend, such as a window that
    /// presents a resident image directly.
    fn as_any(&self) -> &dyn Any;

    /// Builds backend state for a validated module.
    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String>;

    /// Executes a prepared module over the invocation's domain, returning one
    /// image handle per manifest output, owned by the scope's Host. Outputs
    /// may be left resident on the backend's device.
    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<ImageHandle>, ExecutionError>;

    /// Downloads the bytes of a resource this backend left resident:
    /// `byte_len` bytes of little-endian words. The default is for backends
    /// that never do.
    fn download(
        &self,
        device: &(dyn Any + Send + Sync),
        byte_len: usize,
    ) -> Result<Vec<u8>, String> {
        let _ = (device, byte_len);
        Err("this backend keeps no resident resources".into())
    }
}

struct NothingPrepared;

impl Prepared for NothingPrepared {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Backend for orla_cpu::CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn prepare(&self, _module: &Validated) -> Result<Box<dyn Prepared>, String> {
        Ok(Box::new(NothingPrepared))
    }

    fn execute(
        &self,
        _prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<ImageHandle>, ExecutionError> {
        orla_cpu::CpuBackend::execute(self, module, invocation, scope)
    }
}

/// Monotonic counter identifying one published version. Generations are
/// unique across all Kernels in a runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Generation(pub u64);

impl fmt::Display for Generation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "g{}", self.0)
    }
}

/// An immutable published Kernel version.
pub struct KernelVersion {
    id: KernelId,
    generation: Generation,
    module: Validated,
    source: String,
    prepared: Box<dyn Prepared>,
}

impl fmt::Debug for KernelVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KernelVersion({} {})", self.id, self.generation)
    }
}

impl KernelVersion {
    /// The Kernel id.
    pub fn id(&self) -> &KernelId {
        &self.id
    }

    /// The generation this version was published as.
    pub fn generation(&self) -> Generation {
        self.generation
    }

    /// The manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.module.manifest
    }

    /// The validated IR.
    pub fn module(&self) -> &Validated {
        &self.module
    }

    /// The source this version was compiled from.
    pub fn source(&self) -> &str {
        &self.source
    }
}

/// A compiled, validated, and backend-prepared Kernel that has not been
/// published.
pub struct Candidate {
    module: Validated,
    source: String,
    prepared: Box<dyn Prepared>,
}

impl fmt::Debug for Candidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Candidate({})", self.id())
    }
}

impl Candidate {
    /// The id the candidate will publish as.
    pub fn id(&self) -> &KernelId {
        &self.module.manifest.id
    }

    /// The manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.module.manifest
    }

    /// The validated IR.
    pub fn module(&self) -> &Validated {
        &self.module
    }
}

/// Why a candidate could not be prepared.
#[derive(Debug)]
pub enum PrepareError {
    /// The source did not compile.
    Compile(CompileError),
    /// The backend rejected the module; the message is backend specific.
    Backend(String),
}

impl fmt::Display for PrepareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compile(error) => write!(f, "{error}"),
            Self::Backend(message) => write!(f, "backend rejected the kernel: {message}"),
        }
    }
}

impl std::error::Error for PrepareError {}

/// What the publisher believes the registry currently holds for the id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expect {
    /// No version is published yet.
    Absent,
    /// Exactly this generation is current.
    Generation(Generation),
    /// Publish regardless of the current state.
    Any,
}

/// The registry did not match the publisher's expectation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaleCandidate {
    /// The Kernel id.
    pub id: KernelId,
    /// What the publisher expected.
    pub expected: Expect,
    /// What the registry holds.
    pub actual: Option<Generation>,
}

impl fmt::Display for StaleCandidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let actual = match self.actual {
            Some(generation) => generation.to_string(),
            None => "nothing".to_owned(),
        };
        match self.expected {
            Expect::Absent => write!(
                f,
                "cannot publish {} as new: {actual} is already published",
                self.id
            ),
            Expect::Generation(generation) => write!(
                f,
                "cannot replace {} {generation}: the registry holds {actual}",
                self.id
            ),
            Expect::Any => write!(f, "cannot publish {}", self.id),
        }
    }
}

impl std::error::Error for StaleCandidate {}

/// Errors from the combined prepare-and-publish path.
#[derive(Debug)]
pub enum LoadError {
    /// The candidate could not be prepared.
    Prepare(PrepareError),
    /// The registry state did not match.
    Stale(StaleCandidate),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Prepare(error) => write!(f, "{error}"),
            Self::Stale(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// A preparation handed to the worker; redeemed with [`Runtime::poll`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ticket(u64);

impl fmt::Display for Ticket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t{}", self.0)
    }
}

struct Job {
    ticket: Ticket,
    module: Validated,
    source: String,
}

struct Done {
    ticket: Ticket,
    result: Result<Candidate, PrepareError>,
}

/// The thread that prepares modules while frames keep going.
struct Worker {
    jobs: Sender<Job>,
    done: Receiver<Done>,
    handle: JoinHandle<()>,
}

fn prepare_on(
    backend: &dyn Backend,
    module: Validated,
    source: String,
) -> Result<Candidate, PrepareError> {
    let prepared = backend.prepare(&module).map_err(PrepareError::Backend)?;
    Ok(Candidate {
        module,
        source,
        prepared,
    })
}

impl Drop for Runtime {
    fn drop(&mut self) {
        self.stop_worker();
    }
}

/// The Kernel Runtime.
pub struct Runtime {
    host: Host,
    backend: Arc<dyn Backend>,
    kernels: BTreeMap<KernelId, Arc<KernelVersion>>,
    next_generation: u64,
    worker: Option<Worker>,
    next_ticket: u64,
    ready: BTreeMap<Ticket, Result<Candidate, PrepareError>>,
    discarded: BTreeSet<Ticket>,
}

impl Runtime {
    /// Creates a CPU-backed runtime whose Host may hold at most
    /// `budget_words`.
    pub fn new(budget_words: usize) -> Self {
        Self::with_backend(budget_words, Box::new(orla_cpu::CpuBackend))
    }

    /// Creates a runtime over the given backend.
    pub fn with_backend(budget_words: usize, backend: Box<dyn Backend>) -> Self {
        Self {
            host: Host::new(budget_words),
            backend: Arc::from(backend),
            kernels: BTreeMap::new(),
            next_generation: 1,
            worker: None,
            next_ticket: 1,
            ready: BTreeMap::new(),
            discarded: BTreeSet::new(),
        }
    }

    /// Starts or stops the preparation worker. With a worker,
    /// [`Runtime::prepare_later`] returns before the backend has built
    /// anything; without one it prepares inline. Stopping waits for the
    /// jobs in flight and keeps their results for [`Runtime::poll`].
    pub fn set_background(&mut self, enabled: bool) {
        if enabled == self.worker.is_some() {
            return;
        }
        if !enabled {
            self.stop_worker();
            return;
        }
        let (jobs, inbox) = channel::<Job>();
        let (outbox, done) = channel::<Done>();
        let backend = Arc::clone(&self.backend);
        let handle = std::thread::Builder::new()
            .name("orla-prepare".into())
            .spawn(move || {
                for job in inbox {
                    let result = prepare_on(backend.as_ref(), job.module, job.source);
                    if outbox
                        .send(Done {
                            ticket: job.ticket,
                            result,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .expect("the preparation thread can be spawned");
        self.worker = Some(Worker { jobs, done, handle });
    }

    /// Whether a preparation worker is running.
    pub fn background(&self) -> bool {
        self.worker.is_some()
    }

    fn stop_worker(&mut self) {
        if let Some(Worker { jobs, done, handle }) = self.worker.take() {
            drop(jobs);
            let _ = handle.join();
            for finished in done.try_iter() {
                self.file(finished);
            }
        }
    }

    fn file(&mut self, finished: Done) {
        if !self.discarded.remove(&finished.ticket) {
            self.ready.insert(finished.ticket, finished.result);
        }
    }

    fn collect(&mut self) {
        let finished: Vec<Done> = match &self.worker {
            Some(worker) => worker.done.try_iter().collect(),
            None => Vec::new(),
        };
        for done in finished {
            self.file(done);
        }
    }

    /// Hands a validated module to the worker, or prepares it now when
    /// there is none. `source` is kept for inspection.
    pub fn prepare_later(&mut self, module: Validated, source: String) -> Ticket {
        let ticket = Ticket(self.next_ticket);
        self.next_ticket += 1;
        match &self.worker {
            Some(worker) => {
                if let Err(unsent) = worker.jobs.send(Job {
                    ticket,
                    module,
                    source,
                }) {
                    let Job { module, source, .. } = unsent.0;
                    let result = prepare_on(self.backend.as_ref(), module, source);
                    self.ready.insert(ticket, result);
                }
            }
            None => {
                let result = prepare_on(self.backend.as_ref(), module, source);
                self.ready.insert(ticket, result);
            }
        }
        ticket
    }

    /// Takes a finished preparation, or `None` while it is still running.
    /// A ticket is redeemed once.
    pub fn poll(&mut self, ticket: Ticket) -> Option<Result<Candidate, PrepareError>> {
        self.collect();
        self.ready.remove(&ticket)
    }

    /// Forgets a preparation whose result is no longer wanted.
    pub fn discard(&mut self, ticket: Ticket) {
        self.collect();
        if self.ready.remove(&ticket).is_none() {
            self.discarded.insert(ticket);
        }
    }

    /// Turns a candidate into a version that is never registered, under
    /// the given id: it executes like any pinned version but does not
    /// appear in the registry.
    pub fn detach(&mut self, candidate: Candidate, id: KernelId) -> Arc<KernelVersion> {
        let generation = Generation(self.next_generation);
        self.next_generation += 1;
        Arc::new(KernelVersion {
            id,
            generation,
            module: candidate.module,
            source: candidate.source,
            prepared: candidate.prepared,
        })
    }

    /// Name of the backend executing Kernels.
    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    /// The backend executing Kernels.
    pub fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }

    /// The Host, for interfaces that import and export resources.
    pub fn host(&self) -> &Host {
        &self.host
    }

    /// Brings a resident Field back into Host memory through the backend
    /// that owns it; a no-op for Fields already in memory.
    pub fn materialize(&mut self, handle: ImageHandle) -> Result<(), ExecutionError> {
        let Some(device) = self
            .host
            .resident(handle)
            .map_err(|e| ExecutionError::Abi(e.into()))?
        else {
            return Ok(());
        };
        let domain = self
            .host
            .domain(handle)
            .map_err(|e| ExecutionError::Abi(e.into()))?;
        let words_per_element = self
            .host
            .schema(handle)
            .map_err(|e| ExecutionError::Abi(e.into()))?
            .words();
        let byte_len = domain.count() * words_per_element * 4;
        let bytes = self
            .backend
            .download(device, byte_len)
            .map_err(ExecutionError::Device)?;
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        self.host
            .materialize(handle, words)
            .map_err(|e| ExecutionError::Abi(e.into()))
    }

    /// The image view of a Field of pixels, materialized if it was
    /// resident. This is an observation boundary: it is where device work
    /// is waited for.
    pub fn image(&mut self, handle: ImageHandle) -> Result<Image, ExecutionError> {
        self.materialize(handle)?;
        self.host
            .image(handle)
            .map_err(|e| ExecutionError::Abi(e.into()))
    }

    /// Mutable Host access, for interfaces that import and release resources.
    pub fn host_mut(&mut self) -> &mut Host {
        &mut self.host
    }

    /// Compiles, validates, and backend-prepares source without touching the
    /// registry.
    pub fn prepare(&self, source: &str) -> Result<Candidate, PrepareError> {
        let module = orla_compiler::compile(source).map_err(PrepareError::Compile)?;
        self.prepare_module(module, source)
    }

    /// Prepares an already parsed Kernel. `id` names an anonymous inline
    /// Kernel; `source` is kept for inspection.
    pub fn prepare_kernel(
        &self,
        kernel: &orla_language::Kernel,
        id: Option<&KernelId>,
        source: &str,
    ) -> Result<Candidate, PrepareError> {
        let module = orla_compiler::compile_kernel(kernel, id).map_err(PrepareError::Compile)?;
        self.prepare_module(module, source)
    }

    fn prepare_module(&self, module: Validated, source: &str) -> Result<Candidate, PrepareError> {
        prepare_on(self.backend.as_ref(), module, source.to_owned())
    }

    /// Prepares a module the runtime did not compile from source (a fused
    /// group, say) inline, as a detached version; see [`Runtime::detach`].
    pub fn prepare_detached(
        &mut self,
        module: Validated,
        id: KernelId,
        source: String,
    ) -> Result<Arc<KernelVersion>, PrepareError> {
        let candidate = self.prepare_module(module, &source)?;
        Ok(self.detach(candidate, id))
    }

    /// Atomically publishes a candidate if `expect` matches the registry.
    /// Existing `Arc<KernelVersion>` holders keep their pinned version.
    pub fn publish(
        &mut self,
        candidate: Candidate,
        expect: Expect,
    ) -> Result<Arc<KernelVersion>, StaleCandidate> {
        let id = candidate.id().clone();
        let actual = self.kernels.get(&id).map(|version| version.generation);
        let matches = match expect {
            Expect::Absent => actual.is_none(),
            Expect::Generation(generation) => actual == Some(generation),
            Expect::Any => true,
        };
        if !matches {
            return Err(StaleCandidate {
                id,
                expected: expect,
                actual,
            });
        }
        let generation = Generation(self.next_generation);
        self.next_generation += 1;
        let version = Arc::new(KernelVersion {
            id: id.clone(),
            generation,
            module: candidate.module,
            source: candidate.source,
            prepared: candidate.prepared,
        });
        self.kernels.insert(id, Arc::clone(&version));
        Ok(version)
    }

    /// Prepares and publishes in one step, replacing any current version.
    pub fn load(&mut self, source: &str) -> Result<Arc<KernelVersion>, LoadError> {
        let candidate = self.prepare(source).map_err(LoadError::Prepare)?;
        self.publish(candidate, Expect::Any)
            .map_err(LoadError::Stale)
    }

    /// The current version of a Kernel.
    pub fn kernel(&self, id: &KernelId) -> Option<Arc<KernelVersion>> {
        self.kernels.get(id).cloned()
    }

    /// All current versions in id order.
    pub fn kernels(&self) -> impl Iterator<Item = &Arc<KernelVersion>> {
        self.kernels.values()
    }

    /// Removes a Kernel from the registry. Pinned versions stay executable.
    pub fn retire(&mut self, id: &KernelId) -> Option<Arc<KernelVersion>> {
        self.kernels.remove(id)
    }

    /// Executes a pinned version. The Host is scoped to exactly the
    /// capabilities the manifest declares.
    pub fn execute(
        &mut self,
        version: &KernelVersion,
        invocation: &Invocation,
    ) -> Result<Vec<ImageHandle>, ExecutionError> {
        let granted = version.manifest().requires.clone();
        let mut scope = Scope::new(&mut self.host, granted);
        self.backend.execute(
            version.prepared.as_ref(),
            &version.module,
            invocation,
            &mut scope,
        )
    }
}
