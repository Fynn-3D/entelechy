#![warn(missing_docs)]
//! Kernel Runtime (L4).
//!
//! The runtime owns the Host, the backend it is given, and the registry of
//! published Kernel versions. It takes validated IR, never source: what
//! compiled a Kernel is the caller's business, and which backend runs it is
//! the caller's choice — every backend implements the same
//! [`Backend`] contract from `orla-rules`, none is built in. A version is
//! immutable once published. Replacing a Kernel is a two-step act:
//! [`Runtime::prepare_module`] lets the backend build whatever it needs (a shader pipeline, say) into a
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

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

pub use orla_abi::pad;
pub use orla_abi::{
    ABI_VERSION, Context, Domain, Encoding, FieldData, Handle, Host, HostError, Image, InputState,
    Invocation, Pixel, Scalar, Schema, Scope, Shape, Value,
};
pub use orla_abi::{DEFAULT_WORK, MAX_WORK, encoding, file};
pub use orla_ir::{KernelId, Manifest, Validated};
pub use orla_rules::{Backend, Check, ExecutionError, NativeOp, Prepared};

mod later;
pub use later::Run;

/// Monotonic counter identifying one published version. Generations are
/// unique across all Kernels in a runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Generation(pub u64);

impl fmt::Display for Generation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "g{}", self.0)
    }
}

/// What a version executes: compiled IR the backend prepared, or one of
/// the backend's native operations.
enum Body {
    Kernel {
        /// The Kernel as it was compiled, which fusion and inspection read.
        module: Box<Validated>,
        /// The module the backend runs: the Kernel with its work
        /// accounted ([`orla_ir::account`]).
        run: Box<Validated>,
        prepared: Box<dyn Prepared>,
    },
    Native(NativeOp),
}

/// An immutable published Kernel version.
pub struct KernelVersion {
    id: KernelId,
    generation: Generation,
    manifest: Manifest,
    source: String,
    body: Body,
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
        &self.manifest
    }

    /// The validated IR, or `None` for a native operation.
    pub fn module(&self) -> Option<&Validated> {
        match &self.body {
            Body::Kernel { module, .. } => Some(module),
            Body::Native(_) => None,
        }
    }

    /// The native operation this version stands for, if it is one.
    pub fn native(&self) -> Option<NativeOp> {
        match &self.body {
            Body::Native(op) => Some(*op),
            Body::Kernel { .. } => None,
        }
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
    /// The module the backend prepared and runs: `module` with its work
    /// accounted.
    run: Validated,
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
    /// The backend rejected the module; the message is backend specific.
    Backend(String),
    /// Accounting the module's work made it invalid: a fault of the
    /// runtime, reported rather than taken down with.
    Accounting(String),
    /// The backend panicked preparing the module: a fault of the
    /// backend, caught so the Kernel running and the worker live on.
    Panicked(String),
}

impl fmt::Display for PrepareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(message) => write!(f, "backend rejected the kernel: {message}"),
            Self::Accounting(message) => {
                write!(f, "accounting the kernel's work made it invalid: {message}")
            }
            Self::Panicked(message) => {
                write!(f, "the backend failed preparing the kernel: {message}")
            }
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
        if self.actual == Some(Generation(0)) {
            return write!(
                f,
                "cannot publish {}: it is a native operation of the backend",
                self.id
            );
        }
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
    // A backend runs the Kernel with its work accounted in its IR, so
    // where work is counted is one rule, not one a backend.
    let run = orla_ir::account(&module).map_err(|e| PrepareError::Accounting(e.to_string()))?;
    // A backend that panics on a module is a bug; a live session
    // reports it as a rejection, as it would any other, and keeps
    // running what it ran. Preparing touches nothing shared: the
    // candidate is dropped and the registry never saw it.
    let prepared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| backend.prepare(&run)))
        .map_err(|panic| {
            let said = panic
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a panic".to_owned());
            PrepareError::Panicked(said)
        })?
        .map_err(PrepareError::Backend)?;
    Ok(Candidate {
        module,
        run,
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
    /// The backend ordered work (a scan) runs on when it is not the
    /// main one: a GPU walks a scan on one thread, so a GPU runtime
    /// hands scans to the JIT. Its inputs are brought into Host memory
    /// first; its outputs are Host Fields the main backend uploads.
    ordered: Option<Arc<dyn Backend>>,
    kernels: BTreeMap<KernelId, Arc<KernelVersion>>,
    next_generation: u64,
    worker: Option<Worker>,
    next_ticket: u64,
    /// What the device took over the last execution, when the backend
    /// measured it; see [`Runtime::last_measured`].
    last_measured: Option<std::time::Duration>,
    /// Bytes carried between Host memory and a device by executions
    /// since it was last taken; see [`Runtime::take_moved`].
    moved: Moved,
    /// What executions since they were last taken answered with besides
    /// their outputs: whether a run on a device that is not waited for
    /// failed after all; see [`Runtime::take_checks`].
    checks: Vec<Box<dyn Check>>,
    ready: BTreeMap<Ticket, Result<Candidate, PrepareError>>,
    discarded: BTreeSet<Ticket>,
    runs: later::Runs,
}

impl Runtime {
    /// Creates a runtime over the given backend.
    pub fn with_backend(budget_words: usize, backend: Box<dyn Backend>) -> Self {
        Self::with_backends(budget_words, backend, None)
    }

    /// Creates a runtime over `backend`, with `ordered` running the scans
    /// when given; see [`Runtime::executes_on`].
    pub fn with_backends(
        budget_words: usize,
        backend: Box<dyn Backend>,
        ordered: Option<Box<dyn Backend>>,
    ) -> Self {
        let mut runtime = Self {
            host: Host::new(budget_words),
            backend: Arc::from(backend),
            ordered: ordered.map(Arc::from),
            kernels: BTreeMap::new(),
            next_generation: 1,
            worker: None,
            next_ticket: 1,
            last_measured: None,
            moved: Moved::default(),
            checks: Vec::new(),
            ready: BTreeMap::new(),
            discarded: BTreeSet::new(),
            runs: later::Runs::new(),
        };
        // Native operations are published from the start, at generation
        // zero, so graphs bind to them like any Kernel.
        for op in NativeOp::ALL {
            runtime.kernels.insert(
                op.id(),
                Arc::new(KernelVersion {
                    id: op.id(),
                    generation: Generation(0),
                    manifest: op.manifest(),
                    source: op.source().to_owned(),
                    body: Body::Native(op),
                }),
            );
        }
        runtime
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
        let ordered = self.ordered.clone();
        let handle = std::thread::Builder::new()
            .name("orla-prepare".into())
            // A backend walks a module's nested blocks by recursion; the
            // language bounds how deep they nest, and this is room for
            // that bound (orla_language::DEEP_STACK, which the runtime
            // does not depend on).
            .stack_size(256 << 20)
            .spawn(move || {
                for job in inbox {
                    let chosen = match &ordered {
                        Some(ordered) if goes_ordered(backend.as_ref(), &job.module) => ordered,
                        _ => &backend,
                    };
                    let result = prepare_on(chosen.as_ref(), job.module, job.source);
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
                    let result = prepare_on(self.backend_for(&module), module, source);
                    self.ready.insert(ticket, result);
                }
            }
            None => {
                let result = prepare_on(self.backend_for(&module), module, source);
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

    /// Preparations the worker has finished or been told to forget that
    /// nothing has collected. A session that publishes and evaluates in
    /// step leaves none; a growing number is work started and never
    /// asked for, which would hold a module and a backend's prepared
    /// state with it.
    pub fn uncollected(&self) -> usize {
        self.ready.len() + self.discarded.len()
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
            manifest: candidate.module.manifest.clone(),
            source: candidate.source,
            body: Body::Kernel {
                module: Box::new(candidate.module),
                run: Box::new(candidate.run),
                prepared: candidate.prepared,
            },
        })
    }

    /// Name of the backend executing Kernels.
    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    /// Waits for the backend's device to finish what it was given. See
    /// [`Backend::settle_device`].
    pub fn settle_device(&self) {
        self.backend().settle_device();
    }

    /// Whether the backend finishes its work on a device, so that what
    /// a frame gave it is done when [`Runtime::settle_device`] returns
    /// and not when `execute` did.
    pub fn has_device(&self) -> bool {
        self.backend().has_device()
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
    pub fn materialize(&mut self, handle: Handle) -> Result<(), ExecutionError> {
        self.bring_home(handle).map(|_| ())
    }

    /// Brings a resident Field's words into Host memory, answering how
    /// many bytes came back.
    fn bring_home(&mut self, handle: Handle) -> Result<u64, ExecutionError> {
        let Some(device) = self
            .host
            .resident(handle)
            .map_err(|e| ExecutionError::Abi(e.into()))?
        else {
            return Ok(0);
        };
        let domain = self
            .host
            .domain(handle)
            .map_err(|e| ExecutionError::Abi(e.into()))?;
        // The memory as kept: packed when the Field keeps scalars narrow.
        let byte_len = self
            .host
            .schema(handle)
            .map_err(|e| ExecutionError::Abi(e.into()))?
            .storage_words(domain.count())
            * 4;
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
            .map_err(|e| ExecutionError::Abi(e.into()))?;
        Ok(byte_len as u64)
    }

    /// The image view of a Field of pixels, materialized if it was
    /// resident. This is an observation boundary: it is where device work
    /// is waited for.
    pub fn image(&mut self, handle: Handle) -> Result<Image, ExecutionError> {
        self.materialize(handle)?;
        self.host
            .image(handle)
            .map_err(|e| ExecutionError::Abi(e.into()))
    }

    /// Mutable Host access, for interfaces that import and release resources.
    pub fn host_mut(&mut self) -> &mut Host {
        &mut self.host
    }

    /// Backend-prepares a module already compiled and validated; `source`
    /// is kept for inspection. The runtime takes IR: what compiled it is
    /// the caller's business.
    pub fn prepare_module(
        &self,
        module: Validated,
        source: &str,
    ) -> Result<Candidate, PrepareError> {
        prepare_on(self.backend_for(&module), module, source.to_owned())
    }

    /// The backend a module is prepared on and executes on: the ordered
    /// one, when there is one, for a scan and for what the main one does
    /// not take; else the main one.
    fn backend_for(&self, module: &Validated) -> &dyn Backend {
        match &self.ordered {
            Some(ordered) if goes_ordered(self.backend.as_ref(), module) => ordered.as_ref(),
            _ => self.backend.as_ref(),
        }
    }

    /// Whether a module runs on the ordered backend: there is one, and
    /// the module is a scan or one the main backend does not take.
    fn routed(&self, module: &Validated) -> bool {
        self.ordered.is_some() && goes_ordered(self.backend.as_ref(), module)
    }

    /// What the device took over the last execution, by its own clock,
    /// when the backend measured it (a GPU under a profile); `None`
    /// otherwise, and the caller's own timing is the answer.
    pub fn last_measured(&self) -> Option<std::time::Duration> {
        self.last_measured
    }

    /// What measuring one node costs before the node does anything; see
    /// [`Backend::measurement_floor`].
    pub fn measurement_floor(&self) -> std::time::Duration {
        self.backend.measurement_floor()
    }

    /// The name of the backend a version executes on.
    pub fn executes_on(&self, version: &KernelVersion) -> &'static str {
        self.placement(version).backend
    }

    /// Where a version executes, and why when it is not the main
    /// backend: what a report shows beside each node, so a node that
    /// leaves the device, and pays for the trip, says so.
    pub fn placement(&self, version: &KernelVersion) -> Placement {
        let Some(module) = version.module() else {
            return Placement {
                backend: self.backend.name(),
                routed: None,
            };
        };
        if !self.routed(module) {
            return Placement {
                backend: self.backend.name(),
                routed: None,
            };
        }
        let why = match self.backend.declines(module) {
            Some(reason) => format!("not on {}: {reason}", self.backend.name()),
            None => format!(
                "not on {}: a scan runs its steps in order, on one thread",
                self.backend.name()
            ),
        };
        Placement {
            backend: self.backend_for(module).name(),
            routed: Some(why),
        }
    }

    /// The checks executions since the last call answered with, in the
    /// order they ran: a device's run is not waited for, so whether it
    /// failed is known when a check is waited on. A caller that commits
    /// nothing before waiting on every check it took is held to the
    /// reference's failures exactly.
    pub fn take_checks(&mut self) -> Vec<Box<dyn Check>> {
        std::mem::take(&mut self.checks)
    }

    /// The bytes executions carried between Host memory and a device
    /// since the last call, and none after it: what one node's run cost
    /// in traffic, when the caller takes it around the run.
    pub fn take_moved(&mut self) -> Moved {
        std::mem::take(&mut self.moved)
    }

    /// The name of the backend ordered work runs on, when it is not the
    /// main one.
    pub fn ordered_backend_name(&self) -> Option<&'static str> {
        self.ordered.as_ref().map(|backend| backend.name())
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
        if self.kernels.get(&id).is_some_and(|v| v.native().is_some()) {
            return Err(StaleCandidate {
                id,
                expected: expect,
                actual,
            });
        }
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
            manifest: candidate.module.manifest.clone(),
            source: candidate.source,
            body: Body::Kernel {
                module: Box::new(candidate.module),
                run: Box::new(candidate.run),
                prepared: candidate.prepared,
            },
        });
        self.kernels.insert(id, Arc::clone(&version));
        Ok(version)
    }

    /// Prepares and publishes in one step, replacing any current version.
    /// `source` is kept for inspection, as [`Runtime::prepare_module`] keeps it.
    pub fn load(
        &mut self,
        module: Validated,
        source: &str,
    ) -> Result<Arc<KernelVersion>, LoadError> {
        let candidate = self
            .prepare_module(module, source)
            .map_err(LoadError::Prepare)?;
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

    /// Executes the same Kernel `repeat` times, each time reading input
    /// `input` from the output `output` the time before gave: what a
    /// node placed several times over its own output is. The Fields
    /// between the runs are made and released here and nothing else
    /// ever sees them, which is the point of asking for the whole run
    /// at once.
    pub fn execute_repeated(
        &mut self,
        version: &KernelVersion,
        invocation: &Invocation,
        repeat: u32,
        chain: (usize, usize),
    ) -> Result<Vec<Handle>, ExecutionError> {
        // A backend that runs the whole thing itself keeps every Field
        // between the times to itself, which is the point of asking.
        if let Body::Kernel {
            module,
            run,
            prepared,
        } = &version.body
            && !self.routed(module)
        {
            let granted = version.manifest().requires.clone();
            let backend = Arc::clone(&self.backend);
            let before = backend.uploaded();
            let answered = {
                let mut scope = Scope::new(&mut self.host, granted);
                backend.execute_repeated(
                    prepared.as_ref(),
                    run,
                    invocation,
                    &mut scope,
                    repeat,
                    chain,
                )
            };
            self.moved.uploaded += backend.uploaded().saturating_sub(before);
            if let Some(outputs) = answered {
                self.last_measured = backend.last_device_time();
                return held_to(version.manifest(), &mut self.host, outputs);
            }
        }
        let (input, output) = chain;
        let mut invocation = invocation.clone();
        let mut held: Option<Vec<Handle>> = None;
        for _ in 0..repeat.max(1) {
            let outputs = match self.execute(version, &invocation) {
                Ok(outputs) => outputs,
                Err(error) => {
                    for handle in held.into_iter().flatten() {
                        let _ = self.host.release(handle);
                    }
                    return Err(error);
                }
            };
            invocation.inputs[input] = outputs[output];
            // What the time before gave is read by this one and by
            // nothing else, so it goes now that it has been read.
            if let Some(gone) = held.replace(outputs) {
                for handle in gone {
                    let _ = self.host.release(handle);
                }
            }
        }
        Ok(held.expect("a run of at least one"))
    }

    /// Executes a pinned version. The Host is scoped to exactly the
    /// capabilities the manifest declares.
    pub fn execute(
        &mut self,
        version: &KernelVersion,
        invocation: &Invocation,
    ) -> Result<Vec<Handle>, ExecutionError> {
        let granted = version.manifest().requires.clone();
        // Nothing carries over: a backend that does not measure this
        // execution must answer `None` here, not the previous node's
        // number. A native operation used to leave it standing, and its
        // neighbour's time was printed as its own.
        self.last_measured = None;
        match &version.body {
            Body::Kernel {
                module,
                run,
                prepared,
            } => {
                let ordered = self.routed(module);
                if ordered {
                    // The ordered backend reads through Host memory:
                    // whatever the main one left resident comes back
                    // first.
                    for handle in &invocation.inputs {
                        self.moved.downloaded += self.bring_home(*handle)?;
                    }
                }
                let backend = if ordered {
                    Arc::clone(self.ordered.as_ref().expect("checked"))
                } else {
                    Arc::clone(&self.backend)
                };
                let before = backend.uploaded();
                let outputs = {
                    let mut scope = Scope::new(&mut self.host, granted);
                    backend.execute_checked(prepared.as_ref(), run, invocation, &mut scope)
                };
                self.moved.uploaded += backend.uploaded().saturating_sub(before);
                self.last_measured = backend.last_device_time();
                let outputs = outputs.map(|(outputs, check)| {
                    self.checks.extend(check);
                    outputs
                });
                held_to(version.manifest(), &mut self.host, outputs)
            }
            Body::Native(op) => {
                let backend = Arc::clone(&self.backend);
                let before = backend.uploaded();
                let outputs = {
                    let mut scope = Scope::new(&mut self.host, granted);
                    backend.execute_native(*op, invocation, &mut scope)
                };
                self.moved.uploaded += backend.uploaded().saturating_sub(before);
                self.last_measured = backend.last_device_time();
                held_to(version.manifest(), &mut self.host, outputs)
            }
        }
    }
}

/// What a backend handed back, held to the manifest: one Field per
/// declared output, in order. A backend that answers with another count
/// has broken its contract; what it made is released and the execution
/// fails, rather than a reader indexing past what exists.
pub(crate) fn held_to(
    manifest: &Manifest,
    host: &mut Host,
    outputs: Result<Vec<Handle>, ExecutionError>,
) -> Result<Vec<Handle>, ExecutionError> {
    let outputs = outputs?;
    if outputs.len() == manifest.outputs.len() {
        return Ok(outputs);
    }
    for handle in &outputs {
        let _ = host.release(*handle);
    }
    Err(ExecutionError::Device(format!(
        "the backend gave {} Field(s) for `{}`, which declares {} output(s)",
        outputs.len(),
        manifest.id,
        manifest.outputs.len()
    )))
}

/// Whether a module belongs on the ordered backend rather than `main`:
/// a scan, or a module `main` does not take.
fn goes_ordered(main: &dyn Backend, module: &Validated) -> bool {
    module.manifest.mode == orla_ir::Mode::Scan || main.declines(module).is_some()
}

/// Where a version executes: the backend's name, and why it is not the
/// main backend when it is not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    /// The backend that runs it.
    pub backend: &'static str,
    /// Why the main backend does not, when it does not.
    pub routed: Option<String>,
}

/// Bytes carried between Host memory and a device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Moved {
    /// Brought to a device from Host memory.
    pub uploaded: u64,
    /// Brought back to Host memory from a device, for a backend that
    /// reads there.
    pub downloaded: u64,
}
