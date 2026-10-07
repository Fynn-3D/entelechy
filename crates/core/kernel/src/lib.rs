#![warn(missing_docs)]
//! Kernel Runtime (L4).
//!
//! The runtime owns the Host, the backend it is given, and the registry
//! of published Kernel versions. It takes validated IR; every backend
//! implements the [`Backend`] contract from `orla-rules`. A published
//! version is immutable. [`Runtime::prepare_module`] builds a
//! [`Candidate`] outside the registry; [`Runtime::publish`] swaps it in
//! atomically when the caller's expected generation still holds. A failed
//! preparation leaves the running Kernel untouched, and holders of an
//! `Arc<KernelVersion>` keep executing the version they pinned.
//!
//! [`Runtime::prepare_later`] hands preparation to a worker and returns a
//! [`Ticket`]; [`Runtime::poll`] collects the [`Candidate`]. Without a
//! worker the same calls prepare inline, deterministically.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

pub use orla_abi::pad;
pub use orla_abi::{
    ABI_VERSION, Capabilities, Capability, Context, Domain, Encoding, FieldData, Handle, Host,
    HostError, Image, InputState, Invocation, Number, Pixel, Scalar, Schema, Scope, Shape, Value,
};
pub use orla_abi::{DEFAULT_WORK, MAX_WORK, encoding};
pub use orla_ir::{KernelId, Manifest, Validated};
pub use orla_rules::{
    Backend, Check, Downloading, ExecutionError, Extent, NativeOp, Prepared, extent,
};

/// The operations a run may take, by its elements times its Kernel's
/// typical work, and go to the ordered backend when the main one is a
/// device: about what a device's dispatch costs in time, spent on the
/// processor instead.
pub const DEFAULT_SMALL: u64 = 400_000;

/// The elements a run may have and go to the ordered backend however much
/// each takes: fewer than one of a device's groups holds, which the device
/// runs a thread each.
pub const FEW_ELEMENTS: u64 = 64;

/// The most a run of [`FEW_ELEMENTS`] may read of what the device made in
/// the same evaluation and still go to the ordered backend, which waits
/// for it and brings it down.
pub const FEW_FRESH_BYTES: u64 = 1 << 20;

/// The operations past which a run of [`FEW_ELEMENTS`] that reads what the
/// device made in the same evaluation still goes to the ordered backend:
/// run a thread at a time on a device they take longer (two thirds of a
/// millisecond, at the 6.6 ns an operation measured) than waiting for the
/// device costs; fewer stay, the device's work kept whole.
pub const FEW_LONG_WORK: u64 = 100_000;

pub mod clock;
mod later;
mod prepare;
pub use later::Run;

/// Monotonic counter identifying one published version, unique across
/// all Kernels in a runtime.
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
        /// The Kernel as compiled, read by fusion and inspection.
        module: Box<Validated>,
        /// The module the backend runs: the Kernel with its work accounted
        /// ([`orla_ir::account`]).
        run: Box<Validated>,
        prepared: Box<dyn Prepared>,
        /// Why the main backend left it to the ordered one, decided when it
        /// was prepared; `None` when it runs on the main backend.
        routed: Option<String>,
        /// The operations one element takes as a rule
        /// ([`Validated::typical_work`]), for sizing a run.
        typical: u64,
        /// The module prepared on the ordered backend, the first time a run
        /// small enough goes there; `None` when that backend refuses it.
        on_ordered: std::sync::OnceLock<Option<Box<dyn Prepared>>>,
        /// Why the main backend left it at a run, having prepared it (its
        /// device's own compiler failed on it when the run needed it
        /// built): the runs from then on go to the ordered backend.
        left: std::sync::OnceLock<String>,
        /// Whether a run went to the main backend, so a replacement is
        /// finished there ([`Backend::finish`]) before it takes this one's
        /// place.
        ran_main: std::sync::atomic::AtomicBool,
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

    /// Why the main backend left this version to the ordered one, when it
    /// did: in preparing it, or at a run.
    pub(crate) fn routed(&self) -> Option<&str> {
        match &self.body {
            Body::Kernel { routed, left, .. } => {
                routed.as_deref().or_else(|| left.get().map(String::as_str))
            }
            Body::Native(_) => None,
        }
    }

    /// Where runs of this version went: the main backend, and the ordered
    /// one for runs the size sent there.
    pub(crate) fn ahead(&self) -> Ahead {
        match &self.body {
            Body::Kernel {
                ran_main,
                on_ordered,
                routed: None,
                ..
            } => Ahead {
                main: ran_main.load(std::sync::atomic::Ordering::Relaxed),
                ordered: on_ordered.get().is_some_and(Option::is_some),
            },
            _ => Ahead::default(),
        }
    }

    /// What the ordered backend runs this version from: what was prepared
    /// there when the main backend left it in preparing it, or else what
    /// was prepared there the first time a run went there; `None` when
    /// that backend refuses it.
    pub(crate) fn on_ordered(&self, ordered: &dyn Backend) -> Option<&dyn Prepared> {
        match &self.body {
            Body::Kernel {
                routed: Some(_),
                prepared,
                ..
            } => Some(prepared.as_ref()),
            Body::Kernel {
                run, on_ordered, ..
            } => on_ordered
                .get_or_init(|| prepared_quietly(ordered, run))
                .as_deref(),
            Body::Native(_) => None,
        }
    }

    /// After a run on `main` failed: whether `main` now declines this
    /// version (its device's own compiler failed on it as the run needed
    /// it built), recorded once, the runs from then on going to the
    /// ordered backend.
    pub(crate) fn leaves(&self, main: &dyn Backend) -> bool {
        let Body::Kernel {
            run, routed, left, ..
        } = &self.body
        else {
            return false;
        };
        if routed.is_some() || left.get().is_some() {
            return false;
        }
        match main.declines(run) {
            Some(why) => {
                let _ = left.set(why);
                true
            }
            None => false,
        }
    }
}

/// A compiled, validated, backend-prepared Kernel not yet published.
pub struct Candidate {
    module: Validated,
    /// The module the backend prepared: `module` with its work accounted.
    run: Validated,
    source: String,
    prepared: Box<dyn Prepared>,
    /// Why the main backend left it to the ordered one, if it did.
    routed: Option<String>,
    /// What it was prepared with on the ordered backend before it was
    /// ready, for runs the size sends there; `None` when it was not.
    ordered_ahead: Option<Option<Box<dyn Prepared>>>,
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

    /// The version made of it, as `id` at `generation`.
    fn into_version(self, id: KernelId, generation: Generation) -> KernelVersion {
        KernelVersion {
            id,
            generation,
            manifest: self.module.manifest.clone(),
            source: self.source,
            body: Body::Kernel {
                typical: self.run.typical_work(),
                module: Box::new(self.module),
                run: Box::new(self.run),
                prepared: self.prepared,
                routed: self.routed,
                on_ordered: {
                    let on = std::sync::OnceLock::new();
                    if let Some(ahead) = self.ordered_ahead {
                        let _ = on.set(ahead);
                    }
                    on
                },
                left: std::sync::OnceLock::new(),
                ran_main: std::sync::atomic::AtomicBool::new(false),
            },
        }
    }
}

/// Why a candidate could not be prepared.
#[derive(Debug)]
pub enum PrepareError {
    /// The backend rejected the module; the message is backend-specific.
    Backend(String),
    /// Accounting the module's work made it invalid: a runtime fault,
    /// reported.
    Accounting(String),
    /// The backend panicked preparing the module; caught, so the running
    /// Kernel and the worker live on.
    Panicked(String),
    /// The Kernel requires capabilities the runtime does not grant.
    NotGranted {
        /// The Kernel.
        id: KernelId,
        /// Required capabilities that are not granted.
        missing: Capabilities,
    },
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
            Self::NotGranted { id, missing } => write!(
                f,
                "{id} requires {missing}, which this session does not grant"
            ),
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

/// Prepares a module on `main`, or on `ordered` when there is one and
/// `main` declines the module; the candidate keeps `main`'s reason, so
/// where it runs is decided once.
/// Whether every capability `module` requires is in `grants`.
fn granted_by(grants: &Capabilities, module: &Validated) -> Result<(), PrepareError> {
    let missing: Capabilities = module
        .manifest
        .requires
        .iter()
        .filter(|capability| !grants.contains(*capability))
        .collect();
    if missing.iter().next().is_none() {
        return Ok(());
    }
    Err(PrepareError::NotGranted {
        id: module.manifest.id.clone(),
        missing,
    })
}

/// A runtime's backends and grants, to prepare modules on many threads at
/// once ([`Runtime::preparer`]); a candidate it makes is the one the
/// runtime would make.
#[derive(Clone)]
pub struct Preparer {
    backend: Arc<dyn Backend>,
    ordered: Option<Arc<dyn Backend>>,
    grants: Capabilities,
}

impl Preparer {
    /// Finishes `version` on the main backend ([`Backend::finish`]), where
    /// it was prepared there, so its first run there waits for nothing;
    /// where that backend could not build it after all, its runs go to the
    /// ordered backend from now on. A panic in the backend leaves it for
    /// that run to report.
    pub fn finish(&self, version: &KernelVersion) {
        if let Body::Kernel { prepared, .. } = &version.body
            && version.routed().is_none()
        {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.backend.finish(prepared.as_ref())
            }));
            if self.ordered.is_some() {
                version.leaves(self.backend.as_ref());
            }
        }
    }

    /// Prepares `module` to replace `running` as [`Runtime::prepare_later`]
    /// does: made ready where `running` ran.
    pub fn prepare_replacing(
        &self,
        module: Validated,
        source: &str,
        running: &KernelVersion,
    ) -> Result<Candidate, PrepareError> {
        granted_by(&self.grants, &module)?;
        prepare_finished(
            self.backend.as_ref(),
            self.ordered.as_deref(),
            module,
            source.to_owned(),
            running.ahead(),
        )
    }

    /// Prepares `module` as [`Runtime::prepare_module`] does.
    pub fn prepare(&self, module: Validated, source: &str) -> Result<Candidate, PrepareError> {
        granted_by(&self.grants, &module)?;
        prepare_on(
            self.backend.as_ref(),
            self.ordered.as_deref(),
            module,
            source.to_owned(),
        )
    }
}

fn prepare_on(
    main: &dyn Backend,
    ordered: Option<&dyn Backend>,
    module: Validated,
    source: String,
) -> Result<Candidate, PrepareError> {
    let routed = ordered.and_then(|_| main.declines(&module));
    let backend = match ordered {
        Some(ordered) if routed.is_some() => ordered,
        _ => main,
    };
    // Work accounting is one rule in the IR, shared by every backend.
    let run = orla_ir::account(&module).map_err(|e| PrepareError::Accounting(e.to_string()))?;
    let prepared = prepared_on(backend, &run);
    // The main backend found in preparing it that it cannot run it after
    // all (its device's own compiler failed on a valid shader): it says so
    // now, and the ordered backend runs it, as it runs what the main one
    // declines beforehand.
    let (prepared, routed) = match (prepared, ordered) {
        (Err(error), Some(ordered)) if routed.is_none() => match main.declines(&run) {
            Some(why) => (prepared_on(ordered, &run)?, Some(why)),
            None => return Err(error),
        },
        (prepared, _) => (prepared?, routed),
    };
    Ok(Candidate {
        module,
        run,
        source,
        prepared,
        routed,
        ordered_ahead: None,
    })
}

/// Where a running version ran, so its replacement is made ready there
/// before it takes its place: finished on the main backend
/// ([`Backend::finish`]), prepared on the ordered one for runs the size
/// sends there.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Ahead {
    main: bool,
    ordered: bool,
}

/// `module` prepared as [`prepare_on`] does, made ready where `ahead`
/// says; prepared again where the main backend could not build it after
/// all, for the ordered backend to run.
pub(crate) fn prepare_finished(
    main: &dyn Backend,
    ordered: Option<&dyn Backend>,
    module: Validated,
    source: String,
    ahead: Ahead,
) -> Result<Candidate, PrepareError> {
    let mut candidate = prepare_on(main, ordered, module, source)?;
    if candidate.routed.is_some() {
        return Ok(candidate);
    }
    if ahead.main {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            main.finish(candidate.prepared.as_ref())
        }));
        if ordered.is_some() && main.declines(&candidate.run).is_some() {
            return prepare_on(main, ordered, candidate.module, candidate.source);
        }
    }
    if let Some(ordered) = ordered.filter(|_| ahead.ordered) {
        candidate.ordered_ahead = Some(prepared_quietly(ordered, &candidate.run));
    }
    Ok(candidate)
}

/// `run` prepared on `backend`. A backend panic is reported as a
/// rejection; the candidate is dropped unseen by the registry.
fn prepared_on(backend: &dyn Backend, run: &Validated) -> Result<Box<dyn Prepared>, PrepareError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| backend.prepare(run)))
        .map_err(|panic| {
            let said = panic
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a panic".to_owned());
            PrepareError::Panicked(said)
        })?
        .map_err(PrepareError::Backend)
}

/// `run` prepared on `backend`, or `None` where it declines or fails it: a
/// run small enough for the ordered backend then stays on the main one.
fn prepared_quietly(backend: &dyn Backend, run: &Validated) -> Option<Box<dyn Prepared>> {
    if backend.declines(run).is_some() {
        return None;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| backend.prepare(run)))
        .ok()?
        .ok()
}

/// What executions answer besides their outputs, kept until taken: bytes
/// moved between Host and device, the device's time for the last
/// execution, and checks of unwaited device runs.
#[derive(Default)]
struct Telemetry {
    /// The device's time for the last execution, when measured; see
    /// [`Runtime::last_measured`].
    last_measured: Option<std::time::Duration>,
    /// Bytes carried since last taken; see [`Runtime::take_moved`].
    moved: Moved,
    /// Whether an unwaited device run failed; see [`Runtime::take_checks`].
    checks: Vec<Box<dyn Check>>,
}

impl Telemetry {
    /// Counts what `backend` uploaded since it said `before`.
    fn carried(&mut self, backend: &dyn Backend, before: u64) {
        self.moved.uploaded += backend.uploaded().saturating_sub(before);
    }

    /// Keeps what `backend` measured of the execution just done.
    fn measured(&mut self, backend: &dyn Backend) {
        self.last_measured = backend.last_device_time();
    }
}

/// The Kernel Runtime.
pub struct Runtime {
    host: Host,
    backend: Arc<dyn Backend>,
    /// The backend for ordered work (a scan) when the main one hands it off,
    /// as a GPU runtime does to the JIT. Its inputs are brought into Host
    /// memory; its outputs are Host Fields the main backend uploads.
    ordered: Option<Arc<dyn Backend>>,
    kernels: BTreeMap<KernelId, Arc<KernelVersion>>,
    next_generation: u64,
    /// Kernels being prepared, on a worker or inline.
    preparer: prepare::Preparer,
    /// What executions answered besides their outputs.
    telemetry: Telemetry,
    runs: later::Runs,
    /// The capabilities a Kernel may be given; all of them unless set by
    /// [`Runtime::set_grants`]. A Kernel is prepared only when its
    /// requirements are granted, and runs with exactly those.
    grants: Capabilities,
    /// The most operations a run may take, by its elements and its
    /// Kernel's typical work, and go to the ordered backend: a device's
    /// dispatch costs more than a small run takes on the processor. Zero
    /// keeps every run on the main backend.
    small: u64,
    /// Fields the main backend made in this evaluation: a small run that
    /// reads one stays on the main backend, as bringing it back would wait
    /// for the device's work before it. A Field made before, state or an
    /// input, comes back once while the device has little else to do.
    fresh: std::collections::HashSet<Handle>,
    /// Where the last execution ran, and why when off the main backend.
    last_placement: Option<Placement>,
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
            preparer: prepare::Preparer::default(),
            telemetry: Telemetry::default(),
            runs: later::Runs::new(),
            grants: Capability::ALL.into_iter().collect(),
            small: DEFAULT_SMALL,
            fresh: std::collections::HashSet::new(),
            last_placement: None,
        };
        // Native operations are published at generation zero, so graphs
        // bind to them like any Kernel.
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
    /// [`Runtime::prepare_later`] returns before the backend builds; without
    /// one (or where no thread can start) it prepares inline. Stopping waits
    /// for jobs in flight and keeps their results for [`Runtime::poll`].
    pub fn set_background(&mut self, enabled: bool) {
        if enabled == self.preparer.running() {
            return;
        }
        if enabled {
            self.preparer
                .start(Arc::clone(&self.backend), self.ordered.clone());
        } else {
            self.preparer.stop();
        }
    }

    /// Whether a preparation worker is running.
    pub fn background(&self) -> bool {
        self.preparer.running()
    }

    /// Hands a validated module to the worker, or prepares it now when
    /// there is none. `source` is kept for inspection. A replacement is
    /// made ready where the version it replaces ran: finished on the main
    /// backend ([`Backend::finish`]), prepared on the ordered one for runs
    /// the size sends there; so the runs after it takes that version's
    /// place do not wait for it.
    pub fn prepare_later(&mut self, module: Validated, source: String) -> Ticket {
        let ticket = self.preparer.ticket();
        if let Err(refused) = self.granted(&module) {
            self.preparer.finish(ticket, Err(refused));
            return ticket;
        }
        let ahead = self
            .kernels
            .get(&module.manifest.id)
            .map(|running| running.ahead())
            .unwrap_or_default();
        if let Some((module, source)) = self.preparer.send(ticket, module, source, ahead) {
            let result = prepare_finished(
                self.backend.as_ref(),
                self.ordered.as_deref(),
                module,
                source,
                ahead,
            );
            self.preparer.finish(ticket, result);
        }
        ticket
    }

    /// Takes a finished preparation, or `None` while it runs. A ticket is
    /// redeemed once.
    pub fn poll(&mut self, ticket: Ticket) -> Option<Result<Candidate, PrepareError>> {
        self.preparer.poll(ticket)
    }

    /// Uncollected preparations, finished or forgotten. Each holds a module
    /// and its prepared backend state; a session that publishes and evaluates
    /// in step leaves none.
    pub fn uncollected(&self) -> usize {
        self.preparer.uncollected()
    }

    /// Forgets a preparation whose result is unwanted.
    pub fn discard(&mut self, ticket: Ticket) {
        self.preparer.discard(ticket);
    }

    /// Turns a candidate into an unregistered version under the given id,
    /// executable like any pinned version.
    pub fn detach(&mut self, candidate: Candidate, id: KernelId) -> Arc<KernelVersion> {
        let generation = Generation(self.next_generation);
        self.next_generation += 1;
        Arc::new(candidate.into_version(id, generation))
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

    /// A mark of every run submitted so far ([`Backend::submitted`]).
    pub fn submitted(&self) -> Option<Box<dyn Check>> {
        self.backend().submitted()
    }

    /// Whether frames are clocked on the device ([`Backend::set_clocked`]).
    pub fn set_clocked(&self, on: bool) {
        self.backend().set_clocked(on);
    }

    /// Ends a clocked frame ([`Backend::end_frame`]).
    pub fn end_frame(&self) {
        self.backend().end_frame();
    }

    /// The device's busy time for the latest clocked frame it has done
    /// ([`Backend::frame_device_time`]).
    pub fn frame_device_time(&self) -> Option<std::time::Duration> {
        self.backend().frame_device_time()
    }

    /// Whether the backend's device work is done only when
    /// [`Runtime::settle_device`] returns, not when `execute` does.
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

    /// Brings a resident Field back into Host memory through its owning
    /// backend; a Field in memory stays as it is.
    pub fn materialize(&mut self, handle: Handle) -> Result<(), ExecutionError> {
        self.bring_home(handle).map(|_| ())
    }

    /// Whether two Fields hold the same words: compared in Host memory
    /// when both are there, on the device when both are resident on it.
    /// Fields of different extents or layouts, or apart, are not.
    pub fn same_words(&self, a: Handle, b: Handle) -> bool {
        if a == b {
            return true;
        }
        let host = &self.host;
        if let (Ok(x), Ok(y)) = (host.data(a), host.data(b)) {
            return x == y;
        }
        let (Ok(domain), Ok(other), Ok(schema), Ok(layout)) = (
            host.domain(a),
            host.domain(b),
            host.schema(a),
            host.schema(b),
        ) else {
            return false;
        };
        if domain != other || schema != layout {
            return false;
        }
        let (Ok(Some(x)), Ok(Some(y))) = (host.resident(a), host.resident(b)) else {
            return false;
        };
        let byte_len = schema.storage_words(domain.count()) * 4;
        self.backend.same_resident(x, y, byte_len).unwrap_or(false)
    }

    /// A Field's words taken later ([`Fetch::take`]): from Host memory
    /// now, or from the device once the work submitted before this call
    /// has run, not waiting for what is submitted since. The Field may be
    /// released meanwhile.
    pub fn fetch(&self, handle: Handle) -> Result<Fetch, ExecutionError> {
        if let Ok(data) = self.host.data(handle) {
            return Ok(Fetch::Here(data.clone()));
        }
        let host = |e: orla_abi::HostError| ExecutionError::Abi(e.into());
        let device = self
            .host
            .resident(handle)
            .map_err(host)?
            .ok_or_else(|| ExecutionError::Device("the Field holds no words".into()))?;
        let domain = self.host.domain(handle).map_err(host)?;
        let schema = self.host.schema(handle).map_err(host)?.clone();
        let byte_len = schema.storage_words(domain.count()) * 4;
        let coming = self
            .backend
            .download_later(device, byte_len)
            .map_err(ExecutionError::Device)?;
        Ok(Fetch::Coming {
            schema,
            domain,
            coming,
        })
    }

    /// Brings a resident Field's words into Host memory and answers the
    /// bytes moved.
    fn bring_home(&mut self, handle: Handle) -> Result<u64, ExecutionError> {
        // Words kept in memory beside the device's need no trip.
        if self.host.data(handle).is_ok() {
            return Ok(0);
        }
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
        // The memory as kept: packed when narrow.
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

    /// The image view of a Field of pixels, materialized if resident: an
    /// observation boundary, where device work is waited for.
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

    /// Backend-prepares a compiled, validated module; `source` is kept for
    /// inspection.
    pub fn prepare_module(
        &self,
        module: Validated,
        source: &str,
    ) -> Result<Candidate, PrepareError> {
        self.granted(&module)?;
        prepare_on(
            self.backend.as_ref(),
            self.ordered.as_deref(),
            module,
            source.to_owned(),
        )
    }

    /// Sets the capabilities a Kernel may be given, by whoever runs the
    /// session. A Kernel requiring more is refused at preparation, before any
    /// swap; a published one runs with only what is granted of what it
    /// requires.
    pub fn set_grants(&mut self, grants: Capabilities) {
        self.grants = grants;
    }

    /// The capabilities a Kernel may be given.
    pub fn grants(&self) -> &Capabilities {
        &self.grants
    }

    /// Whether every capability `module` requires is granted: checked first
    /// by preparation, and asked by callers before committing to a module.
    pub fn granted(&self, module: &Validated) -> Result<(), PrepareError> {
        granted_by(&self.grants, module)
    }

    /// What prepares modules as [`Runtime::prepare_module`] does, from any
    /// thread: this runtime's backends and grants, as they are now.
    pub fn preparer(&self) -> Preparer {
        Preparer {
            backend: Arc::clone(&self.backend),
            ordered: self.ordered.clone(),
            grants: self.grants.clone(),
        }
    }

    /// What a version runs with: what it requires, of what is granted.
    pub(crate) fn scope_grants(&self, version: &KernelVersion) -> Capabilities {
        version
            .manifest()
            .requires
            .iter()
            .filter(|capability| self.grants.contains(*capability))
            .collect()
    }

    /// The device's own time for the last execution, when the backend
    /// measured it (a GPU under a profile); else `None`.
    pub fn last_measured(&self) -> Option<std::time::Duration> {
        self.telemetry.last_measured
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

    /// Where a version executes and, off the main backend, why: shown beside
    /// each node in a report.
    pub fn placement(&self, version: &KernelVersion) -> Placement {
        match (version.routed(), &self.ordered) {
            (Some(why), Some(ordered)) => Placement {
                backend: ordered.name(),
                routed: Some(format!("not on {}: {why}", self.backend.name())),
            },
            _ => Placement {
                backend: self.backend.name(),
                routed: None,
            },
        }
    }

    /// Sets the most operations a run may take and go to the ordered
    /// backend (see [`DEFAULT_SMALL`]); zero keeps every run on the main one.
    pub fn set_small(&mut self, operations: u64) {
        self.small = operations;
    }

    /// Starts an evaluation: no Field is yet the main backend's own of it.
    pub fn begin_evaluation(&mut self) {
        self.fresh.clear();
    }

    /// Where the last execution ran, and why when off the main backend.
    pub fn last_placement(&self) -> Option<&Placement> {
        self.last_placement.as_ref()
    }

    /// Whether the backend the last execution ran on writes the same words
    /// for the same run every time ([`Backend::repeats_exactly`]).
    pub fn last_repeats_exactly(&self) -> bool {
        let Some(placement) = &self.last_placement else {
            return false;
        };
        let mut ran = std::iter::once(&self.backend)
            .chain(self.ordered.as_ref())
            .filter(|backend| backend.name() == placement.backend)
            .peekable();
        ran.peek().is_some() && ran.all(|backend| backend.repeats_exactly())
    }

    /// Why a run of a Kernel the main backend takes goes to the ordered one
    /// by its size, or `None`: its elements times its typical work are at
    /// most `small`, and it reads no Field the main backend made in this
    /// evaluation. The choice reads the run and nothing of time.
    fn sized_off(&self, run: &Validated, typical: u64, invocation: &Invocation) -> Option<String> {
        if self.small == 0 || self.ordered.is_none() || !self.backend.has_device() {
            return None;
        }
        let elements = match run.manifest.mode {
            orla_ir::Mode::Reduce | orla_ir::Mode::Prefix { .. } => invocation
                .inputs
                .first()
                .and_then(|input| self.host.domain(*input).ok())
                .map_or(0, |domain| domain.count()),
            _ => invocation.domain.count(),
        } as u64;
        // A scatter carries its target whole: where it runs, the target
        // is copied and written there, and its readers take it from there.
        let carried = match run.manifest.mode {
            orla_ir::Mode::Scatter { base, .. } => invocation
                .inputs
                .get(base as usize)
                .and_then(|input| self.host.domain(*input).ok())
                .map_or(0, |domain| domain.count() as u64),
            _ => 0,
        };
        // What it reads that the device made in this evaluation, in bytes:
        // the ordered backend waits for it and brings it down.
        let fresh: u64 = invocation
            .inputs
            .iter()
            .filter(|input| self.fresh.contains(*input))
            .map(
                |input| match (self.host.schema(*input), self.host.domain(*input)) {
                    (Ok(schema), Ok(domain)) => schema.storage_words(domain.count()) as u64 * 4,
                    _ => u64::MAX / 4,
                },
            )
            .fold(0, u64::saturating_add);
        let work = elements.saturating_mul(typical).saturating_add(carried);
        // Fewer elements than one of the device's groups holds gain nothing
        // there, the device running them a thread each, far slower than
        // the processor. They leave it, but for one that reads what the
        // device made in this evaluation: the processor would wait for
        // the device and bring it down, breaking the device's work in
        // two, which pays only when the run itself is long
        // ([`FEW_LONG_WORK`]) and what comes down is not large.
        let few = elements.max(carried) <= FEW_ELEMENTS
            && (fresh == 0 || (work > FEW_LONG_WORK && fresh <= FEW_FRESH_BYTES));
        if few {
            return Some(format!(
                "few: {elements} elements, which a device runs a thread each"
            ));
        }
        if fresh > 0 {
            return None;
        }
        (work <= self.small).then(|| format!("small: {elements} elements, about {work} operations"))
    }

    /// The checks executions answered since the last call, in run order.
    /// Device runs are not waited for; a check reports failure when waited
    /// on. A caller that commits only after waiting on every check matches
    /// the reference's failures exactly.
    pub fn take_checks(&mut self) -> Vec<Box<dyn Check>> {
        std::mem::take(&mut self.telemetry.checks)
    }

    /// The bytes moved between Host memory and a device since the last call;
    /// taken around one node's run, its traffic.
    pub fn take_moved(&mut self) -> Moved {
        std::mem::take(&mut self.telemetry.moved)
    }

    /// The ordered backend's name, when it is not the main one.
    pub fn ordered_backend_name(&self) -> Option<&'static str> {
        self.ordered.as_ref().map(|backend| backend.name())
    }

    /// Prepares a module not compiled from source (a fused group, say)
    /// inline, as a detached version; see [`Runtime::detach`].
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
        self.expected(candidate.id(), expect)?;
        Ok(self.swap_in(candidate))
    }

    /// Publishes several candidates as one change: every expectation is
    /// checked before any swap, so all are published or none. For Kernels
    /// that must change together, such as two sides of one interface.
    pub fn publish_all(
        &mut self,
        candidates: Vec<(Candidate, Expect)>,
    ) -> Result<Vec<Arc<KernelVersion>>, StaleCandidate> {
        let mut seen = std::collections::BTreeSet::new();
        for (candidate, expect) in &candidates {
            self.expected(candidate.id(), *expect)?;
            // One change publishes an id once.
            if !seen.insert(candidate.id().clone()) {
                return Err(StaleCandidate {
                    id: candidate.id().clone(),
                    expected: *expect,
                    actual: None,
                });
            }
        }
        Ok(candidates
            .into_iter()
            .map(|(candidate, _)| self.swap_in(candidate))
            .collect())
    }

    /// Restores exactly these versions and generations, or no version where
    /// an id had none: the undo of a [`Runtime::publish_all`] whose change
    /// was refused. An agent's expectation of what runs holds again.
    pub fn reinstate(&mut self, versions: Vec<(KernelId, Option<Arc<KernelVersion>>)>) {
        for (id, version) in versions {
            match version {
                Some(version) => {
                    self.kernels.insert(id, version);
                }
                None => {
                    self.kernels.remove(&id);
                }
            }
        }
    }

    /// Whether `expect` holds for `id` now and `id` is not a native
    /// operation.
    fn expected(&self, id: &KernelId, expect: Expect) -> Result<(), StaleCandidate> {
        let actual = self.kernels.get(id).map(|version| version.generation);
        let native = self.kernels.get(id).is_some_and(|v| v.native().is_some());
        let matches = match expect {
            Expect::Absent => actual.is_none(),
            Expect::Generation(generation) => actual == Some(generation),
            Expect::Any => true,
        };
        if native || !matches {
            return Err(StaleCandidate {
                id: id.clone(),
                expected: expect,
                actual,
            });
        }
        Ok(())
    }

    /// Swaps a checked candidate in as the id's next generation.
    fn swap_in(&mut self, candidate: Candidate) -> Arc<KernelVersion> {
        let id = candidate.id().clone();
        let generation = Generation(self.next_generation);
        self.next_generation += 1;
        let version = Arc::new(candidate.into_version(id.clone(), generation));
        self.kernels.insert(id, Arc::clone(&version));
        version
    }

    /// Prepares and publishes in one step, replacing any current version.
    /// `source` is kept for inspection.
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

    /// Removes a Kernel from the registry; pinned versions stay executable.
    /// A native operation stays and answers `None`.
    pub fn retire(&mut self, id: &KernelId) -> Option<Arc<KernelVersion>> {
        // A native operation is the Host's own and is never replaced.
        if self
            .kernels
            .get(id)
            .is_some_and(|version| version.native().is_some())
        {
            return None;
        }
        self.kernels.remove(id)
    }

    /// Executes one Kernel `repeat` times, each time reading input `input`
    /// from the previous time's output `output`: a node placed several times
    /// over its own output. The Fields between runs are made and released
    /// here, visible to nothing else.
    pub fn execute_repeated(
        &mut self,
        version: &KernelVersion,
        invocation: &Invocation,
        repeat: u32,
        chain: (usize, usize),
    ) -> Result<Vec<Handle>, ExecutionError> {
        // A backend that runs every time itself keeps the Fields between.
        if let Body::Kernel {
            run,
            prepared,
            typical,
            ran_main,
            ..
        } = &version.body
            && version.routed().is_none()
            && self.sized_off(run, *typical, invocation).is_none()
        {
            let granted = self.scope_grants(version);
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
            self.telemetry.carried(backend.as_ref(), before);
            // The main backend could not build it as the runs needed it: they
            // are made one by one where it now goes.
            let left = matches!(answered, Some(Err(_)))
                && self.ordered.is_some()
                && version.leaves(backend.as_ref());
            if let Some(outputs) = answered.filter(|_| !left) {
                self.telemetry.measured(backend.as_ref());
                if outputs.is_ok() {
                    ran_main.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                self.last_placement = Some(Placement {
                    backend: backend.name(),
                    routed: None,
                });
                let outputs = held_to(version.manifest(), &mut self.host, outputs)?;
                self.fresh.extend(outputs.iter().copied());
                return Ok(outputs);
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
            // The previous output has one reader, so it goes once read.
            if let Some(gone) = held.replace(outputs) {
                for handle in gone {
                    let _ = self.host.release(handle);
                }
            }
        }
        Ok(held.expect("a run of at least one"))
    }

    /// Executes a pinned version, with the Host scoped to exactly the
    /// capabilities the manifest declares.
    pub fn execute(
        &mut self,
        version: &KernelVersion,
        invocation: &Invocation,
    ) -> Result<Vec<Handle>, ExecutionError> {
        let granted = self.scope_grants(version);
        // Cleared per execution: a backend that does not measure this one
        // answers `None`.
        self.telemetry.last_measured = None;
        match &version.body {
            Body::Kernel {
                run,
                prepared,
                typical,
                ran_main,
                ..
            } => {
                // The ordered backend, when the main one left the Kernel to
                // it or the run is small enough; the main one otherwise.
                let sized = match version.routed() {
                    Some(_) => None,
                    None => self.sized_off(run, *typical, invocation),
                };
                let off: Option<(Arc<dyn Backend>, &dyn Prepared, String)> =
                    match (&self.ordered, version.routed(), sized) {
                        (Some(ordered), Some(why), _) => {
                            version.on_ordered(ordered.as_ref()).map(|on| {
                                (
                                    Arc::clone(ordered),
                                    on,
                                    format!("not on {}: {why}", self.backend.name()),
                                )
                            })
                        }
                        (Some(ordered), None, Some(why)) => version
                            .on_ordered(ordered.as_ref())
                            .map(|on| (Arc::clone(ordered), on, why)),
                        _ => None,
                    };
                if off.is_some() {
                    // The ordered backend reads Host memory, so resident
                    // Fields come back first.
                    for handle in &invocation.inputs {
                        let brought = self.bring_home(*handle)?;
                        self.telemetry.moved.downloaded += brought;
                    }
                }
                let (backend, prepared, placement) = match off {
                    Some((backend, on, why)) => {
                        let name = backend.name();
                        (
                            backend,
                            on,
                            Placement {
                                backend: name,
                                routed: Some(why),
                            },
                        )
                    }
                    None => (
                        Arc::clone(&self.backend),
                        prepared.as_ref(),
                        Placement {
                            backend: self.backend.name(),
                            routed: None,
                        },
                    ),
                };
                let before = backend.uploaded();
                let outputs = {
                    let mut scope = Scope::new(&mut self.host, granted);
                    backend.execute_checked(prepared, run, invocation, &mut scope)
                };
                self.telemetry.carried(backend.as_ref(), before);
                self.telemetry.measured(backend.as_ref());
                let outputs = outputs.map(|(outputs, check)| {
                    self.telemetry.checks.extend(check);
                    outputs
                });
                let main = placement.routed.is_none();
                if main {
                    // The main backend could not build it as this run needed
                    // it: the run is made again where it now goes.
                    if outputs.is_err()
                        && self.ordered.is_some()
                        && version.leaves(self.backend.as_ref())
                    {
                        return self.execute(version, invocation);
                    }
                    if outputs.is_ok() {
                        ran_main.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                self.last_placement = Some(placement);
                let outputs = held_to(version.manifest(), &mut self.host, outputs)?;
                if main {
                    self.fresh.extend(outputs.iter().copied());
                }
                Ok(outputs)
            }
            Body::Native(op) => {
                self.last_placement = Some(Placement {
                    backend: self.backend.name(),
                    routed: None,
                });
                let backend = Arc::clone(&self.backend);
                let before = backend.uploaded();
                let outputs = {
                    let mut scope = Scope::new(&mut self.host, granted);
                    backend.execute_native(*op, invocation, &mut scope)
                };
                self.telemetry.carried(backend.as_ref(), before);
                self.telemetry.measured(backend.as_ref());
                let outputs = held_to(version.manifest(), &mut self.host, outputs)?;
                self.fresh.extend(outputs.iter().copied());
                Ok(outputs)
            }
        }
    }
}

/// What a backend handed back, held to the manifest: one Field per
/// declared output, in order. Another count is a broken contract: what
/// it made is released and the execution fails.
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

/// Where a version executes: the backend's name, and why it is not the
/// main backend when it is not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    /// The backend that runs it.
    pub backend: &'static str,
    /// Why the main backend does not run it, when it does not.
    pub routed: Option<String>,
}

/// Bytes carried between Host memory and a device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Moved {
    /// Brought to a device from Host memory.
    pub uploaded: u64,
    /// Brought back to Host memory from a device, for a backend that reads
    /// there.
    pub downloaded: u64,
}

/// A Field's words on their way: [`Runtime::fetch`].
pub enum Fetch {
    /// In Host memory already.
    Here(FieldData),
    /// Coming from a device.
    Coming {
        /// How the words are laid out.
        schema: Schema,
        /// Over what extent.
        domain: Domain,
        /// The words on their way.
        coming: Box<dyn Downloading>,
    },
}

impl Fetch {
    /// The Field's words, once they have come.
    pub fn take(self) -> Result<FieldData, ExecutionError> {
        match self {
            Self::Here(data) => Ok(data),
            Self::Coming {
                schema,
                domain,
                coming,
            } => {
                let bytes = coming.take().map_err(ExecutionError::Device)?;
                let words = bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                FieldData::from_storage(schema, domain, words)
                    .map_err(|e| ExecutionError::Abi(e.into()))
            }
        }
    }
}
