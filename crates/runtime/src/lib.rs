#![warn(missing_docs)]
//! Kernel Runtime (L4).
//!
//! The runtime owns the Host, the backend, and the registry of published
//! Kernel versions. A version is immutable once published. Replacing a Kernel
//! is a two-step act: [`Runtime::prepare`] compiles and validates a
//! [`Candidate`] without touching the registry; [`Runtime::publish`] swaps it
//! in atomically, and only if the caller's expectation about the current
//! generation still holds. A failed preparation leaves the running Kernel
//! untouched. Callers that hold an `Arc<KernelVersion>` keep executing the
//! version they pinned, whatever the registry does afterwards.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

pub use orla_abi::{Domain, Host, HostError, Image, ImageHandle, Invocation, Scope, Value};
pub use orla_compiler::CompileError;
pub use orla_cpu::ExecutionError;
pub use orla_ir::{KernelId, Manifest, Validated};

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
#[derive(Debug)]
pub struct KernelVersion {
    id: KernelId,
    generation: Generation,
    module: Validated,
    source: String,
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

/// A compiled and validated Kernel that has not been published.
#[derive(Debug)]
pub struct Candidate {
    module: Validated,
    source: String,
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
    /// The source did not compile.
    Compile(CompileError),
    /// The registry state did not match.
    Stale(StaleCandidate),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compile(error) => write!(f, "{error}"),
            Self::Stale(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// The Kernel Runtime.
pub struct Runtime {
    host: Host,
    backend: orla_cpu::CpuBackend,
    kernels: BTreeMap<KernelId, Arc<KernelVersion>>,
    next_generation: u64,
}

impl Runtime {
    /// Creates a runtime whose Host may hold at most `budget_pixels`.
    pub fn new(budget_pixels: usize) -> Self {
        Self {
            host: Host::new(budget_pixels),
            backend: orla_cpu::CpuBackend,
            kernels: BTreeMap::new(),
            next_generation: 1,
        }
    }

    /// The Host, for interfaces that import and export resources.
    pub fn host(&self) -> &Host {
        &self.host
    }

    /// Mutable Host access, for interfaces that import and release resources.
    pub fn host_mut(&mut self) -> &mut Host {
        &mut self.host
    }

    /// Compiles and validates source without touching the registry.
    pub fn prepare(&self, source: &str) -> Result<Candidate, CompileError> {
        let module = orla_compiler::compile(source)?;
        Ok(Candidate {
            module,
            source: source.to_owned(),
        })
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
        });
        self.kernels.insert(id, Arc::clone(&version));
        Ok(version)
    }

    /// Prepares and publishes in one step, replacing any current version.
    pub fn load(&mut self, source: &str) -> Result<Arc<KernelVersion>, LoadError> {
        let candidate = self.prepare(source).map_err(LoadError::Compile)?;
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
        self.backend
            .execute(&version.module, invocation, &mut scope)
    }
}
