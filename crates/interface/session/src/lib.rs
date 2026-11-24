//! A live session over one `.esn` document and the libraries it uses.
//!
//! The session reads its documents through `entelechy-documents`, which
//! parses them, publishes their Kernels, follows their `use` items and
//! lowers the Graph. Each [`Session::sync`] republishes only Kernels whose
//! text changed; a rejected replacement leaves the running version.
//!
//! The [session](session) opens the backends, keeps the runtime and
//! evaluator, and orders the documents' sync. What a caller
//! [observes](observe) is read from the parts it is handed.
//!
//! Every interface (command line, Player, C ABI, importer, exporter) opens
//! a [`Session`]. The wire is `entelechy-protocol`; file formats are
//! `entelechy-formats`.

#![warn(missing_docs)]

pub mod apart;
pub mod bench;
pub mod bundle;
mod clipboard;
mod clocked;
mod devices;
mod memory;
#[cfg(feature = "documents")]
mod observe;
#[cfg(feature = "documents")]
mod options;
pub mod play;
pub mod preparing;
pub mod remember;
#[cfg(feature = "documents")]
pub mod replay;
mod session;
pub mod watch;

use std::fmt;
use std::path::PathBuf;

use entelechy_formats::ppm;
#[cfg(feature = "documents")]
use entelechy_kernel::{Generation, KernelId, Validated};

#[cfg(feature = "documents")]
pub use entelechy_documents::history::{Attempt, Change, Origin, Record};
#[cfg(feature = "documents")]
pub use entelechy_documents::store::{
    Arrived, Disk, FetchFn, FetchLaterFn, Kept, Later, Memory, Parsed, Store,
};
#[cfg(feature = "documents")]
pub use entelechy_documents::{
    Allowance, Diagnostic, Event, Fix, Refusal, SourcePlace, history, overrides, package,
    parse_allow, store,
};
/// How a device is waited for: blocking, or giving a page turns.
pub use entelechy_gpu::Wait;
#[cfg(feature = "documents")]
pub use options::{SessionOptions, environment_packages};

/// The world at one moment, taken for its caller to keep
/// ([`Session::snapshot`]).
#[cfg(feature = "documents")]
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    /// The revision it was taken at.
    pub revision: u64,
    /// The frames the revision had run.
    pub frame: u64,
    /// Every published Kernel: id, generation and source, readable
    /// without the session.
    pub laws: Vec<(KernelId, Generation, String)>,
    /// Every port's committed state.
    pub states: Vec<StateCopy>,
    /// The scene document the running graph was built from, if any.
    pub scene: Option<String>,
}

/// What structure a restored snapshot runs in.
#[cfg(feature = "documents")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Structure {
    /// The scene as it stands.
    #[default]
    Current,
    /// The snapshot's scene document, written back first.
    Snapshot,
}

/// What laws a restored snapshot runs under.
#[cfg(feature = "documents")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Laws {
    /// The running laws; state migrates where a layout changed.
    Current,
    /// The snapshot's own Kernels, published back.
    Snapshot,
}

/// One port's committed state in a snapshot.
#[cfg(feature = "documents")]
#[derive(Clone, Debug, PartialEq)]
pub struct StateCopy {
    /// The producing node.
    pub node: String,
    /// Its port.
    pub port: String,
    /// The element layout.
    pub schema: entelechy_kernel::Schema,
    /// The extent.
    pub domain: entelechy_kernel::Domain,
    /// The Kernel's words, one per scalar.
    pub words: Vec<u32>,
}
pub use clocked::DeviceTimes;
pub use memory::{HeldBytes, MemoryMap};
pub use remember::Warm;
pub use session::{Session, StepError, StepOutcome, StepStop};

#[cfg(feature = "documents")]
pub use entelechy_documents::sources::{
    Declarations, PACKAGE_VERSION, declarations_for, declarations_within, locate_document,
    package_manifest, read_document, used_documents,
};
#[cfg(feature = "documents")]
pub use observe::parse_target;

/// What `trace` returns: the reference interpreter's record of one
/// element, re-exported so interfaces read it here.
#[cfg(feature = "documents")]
pub use entelechy_cpu::{CALL_RECORDS, Trace, TraceFault, TraceValue};

/// A traced element: the trace, the Kernel it ran, and where its failure,
/// if any, is written.
#[cfg(feature = "documents")]
#[derive(Clone, Debug)]
pub struct Traced {
    /// What the interpreter recorded.
    pub trace: Trace,
    /// The module it ran: the trace's types name its structs.
    pub module: Validated,
    /// The node's Kernel.
    pub kernel: KernelId,
    /// Where the element's failure is written, when it failed and a
    /// document owns the Kernel.
    pub place: Option<SourcePlace>,
}

pub use clipboard::{Clipboard, Held};

#[cfg(feature = "documents")]
pub use entelechy_documents::check::{
    Checked, LARGE, check_document, check_document_with, large, large_past,
};

/// A runtime built for a session ([`Session::build`]).
pub struct Built {
    /// The runtime.
    pub runtime: entelechy_kernel::Runtime,
    /// The device's display, when the backend has one.
    pub display: Option<entelechy_gpu::present::Display>,
    /// The backends tried first and passed over, as `name: why`.
    pub passed_over: Vec<String>,
}

/// A `publish` being prepared ([`Session::publish_later`]), finished by
/// [`Session::publishing`].
#[cfg(feature = "documents")]
pub struct Publishing {
    ticket: entelechy_kernel::Ticket,
    expect: entelechy_kernel::Expect,
    source: String,
}

/// A session opened and its scene read ([`Session::open_loaded`]).
#[cfg(feature = "documents")]
pub struct Loaded {
    /// The session, its scene loaded.
    pub session: Session,
    /// The device's display, when the backend has one.
    pub display: Option<entelechy_gpu::present::Display>,
    /// What reading the scene said.
    pub events: Vec<Event>,
    /// The backends tried first and passed over, as `name: why`.
    pub passed_over: Vec<String>,
}

/// A session opened over a compiled [`bundle::Bundle`] ([`Session::open_bundle`]).
pub struct LoadedBundle {
    /// The session, its bundle loaded or preparing in turns.
    pub session: Session,
    /// The device's display, when the backend has one.
    pub display: Option<entelechy_gpu::present::Display>,
    /// What opening the bundle said.
    pub said: Vec<String>,
    /// The backends tried first and passed over, as `name: why`.
    pub passed_over: Vec<String>,
}

/// Parses a window given on a command line: `<w>x<h>` or `<w>x<h>@<scale>`,
/// returning the size in pixels and the scale factor (one when absent).
pub fn parse_window(text: &str) -> Result<([f32; 2], f32), String> {
    let (size, scale) = match text.split_once('@') {
        Some((size, scale)) => (
            size,
            scale
                .parse::<f32>()
                .ok()
                .filter(|s| *s > 0.0)
                .ok_or_else(|| "--window's scale must be a positive number".to_string())?,
        ),
        None => (text, 1.0),
    };
    let (w, h) = size
        .split_once('x')
        .ok_or_else(|| "--window needs <w>x<h> or <w>x<h>@<scale>".to_string())?;
    let side = |text: &str| {
        text.parse::<u32>()
            .ok()
            .filter(|n| *n > 0)
            .map(|n| n as f32)
            .ok_or_else(|| "--window's sides must be positive integers".to_string())
    };
    Ok(([side(w)?, side(h)?], scale))
}

/// The most pixels a side of the pointer's picture may have: what a
/// browser takes for a page's pointer.
pub const POINTER_MOST: u32 = 128;

/// The default word budget of a session's Host: two gibibytes. It is a
/// net that stops a runaway scene before the machine swaps; `--memory`
/// raises it.
pub const DEFAULT_BUDGET_WORDS: usize = 512 * 1024 * 1024;

/// The words in a number of mebibytes, which is how `--memory` asks.
pub fn budget_of_megabytes(megabytes: usize) -> usize {
    megabytes.saturating_mul(1024 * 1024 / 4)
}

/// What a session's Host may hold and spend: Field words held at once
/// (`--memory`), loop iterations one Kernel run may take across all its
/// elements (`--work`), and the operations a run may take and go to the
/// processor when the main backend is a device (`--small`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Words the Host may hold at once.
    pub words: usize,
    /// Loop iterations one run may take.
    pub work: u64,
    /// Operations a run may take, by its elements and its Kernel's
    /// typical work, and go to the ordered backend; zero keeps every run
    /// on the main one.
    pub small: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            words: DEFAULT_BUDGET_WORDS,
            work: entelechy_kernel::DEFAULT_WORK,
            small: entelechy_kernel::DEFAULT_SMALL,
        }
    }
}

impl Limits {
    /// Whether `flag` sets a limit: `--memory`, `--work` or `--small`.
    pub fn takes(flag: &str) -> bool {
        matches!(flag, "--memory" | "--work" | "--small")
    }

    /// Sets the limit `flag` names from its value: `--memory <MB>` in
    /// whole mebibytes, `--work <M>` in millions of iterations, `--small
    /// <operations>` in operations, zero allowed.
    pub fn read(&mut self, flag: &str, text: &str) -> Result<(), String> {
        let whole = |unit: &str| -> Result<u64, String> {
            text.parse::<u64>().ok().filter(|n| *n > 0).ok_or_else(|| {
                format!("{flag} takes a whole number of {unit}, at least one, found `{text}`")
            })
        };
        match flag {
            "--memory" => self.words = budget_of_megabytes(whole("mebibytes")? as usize),
            "--work" => {
                let millions = whole("millions of iterations")?;
                let most = entelechy_kernel::MAX_WORK / 1_000_000;
                if millions > most {
                    return Err(format!(
                        "--work is at most {most} million iterations, found {millions}"
                    ));
                }
                self.work = millions * 1_000_000;
            }
            "--small" => {
                self.small = text.parse::<u64>().map_err(|_| {
                    format!("--small takes a whole number of operations, found `{text}`")
                })?;
            }
            _ => return Err(format!("{flag} sets no limit")),
        }
        Ok(())
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
    Host(entelechy_kernel::HostError),
    /// A resident image could not be brought back for observation.
    Observe(String),
    /// The file's extension names no format that takes the Field.
    Unwritable(PathBuf, String),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(path, error) => write!(f, "{}: {error}", path.display()),
            Self::Image(path, error) => write!(f, "{}: {error}", path.display()),
            Self::NoScene => f.write_str("no scene has been loaded"),
            Self::Host(error) => write!(f, "{error}"),
            Self::Observe(message) => write!(f, "cannot observe image: {message}"),
            Self::Unwritable(path, why) => write!(f, "cannot write {}: {why}", path.display()),
        }
    }
}

impl std::error::Error for SessionError {}
