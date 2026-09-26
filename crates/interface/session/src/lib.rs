//! A live session over one `.osn` document and the libraries it uses.
//!
//! The session parses the document, publishes its inline Kernels, follows
//! its `use` items to `.orla` libraries, and builds the Graph. Each
//! [`Session::sync`] republishes only Kernels whose text changed; a
//! rejected replacement leaves the running version. Source resolution
//! lives here.
//!
//! Four parts each own their state and reach the others only through what
//! they are handed: the [scene](scene) reads documents and lowers the
//! Graph; the [ledger](kernels) knows which document published which
//! Kernel; the [inputs](inputs) own the Fields a scene's inputs hold; and
//! [source resolution](sources) is stateless. The [session](session) keeps
//! the runtime and evaluator and orders the sync. What a caller
//! [observes](observe) is read from the parts it is handed.
//!
//! Every interface (command line, Player, C ABI, importer, exporter) opens
//! a [`Session`]. The wire is `orla-protocol`; file formats are
//! `orla-formats`.

#![warn(missing_docs)]

pub mod bench;
mod check;
pub mod history;
mod inputs;
mod kernels;
mod observe;
mod options;
pub mod play;
pub mod replay;
mod scene;
pub mod search;
mod session;
mod sources;
pub mod store;
mod writes;

use std::fmt;
use std::path::PathBuf;

use orla_formats::ppm;
use orla_kernel::{Generation, KernelId};

pub use history::{Attempt, Change, Origin, Record};
pub use options::{SessionOptions, environment_packages};
/// How a device is waited for: blocking, or giving a page turns.
pub use orla_gpu::Wait;
pub use store::{Disk, Memory, Store};

/// The world at one moment, taken for its caller to keep
/// ([`Session::snapshot`]).
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Structure {
    /// The scene as it stands.
    #[default]
    Current,
    /// The snapshot's scene document, written back first.
    Snapshot,
}

/// What laws a restored snapshot runs under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Laws {
    /// The running laws; state migrates where a layout changed.
    Current,
    /// The snapshot's own Kernels, published back.
    Snapshot,
}

/// One port's committed state in a snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct StateCopy {
    /// The producing node.
    pub node: String,
    /// Its port.
    pub port: String,
    /// The element layout.
    pub schema: orla_kernel::Schema,
    /// The extent.
    pub domain: orla_kernel::Domain,
    /// The Kernel's words, one per scalar.
    pub words: Vec<u32>,
}
pub use session::Session;

pub use observe::parse_target;
pub use sources::{
    Declarations, PACKAGE_VERSION, declarations_for, declarations_within, package_manifest,
    read_document, used_documents,
};

/// What `trace` returns: the reference interpreter's record of one
/// element, re-exported so interfaces read it here.
pub use orla_cpu::{CALL_RECORDS, Trace, TraceValue};

/// What the person running a session allows its scene beyond its own
/// directory. Only a command line grants it; the scene never does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Allowance {
    /// Inputs may be read from files in this directory and under it.
    FileRead(PathBuf),
    /// Outputs may be written to files in this directory and under it.
    FileWrite(PathBuf),
}

pub use check::{Checked, check_document};

/// A runtime built for a session ([`Session::build`]).
pub struct Built {
    /// The runtime.
    pub runtime: orla_kernel::Runtime,
    /// The device's display, when the backend has one.
    pub display: Option<orla_gpu::present::Display>,
    /// The backends tried first and passed over, as `name: why`.
    pub passed_over: Vec<String>,
}

/// A `publish` being prepared ([`Session::publish_later`]), finished by
/// [`Session::publishing`].
pub struct Publishing {
    ticket: orla_kernel::Ticket,
    expect: orla_kernel::Expect,
    source: String,
}

/// A session opened and its scene read ([`Session::open_loaded`]).
pub struct Loaded {
    /// The session, its scene loaded.
    pub session: Session,
    /// The device's display, when the backend has one.
    pub display: Option<orla_gpu::present::Display>,
    /// What reading the scene said.
    pub events: Vec<Event>,
    /// The backends tried first and passed over, as `name: why`.
    pub passed_over: Vec<String>,
}

/// Parses an allowance given on a command line: `file.read=<dir>` or
/// `file.write=<dir>`.
pub fn parse_allow(text: &str) -> Result<Allowance, String> {
    match text.split_once('=') {
        Some(("file.read", dir)) if !dir.is_empty() => Ok(Allowance::FileRead(PathBuf::from(dir))),
        Some(("file.write", dir)) if !dir.is_empty() => {
            Ok(Allowance::FileWrite(PathBuf::from(dir)))
        }
        _ => Err(format!(
            "--allow takes file.read=<directory> or file.write=<directory>, found `{text}`"
        )),
    }
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

/// The default word budget of a session's Host: two gibibytes. It is a
/// net that stops a runaway scene before the machine swaps; `--memory`
/// raises it.
pub const DEFAULT_BUDGET_WORDS: usize = 512 * 1024 * 1024;

/// The words in a number of mebibytes, which is how `--memory` asks.
pub fn budget_of_megabytes(megabytes: usize) -> usize {
    megabytes.saturating_mul(1024 * 1024 / 4)
}

/// What a session's Host may hold and spend: Field words held at once
/// (`--memory`), and loop iterations one Kernel run may take across all
/// its elements (`--work`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Words the Host may hold at once.
    pub words: usize,
    /// Loop iterations one run may take.
    pub work: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            words: DEFAULT_BUDGET_WORDS,
            work: orla_kernel::DEFAULT_WORK,
        }
    }
}

impl Limits {
    /// Whether `flag` sets a limit: `--memory` or `--work`.
    pub fn takes(flag: &str) -> bool {
        matches!(flag, "--memory" | "--work")
    }

    /// Sets the limit `flag` names from its value: `--memory <MB>` in
    /// whole mebibytes, `--work <M>` in millions of iterations.
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
                let most = orla_kernel::MAX_WORK / 1_000_000;
                if millions > most {
                    return Err(format!(
                        "--work is at most {most} million iterations, found {millions}"
                    ));
                }
                self.work = millions * 1_000_000;
            }
            _ => return Err(format!("{flag} sets no limit")),
        }
        Ok(())
    }
}

/// What is wrong with a document, and where when known. The position is
/// data an interface hands an agent as is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    /// What is wrong.
    pub message: String,
    /// Line and column, from 1, when the problem has a place.
    pub at: Option<(u32, u32)>,
}

impl Diagnostic {
    /// A diagnostic with no place.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            at: None,
        }
    }

    /// A diagnostic at a place in its document.
    pub fn at(span: orla_language::Span, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            at: Some((span.line, span.column)),
        }
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.at {
            Some((line, column)) => write!(f, "{line}:{column}: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl From<&orla_language::ParseError> for Diagnostic {
    fn from(error: &orla_language::ParseError) -> Self {
        Self::at(error.span, error.message.clone())
    }
}

impl From<&orla_scene::DocumentError> for Diagnostic {
    fn from(error: &orla_scene::DocumentError) -> Self {
        Self::at(error.span, error.message.clone())
    }
}

impl From<&orla_compiler::CompileError> for Diagnostic {
    fn from(error: &orla_compiler::CompileError) -> Self {
        match error {
            orla_compiler::CompileError::Parse(error) => error.into(),
            orla_compiler::CompileError::Semantic { span, message } => {
                Self::at(*span, message.clone())
            }
            other => Self::new(other.to_string()),
        }
    }
}

impl From<String> for Diagnostic {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}

/// Something the session observed while synchronizing with the filesystem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// A changed Kernel compiled and is being prepared; the running
    /// version stays until a later sync publishes the replacement.
    Preparing {
        /// Document that defines it.
        path: PathBuf,
        /// Kernel id.
        id: KernelId,
    },
    /// A Kernel was compiled and published.
    Published {
        /// Document that defines it.
        path: PathBuf,
        /// Kernel id.
        id: KernelId,
        /// New generation.
        generation: Generation,
    },
    /// A changed Kernel was refused; the previous version stays.
    Rejected {
        /// Document that defines it.
        path: PathBuf,
        /// Diagnostic, placed in that document when it has a place.
        error: Diagnostic,
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
    /// An input's text was written into the Host as rows of character
    /// codes.
    Text {
        /// The input.
        name: String,
        /// How many characters.
        chars: usize,
    },
    /// An input's file was read into the Host.
    Loaded {
        /// The input.
        name: String,
        /// The file.
        path: PathBuf,
        /// What came in: shape, extent, layout, and what the file adds
        /// (a sound's rate).
        detail: String,
    },
    /// The changed scene document was refused; the previous graph stays.
    SceneRejected(Diagnostic),
    /// An input's file is absent; the input reads its fallback.
    Absent {
        /// The input.
        name: String,
        /// The file.
        path: PathBuf,
    },
    /// An output was written to its file after a committed evaluation.
    Wrote {
        /// The output.
        output: String,
        /// The file.
        path: PathBuf,
        /// How many bytes.
        bytes: usize,
    },
    /// An output's write was refused; said once until the reason changes
    /// or a write succeeds.
    WriteRefused {
        /// The output.
        output: String,
        /// The file.
        path: PathBuf,
        /// Why.
        why: String,
    },
    /// A node runs off the session's main backend, and why. Said once
    /// per placement change.
    Routed {
        /// The node.
        node: String,
        /// The backend that runs it.
        backend: &'static str,
        /// Why it leaves the main backend.
        why: String,
    },
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preparing { path, id } => write!(f, "preparing {id} from {}", path.display()),
            Self::Published {
                path,
                id,
                generation,
            } => write!(f, "published {id} {generation} from {}", path.display()),
            Self::Rejected { path, error } => match error.at {
                Some(_) => write!(f, "rejected {}:{error}", path.display()),
                None => write!(f, "rejected {}: {error}", path.display()),
            },
            Self::Retired { path, id } => write!(f, "retired {id} ({})", path.display()),
            Self::SceneLoaded => f.write_str("scene loaded"),
            Self::Loaded { name, path, detail } => {
                write!(f, "loaded {name} from {}: {detail}", path.display())
            }
            Self::Text { name, chars } => write!(f, "text {name}: {chars} characters"),
            Self::SceneRejected(error) => write!(f, "scene rejected: {error}"),
            Self::Absent { name, path } => write!(
                f,
                "input `{name}`: {} is not there; reading its fallback",
                path.display()
            ),
            Self::Wrote {
                output,
                path,
                bytes,
            } => write!(f, "wrote `{output}` to {} ({bytes} bytes)", path.display()),
            Self::WriteRefused { output, path, why } => {
                write!(f, "did not write `{output}` to {}: {why}", path.display())
            }
            Self::Routed { node, backend, why } => {
                write!(f, "node `{node}` runs on {backend} ({why})")
            }
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
    Host(orla_kernel::HostError),
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
