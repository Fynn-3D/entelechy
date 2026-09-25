//! A live session over one `.osn` document and the libraries it uses.
//!
//! The document is the graph: the session parses it, publishes its inline
//! Kernels, follows its `use` items to `.orla` libraries, and builds the
//! Graph. On every [`Session::sync`] it re-reads what changed on disk and
//! republishes only the Kernels whose text changed, keeping a running
//! version whenever a replacement is rejected. Source resolution lives here,
//! not in the runtimes.
//!
//! A session is four parts, each owning its own state and reaching the
//! others only through what it is handed: the [scene](scene) reads the
//! documents and lowers the Graph; the [ledger](kernels) knows which
//! document published which Kernel and publishes through the runtime it
//! is given; the [inputs](inputs) own the Fields a scene's inputs hold in
//! the Host; and [source resolution](sources) holds nothing at all. The
//! [session](session) keeps the runtime and the evaluator and says in
//! what order the parts sync; its fields are private to its own module,
//! so no part reaches them. What a caller [observes](observe) is read
//! from the parts it is handed, never from a session.
//!
//! This crate is what every interface drives: the command line, the
//! Player, the C ABI, the importer and the exporter each open a
//! [`Session`], and none of them depends on another. The wire an agent
//! speaks to it is `orla-protocol`, and the file formats it reads and
//! writes are `orla-formats`; neither is the session's own.

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
mod session;
mod sources;
pub mod store;
mod writes;

use std::fmt;
use std::path::PathBuf;

use orla_formats::ppm;
use orla_runtime::{Generation, KernelId};

pub use history::{Attempt, Change, Origin, Record};
pub use options::SessionOptions;
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
    /// Every published Kernel: its id, generation and source, so the
    /// snapshot needs no memory of the session's to be read.
    pub laws: Vec<(KernelId, Generation, String)>,
    /// Every port's committed state.
    pub states: Vec<StateCopy>,
    /// The scene document the running graph was built from, when one
    /// was: the structure, which a restore may bring back too.
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
    /// The laws that run now: the snapshot's state goes under them,
    /// migrated where a layout changed.
    Current,
    /// The snapshot's own: its Kernels are published back with it.
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
    pub schema: orla_runtime::Schema,
    /// The extent.
    pub domain: orla_runtime::Domain,
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
/// element, which only the interpreter keeps, so an interface reads it
/// here rather than reaching for that backend itself.
pub use orla_cpu::{CALL_RECORDS, Trace, TraceValue};

/// What the person running a session allows its scene beyond its own
/// directory: given on a command line, never by what the session runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Allowance {
    /// Inputs may be read from files in this directory and under it.
    FileRead(PathBuf),
    /// Outputs may be written to files in this directory and under it.
    FileWrite(PathBuf),
}

pub use check::{Checked, check_document};

/// A runtime built for a session ([`Session::build`]): the runtime, the
/// display a device shows what stays on it through, and the backends
/// `best` passed over for it, each with why.
pub struct Built {
    /// The runtime.
    pub runtime: orla_runtime::Runtime,
    /// The device's display, when the backend has one.
    pub display: Option<orla_gpu::present::Display>,
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

/// How many words a session's Host may hold at once unless it is told
/// otherwise: two gibibytes of them. A budget is a net and not a
/// policy — it stops a scene that has run away before the machine
/// starts swapping — so the default is generous enough for a scene
/// with millions of triangles in it and `--memory` states the policy
/// where a scene wants more.
pub const DEFAULT_BUDGET_WORDS: usize = 512 * 1024 * 1024;

/// The words in a number of mebibytes, which is how `--memory` asks.
pub fn budget_of_megabytes(megabytes: usize) -> usize {
    megabytes.saturating_mul(1024 * 1024 / 4)
}

/// What a session's Host may hold and spend: the words of Fields it
/// holds at once (`--memory`), and the loop iterations one run of a
/// Kernel may take, all its elements together (`--work`). Both are nets
/// and not policies, and a scene that wants more says so.
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
            work: orla_runtime::DEFAULT_WORK,
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
                let most = orla_runtime::MAX_WORK / 1_000_000;
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

/// What is wrong with a document, and where in it when that is known:
/// the line and column from 1. Carried as it was found, so an interface
/// hands an agent the position as data rather than reading it back out
/// of a sentence.
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
    /// A changed Kernel compiled and went to the preparation worker; the
    /// running version stays until a later sync publishes the replacement.
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
    /// A Kernel changed but was not accepted; the previous version stays.
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
    /// An input's text was written into the Host as rows of its
    /// characters' codes, at first or after it changed.
    Text {
        /// The input.
        name: String,
        /// How many characters.
        chars: usize,
    },
    /// An input's file was read into the Host, at first or after it changed.
    Loaded {
        /// The input.
        name: String,
        /// The file.
        path: PathBuf,
        /// What came in: shape, extent, layout, and what the file adds
        /// (a sound's rate).
        detail: String,
    },
    /// The scene document changed but does not parse; the previous graph stays.
    SceneRejected(Diagnostic),
    /// An input's file is not there, and the input reads its fallback
    /// until it is.
    Absent {
        /// The input.
        name: String,
        /// The file.
        path: PathBuf,
    },
    /// An output was written to its file, after an evaluation that
    /// changed it committed.
    Wrote {
        /// The output.
        output: String,
        /// The file.
        path: PathBuf,
        /// How many bytes.
        bytes: usize,
    },
    /// An output was not written to its file, and why; said once until
    /// the reason changes or a write succeeds.
    WriteRefused {
        /// The output.
        output: String,
        /// The file.
        path: PathBuf,
        /// Why.
        why: String,
    },
    /// A node will run off the session's main backend, and why: what it
    /// costs a frame to take its inputs there and its outputs back is
    /// the scene's to know before it measures anything. Said once, when
    /// the node's placement changes.
    Routed {
        /// The node.
        node: String,
        /// The backend that runs it.
        backend: &'static str,
        /// Why the main backend does not.
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
    Host(orla_runtime::HostError),
    /// A resident image could not be brought back for observation.
    Observe(String),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(path, error) => write!(f, "{}: {error}", path.display()),
            Self::Image(path, error) => write!(f, "{}: {error}", path.display()),
            Self::NoScene => f.write_str("no scene has been loaded"),
            Self::Host(error) => write!(f, "{error}"),
            Self::Observe(message) => write!(f, "cannot observe image: {message}"),
        }
    }
}

impl std::error::Error for SessionError {}
