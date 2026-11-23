//! The documents a session runs: one `.esn` scene and the `.ent`
//! libraries it uses, what they publish, the Graph they lower to, and the
//! files they read and write.
//!
//! Four parts each own their state and reach the others only through what
//! they are handed: the [scene](scene) reads documents and lowers the
//! Graph; the [ledger](kernels) knows which document published which
//! Kernel; the [inputs](inputs) own the Fields a scene's inputs hold; and
//! [source resolution](sources) is stateless. The [store](store) is where
//! documents are read from, the [history](history) what changed them.
//!
//! Nothing here runs a Kernel: `entelechy-session` opens the backends,
//! keeps the runtime and evaluator, and orders the parts' sync.

#![warn(missing_docs)]

#[cfg(feature = "documents")]
pub mod check;
#[cfg(feature = "documents")]
pub mod devices;
#[cfg(feature = "documents")]
pub mod history;
#[cfg(feature = "documents")]
pub mod ignore;
#[cfg(feature = "documents")]
pub mod inputs;
#[cfg(feature = "documents")]
pub mod kernels;
#[cfg(feature = "documents")]
pub mod overrides;
#[cfg(feature = "documents")]
pub mod package;
#[cfg(feature = "documents")]
pub mod reads;
#[cfg(feature = "documents")]
pub mod scene;
#[cfg(feature = "documents")]
pub mod sources;
#[cfg(feature = "documents")]
pub mod spread;
pub mod store;
#[cfg(feature = "documents")]
pub mod writes;

#[cfg(feature = "documents")]
mod items {
    use std::fmt;
    use std::path::PathBuf;

    use entelechy_kernel::{Generation, KernelId};

    use crate::check;

    /// Where in its document a Kernel's failure is written.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct SourcePlace {
        /// The document.
        pub path: PathBuf,
        /// Its line, 1-based.
        pub line: u32,
        /// Its column in characters, 1-based.
        pub column: u32,
        /// The function it is in, `None` the Kernel's body.
        pub function: Option<String>,
    }

    impl fmt::Display for SourcePlace {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}:{}:{}", self.path.display(), self.line, self.column)?;
            if let Some(function) = &self.function {
                write!(f, ", in `{function}`")?;
            }
            Ok(())
        }
    }

    /// What the person running a session allows its scene beyond its own
    /// directory. Only a command line grants it; the scene never does.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum Allowance {
        /// Inputs may be read from files in this directory and under it.
        FileRead(PathBuf),
        /// Outputs may be written to files in this directory and under it.
        FileWrite(PathBuf),
        /// Files in this directory and under it may be removed and renamed
        /// by an output `to files`.
        FileDelete(PathBuf),
        /// An input `= clipboard` may read the system's clipboard.
        ClipboardRead,
        /// An output `to clipboard` may write the system's clipboard.
        ClipboardWrite,
    }

    /// Parses an allowance given on a command line: `file.read=<dir>`,
    /// `file.write=<dir>`, `file.delete=<dir>`, `clipboard.read` or
    /// `clipboard.write`.
    pub fn parse_allow(text: &str) -> Result<Allowance, String> {
        match text {
            "clipboard.read" => return Ok(Allowance::ClipboardRead),
            "clipboard.write" => return Ok(Allowance::ClipboardWrite),
            _ => {}
        }
        match text.split_once('=') {
            Some(("file.read", dir)) if !dir.is_empty() => {
                Ok(Allowance::FileRead(PathBuf::from(dir)))
            }
            Some(("file.write", dir)) if !dir.is_empty() => {
                Ok(Allowance::FileWrite(PathBuf::from(dir)))
            }
            Some(("file.delete", dir)) if !dir.is_empty() => {
                Ok(Allowance::FileDelete(PathBuf::from(dir)))
            }
            _ => Err(format!(
                "--allow takes file.read=<directory>, file.write=<directory>, file.delete=<directory>, clipboard.read or clipboard.write, found `{text}`"
            )),
        }
    }

    /// A change of several documents refused: the document at fault and why.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Refusal {
        /// The document, by its one name.
        pub path: std::path::PathBuf,
        /// Why, and where in it.
        pub diagnostic: Diagnostic,
    }

    /// What is wrong with a document, and where when known. The position is
    /// data an interface hands an agent as is.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Diagnostic {
        /// What is wrong.
        pub message: String,
        /// Line and column, from 1, when the problem has a place.
        pub at: Option<(u32, u32)>,
        /// The change that mends it, when one change does: data an agent
        /// hands back as a compose.
        pub fix: Option<Box<Fix>>,
    }

    /// A change that mends a diagnostic: `using` written as a `use` item of
    /// `document` (`use "{using}";`), so the document sees what it names.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Fix {
        /// The document to change, by its one name.
        pub document: std::path::PathBuf,
        /// The path its new `use` item names, spelled as the document would.
        pub using: String,
        /// The document whole with the item added, when the refused change
        /// wrote it: what is on disk is not what the change asked for.
        pub text: Option<String>,
    }

    impl Fix {
        /// The `use` item the fix adds.
        pub fn item(&self) -> String {
            format!("use \"{}\";", self.using)
        }
    }

    impl Diagnostic {
        /// A diagnostic with no place.
        pub fn new(message: impl Into<String>) -> Self {
            Self {
                message: message.into(),
                at: None,
                fix: None,
            }
        }

        /// A diagnostic at a place in its document; one whose span is no
        /// place has none.
        pub fn at(span: entelechy_language::Span, message: impl Into<String>) -> Self {
            Self {
                message: message.into(),
                at: span.is_placed().then_some((span.line, span.column)),
                fix: None,
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

    impl From<&entelechy_language::ParseError> for Diagnostic {
        fn from(error: &entelechy_language::ParseError) -> Self {
            Self::at(error.span, error.message.clone())
        }
    }

    impl From<&entelechy_scene::DocumentError> for Diagnostic {
        fn from(error: &entelechy_scene::DocumentError) -> Self {
            Self::at(error.span, error.message.clone())
        }
    }

    impl From<&entelechy_compiler::CompileError> for Diagnostic {
        fn from(error: &entelechy_compiler::CompileError) -> Self {
            match error {
                entelechy_compiler::CompileError::Parse(error) => error.into(),
                entelechy_compiler::CompileError::Semantic { span, message, .. } => {
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
        /// A file was removed by an output's operations.
        Removed {
            /// The output.
            output: String,
            /// The file.
            path: PathBuf,
        },
        /// A file was renamed by an output's operations.
        Renamed {
            /// The output.
            output: String,
            /// The file as it was.
            from: PathBuf,
            /// The file as it is.
            to: PathBuf,
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
        /// A Kernel just published is large ([`large`](check::large)): what it carries
        /// through its branches, the structs and the functions that carry
        /// most. Said for each version published so.
        Large {
            /// Document that defines it.
            path: PathBuf,
            /// Kernel id.
            id: KernelId,
            /// What makes it so.
            why: String,
        },
        /// A node with no `domain` whose Kernel compares its element with its
        /// first input's count, so the elements past that count, which the
        /// test is for, are never run ([`check::defaulted_domain`]). Said once
        /// when the scene loads, and again when it changes.
        Unrun {
            /// The node.
            node: String,
            /// What is said of it.
            why: String,
        },
        /// A large output written to a file was brought to the Host as a new
        /// Field on each of `frames` evaluations in a row and found each time
        /// to be the file as it stands: the node it reads runs every time and
        /// makes the same, and each time costs the Field fetched, encoded and
        /// hashed. Said once, until the output is written or left as it was.
        BroughtUnchanged {
            /// The output.
            output: String,
            /// The file.
            path: PathBuf,
            /// The bytes brought each time.
            bytes: usize,
            /// The evaluations in a row.
            frames: u32,
        },
        /// Kernels a document of the scene publishes and no node uses, each
        /// compiled and prepared all the same. Said when the scene loads, and
        /// again when they change.
        Idle {
            /// The document.
            path: PathBuf,
            /// The Kernels.
            ids: Vec<KernelId>,
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
                Self::Removed { output, path } => {
                    write!(f, "`{output}` removed {}", path.display())
                }
                Self::Renamed { output, from, to } => write!(
                    f,
                    "`{output}` renamed {} to {}",
                    from.display(),
                    to.display()
                ),
                Self::WriteRefused { output, path, why } => {
                    write!(f, "did not write `{output}` to {}: {why}", path.display())
                }
                Self::Routed { node, backend, why } => {
                    write!(f, "node `{node}` runs on {backend} ({why})")
                }
                Self::Large { path, why, .. } => write!(f, "large: {}: {why}", path.display()),
                Self::Unrun { why, .. } => f.write_str(why),
                Self::BroughtUnchanged {
                    output,
                    path,
                    bytes,
                    frames,
                } => write!(
                    f,
                    "output `{output}` was brought to the Host to be written to {} on each of the last {frames} frames, {:.1} MiB each time, and found unchanged: what it reads runs every frame; a node that sleeps (`awake`) until the save is due gives the same Field until then, and nothing is brought",
                    path.display(),
                    *bytes as f64 / (1 << 20) as f64
                ),
                Self::Idle { path, ids } => f.write_str(&check::idle_said(path, ids)),
            }
        }
    }
}

#[cfg(feature = "documents")]
pub use items::*;
