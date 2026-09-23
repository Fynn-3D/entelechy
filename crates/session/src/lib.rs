//! A live session over one `.osn` document and the libraries it uses.
//!
//! The document is the graph: the session parses it, publishes its inline
//! Kernels, follows its `use` items to `.orla` libraries, and builds the
//! Graph. On every [`Session::sync`] it re-reads what changed on disk and
//! republishes only the Kernels whose text changed, keeping a running
//! version whenever a replacement is rejected. Source resolution lives here,
//! not in the runtimes.
//!
//! This crate is what every interface drives: the command line, the
//! Player, the C ABI, the importer and the exporter each open a
//! [`Session`] and speak the [`protocol`] to it, and none of them
//! depends on another. The file formats a session reads and writes
//! travel with it.

#![warn(missing_docs)]

pub mod bench;
pub mod inflate;
pub mod png;
pub mod ppm;
pub mod protocol;
pub mod wav;
pub mod websocket;

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use orla_graph::{EvaluateError, Evaluator, Graph, Report};
use orla_language::{Document, Kernel};
use orla_runtime::{
    Backend, Context, Expect, FieldData, Generation, Handle, KernelId, Runtime, Shape, Ticket,
    Value,
};

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
    SceneRejected(String),
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
            Self::Rejected { path, error } => write!(f, "rejected {}:{error}", path.display()),
            Self::Retired { path, id } => write!(f, "retired {id} ({})", path.display()),
            Self::SceneLoaded => f.write_str("scene loaded"),
            Self::Loaded { name, path, detail } => {
                write!(f, "loaded {name} from {}: {detail}", path.display())
            }
            Self::Text { name, chars } => write!(f, "text {name}: {chars} characters"),
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

/// A file an input is read from.
struct Asset {
    path: PathBuf,
    stamp: Option<Stamp>,
}

/// An input read frame by frame from a directory: its files in name
/// order, and which of them the Host holds now.
struct Sequence {
    dir: PathBuf,
    ty: String,
    rate: u32,
    files: Vec<PathBuf>,
    current: Option<usize>,
}

/// The file extensions a frame may have.
const FRAME_EXTENSIONS: [&str; 5] = ["png", "ppm", "pgm", "field", "wav"];

/// The files of `dir` that can be frames, in name order.
fn frame_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| FRAME_EXTENSIONS.contains(&e))
        })
        .collect();
    files.sort();
    Ok(files)
}

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

/// The format of an `.orlapkg` archive and of a package's manifest.
pub const PACKAGE_VERSION: u32 = 1;

/// A `use` target: a file; every `.orla` directly inside a directory; or
/// a package, a directory with a `package.osn` (found beside the
/// document, or by name in the directories `ORLA_PACKAGES` lists),
/// whose manifest's own `use`s are followed once its versions fit.
fn expand_use(path: &Path, trailing_slash: bool) -> Result<Vec<PathBuf>, String> {
    // One file, one path: a Kernel reached by two spellings (a scene's
    // `use` of a package and its preset's `use "../kernels/"`, say) is
    // owned by one document, not published twice.
    let canonical = |file: PathBuf| file.canonicalize().unwrap_or(file);
    let path = resolve_package(path);
    // A `use` written inside a package, reached by the package's name:
    // the way through it is the same, wherever the package itself is.
    let path = match path.exists() {
        true => path,
        false => in_package(&path).unwrap_or(path),
    };
    if path.join("package.osn").is_file() {
        return package_files(&path).map(|files| files.into_iter().map(canonical).collect());
    }
    if trailing_slash || path.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&path)
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .map(|entry| entry.path())
                    .filter(|path| path.extension().is_some_and(|ext| ext == "orla"))
                    .collect()
            })
            .unwrap_or_default();
        files.sort();
        Ok(files.into_iter().map(canonical).collect())
    } else {
        Ok(vec![canonical(path.to_path_buf())])
    }
}

/// A `use` that names nothing beside the document is looked for by its
/// A document a scene reaches: the file at that path, or, where there
/// is none and the path runs through a package's name, the file inside
/// the package `ORLA_PACKAGES` finds. A preset is reached the way the
/// package that holds it is.
pub fn read_document(path: &Path) -> Result<String, String> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(text),
        Err(missing) => match in_package(path) {
            Some(inside) => std::fs::read_to_string(inside).map_err(|error| error.to_string()),
            None => Err(missing.to_string()),
        },
    }
}

/// A path whose first part names a package: `picture.orlapkg/graphs/x.osn`
/// where the package is one `ORLA_PACKAGES` lists. `None` when no part of
/// it names a package that is there.
fn in_package(path: &Path) -> Option<PathBuf> {
    let mut parts = path.components();
    let mut rest = PathBuf::new();
    // The package's name is the last part that ends in `.orlapkg`; what
    // follows it is the way to the file inside.
    let mut name: Option<PathBuf> = None;
    for part in parts.by_ref() {
        let piece = Path::new(part.as_os_str());
        if piece.extension().is_some_and(|ext| ext == "orlapkg") {
            name = Some(piece.to_path_buf());
            rest = PathBuf::new();
        } else if name.is_some() {
            rest.push(piece);
        }
    }
    let name = name?;
    let root = resolve_package(&name);
    let inside = root.join(rest);
    inside.exists().then_some(inside)
}

/// name in each directory of `ORLA_PACKAGES`, `:`-separated.
fn resolve_package(path: &Path) -> PathBuf {
    if path.exists() {
        return path.to_path_buf();
    }
    let Some(name) = path.file_name() else {
        return path.to_path_buf();
    };
    let Some(dirs) = std::env::var_os("ORLA_PACKAGES") else {
        return path.to_path_buf();
    };
    for dir in std::env::split_paths(&dirs) {
        let candidate = dir.join(name);
        if candidate.join("package.osn").is_file() {
            return candidate;
        }
    }
    path.to_path_buf()
}

/// The Kernel files a package brings in: what its manifest uses, once
/// the manifest's versions fit this Orla.
fn package_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let manifest = dir.join("package.osn");
    let declared = package_manifest(&manifest)?;
    let text =
        std::fs::read_to_string(&manifest).map_err(|e| format!("{}: {e}", manifest.display()))?;
    let document =
        orla_language::parse_document(&text).map_err(|e| format!("{}:{e}", manifest.display()))?;
    let mut files = Vec::new();
    for item in &document.uses {
        let path = dir.join(&item.path.value);
        if path.join("package.osn").is_file() && path.canonicalize().ok() == dir.canonicalize().ok()
        {
            return Err(format!(
                "{}: a package cannot use itself",
                declared.id.value
            ));
        }
        files.extend(expand_use(&path, item.path.value.ends_with('/'))?);
    }
    Ok(files)
}

/// A package's manifest, checked: it declares a package, and needs no
/// newer language or IR than this Orla has.
pub fn package_manifest(manifest: &Path) -> Result<orla_language::PackageDecl, String> {
    let text =
        std::fs::read_to_string(manifest).map_err(|e| format!("{}: {e}", manifest.display()))?;
    let document =
        orla_language::parse_document(&text).map_err(|e| format!("{}:{e}", manifest.display()))?;
    let Some(declared) = document.package else {
        return Err(format!(
            "{}: a package manifest begins `package name@version {{ language = N; ir = N; }}`",
            manifest.display()
        ));
    };
    if declared.language > orla_language::LANGUAGE_VERSION {
        return Err(format!(
            "package {} needs language {}, and this Orla has language {}",
            declared.id.value,
            declared.language,
            orla_language::LANGUAGE_VERSION
        ));
    }
    if declared.ir > orla_ir::IR_VERSION {
        return Err(format!(
            "package {} needs ir {}, and this Orla has ir {}",
            declared.id.value,
            declared.ir,
            orla_ir::IR_VERSION
        ));
    }
    Ok(declared)
}

/// What a document shares with the Kernels that use it: structs, type
/// aliases, and functions.
pub type Declarations = (
    Vec<orla_language::StructDecl>,
    Vec<orla_language::TypeAlias>,
    Vec<orla_language::FnDecl>,
);

/// Every document a `use` list reaches, recursively, each once, in the
/// order reached: what a check compiles beyond the file it was given.
pub fn used_documents(base: &Path, uses: &[orla_language::Use]) -> Result<Vec<PathBuf>, String> {
    fn gather(
        base: &Path,
        uses: &[orla_language::Use],
        visited: &mut std::collections::BTreeSet<PathBuf>,
        out: &mut Vec<PathBuf>,
    ) -> Result<(), String> {
        for item in uses {
            let path = base.join(&item.path.value);
            for file in expand_use(&path, item.path.value.ends_with('/'))? {
                let key = file.canonicalize().unwrap_or_else(|_| file.clone());
                if !visited.insert(key) {
                    continue;
                }
                let text = std::fs::read_to_string(&file)
                    .map_err(|e| format!("{}: {e}", file.display()))?;
                let document = orla_language::parse_document(&text)
                    .map_err(|e| format!("{}:{e}", file.display()))?;
                let dir = file
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .to_path_buf();
                gather(&dir, &document.uses, visited, out)?;
                out.push(file);
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    gather(base, uses, &mut std::collections::BTreeSet::new(), &mut out)?;
    Ok(out)
}

/// The structs and type aliases a document's `use` items bring in, from
/// the used files and directories and, recursively, what they use. The
/// same declaration reached twice is one; two different declarations of
/// one name are refused.
pub fn declarations_for(base: &Path, uses: &[orla_language::Use]) -> Result<Declarations, String> {
    declarations_within(None, base, uses)
}

/// The declarations a document at `origin` inherits from what it uses.
/// Naming the origin is what lets a `use` that comes back to it be
/// reported as the cycle it is rather than as a name declared twice.
pub fn declarations_within(
    origin: Option<&Path>,
    base: &Path,
    uses: &[orla_language::Use],
) -> Result<Declarations, String> {
    #[allow(clippy::too_many_arguments)]
    fn gather(
        base: &Path,
        uses: &[orla_language::Use],
        visited: &mut std::collections::BTreeSet<PathBuf>,
        stack: &mut Vec<PathBuf>,
        structs: &mut Vec<(orla_language::StructDecl, PathBuf)>,
        types: &mut Vec<(orla_language::TypeAlias, PathBuf)>,
        functions: &mut Vec<(orla_language::FnDecl, PathBuf)>,
    ) -> Result<(), String> {
        for item in uses {
            let path = base.join(&item.path.value);
            for file in expand_use(&path, item.path.value.ends_with('/'))? {
                let key = file.canonicalize().unwrap_or_else(|_| file.clone());
                if let Some(at) = stack.iter().position(|open| *open == key) {
                    let names: Vec<String> = stack[at..]
                        .iter()
                        .chain(std::iter::once(&key))
                        .map(|path| {
                            path.file_name()
                                .map(|name| name.to_string_lossy().into_owned())
                                .unwrap_or_else(|| path.display().to_string())
                        })
                        .collect();
                    let cycle = if names.len() == 3 && names[0] == names[2] {
                        format!("`{}` and `{}` use each other", names[0], names[1])
                    } else {
                        format!("`{}` use one another in a ring", names.join("` -> `"))
                    };
                    return Err(format!(
                        "{cycle}; what two documents share belongs in a third they both use"
                    ));
                }
                if !visited.insert(key.clone()) {
                    continue;
                }
                stack.push(key);
                let text = std::fs::read_to_string(&file)
                    .map_err(|e| format!("{}: {e}", file.display()))?;
                let document = orla_language::parse_document(&text)
                    .map_err(|e| format!("{}:{e}", file.display()))?;
                let dir = file
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .to_path_buf();
                let nested = gather(
                    &dir,
                    &document.uses,
                    visited,
                    stack,
                    structs,
                    types,
                    functions,
                );
                stack.pop();
                nested?;
                for decl in &document.functions {
                    // A function has no layout to compare; one name, one file.
                    if let Some((_, other)) = functions
                        .iter()
                        .find(|(f, _)| f.name.value == decl.name.value)
                    {
                        return Err(format!(
                            "fn `{}` is declared by both {} and {}",
                            decl.name.value,
                            other.display(),
                            file.display()
                        ));
                    }
                    functions.push((decl.clone(), file.clone()));
                }
                for decl in &document.structs {
                    match structs
                        .iter()
                        .find(|(s, _)| s.name.value == decl.name.value)
                    {
                        Some((same, _)) if same.fields == decl.fields => {}
                        Some((_, other)) => {
                            return Err(format!(
                                "struct `{}` is declared differently by {} and {}",
                                decl.name.value,
                                other.display(),
                                file.display()
                            ));
                        }
                        None => structs.push((decl.clone(), file.clone())),
                    }
                }
                for decl in &document.types {
                    match types.iter().find(|(t, _)| t.name.value == decl.name.value) {
                        Some((same, _)) if same.target.value == decl.target.value => {}
                        Some((_, other)) => {
                            return Err(format!(
                                "type `{}` is declared differently by {} and {}",
                                decl.name.value,
                                other.display(),
                                file.display()
                            ));
                        }
                        None => types.push((decl.clone(), file.clone())),
                    }
                }
            }
        }
        Ok(())
    }
    let mut visited = std::collections::BTreeSet::new();
    let mut stack: Vec<PathBuf> = origin
        .map(|path| vec![path.canonicalize().unwrap_or_else(|_| path.to_path_buf())])
        .unwrap_or_default();
    let (mut structs, mut types, mut functions) = (Vec::new(), Vec::new(), Vec::new());
    gather(
        base,
        uses,
        &mut visited,
        &mut stack,
        &mut structs,
        &mut types,
        &mut functions,
    )?;
    Ok((
        structs.into_iter().map(|(s, _)| s).collect(),
        types.into_iter().map(|(t, _)| t).collect(),
        functions.into_iter().map(|(f, _)| f).collect(),
    ))
}

/// Kernels a document defines: id, source text, and the parsed Kernel.
struct Defined {
    id: KernelId,
    text: String,
    kernel: Kernel,
}

/// A scene, the libraries it uses, and the runtime state built from them.
pub struct Session {
    root: PathBuf,
    extra_uses: Vec<PathBuf>,
    runtime: Runtime,
    evaluator: Evaluator,
    graph: Option<Graph>,
    root_stamp: Option<Stamp>,
    /// `use` targets from the last good parse of the root and its
    /// sub-graphs, resolved.
    root_uses: Vec<(PathBuf, bool)>,
    /// Sub-graph documents placed by the root, with the stamps they were
    /// loaded at; a change to any of them rebuilds the graph.
    sub_graphs: BTreeMap<PathBuf, Option<Stamp>>,
    /// The scene's shared declarations, from its `use`s and its own top
    /// level, which a Kernel published through the protocol inherits.
    shared_structs: Vec<orla_language::StructDecl>,
    shared_types: Vec<orla_language::TypeAlias>,
    shared_functions: Vec<orla_language::FnDecl>,
    /// Caller overrides of graph parameters.
    params: BTreeMap<String, Value>,
    /// Kernel text by id for every document that published Kernels; the
    /// root document is one of them.
    documents: BTreeMap<PathBuf, DocumentState>,
    owners: BTreeMap<KernelId, PathBuf>,
    externals: BTreeMap<String, Handle>,
    /// Inputs read from files, with the stamp each was last read at.
    assets: BTreeMap<String, Asset>,
    /// Inputs read frame by frame from directories.
    sequences: BTreeMap<String, Sequence>,
    /// What went wrong selecting a frame, reported by the next sync.
    frame_events: Vec<Event>,
    /// Inputs written as text, with the text each holds.
    texts: BTreeMap<String, String>,
    /// The window a headless session reports when an evaluation names
    /// none: its size in pixels and its scale factor.
    window: Option<([f32; 2], f32)>,
    /// Replacements of running Kernels the runtime is still preparing.
    pending: BTreeMap<KernelId, PendingPublish>,
}

struct PendingPublish {
    ticket: Ticket,
    path: PathBuf,
    expect: Expect,
}

#[derive(Default)]
struct DocumentState {
    stamp: Option<Stamp>,
    kernels: BTreeMap<KernelId, String>,
    /// The Kernels as they were parsed, which a trace compiles again
    /// without the optimizer so that it walks what the source says.
    asts: BTreeMap<KernelId, Kernel>,
    /// The shared declarations the document's Kernels were compiled
    /// with; when they change, every Kernel republishes even though its
    /// own text did not.
    declarations: String,
}

/// The name a diagnostic says it could not find: the first backticked
/// word of a message that says something is not a function, a struct or
/// a type.
fn missing_name(message: &str) -> Option<String> {
    let rest = message.rsplit(": ").next().unwrap_or(message);
    let name = rest.strip_prefix('`')?.split('`').next()?;
    let says = rest.contains("is not a builtin")
        || rest.contains("is not a struct")
        || rest.contains("is not a type")
        || rest.contains("is not a port type");
    (says && !name.is_empty()).then(|| name.to_owned())
}

/// Where a name is declared, among the documents `search` names: the
/// file, and whether it declares it as a function, a struct or a type.
fn declared_in(name: &str, search: &[PathBuf]) -> Option<(PathBuf, &'static str)> {
    for path in search {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let Ok(document) = orla_language::parse_document(&text) else {
            continue;
        };
        let kinds = [
            (
                "fn",
                document.functions.iter().any(|d| d.name.value == name),
            ),
            (
                "struct",
                document.structs.iter().any(|d| d.name.value == name),
            ),
            ("type", document.types.iter().any(|d| d.name.value == name)),
        ];
        if let Some((kind, _)) = kinds.into_iter().find(|(_, found)| *found) {
            return Some((path.clone(), kind));
        }
    }
    None
}

/// `path` as a `use` would spell it from `from`, when they share a root.
fn relative_to(path: &Path, from: &Path) -> String {
    let (path, from) = (
        path.canonicalize().unwrap_or_else(|_| path.to_path_buf()),
        from.canonicalize().unwrap_or_else(|_| from.to_path_buf()),
    );
    if let Ok(rest) = path.strip_prefix(&from) {
        return rest.display().to_string();
    }
    let mut up = PathBuf::new();
    let mut base = from.as_path();
    while let Some(parent) = base.parent() {
        up.push("..");
        if let Ok(rest) = path.strip_prefix(parent) {
            return up.join(rest).display().to_string();
        }
        base = parent;
    }
    path.display().to_string()
}

/// A vector's component by name, as a Kernel spells it (`x` or `r` and
/// the rest): which word it is, and what type it holds.
fn component_of(ty: orla_ir::Type, name: &str) -> Option<(usize, orla_ir::Type)> {
    use orla_ir::Type;
    let (lanes, element) = match ty {
        Type::Vec2 => (2, Type::Float),
        Type::Vec3 => (3, Type::Float),
        Type::Vec4 => (4, Type::Float),
        Type::IVec2 => (2, Type::Int),
        Type::IVec3 => (3, Type::Int),
        _ => return None,
    };
    let at = ["xyzw", "rgba"]
        .iter()
        .find_map(|alphabet| alphabet.find(name))?;
    (name.len() == 1 && at < lanes).then_some((at, element))
}

/// `node.port[index].field.field` as a replay writes it and the
/// protocol takes it: the node, the port, the element, and the path
/// through the struct to a value.
pub fn parse_target(text: &str) -> Result<(String, String, u32, Vec<String>), String> {
    let (head, rest) = text
        .split_once('[')
        .ok_or_else(|| "name an element as `node.port[index]`".to_string())?;
    let (index, tail) = rest
        .split_once(']')
        .ok_or_else(|| "name an element as `node.port[index]`".to_string())?;
    let (node, port) = head
        .rsplit_once('.')
        .ok_or_else(|| "name `node.port`".to_string())?;
    let index = index
        .parse::<u32>()
        .map_err(|_| format!("element index `{index}` is not a whole number"))?;
    let path: Vec<String> = tail
        .split('.')
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect();
    Ok((node.to_owned(), port.to_owned(), index, path))
}

/// A key for a set of shared declarations.
fn declarations_key(
    structs: &[orla_language::StructDecl],
    types: &[orla_language::TypeAlias],
    functions: &[orla_language::FnDecl],
) -> String {
    format!("{structs:?}{types:?}{functions:?}")
}

impl Session {
    /// Creates a CPU-backed session; call [`Session::sync`] to load files.
    /// `library` is an extra directory or file of Kernels to use.
    pub fn open(scene: &Path, library: Option<&Path>, budget_words: usize) -> Self {
        Self::open_with(
            scene,
            library,
            Runtime::with_backend(budget_words, Box::new(orla_cpu::CpuBackend)),
        )
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
            sub_graphs: BTreeMap::new(),
            shared_structs: Vec::new(),
            shared_types: Vec::new(),
            shared_functions: Vec::new(),
            params: BTreeMap::new(),
            documents: BTreeMap::new(),
            owners: BTreeMap::new(),
            pending: BTreeMap::new(),
            externals: BTreeMap::new(),
            assets: BTreeMap::new(),
            sequences: BTreeMap::new(),
            frame_events: Vec::new(),
            texts: BTreeMap::new(),
            window: None,
        }
    }

    /// The runtime, for inspection.
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Builds a runtime over a named backend: `cpu`, or `gpu` when a device
    /// is available.
    pub fn runtime_for(backend: &str, budget_words: usize) -> Result<Runtime, String> {
        match backend {
            "cpu" => Ok(Runtime::with_backend(
                budget_words,
                Box::new(orla_cpu::CpuBackend),
            )),
            "gpu" => {
                // A GPU walks a scan on one thread; the JIT takes the
                // scans, the GPU everything else.
                let gpu = orla_gpu::GpuBackend::new().map_err(|error| error.to_string())?;
                let backend: Box<dyn Backend> = Box::new(gpu);
                let ordered: Option<Box<dyn Backend>> = orla_jit::JitBackend::new()
                    .ok()
                    .map(|jit| Box::new(jit) as Box<dyn Backend>);
                Ok(Runtime::with_backends(budget_words, backend, ordered))
            }
            "jit" => {
                let jit = orla_jit::JitBackend::new()?;
                let backend: Box<dyn Backend> = Box::new(jit);
                Ok(Runtime::with_backend(budget_words, backend))
            }
            other => Err(format!("unknown backend `{other}`; use cpu, gpu, or jit")),
        }
    }

    /// Mutable runtime access for interfaces that need to observe images.
    pub fn runtime_mut(&mut self) -> &mut Runtime {
        &mut self.runtime
    }

    /// The current graph, if the scene loaded.
    pub fn graph(&self) -> Option<&Graph> {
        self.graph.as_ref()
    }

    /// The scene document.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Overrides a graph parameter for later evaluations, in the shape of
    /// its declared type: `[[1, 0], [0, 1]]` for a `Mat2`, an integer for a
    /// float. A value of another layout is refused. Unknown names are kept
    /// as given and take effect if the graph later declares them.
    pub fn set_param(&mut self, name: &str, value: Value) -> Result<(), String> {
        let declared = self
            .graph
            .as_ref()
            .and_then(|graph| graph.params.iter().find(|p| p.name == name));
        let value = match declared {
            Some(param) => value
                .shaped_like(&param.default)
                .ok_or_else(|| format!("param `{name}` is a {}; `{value}` is not one", param.ty))?,
            None => value,
        };
        let previous = self.params.insert(name.to_owned(), value.clone());
        // A parameter that sizes the graph (a node's `times`) changes its
        // structure: the document is lowered again on the next sync.
        if previous != Some(value)
            && self
                .graph
                .as_ref()
                .is_some_and(|graph| graph.times_params.iter().any(|p| p == name))
        {
            self.root_stamp = None;
        }
        Ok(())
    }

    /// Current graph parameter overrides.
    pub fn params(&self) -> &BTreeMap<String, Value> {
        &self.params
    }

    /// Every document the session currently watches.
    pub fn watched(&self) -> Vec<PathBuf> {
        self.documents
            .keys()
            .cloned()
            .chain(self.assets.values().map(|asset| asset.path.clone()))
            .chain(self.sequences.values().map(|sequence| sequence.dir.clone()))
            .collect()
    }

    /// Reads a file as its bytes: `Rows<Int>`, one element a byte, 0 to
    /// 255. What a Kernel does with them is the Kernel's business, which
    /// is how a scene reads a format nothing here knows.
    pub fn load_bytes(path: &Path) -> Result<(FieldData, String), String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        if bytes.is_empty() {
            return Err("the file is empty; a Field has no empty extent".into());
        }
        let words: Vec<u32> = bytes.iter().map(|byte| u32::from(*byte)).collect();
        let data = FieldData::from_storage(
            orla_runtime::Schema::new(vec![orla_runtime::Scalar::Int]),
            orla_runtime::Domain::Rows(words.len() as u32),
            words,
        )
        .map_err(|e| e.to_string())?;
        let detail = format!("{} byte(s)", bytes.len());
        Ok((data, detail))
    }

    /// Reads a Field file: `.field` as the Host writes it; `.png`, `.ppm`,
    /// and `.pgm` as a grid of pixels, or as a `Grid<Float>` of luminance
    /// when `want` says so; `.wav` as rows of one or two floats. Returns
    /// the Field and a line saying what came in.
    pub fn load_field(path: &Path, want: Option<&str>) -> Result<(FieldData, String), String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let gray = want.is_some_and(|ty| without_encoding(ty) == "Grid<Float>");
        let describe = |data: &FieldData| {
            format!(
                "{} {} {}",
                data.domain().shape(),
                data.domain().count(),
                data.schema()
            )
        };
        match path.extension().and_then(|e| e.to_str()) {
            Some("field") => {
                let data = orla_runtime::file::decode(&bytes).map_err(|e| e.to_string())?;
                let detail = describe(&data);
                Ok((data, detail))
            }
            Some("png") => {
                let data = png::decode_field(&bytes, gray).map_err(|e| e.to_string())?;
                let detail = describe(&data);
                Ok((data, detail))
            }
            Some("ppm") | Some("pgm") => {
                let data = ppm::decode_field(&bytes).map_err(|e| e.to_string())?;
                let data = if gray && data.schema().is_pixel() {
                    let words = data
                        .storage()
                        .chunks_exact(4)
                        .map(|p| {
                            (f32::from_bits(p[0]) * 0.2126
                                + f32::from_bits(p[1]) * 0.7152
                                + f32::from_bits(p[2]) * 0.0722)
                                .to_bits()
                        })
                        .collect();
                    FieldData::from_storage(
                        orla_runtime::Schema::new(vec![orla_runtime::Scalar::Float]),
                        data.domain(),
                        words,
                    )
                    .map_err(|e| e.to_string())?
                } else {
                    data
                };
                let detail = describe(&data);
                Ok((data, detail))
            }
            Some("wav") => {
                let sound = wav::decode(&bytes).map_err(|e| e.to_string())?;
                let detail = format!("{} at {} Hz", describe(&sound.field), sound.rate);
                Ok((sound.field, detail))
            }
            _ => Err("unknown Field file format; use .field, .png, .ppm, .pgm, or .wav".into()),
        }
    }

    /// Reads every input's file that is new or changed since it was last
    /// read, and forgets inputs the scene no longer declares.
    fn sync_assets(&mut self, events: &mut Vec<Event>) {
        let wanted: Vec<orla_graph::Asset> = self
            .graph
            .as_ref()
            .map(|graph| graph.assets.clone())
            .unwrap_or_default();
        let stale: Vec<String> = self
            .assets
            .keys()
            .filter(|name| !wanted.iter().any(|asset| &asset.name == *name))
            .cloned()
            .collect();
        for name in stale {
            self.assets.remove(&name);
            if let Some(handle) = self.externals.remove(&name) {
                let _ = self.runtime.host_mut().release(handle);
            }
        }
        for orla_graph::Asset {
            name,
            path,
            ty,
            bytes,
        } in wanted
        {
            let current = stamp(&path);
            if self
                .assets
                .get(&name)
                .is_some_and(|asset| asset.path == path && asset.stamp == current)
            {
                continue;
            }
            self.assets.insert(
                name.clone(),
                Asset {
                    path: path.clone(),
                    stamp: current,
                },
            );
            let read = if bytes {
                Self::load_bytes(&path)
            } else {
                Self::load_field(&path, Some(&ty))
            };
            let read = read.and_then(|(data, detail)| {
                let data = kept_as(data, &ty)?;
                let detail = match data.schema().uniform_encoding() {
                    Some(encoding) if !data.schema().is_wide() => {
                        format!("{detail}, kept as {encoding}")
                    }
                    _ => detail,
                };
                Ok((data, detail))
            });
            let (data, detail) = match read {
                Ok(loaded) => loaded,
                Err(error) => {
                    events.push(Event::Rejected {
                        path: path.clone(),
                        error: format!(" {error}"),
                    });
                    continue;
                }
            };
            match self.runtime.host_mut().adopt(data) {
                Ok(handle) => {
                    if let Some(previous) = self.externals.insert(name.clone(), handle) {
                        let _ = self.runtime.host_mut().release(previous);
                    }
                    events.push(Event::Loaded { name, path, detail });
                }
                Err(error) => events.push(Event::Rejected {
                    path,
                    error: format!(" {error}"),
                }),
            }
        }
    }

    /// What a scene output of this name is wired to, when the name is
    /// an output rather than a node.
    fn output_source(
        &self,
        node: &orla_graph::NodeId,
        port: &str,
    ) -> Option<(orla_graph::NodeId, String)> {
        let graph = self.graph.as_ref()?;
        if graph.nodes.iter().any(|n| n.id == *node) {
            return None;
        }
        let (name, _) = graph
            .outputs
            .iter()
            .find(|(name, _)| name == &node.0 && port == "result")?;
        match &graph.outputs.iter().find(|(o, _)| o == name)?.1 {
            orla_graph::Source::Node { node, port } => Some((node.clone(), port.clone())),
            orla_graph::Source::External(_) => None,
        }
    }

    /// Why a node's output is not there to be saved: the name may be a
    /// placed graph, which is its nodes by the time the graph runs, or
    /// a node of one written without its repetition, or no node at all.
    fn why_not_saved(&self, node: &orla_graph::NodeId, port: &str) -> String {
        let Some(graph) = self.graph.as_ref() else {
            return format!("`{node}.{port}` has no value; no scene is loaded");
        };
        let named: Vec<&str> = graph
            .nodes
            .iter()
            .map(|n| n.id.0.as_str())
            .filter(|id| {
                id.starts_with(&format!("{node}."))
                    || id.starts_with(&format!("{node}/"))
                    || id.split('/').next() == Some(node.0.as_str())
            })
            .collect();
        if !graph.nodes.iter().any(|n| n.id == *node) {
            if let Some(last) = named.last() {
                return format!(
                    "`{node}` is not a node: it is placed as {} of them, the last `{last}`, which is what a `--save` names (or give the scene an `output`)",
                    named.len()
                );
            }
            return format!("`{node}` is not a node of this scene");
        }
        format!("`{node}.{port}` has no value; pin it and evaluate")
    }

    /// Writes a node's output as a file: `.field` for any Field, `.png` or
    /// `.ppm` for a grid of pixels (a `Grid<Float>` as gray), `.pgm` for a
    /// `Grid<Float>`, `.wav` at 48 kHz for rows of one or two floats. The
    /// node must have been evaluated and kept materialized (pinned).
    pub fn save(
        &mut self,
        node: &orla_graph::NodeId,
        port: &str,
        path: &Path,
    ) -> Result<(), String> {
        // A scene's own output name stands for what it is wired to, so
        // `--save settled=...` works with the names the scene reads by.
        let (node, port) = match self.output_source(node, port) {
            Some((node, port)) => (node, port),
            None => (node.clone(), port.to_owned()),
        };
        let (node, port) = (&node, port.as_str());
        let handle = match self.cached(node, port) {
            Some(handle) => handle,
            None => return Err(self.why_not_saved(node, port)),
        };
        self.runtime
            .materialize(handle)
            .map_err(|e| e.to_string())?;
        let host = self.runtime.host();
        let data = host.data(handle).map_err(|e| e.to_string())?;
        let domain = data.domain();
        let bytes = match path.extension().and_then(|e| e.to_str()) {
            Some("field") => orla_runtime::file::encode_data(data),
            Some("ppm") => {
                if !data.schema().is_pixel() || domain.shape() != Shape::Grid {
                    return Err(format!(
                        "`{node}.{port}` is not a grid of pixels; save it as .field"
                    ));
                }
                let image = self.runtime.image(handle).map_err(|e| e.to_string())?;
                ppm::encode(&image)
            }
            Some("pgm") => {
                if data.schema().value_words() != 1 || domain.shape() != Shape::Grid {
                    return Err(format!(
                        "`{node}.{port}` is not a Grid<Float>; save it as .field"
                    ));
                }
                let samples: Vec<f32> = data.values().iter().map(|w| f32::from_bits(*w)).collect();
                ppm::encode_gray(domain.width(), domain.height(), &samples)
            }
            Some("png") => {
                if domain.shape() != Shape::Grid {
                    return Err(format!("`{node}.{port}` is not a grid; save it as .field"));
                }
                let values = data.values();
                let pixels: Vec<[f32; 4]> = if data.schema().is_pixel() {
                    values
                        .chunks_exact(4)
                        .map(|p| {
                            [
                                f32::from_bits(p[0]),
                                f32::from_bits(p[1]),
                                f32::from_bits(p[2]),
                                f32::from_bits(p[3]),
                            ]
                        })
                        .collect()
                } else if data.schema().value_words() == 1 {
                    values
                        .iter()
                        .map(|w| {
                            let v = f32::from_bits(*w);
                            [v, v, v, 1.0]
                        })
                        .collect()
                } else {
                    return Err(format!(
                        "`{node}.{port}` is neither pixels nor a Grid<Float>; save it as .field"
                    ));
                };
                png::encode(domain.width(), domain.height(), &pixels)
            }
            Some("wav") => {
                let channels = data.schema().value_words();
                if domain.shape() != Shape::Rows || channels > 2 {
                    return Err(format!(
                        "`{node}.{port}` is not rows of one or two floats; save it as .field"
                    ));
                }
                wav::encode(&data.values(), channels, 48_000)
            }
            _ => {
                return Err(
                    "unknown Field file format; use .field, .png, .ppm, .pgm, or .wav".into(),
                );
            }
        };
        std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
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
            let _ = self.runtime.host_mut().release(previous);
        }
        Ok(())
    }

    /// Reconciles the runtime with the files on disk and reports what changed.
    pub fn sync(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        self.collect_pending(&mut events);
        self.sync_root(&mut events);
        // A library that changed may have changed a declaration the root's
        // Kernels inherit, so the root is read again after the libraries.
        if self.sync_libraries(&mut events) {
            self.root_stamp = None;
            self.sync_root(&mut events);
        }
        self.sync_assets(&mut events);
        self.sync_sequences(&mut events);
        self.sync_texts(&mut events);
        events.append(&mut self.frame_events);
        events
    }

    /// Lists every `frames` input's directory anew, and forgets sequences
    /// the scene no longer declares. A listing that changed starts over:
    /// the frame whose turn it is comes in at the next evaluation.
    fn sync_sequences(&mut self, events: &mut Vec<Event>) {
        let wanted: Vec<(String, PathBuf, String, u32)> = self
            .graph
            .as_ref()
            .map(|graph| graph.sequences.clone())
            .unwrap_or_default();
        let stale: Vec<String> = self
            .sequences
            .keys()
            .filter(|name| !wanted.iter().any(|(wanted, ..)| wanted == *name))
            .cloned()
            .collect();
        for name in stale {
            self.sequences.remove(&name);
            if let Some(handle) = self.externals.remove(&name) {
                let _ = self.runtime.host_mut().release(handle);
            }
        }
        for (name, dir, ty, rate) in wanted {
            let files = match frame_files(&dir) {
                Ok(files) => files,
                Err(error) => {
                    if self.sequences.remove(&name).is_some()
                        && let Some(handle) = self.externals.remove(&name)
                    {
                        let _ = self.runtime.host_mut().release(handle);
                    }
                    events.push(Event::Rejected {
                        path: dir,
                        error: format!(" {error}"),
                    });
                    continue;
                }
            };
            if self.sequences.get(&name).is_some_and(|sequence| {
                sequence.dir == dir
                    && sequence.ty == ty
                    && sequence.rate == rate
                    && sequence.files == files
            }) {
                continue;
            }
            if files.is_empty() {
                events.push(Event::Rejected {
                    path: dir.clone(),
                    error: " holds no frame: a .png, .ppm, .pgm, .field or .wav file".into(),
                });
            } else {
                events.push(Event::Loaded {
                    name: name.clone(),
                    path: dir.clone(),
                    detail: format!("{} frame(s) at {rate} a second", files.len()),
                });
            }
            self.sequences.insert(
                name,
                Sequence {
                    dir,
                    ty,
                    rate,
                    files,
                    current: None,
                },
            );
        }
    }

    /// Brings in the frame whose turn it is at `time` for every `frames`
    /// input: frame `floor(time * rate)`, wrapping at the end. A frame
    /// that cannot be read is reported by the next sync and the input
    /// keeps the frame it had.
    fn select_frames(&mut self, time: f32) {
        let names: Vec<String> = self.sequences.keys().cloned().collect();
        for name in names {
            let Some(sequence) = self.sequences.get(&name) else {
                continue;
            };
            if sequence.files.is_empty() {
                continue;
            }
            let count = sequence.files.len() as i64;
            let index = ((time * sequence.rate as f32).floor() as i64).rem_euclid(count) as usize;
            if sequence.current == Some(index) {
                continue;
            }
            let path = sequence.files[index].clone();
            let ty = sequence.ty.clone();
            let loaded = Self::load_field(&path, Some(&ty)).and_then(|(data, _)| {
                let data = kept_as(data, &ty)?;
                self.runtime
                    .host_mut()
                    .adopt(data)
                    .map_err(|e| e.to_string())
            });
            match loaded {
                Ok(handle) => {
                    if let Some(previous) = self.externals.insert(name.clone(), handle) {
                        let _ = self.runtime.host_mut().release(previous);
                    }
                    if let Some(sequence) = self.sequences.get_mut(&name) {
                        sequence.current = Some(index);
                    }
                }
                Err(error) => self.frame_events.push(Event::Rejected {
                    path,
                    error: format!(" {error}"),
                }),
            }
        }
    }

    /// Writes every input's text that is new or changed into the Host as
    /// `Rows<Int>` of its characters' codes, and forgets texts the scene no
    /// longer declares.
    fn sync_texts(&mut self, events: &mut Vec<Event>) {
        let wanted: Vec<(String, String)> = self
            .graph
            .as_ref()
            .map(|graph| graph.texts.clone())
            .unwrap_or_default();
        let stale: Vec<String> = self
            .texts
            .keys()
            .filter(|name| !wanted.iter().any(|(wanted, _)| wanted == *name))
            .cloned()
            .collect();
        for name in stale {
            self.texts.remove(&name);
            if let Some(handle) = self.externals.remove(&name) {
                let _ = self.runtime.host_mut().release(handle);
            }
        }
        for (name, text) in wanted {
            if self.texts.get(&name) == Some(&text) {
                continue;
            }
            let words: Vec<u32> = text.chars().map(|c| c as u32).collect();
            let chars = words.len();
            let data = FieldData::from_storage(
                orla_runtime::Schema::new(vec![orla_runtime::Scalar::Int]),
                orla_runtime::Domain::Rows(chars as u32),
                words,
            );
            let adopted = data.and_then(|data| self.runtime.host_mut().adopt(data));
            match adopted {
                Ok(handle) => {
                    if let Some(previous) = self.externals.insert(name.clone(), handle) {
                        let _ = self.runtime.host_mut().release(previous);
                    }
                    self.texts.insert(name.clone(), text);
                    events.push(Event::Text { name, chars });
                }
                Err(error) => events.push(Event::SceneRejected(format!(
                    "input `{name}`'s text: {error}"
                ))),
            }
        }
    }

    /// Prepares replacements of running Kernels and fused groups on a
    /// worker thread, so a republish never stalls the caller; a
    /// replacement publishes on the first sync after it is ready.
    pub fn set_background(&mut self, enabled: bool) {
        self.runtime.set_background(enabled);
    }

    /// Whether replacements prepare on a worker thread.
    pub fn background(&self) -> bool {
        self.runtime.background()
    }

    /// Publishes every replacement the worker has finished.
    fn collect_pending(&mut self, events: &mut Vec<Event>) {
        let ids: Vec<KernelId> = self.pending.keys().cloned().collect();
        for id in ids {
            let ticket = self.pending[&id].ticket;
            let Some(result) = self.runtime.poll(ticket) else {
                continue;
            };
            let PendingPublish { path, expect, .. } =
                self.pending.remove(&id).expect("pending entry exists");
            match result {
                Ok(candidate) => match self.runtime.publish(candidate, expect) {
                    Ok(version) => {
                        events.push(Event::Published {
                            path: path.clone(),
                            id: id.clone(),
                            generation: version.generation(),
                        });
                        self.owners.insert(id, path);
                    }
                    Err(error) => events.push(Event::Rejected {
                        path,
                        error: format!(" {error}"),
                    }),
                },
                Err(error) => events.push(Event::Rejected {
                    path,
                    error: error.to_string(),
                }),
            }
        }
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
        let sub_graph_changed = self
            .sub_graphs
            .iter()
            .any(|(path, loaded)| stamp(path) != *loaded);
        if current == self.root_stamp && self.graph.is_some() && !sub_graph_changed {
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
        let base = self
            .root
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let mut loader = |path: &Path| read_document(path);
        let mut declare = |base: &Path, uses: &[orla_language::Use]| declarations_for(base, uses);
        let lowered = match orla_scene::load_with_params(
            &document,
            &text,
            &name,
            &base,
            &mut loader,
            &mut declare,
            &self.params,
        ) {
            Ok(lowered) => lowered,
            Err(error) => {
                events.push(Event::SceneRejected(error.to_string()));
                return;
            }
        };
        // Kernels defined by the root: its top-level ones and the inline
        // ones of its own nodes. Inline Kernels of sub-graphs belong to
        // their documents.
        let mut by_document: BTreeMap<PathBuf, Vec<Defined>> = BTreeMap::new();
        let root = self.root.clone();
        let root_key = PathBuf::from(&name);
        // The root's own top-level Kernels see the shared declarations too.
        let mut document = document;
        let mut declarations = String::new();
        match declarations_for(&base, &document.uses) {
            Ok((structs, types, functions)) => {
                declarations = declarations_key(&structs, &types, &functions);
                self.shared_structs = structs.iter().chain(&document.structs).cloned().collect();
                self.shared_types = types.iter().chain(&document.types).cloned().collect();
                self.shared_functions = functions
                    .iter()
                    .chain(&document.functions)
                    .cloned()
                    .collect();
                document.inherit(&structs, &types, &functions);
            }
            Err(error) => events.push(Event::Rejected {
                path: root.clone(),
                error: format!(" {error}"),
            }),
        }
        for kernel in &document.kernels {
            match named_id(kernel) {
                Ok(id) => by_document.entry(root.clone()).or_default().push(Defined {
                    id,
                    text: text[kernel.range.clone()].to_owned(),
                    kernel: kernel.clone(),
                }),
                Err(message) => events.push(Event::Rejected {
                    path: root.clone(),
                    error: message,
                }),
            }
        }
        for inline in &lowered.inline {
            let owner = if inline.document == root_key {
                root.clone()
            } else {
                inline.document.clone()
            };
            by_document.entry(owner).or_default().push(Defined {
                id: inline.id.clone(),
                text: inline.text.clone(),
                kernel: inline.kernel.clone(),
            });
        }
        let mut sub_graphs = BTreeMap::new();
        for path in &lowered.documents {
            sub_graphs.insert(path.clone(), stamp(path));
            by_document.entry(path.clone()).or_default();
        }
        by_document.entry(root.clone()).or_default();
        // Documents that were graph documents before and are not any more
        // release their kernels.
        let gone: Vec<PathBuf> = self
            .sub_graphs
            .keys()
            .filter(|path| !sub_graphs.contains_key(*path))
            .cloned()
            .collect();
        for path in gone {
            self.retire_document(&path, events);
        }
        for (path, defined) in by_document {
            let current = if path == root { current } else { stamp(&path) };
            let key = if path == root {
                declarations.clone()
            } else {
                // A sub-graph document's inline Kernels inherit through
                // its own `use` items, resolved during lowering.
                let dir = path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .to_path_buf();
                std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|text| orla_language::parse_document(&text).ok())
                    .and_then(|doc| declarations_for(&dir, &doc.uses).ok())
                    .map(|(s, t, f)| declarations_key(&s, &t, &f))
                    .unwrap_or_default()
            };
            self.publish_document(&path, current, defined, &key, events);
        }
        self.sub_graphs = sub_graphs;
        self.root_uses = lowered.uses;
        self.graph = Some(lowered.graph);
        events.push(Event::SceneLoaded);
    }

    /// Loads the libraries the root uses; returns whether any changed.
    fn sync_libraries(&mut self, events: &mut Vec<Event>) -> bool {
        let mut changed = false;
        let mut wanted: Vec<PathBuf> = Vec::new();
        for (path, trailing_slash) in &self.root_uses {
            wanted.extend(expand_use(path, *trailing_slash).unwrap_or_default());
        }
        for path in &self.extra_uses {
            wanted.extend(expand_use(path, false).unwrap_or_default());
        }
        wanted.retain(|path| *path != self.root && !self.sub_graphs.contains_key(path));
        wanted.sort();
        wanted.dedup();
        let stale: Vec<PathBuf> = self
            .documents
            .keys()
            .filter(|path| {
                **path != self.root
                    && !wanted.contains(path)
                    && !self.sub_graphs.contains_key(*path)
            })
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
            changed = true;
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
            // A library's own `use` items bring in shared declarations.
            let dir = path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf();
            let mut document = document;
            let declarations = match declarations_within(Some(&path), &dir, &document.uses) {
                Ok((structs, types, functions)) => {
                    document.inherit(&structs, &types, &functions);
                    declarations_key(&structs, &types, &functions)
                }
                Err(error) => {
                    events.push(Event::Rejected {
                        path: path.clone(),
                        error: format!(" {error}"),
                    });
                    self.documents.entry(path).or_default().stamp = current;
                    continue;
                }
            };
            let mut defined = Vec::new();
            for kernel in &document.kernels {
                match named_id(kernel) {
                    Ok(id) => defined.push(Defined {
                        id,
                        text: text[kernel.range.clone()].to_owned(),
                        kernel: kernel.clone(),
                    }),
                    Err(message) => events.push(Event::Rejected {
                        path: path.clone(),
                        error: message,
                    }),
                }
            }
            self.publish_document(&path, current, defined, &declarations, events);
        }
        changed
    }

    /// A diagnostic with the file that declares the name it could not
    /// find, when the session knows one: a `use` away is the usual
    /// reason, and saying which file says what to write.
    fn located(&self, message: String, document: &Path) -> String {
        let Some(name) = missing_name(&message) else {
            return message;
        };
        // What this document already uses is not the answer; every other
        // document the session knows, and every `.orla` beside this one
        // and beside the scene, is where to look.
        let used: Vec<PathBuf> = std::fs::read_to_string(document)
            .ok()
            .and_then(|text| orla_language::parse_document(&text).ok())
            .map(|parsed| {
                used_documents(
                    document.parent().unwrap_or_else(|| Path::new(".")),
                    &parsed.uses,
                )
                .unwrap_or_default()
            })
            .unwrap_or_default();
        let mut search: Vec<PathBuf> = Vec::new();
        let siblings = |dir: &Path, search: &mut Vec<PathBuf>| {
            if let Ok(entries) = std::fs::read_dir(dir) {
                let mut files: Vec<PathBuf> = entries
                    .filter_map(Result::ok)
                    .map(|entry| entry.path())
                    .filter(|path| path.extension().is_some_and(|ext| ext == "orla"))
                    .collect();
                files.sort();
                search.append(&mut files);
            }
        };
        siblings(
            document.parent().unwrap_or_else(|| Path::new(".")),
            &mut search,
        );
        siblings(
            self.root.parent().unwrap_or_else(|| Path::new(".")),
            &mut search,
        );
        // The libraries the session knows, and what sits beside them: a
        // shared directory usually holds more than one.
        let known: Vec<PathBuf> = self.documents.keys().cloned().collect();
        for path in &known {
            siblings(path.parent().unwrap_or_else(|| Path::new(".")), &mut search);
        }
        search.extend(known);
        search.retain(|path| {
            path != document
                && !used.iter().any(|seen| seen == path)
                && path.extension().is_some_and(|ext| ext == "orla")
        });
        search.dedup();
        let Some((file, kind)) = declared_in(&name, &search) else {
            return message;
        };
        let spelling = relative_to(&file, document.parent().unwrap_or_else(|| Path::new(".")));
        format!(
            "{message}; {kind} `{name}` is declared in {spelling}, which this document does not use: `use \"{spelling}\";`"
        )
    }

    /// Publishes the Kernels a document now defines: retires the ones that
    /// left, keeps the unchanged, republishes the changed, and reports each.
    /// `declarations` keys what the Kernels inherit; a change there
    /// republishes them all.
    fn publish_document(
        &mut self,
        path: &Path,
        current: Option<Stamp>,
        defined: Vec<Defined>,
        declarations: &str,
        events: &mut Vec<Event>,
    ) {
        let previous_state = self.documents.remove(path).unwrap_or_default();
        let previous = if previous_state.declarations == declarations {
            previous_state.kernels
        } else {
            BTreeMap::new()
        };
        let mut kernels = BTreeMap::new();
        let mut asts = BTreeMap::new();
        for id in previous.keys() {
            if !defined.iter().any(|d| &d.id == id) {
                self.retire(path, id, events);
            }
        }
        for Defined { id, text, kernel } in defined {
            asts.insert(id.clone(), kernel.clone());
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
            if let Some(running) = self.runtime.kernel(&id) {
                // Something is running: its replacement may take its time.
                match orla_compiler::compile_kernel(&kernel, Some(&id)) {
                    Ok(module) => {
                        if let Some(superseded) = self.pending.remove(&id) {
                            self.runtime.discard(superseded.ticket);
                        }
                        let ticket = self.runtime.prepare_later(module, text.clone());
                        self.pending.insert(
                            id.clone(),
                            PendingPublish {
                                ticket,
                                path: path.to_path_buf(),
                                expect: Expect::Generation(running.generation()),
                            },
                        );
                        if self.runtime.background() {
                            events.push(Event::Preparing {
                                path: path.to_path_buf(),
                                id: id.clone(),
                            });
                        }
                        self.collect_pending(events);
                    }
                    Err(error) => events.push(Event::Rejected {
                        path: path.to_path_buf(),
                        error: self.located(error.to_string(), path),
                    }),
                }
                kernels.insert(id, text);
                continue;
            }
            let prepared = orla_compiler::compile_kernel(&kernel, Some(&id))
                .map_err(|error| error.to_string())
                .and_then(|module| {
                    self.runtime
                        .prepare_module(module, &text)
                        .map_err(|error| error.to_string())
                });
            match prepared {
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
                    error: self.located(error.to_string(), path),
                }),
            }
            kernels.insert(id, text);
        }
        self.documents.insert(
            path.to_path_buf(),
            DocumentState {
                stamp: current,
                kernels,
                asts,
                declarations: declarations.to_owned(),
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
            if let Some(pending) = self.pending.remove(id) {
                self.runtime.discard(pending.ticket);
            }
            if self.runtime.retire(id).is_some() {
                events.push(Event::Retired {
                    path: path.to_path_buf(),
                    id: id.clone(),
                });
            }
        }
    }

    /// Publishes Kernel source that did not come from a document, for the
    /// creation protocol. The source holds exactly one Kernel. Ownership
    /// stays with documents: a document that later defines the id
    /// republishes it from the file.
    pub fn publish_source(
        &mut self,
        source: &str,
        expect: Expect,
    ) -> Result<std::sync::Arc<orla_runtime::KernelVersion>, String> {
        let candidate = self.prepare_source(source)?;
        self.runtime
            .publish(candidate, expect)
            .map_err(|e| e.to_string())
    }

    /// Prepares `source`, a document holding one Kernel, without
    /// publishing it. The Kernel sees the declarations the source makes
    /// or `use`s, and the scene's shared ones after those, so a Kernel
    /// from a file republishes through the protocol as it is.
    pub fn prepare_source(&self, source: &str) -> Result<orla_runtime::Candidate, String> {
        let mut document = orla_language::parse_document(source).map_err(|e| e.to_string())?;
        if document.kernels.len() != 1 {
            return Err(format!(
                "expected one kernel, found {}",
                document.kernels.len()
            ));
        }
        let base = self.root.parent().unwrap_or_else(|| Path::new("."));
        let (mut structs, mut types, mut functions) = if document.uses.is_empty() {
            (Vec::new(), Vec::new(), Vec::new())
        } else {
            declarations_for(base, &document.uses).map_err(|e| e.to_string())?
        };
        let root = self.root.clone();
        // The scene's shared declarations come after the source's own,
        // and a name the source declares, at its top or inside its
        // Kernel, is the source's: a Kernel may redefine a shared struct,
        // which is how a state's layout changes while it runs.
        for shared in &self.shared_structs {
            let declared = |d: &orla_language::StructDecl| d.name.value == shared.name.value;
            if !document.structs.iter().any(declared)
                && !structs.iter().any(declared)
                && !document.kernels[0].structs.iter().any(declared)
            {
                structs.push(shared.clone());
            }
        }
        for shared in &self.shared_types {
            let declared = |d: &orla_language::TypeAlias| d.name.value == shared.name.value;
            if !document.types.iter().any(declared)
                && !types.iter().any(declared)
                && !document.kernels[0].types.iter().any(declared)
            {
                types.push(shared.clone());
            }
        }
        for shared in &self.shared_functions {
            let declared = |d: &orla_language::FnDecl| d.name.value == shared.name.value;
            if !document.functions.iter().any(declared)
                && !functions.iter().any(declared)
                && !document.kernels[0].functions.iter().any(declared)
            {
                functions.push(shared.clone());
            }
        }
        document.inherit(&structs, &types, &functions);
        let module = orla_compiler::compile_kernel(&document.kernels[0], None)
            .map_err(|e| self.located(e.to_string(), &root))?;
        self.runtime
            .prepare_module(module, source)
            .map_err(|e| e.to_string())
    }

    /// Retires a published Kernel; returns whether one was registered.
    pub fn retire_kernel(&mut self, id: &KernelId) -> bool {
        self.owners.remove(id);
        if let Some(pending) = self.pending.remove(id) {
            self.runtime.discard(pending.ticket);
        }
        for state in self.documents.values_mut() {
            state.kernels.remove(id);
        }
        self.runtime.retire(id).is_some()
    }

    /// The Field an input of the scene holds now: a file's, a text's, a
    /// frame's, or what the caller supplied.
    pub fn external(&self, name: &str) -> Option<Handle> {
        self.externals.get(name).copied()
    }

    /// The cached image of a node's output port, if it was evaluated.
    pub fn cached(&self, node: &orla_graph::NodeId, port: &str) -> Option<Handle> {
        self.evaluator.cached(node, port)
    }

    /// How many elements a node's output holds: what a node whose
    /// `domain` a Kernel decided came out as.
    pub fn element_count(&self, node: &orla_graph::NodeId, port: &str) -> Result<usize, String> {
        let handle = self
            .cached(node, port)
            .ok_or_else(|| "the node has no materialized output; is it wired?".to_string())?;
        self.runtime()
            .host()
            .domain(handle)
            .map(|domain| domain.count())
            .map_err(|e| e.to_string())
    }

    /// Evaluates the current graph under `context`. Unchanged nodes are
    /// Waits for every `async` node's run off the frame and takes it in;
    /// the next evaluation reads what landed. See
    /// [`orla_graph::Evaluator::settle`].
    pub fn settle(&mut self) -> Result<Vec<orla_graph::NodeId>, EvaluateError> {
        self.evaluator.settle(&mut self.runtime)
    }

    /// Whether any `async` node's run is still going.
    pub fn has_running(&self) -> bool {
        self.evaluator.has_running()
    }

    /// reused; state commits only when the whole evaluation succeeds.
    pub fn evaluate(&mut self, context: &Context) -> Result<Report, EvaluateError> {
        self.select_frames(context.time);
        let graph = self.graph.as_ref().ok_or_else(|| {
            EvaluateError::Graph(orla_graph::GraphError::UnknownOutput(
                orla_graph::Source::External("<no scene>".into()),
            ))
        })?;
        self.evaluator.evaluate(
            graph,
            &mut self.runtime,
            &self.externals,
            context,
            &self.params,
        )
    }

    /// Runs one element of `node`'s Kernel on the interpreter and returns
    /// every binding of it: element `at` of a map, step `at[0]` of a scan.
    /// The scene is evaluated first under `context`, with the node's
    /// sources pinned so their Fields exist, and the node's invocation is
    /// what an evaluation would give it now; nothing is committed by the
    /// trace itself. Pins the trace added are removed again.
    pub fn trace(
        &mut self,
        node: &orla_graph::NodeId,
        at: [u32; 3],
        context: &Context,
    ) -> Result<(orla_cpu::Trace, KernelId), String> {
        let graph = self.graph.as_ref().ok_or("no scene is loaded")?;
        let target = graph
            .nodes
            .iter()
            .find(|n| &n.id == node)
            .ok_or_else(|| format!("no node named `{node}`"))?
            .clone();
        let version = self
            .runtime
            .kernel(&target.kernel)
            .ok_or_else(|| format!("`{}` is not published", target.kernel))?;
        if version.module().is_none() {
            return Err(format!(
                "`{}` is a native operation; there is nothing to trace",
                target.kernel
            ));
        }
        // The sources the node reads must have Fields: pin them for this
        // evaluation, then let fusion have them back.
        let mut pinned = Vec::new();
        for binding in target.inputs.values() {
            let sources = match binding {
                orla_graph::Binding::Now(source) => vec![source],
                orla_graph::Binding::Previous {
                    source, initial, ..
                } => vec![source, initial],
            };
            for source in sources {
                if let orla_graph::Source::Node { node, .. } = source
                    && !self.evaluator.pins().contains(node)
                {
                    self.evaluator.pin(node.clone());
                    pinned.push(node.clone());
                }
            }
        }
        let traced = (|| {
            self.evaluate(context).map_err(|error| error.to_string())?;
            // The graph is read while the runtime is written to (a
            // `value(port)` domain brings a Field to the Host), so the
            // scene is taken out for the call and put back.
            let graph = self.graph.take().ok_or("no scene is loaded")?;
            let invocation = self
                .evaluator
                .invocation(
                    &graph,
                    &target,
                    &version,
                    &mut self.runtime,
                    &self.externals,
                    context,
                    &self.params,
                )
                .map_err(|error| error.to_string());
            self.graph = Some(graph);
            let invocation = invocation?;
            for handle in &invocation.inputs {
                self.runtime
                    .materialize(*handle)
                    .map_err(|error| error.to_string())?;
            }
            // A trace walks the Kernel as written: the same source
            // without the optimizer, so its functions are still called
            // and its `let`s are all there. The running module answers
            // for a Kernel no document owns.
            let written = self.unoptimized(&target.kernel);
            let module = written
                .as_ref()
                .unwrap_or_else(|| version.module().expect("checked above"));
            let granted = module.manifest.requires.clone();
            let mut scope = orla_runtime::Scope::new(self.runtime.host_mut(), granted);
            orla_cpu::trace(module, &invocation, &mut scope, at).map_err(|error| error.to_string())
        })();
        for node in pinned {
            self.evaluator.unpin(&node);
        }
        traced.map(|trace| (trace, target.kernel.clone()))
    }

    /// One element field of a node's output: the value's type and its
    /// words. `node.port[index].field.field` is how a replay and the
    /// protocol both name it; [`parse_target`] reads that spelling.
    pub fn element_field(
        &mut self,
        node: &orla_graph::NodeId,
        port: &str,
        index: u32,
        path: &[String],
    ) -> Result<(orla_ir::Type, Vec<u32>), String> {
        use orla_ir::Type;
        let kernel = self
            .graph()
            .and_then(|graph| graph.nodes.iter().find(|n| &n.id == node))
            .map(|n| n.kernel.clone())
            .ok_or_else(|| format!("no node `{node}` in the scene"))?;
        let version = self
            .runtime()
            .kernel(&kernel)
            .ok_or_else(|| format!("`{kernel}` is not published"))?;
        let output = version
            .manifest()
            .outputs
            .iter()
            .find(|o| o.name == port)
            .ok_or_else(|| format!("`{kernel}` has no output `{port}`"))?;
        let structs: Vec<orla_ir::StructDef> = version
            .module()
            .map(|m| m.structs.clone())
            .unwrap_or_default();
        // Follow the field path through the struct table to a word
        // offset and a type.
        let mut ty = output.element();
        let mut offset = 0usize;
        let words_of = |ty: Type| orla_ir::schema_of(ty, &structs).map_or(1, |s| s.value_words());
        for field in path {
            // A struct's field by name, or a vector's component by the
            // letter the language spells it with.
            if let Some((component, element)) = component_of(ty, field) {
                offset += component;
                ty = element;
                continue;
            }
            let Type::Struct(id) = ty else {
                return Err(format!(
                    "`{field}` names a field or a component, but the value is a {ty}"
                ));
            };
            let def = &structs[id as usize];
            let mut found = None;
            for f in &def.fields {
                if &f.name == field {
                    found = Some(f.ty);
                    break;
                }
                offset += words_of(f.ty);
            }
            ty = found.ok_or_else(|| format!("struct `{}` has no field `{field}`", def.name))?;
        }
        let handle = self
            .cached(node, port)
            .ok_or_else(|| "the node has no materialized output; is it wired?".to_string())?;
        self.runtime_mut()
            .materialize(handle)
            .map_err(|e| e.to_string())?;
        let element = self
            .runtime()
            .host()
            .element(handle, index)
            .map_err(|e| e.to_string())?;
        Ok((ty, element[offset..offset + words_of(ty)].to_vec()))
    }

    /// The Kernel as its document wrote it, compiled without the
    /// optimizer; `None` for one no document owns, or one whose source
    /// no longer compiles on its own.
    fn unoptimized(&self, id: &KernelId) -> Option<orla_runtime::Validated> {
        let kernel = self
            .documents
            .values()
            .find_map(|state| state.asts.get(id))?;
        orla_compiler::compile_kernel_unoptimized(kernel, Some(id)).ok()
    }

    /// Evaluates only what `demand` needs; see
    /// [`orla_graph::Evaluator::evaluate_for`].
    pub fn evaluate_for(
        &mut self,
        context: &Context,
        demand: &std::collections::BTreeSet<orla_graph::NodeId>,
    ) -> Result<Report, EvaluateError> {
        self.select_frames(context.time);
        let graph = self.graph.as_ref().ok_or_else(|| {
            EvaluateError::Graph(orla_graph::GraphError::UnknownOutput(
                orla_graph::Source::External("<no scene>".into()),
            ))
        })?;
        self.evaluator.evaluate_for(
            graph,
            &mut self.runtime,
            &self.externals,
            context,
            &self.params,
            demand,
        )
    }

    /// Replaces the committed state of `node.port` with `words`; see
    /// [`orla_graph::Evaluator::write`].
    pub fn write(
        &mut self,
        node: &orla_graph::NodeId,
        port: &str,
        words: Vec<u32>,
    ) -> Result<(), EvaluateError> {
        let graph = self.graph.as_ref().ok_or_else(|| {
            EvaluateError::Graph(orla_graph::GraphError::UnknownOutput(
                orla_graph::Source::External("<no scene>".into()),
            ))
        })?;
        self.evaluator
            .write(graph, &mut self.runtime, node, port, words)
    }

    /// Forgets committed state; `<~` bindings restart from their initial
    /// sources on the next evaluation.
    pub fn reset_state(&mut self) {
        self.evaluator.reset_state(&mut self.runtime);
    }

    /// Sets the window a headless session reports when an evaluation names
    /// none: the size in pixels and the scale factor; `None` for no window.
    pub fn set_window(&mut self, window: Option<([f32; 2], f32)>) {
        self.window = window;
    }

    /// The window a headless session reports, if one was set.
    pub fn window(&self) -> Option<([f32; 2], f32)> {
        self.window
    }

    /// Makes the backend wait for its device after every node, so a
    /// profile measures the work rather than the submission; on the GPU
    /// that is the difference between a dispatch and a picture. The CPU
    /// and the JIT are measured as they are.
    pub fn set_measured(&mut self, on: bool) {
        if let Some(gpu) = self
            .runtime
            .backend()
            .as_any()
            .downcast_ref::<orla_gpu::GpuBackend>()
        {
            gpu.set_measured(on);
        }
    }

    /// What measuring one node costs before the node does anything, so
    /// a profile can take it off what it prints; see
    /// [`orla_runtime::Backend::measurement_floor`]. Ask after a
    /// measured run: a backend only knows once it has tried.
    pub fn measurement_floor(&self) -> std::time::Duration {
        self.runtime().measurement_floor()
    }

    /// Whether a profile's numbers are what the device's own clock said
    /// about the work, rather than what the call and its wait took.
    pub fn measures_the_device(&self) -> bool {
        self.runtime()
            .backend()
            .as_any()
            .downcast_ref::<orla_gpu::GpuBackend>()
            .is_some_and(|gpu| gpu.device_clock_answered())
    }

    /// Enables or disables fusion of pointwise chains.
    pub fn set_fusion(&mut self, enabled: bool) {
        self.evaluator.set_fusion(enabled);
    }

    /// Whether fusion is enabled.
    pub fn fusion(&self) -> bool {
        self.evaluator.fusion()
    }

    /// Keeps a node materialized so its image can be read.
    pub fn pin(&mut self, node: orla_graph::NodeId) {
        self.evaluator.pin(node);
    }

    /// What would execute right now, in order, with fusion applied.
    pub fn plan(&mut self) -> Result<Vec<orla_graph::PlanEntry>, EvaluateError> {
        let graph = self.graph.as_ref().ok_or_else(|| {
            EvaluateError::Graph(orla_graph::GraphError::UnknownOutput(
                orla_graph::Source::External("<no scene>".into()),
            ))
        })?;
        self.evaluator
            .plan_summary(graph, &mut self.runtime, &self.externals, &self.params)
    }

    /// Lets a node be fused away again.
    pub fn unpin(&mut self, node: &orla_graph::NodeId) {
        self.evaluator.unpin(node);
    }

    /// Writes a cached image to a PPM file, materializing it if it was
    /// resident on the device.
    pub fn export(&mut self, image: Handle, path: &Path) -> Result<(), SessionError> {
        let image = self
            .runtime
            .image(image)
            .map_err(|e| SessionError::Observe(e.to_string()))?;
        ppm::write(path, &image).map_err(|e| SessionError::Image(path.to_path_buf(), e))
    }

    /// Reads one pixel of a cached image, materializing it if needed.
    pub fn read_pixel(
        &mut self,
        image: Handle,
        x: u32,
        y: u32,
    ) -> Result<orla_runtime::Pixel, SessionError> {
        // Four words out of the Field, not a copy of the picture: a
        // replay checks a handful of pixels a frame, and the picture it
        // checks them in is thirty-three megabytes at 1920x1080. The
        // Field comes to the Host once; every pixel after that is an
        // index.
        self.runtime
            .materialize(image)
            .map_err(|e| SessionError::Observe(e.to_string()))?;
        let host = self.runtime.host();
        let domain = host.domain(image).map_err(SessionError::Host)?;
        if !host.schema(image).map_err(SessionError::Host)?.is_pixel() {
            return Err(SessionError::Observe(
                "the output is not a picture".to_owned(),
            ));
        }
        let (width, height) = (domain.width(), domain.height() * domain.depth());
        if x >= width || y >= height {
            return Err(SessionError::Host(orla_runtime::HostError::OutOfBounds {
                x,
                y,
                width,
                height,
            }));
        }
        let words = host
            .element(image, y * width + x)
            .map_err(SessionError::Host)?;
        Ok([
            f32::from_bits(words[0]),
            f32::from_bits(words[1]),
            f32::from_bits(words[2]),
            f32::from_bits(words[3]),
        ])
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
        && document.params.is_empty()
}

/// A port type without the encoding it keeps its scalars in:
/// `Grid<Vec4 as unorm8>` is `Grid<Vec4>`.
fn without_encoding(ty: &str) -> String {
    match ty.split_once(" as ") {
        Some((head, tail)) => format!("{head}{}", tail.find('>').map_or("", |at| &tail[at..])),
        None => ty.to_owned(),
    }
}

/// A Field read from a file, kept as the input's type says: what a
/// file decodes to is whole words, and an input declared
/// `Grid<Vec4 as unorm8>` keeps it as bytes from the moment it lands.
fn kept_as(data: FieldData, ty: &str) -> Result<FieldData, String> {
    let Some(name) = ty
        .split_once(" as ")
        .and_then(|(_, tail)| tail.split('>').next())
    else {
        return Ok(data);
    };
    let encoding = orla_runtime::Encoding::parse(name.trim())
        .ok_or_else(|| format!("`{name}` is not an encoding"))?;
    let schema = data
        .schema()
        .with_encoding(encoding)
        .map_err(|why| format!("what the file holds ({}) {why}", data.schema().describe()))?;
    FieldData::from_values(schema, data.domain(), &data.values()).map_err(|e| e.to_string())
}
