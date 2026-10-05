//! Source resolution: what a `use` reaches (a file, a directory of
//! `.orla`, or a package beside the document or in the store's
//! [`Store::packages`]),
//! inherited declarations, and where a missing name is declared.
//! Stateless; reads through the given store.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use orla_kernel::KernelId;
use orla_language::{Document, Kernel};

use crate::store::{Kind, Parsed, Stamp, Store};

/// The format of an `.orlapkg` archive and of a package's manifest.
pub const PACKAGE_VERSION: u32 = 1;

/// What a document's `use`s reach, and the stamp of everything the
/// expansion looked at: each directory it listed or looked in, and each
/// file it read. While none of them changed, neither did what it reached.
#[derive(Debug)]
pub(crate) struct Expanded {
    uses: Vec<(PathBuf, bool)>,
    /// The files reached, in the order of the `use`s.
    pub files: Vec<PathBuf>,
    watched: Vec<(PathBuf, Option<Stamp>)>,
}

impl Expanded {
    /// Every file `uses` reach (a `use` that reaches nothing gives
    /// nothing), noting what the expansion looked at.
    pub(crate) fn of(store: &dyn Store, uses: Vec<(PathBuf, bool)>) -> Self {
        let watching = Watching {
            store,
            seen: Mutex::new(BTreeSet::new()),
        };
        let files = uses
            .iter()
            .flat_map(|(path, trailing_slash)| {
                expand_use(&watching, path, *trailing_slash).unwrap_or_default()
            })
            .collect();
        let watched = watching
            .seen
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .into_iter()
            .map(|path| {
                let stamp = store.stamp(&path);
                (path, stamp)
            })
            .collect();
        Self {
            uses,
            files,
            watched,
        }
    }

    /// Whether expanding `uses` again would give the same files: the
    /// same `use`s, and a store that stamps directories stamping what the
    /// expansion looked at as it did then.
    pub(crate) fn holds(&self, store: &dyn Store, uses: &[(PathBuf, bool)]) -> bool {
        store.stamps_directories()
            && self.uses == uses
            && self
                .watched
                .iter()
                .all(|(path, stamp)| store.stamp(path) == *stamp)
    }
}

/// A store that notes what is looked at through it: a directory listed,
/// the directory a path is looked for in, a file read.
struct Watching<'a> {
    store: &'a dyn Store,
    seen: Mutex<BTreeSet<PathBuf>>,
}

impl Watching<'_> {
    fn note(&self, path: &Path) {
        self.seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(path.to_path_buf());
    }

    fn note_parent(&self, path: &Path) {
        self.note(path.parent().unwrap_or(Path::new("")));
    }
}

impl Store for Watching<'_> {
    fn read(&self, path: &Path) -> std::io::Result<Vec<u8>> {
        self.note(path);
        self.store.read(path)
    }
    fn read_start(&self, path: &Path, bytes: usize) -> std::io::Result<Vec<u8>> {
        self.note(path);
        self.store.read_start(path, bytes)
    }
    fn write(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
        self.store.write(path, bytes)
    }
    fn kind(&self, path: &Path) -> Option<Kind> {
        self.note_parent(path);
        self.store.kind(path)
    }
    fn list(&self, dir: &Path) -> std::io::Result<Vec<PathBuf>> {
        self.note(dir);
        self.store.list(dir)
    }
    fn stamp(&self, path: &Path) -> Option<Stamp> {
        self.note(path);
        self.store.stamp(path)
    }
    fn canonical(&self, path: &Path) -> std::io::Result<PathBuf> {
        self.note_parent(path);
        self.store.canonical(path)
    }
    fn whole(&self, path: &Path) -> PathBuf {
        self.store.whole(path)
    }
    fn packages(&self) -> &[PathBuf] {
        self.store.packages()
    }
    fn parsed(&self) -> Option<&Parsed> {
        self.store.parsed()
    }
}

/// A `use` target's files: a file; each `.orla` directly in a
/// directory; or a package (a directory with `package.osn`, beside the
/// document or in the store's packages), whose manifest `use`s are followed
/// once its versions fit.
pub(crate) fn expand_use(
    store: &dyn Store,
    path: &Path,
    trailing_slash: bool,
) -> Result<Vec<PathBuf>, String> {
    // Canonical paths: a file reached by two spellings is one document.
    let canonical = |file: PathBuf| store.canonical(&file).unwrap_or(file);
    let path = resolve_package(store, path);
    // A path through a package's name resolves inside that package.
    let path = match store.exists(&path) {
        true => path,
        false => in_package(store, &path).unwrap_or(path),
    };
    if store.is_file(&path.join("package.osn")) {
        return package_files(store, &path).map(|files| files.into_iter().map(canonical).collect());
    }
    if trailing_slash || store.is_dir(&path) {
        let mut files: Vec<PathBuf> = store
            .list(&path)
            .unwrap_or_default()
            .into_iter()
            .filter(|path| path.extension().is_some_and(|ext| ext == "orla"))
            .collect();
        files.sort();
        Ok(files.into_iter().map(canonical).collect())
    } else {
        Ok(vec![canonical(path.to_path_buf())])
    }
}

/// Refuses a `use` from a document in `base` of a file inside a package
/// the document is not in: a package is used whole, through its manifest,
/// whose versions it is held to. A package's own documents use its files.
fn used_through_its_manifest(store: &dyn Store, base: &Path, path: &Path) -> Result<(), String> {
    // Only a path through a package's name reaches inside one.
    if !path.components().any(|part| {
        Path::new(part.as_os_str())
            .extension()
            .is_some_and(|ext| ext == "orlapkg")
    }) {
        return Ok(());
    }
    // Where a path lies, through a package's name when it is spelled so.
    let located = |path: &Path| {
        let path = resolve_package(store, path);
        let path = match store.exists(&path) {
            true => path,
            false => in_package(store, &path).unwrap_or(path),
        };
        store.canonical(&path).unwrap_or(path)
    };
    let (base, path) = (located(base), located(path));
    let canonical = |p: &Path| store.canonical(p).unwrap_or_else(|_| p.to_path_buf());
    let Some(package) = package_holding(store, &path) else {
        return Ok(());
    };
    if base.starts_with(canonical(&package)) {
        return Ok(());
    }
    let name = package.file_name().unwrap_or_default().to_string_lossy();
    Err(format!(
        "`{}` is inside the package {name}; use the package whole, `use \".../{name}\";`, which its manifest is held to",
        path.display()
    ))
}

/// The package `path` lies inside: the nearest directory above it named
/// `*.orlapkg` that holds a manifest.
pub(crate) fn package_holding(store: &dyn Store, path: &Path) -> Option<PathBuf> {
    path.ancestors()
        .skip(1)
        .find(|dir| {
            dir.extension().is_some_and(|ext| ext == "orlapkg")
                && store.is_file(&dir.join("package.osn"))
        })
        .map(Path::to_path_buf)
}

/// A document's text: the file at `path`, else, for a path through a
/// package's name, the file inside that package.
pub fn read_document(store: &dyn Store, path: &Path) -> Result<String, String> {
    match store.read_text(path) {
        Ok(text) => Ok(text),
        Err(missing) => match in_package(store, path) {
            Some(inside) => store.read_text(&inside).map_err(|error| error.to_string()),
            None => Err(missing.to_string()),
        },
    }
}

/// Resolves a path through a package name
/// (`image.orlapkg/graphs/x.osn`) to the existing file inside it.
fn in_package(store: &dyn Store, path: &Path) -> Option<PathBuf> {
    let mut rest = PathBuf::new();
    // The last `.orlapkg` part names the package, the path up to it where
    // it lies; the rest is inside.
    let mut name: Option<PathBuf> = None;
    let mut before = PathBuf::new();
    for part in path.components() {
        let piece = Path::new(part.as_os_str());
        before.push(piece);
        if piece.extension().is_some_and(|ext| ext == "orlapkg") {
            name = Some(before.clone());
            rest = PathBuf::new();
        } else if name.is_some() {
            rest.push(piece);
        }
    }
    let name = name?;
    let root = resolve_package(store, &name);
    let inside = root.join(rest);
    store.exists(&inside).then_some(inside)
}

/// A missing path, looked up by name as a package: beside the package
/// the path lies in, when it lies in one, so a package names the
/// packages it uses wherever they are kept together; then in each of
/// the store's [`Store::packages`].
fn resolve_package(store: &dyn Store, path: &Path) -> PathBuf {
    if store.exists(path) {
        return path.to_path_buf();
    }
    let Some(name) = path.file_name() else {
        return path.to_path_buf();
    };
    let beside =
        package_holding(store, path).and_then(|package| Some(package.parent()?.to_path_buf()));
    for dir in beside.iter().chain(store.packages()) {
        let candidate = dir.join(name);
        if store.is_file(&candidate.join("package.osn")) {
            return candidate;
        }
    }
    path.to_path_buf()
}

/// The Kernel files a package's manifest uses, once its versions fit.
pub(crate) fn package_files(store: &dyn Store, dir: &Path) -> Result<Vec<PathBuf>, String> {
    let manifest = dir.join("package.osn");
    let declared = package_manifest(store, &manifest)?;
    let text = store
        .read_text(&manifest)
        .map_err(|e| format!("{}: {e}", manifest.display()))?;
    let document =
        orla_language::parse_document(&text).map_err(|e| format!("{}:{e}", manifest.display()))?;
    let mut files = Vec::new();
    for item in &document.uses {
        let path = dir.join(&item.path.value);
        if store.is_file(&path.join("package.osn"))
            && store.canonical(&path).ok() == store.canonical(dir).ok()
        {
            return Err(format!(
                "{}: a package cannot use itself",
                declared.id.value
            ));
        }
        files.extend(expand_use(store, &path, item.path.value.ends_with('/'))?);
    }
    Ok(files)
}

/// A checked package manifest: it declares a package whose language and
/// IR versions this Orla supports.
pub fn package_manifest(
    store: &dyn Store,
    manifest: &Path,
) -> Result<orla_language::PackageDecl, String> {
    let text = store
        .read_text(manifest)
        .map_err(|e| format!("{}: {e}", manifest.display()))?;
    let document =
        orla_language::parse_document(&text).map_err(|e| format!("{}:{e}", manifest.display()))?;
    let Some(declared) = document.package else {
        return Err(format!(
            "{}: a package manifest begins `package name@version {{ language = N; ir = N; }}`",
            manifest.display()
        ));
    };
    // A directory named `name.orlapkg` holds the package `name`: the
    // name a `use` finds it by is the name it declares.
    let dir = manifest.parent().unwrap_or_else(|| Path::new("."));
    if dir.extension().is_some_and(|ext| ext == "orlapkg") {
        let named = dir.file_stem().unwrap_or_default().to_string_lossy();
        let declares = declared.id.value.split('@').next().unwrap_or_default();
        if named != declares {
            return Err(format!(
                "{}: the directory `{named}.orlapkg` declares the package `{}`; a package's directory is named for it, `{declares}.orlapkg`",
                manifest.display(),
                declared.id.value
            ));
        }
    }
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

/// Declarations a document shares: structs, type aliases, functions.
pub type Declarations = orla_language::Declarations;

/// Every document a `use` list reaches, recursively, each once,
/// dependencies first. A name is one package: two packages of one name
/// in the reach are refused, naming both.
pub fn used_documents(
    store: &dyn Store,
    base: &Path,
    uses: &[orla_language::Use],
) -> Result<Vec<PathBuf>, String> {
    use std::collections::{BTreeMap, BTreeSet};
    struct Reach {
        visited: BTreeSet<PathBuf>,
        packages: BTreeMap<String, PathBuf>,
        out: Vec<PathBuf>,
    }
    fn gather(
        store: &dyn Store,
        base: &Path,
        uses: &[orla_language::Use],
        reach: &mut Reach,
    ) -> Result<(), String> {
        for item in uses {
            let path = base.join(&item.path.value);
            used_through_its_manifest(store, base, &path)?;
            for file in expand_use(store, &path, item.path.value.ends_with('/'))? {
                let key = store.canonical(&file).unwrap_or_else(|_| file.clone());
                if !reach.visited.insert(key) {
                    continue;
                }
                if let Some(package) = package_holding(store, &file) {
                    one_package_of_its_name(store, &mut reach.packages, package)?;
                }
                let document = store.document(&file)?;
                let dir = file
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .to_path_buf();
                gather(store, &dir, &document.uses, reach)?;
                reach.out.push(file);
            }
        }
        Ok(())
    }
    let mut reach = Reach {
        visited: BTreeSet::new(),
        packages: BTreeMap::new(),
        out: Vec::new(),
    };
    gather(store, base, uses, &mut reach)?;
    Ok(reach.out)
}

/// Records `package` as the one package of its name in a reach;
/// another of the name already there refuses the reach.
fn one_package_of_its_name(
    store: &dyn Store,
    packages: &mut std::collections::BTreeMap<String, PathBuf>,
    package: PathBuf,
) -> Result<(), String> {
    let canonical = |p: &Path| store.canonical(p).unwrap_or_else(|_| p.to_path_buf());
    let name = package
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    match packages.get(&name) {
        Some(held) if canonical(held) != canonical(&package) => {
            let id = |dir: &Path| {
                package_manifest(store, &dir.join("package.osn"))
                    .map(|declared| declared.id.value)
                    .unwrap_or_else(|_| name.clone())
            };
            Err(format!(
                "two packages named `{name}` are reached, {} ({}) and {} ({}); a scene reaches one package of a name",
                held.display(),
                id(held),
                package.display(),
                id(&package)
            ))
        }
        Some(_) => Ok(()),
        None => {
            packages.insert(name, package);
            Ok(())
        }
    }
}

/// The declarations a document's `use` items bring in, recursively.
/// Functions, structs and types are scoped by document: a name reaches
/// what its own document declares, or else the one declaration of that
/// name in the documents it uses (an identical struct or type declared
/// by several is one). `own` is what the document that uses declares
/// itself; each declaration it sees by a name keeps the name, and any
/// other of that name is renamed `name@path` (see [`scoped`] and
/// [`scoped_types`]), each gathered declaration naming what its own
/// document sees.
pub fn declarations_for(
    store: &dyn Store,
    base: &Path,
    uses: &[orla_language::Use],
    own: orla_language::Own<'_>,
) -> Result<Declarations, String> {
    declarations_within(store, None, base, uses, own)
}

/// What a walk of the `use` items finds: each document reached, after
/// the ones it uses, with the documents it uses directly; and the one
/// package of each name.
#[derive(Default)]
struct Reach {
    documents: Vec<(PathBuf, std::sync::Arc<Document>)>,
    uses: std::collections::HashMap<PathBuf, Vec<PathBuf>>,
    packages: std::collections::BTreeMap<String, PathBuf>,
}

/// As [`declarations_for`], for a document at `origin`, so a `use`
/// back to it reports as a cycle.
pub fn declarations_within(
    store: &dyn Store,
    origin: Option<&Path>,
    base: &Path,
    uses: &[orla_language::Use],
    own: orla_language::Own<'_>,
) -> Result<Declarations, String> {
    /// The documents `uses` names directly, after walking each one.
    fn gather(
        store: &dyn Store,
        base: &Path,
        uses: &[orla_language::Use],
        visited: &mut std::collections::BTreeSet<PathBuf>,
        stack: &mut Vec<PathBuf>,
        reach: &mut Reach,
    ) -> Result<Vec<PathBuf>, String> {
        let mut direct = Vec::new();
        for item in uses {
            let path = base.join(&item.path.value);
            used_through_its_manifest(store, base, &path)?;
            for file in expand_use(store, &path, item.path.value.ends_with('/'))? {
                let key = store.canonical(&file).unwrap_or_else(|_| file.clone());
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
                if !direct.contains(&file) {
                    direct.push(file.clone());
                }
                if !visited.insert(key.clone()) {
                    continue;
                }
                if let Some(package) = package_holding(store, &file) {
                    one_package_of_its_name(store, &mut reach.packages, package)?;
                }
                stack.push(key);
                let document = store.document(&file)?;
                let dir = file
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .to_path_buf();
                let nested = gather(store, &dir, &document.uses, visited, stack, reach);
                stack.pop();
                reach.uses.insert(file.clone(), nested?);
                reach.documents.push((file.clone(), document));
            }
        }
        Ok(direct)
    }
    let mut visited = std::collections::BTreeSet::new();
    let mut stack: Vec<PathBuf> = origin
        .map(|path| vec![store.canonical(path).unwrap_or_else(|_| path.to_path_buf())])
        .unwrap_or_default();
    let mut reach = Reach::default();
    let direct = gather(store, base, uses, &mut visited, &mut stack, &mut reach)?;
    // The names are the language's: it is handed the documents and which
    // uses which.
    let place = |path: &Path| reach.documents.iter().position(|(p, _)| p == path);
    let sources: Vec<orla_language::Source<'_>> = reach
        .documents
        .iter()
        .map(|(path, document)| orla_language::Source {
            name: path.display().to_string(),
            functions: &document.functions,
            structs: &document.structs,
            types: &document.types,
            uses: reach
                .uses
                .get(path)
                .map_or(&[][..], Vec::as_slice)
                .iter()
                .filter_map(|p| place(p))
                .collect(),
        })
        .collect();
    let direct: Vec<usize> = direct.iter().filter_map(|p| place(p)).collect();
    // What the language makes of these documents depends on them alone:
    // a reach whose every document is as it was, from the same `use`
    // items and own declarations, is named as it was before.
    let key = scope_key(store, base, uses, own, &reach.documents);
    if let (Some(parsed), Some(key)) = (store.parsed(), &key)
        && let Some(named) = parsed
            .scoped
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(key)
    {
        return named.clone();
    }
    let named = orla_language::scope(&sources, &direct, own, &|name| {
        builtin_names().contains(name)
    });
    if let (Some(parsed), Some(key)) = (store.parsed(), key) {
        let mut scoped = parsed
            .scoped
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // A long session's edits leave keys behind; they go when there
        // are many.
        if scoped.len() >= 4096 {
            scoped.clear();
        }
        scoped.insert(key, named.clone());
    }
    named
}

/// What a scope is named from: where the `use` items are written and
/// what they say, the own declarations' key, and each document reached
/// with its stamp; none when a document has no stamp.
fn scope_key(
    store: &dyn Store,
    base: &Path,
    uses: &[orla_language::Use],
    own: orla_language::Own<'_>,
    reached: &[(PathBuf, std::sync::Arc<Document>)],
) -> Option<String> {
    use std::fmt::Write;
    let mut key = format!("{}", base.display());
    for item in uses {
        let _ = write!(key, "|use {}", item.path.value);
    }
    let _ = write!(
        key,
        "|own {}",
        declarations_key(store, own.structs, own.types, own.functions)
    );
    for (path, _) in reached {
        let stamp = store.stamp(path)?;
        let _ = write!(key, "|{}@{stamp:?}", path.display());
    }
    Some(key)
}

/// `path` as a document beside `document` names it: from `document`'s
/// directory when it lies under it, whole otherwise.
fn shown_from(store: &dyn Store, document: &Path, path: &Path) -> String {
    let base = document.parent().unwrap_or_else(|| Path::new("."));
    let base = store.canonical(base).unwrap_or_else(|_| base.to_path_buf());
    let path = store.canonical(path).unwrap_or_else(|_| path.to_path_buf());
    path.strip_prefix(&base)
        .unwrap_or(&path)
        .display()
        .to_string()
}

/// The function a diagnostic says hides the builtin of its name.
fn hidden_builtin(message: &str) -> Option<&str> {
    let rest = message.split("which hides the builtin `").nth(1)?;
    rest.split('`').next().filter(|name| !name.is_empty())
}

/// The language's builtins by name, gathered once.
fn builtin_names() -> &'static BTreeSet<&'static str> {
    static NAMES: std::sync::OnceLock<BTreeSet<&'static str>> = std::sync::OnceLock::new();
    NAMES.get_or_init(|| {
        orla_compiler::help::builtins()
            .into_iter()
            .map(|b| b.name)
            .collect()
    })
}

/// The backticked name a diagnostic reports as not a builtin, struct
/// or type, or as an unknown type.
fn missing_name(message: &str) -> Option<String> {
    let rest = message.rsplit(": ").next().unwrap_or(message);
    if let Some(unknown) = rest.strip_prefix("unknown type `") {
        let name = unknown.split('`').next()?;
        return (!name.is_empty()).then(|| name.to_owned());
    }
    let name = rest.strip_prefix('`')?.split('`').next()?;
    let says = rest.contains("is not a builtin")
        || rest.contains("is not a struct")
        || rest.contains("is not a type")
        || rest.contains("is not a port type");
    (says && !name.is_empty()).then(|| name.to_owned())
}

/// The document an error of a Kernel of `document` is written in: the
/// one a declaration the Kernel inherits was written in, when the error is
/// in one, else `document`. `held` is what the Kernel holds.
pub(crate) fn written_in(
    document: &Path,
    held: orla_language::Held<'_>,
    error: &orla_compiler::CompileError,
) -> PathBuf {
    error
        .declaration()
        .and_then(|name| held.origin_of(name))
        .map_or_else(
            || document.to_path_buf(),
            |origin| PathBuf::from(&*origin.document),
        )
}

/// What an error of a Kernel says, each name the Kernel knows an
/// inherited declaration by written as that declaration was written.
pub(crate) fn said(
    held: orla_language::Held<'_>,
    error: &orla_compiler::CompileError,
) -> crate::Diagnostic {
    let d = crate::Diagnostic::from(error);
    crate::Diagnostic {
        message: held.written(&d.message),
        ..d
    }
}

/// What the declarations `inherited` and a document's own hold.
pub(crate) fn held<'a>(
    inherited: (
        &'a [orla_language::StructDecl],
        &'a [orla_language::TypeAlias],
        &'a [orla_language::FnDecl],
    ),
) -> orla_language::Held<'a> {
    orla_language::Held {
        structs: inherited.0,
        types: inherited.1,
        functions: inherited.2,
    }
}

/// Checks the functions `document`, at `path`, declares, against it and
/// `inherited`, once; a fault is placed in the document it is written in.
pub(crate) fn functions_checked(
    path: &Path,
    document: &Document,
    inherited: (
        &[orla_language::StructDecl],
        &[orla_language::TypeAlias],
        &[orla_language::FnDecl],
    ),
) -> Result<(), crate::Diagnostic> {
    orla_compiler::check_functions(document, inherited).map_err(|error| {
        let at = written_in(path, held(inherited), &error);
        let d = said(held(inherited), &error);
        if at == path {
            d
        } else {
            crate::Diagnostic::new(format!("{}:{d}", at.display()))
        }
    })
}

/// The first file in `search` declaring `name`, and its kind.
fn declared_in(
    store: &dyn Store,
    name: &str,
    search: &[PathBuf],
) -> Option<(PathBuf, &'static str)> {
    for path in search {
        let Ok(document) = store.document(path) else {
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
fn relative_to(store: &dyn Store, path: &Path, from: &Path) -> String {
    let (path, from) = (
        store.canonical(path).unwrap_or_else(|_| path.to_path_buf()),
        store.canonical(from).unwrap_or_else(|_| from.to_path_buf()),
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

/// A key for a set of shared declarations: a hash, 128 bits, of all
/// they say and where (every name, type, body and place written), so a
/// change to any of them is another key. A function body is hashed once
/// while it is held: the store keeps each body's hash beside the
/// documents it parsed, by the body, and a body is shared by every
/// declaration that inherits it.
pub(crate) fn declarations_key(
    store: &dyn Store,
    structs: &[orla_language::StructDecl],
    types: &[orla_language::TypeAlias],
    functions: &[orla_language::FnDecl],
) -> String {
    use std::fmt::Write;
    let mut key = Twice::default();
    let _ = write!(key, "{structs:?}{types:?}");
    for function in functions {
        let orla_language::FnDecl {
            name,
            generics,
            args,
            ret,
            body,
            origin,
        } = function;
        let _ = write!(key, "fn{name:?}{generics:?}{args:?}{ret:?}{origin:?}");
        let _ = write!(key, "{:032x}", body_hash(store, body));
    }
    format!("{:032x}", key.finish())
}

/// The bodies hashed, by where each lies, with the body held weakly: an
/// entry whose body is gone is not its address's any more, and its
/// allocation is not reused while the entry holds it.
pub(crate) type BodyHashes =
    std::collections::HashMap<usize, (std::sync::Weak<orla_language::Block>, u128)>;

fn lock(parsed: &Parsed) -> std::sync::MutexGuard<'_, BodyHashes> {
    parsed
        .bodies
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A body's hash, from the store's bodies when it was hashed while held.
fn body_hash(store: &dyn Store, body: &std::sync::Arc<orla_language::Block>) -> u128 {
    use std::fmt::Write;
    let at = std::sync::Arc::as_ptr(body) as usize;
    if let Some(parsed) = store.parsed()
        && let Some((held, hash)) = lock(parsed).get(&at)
        && held
            .upgrade()
            .is_some_and(|alive| std::sync::Arc::ptr_eq(&alive, body))
    {
        return *hash;
    }
    let mut hashed = Twice::default();
    let _ = write!(hashed, "{body:?}");
    let hash = hashed.finish();
    if let Some(parsed) = store.parsed() {
        let mut bodies = lock(parsed);
        // Bodies dropped leave their entries; they go when there are many.
        if bodies.len() >= 4096 {
            bodies.retain(|_, (held, _)| held.strong_count() > 0);
        }
        bodies.insert(at, (std::sync::Arc::downgrade(body), hash));
    }
    hash
}

/// Text hashed two ways at once, for a 128-bit hash: SipHash with its
/// fixed keys, one of the two begun with a byte the other is not.
struct Twice(
    std::collections::hash_map::DefaultHasher,
    std::collections::hash_map::DefaultHasher,
);

impl Default for Twice {
    fn default() -> Self {
        use std::hash::Hasher;
        let mut second = std::collections::hash_map::DefaultHasher::new();
        second.write_u8(1);
        Self(std::collections::hash_map::DefaultHasher::new(), second)
    }
}

impl Twice {
    fn finish(&self) -> u128 {
        use std::hash::Hasher;
        (u128::from(self.0.finish()) << 64) | u128::from(self.1.finish())
    }
}

impl std::fmt::Write for Twice {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        use std::hash::Hasher;
        self.0.write(text.as_bytes());
        self.1.write(text.as_bytes());
        Ok(())
    }
}

pub(crate) fn named_id(kernel: &Kernel) -> Result<KernelId, crate::Diagnostic> {
    match &kernel.id {
        Some(id) => id
            .value
            .parse::<KernelId>()
            .map_err(|error| crate::Diagnostic::at(id.span, error.to_string())),
        None => Err(crate::Diagnostic {
            message: "a kernel outside a node needs an id".into(),
            at: Some((1, 1)),
        }),
    }
}

pub(crate) fn is_library(document: &Document) -> bool {
    document.nodes.is_empty()
        && document.inputs.is_empty()
        && document.outputs.is_empty()
        && document.params.is_empty()
}

/// Whether `package` is one the store finds by name: it lies in one of
/// the store's packages directories.
fn by_name(store: &dyn Store, package: &Path) -> bool {
    let canonical = |path: &Path| store.canonical(path).unwrap_or_else(|_| path.to_path_buf());
    let held = canonical(package);
    store
        .packages()
        .iter()
        .any(|dir| held.parent() == Some(canonical(dir).as_path()))
}

/// `error` extended with the `use` that would bring in its missing name,
/// when a file declaring it is found. `known` are the caller's
/// documents; `root` is their scene.
pub(crate) fn located(
    store: &dyn Store,
    error: crate::Diagnostic,
    document: &Path,
    root: &Path,
    known: &[PathBuf],
) -> crate::Diagnostic {
    // A function hiding a builtin: the document that declares it.
    // A builtin is hidden only in the document of the function hiding it.
    if hidden_builtin(&error.message).is_some() {
        let shown = shown_from(store, document, document);
        return crate::Diagnostic {
            message: format!("{}, declared in {shown}", error.message),
            ..error
        };
    }
    let Some(name) = missing_name(&error.message) else {
        return error;
    };
    // Search known documents and `.orla` files beside this document,
    // the scene and each known one, excluding what this already uses.
    let used: Vec<PathBuf> = store
        .document(document)
        .ok()
        .map(|parsed| {
            used_documents(
                store,
                document.parent().unwrap_or_else(|| Path::new(".")),
                &parsed.uses,
            )
            .unwrap_or_default()
        })
        .unwrap_or_default();
    // Two documents this one uses each declare the name: it reaches
    // neither. A struct or type declared alike by both is one.
    let kind = |parsed: &Document| {
        if parsed.functions.iter().any(|f| f.name.value == name) {
            Some("fn")
        } else if parsed.structs.iter().any(|d| d.name.value == name) {
            Some("struct")
        } else if parsed.types.iter().any(|d| d.name.value == name) {
            Some("type")
        } else {
            None
        }
    };
    // What it names in its own `use` items: the names it sees are theirs.
    let base = document.parent().unwrap_or_else(|| Path::new("."));
    let direct: Vec<PathBuf> = store
        .document(document)
        .ok()
        .map(|parsed| {
            parsed
                .uses
                .iter()
                .flat_map(|item| {
                    expand_use(
                        store,
                        &base.join(&item.path.value),
                        item.path.value.ends_with('/'),
                    )
                    .unwrap_or_default()
                })
                .collect()
        })
        .unwrap_or_default();
    let declares = |path: &&PathBuf| {
        store
            .document(path)
            .is_ok_and(|parsed| kind(&parsed).is_some())
    };
    let declaring: Vec<&PathBuf> = direct.iter().filter(declares).collect();
    // Declared only further down: reached through another document's
    // `use`, which brings its names to that document alone.
    if declaring.is_empty()
        && let Some(deeper) = used.iter().find(declares)
    {
        let what = store
            .document(deeper)
            .ok()
            .and_then(|parsed| kind(&parsed))
            .unwrap_or("fn");
        let spelling = match package_holding(store, deeper) {
            // A package the store finds by name is used by its name.
            Some(package) if by_name(store, &package) => package
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| relative_to(store, &package, base)),
            Some(package) => relative_to(store, &package, base),
            None => relative_to(store, deeper, base),
        };
        return crate::Diagnostic {
            message: format!(
                "{}; {what} `{name}` is declared in {}, which this document reaches only through another's `use`: a document sees what it uses itself, `use \"{spelling}\";`",
                error.message,
                relative_to(store, deeper, base)
            ),
            at: error.at,
        };
    }
    let what = declaring
        .first()
        .and_then(|path| store.document(path).ok())
        .and_then(|parsed| kind(&parsed))
        .unwrap_or("fn");
    if declaring.len() > 1 {
        let names: Vec<String> = declaring
            .iter()
            .map(|path| relative_to(store, path, base))
            .collect();
        return crate::Diagnostic {
            message: format!(
                "{}; {what} `{name}` is declared by {}, which this document uses, and a name reaches one declaration: declare `{name}` in this document, or rename one of them",
                error.message,
                names.join(" and ")
            ),
            at: error.at,
        };
    }
    let mut search: Vec<PathBuf> = Vec::new();
    let siblings = |dir: &Path, search: &mut Vec<PathBuf>| {
        if let Ok(entries) = store.list(dir) {
            let mut files: Vec<PathBuf> = entries
                .into_iter()
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
    siblings(root.parent().unwrap_or_else(|| Path::new(".")), &mut search);
    for path in known {
        siblings(path.parent().unwrap_or_else(|| Path::new(".")), &mut search);
    }
    search.extend(known.iter().cloned());
    search.retain(|path| {
        path != document
            && !used.iter().any(|seen| seen == path)
            && path.extension().is_some_and(|ext| ext == "orla")
    });
    search.dedup();
    let Some((file, kind)) = declared_in(store, &name, &search) else {
        return error;
    };
    let spelling = relative_to(
        store,
        &file,
        document.parent().unwrap_or_else(|| Path::new(".")),
    );
    crate::Diagnostic {
        message: format!(
            "{}; {kind} `{name}` is declared in {spelling}, which this document does not use: `use \"{spelling}\";`",
            error.message
        ),
        at: error.at,
    }
}

/// The nodes of `graph` binding a Kernel nothing the scene publishes: not
/// its own (`own`, its document's and its inline ones), not a native
/// operation, not one a document its `use`s name writes. Each comes with
/// why, when a document the scene reaches through another writes it: a
/// scene publishes the Kernels of the documents it uses itself, and
/// reaches only the declarations of what those use.
pub(crate) fn unpublished(
    store: &dyn Store,
    root: &Path,
    graph: &orla_graph::Graph,
    uses: &[(PathBuf, bool)],
    own: &[KernelId],
) -> Vec<(orla_graph::NodeId, KernelId, Option<String>)> {
    let files = Expanded::of(store, uses.to_vec()).files;
    let writes = |path: &Path, id: &KernelId| {
        store.document(path).is_ok_and(|parsed| {
            parsed
                .kernels
                .iter()
                .any(|k| named_id(k).ok().as_ref() == Some(id))
        })
    };
    let shown = |path: &Path| shown_from(store, root, path);
    graph
        .nodes
        .iter()
        .filter(|node| !node.kernel.name().starts_with("native."))
        .filter(|node| !own.contains(&node.kernel))
        .filter(|node| !files.iter().any(|file| writes(file, &node.kernel)))
        .map(|node| {
            let through = files.iter().find_map(|file| {
                let parsed = store.document(file).ok()?;
                let dir = file.parent().unwrap_or_else(|| Path::new("."));
                used_documents(store, dir, &parsed.uses)
                    .ok()?
                    .into_iter()
                    .find(|reached| writes(reached, &node.kernel))
                    .map(|reached| (reached, file.clone()))
            });
            let why = through.map(|(reached, via)| {
                format!(
                    "`{}` is written in {}, which the scene reaches only through {}: a scene publishes the Kernels of the documents it uses itself, so it wants `use \"{}\";`",
                    node.kernel,
                    shown(&reached),
                    shown(&via),
                    shown(&reached)
                )
            });
            (node.id.clone(), node.kernel.clone(), why)
        })
        .collect()
}
