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
    let Some(package) = path.ancestors().skip(1).find(|dir| {
        dir.extension().is_some_and(|ext| ext == "orlapkg")
            && store.is_file(&dir.join("package.osn"))
    }) else {
        return Ok(());
    };
    if base.starts_with(canonical(package)) {
        return Ok(());
    }
    let name = package.file_name().unwrap_or_default().to_string_lossy();
    Err(format!(
        "`{}` is inside the package {name}; use the package whole, `use \".../{name}\";`, which its manifest is held to",
        path.display()
    ))
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
/// (`picture.orlapkg/graphs/x.osn`) to the existing file inside it.
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

/// A missing path, looked up by name as a package in each of the store's
/// [`Store::packages`].
fn resolve_package(store: &dyn Store, path: &Path) -> PathBuf {
    if store.exists(path) {
        return path.to_path_buf();
    }
    let Some(name) = path.file_name() else {
        return path.to_path_buf();
    };
    for dir in store.packages() {
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
pub type Declarations = (
    Vec<orla_language::StructDecl>,
    Vec<orla_language::TypeAlias>,
    Vec<orla_language::FnDecl>,
);

/// Every document a `use` list reaches, recursively, each once,
/// dependencies first.
pub fn used_documents(
    store: &dyn Store,
    base: &Path,
    uses: &[orla_language::Use],
) -> Result<Vec<PathBuf>, String> {
    fn gather(
        store: &dyn Store,
        base: &Path,
        uses: &[orla_language::Use],
        visited: &mut std::collections::BTreeSet<PathBuf>,
        out: &mut Vec<PathBuf>,
    ) -> Result<(), String> {
        for item in uses {
            let path = base.join(&item.path.value);
            used_through_its_manifest(store, base, &path)?;
            for file in expand_use(store, &path, item.path.value.ends_with('/'))? {
                let key = store.canonical(&file).unwrap_or_else(|_| file.clone());
                if !visited.insert(key) {
                    continue;
                }
                let document = store.document(&file)?;
                let dir = file
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .to_path_buf();
                gather(store, &dir, &document.uses, visited, out)?;
                out.push(file);
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    gather(
        store,
        base,
        uses,
        &mut std::collections::BTreeSet::new(),
        &mut out,
    )?;
    Ok(out)
}

/// The declarations a document's `use` items bring in, recursively. An
/// identical declaration reached twice is one; conflicting ones are
/// refused.
pub fn declarations_for(
    store: &dyn Store,
    base: &Path,
    uses: &[orla_language::Use],
) -> Result<Declarations, String> {
    declarations_within(store, None, base, uses)
}

/// Where each gathered declaration is, by name.
#[derive(Default)]
struct Named {
    structs: std::collections::HashMap<String, usize>,
    types: std::collections::HashMap<String, usize>,
    functions: std::collections::HashMap<String, usize>,
}

/// As [`declarations_for`], for a document at `origin`, so a `use`
/// back to it reports as a cycle.
pub fn declarations_within(
    store: &dyn Store,
    origin: Option<&Path>,
    base: &Path,
    uses: &[orla_language::Use],
) -> Result<Declarations, String> {
    #[allow(clippy::too_many_arguments)]
    fn gather(
        store: &dyn Store,
        base: &Path,
        uses: &[orla_language::Use],
        visited: &mut std::collections::BTreeSet<PathBuf>,
        stack: &mut Vec<PathBuf>,
        structs: &mut Vec<(orla_language::StructDecl, PathBuf)>,
        types: &mut Vec<(orla_language::TypeAlias, PathBuf)>,
        functions: &mut Vec<(orla_language::FnDecl, PathBuf)>,
        named: &mut Named,
    ) -> Result<(), String> {
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
                if !visited.insert(key.clone()) {
                    continue;
                }
                stack.push(key);
                let document = store.document(&file)?;
                let dir = file
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .to_path_buf();
                let nested = gather(
                    store,
                    &dir,
                    &document.uses,
                    visited,
                    stack,
                    structs,
                    types,
                    functions,
                    named,
                );
                stack.pop();
                nested?;
                for decl in &document.functions {
                    // A function name is declared in one file only.
                    if let Some((_, other)) = named
                        .functions
                        .get(&decl.name.value)
                        .map(|at| &functions[*at])
                    {
                        return Err(format!(
                            "fn `{}` is declared by both {} and {}",
                            decl.name.value,
                            other.display(),
                            file.display()
                        ));
                    }
                    named
                        .functions
                        .insert(decl.name.value.clone(), functions.len());
                    functions.push((decl.clone(), file.clone()));
                }
                for decl in &document.structs {
                    match named.structs.get(&decl.name.value).map(|at| &structs[*at]) {
                        // Same fields in the same order: one declaration.
                        Some((same, _))
                            if same
                                .fields
                                .iter()
                                .map(|(name, ty)| (&name.value, &ty.value))
                                .eq(decl
                                    .fields
                                    .iter()
                                    .map(|(name, ty)| (&name.value, &ty.value))) => {}
                        Some((_, other)) => {
                            return Err(format!(
                                "struct `{}` is declared differently by {} and {}",
                                decl.name.value,
                                other.display(),
                                file.display()
                            ));
                        }
                        None => {
                            named.structs.insert(decl.name.value.clone(), structs.len());
                            structs.push((decl.clone(), file.clone()));
                        }
                    }
                }
                for decl in &document.types {
                    match named.types.get(&decl.name.value).map(|at| &types[*at]) {
                        Some((same, _)) if same.target.value == decl.target.value => {}
                        Some((_, other)) => {
                            return Err(format!(
                                "type `{}` is declared differently by {} and {}",
                                decl.name.value,
                                other.display(),
                                file.display()
                            ));
                        }
                        None => {
                            named.types.insert(decl.name.value.clone(), types.len());
                            types.push((decl.clone(), file.clone()));
                        }
                    }
                }
            }
        }
        Ok(())
    }
    let mut visited = std::collections::BTreeSet::new();
    let mut stack: Vec<PathBuf> = origin
        .map(|path| vec![store.canonical(path).unwrap_or_else(|_| path.to_path_buf())])
        .unwrap_or_default();
    let (mut structs, mut types, mut functions) = (Vec::new(), Vec::new(), Vec::new());
    gather(
        store,
        base,
        uses,
        &mut visited,
        &mut stack,
        &mut structs,
        &mut types,
        &mut functions,
        &mut Named::default(),
    )?;
    Ok((
        structs.into_iter().map(|(s, _)| s).collect(),
        types.into_iter().map(|(t, _)| t).collect(),
        functions.into_iter().map(|(f, _)| f).collect(),
    ))
}

/// The backticked name a diagnostic reports as not a builtin, struct
/// or type.
fn missing_name(message: &str) -> Option<String> {
    let rest = message.rsplit(": ").next().unwrap_or(message);
    let name = rest.strip_prefix('`')?.split('`').next()?;
    let says = rest.contains("is not a builtin")
        || rest.contains("is not a struct")
        || rest.contains("is not a type")
        || rest.contains("is not a port type");
    (says && !name.is_empty()).then(|| name.to_owned())
}

/// The document an error of a Kernel in `document` is written in:
/// `document`, or, for one in the body of a function `document` does not
/// declare but uses, the used document that declares it.
pub(crate) fn written_in(
    store: &dyn Store,
    document: &Path,
    error: &orla_compiler::CompileError,
) -> PathBuf {
    let Some(function) = error.function() else {
        return document.to_path_buf();
    };
    let Ok(parsed) = store.document(document) else {
        return document.to_path_buf();
    };
    if parsed.functions.iter().any(|f| f.name.value == function) {
        return document.to_path_buf();
    }
    let base = document.parent().unwrap_or_else(|| Path::new("."));
    let used = used_documents(store, base, &parsed.uses).unwrap_or_default();
    match declared_in(store, function, &used) {
        Some((path, "fn")) => path,
        _ => document.to_path_buf(),
    }
}

/// Checks the functions `document`, at `path`, declares, against it and
/// `inherited`, once; a fault is placed in the document it is written in.
pub(crate) fn functions_checked(
    store: &dyn Store,
    path: &Path,
    document: &Document,
    inherited: (
        &[orla_language::StructDecl],
        &[orla_language::TypeAlias],
        &[orla_language::FnDecl],
    ),
) -> Result<(), crate::Diagnostic> {
    orla_compiler::check_functions(document, inherited).map_err(|error| {
        let at = written_in(store, path, &error);
        if at == path {
            crate::Diagnostic::from(&error)
        } else {
            crate::Diagnostic::new(format!("{}:{error}", at.display()))
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

/// A key for a set of shared declarations.
pub(crate) fn declarations_key(
    structs: &[orla_language::StructDecl],
    types: &[orla_language::TypeAlias],
    functions: &[orla_language::FnDecl],
) -> String {
    format!("{structs:?}{types:?}{functions:?}")
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
