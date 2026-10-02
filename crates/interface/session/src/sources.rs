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

/// A struct or a type a document declares.
#[derive(Clone)]
enum Declared {
    Struct(orla_language::StructDecl),
    Type(orla_language::TypeAlias),
}

impl Declared {
    fn name(&self) -> &str {
        match self {
            Declared::Struct(d) => &d.name.value,
            Declared::Type(d) => &d.name.value,
        }
    }

    /// Whether `other` declares the same: a struct with the same fields in
    /// the same order, or a type of the same target.
    fn same(&self, other: &Declared) -> bool {
        match (self, other) {
            (Declared::Struct(a), Declared::Struct(b)) => a
                .fields
                .iter()
                .map(|(name, ty)| (&name.value, &ty.value))
                .eq(b.fields.iter().map(|(name, ty)| (&name.value, &ty.value))),
            (Declared::Type(a), Declared::Type(b)) => a.target.value == b.target.value,
            _ => false,
        }
    }

    /// This declaration named `name`, each struct or type named in it that
    /// `rename` maps written by the new name.
    fn renamed(&self, name: String, rename: &dyn Fn(&str) -> Option<String>) -> Declared {
        let retype = |ty: &orla_language::Spanned<String>| orla_language::Spanned {
            value: orla_language::renamed_type(&ty.value, rename),
            span: ty.span,
        };
        match self {
            Declared::Struct(d) => Declared::Struct(orla_language::StructDecl {
                name: orla_language::Spanned {
                    value: name,
                    span: d.name.span,
                },
                fields: d
                    .fields
                    .iter()
                    .map(|(field, ty)| (field.clone(), retype(ty)))
                    .collect(),
            }),
            // An enum's variants name no type.
            Declared::Type(d) => Declared::Type(orla_language::TypeAlias {
                name: orla_language::Spanned {
                    value: name,
                    span: d.name.span,
                },
                target: if d.variants().is_some() {
                    d.target.clone()
                } else {
                    retype(&d.target)
                },
            }),
        }
    }
}

/// What a walk of the `use` items gathers: each document's functions,
/// dependencies first; each distinct struct or type, with every document
/// that declares it; and the documents each one uses directly.
#[derive(Default)]
struct Reach {
    functions: Vec<(orla_language::FnDecl, PathBuf)>,
    declared: Vec<(Declared, Vec<PathBuf>)>,
    uses: std::collections::HashMap<PathBuf, Vec<PathBuf>>,
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
                stack.push(key);
                let document = store.document(&file)?;
                let dir = file
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .to_path_buf();
                let nested = gather(store, &dir, &document.uses, visited, stack, reach);
                stack.pop();
                reach.uses.insert(file.clone(), nested?);
                for decl in &document.functions {
                    reach.functions.push((decl.clone(), file.clone()));
                }
                let declared = document
                    .structs
                    .iter()
                    .map(|d| Declared::Struct(d.clone()))
                    .chain(document.types.iter().map(|d| Declared::Type(d.clone())));
                for decl in declared {
                    match reach
                        .declared
                        .iter_mut()
                        .find(|(d, _)| d.name() == decl.name() && d.same(&decl))
                    {
                        Some((_, files)) => {
                            if !files.contains(&file) {
                                files.push(file.clone());
                            }
                        }
                        None => reach.declared.push((decl, vec![file.clone()])),
                    }
                }
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
    let (structs, types) = scoped_types(&mut reach, &direct, own)?;
    Ok((structs, types, scoped(&reach, &direct, own.functions)?))
}

/// The documents `from` reaches through `use`, recursively.
fn closure<'a>(
    from: &'a [PathBuf],
    uses: &'a std::collections::HashMap<PathBuf, Vec<PathBuf>>,
    out: &mut std::collections::BTreeSet<&'a Path>,
) {
    for file in from {
        if out.insert(file) {
            closure(uses.get(file).map_or(&[][..], Vec::as_slice), uses, out);
        }
    }
}

/// The gathered structs and types, each under one name in the using
/// document's reach, as [`scoped`] names functions: a name two distinct
/// declarations share, or one the using document declares itself, is
/// the using document's own, or else the one it reaches; every other
/// declaration of it is renamed `name@path`; the using document's own
/// declaration alike to the one it reaches is that one, and no clash.
/// Each gathered declaration,
/// and each gathered function, then names the structs and types its own
/// document sees. A name a document reaches two declarations of, and
/// declares neither, is refused where that document names it.
fn scoped_types(
    reach: &mut Reach,
    direct: &[PathBuf],
    own: orla_language::Own<'_>,
) -> Result<
    (
        Vec<orla_language::StructDecl>,
        Vec<orla_language::TypeAlias>,
    ),
    String,
> {
    use std::collections::{BTreeSet, HashMap};
    let named = |name: &str| {
        reach
            .declared
            .iter()
            .filter(|(d, _)| d.name() == name)
            .count()
    };
    // What the using document declares itself of a name, if anything: a
    // declaration alike to the one it reaches is that one.
    let own_decl = |name: &str| -> Option<Declared> {
        own.structs
            .iter()
            .find(|d| d.name.value == name)
            .map(|d| Declared::Struct(d.clone()))
            .or_else(|| {
                own.types
                    .iter()
                    .find(|d| d.name.value == name)
                    .map(|d| Declared::Type(d.clone()))
            })
    };
    let own_differs = |name: &str| {
        own_decl(name).is_some_and(|mine| {
            reach
                .declared
                .iter()
                .any(|(d, _)| d.name() == name && !d.same(&mine))
        })
    };
    let clashing = |name: &str| named(name) > 1 || own_differs(name);
    // A reached declaration the using document makes alike itself is its
    // own, and is not gathered again.
    let own_alike = |d: &Declared| own_decl(d.name()).is_some_and(|mine| d.same(&mine));
    let split = |declared: Vec<Declared>| {
        let mut structs = Vec::new();
        let mut types = Vec::new();
        for decl in declared {
            match decl {
                Declared::Struct(d) => structs.push(d),
                Declared::Type(d) => types.push(d),
            }
        }
        (structs, types)
    };
    // Nothing declared twice and nothing the using document declares:
    // every name is plain, as written.
    if !reach.declared.iter().any(|(d, _)| clashing(d.name())) {
        return Ok(split(
            reach
                .declared
                .iter()
                .filter(|(d, _)| !own_alike(d))
                .map(|(d, _)| d.clone())
                .collect(),
        ));
    }
    let mut top = BTreeSet::new();
    closure(direct, &reach.uses, &mut top);
    let reached_from = |file: &Path| {
        let mut reached = BTreeSet::new();
        closure(
            reach.uses.get(file).map_or(&[][..], Vec::as_slice),
            &reach.uses,
            &mut reached,
        );
        reached
    };
    let declared = &reach.declared;
    // The declarations of `name` a set of documents reaches.
    let reaching = |name: &str, reached: &BTreeSet<&Path>| -> Vec<usize> {
        (0..declared.len())
            .filter(|&i| {
                declared[i].0.name() == name
                    && declared[i]
                        .1
                        .iter()
                        .any(|file| reached.contains(file.as_path()))
            })
            .collect()
    };
    let plain = |i: usize| {
        let name = declared[i].0.name();
        !own_differs(name) && reaching(name, &top) == [i]
    };
    let out_name = |i: usize| {
        let name = declared[i].0.name();
        if plain(i) {
            name.to_owned()
        } else {
            scoped_name(name, &declared[i].1[0])
        }
    };
    // How a document names what it sees: the declaration it makes itself,
    // or the one it reaches; two it reaches and neither its own is refused.
    let mut refused: Option<String> = None;
    let mut reaches: HashMap<PathBuf, BTreeSet<&Path>> = HashMap::new();
    let mut view = |file: &Path, name: &str, refused: &mut Option<String>| -> Option<String> {
        if !clashing(name) {
            return None;
        }
        let target = match (0..declared.len())
            .find(|&i| declared[i].0.name() == name && declared[i].1.iter().any(|f| f == file))
        {
            Some(i) => i,
            None => {
                let reached = reaches
                    .entry(file.to_path_buf())
                    .or_insert_with(|| reached_from(file));
                let found = reaching(name, reached);
                match found.as_slice() {
                    [] => return None,
                    [one] => *one,
                    many => {
                        let files: Vec<String> = many
                            .iter()
                            .map(|&i| declared[i].1[0].display().to_string())
                            .collect();
                        refused.get_or_insert_with(|| {
                            format!(
                                "`{name}` is declared differently by {}, which {} uses: declare `{name}` in it, or rename one of them",
                                files.join(" and "),
                                file.display()
                            )
                        });
                        return None;
                    }
                }
            }
        };
        let renamed = out_name(target);
        (renamed != name).then_some(renamed)
    };
    let mut out = Vec::with_capacity(declared.len());
    for (i, (decl, files)) in declared.iter().enumerate() {
        if own_alike(decl) {
            continue;
        }
        let names = std::cell::RefCell::new(|name: &str| view(&files[0], name, &mut refused));
        let rename = |name: &str| (names.borrow_mut())(name);
        out.push(decl.renamed(out_name(i), &rename));
    }
    let functions: Vec<(orla_language::FnDecl, PathBuf)> = reach
        .functions
        .iter()
        .map(|(decl, file)| {
            let names = std::cell::RefCell::new(|name: &str| view(file, name, &mut refused));
            let rename = |name: &str| (names.borrow_mut())(name);
            (decl.retyped(&rename), file.clone())
        })
        .collect();
    if let Some(refused) = refused {
        return Err(refused);
    }
    reach.functions = functions;
    Ok(split(out))
}

/// The name a function of `file` takes where another function of the
/// document that uses holds its own name.
pub(crate) fn scoped_name(name: &str, file: &Path) -> String {
    format!("{name}@{}", file.display())
}

/// The file a `name@path` function name was renamed from, and its name.
pub(crate) fn unscoped(name: &str) -> Option<(&str, &Path)> {
    name.split_once('@')
        .map(|(name, path)| (name, Path::new(path)))
}

/// The gathered functions, each under one name in the using document's
/// reach. Each document sees its own functions and those of every
/// document it reaches through `use`; a name two of those declare is
/// the document's own, or else ambiguous. The using document's view
/// keeps plain names: each function it sees by name keeps that name,
/// every other is renamed `name@path`, and each body's calls follow its
/// own document's view.
fn scoped(
    reach: &Reach,
    direct: &[PathBuf],
    own: &[orla_language::FnDecl],
) -> Result<Vec<orla_language::FnDecl>, String> {
    use std::collections::{BTreeSet, HashMap};
    // Which documents declare each name.
    let mut declaring: HashMap<&str, Vec<&Path>> = HashMap::new();
    for (decl, file) in &reach.functions {
        let files = declaring.entry(decl.name.value.as_str()).or_default();
        if !files.contains(&file.as_path()) {
            files.push(file);
        }
    }
    // Nothing declared twice and nothing the user shadows: every name is
    // plain, as written.
    let clashing = |name: &str| {
        declaring.get(name).is_some_and(|files| files.len() > 1)
            || own.iter().any(|f| f.name.value == name)
    };
    if !reach
        .functions
        .iter()
        .any(|(decl, _)| clashing(&decl.name.value))
    {
        return Ok(reach
            .functions
            .iter()
            .map(|(decl, _)| decl.clone())
            .collect());
    }
    // Each document's reach: what it uses, recursively.
    let mut top = BTreeSet::new();
    closure(direct, &reach.uses, &mut top);
    // The document a name reaches from `reached`: one declaring it.
    let resolve = |name: &str, reached: &BTreeSet<&Path>| -> Option<&Path> {
        let files: Vec<&Path> = declaring
            .get(name)?
            .iter()
            .copied()
            .filter(|file| reached.contains(file))
            .collect();
        (files.len() == 1).then(|| files[0])
    };
    let seen_by_user = |name: &str, file: &Path| {
        !own.iter().any(|f| f.name.value == name) && resolve(name, &top) == Some(file)
    };
    let internal = |name: &str, file: &Path| {
        if seen_by_user(name, file) {
            name.to_owned()
        } else {
            scoped_name(name, file)
        }
    };
    let mut reaches: HashMap<&Path, BTreeSet<&Path>> = HashMap::new();
    let mut out = Vec::with_capacity(reach.functions.len());
    for (decl, file) in &reach.functions {
        let reached = reaches.entry(file.as_path()).or_insert_with(|| {
            let mut reached = BTreeSet::new();
            closure(
                reach.uses.get(file).map_or(&[][..], Vec::as_slice),
                &reach.uses,
                &mut reached,
            );
            reached
        });
        let reached = &*reached;
        let declares = |name: &str| {
            reach
                .functions
                .iter()
                .any(|(d, f)| f == file && d.name.value == name)
        };
        let rename = |name: &str| -> Option<String> {
            let target = if declares(name) {
                file.as_path()
            } else {
                resolve(name, reached)?
            };
            let renamed = internal(name, target);
            (renamed != name).then_some(renamed)
        };
        out.push(decl.renamed(internal(&decl.name.value, file), &rename));
    }
    Ok(out)
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

/// The document an error of a Kernel in `document` is written in:
/// `document`, or, for one in a declaration the Kernel inherits from a
/// used document, the used document that declares it. `kernel` is the
/// byte range of the Kernel compiled, when one is: a declaration of its
/// own keeps its name, and the inherited one of that name is renamed out
/// of its way.
pub(crate) fn written_in(
    store: &dyn Store,
    document: &Path,
    kernel: Option<&std::ops::Range<usize>>,
    error: &orla_compiler::CompileError,
) -> PathBuf {
    match error.declaration() {
        Some(name) => declaration_written_in(store, document, kernel, name),
        None => document.to_path_buf(),
    }
}

/// The document a declaration a Kernel of `document` names is written
/// in: `document` for its own, or the used document that declares it.
/// The name is the one the Kernel knows it by: `name@path` for one a
/// clash renamed, `name@shared` for an inherited one the Kernel's own
/// shadows, `name<binding>` for an instance of a function taking a
/// Field.
pub(crate) fn declaration_written_in(
    store: &dyn Store,
    document: &Path,
    kernel: Option<&std::ops::Range<usize>>,
    name: &str,
) -> PathBuf {
    let name = name.split('<').next().unwrap_or(name);
    let (name, shared) = match name.strip_suffix("@shared") {
        Some(name) => (name, true),
        None => (name, false),
    };
    if let Some((_, path)) = unscoped(name) {
        return path.to_path_buf();
    }
    let Ok(parsed) = store.document(document) else {
        return document.to_path_buf();
    };
    let declares = |functions: &[orla_language::FnDecl],
                    structs: &[orla_language::StructDecl],
                    types: &[orla_language::TypeAlias]| {
        functions.iter().any(|d| d.name.value == name)
            || structs.iter().any(|d| d.name.value == name)
            || types.iter().any(|d| d.name.value == name)
    };
    let own_kernel = || {
        parsed
            .kernels
            .iter()
            .filter(|k| Some(&k.range) == kernel)
            .any(|k| declares(&k.functions, &k.structs, &k.types))
    };
    if declares(&parsed.functions, &parsed.structs, &parsed.types) || (!shared && own_kernel()) {
        return document.to_path_buf();
    }
    let base = document.parent().unwrap_or_else(|| Path::new("."));
    let used = used_documents(store, base, &parsed.uses).unwrap_or_default();
    match declared_in(store, name, &used) {
        Some((path, _)) => path,
        None => document.to_path_buf(),
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
        let at = written_in(store, path, None, &error);
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
    let declaring: Vec<&PathBuf> = used
        .iter()
        .filter(|path| {
            store
                .document(path)
                .is_ok_and(|parsed| kind(&parsed).is_some())
        })
        .collect();
    let what = declaring
        .first()
        .and_then(|path| store.document(path).ok())
        .and_then(|parsed| kind(&parsed))
        .unwrap_or("fn");
    if declaring.len() > 1 {
        let base = document.parent().unwrap_or_else(|| Path::new("."));
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
