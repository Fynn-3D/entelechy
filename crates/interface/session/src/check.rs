//! `entelechy check`: a document checked through a store, without a session.

use std::path::Path;

use crate::sources::{held, located, said, written_in};
use crate::{Store, declarations_for, locate_document, read_document, used_documents};

/// The words a Kernel may carry through its branches before it is said
/// to be large ([`entelechy_ir::carried`]): a device's compiler takes long over
/// more (seconds, past in proportion), or fails on it.
pub const LARGE: u64 = 5_000;

/// What makes a Kernel large, when it is ([`LARGE`]): the words it carries
/// through its branches, the structs they are, and the functions that
/// carry most, each with the calls that reach it.
pub fn large(module: &entelechy_ir::Validated) -> Option<String> {
    large_past(module, LARGE)
}

/// [`large`], past `limit` words.
pub fn large_past(module: &entelechy_ir::Validated, limit: u64) -> Option<String> {
    let carried = entelechy_ir::carried(module);
    if carried.total <= limit {
        return None;
    }
    let mut by_struct: Vec<usize> = (0..carried.structs.len())
        .filter(|&s| carried.structs[s] > 0)
        .collect();
    let words = |s: usize| {
        entelechy_ir::words(&module.structs, entelechy_ir::Type::Struct(s as u32)) as u64
    };
    by_struct.sort_by_key(|&s| std::cmp::Reverse(carried.structs[s] * words(s)));
    let structs: Vec<String> = by_struct
        .into_iter()
        .take(3)
        .map(|s| {
            format!(
                "`{}` ({} words) {} times",
                module.structs[s].name,
                words(s),
                carried.structs[s]
            )
        })
        .collect();
    let sizes = entelechy_ir::sizes(module);
    let name = |f: u32| module.functions[f as usize].name.clone();
    let mut by_function: Vec<usize> = (0..carried.functions.len())
        .filter(|&f| carried.functions[f] > 0)
        .collect();
    by_function.sort_by_key(|&f| std::cmp::Reverse(carried.functions[f]));
    let mut most: Vec<(u64, String)> = by_function
        .into_iter()
        .take(3)
        .map(|f| {
            let way: Vec<String> = sizes.way_to(f as u32).into_iter().map(name).collect();
            let by = if way.len() <= 1 {
                "called by the Kernel itself".to_owned()
            } else {
                format!("by {}", way.join(" → "))
            };
            (
                carried.functions[f],
                format!("`{}` ({}, {by})", name(f as u32), carried.functions[f]),
            )
        })
        .collect();
    if carried.roots > 0 {
        most.push((
            carried.roots,
            format!("the Kernel's own body ({})", carried.roots),
        ));
    }
    most.sort_by_key(|(words, _)| std::cmp::Reverse(*words));
    let most: Vec<String> = most.into_iter().take(3).map(|(_, said)| said).collect();
    Some(format!(
        "`{}` carries {} words through its branches, past the {limit} a device's compiler takes in comfort: {}; most in {}. A narrower struct, or state kept in several rows, carries fewer",
        module.manifest.id,
        carried.total,
        structs.join(", "),
        most.join(", ")
    ))
}

/// The Kernels each document of a scene publishes that no node of its
/// graph uses, by document: each is compiled and prepared with the scene
/// all the same. A package's are left out: they are its verbs, used or
/// not.
pub(crate) fn idle(
    store: &dyn Store,
    graph: &entelechy_graph::Graph,
    published: impl IntoIterator<Item = (std::path::PathBuf, Vec<entelechy_ir::KernelId>)>,
) -> Vec<(std::path::PathBuf, Vec<entelechy_ir::KernelId>)> {
    let bound: std::collections::BTreeSet<&entelechy_ir::KernelId> =
        graph.nodes.iter().map(|node| &node.kernel).collect();
    published
        .into_iter()
        .filter(|(path, _)| crate::sources::package_holding(store, path).is_none())
        .filter_map(|(path, ids)| {
            let ids: Vec<entelechy_ir::KernelId> =
                ids.into_iter().filter(|id| !bound.contains(id)).collect();
            (!ids.is_empty()).then_some((path, ids))
        })
        .collect()
}

/// How [`idle`] Kernels are said.
pub(crate) fn idle_said(path: &Path, ids: &[entelechy_ir::KernelId]) -> String {
    let ids: Vec<String> = ids.iter().map(|id| format!("`{id}`")).collect();
    format!(
        "{}: publishes {}, which no node of the scene uses: compiled and prepared with it all the same",
        path.display(),
        ids.join(", ")
    )
}

/// What checking a document found: the lines to report, in order.
#[derive(Debug, Default)]
pub struct Checked {
    /// Each line, and whether it is a problem.
    pub lines: Vec<(bool, String)>,
}

impl Checked {
    /// How many problems were found.
    pub fn problems(&self) -> usize {
        self.lines.iter().filter(|(problem, _)| *problem).count()
    }
}

/// What the language finds wrong in `documents` (those of `root`, a
/// file or a directory, each checked as [`check_document`] checks it), a
/// line a problem: `file:line:column: message`, the file relative to
/// `root` (to its directory when it is a file), a problem with no place at
/// line and column 0; in order of file and place, each once. A newline
/// alone when there is none, as a Field holds one character at least.
pub(crate) fn problems_text(
    store: &dyn Store,
    root: &Path,
    documents: &[std::path::PathBuf],
) -> String {
    let base = if store.kind(root) == Some(crate::store::Kind::File) {
        root.parent().unwrap_or(root)
    } else {
        root
    };
    // Compared by their one name: a document a `use` reaches is said by
    // it, a link read through.
    let named = |path: &Path| store.canonical(path).unwrap_or_else(|_| store.whole(path));
    let base = named(base);
    let mut found = std::collections::BTreeSet::new();
    // The documents are checked against one file system: what each used
    // one was found to be, by whichever used it first, holds for all.
    let pass = Pass::default();
    for document in documents {
        let checked = check_in(
            store,
            document,
            false,
            &crate::overrides::Overrides::default(),
            Some(&pass),
        );
        for (problem, line) in checked.lines {
            if !problem {
                continue;
            }
            let (file, at, message) = placed(&line);
            // A document that fails where it is used is said after its
            // user's name: `user.ent:/path/used.ent`.
            let file = file
                .rsplit_once(':')
                .map(|(_, used)| used)
                .filter(|used| store.exists(Path::new(used)))
                .unwrap_or(&file);
            let file = named(Path::new(file));
            let file = file
                .strip_prefix(&base)
                .unwrap_or(&file)
                .display()
                .to_string();
            found.insert((file, at.0, at.1, message));
        }
    }
    if found.is_empty() {
        return "\n".to_owned();
    }
    found
        .into_iter()
        .map(|(file, line, column, message)| format!("{file}:{line}:{column}: {message}\n"))
        .collect()
}

/// What `documents` define (those of `root`, a file or a directory), a
/// line a definition: `file:line:column: kind name: signature`
/// ([`entelechy_language::definitions`]), the file relative to `root`
/// (to its directory when it is a file), `: signature` left out when
/// there is none; in order of file and place. A document that does not
/// parse defines nothing. A newline alone when there is nothing, as a
/// Field holds one character at least.
pub(crate) fn outline_text(
    store: &dyn Store,
    root: &Path,
    documents: &[std::path::PathBuf],
) -> String {
    let base = if store.kind(root) == Some(crate::store::Kind::File) {
        root.parent().unwrap_or(root)
    } else {
        root
    };
    let named = |path: &Path| store.canonical(path).unwrap_or_else(|_| store.whole(path));
    let base = named(base);
    let mut text = String::new();
    for document in documents {
        let file = named(document);
        let file = file
            .strip_prefix(&base)
            .unwrap_or(&file)
            .display()
            .to_string();
        // Parsed once per stamp: a document unchanged is not parsed again.
        let Ok(parsed) = store.document(document) else {
            continue;
        };
        for d in entelechy_language::definitions(&parsed) {
            let signature = if d.signature.is_empty() {
                String::new()
            } else {
                format!(": {}", d.signature)
            };
            text.push_str(&format!(
                "{file}:{}:{}: {} {}{signature}\n",
                d.span.line, d.span.column, d.kind, d.name
            ));
        }
    }
    if text.is_empty() {
        return "\n".to_owned();
    }
    text
}

/// A problem's line as [`check_document`] says it, taken apart: its
/// file, its line and column (0 and 0 when it has no place), its message.
fn placed(line: &str) -> (String, (u32, u32), String) {
    let place = |rest: &str| -> Option<(u32, u32, String)> {
        let (line, rest) = rest.split_once(':')?;
        let (column, message) = rest.split_once(": ")?;
        Some((line.parse().ok()?, column.parse().ok()?, message.to_owned()))
    };
    let colons = line.match_indices(':').map(|(at, _)| at);
    for at in colons.clone() {
        if let Some((l, c, message)) = place(&line[at + 1..]) {
            return (line[..at].to_owned(), (l, c), message);
        }
    }
    for at in colons {
        if let Some(message) = line[at + 1..].strip_prefix(' ') {
            return (line[..at].to_owned(), (0, 0), message.to_owned());
        }
    }
    (String::new(), (0, 0), line.to_owned())
}

/// Checks that the document at `path` parses, lowers with what it
/// reaches, compiles with the documents it uses, binds each node to its
/// Kernel's ports ([`entelechy_graph::binding_fault`]), and orders its graph.
/// `show_ir` also reports each Kernel's accounted IR.
pub fn check_document(store: &dyn Store, path: &Path, show_ir: bool) -> Checked {
    check_document_with(
        store,
        path,
        show_ir,
        &crate::overrides::Overrides::default(),
    )
}

/// [`check_document`], the scene's inputs and outputs placed as
/// `overrides` say (`--input`, `--output`): a name it does not read or
/// write from a file is a problem.
pub fn check_document_with(
    store: &dyn Store,
    path: &Path,
    show_ir: bool,
    overrides: &crate::overrides::Overrides,
) -> Checked {
    check_in(store, path, show_ir, overrides, None)
}

/// What a check of many documents at once ([`problems_text`]) found of
/// each document they use, by the document and whether its Kernels were
/// published: the file system holds still while it runs, so a document
/// used by many is looked at once.
#[derive(Default)]
struct Pass {
    used: std::sync::Mutex<std::collections::HashMap<(std::path::PathBuf, bool), UsedChecked>>,
}

/// [`check_document_with`], within `pass` when it is one of many.
fn check_in(
    store: &dyn Store,
    path: &Path,
    show_ir: bool,
    overrides: &crate::overrides::Overrides,
    pass: Option<&Pass>,
) -> Checked {
    let mut checked = Checked::default();
    let file = path.display().to_string();
    let file = file.as_str();
    macro_rules! say {
        // One fault in one place is one problem, however many Kernels
        // meet it.
        (problem, $($arg:tt)*) => {{
            let line = format!($($arg)*);
            if !checked.lines.iter().any(|(problem, had)| *problem && *had == line) {
                checked.lines.push((true, line));
            }
        }};
        (note, $($arg:tt)*) => { checked.lines.push((false, format!($($arg)*))) };
    }
    let text = match store.read_text(path) {
        Ok(text) => text,
        Err(error) => {
            say!(problem, "{file}: {error}");
            return checked;
        }
    };
    let document = match entelechy_language::parse_document(&text) {
        Ok(document) => document,
        Err(error) => {
            say!(problem, "{file}:{error}");
            return checked;
        }
    };
    let name = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("scene");
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut loader = |path: &Path| locate_document(store, path);
    let mut declare =
        |base: &Path, uses: &[entelechy_language::Use], own: entelechy_language::Own<'_>| {
            declarations_for(store, base, uses, own)
        };
    let lowered =
        match entelechy_scene::load_with(&document, &text, name, base, &mut loader, &mut declare) {
            Ok(lowered) => lowered,
            Err(error) => {
                say!(problem, "{file}:{error}");
                return checked;
            }
        };
    if let Err(error) = overrides.apply(&mut lowered.files.clone(), &lowered.graph.externals) {
        say!(problem, "{file}: {error}");
    }
    let mut document = document;
    if let Ok(((structs, types, functions), reach)) = crate::sources::declarations_keyed(
        store,
        None,
        base,
        &document.uses,
        entelechy_language::Own::of(&document),
    ) {
        for error in functions_checked(
            store,
            path,
            reach.as_deref(),
            &document,
            (&structs, &types, &functions),
        ) {
            let held = held((&structs, &types, &functions));
            let at = written_in(path, held, &error);
            let said = located(store, said(held, &error), &at, path, &[]);
            say!(problem, "{}:{said}", at.display());
        }
        document.resolve((&structs, &types, &functions));
    }
    // Each Kernel compiled, its manifest and structs, which the nodes
    // binding it are held to.
    let mut manifests: Vec<Compiled> = Vec::new();
    let mut kernels: Vec<(Option<entelechy_ir::KernelId>, &entelechy_language::Kernel)> = document
        .kernels
        .iter()
        .map(|kernel| (None, kernel))
        .collect();
    kernels.extend(
        lowered
            .inline
            .iter()
            .map(|inline| (Some(inline.id.clone()), &inline.kernel)),
    );
    for (id, kernel) in kernels {
        // The document's own Kernels as a session compiles them; an inline
        // one, which no session reads from a document, as it is.
        let module = match &id {
            Some(id) => entelechy_compiler::compile_kernel(kernel, Some(id)),
            None => crate::kernels::compiled_in(store, path, &text, kernel),
        };
        match module {
            Ok(module) => {
                manifests.push((
                    module.manifest.clone(),
                    module.structs.clone(),
                    first_counted_against(&module),
                ));
                if let Some(large) = large(&module) {
                    say!(note, "{file}: {large}");
                }
                say!(
                    note,
                    "{file}: {}\n{}",
                    module.manifest.id,
                    module
                        .manifest
                        .named(&module.structs)
                        .to_string()
                        .trim_end()
                );
                // The accounted module, as a backend runs it.
                if show_ir {
                    match entelechy_ir::account(&module) {
                        Ok(run) => say!(note, "{run}"),
                        Err(error) => say!(note, "{file}: accounting: {error}"),
                    }
                }
            }
            Err(error) => {
                let held = entelechy_language::Held::of(kernel);
                let at = written_in(path, held, &error);
                let said = located(store, said(held, &error), &at, path, &[]);
                say!(problem, "{}:{said}", at.display());
            }
        }
    }
    let root = path;
    let own_compiled = manifests.len();
    // The documents whose Kernels the scene publishes, as a session
    // publishes them: those its `use` items name. Of a document those use
    // in turn, the declarations are checked, which Kernels inherit, and
    // its Kernels are not the scene's.
    let published = crate::sources::Expanded::of(store, lowered.uses.clone()).files;
    let publishes = |path: &Path| {
        let path = store.canonical(path).unwrap_or_else(|_| path.to_path_buf());
        published
            .iter()
            .any(|file| store.canonical(file).unwrap_or_else(|_| file.clone()) == path)
    };
    // A Kernel id is one Kernel: one the scene publishes from two
    // documents, a session publishes from the first and refuses from the
    // second.
    let canonical = |path: &Path| store.canonical(path).unwrap_or_else(|_| path.to_path_buf());
    let mut owners: Vec<(entelechy_ir::KernelId, std::path::PathBuf)> = document
        .kernels
        .iter()
        .filter_map(|kernel| crate::sources::named_id(kernel).ok())
        .map(|id| (id, canonical(path)))
        .collect();
    for file in &published {
        let Ok(parsed) = store.document(file) else {
            continue;
        };
        let file = canonical(file);
        for id in parsed
            .kernels
            .iter()
            .filter_map(|kernel| crate::sources::named_id(kernel).ok())
        {
            match owners.iter().find(|(owned, _)| *owned == id) {
                Some((_, owner)) if *owner != file => say!(
                    problem,
                    "{}: {id} is already published by {}; kernel ids are unique",
                    file.display(),
                    owner.display()
                ),
                Some(_) => {}
                None => owners.push((id, file.clone())),
            }
        }
    }
    // The used documents, each with its own declarations.
    match used_documents(store, base, &document.uses) {
        Ok(used) => {
            // Each used document checked on its own, all at once; what
            // each found is said in their order.
            let found = crate::spread::map(&used, |used_path| {
                let published = publishes(used_path);
                let Some(pass) = pass else {
                    return check_used(store, used_path, root, published, show_ir);
                };
                let key = (used_path.to_path_buf(), published);
                let kept = pass
                    .used
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get(&key)
                    .cloned();
                kept.unwrap_or_else(|| {
                    let found = check_used(store, used_path, root, published, show_ir);
                    pass.used
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .insert(key, found.clone());
                    found
                })
            });
            for (lines, modules) in found {
                manifests.extend(modules);
                for (problem, line) in lines {
                    if problem {
                        say!(problem, "{line}");
                    } else {
                        say!(note, "{line}");
                    }
                }
            }
            if !used.is_empty() {
                say!(
                    note,
                    "{file}: {} kernel(s) compiled from {} used document(s)",
                    manifests.len() - own_compiled,
                    used.len()
                );
            }
        }
        Err(error) => {
            say!(problem, "{file}: {error}");
        }
    }
    // Each node's Kernel published by the scene, as a session publishes
    // it: its own, its inline ones, those of the documents it uses.
    let own: Vec<entelechy_ir::KernelId> = document
        .kernels
        .iter()
        .filter_map(|kernel| crate::sources::named_id(kernel).ok())
        .chain(lowered.inline.iter().map(|inline| inline.id.clone()))
        .collect();
    let unpublished = crate::sources::unpublished(store, path, &lowered.graph, &lowered.uses, &own);
    // Each node held to its Kernel's ports, as an evaluation binds it: a
    // Kernel the scene publishes, or one of the backend's natives.
    let natives: Vec<Compiled> = entelechy_kernel::NativeOp::ALL
        .iter()
        .map(|op| (op.manifest(), op.structs(), false))
        .collect();
    // What a source is, as far as a check can know it: a node's output by
    // its Kernel's port, an input read from a file by the shape it is
    // declared.
    let bound = |source: &entelechy_graph::Source| -> Option<entelechy_graph::Bound> {
        match source {
            entelechy_graph::Source::Node { node, port } => {
                let producer = lowered.graph.node(node)?;
                let (manifest, _, _) = manifests
                    .iter()
                    .chain(&natives)
                    .find(|(m, _, _)| m.id == producer.kernel)?;
                let output = manifest.outputs.iter().find(|o| o.name == *port)?;
                Some((output.ty.shape, Some(output.ty.schema.clone())))
            }
            entelechy_graph::Source::External(name) => {
                // What a check finds is text, as bytes are, and a
                // directory's files, listed or read, are words.
                if lowered
                    .files
                    .checks
                    .iter()
                    .chain(&lowered.files.files)
                    .chain(&lowered.files.outlines)
                    .any(|(checked, _)| checked == name)
                    || lowered
                        .files
                        .file_nodes
                        .iter()
                        .flat_map(|node| &node.outputs)
                        .any(|(_, external)| external == name)
                {
                    return Some((
                        entelechy_ir::Shape::Rows,
                        entelechy_ir::schema_of(entelechy_ir::Type::Int, &[]),
                    ));
                }
                let asset = lowered.files.assets.iter().find(|a| a.name == *name)?;
                let ty = asset.ty.trim();
                let shape = if asset.bytes || ty.starts_with("Rows<") {
                    entelechy_ir::Shape::Rows
                } else if ty.starts_with("Grid<") || ty == "Image" {
                    entelechy_ir::Shape::Grid
                } else if ty.starts_with("Volume<") {
                    entelechy_ir::Shape::Volume
                } else {
                    return None;
                };
                // A layout too where the element is one of the language's
                // own, kept as its words: a struct's is its document's.
                let element = if asset.bytes {
                    "Int"
                } else if ty == "Image" {
                    "Vec4"
                } else {
                    ty.split_once('<')
                        .and_then(|(_, rest)| rest.strip_suffix('>'))
                        .unwrap_or("")
                        .trim()
                };
                let schema = element
                    .parse::<entelechy_ir::Type>()
                    .ok()
                    .filter(|_| !element.contains(" as "))
                    .and_then(|element| entelechy_ir::schema_of(element, &[]));
                Some((shape, schema))
            }
        }
    };
    for node in &lowered.graph.nodes {
        if unpublished.iter().any(|(id, _, _)| id == &node.id) {
            continue;
        }
        let Some((manifest, structs, counted)) = manifests
            .iter()
            .chain(&natives)
            .find(|(m, _, _)| m.id == node.kernel)
        else {
            continue;
        };
        if let Err(error) = entelechy_graph::binding_fault(&lowered.graph, node, manifest, structs)
        {
            let at = node.at.clone().unwrap_or_else(|| file.to_owned());
            say!(problem, "{at}: {error}");
        } else if let Err(error) =
            entelechy_graph::type_fault(&lowered.graph, node, manifest, structs, &bound)
        {
            let at = node.at.clone().unwrap_or_else(|| file.to_owned());
            say!(problem, "{at}: {error}");
        }
        if let Some(why) = defaulted_domain(node, manifest, *counted) {
            let at = node.at.clone().unwrap_or_else(|| file.to_owned());
            say!(note, "{at}: {why}");
        }
        if let Some((port, output, first)) =
            entelechy_graph::borrowed_domain(&lowered.graph, node, manifest)
        {
            let at = node.at.clone().unwrap_or_else(|| file.to_owned());
            say!(
                note,
                "{at}: node `{}` carries its own `{output}` back through `{port}` and has no `domain`: it runs over its first input `{first}`, so its state takes that input's size; write its `domain`, or make `{port}` its first input",
                node.id
            );
        }
    }
    // A frame runs what something reads: a node nothing reads (no output,
    // state, input's fallback while its file is absent, font node's codes
    // or files node's paths) never runs.
    let fallbacks = lowered
        .graph
        .fallbacks
        .iter()
        .map(|(_, source)| source)
        .chain(lowered.files.fonts.iter().map(|font| &font.codes))
        .chain(lowered.files.file_nodes.iter().map(|node| &node.paths))
        .filter_map(|source| match source {
            entelechy_graph::Source::Node { node, .. } => Some(node.clone()),
            entelechy_graph::Source::External(_) => None,
        });
    let read = lowered.graph.read_through(fallbacks);
    // Said once a chain, at its top: a node only an unread one reads goes
    // with it.
    let read_at_all: std::collections::BTreeSet<&entelechy_graph::NodeId> = lowered
        .graph
        .nodes
        .iter()
        .flat_map(|node| node.inputs.values())
        .filter_map(|binding| match binding {
            entelechy_graph::Binding::Now(entelechy_graph::Source::Node { node, .. }) => Some(node),
            _ => None,
        })
        .collect();
    for node in lowered
        .graph
        .nodes
        .iter()
        .filter(|node| !read.contains(&node.id) && !read_at_all.contains(&node.id))
    {
        let at = node.at.clone().unwrap_or_else(|| file.to_owned());
        say!(
            note,
            "{at}: nothing reads node `{}` (no output or state reaches it), so a frame does not run it, nor what only it reads; a replay or a reader that pins it runs it",
            node.id
        );
    }
    for (node, kernel, why) in unpublished {
        let at = lowered
            .graph
            .node(&node)
            .and_then(|n| n.at.clone())
            .unwrap_or_else(|| file.to_owned());
        match why {
            Some(why) => say!(
                problem,
                "{at}: node `{node}` binds `{kernel}`, which the scene does not publish; {why}"
            ),
            None => say!(
                problem,
                "{at}: node `{node}` binds `{kernel}`, which nothing the scene uses publishes"
            ),
        }
    }
    // A Kernel the scene publishes and no node uses is compiled and
    // prepared with it all the same: the scene's own, and those of every
    // document it publishes, its placed graphs' included.
    if !document.nodes.is_empty() {
        let own: Vec<entelechy_ir::KernelId> = document
            .kernels
            .iter()
            .filter_map(|kernel| crate::sources::named_id(kernel).ok())
            .collect();
        // A file reached by more than one spelling of its path, once.
        let mut files: Vec<std::path::PathBuf> = published
            .iter()
            .map(|file| store.canonical(file).unwrap_or_else(|_| file.clone()))
            .collect();
        files.sort();
        files.dedup();
        let theirs = files.iter().map(|file| {
            let ids = read_document(store, file)
                .ok()
                .and_then(|text| entelechy_language::parse_document(&text).ok())
                .map(|used| {
                    used.kernels
                        .iter()
                        .filter_map(|kernel| crate::sources::named_id(kernel).ok())
                        .collect()
                })
                .unwrap_or_default();
            (file.clone(), ids)
        });
        let scene = std::iter::once((path.to_path_buf(), own)).chain(theirs);
        for (at, ids) in idle(store, &lowered.graph, scene) {
            say!(note, "{}", idle_said(&at, &ids));
        }
    }
    if !document.nodes.is_empty() {
        match lowered.graph.order() {
            Ok(order) => say!(
                note,
                "{file}: graph `{}` with {} node(s), {} use(s)",
                lowered.graph.name,
                order.len(),
                document.uses.len()
            ),
            Err(error) => match crate::scene::graph_diagnostic(&document, &error) {
                placed if placed.at.is_some() => say!(problem, "{file}:{placed}"),
                unplaced => say!(problem, "{file}: {unplaced}"),
            },
        }
    }
    checked
}

/// What checking one document a scene uses found, in order, and the
/// Kernels that compiled, each its manifest and structs: its declarations, and its Kernels when the
/// scene publishes them (`published`).
/// A compiled Kernel's manifest, the structs its ports name, and whether
/// it weighs its element against its first input's count
/// ([`first_counted_against`]).
pub(crate) type Compiled = (entelechy_ir::Manifest, Vec<entelechy_ir::StructDef>, bool);
/// What checking a used document found ([`check_used`]): its lines, and its
/// Kernels compiled.
pub(crate) type UsedChecked = (Vec<(bool, String)>, Vec<Compiled>);

/// Whether `module` compares its element with its first input's count
/// (`element() < count(first)`, as a join of two rows does): a test that,
/// run over that input (a node with no `domain`), is always decided the
/// same way.
pub(crate) fn first_counted_against(module: &entelechy_ir::Module) -> bool {
    use entelechy_ir::{BinaryOp, Op};
    fn walk(
        body: &[entelechy_ir::Instruction],
        elements: &mut Vec<entelechy_ir::ValueId>,
        counts: &mut Vec<entelechy_ir::ValueId>,
        tests: &mut Vec<(entelechy_ir::ValueId, entelechy_ir::ValueId)>,
    ) {
        for instruction in body {
            match &instruction.op {
                Op::Element => elements.push(instruction.id),
                Op::Count(0) => counts.push(instruction.id),
                Op::Binary(
                    BinaryOp::Less
                    | BinaryOp::LessEqual
                    | BinaryOp::Greater
                    | BinaryOp::GreaterEqual
                    | BinaryOp::Equal
                    | BinaryOp::NotEqual,
                    a,
                    b,
                ) => tests.push((*a, *b)),
                op => {
                    for block in op.blocks() {
                        walk(&block.body, elements, counts, tests);
                    }
                }
            }
        }
    }
    if module.manifest.inputs.is_empty() {
        return false;
    }
    let (mut elements, mut counts, mut tests) = (Vec::new(), Vec::new(), Vec::new());
    walk(&module.entry.body, &mut elements, &mut counts, &mut tests);
    tests.iter().any(|(a, b)| {
        (elements.contains(a) && counts.contains(b)) || (elements.contains(b) && counts.contains(a))
    })
}

/// What to say of a node with no `domain`, which runs over its first input,
/// when its Kernel compares its element with that input's count
/// (`counted`): the elements past it, which such a test is for, are never
/// run, as a join's second input would not be.
pub(crate) fn defaulted_domain(
    node: &entelechy_graph::Node,
    manifest: &entelechy_ir::Manifest,
    counted: bool,
) -> Option<String> {
    if node.domain.is_some() || !counted || manifest.mode == entelechy_ir::Mode::Reduce {
        return None;
    }
    let first = &manifest.inputs.first()?.name;
    Some(format!(
        "node `{}` has no `domain`, so it runs over its first input `{first}`, one element each; its Kernel compares `element()` with `count({first})`, which over that domain is decided the same way every element, so elements past `{first}`'s are never run; write its `domain`",
        node.id
    ))
}

/// What the compiler finds wrong in `document`'s functions (the one at
/// `path`), its inherited declarations given: kept by the document's
/// stamp and its reach's key (every document it reaches with its stamp,
/// [`crate::sources::declarations_keyed`]) where the store keeps what it
/// works out, so a check of many documents using one checks its functions
/// once, and a document unchanged since the last check, with all it
/// reaches, is not checked again.
fn functions_checked(
    store: &dyn Store,
    path: &Path,
    reach: Option<&str>,
    document: &entelechy_language::Document,
    inherited: (
        &[entelechy_language::StructDecl],
        &[entelechy_language::TypeAlias],
        &[entelechy_language::FnDecl],
    ),
) -> Vec<entelechy_compiler::CompileError> {
    let key = store
        .stamp(path)
        .zip(reach)
        .map(|(stamp, reach)| format!("{}@{stamp:?}|{reach}", path.display()));
    let kept = |parsed: &crate::store::Parsed, key: &String| {
        parsed
            .functions_checked
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(key)
            .cloned()
    };
    if let (Some(parsed), Some(key)) = (store.parsed(), &key)
        && let Some(errors) = kept(parsed, key)
    {
        return errors;
    }
    let errors = entelechy_compiler::check_functions_each(document, inherited);
    if let (Some(parsed), Some(key)) = (store.parsed(), key) {
        let mut checked = parsed
            .functions_checked
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // A long session's edits leave keys behind; they go when there
        // are many.
        if checked.len() >= 4096 {
            checked.clear();
        }
        checked.insert(key, errors.clone());
    }
    errors
}

/// What checking `path`, a document a checked document uses, finds: what
/// the compiler finds wrong in its functions and, when `published`, its
/// Kernels, said from `root`. Kept where the store keeps what it works out,
/// by the document, its stamp, its reach's key and how it is said, so a
/// document used by many is checked once, and again only when it or what
/// it reaches changes.
fn check_used(
    store: &dyn Store,
    path: &Path,
    root: &Path,
    published: bool,
    show_ir: bool,
) -> UsedChecked {
    let mut lines = Vec::new();
    let mut compiled = Vec::new();
    let text = match store.read_text(path) {
        Ok(text) => text,
        Err(error) => {
            lines.push((true, format!("{}: {error}", path.display())));
            return (lines, compiled);
        }
    };
    // Parsed once per stamp, however many documents use it.
    let Ok(parsed) = store.document(path) else {
        return (lines, compiled); // reported by `used_documents`
    };
    let mut used_document = (*parsed).clone();
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut key = None;
    match crate::sources::declarations_keyed(
        store,
        Some(path),
        dir,
        &used_document.uses,
        entelechy_language::Own::of(&used_document),
    ) {
        Ok(((structs, types, functions), reach)) => {
            key = store
                .stamp(path)
                .zip(reach.as_deref())
                .map(|(stamp, reach)| {
                    format!(
                        "{}@{stamp:?}|{reach}|{}|{published}|{show_ir}",
                        path.display(),
                        root.display()
                    )
                });
            if let (Some(parsed), Some(key)) = (store.parsed(), &key)
                && let Some(found) = parsed
                    .used_checked
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get(key)
            {
                return found.clone();
            }
            for error in functions_checked(
                store,
                path,
                reach.as_deref(),
                &used_document,
                (&structs, &types, &functions),
            ) {
                let held = held((&structs, &types, &functions));
                let at = written_in(path, held, &error);
                let said = located(store, said(held, &error), &at, root, &[]);
                lines.push((true, format!("{}:{said}", at.display())));
            }
            used_document.resolve((&structs, &types, &functions))
        }
        Err(error) => lines.push((true, format!("{}: {error}", path.display()))),
    }
    let kernels: &[entelechy_language::Kernel] = if published {
        &used_document.kernels
    } else {
        &[]
    };
    for kernel in kernels {
        match crate::kernels::compiled_in(store, path, &text, kernel) {
            Ok(module) => {
                compiled.push((
                    module.manifest.clone(),
                    module.structs.clone(),
                    first_counted_against(&module),
                ));
                if let Some(large) = large(&module) {
                    lines.push((false, format!("{}: {large}", path.display())));
                }
                if show_ir {
                    lines.push((
                        false,
                        format!("{}: {}\n{module}", path.display(), module.manifest.id),
                    ));
                }
            }
            Err(error) => {
                let held = entelechy_language::Held::of(kernel);
                let at = written_in(path, held, &error);
                let said = located(store, said(held, &error), &at, root, &[]);
                lines.push((true, format!("{}:{said}", at.display())));
            }
        }
    }
    if let (Some(parsed), Some(key)) = (store.parsed(), key) {
        let mut kept = parsed
            .used_checked
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // A long session's edits leave keys behind; they go when there
        // are many.
        if kept.len() >= 4096 {
            kept.clear();
        }
        kept.insert(key, (lines.clone(), compiled.clone()));
    }
    (lines, compiled)
}
