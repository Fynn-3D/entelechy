//! `orla check`: a document checked through a store, without a session.

use std::path::Path;

use crate::sources::{held, located, said, written_in};
use crate::{Store, declarations_for, declarations_within, read_document, used_documents};

/// The words a Kernel may carry through its branches before it is said
/// to be large ([`orla_ir::carried`]): a device's compiler takes long over
/// more (seconds, past in proportion), or fails on it.
pub const LARGE: u64 = 5_000;

/// What makes a Kernel large, when it is ([`LARGE`]): the words it carries
/// through its branches, the structs they are, and the functions that
/// carry most, each with the calls that reach it.
pub fn large(module: &orla_ir::Validated) -> Option<String> {
    large_past(module, LARGE)
}

/// [`large`], past `limit` words.
pub fn large_past(module: &orla_ir::Validated, limit: u64) -> Option<String> {
    let carried = orla_ir::carried(module);
    if carried.total <= limit {
        return None;
    }
    let mut by_struct: Vec<usize> = (0..carried.structs.len())
        .filter(|&s| carried.structs[s] > 0)
        .collect();
    let words = |s: usize| orla_ir::words(&module.structs, orla_ir::Type::Struct(s as u32)) as u64;
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
    let sizes = orla_ir::sizes(module);
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
    graph: &orla_graph::Graph,
    published: impl IntoIterator<Item = (std::path::PathBuf, Vec<orla_ir::KernelId>)>,
) -> Vec<(std::path::PathBuf, Vec<orla_ir::KernelId>)> {
    let bound: std::collections::BTreeSet<&orla_ir::KernelId> =
        graph.nodes.iter().map(|node| &node.kernel).collect();
    published
        .into_iter()
        .filter(|(path, _)| crate::sources::package_holding(store, path).is_none())
        .filter_map(|(path, ids)| {
            let ids: Vec<orla_ir::KernelId> =
                ids.into_iter().filter(|id| !bound.contains(id)).collect();
            (!ids.is_empty()).then_some((path, ids))
        })
        .collect()
}

/// How [`idle`] Kernels are said.
pub(crate) fn idle_said(path: &Path, ids: &[orla_ir::KernelId]) -> String {
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

/// Checks that the document at `path` parses, lowers with what it
/// reaches, compiles with the documents it uses, binds each node to its
/// Kernel's ports ([`orla_graph::binding_fault`]), and orders its graph.
/// `show_ir` also reports each Kernel's accounted IR.
pub fn check_document(store: &dyn Store, path: &Path, show_ir: bool) -> Checked {
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
    let document = match orla_language::parse_document(&text) {
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
    let mut loader = |path: &Path| read_document(store, path);
    let mut declare = |base: &Path, uses: &[orla_language::Use], own: orla_language::Own<'_>| {
        declarations_for(store, base, uses, own)
    };
    let lowered =
        match orla_scene::load_with(&document, &text, name, base, &mut loader, &mut declare) {
            Ok(lowered) => lowered,
            Err(error) => {
                say!(problem, "{file}:{error}");
                return checked;
            }
        };
    let mut document = document;
    if let Ok((structs, types, functions)) = declarations_for(
        store,
        base,
        &document.uses,
        orla_language::Own::of(&document),
    ) {
        if let Err(error) =
            orla_compiler::check_functions(&document, (&structs, &types, &functions))
        {
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
    let mut kernels: Vec<(Option<orla_ir::KernelId>, &orla_language::Kernel)> = document
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
        match orla_compiler::compile_kernel(kernel, id.as_ref()) {
            Ok(module) => {
                manifests.push((module.manifest.clone(), module.structs.clone()));
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
                    match orla_ir::account(&module) {
                        Ok(run) => say!(note, "{run}"),
                        Err(error) => say!(note, "{file}: accounting: {error}"),
                    }
                }
            }
            Err(error) => {
                let held = orla_language::Held::of(kernel);
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
    // The used documents, each with its own declarations.
    match used_documents(store, base, &document.uses) {
        Ok(used) => {
            // Each used document checked on its own, all at once; what
            // each found is said in their order.
            let found = crate::spread::map(&used, |used_path| {
                check_used(store, used_path, root, publishes(used_path), show_ir)
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
    let own: Vec<orla_ir::KernelId> = document
        .kernels
        .iter()
        .filter_map(|kernel| crate::sources::named_id(kernel).ok())
        .chain(lowered.inline.iter().map(|inline| inline.id.clone()))
        .collect();
    let unpublished = crate::sources::unpublished(store, path, &lowered.graph, &lowered.uses, &own);
    // Each node held to its Kernel's ports, as an evaluation binds it.
    for node in &lowered.graph.nodes {
        if unpublished.iter().any(|(id, _, _)| id == &node.id) {
            continue;
        }
        let Some((manifest, structs)) = manifests.iter().find(|(m, _)| m.id == node.kernel) else {
            continue;
        };
        if let Err(error) = orla_graph::binding_fault(&lowered.graph, node, manifest, structs) {
            let at = node.at.clone().unwrap_or_else(|| file.to_owned());
            say!(problem, "{at}: {error}");
        }
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
        let own: Vec<orla_ir::KernelId> = document
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
                .and_then(|text| orla_language::parse_document(&text).ok())
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
/// A compiled Kernel's manifest, and the structs its ports name.
type Compiled = (orla_ir::Manifest, Vec<orla_ir::StructDef>);

fn check_used(
    store: &dyn Store,
    path: &Path,
    root: &Path,
    published: bool,
    show_ir: bool,
) -> (Vec<(bool, String)>, Vec<Compiled>) {
    let mut lines = Vec::new();
    let mut compiled = Vec::new();
    let text = match store.read_text(path) {
        Ok(text) => text,
        Err(error) => {
            lines.push((true, format!("{}: {error}", path.display())));
            return (lines, compiled);
        }
    };
    let Ok(mut used_document) = orla_language::parse_document(&text) else {
        return (lines, compiled); // reported by `used_documents`
    };
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    match declarations_within(
        store,
        Some(path),
        dir,
        &used_document.uses,
        orla_language::Own::of(&used_document),
    ) {
        Ok((structs, types, functions)) => {
            if let Err(error) =
                orla_compiler::check_functions(&used_document, (&structs, &types, &functions))
            {
                let held = held((&structs, &types, &functions));
                let at = written_in(path, held, &error);
                let said = located(store, said(held, &error), &at, root, &[]);
                lines.push((true, format!("{}:{said}", at.display())));
            }
            used_document.resolve((&structs, &types, &functions))
        }
        Err(error) => lines.push((true, format!("{}: {error}", path.display()))),
    }
    let kernels: &[orla_language::Kernel] = if published {
        &used_document.kernels
    } else {
        &[]
    };
    for kernel in kernels {
        match orla_compiler::compile_kernel(kernel, None) {
            Ok(module) => {
                compiled.push((module.manifest.clone(), module.structs.clone()));
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
                let held = orla_language::Held::of(kernel);
                let at = written_in(path, held, &error);
                let said = located(store, said(held, &error), &at, root, &[]);
                lines.push((true, format!("{}:{said}", at.display())));
            }
        }
    }
    (lines, compiled)
}
