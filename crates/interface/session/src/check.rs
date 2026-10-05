//! `orla check`: a document checked through a store, without a session.

use std::path::Path;

use crate::sources::{held, located, said, written_in};
use crate::{Store, declarations_for, declarations_within, read_document, used_documents};

/// The instructions past which a Kernel is said to be large: a device's
/// compiler may take long over it, or fail on it.
pub const LARGE: usize = 25_000;

/// What makes a Kernel large, when it is ([`LARGE`]): its instructions, and
/// its largest functions, each with the calls that reach it, as the
/// largest are most often reached by a call not meant to bring them.
pub fn large(module: &orla_ir::Validated) -> Option<String> {
    large_past(module, LARGE)
}

/// [`large`], past `limit` instructions.
pub fn large_past(module: &orla_ir::Validated, limit: usize) -> Option<String> {
    let sizes = orla_ir::sizes(module);
    let total = sizes.total();
    if total <= limit {
        return None;
    }
    let mut order: Vec<usize> = (0..sizes.functions.len()).collect();
    order.sort_by_key(|&f| std::cmp::Reverse(sizes.functions[f]));
    let name = |f: u32| module.functions[f as usize].name.clone();
    let largest: Vec<String> = order
        .into_iter()
        .take(3)
        .map(|f| {
            let way: Vec<String> = sizes.way_to(f as u32).into_iter().map(name).collect();
            let by = if way.len() <= 1 {
                "called by the Kernel itself".to_owned()
            } else {
                format!("by {}", way.join(" → "))
            };
            format!("`{}` ({}, {by})", name(f as u32), sizes.functions[f])
        })
        .collect();
    Some(format!(
        "`{}` is {total} instructions, past the {limit} a device's compiler takes in comfort; its largest functions: {}",
        module.manifest.id,
        largest.join(", ")
    ))
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
/// reaches, compiles with the documents it uses, and orders its graph.
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
            say!(problem, "{}:{}", at.display(), said(held, &error));
        }
        document.resolve((&structs, &types, &functions));
    }
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
                let said = located(store, said(held, &error), path, path, &[]);
                say!(problem, "{}:{said}", at.display());
            }
        }
    }
    let root = path;
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
            let mut compiled = 0;
            for (lines, count) in found {
                compiled += count;
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
                    "{file}: {compiled} kernel(s) compiled from {} used document(s)",
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
    for (node, kernel, why) in
        crate::sources::unpublished(store, path, &lowered.graph, &lowered.uses, &own)
    {
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

/// What checking one document a scene uses found, in order, and how many
/// of its Kernels compiled: its declarations, and its Kernels when the
/// scene publishes them (`published`).
fn check_used(
    store: &dyn Store,
    path: &Path,
    root: &Path,
    published: bool,
    show_ir: bool,
) -> (Vec<(bool, String)>, usize) {
    let mut lines = Vec::new();
    let mut compiled = 0;
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
                lines.push((true, format!("{}:{}", at.display(), said(held, &error))));
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
                compiled += 1;
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
                let said = located(store, said(held, &error), path, root, &[]);
                lines.push((true, format!("{}:{said}", at.display())));
            }
        }
    }
    (lines, compiled)
}
