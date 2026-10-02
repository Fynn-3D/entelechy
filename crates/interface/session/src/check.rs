//! `orla check`: a document checked through a store, without a session.

use std::path::Path;

use crate::sources::{held, located, said, written_in};
use crate::{Store, declarations_for, declarations_within, read_document, used_documents};

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
        document.inherit(&structs, &types, &functions);
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
            let mut compiled = 0;
            for path in &used {
                let text = match store.read_text(path) {
                    Ok(text) => text,
                    Err(error) => {
                        say!(problem, "{}: {error}", path.display());
                        continue;
                    }
                };
                let Ok(mut used_document) = orla_language::parse_document(&text) else {
                    continue; // reported by `used_documents`
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
                        if let Err(error) = orla_compiler::check_functions(
                            &used_document,
                            (&structs, &types, &functions),
                        ) {
                            let held = held((&structs, &types, &functions));
                            let at = written_in(path, held, &error);
                            say!(problem, "{}:{}", at.display(), said(held, &error));
                        }
                        used_document.inherit(&structs, &types, &functions)
                    }
                    Err(error) => {
                        say!(problem, "{}: {error}", path.display());
                    }
                }
                let kernels: &[orla_language::Kernel] = if publishes(path) {
                    &used_document.kernels
                } else {
                    &[]
                };
                for kernel in kernels {
                    match orla_compiler::compile_kernel(kernel, None) {
                        Ok(module) => {
                            compiled += 1;
                            if show_ir {
                                say!(note, "{}: {}\n{module}", path.display(), module.manifest.id);
                            }
                        }
                        Err(error) => {
                            let held = orla_language::Held::of(kernel);
                            let at = written_in(path, held, &error);
                            let said = located(store, said(held, &error), path, root, &[]);
                            say!(problem, "{}:{said}", at.display());
                        }
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
