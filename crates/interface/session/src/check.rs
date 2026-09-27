//! `orla check`: a document checked through a store, without a session.

use std::path::Path;

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
        (problem, $($arg:tt)*) => { checked.lines.push((true, format!($($arg)*))) };
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
    let mut declare =
        |base: &Path, uses: &[orla_language::Use]| declarations_for(store, base, uses);
    let lowered =
        match orla_scene::load_with(&document, &text, name, base, &mut loader, &mut declare) {
            Ok(lowered) => lowered,
            Err(error) => {
                say!(problem, "{file}:{error}");
                return checked;
            }
        };
    let mut document = document;
    if let Ok((structs, types, functions)) = declarations_for(store, base, &document.uses) {
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
                say!(problem, "{file}:{error}");
            }
        }
    }
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
                match declarations_within(store, Some(path), dir, &used_document.uses) {
                    Ok((structs, types, functions)) => {
                        used_document.inherit(&structs, &types, &functions)
                    }
                    Err(error) => {
                        say!(problem, "{}: {error}", path.display());
                    }
                }
                for kernel in &used_document.kernels {
                    match orla_compiler::compile_kernel(kernel, None) {
                        Ok(module) => {
                            compiled += 1;
                            if show_ir {
                                say!(note, "{}: {}\n{module}", path.display(), module.manifest.id);
                            }
                        }
                        Err(error) => {
                            say!(problem, "{}:{error}", path.display());
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
