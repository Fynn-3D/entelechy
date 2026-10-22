//! A scene document's surface is a contract with every written scene.
//! `tests/surface/` holds documents covering every scene form (wiring,
//! values, state, inline Kernels, extents, Field nodes, native operations,
//! placed graphs, repetition, Host inputs and nodes), each beside the Graph it
//! lowers to at `LANGUAGE_VERSION`.
//!
//! A changed lowering waits for a new `LANGUAGE_VERSION`. Every Kernel a
//! document writes in place compiles. `ENTELECHY_BLESS=1` records new
//! documents and refuses to change recorded ones until the version is
//! raised. `parts/` holds documents others place, not lowered alone.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use entelechy_language::LANGUAGE_VERSION;

fn surface() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/surface")
}

/// A path as the document's directory sees it.
fn relative(path: &Path, base: &Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .display()
        .to_string()
}

/// The document lowered and written out: the printed Graph, then its Host
/// files, stepped frames, sizing params, inline Kernels and uses.
fn lowered(path: &Path) -> Result<String, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let document = entelechy_language::parse_document(&text).map_err(|e| e.to_string())?;
    let base = path.parent().unwrap().to_path_buf();
    let name = path.file_stem().unwrap().to_string_lossy().into_owned();
    let mut loader = |path: &Path| {
        std::fs::read_to_string(path)
            .map(|text| (path.to_path_buf(), text))
            .map_err(|e| e.to_string())
    };
    let mut declare = |_: &Path, _: &[entelechy_language::Use], _: entelechy_language::Own<'_>| {
        Ok(entelechy_language::Declarations::default())
    };
    let lowered = entelechy_scene::load_with_params(
        &document,
        &text,
        &name,
        &base,
        &mut loader,
        &mut declare,
        &BTreeMap::new(),
    )
    .map_err(|e| e.to_string())?;
    // Every Kernel written in place compiles, filled Fields' included.
    for inline in &lowered.inline {
        entelechy_compiler::compile_kernel(&inline.kernel, Some(&inline.id))
            .map_err(|e| format!("the Kernel `{}` it writes does not compile: {e}", inline.id))?;
    }
    let graph = &lowered.graph;
    let mut out = entelechy_scene::print(graph);
    for asset in &lowered.files.assets {
        let kind = if asset.bytes { " bytes" } else { "" };
        out.push_str(&format!(
            "// file {} {} {}{kind}\n",
            asset.name,
            relative(&asset.path, &base),
            asset.ty
        ));
    }
    for (name, dir, ty, rate) in &lowered.files.sequences {
        out.push_str(&format!(
            "// frames {name} {} {ty} {rate}\n",
            relative(dir, &base)
        ));
    }
    for (name, path) in &lowered.files.checks {
        out.push_str(&format!("// checks {name} {}\n", relative(path, &base)));
    }
    for (name, path) in &lowered.files.outlines {
        out.push_str(&format!("// outline {name} {}\n", relative(path, &base)));
    }
    for (name, dir) in &lowered.files.files {
        out.push_str(&format!("// directory {name} {}\n", relative(dir, &base)));
    }
    for node in &lowered.files.file_nodes {
        out.push_str(&format!(
            "// files node {} {} paths from {}, read as {}\n",
            node.node,
            relative(&node.dir, &base),
            node.paths,
            node.outputs
                .iter()
                .map(|(_, external)| external.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    for node in &lowered.files.field_nodes {
        out.push_str(&format!(
            "// fields node {} {} {} paths from {}, read as {}\n",
            node.node,
            node.ty,
            relative(&node.dir, &base),
            node.paths,
            node.outputs
                .iter()
                .map(|(_, external)| external.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    for write in &lowered.files.writes {
        let kind = if write.files {
            " files"
        } else if write.bytes {
            " bytes"
        } else {
            ""
        };
        out.push_str(&format!(
            "// writes {} {}{kind}\n",
            write.output,
            relative(&write.path, &base)
        ));
    }
    for param in &graph.times_params {
        out.push_str(&format!("// sizes {param}\n"));
    }
    for inline in &lowered.inline {
        out.push_str(&format!(
            "// inline {} {} from {}\n",
            inline.node,
            inline.id,
            relative(&inline.document, &base)
        ));
        for line in inline.text.lines() {
            out.push_str(&format!("//   {line}\n"));
        }
    }
    for (path, directory) in &lowered.uses {
        let kind = if *directory { " directory" } else { "" };
        out.push_str(&format!("// use {}{kind}\n", relative(path, &base)));
    }
    Ok(out)
}

#[test]
fn every_scene_form_lowers_as_recorded() {
    let bless = std::env::var_os("ENTELECHY_BLESS").is_some();
    let mut documents: Vec<PathBuf> = std::fs::read_dir(surface())
        .unwrap()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "esn"))
        .collect();
    documents.sort();
    assert!(documents.len() >= 6, "found only {documents:?}");
    let mut broken = Vec::new();
    let mut new = Vec::new();
    for document in &documents {
        let record = document.with_extension("lowered");
        let now = lowered(document)
            .map(|graph| format!("language {LANGUAGE_VERSION}\n{graph}"))
            .map_err(|error| format!("{}: {error}", relative(document, &surface())));
        let recorded = std::fs::read_to_string(&record).ok();
        let (version, before) = match &recorded {
            Some(text) => {
                let (head, rest) = text.split_once('\n').unwrap_or((text, ""));
                let version: u32 = head
                    .strip_prefix("language ")
                    .and_then(|n| n.parse().ok())
                    .unwrap_or(0);
                (version, Some(rest.to_owned()))
            }
            None => (0, None),
        };
        match (&now, &before) {
            (Ok(now), Some(before))
                if now.split_once('\n').map(|(_, rest)| rest) == Some(before) =>
            {
                if bless && version != LANGUAGE_VERSION {
                    std::fs::write(&record, now).unwrap();
                }
            }
            (Ok(now), None) => {
                new.push(relative(document, &surface()));
                if bless {
                    std::fs::write(&record, now).unwrap();
                }
            }
            (Ok(now), Some(_)) => {
                if bless && LANGUAGE_VERSION > version {
                    std::fs::write(&record, now).unwrap();
                } else {
                    broken.push(format!(
                        "{} lowers differently",
                        relative(document, &surface())
                    ));
                }
            }
            (Err(error), _) => broken.push(error.clone()),
        }
    }
    assert!(
        broken.is_empty(),
        "scenes written for language {LANGUAGE_VERSION} would change meaning or stop loading:\n{}\nraise LANGUAGE_VERSION, then record with `ENTELECHY_BLESS=1 cargo test -p entelechy-scene --test surface`",
        broken.join("\n")
    );
    assert!(
        new.is_empty() || bless,
        "new scene forms, not yet recorded: {new:?}; record them with \
         `ENTELECHY_BLESS=1 cargo test -p entelechy-scene --test surface`"
    );
}
