//! What a scene document may say is a contract with every scene already
//! written. `tests/surface/` holds documents that between them write
//! every form a scene has — wiring, values, state, inline Kernels,
//! extents, Field nodes, native operations, placed graphs, repetition,
//! inputs the Host reads — and beside each, the Graph it lowers to at
//! `LANGUAGE_VERSION`.
//!
//! A document that stops lowering, or lowers to something else, breaks
//! scenes already written, and waits for a new `LANGUAGE_VERSION`. Every
//! Kernel a document writes in place, a filled Field's included, must
//! compile: a lowering that does not was never a scene. A new
//! document is the surface growing. `ORLA_BLESS=1` records what is new
//! and refuses to record a change to what was recorded until the version
//! is raised. `parts/` holds documents the others place, which are not
//! lowered alone.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use orla_language::LANGUAGE_VERSION;

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

/// The document lowered and written out: the Graph as it prints, then
/// what printing leaves out — the files the Host reads, the frames it
/// steps, the parameters that size the graph, the Kernels written in
/// place, and what the document uses.
fn lowered(path: &Path) -> Result<String, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let document = orla_language::parse_document(&text).map_err(|e| e.to_string())?;
    let base = path.parent().unwrap().to_path_buf();
    let name = path.file_stem().unwrap().to_string_lossy().into_owned();
    let mut loader = |path: &Path| std::fs::read_to_string(path).map_err(|e| e.to_string());
    let mut declare = |_: &Path, _: &[orla_language::Use]| Ok((Vec::new(), Vec::new(), Vec::new()));
    let lowered = orla_scene::load_with_params(
        &document,
        &text,
        &name,
        &base,
        &mut loader,
        &mut declare,
        &BTreeMap::new(),
    )
    .map_err(|e| e.to_string())?;
    // A lowering is only a scene if what it wrote compiles: every Kernel
    // written in place, a filled Field's among them.
    for inline in &lowered.inline {
        orla_compiler::compile_kernel(&inline.kernel, Some(&inline.id))
            .map_err(|e| format!("the Kernel `{}` it writes does not compile: {e}", inline.id))?;
    }
    let graph = &lowered.graph;
    let mut out = graph.to_string();
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
    let bless = std::env::var_os("ORLA_BLESS").is_some();
    let mut documents: Vec<PathBuf> = std::fs::read_dir(surface())
        .unwrap()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "osn"))
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
        "scenes written for language {LANGUAGE_VERSION} would change meaning or stop loading:\n{}\nraise LANGUAGE_VERSION, then record with `ORLA_BLESS=1 cargo test -p orla-scene --test surface`",
        broken.join("\n")
    );
    assert!(
        new.is_empty() || bless,
        "new scene forms, not yet recorded: {new:?}; record them with \
         `ORLA_BLESS=1 cargo test -p orla-scene --test surface`"
    );
}
