//! A scene as one page: a single HTML file holding the Player in a page,
//! Entelechy's bundle runtime (`entelechy_single`), and the scene's compiled
//! binary [`Bundle`](entelechy_session::bundle::Bundle). Nothing is fetched; it
//! opens from the disk.
//!
//! The exporter compiles and packs. How the page starts is the web bridge's
//! (`bridges/web/single.html`, a template): an `@@include:<file>@@` is a file
//! beside the template put in as it is, an `@@pack:<file>@@` that file packed
//! (a zlib stream in UTF-8 base122, as everything large in the page is), an
//! `@@slot:<name>@@` what is made here.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use entelechy_formats::base122::base122;
use entelechy_formats::deflate::zlib_tight;
use entelechy_session::{Event, Limits, Session, SessionOptions, Wait};

/// A scene compiled into its binary [`Bundle`](entelechy_session::bundle::Bundle)
/// bytes, with counts for the line the exporter prints.
pub struct Compiled {
    pub bundle: Vec<u8>,
    pub modules: usize,
    pub inputs: usize,
    /// Each font node subsetted by `--chars`: `(node, font_file_bytes, atlas_bytes)`.
    pub fonts: Vec<(String, usize, usize)>,
}

/// Loads `scene` as a front does, checks that nothing in it or its inputs was
/// refused, bakes every static input and font atlas (`chars` keeping those
/// characters alone when given), and encodes the binary bundle. No host path
/// is written into the bundle.
pub fn compile(scene: &Path, chars: Option<&str>) -> Result<Compiled, String> {
    let under = SessionOptions::from_environment().store();
    let loaded = Session::open_loaded(
        Arc::clone(&under),
        scene,
        None,
        "cpu",
        Limits::default(),
        Wait::blocking(),
        false,
    )
    .map_err(|why| format!("{}: {why}", scene.display()))?;
    let mut session = loaded.session;
    let mut events = loaded.events;
    events.extend(session.sync());
    for event in &events {
        match event {
            Event::SceneRejected(error) => return Err(format!("{}: {error}", scene.display())),
            Event::Rejected { path, error } => {
                return Err(format!("{}: {error}", path.display()));
            }
            _ => {}
        }
    }
    let mut fonts = Vec::new();
    let bundled = session
        .bundle_with_fonts(|node, path| {
            let bytes = under
                .read(path)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            let (fields, _) = entelechy_import::font::read(&bytes, chars)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            if chars.is_some() {
                let kept = (fields.glyphs.storage().len()
                    + fields.segments.storage().len()
                    + fields.metrics.storage().len())
                    * 4;
                fonts.push((node.to_owned(), bytes.len(), kept));
            }
            Ok((fields.glyphs, fields.segments, fields.metrics))
        })
        .map_err(|why| format!("{}: {why}", scene.display()))?;
    let modules = bundled.modules.len();
    let inputs = bundled.inputs.len();
    let bundle = bundled.encode();
    Ok(Compiled {
        bundle,
        modules,
        inputs,
        fonts,
    })
}

/// Bytes as the page carries what is large: a zlib stream in UTF-8 base122,
/// which the browser inflates.
fn packed(bytes: &[u8]) -> String {
    base122(&zlib_tight(bytes))
}

/// Text put inside a `<script>`: nothing in it may end the element.
fn inside_script(text: &str) -> String {
    text.replace("</script", "<\\/script")
}

/// `template` with every `@@include:<file>@@` replaced by that file from
/// `beside`, every `@@pack:<file>@@` by that file packed, and every
/// `@@slot:<name>@@` by its text in `slots`. A slot the template names and
/// `slots` lacks, or a file not there, is an error.
pub fn fill(
    template: &str,
    beside: &Path,
    slots: &BTreeMap<&str, String>,
) -> Result<String, String> {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(at) = rest.find("@@") {
        out.push_str(&rest[..at]);
        let after = &rest[at + 2..];
        let end = after
            .find("@@")
            .ok_or("the page's template has an `@@` that is not closed")?;
        let mark = &after[..end];
        if let Some(file) = mark.strip_prefix("include:") {
            let path = beside.join(file);
            let text =
                std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            out.push_str(&inside_script(&text));
        } else if let Some(file) = mark.strip_prefix("pack:") {
            let path = beside.join(file);
            let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            out.push_str(&packed(&bytes));
        } else if let Some(name) = mark.strip_prefix("slot:") {
            out.push_str(slots.get(name).ok_or_else(|| {
                format!("the page's template names a slot `{name}` the exporter does not fill")
            })?);
        } else {
            return Err(format!(
                "the page's template has `@@{mark}@@`, neither an include, a pack nor a slot"
            ));
        }
        rest = &after[end + 2..];
    }
    out.push_str(rest);
    Ok(out)
}

/// What a page was made of, for the line the exporter prints.
pub struct Made {
    pub page: String,
    pub modules: usize,
    pub inputs: usize,
    pub bundle_bytes: usize,
    pub runtime_bytes: usize,
    /// Each font node subsetted by `--chars`: `(node, font_file_bytes, atlas_bytes)`.
    pub fonts: Vec<(String, usize, usize)>,
}

/// `scene` as one page. `web` is where Entelechy for the web is: the
/// directory with `bridges/web/` (the template and its player) and
/// `target/page/` (`entelechy_single.{js,_bg.wasm}`). `backend` is what runs
/// it in the page (`gpu` or `jit`, or `best` when none is named). `chars`,
/// given, is the characters the scene's font nodes keep.
pub fn page(
    scene: &Path,
    web: &Path,
    backend: Option<&str>,
    chars: Option<&str>,
) -> Result<Made, String> {
    let backend = match backend {
        None | Some("best") => "best",
        Some(name @ ("gpu" | "jit")) => name,
        Some("cpu") => return Err("a page runs on gpu or jit, not cpu".into()),
        Some(other) => return Err(format!("--backend for a page is gpu or jit, not `{other}`")),
    };
    let compiled = compile(scene, chars)?;
    let bridge = web.join("bridges").join("web");
    let built = web.join("target").join("page");
    let read = |path: PathBuf| {
        std::fs::read(&path).map_err(|e| {
            format!(
                "{}: {e} (Entelechy for the web is looked for in --web, `{}`)",
                path.display(),
                web.display()
            )
        })
    };
    let template =
        String::from_utf8(read(bridge.join("single.html"))?).map_err(|e| e.to_string())?;
    let glue = read(built.join("entelechy_single.js"))?;
    let runtime = read(built.join("entelechy_single_bg.wasm"))?;
    let title = scene
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut slots = BTreeMap::new();
    slots.insert("title", title.replace('&', "&amp;").replace('<', "&lt;"));
    slots.insert("backend", backend.to_owned());
    slots.insert("glue", packed(&glue));
    slots.insert("runtime", packed(&runtime));
    slots.insert("bundle", packed(&compiled.bundle));
    Ok(Made {
        page: fill(&template, &bridge, &slots)?,
        modules: compiled.modules,
        inputs: compiled.inputs,
        bundle_bytes: compiled.bundle.len(),
        runtime_bytes: runtime.len(),
        fonts: compiled.fonts,
    })
}
