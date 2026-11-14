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
use entelechy_formats::font::CompactFont;
use entelechy_session::bundle::BundledAtlas;
use entelechy_session::{Event, Limits, Session, SessionOptions, Wait};

/// A scene compiled into its binary [`Bundle`](entelechy_session::bundle::Bundle)
/// bytes, with counts for the line the exporter prints.
pub struct Compiled {
    pub bundle: Vec<u8>,
    pub modules: usize,
    pub inputs: usize,
    /// Each font node subsetted (by `--chars` or static `codes`):
    /// `(node, font_file_bytes, compact_bytes)`.
    pub fonts: Vec<(String, usize, usize)>,
}

/// Loads `scene` as a front does (ignoring any runtime save files the scene
/// writes for itself), checks that nothing in it or its inputs was refused,
/// bakes every static input and compact font atlas (`chars` keeping those
/// characters alone when given, else static `codes` when invariant), and
/// encodes the binary bundle. No host path is written into the bundle.
pub fn compile(scene: &Path, chars: Option<&str>) -> Result<Compiled, String> {
    let under = SessionOptions::from_environment().store();
    let built = Session::build("cpu", Limits::default(), Wait::blocking())
        .map_err(|why| format!("{}: {why}", scene.display()))?;
    let mut session = Session::open_in(Arc::clone(&under), scene, None, built.runtime);
    session.start_from_nothing_saved();
    let events = session.sync();
    for event in &events {
        match event {
            Event::SceneRejected(error) => return Err(format!("{}: {error}", scene.display())),
            Event::Rejected { path, error } => {
                return Err(format!("{}: {error}", path.display()));
            }
            _ => {}
        }
    }
    if session.graph().is_none() {
        return Err(format!("{}: scene did not load", scene.display()));
    }
    let mut fonts = Vec::new();
    let mut font_cache: BTreeMap<PathBuf, (usize, usize, CompactFont)> = BTreeMap::new();
    let bundled = session
        .bundle_with_fonts(|node, path, static_chars| {
            let chosen = chars.or(static_chars);
            let (raw_len, kept_len, compact) = if let Some(cached) = font_cache.get(path) {
                cached.clone()
            } else {
                let bytes = under
                    .read(path)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                let (compact, _) =
                    entelechy_import::font::read_compact(&bytes, chosen, chars.is_none())
                        .map_err(|e| format!("{}: {e}", path.display()))?;
                let mut encoded = Vec::new();
                compact.write_to(&mut encoded);
                let entry = (bytes.len(), encoded.len(), compact);
                font_cache.insert(path.to_path_buf(), entry.clone());
                entry
            };
            if chosen.is_some() {
                fonts.push((node.to_owned(), raw_len, kept_len));
            }
            Ok(BundledAtlas::Compact(compact))
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

/// How an exported page handles the browser's pointer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PointerMode {
    /// Free browser cursor mapped into the canvas.
    #[default]
    Free,
    /// Captured on click, stepped by relative motion and clamped to the output's edges.
    Hold,
    /// Captured on click, stepped by unbounded relative motion for 360-degree camera aim.
    Lock,
}

impl PointerMode {
    /// The slot text written into `single.html`'s `data-pointer`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Free => "free",
            Self::Hold => "hold",
            Self::Lock => "lock",
        }
    }
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

/// Default page title derived from `scene`: the scene's stem, or its owning
/// directory's name when the stem is the generic entry name `scene` or `world`.
pub fn default_title(scene: &Path) -> String {
    let stem = scene
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    if matches!(stem.as_str(), "scene" | "world") {
        let mut dir = scene.parent();
        while let Some(parent) = dir {
            let name = parent
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            if matches!(name.as_str(), "graph" | "graphs") {
                dir = parent.parent();
                continue;
            }
            if !name.is_empty() && name != "." && name != ".." {
                return name;
            }
            break;
        }
    }
    stem
}

fn escape_html(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// `scene` as one page with a free pointer ([`page_with_pointer`]).
pub fn page(
    scene: &Path,
    web: &Path,
    backend: Option<&str>,
    chars: Option<&str>,
) -> Result<Made, String> {
    page_with_options(scene, web, backend, chars, PointerMode::Free, None)
}

/// `scene` as one page with `pointer` handling and its default title
/// ([`page_with_options`]).
pub fn page_with_pointer(
    scene: &Path,
    web: &Path,
    backend: Option<&str>,
    chars: Option<&str>,
    pointer: PointerMode,
) -> Result<Made, String> {
    page_with_options(scene, web, backend, chars, pointer, None)
}

/// `scene` as one page. `web` is where Entelechy for the web is: the
/// directory with `bridges/web/` (the template and its player) and
/// `target/page/` (`entelechy_single.{js,_bg.wasm}`). `backend` is what runs
/// it in the page (`gpu` or `jit`, or `best` when none is named). `chars`,
/// given, is the characters the scene's font nodes keep; `pointer` says
/// whether clicking captures the browser cursor (`hold` clamped, `lock`
/// unbounded); `title`, given, names the page instead of [`default_title`].
pub fn page_with_options(
    scene: &Path,
    web: &Path,
    backend: Option<&str>,
    chars: Option<&str>,
    pointer: PointerMode,
    title: Option<&str>,
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
    let chosen_title = title
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| default_title(scene));
    let mut slots = BTreeMap::new();
    slots.insert("title", escape_html(&chosen_title));
    slots.insert("backend", backend.to_owned());
    slots.insert("pointer", pointer.as_str().to_owned());
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
