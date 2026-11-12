//! A scene as one page: a single HTML file that is the Player in a page,
//! Entelechy built for a page, and every file the scene reads. Nothing
//! is fetched; it opens from the disk.
//!
//! The exporter gathers and packs. How the page starts is the web
//! bridge's (`bridges/web/single.html`, a template): an `@@include:<file>@@`
//! is a file beside the template put in as it is, an `@@slot:<name>@@` is
//! what is made here.

use std::collections::BTreeMap;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use entelechy_formats::base64::base64;
use entelechy_formats::deflate::zlib;
use entelechy_session::store::{Kind, Stamp};
use entelechy_session::{Later, Limits, Parsed, Session, SessionOptions, Store};

/// A store that answers as another does and keeps the path of every file
/// read through it: what a scene reads as it loads is what its page must
/// carry.
struct Recording {
    under: Arc<dyn Store>,
    read: Mutex<Vec<PathBuf>>,
}

impl Recording {
    fn note(&self, path: &Path) {
        if let Ok(mut read) = self.read.lock() {
            read.push(self.under.whole(path));
        }
    }
}

impl Store for Recording {
    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        let bytes = self.under.read(path)?;
        self.note(path);
        Ok(bytes)
    }
    fn read_start(&self, path: &Path, bytes: usize) -> io::Result<Vec<u8>> {
        let start = self.under.read_start(path, bytes)?;
        self.note(path);
        Ok(start)
    }
    fn read_words(&self, path: &Path, from: usize) -> io::Result<(Vec<u32>, usize)> {
        let words = self.under.read_words(path, from)?;
        self.note(path);
        Ok(words)
    }
    fn read_later(&self, path: &Path) -> Box<dyn Later> {
        if self.under.kind(path).is_some() {
            self.note(path);
        }
        self.under.read_later(path)
    }
    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        self.under.write(path, bytes)
    }
    fn remove(&self, path: &Path) -> io::Result<()> {
        self.under.remove(path)
    }
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.under.rename(from, to)
    }
    fn remove_empty(&self, dir: &Path) -> io::Result<()> {
        self.under.remove_empty(dir)
    }
    fn kind(&self, path: &Path) -> Option<Kind> {
        self.under.kind(path)
    }
    fn list(&self, dir: &Path) -> io::Result<Vec<PathBuf>> {
        self.under.list(dir)
    }
    fn stamp(&self, path: &Path) -> Option<Stamp> {
        self.under.stamp(path)
    }
    fn canonical(&self, path: &Path) -> io::Result<PathBuf> {
        self.under.canonical(path)
    }
    fn whole(&self, path: &Path) -> PathBuf {
        self.under.whole(path)
    }
    fn packages(&self) -> &[PathBuf] {
        self.under.packages()
    }
    fn parsed(&self) -> Option<&Parsed> {
        self.under.parsed()
    }
    fn stamps_directories(&self) -> bool {
        self.under.stamps_directories()
    }
    fn sync_begins(&self) {
        self.under.sync_begins();
    }
    fn sync_ends(&self) {
        self.under.sync_ends();
    }
}

/// What a scene reads, as its page holds it: each file under the name the
/// page gives it, and the scene's own.
pub struct Gathered {
    pub files: Vec<(String, Vec<u8>)>,
    pub scene: String,
}

/// The directory every one of `paths` is in.
fn common(paths: &[PathBuf]) -> PathBuf {
    let mut shared: Option<Vec<Component>> = None;
    for path in paths {
        let dir: Vec<Component> = path
            .parent()
            .unwrap_or(Path::new(""))
            .components()
            .collect();
        shared = Some(match shared {
            None => dir,
            Some(so_far) => so_far
                .iter()
                .zip(&dir)
                .take_while(|(a, b)| a == b)
                .map(|(a, _)| *a)
                .collect(),
        });
    }
    shared.unwrap_or_default().iter().collect()
}

fn slashed(path: &Path) -> String {
    path.components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

/// Loads `scene` as a front does (packages found where every front finds
/// them) and gathers every file it read. In the page a package found by
/// its name is in `packages/`, and everything else keeps its place under
/// `play/`, counted from the directory all of it is in: so what names a
/// file beside it, or a package by its name, finds it there too, and no
/// directory of the machine it was exported on is written into the page.
pub fn gather(scene: &Path) -> Result<Gathered, String> {
    let under = SessionOptions::from_environment().store();
    let recording = Arc::new(Recording {
        under: Arc::clone(&under),
        read: Mutex::new(Vec::new()),
    });
    let store: Arc<dyn Store> = recording.clone();
    let loaded = Session::open_loaded(
        store,
        scene,
        None,
        "cpu",
        Limits::default(),
        entelechy_session::Wait::blocking(),
        false,
    )
    .map_err(|why| format!("{}: {why}", scene.display()))?;
    let mut session = loaded.session;
    // (What a scene's inputs name is read as the scene is first run.)
    session.sync();
    drop(session);
    let whole_scene = under.whole(scene);
    let mut read: Vec<PathBuf> = recording
        .read
        .lock()
        .map_err(|_| "the files read were lost".to_string())?
        .clone();
    read.push(whole_scene.clone());
    read.sort();
    read.dedup();
    let shelves: Vec<PathBuf> = under
        .packages()
        .iter()
        .map(|dir| under.whole(dir))
        .collect();
    let shelf_of = |path: &Path| shelves.iter().find(|shelf| path.starts_with(shelf));
    let own: Vec<PathBuf> = read
        .iter()
        .filter(|p| shelf_of(p).is_none())
        .cloned()
        .collect();
    let root = common(&own);
    let named = |path: &Path| -> Result<String, String> {
        let inside = match shelf_of(path) {
            Some(shelf) => {
                Path::new("packages").join(path.strip_prefix(shelf).map_err(|e| e.to_string())?)
            }
            None => Path::new("play").join(path.strip_prefix(&root).map_err(|e| e.to_string())?),
        };
        Ok(slashed(&inside))
    };
    let mut files = BTreeMap::new();
    for path in &read {
        let bytes = under
            .read(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        files.insert(named(path)?, bytes);
    }
    Ok(Gathered {
        scene: named(&whole_scene)?,
        files: files.into_iter().collect(),
    })
}

/// The files one after another, as the page's template says: a count,
/// then for each a path's length, the path, the bytes' length, the bytes;
/// lengths little-endian in four bytes.
pub fn archive(files: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut out = (files.len() as u32).to_le_bytes().to_vec();
    for (path, bytes) in files {
        out.extend_from_slice(&(path.len() as u32).to_le_bytes());
        out.extend_from_slice(path.as_bytes());
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    out
}

/// Text put inside a `<script>`: nothing in it may end the element.
fn inside_script(text: &str) -> String {
    text.replace("</script", "<\\/script")
}

/// `template` with every `@@include:<file>@@` replaced by that file from
/// `beside` and every `@@slot:<name>@@` by its text in `slots`. A slot
/// the template names and `slots` lacks, or a file not there, is an error.
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
        } else if let Some(name) = mark.strip_prefix("slot:") {
            out.push_str(slots.get(name).ok_or_else(|| {
                format!("the page's template names a slot `{name}` the exporter does not fill")
            })?);
        } else {
            return Err(format!(
                "the page's template has `@@{mark}@@`, neither an include nor a slot"
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
    pub files: usize,
    pub files_bytes: usize,
    pub runtime_bytes: usize,
}

/// `scene` as one page. `web` is where Entelechy for the web is: the
/// directory with `bridges/web/` (the template and the Player in a page)
/// and `target/page/` (Entelechy built for a page), as a web release and
/// a checkout that has built it both are. `backend` is what runs it in
/// the page (`best` when none is named, as the Player chooses).
pub fn page(scene: &Path, web: &Path, backend: Option<&str>) -> Result<Made, String> {
    let gathered = gather(scene)?;
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
    let glue =
        String::from_utf8(read(built.join("entelechy_ffi.js"))?).map_err(|e| e.to_string())?;
    let runtime = read(built.join("entelechy_ffi_bg.wasm"))?;
    let packed = archive(&gathered.files);
    let title = scene
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut play = serde_json::Map::new();
    play.insert("scene".into(), gathered.scene.clone().into());
    if let Some(backend) = backend {
        play.insert("backend".into(), backend.into());
    }
    let mut slots = BTreeMap::new();
    slots.insert("title", title.replace('<', "&lt;").replace('&', "&amp;"));
    slots.insert("glue", inside_script(&glue));
    slots.insert("runtime", base64(&zlib(&runtime)));
    slots.insert("files", base64(&zlib(&packed)));
    slots.insert(
        "play",
        inside_script(&serde_json::Value::Object(play).to_string()),
    );
    Ok(Made {
        page: fill(&template, &bridge, &slots)?,
        files: gathered.files.len(),
        files_bytes: packed.len(),
        runtime_bytes: runtime.len(),
    })
}
