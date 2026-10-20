//! The Fields a scene's inputs hold: a file (reread on change), a
//! directory frame chosen by time, a text as character codes, or a
//! caller binding. Inputs own their Host handles and release each one
//! they replace or forget. A data table (`.csv`, `.json`) is read as the
//! Kernel reading the input lays out its element, and read again when
//! that layout changes.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use entelechy_graph::{Binding, Graph, Source};
use entelechy_kernel::{FieldData, Handle, Host, Runtime, Schema};

use crate::store::{Stamp, Store};
use crate::{Diagnostic, Event};

/// A file an input is read from, and the element a table was read as and
/// the tables its rows named, as they were when it was read.
struct Asset {
    path: PathBuf,
    stamp: Option<Stamp>,
    table: Option<TableElement>,
    named: Vec<(PathBuf, Option<Stamp>)>,
}

/// The element a data table input is read as: the name of each number it
/// holds and the layout, as the Kernel reading the input lays them out.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TableElement {
    names: Vec<String>,
    schema: Schema,
}

/// What a scene declares of its inputs: its graph, the files and
/// directories they are read from, and the element each data table is
/// read as ([`table_elements`]).
pub(crate) struct Declared<'a> {
    pub(crate) graph: Option<&'a Graph>,
    pub(crate) files: &'a entelechy_scene::Files,
    pub(crate) tables: &'a BTreeMap<String, TableElement>,
}

/// For each scene input a node reads, the element the first node reading
/// it (in the graph's order) lays it out as; an input no published Kernel
/// reads has none.
pub(crate) fn table_elements(
    graph: Option<&Graph>,
    runtime: &Runtime,
) -> BTreeMap<String, TableElement> {
    let mut found = BTreeMap::new();
    let Some(graph) = graph else {
        return found;
    };
    for node in &graph.nodes {
        // The scene inputs it reads that no node before it laid out.
        let reads: Vec<(&String, &String)> = node
            .inputs
            .iter()
            .flat_map(|(port, binding)| {
                let sources = match binding {
                    Binding::Now(source) => vec![source],
                    Binding::Previous {
                        source, initial, ..
                    } => vec![source, initial],
                };
                sources.into_iter().filter_map(move |source| match source {
                    Source::External(name) => Some((port, name)),
                    Source::Node { .. } => None,
                })
            })
            .filter(|(_, name)| !found.contains_key(*name))
            .collect();
        if reads.is_empty() {
            continue;
        }
        let Some(version) = runtime.kernel(&node.kernel) else {
            continue;
        };
        let native;
        let structs: &[entelechy_ir::StructDef] = match (version.module(), version.native()) {
            (Some(module), _) => &module.structs,
            (None, Some(op)) => {
                native = op.structs();
                &native
            }
            (None, None) => &[],
        };
        for (port, name) in reads {
            if found.contains_key(name) {
                continue;
            }
            let Some(declared) = version.manifest().inputs.iter().find(|p| &p.name == port) else {
                continue;
            };
            let ty = declared.element();
            if let (Some(names), Some(schema)) = (
                entelechy_ir::number_names(ty, structs),
                entelechy_ir::schema_of(ty, structs),
            ) {
                found.insert(name.clone(), TableElement { names, schema });
            }
        }
    }
    found
}

/// A directory input: frame files in name order and the one held now.
struct Sequence {
    dir: PathBuf,
    ty: String,
    rate: u32,
    files: Vec<PathBuf>,
    current: Option<usize>,
}

/// A `checks` or an `outline` input: the file or directory it reads,
/// whether it holds what the documents define (else what is wrong in
/// them), and each document found there with its stamp when it was last
/// read.
struct Checks {
    path: PathBuf,
    outline: bool,
    /// The documents, with their stamps, of the text held.
    documents: Vec<(PathBuf, Option<Stamp>)>,
    /// A check of them as they are now, made off the frames while the
    /// session prepares in the background ([`Inputs::set_background`]):
    /// taken in by the first sync after it is done.
    running: Option<Running>,
}

/// A check made off the frames: the documents, with their stamps, it
/// checks, and where its text comes back.
struct Running {
    documents: Vec<(PathBuf, Option<Stamp>)>,
    text: std::sync::mpsc::Receiver<String>,
}

/// A font node's font as last read, and the characters it was last given
/// (their Field's handle, so the same Field is not read twice).
struct Font {
    path: PathBuf,
    stamp: Option<Stamp>,
    bytes: Vec<u8>,
    codes: Vec<i32>,
    seen: Option<Handle>,
}

/// The documents a `checks` input checks: the file, or every `.ent` and
/// `.esn` under the directory (hidden ones left out), in name order.
fn documents_under(store: &dyn Store, path: &Path) -> Vec<PathBuf> {
    match store.kind(path) {
        Some(crate::store::Kind::File) => vec![path.to_path_buf()],
        Some(crate::store::Kind::Dir) => {
            let mut found = Vec::new();
            let mut entries = store.list(path).unwrap_or_default();
            entries.sort();
            for entry in entries {
                let hidden = entry
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_none_or(|name| name.starts_with('.'));
                if hidden {
                    continue;
                }
                let document =
                    matches!(entelechy_formats::extension(&entry).as_str(), "ent" | "esn");
                match store.kind(&entry) {
                    Some(crate::store::Kind::Dir) => found.extend(documents_under(store, &entry)),
                    Some(crate::store::Kind::File) if document => found.push(entry),
                    _ => {}
                }
            }
            found
        }
        None => Vec::new(),
    }
}

/// A `files` input: its directory, and each file found there, by its
/// path under it, with its stamp when it was last read.
struct Listed {
    dir: PathBuf,
    files: Vec<(String, Option<Stamp>)>,
}

/// A `files` node: its directory, each file it listed with its stamp,
/// the paths it was last given and each file it read of them with its
/// stamp, and the Field it last read its paths from.
struct FileNode {
    dir: PathBuf,
    listed: Vec<(String, Option<Stamp>)>,
    paths: Vec<String>,
    opened: Vec<(String, Option<Stamp>)>,
    seen: Option<Handle>,
}

/// A `fields` node: its directory, the paths it was last given, each file
/// they name as it stands, the Field it last read its paths from, and
/// whether what it holds changed since its outputs were last made.
struct FieldNode {
    dir: PathBuf,
    paths: Vec<String>,
    files: BTreeMap<String, Chunk>,
    seen: Option<Handle>,
    changed: bool,
    /// The element as the Kernel reading the result lays it out, as the
    /// last sync found it.
    element: Option<Schema>,
}

/// A Field file a `fields` node names, with the stamp it was asked at.
enum Chunk {
    /// On its way.
    Coming(Box<dyn crate::store::Later>, Option<Stamp>),
    /// In: its elements as the Kernel's words, how many, and their layout.
    In(Vec<u32>, usize, Schema, Option<Stamp>),
    /// Not in: no such file, or one refused and reported.
    Out(Option<Stamp>),
}

impl Chunk {
    fn stamp(&self) -> Option<Stamp> {
        match self {
            Self::Coming(_, stamp) | Self::In(.., stamp) | Self::Out(stamp) => *stamp,
        }
    }
}

/// Starts reading `file` for a `fields` node: off the frames where the
/// session prepares in the background, else now.
fn ask(store: &dyn Store, file: &Path, background: bool) -> Box<dyn crate::store::Later> {
    if background {
        store.read_later(file)
    } else {
        crate::store::read_now(store.read(file))
    }
}

/// A Field file as it came in for a `fields` node whose element is
/// `element` (where a Kernel reads it): its elements as the Kernel's
/// words, how many and their layout, within `room` words.
fn taken(
    arrived: &dyn crate::store::Arrived,
    element: Option<&Schema>,
    room: usize,
) -> Result<(Vec<u32>, usize, Schema), String> {
    let mut want = entelechy_formats::file::HEADER_BYTES;
    let start = loop {
        let start = arrived.start(want);
        if start.len() < want || entelechy_formats::file::header_whole(&start) {
            break start;
        }
        want = want.saturating_mul(2);
    };
    let (schema, domain, from) =
        entelechy_formats::file::layout(&start).map_err(|e| e.to_string())?;
    if let Some(element) = element
        && schema.scalars() != element.scalars()
    {
        return Err(format!(
            "its elements are {schema}, where the node's are {element}"
        ));
    }
    let needed = schema.storage_words(domain.count());
    if needed > room {
        return Err(entelechy_kernel::HostError::BudgetExceeded {
            requested: needed,
            available: room,
        }
        .to_string());
    }
    let (words, bytes) = arrived.words(from);
    let data = entelechy_formats::file::from_words(schema, domain, words, bytes)
        .map_err(|e| e.to_string())?;
    let wide = Schema::new(data.schema().scalars().to_vec());
    Ok((data.values().into_owned(), domain.count(), wide))
}

/// A fields node's outputs from what it holds: `result`, the elements of
/// the files in, in the order named (one zero element where none is), in
/// `element`'s layout, else the first file's; `ranges`, a row a path,
/// where its elements start and how many there are, -1 and 0 for one not
/// in (one such row where no path is named).
fn made_of(
    node: &FieldNode,
    element: Option<&Schema>,
) -> Vec<(&'static str, Result<FieldData, String>)> {
    let schema = element.cloned().or_else(|| {
        node.files.values().find_map(|chunk| match chunk {
            Chunk::In(_, _, schema, _) => Some(schema.clone()),
            _ => None,
        })
    });
    let mut words = Vec::new();
    let mut ranges = Vec::with_capacity(node.paths.len().max(1) * 2);
    let mut start = 0usize;
    for path in &node.paths {
        match node.files.get(path) {
            Some(Chunk::In(chunk, count, _, _)) => {
                words.extend_from_slice(chunk);
                ranges.extend([start as u32, *count as u32]);
                start += count;
            }
            _ => ranges.extend([-1i32 as u32, 0]),
        }
    }
    if ranges.is_empty() {
        ranges.extend([-1i32 as u32, 0]);
    }
    let mut made = vec![(
        "ranges",
        FieldData::from_storage(
            Schema::new(vec![entelechy_kernel::Scalar::Int; 2]),
            entelechy_kernel::Domain::Rows((ranges.len() / 2) as u32),
            ranges,
        )
        .map_err(|e| e.to_string()),
    )];
    if let Some(schema) = schema {
        if start == 0 {
            words = vec![0; schema.value_words()];
        }
        made.push((
            "result",
            FieldData::from_storage(
                schema,
                entelechy_kernel::Domain::Rows(start.max(1) as u32),
                words,
            )
            .map_err(|e| e.to_string()),
        ));
    }
    made
}

/// Every file under `dir` a listing shows: hidden ones, those under hidden
/// directories and those a `.gitignore` on the way leaves out are not
/// ([`crate::ignore`]); each by its path under `dir`, parted by `/`, with
/// its stamp, in path order.
fn listed_under(store: &dyn Store, dir: &Path) -> Vec<(String, Option<Stamp>)> {
    fn walk(
        store: &dyn Store,
        dir: &Path,
        prefix: &str,
        ignore: &crate::ignore::Ignore,
        found: &mut Vec<(String, Option<Stamp>)>,
    ) {
        let ignore = match store.read_text(&dir.join(".gitignore")) {
            Ok(text) => ignore.with(prefix, &text),
            Err(_) => ignore.clone(),
        };
        let mut entries = store.list(dir).unwrap_or_default();
        entries.sort();
        for entry in entries {
            let Some(name) = entry.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.starts_with('.') {
                continue;
            }
            let path = format!("{prefix}{name}");
            match store.kind(&entry) {
                Some(crate::store::Kind::Dir) if !ignore.ignores(&path, true) => {
                    walk(store, &entry, &format!("{path}/"), &ignore, found);
                }
                Some(crate::store::Kind::File) if !ignore.ignores(&path, false) => {
                    found.push((path, store.stamp(&entry)));
                }
                _ => {}
            }
        }
    }
    let mut found = Vec::new();
    if store.kind(dir) == Some(crate::store::Kind::Dir) {
        walk(
            store,
            dir,
            "",
            &crate::ignore::Ignore::default(),
            &mut found,
        );
    }
    found.sort_by(|a, b| a.0.cmp(&b.0));
    found
}

/// The paths a files node's `paths` spell: each path's UTF-8 bytes, ended
/// by a 0 or the Field's end; an empty one is no path.
pub(crate) fn paths_spelled(words: &[u32]) -> Result<Vec<String>, String> {
    let mut paths = Vec::new();
    for part in words.split(|word| *word == 0) {
        if part.is_empty() {
            continue;
        }
        let bytes: Vec<u8> = part
            .iter()
            .map(|word| u8::try_from(*word).map_err(|_| format!("{word} is not a byte of a path")))
            .collect::<Result<_, _>>()?;
        let path = String::from_utf8(bytes).map_err(|_| "a path is not UTF-8".to_owned())?;
        if !paths.contains(&path) {
            paths.push(path);
        }
    }
    Ok(paths)
}

/// Every file under `dir` (hidden ones and those under hidden
/// directories left out), each by its path under `dir`, parted by `/`,
/// in path order.
fn files_under(store: &dyn Store, dir: &Path) -> Vec<String> {
    fn walk(store: &dyn Store, dir: &Path, prefix: &str, found: &mut Vec<String>) {
        let mut entries = store.list(dir).unwrap_or_default();
        entries.sort();
        for entry in entries {
            let Some(name) = entry.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.starts_with('.') {
                continue;
            }
            let path = format!("{prefix}{name}");
            match store.kind(&entry) {
                Some(crate::store::Kind::Dir) => walk(store, &entry, &format!("{path}/"), found),
                Some(crate::store::Kind::File) => found.push(path),
                None => {}
            }
        }
    }
    let mut found = Vec::new();
    if store.kind(dir) == Some(crate::store::Kind::Dir) {
        walk(store, dir, "", &mut found);
    }
    found.sort();
    found
}

/// The file extensions a frame may have.
const FRAME_EXTENSIONS: [&str; 5] = ["png", "ppm", "pgm", "field", "wav"];

/// The files of `dir` that can be frames, in name order.
fn frame_files(store: &dyn Store, dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files: Vec<PathBuf> = store
        .list(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .into_iter()
        .filter(|path| {
            path.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| FRAME_EXTENSIONS.contains(&e))
        })
        .collect();
    files.sort();
    Ok(files)
}

/// The words the Host has room for, which a decoded file may take.
fn room(host: &Host) -> usize {
    host.budget_words().saturating_sub(host.used_words())
}

/// Puts an input just read on the device, when there is one and its words
/// take [`COPY_LEAST`](crate::memory::COPY_LEAST) bytes or more, and lets
/// go of its Host copy: a load then holds one big input's words at a time
/// beside its documents, not all of them. What reads it on the processor
/// brings the words back once. Answers whether its copy went.
fn put_on_device(runtime: &mut Runtime, handle: Handle) -> bool {
    runtime.has_device()
        && runtime
            .host()
            .field_bytes(handle)
            .is_ok_and(|bytes| bytes.host >= crate::memory::COPY_LEAST)
        && runtime.upload(handle).unwrap_or(false)
        && runtime.host_mut().let_go_of_copy(handle).is_ok()
}

/// A data table's rows as the element the Kernel reading input `name`
/// lays out; refused when no Kernel reads it.
fn read_table(
    bytes: &[u8],
    extension: &str,
    name: &str,
    table: Option<&TableElement>,
    room: usize,
    tables: &dyn Fn(&str) -> Result<Vec<String>, String>,
) -> Result<(FieldData, String), String> {
    let table = table.ok_or_else(|| {
        format!("a data table is read as the Kernel reading it lays out its element, and no published Kernel reads `{name}`")
    })?;
    let element = entelechy_formats::table::Element {
        names: &table.names,
        schema: &table.schema,
    };
    let data = entelechy_formats::table::decode(bytes, extension, &element, room, tables)?;
    let detail = format!("{} row(s) of {}", data.domain().count(), data.schema());
    Ok((data, detail))
}

/// The row names of the table `name` beside `dir` (`name.csv`, else
/// `name.json`), read as any input is, and the file and its stamp noted.
fn table_names(
    store: &dyn Store,
    readable: &dyn Fn(&Path) -> Result<(), String>,
    dir: &Path,
    name: &str,
    noted: &std::cell::RefCell<Vec<(PathBuf, Option<Stamp>)>>,
) -> Result<Vec<String>, String> {
    if name.is_empty() || name.contains(['/', '\\']) || name.starts_with('.') {
        return Err(format!(
            "`{name}` is no table's name: a table is named by its file's stem beside this one"
        ));
    }
    for extension in ["csv", "json"] {
        let path = dir.join(format!("{name}.{extension}"));
        if !store.exists(&path) {
            continue;
        }
        readable(&path)?;
        let bytes = store
            .read(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        noted.borrow_mut().push((path.clone(), store.stamp(&path)));
        return entelechy_formats::table::row_names(&bytes, extension)
            .map_err(|why| format!("{}: {why}", path.display()));
    }
    Err(format!(
        "no table `{name}` beside it ({name}.csv or {name}.json)"
    ))
}

/// A file's bytes, when the Field its header describes fits in `room`
/// words: one too large is refused before its body is read. A header
/// longer than the bytes first read is read in twice as many until it is
/// whole or the file ends.
fn read_within(store: &dyn Store, path: &Path, room: usize) -> Result<Vec<u8>, String> {
    head_within(store, path, room)?;
    store.read(path).map_err(|e| e.to_string())
}

/// A file's first bytes, its whole header where it has one, once what
/// they say fits in `room` words.
fn head_within(store: &dyn Store, path: &Path, room: usize) -> Result<Vec<u8>, String> {
    let mut want = entelechy_formats::file::HEADER_BYTES;
    let start = loop {
        let start = store.read_start(path, want).map_err(|e| e.to_string())?;
        if start.len() < want || entelechy_formats::header_whole(&start, path) {
            break start;
        }
        want = want.saturating_mul(2);
    };
    entelechy_formats::fits(&start, path, room)?;
    Ok(start)
}

/// A Field file's Field and its line, its header read first and its
/// elements then as words, never held as bytes beside them.
fn read_field(store: &dyn Store, path: &Path, room: usize) -> Result<(FieldData, String), String> {
    let start = head_within(store, path, room)?;
    let (schema, domain, from) =
        entelechy_formats::file::layout(&start).map_err(|e| e.to_string())?;
    let (words, bytes) = store.read_words(path, from).map_err(|e| e.to_string())?;
    let data = entelechy_formats::file::from_words(schema, domain, words, bytes)
        .map_err(|e| e.to_string())?;
    let detail = entelechy_formats::describe(&data);
    Ok((data, detail))
}

/// The inputs of a scene and the Fields they hold.
#[derive(Default)]
pub(crate) struct Inputs {
    /// The Field each input holds, by name.
    handles: BTreeMap<String, Handle>,
    /// File inputs with their last-read stamps.
    assets: BTreeMap<String, Asset>,
    /// Directory inputs.
    sequences: BTreeMap<String, Sequence>,
    /// Inputs holding what the language finds wrong in documents.
    checks: BTreeMap<String, Checks>,
    /// Inputs holding a directory's files.
    listed: BTreeMap<String, Listed>,
    /// Font nodes' fonts, by node.
    fonts: BTreeMap<String, Font>,
    /// Files nodes' listings and the files they read, by node.
    file_nodes: BTreeMap<String, FileNode>,
    /// Fields nodes' files, by node.
    field_nodes: BTreeMap<String, FieldNode>,
    /// Text inputs with their text.
    texts: BTreeMap<String, String>,
    /// Frame selection failures, reported by the next sync.
    frame_events: Vec<Event>,
    /// Whether checks after the first are made off the frames, as Kernels
    /// prepare while the session prepares in the background.
    background: bool,
}

impl Inputs {
    /// Makes checks after an input's first off the frames: each is taken
    /// in by the first sync after it is done, the text held until then. As
    /// the session's replacements, never in a replay or a run, which take
    /// every check in at once.
    pub(crate) fn set_background(&mut self, enabled: bool) {
        self.background = enabled;
    }

    /// The Field each input holds, by name.
    pub(crate) fn handles(&self) -> &BTreeMap<String, Handle> {
        &self.handles
    }

    /// The files and directories the inputs are read from.
    pub(crate) fn watched(&self) -> impl Iterator<Item = PathBuf> + '_ {
        self.assets
            .values()
            .map(|asset| asset.path.clone())
            .chain(self.sequences.values().map(|sequence| sequence.dir.clone()))
            .chain(self.checks.values().map(|checks| checks.path.clone()))
            .chain(self.listed.values().map(|listed| listed.dir.clone()))
            .chain(self.fonts.values().map(|font| font.path.clone()))
            .chain(self.file_nodes.values().map(|node| node.dir.clone()))
            .chain(self.field_nodes.values().map(|node| node.dir.clone()))
    }

    /// Makes `handle` what input `name` holds, releasing what it held.
    pub(crate) fn hold(&mut self, host: &mut Host, name: &str, handle: Handle) {
        if let Some(previous) = self.handles.insert(name.to_owned(), handle) {
            let _ = host.release(previous);
        }
    }

    /// Whether `name` holds a Field.
    pub(crate) fn holds(&self, name: &str) -> bool {
        self.handles.contains_key(name)
    }

    fn forget(&mut self, host: &mut Host, name: &str) {
        if let Some(handle) = self.handles.remove(name) {
            let _ = host.release(handle);
        }
    }

    /// Syncs inputs to what the scene declares: reads new or changed
    /// files, lists directories, writes texts, forgets undeclared inputs.
    /// Answers the inputs read now that went on the device as they were
    /// read, their Host copies let go ([`put_on_device`]).
    pub(crate) fn sync(
        &mut self,
        store: &std::sync::Arc<dyn Store>,
        declared: &Declared,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) -> Vec<Handle> {
        let mut on_device = Vec::new();
        self.sync_assets(
            store.as_ref(),
            declared,
            readable,
            runtime,
            events,
            &mut on_device,
        );
        let host = runtime.host_mut();
        self.sync_sequences(store.as_ref(), declared.files, readable, host, events);
        self.sync_checks(store, declared.files, readable, host, events);
        self.sync_files(store.as_ref(), declared.files, readable, host, events);
        self.sync_fonts(store.as_ref(), declared.files, readable, host, events);
        self.sync_file_nodes(store.as_ref(), declared.files, readable, host, events);
        self.sync_field_nodes(store.as_ref(), declared, host, events);
        self.sync_texts(declared.graph, host, events);
        on_device
    }

    /// Reads each font node's font when it is new or changed and gives the
    /// node the glyphs of the characters it was last given (none at
    /// first); forgets undeclared ones.
    fn sync_fonts(
        &mut self,
        store: &dyn Store,
        files: &entelechy_scene::Files,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        host: &mut Host,
        events: &mut Vec<Event>,
    ) {
        let stale: Vec<String> = self
            .fonts
            .keys()
            .filter(|node| !files.fonts.iter().any(|font| &font.node == *node))
            .cloned()
            .collect();
        for node in stale {
            self.fonts.remove(&node);
            for port in entelechy_scene::FONT_OUTPUTS {
                self.forget(host, &format!("{node}.{port}"));
            }
        }
        for font in &files.fonts {
            // Found as an input's file is: there, else through a package's
            // name (`cjk.entpkg/fonts/x.ttf`).
            let path = crate::sources::file_at(store, &font.path);
            let stamp = store.stamp(&path);
            if self
                .fonts
                .get(&font.node)
                .is_some_and(|known| known.path == path && known.stamp == stamp)
            {
                continue;
            }
            let bytes = readable(&path).and_then(|()| store.read(&path).map_err(|e| e.to_string()));
            let bytes = match bytes {
                Ok(bytes) => bytes,
                Err(why) => {
                    // The node still gives Fields, the missing glyph's, so
                    // the frames that read it run.
                    self.fonts.remove(&font.node);
                    if let Err(error) =
                        self.hold_fields(font, entelechy_import::font::placeholder(), host)
                    {
                        events.push(Event::Rejected {
                            path: font.path.clone(),
                            error: Diagnostic::new(error),
                        });
                    }
                    events.push(Event::Rejected {
                        path: font.path.clone(),
                        error: Diagnostic::new(why),
                    });
                    continue;
                }
            };
            let codes = self
                .fonts
                .remove(&font.node)
                .filter(|known| known.path == path)
                .map(|known| known.codes)
                .unwrap_or_default();
            self.fonts.insert(
                font.node.clone(),
                Font {
                    path: path.clone(),
                    stamp,
                    bytes,
                    codes,
                    seen: None,
                },
            );
            if let Err(why) = self.give_glyphs(font, host) {
                if let Err(error) =
                    self.hold_fields(font, entelechy_import::font::placeholder(), host)
                {
                    events.push(Event::Rejected {
                        path: font.path.clone(),
                        error: Diagnostic::new(error),
                    });
                }
                events.push(Event::Rejected {
                    path: font.path.clone(),
                    error: Diagnostic::new(why),
                });
            } else {
                events.push(Event::Loaded {
                    name: font.node.clone(),
                    path: font.path.clone(),
                    detail: "a font node's font".into(),
                });
            }
        }
    }

    /// Lists each files node's directory when what it holds changed, and
    /// reads again the files it was given that changed; a new node holds
    /// an empty `opened` until it is given paths; forgets undeclared ones.
    fn sync_file_nodes(
        &mut self,
        store: &dyn Store,
        files: &entelechy_scene::Files,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        host: &mut Host,
        events: &mut Vec<Event>,
    ) {
        let stale: Vec<String> = self
            .file_nodes
            .keys()
            .filter(|node| !files.file_nodes.iter().any(|wanted| &wanted.node == *node))
            .cloned()
            .collect();
        for node in stale {
            self.file_nodes.remove(&node);
            for port in entelechy_scene::FILES_OUTPUTS {
                self.forget(host, &format!("{node}.{port}"));
            }
        }
        for node in &files.file_nodes {
            let fresh = self
                .file_nodes
                .get(&node.node)
                .is_none_or(|known| known.dir != node.dir);
            if fresh {
                self.file_nodes.insert(
                    node.node.clone(),
                    FileNode {
                        dir: node.dir.clone(),
                        listed: Vec::new(),
                        paths: Vec::new(),
                        opened: Vec::new(),
                        seen: None,
                    },
                );
            }
            // The listing, when a file came, went or changed.
            let listed = match readable(&node.dir) {
                Ok(()) => listed_under(store, &node.dir),
                Err(why) => {
                    events.push(Event::Rejected {
                        path: node.dir.clone(),
                        error: Diagnostic::new(why),
                    });
                    Vec::new()
                }
            };
            let known = self.file_nodes.get(&node.node).expect("just placed");
            if fresh || known.listed != listed {
                let sized: Vec<(String, u64)> = listed
                    .iter()
                    .map(|(path, stamp)| {
                        let size = stamp.and_then(|stamp| stamp.length()).unwrap_or_else(|| {
                            store
                                .read(&node.dir.join(path))
                                .map_or(0, |bytes| bytes.len() as u64)
                        });
                        (path.clone(), size)
                    })
                    .collect();
                match self.hold_output(
                    node,
                    "listing",
                    entelechy_formats::directory::pack_listing(&sized),
                    host,
                ) {
                    Ok(()) => events.push(Event::Loaded {
                        name: format!("{}.listing", node.node),
                        path: node.dir.clone(),
                        detail: format!("{} file(s) listed", sized.len()),
                    }),
                    Err(why) => events.push(Event::Rejected {
                        path: node.dir.clone(),
                        error: Diagnostic::new(why),
                    }),
                }
                self.file_nodes
                    .get_mut(&node.node)
                    .expect("just placed")
                    .listed = listed;
            }
            // What it read, when one of those files changed.
            let known = self.file_nodes.get(&node.node).expect("just placed");
            let changed = fresh
                || known
                    .opened
                    .iter()
                    .any(|(path, stamp)| store.stamp(&node.dir.join(path)) != *stamp)
                || known.paths.iter().any(|path| {
                    !known.opened.iter().any(|(read, _)| read == path)
                        && store.stamp(&node.dir.join(path)).is_some()
                });
            if changed {
                let paths = known.paths.clone();
                if let Err(why) = self.read_opened(store, readable, node, &paths, host) {
                    events.push(Event::Rejected {
                        path: node.dir.clone(),
                        error: Diagnostic::new(format!("files node `{}`: {why}", node.node)),
                    });
                }
            }
        }
    }

    /// Takes in the files each fields node named that came in, asks again
    /// for those that changed since they were read, and makes its outputs
    /// again where what it holds changed; forgets undeclared ones.
    fn sync_field_nodes(
        &mut self,
        store: &dyn Store,
        declared: &Declared,
        host: &mut Host,
        events: &mut Vec<Event>,
    ) {
        let files = declared.files;
        let stale: Vec<String> = self
            .field_nodes
            .keys()
            .filter(|node| !files.field_nodes.iter().any(|wanted| &wanted.node == *node))
            .cloned()
            .collect();
        for node in stale {
            self.field_nodes.remove(&node);
            for port in entelechy_scene::FIELDS_OUTPUTS {
                self.forget(host, &format!("{node}.{port}"));
            }
        }
        let background = self.background;
        for node in &files.field_nodes {
            if self
                .field_nodes
                .get(&node.node)
                .is_none_or(|known| known.dir != node.dir)
            {
                self.field_nodes.insert(
                    node.node.clone(),
                    FieldNode {
                        dir: node.dir.clone(),
                        paths: Vec::new(),
                        files: BTreeMap::new(),
                        seen: None,
                        changed: true,
                        element: None,
                    },
                );
            }
            // The element as the Kernel reading the result lays it out.
            let element = node
                .outputs
                .iter()
                .find(|(port, _)| port == "result")
                .and_then(|(_, external)| declared.tables.get(external))
                .map(|table| table.schema.clone());
            let known = self.field_nodes.get_mut(&node.node).expect("just placed");
            if known.element != element {
                known.element = element;
                known.changed = true;
            }
            // A file that changed since it was asked is asked again.
            for (path, chunk) in &mut known.files {
                let file = node.dir.join(path);
                let stamp = store.stamp(&file);
                if !matches!(chunk, Chunk::Coming(..)) && stamp != chunk.stamp() {
                    *chunk = Chunk::Coming(ask(store, &file, background), stamp);
                }
            }
            self.settle_field_node(node, host, events);
        }
    }

    /// Takes in the files fields node `node` named that came in, and makes
    /// its outputs again where what it holds changed.
    fn settle_field_node(
        &mut self,
        node: &entelechy_scene::FieldsNode,
        host: &mut Host,
        events: &mut Vec<Event>,
    ) {
        let room = room(host);
        let Some(known) = self.field_nodes.get_mut(&node.node) else {
            return;
        };
        for (path, chunk) in &mut known.files {
            let Chunk::Coming(later, stamp) = chunk else {
                continue;
            };
            let Some(read) = later.poll() else {
                continue;
            };
            let stamp = *stamp;
            let file = node.dir.join(path);
            known.changed = true;
            *chunk = match read {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Chunk::Out(stamp),
                Err(error) => {
                    events.push(Event::Rejected {
                        path: file,
                        error: Diagnostic::new(format!("fields node `{}`: {error}", node.node)),
                    });
                    Chunk::Out(stamp)
                }
                Ok(arrived) => match taken(arrived.as_ref(), known.element.as_ref(), room) {
                    Ok((words, count, schema)) => Chunk::In(words, count, schema, stamp),
                    Err(why) => {
                        events.push(Event::Rejected {
                            path: file,
                            error: Diagnostic::new(format!(
                                "fields node `{}`: {path}: {why}",
                                node.node
                            )),
                        });
                        Chunk::Out(stamp)
                    }
                },
            };
        }
        if !known.changed {
            return;
        }
        known.changed = false;
        let made = made_of(known, known.element.as_ref());
        for (port, data) in made {
            if let Err(why) = self.hold_field_output(node, port, data, host) {
                events.push(Event::Rejected {
                    path: node.dir.clone(),
                    error: Diagnostic::new(format!("fields node `{}`: {why}", node.node)),
                });
            }
        }
    }

    /// Holds `data` as fields node `node`'s output `port`.
    fn hold_field_output(
        &mut self,
        node: &entelechy_scene::FieldsNode,
        port: &str,
        data: Result<FieldData, String>,
        host: &mut Host,
    ) -> Result<(), String> {
        let Some((_, external)) = node.outputs.iter().find(|(name, _)| name == port) else {
            return Ok(());
        };
        let handle = host.adopt(data?).map_err(|e| e.to_string())?;
        self.hold(host, external, handle);
        Ok(())
    }

    /// Whether fields node `node` last read its paths from `handle`.
    pub(crate) fn field_node_saw(
        &self,
        node: &entelechy_scene::FieldsNode,
        handle: Handle,
    ) -> bool {
        self.field_nodes
            .get(&node.node)
            .is_none_or(|known| known.seen == Some(handle))
    }

    /// Gives fields node `node` the paths `paths` spell (read from the
    /// Field `handle`): a file no longer named is let go, one newly named
    /// asked for, taken in by a sync once it is in. Refused whole, the
    /// paths before staying, where a path is not one under the directory or
    /// is not readable; reported by the next sync.
    pub(crate) fn give_field_paths(
        &mut self,
        store: &dyn Store,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        node: &entelechy_scene::FieldsNode,
        handle: Handle,
        paths: Result<Vec<String>, String>,
        host: &mut Host,
    ) {
        let background = self.background;
        let Some(known) = self.field_nodes.get_mut(&node.node) else {
            return;
        };
        known.seen = Some(handle);
        let checked = paths.and_then(|paths| {
            for path in &paths {
                entelechy_formats::directory::checked_path(path)?;
                readable(&node.dir.join(path))?;
            }
            Ok(paths)
        });
        let paths = match checked {
            Ok(paths) if known.paths == paths => return,
            Ok(paths) => paths,
            Err(why) => {
                self.frame_events.push(Event::Rejected {
                    path: node.dir.clone(),
                    error: Diagnostic::new(format!("fields node `{}`: {why}", node.node)),
                });
                return;
            }
        };
        known.files.retain(|path, _| paths.contains(path));
        for path in &paths {
            if !known.files.contains_key(path) {
                let file = node.dir.join(path);
                let stamp = store.stamp(&file);
                known.files.insert(
                    path.clone(),
                    Chunk::Coming(ask(store, &file, background), stamp),
                );
            }
        }
        known.paths = paths;
        known.changed = true;
        // What is in already (all of it, where nothing is read off the
        // frames) is held for the next evaluation.
        let mut events = Vec::new();
        self.settle_field_node(node, host, &mut events);
        self.frame_events.append(&mut events);
    }

    /// Forgets what the frames gave the font and files nodes, as a scene
    /// just opened holds them: each font node the glyphs of no codes; each
    /// files node no paths, its directory listed again and its `opened`
    /// empty at the next sync; each fields node no paths.
    pub(crate) fn start_fed_over(&mut self, files: &entelechy_scene::Files, host: &mut Host) {
        self.file_nodes.clear();
        self.field_nodes.clear();
        for font in &files.fonts {
            let Some(known) = self.fonts.get_mut(&font.node) else {
                continue;
            };
            known.seen = None;
            if known.codes.is_empty() {
                continue;
            }
            known.codes.clear();
            if let Err(why) = self.give_glyphs(font, host) {
                self.frame_events.push(Event::Rejected {
                    path: font.path.clone(),
                    error: Diagnostic::new(format!("font node `{}`: {why}", font.node)),
                });
            }
        }
    }

    /// Whether files node `node` last read its paths from `handle`.
    pub(crate) fn file_node_saw(&self, node: &entelechy_scene::FilesNode, handle: Handle) -> bool {
        self.file_nodes
            .get(&node.node)
            .is_none_or(|known| known.seen == Some(handle))
    }

    /// Gives files node `node` the paths `paths` spell (read from the Field
    /// `handle`): the files they name read again when they differ from the
    /// last; what fails is reported by the next sync.
    pub(crate) fn give_paths(
        &mut self,
        store: &dyn Store,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        node: &entelechy_scene::FilesNode,
        handle: Handle,
        paths: Result<Vec<String>, String>,
        host: &mut Host,
    ) {
        let Some(known) = self.file_nodes.get_mut(&node.node) else {
            return;
        };
        known.seen = Some(handle);
        let read = match paths {
            Ok(paths) if known.paths == paths => return,
            Ok(paths) => self.read_opened(store, readable, node, &paths, host),
            Err(why) => Err(why),
        };
        if let Err(why) = read {
            self.frame_events.push(Event::Rejected {
                path: node.dir.clone(),
                error: Diagnostic::new(format!("files node `{}`: {why}", node.node)),
            });
        }
    }

    /// Reads the files `paths` names under files node `node`'s directory
    /// and holds them as its `opened`, in the order named; a path naming no
    /// file is left out. Refused whole when a path is not one under the
    /// directory, is not readable, or the files are past the budget.
    fn read_opened(
        &mut self,
        store: &dyn Store,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        node: &entelechy_scene::FilesNode,
        paths: &[String],
        host: &mut Host,
    ) -> Result<(), String> {
        let mut room = room(host);
        let mut read = Vec::with_capacity(paths.len());
        let mut opened = Vec::with_capacity(paths.len());
        for path in paths {
            entelechy_formats::directory::checked_path(path)?;
            let file = node.dir.join(path);
            readable(&file)?;
            let stamp = store.stamp(&file);
            let bytes = match store.read(&file) {
                Ok(bytes) => bytes,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(format!("{path}: {error}")),
            };
            room = room
                .checked_sub(bytes.len() + path.len() + 2)
                .ok_or_else(|| {
                    format!(
                        "the files named under {} are past the memory budget",
                        node.dir.display()
                    )
                })?;
            opened.push((path.clone(), stamp));
            read.push((path.clone(), bytes));
        }
        self.hold_output(
            node,
            "opened",
            entelechy_formats::directory::pack(&read),
            host,
        )?;
        let known = self
            .file_nodes
            .get_mut(&node.node)
            .expect("a files node gives paths");
        known.paths = paths.to_vec();
        known.opened = opened;
        Ok(())
    }

    /// Holds `data` as files node `node`'s output `port`.
    fn hold_output(
        &mut self,
        node: &entelechy_scene::FilesNode,
        port: &str,
        data: Result<FieldData, String>,
        host: &mut Host,
    ) -> Result<(), String> {
        let Some((_, external)) = node.outputs.iter().find(|(name, _)| name == port) else {
            return Ok(());
        };
        let handle = host.adopt(data?).map_err(|e| e.to_string())?;
        self.hold(host, external, handle);
        Ok(())
    }

    /// Whether font node `font` was last given `handle`'s characters.
    pub(crate) fn font_saw(&self, font: &entelechy_scene::FontNode, handle: Handle) -> bool {
        self.fonts
            .get(&font.node)
            .is_none_or(|known| known.seen == Some(handle))
    }

    /// Gives font node `font` the characters `codes` (read from the Field
    /// `handle`): its glyphs again when they differ from the last; what
    /// fails is reported by the next sync.
    pub(crate) fn give_codes(
        &mut self,
        font: &entelechy_scene::FontNode,
        handle: Handle,
        codes: Result<Vec<i32>, String>,
        host: &mut Host,
    ) {
        let Some(known) = self.fonts.get_mut(&font.node) else {
            return;
        };
        known.seen = Some(handle);
        let given = match codes {
            Ok(codes) if known.codes == codes => return,
            Ok(codes) => {
                known.codes = codes;
                self.give_glyphs(font, host)
            }
            Err(why) => Err(why),
        };
        if let Err(why) = given {
            self.frame_events.push(Event::Rejected {
                path: font.path.clone(),
                error: Diagnostic::new(format!("font node `{}`: {why}", font.node)),
            });
        }
    }

    /// Holds font node `font`'s three Fields, of the characters it was
    /// last given.
    fn give_glyphs(
        &mut self,
        font: &entelechy_scene::FontNode,
        host: &mut Host,
    ) -> Result<(), String> {
        let Some(known) = self.fonts.get(&font.node) else {
            return Ok(());
        };
        let fields = entelechy_import::font::glyphs_of(&known.bytes, &known.codes)?;
        self.hold_fields(font, fields, host)
    }

    /// Holds `fields` as font node `font`'s three outputs.
    fn hold_fields(
        &mut self,
        font: &entelechy_scene::FontNode,
        fields: entelechy_import::font::Fields,
        host: &mut Host,
    ) -> Result<(), String> {
        for (port, data) in [
            ("glyphs", fields.glyphs),
            ("outlines", fields.segments),
            ("metrics", fields.metrics),
        ] {
            let Some((_, external)) = font.outputs.iter().find(|(name, _)| name == port) else {
                continue;
            };
            let handle = host.adopt(data).map_err(|e| e.to_string())?;
            self.hold(host, external, handle);
        }
        Ok(())
    }

    /// Frame failures since the last call.
    pub(crate) fn take_frame_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.frame_events)
    }

    /// Reads new or changed input files; forgets undeclared ones.
    fn sync_assets(
        &mut self,
        store: &dyn Store,
        declared: &Declared,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
        on_device: &mut Vec<Handle>,
    ) {
        let Declared {
            graph,
            files,
            tables,
        } = *declared;
        let wanted: Vec<entelechy_scene::Asset> = files.assets.clone();
        let stale: Vec<String> = self
            .assets
            .keys()
            .filter(|name| !wanted.iter().any(|asset| &asset.name == *name))
            .cloned()
            .collect();
        for name in stale {
            self.assets.remove(&name);
            self.forget(runtime.host_mut(), &name);
        }
        for entelechy_scene::Asset {
            name,
            path,
            ty,
            bytes,
        } in wanted
        {
            // Through a package's name, the file inside it.
            let path = crate::sources::file_at(store, &path);
            let current = store.stamp(&path);
            let table = entelechy_formats::table::is_table(&entelechy_formats::extension(&path))
                .then(|| tables.get(&name).cloned())
                .flatten();
            if self.assets.get(&name).is_some_and(|asset| {
                asset.path == path
                    && asset.stamp == current
                    && asset.table == table
                    && asset
                        .named
                        .iter()
                        .all(|(named, stamp)| store.stamp(named) == *stamp)
            }) {
                continue;
            }
            self.assets.insert(
                name.clone(),
                Asset {
                    path: path.clone(),
                    stamp: current,
                    table: table.clone(),
                    named: Vec::new(),
                },
            );
            // Checked at read time, so a refusal reports once per change.
            if let Err(why) = readable(&path) {
                self.forget(runtime.host_mut(), &name);
                events.push(Event::Rejected {
                    path,
                    error: Diagnostic::new(why),
                });
                continue;
            }
            let extension = entelechy_formats::extension(&path);
            let noted = std::cell::RefCell::new(Vec::new());
            let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
            let tables = |table: &str| table_names(store, readable, &dir, table, &noted);
            let read = match bytes {
                false if extension == "field" => read_field(store, &path, room(runtime.host())),
                _ => {
                    read_within(store, &path, room(runtime.host())).and_then(|found| match bytes {
                        true => entelechy_formats::decode_bytes(&found),
                        false if entelechy_formats::table::is_table(&extension) => read_table(
                            &found,
                            &extension,
                            &name,
                            table.as_ref(),
                            room(runtime.host()),
                            &tables,
                        ),
                        false => entelechy_formats::decode(
                            &found,
                            &path,
                            Some(&ty),
                            room(runtime.host()),
                        ),
                    })
                }
            };
            if let Some(asset) = self.assets.get_mut(&name) {
                asset.named = noted.into_inner();
            }
            let read = read.and_then(|(data, detail)| {
                let data = kept_as(data, &ty)?;
                let detail = match data.schema().uniform_encoding() {
                    Some(encoding) if !data.schema().is_wide() => {
                        format!("{detail}, kept as {encoding}")
                    }
                    _ => detail,
                };
                Ok((data, detail))
            });
            let (data, detail) = match read {
                Ok(loaded) => loaded,
                // A missing file with a fallback is absent, not failed.
                Err(_)
                    if !store.exists(&path)
                        && graph.is_some_and(|g| g.fallback(&name).is_some()) =>
                {
                    self.forget(runtime.host_mut(), &name);
                    events.push(Event::Absent { name, path });
                    continue;
                }
                Err(error) => {
                    events.push(Event::Rejected {
                        path: path.clone(),
                        error: Diagnostic::new(error.to_string()),
                    });
                    continue;
                }
            };
            match runtime.host_mut().adopt(data) {
                Ok(handle) => {
                    self.hold(runtime.host_mut(), &name, handle);
                    if put_on_device(runtime, handle) {
                        on_device.push(handle);
                    }
                    events.push(Event::Loaded { name, path, detail });
                }
                Err(error) => events.push(Event::Rejected {
                    path,
                    error: Diagnostic::new(error.to_string()),
                }),
            }
        }
    }

    /// Relists each `frames` directory and forgets undeclared ones. A
    /// changed listing resets; the due frame loads at the next evaluation.
    fn sync_sequences(
        &mut self,
        store: &dyn Store,
        files: &entelechy_scene::Files,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        host: &mut Host,
        events: &mut Vec<Event>,
    ) {
        let wanted: Vec<(String, PathBuf, String, u32)> = files.sequences.clone();
        let stale: Vec<String> = self
            .sequences
            .keys()
            .filter(|name| !wanted.iter().any(|(wanted, ..)| wanted == *name))
            .cloned()
            .collect();
        for name in stale {
            self.sequences.remove(&name);
            self.forget(host, &name);
        }
        for (name, dir, ty, rate) in wanted {
            let files = match readable(&dir).and_then(|()| frame_files(store, &dir)) {
                Ok(files) => files,
                Err(error) => {
                    if self.sequences.remove(&name).is_some() {
                        self.forget(host, &name);
                    }
                    events.push(Event::Rejected {
                        path: dir,
                        error: Diagnostic::new(error.to_string()),
                    });
                    continue;
                }
            };
            if self.sequences.get(&name).is_some_and(|sequence| {
                sequence.dir == dir
                    && sequence.ty == ty
                    && sequence.rate == rate
                    && sequence.files == files
            }) {
                continue;
            }
            if files.is_empty() {
                events.push(Event::Rejected {
                    path: dir.clone(),
                    error: Diagnostic::new(
                        "holds no frame: a .png, .ppm, .pgm, .field or .wav file",
                    ),
                });
            } else {
                events.push(Event::Loaded {
                    name: name.clone(),
                    path: dir.clone(),
                    detail: format!("{} frame(s) at {rate} a second", files.len()),
                });
            }
            self.sequences.insert(
                name,
                Sequence {
                    dir,
                    ty,
                    rate,
                    files,
                    current: None,
                },
            );
        }
    }

    /// Loads frame `floor(time * rate)`, wrapping, for each `frames`
    /// input. An unreadable frame is reported by the next sync and the
    /// input keeps its frame.
    pub(crate) fn select_frames(&mut self, store: &dyn Store, host: &mut Host, time: f64) {
        let names: Vec<String> = self.sequences.keys().cloned().collect();
        for name in names {
            let Some(sequence) = self.sequences.get(&name) else {
                continue;
            };
            if sequence.files.is_empty() {
                continue;
            }
            let count = sequence.files.len() as i64;
            let index =
                ((time * f64::from(sequence.rate)).floor() as i64).rem_euclid(count) as usize;
            if sequence.current == Some(index) {
                continue;
            }
            let path = sequence.files[index].clone();
            let ty = sequence.ty.clone();
            let loaded = read_within(store, &path, room(host))
                .and_then(|found| entelechy_formats::decode(&found, &path, Some(&ty), room(host)))
                .and_then(|(data, _)| {
                    let data = kept_as(data, &ty)?;
                    host.adopt(data).map_err(|e| e.to_string())
                });
            match loaded {
                Ok(handle) => {
                    self.hold(host, &name, handle);
                    if let Some(sequence) = self.sequences.get_mut(&name) {
                        sequence.current = Some(index);
                    }
                }
                Err(error) => self.frame_events.push(Event::Rejected {
                    path,
                    error: Diagnostic::new(error.to_string()),
                }),
            }
        }
    }

    /// Reads each `checks` and `outline` input's documents again when one
    /// of them, or which there are, changed, and holds what was found as
    /// text ([`crate::check::problems_text`],
    /// [`crate::check::outline_text`]); forgets undeclared ones.
    fn sync_checks(
        &mut self,
        store: &std::sync::Arc<dyn Store>,
        files: &entelechy_scene::Files,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        host: &mut Host,
        events: &mut Vec<Event>,
    ) {
        let stale: Vec<String> = self
            .checks
            .keys()
            .filter(|name| {
                !files
                    .checks
                    .iter()
                    .chain(&files.outlines)
                    .any(|(wanted, _)| wanted == *name)
            })
            .cloned()
            .collect();
        for name in stale {
            self.checks.remove(&name);
            self.forget(host, &name);
        }
        let wanted = files
            .checks
            .iter()
            .map(|(name, path)| (name, path, false))
            .chain(files.outlines.iter().map(|(name, path)| (name, path, true)));
        for (name, path, outline) in wanted {
            // A check made off the frames, done, is taken in first.
            let finished = self.checks.get_mut(name).and_then(|checks| {
                let running = checks.running.as_ref()?;
                match running.text.try_recv() {
                    Ok(text) => Some(Some((text, checks.running.take()?.documents))),
                    Err(std::sync::mpsc::TryRecvError::Empty) => None,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        checks.running = None;
                        Some(None)
                    }
                }
            });
            if let Some(Some((text, documents))) = finished {
                let checks = self.checks.get_mut(name).expect("a check ran for it");
                checks.documents = documents.clone();
                self.take_check(host, name, path, outline, &text, documents.len(), events);
            }
            let documents = documents_under(store.as_ref(), path);
            let stamps: Vec<(PathBuf, Option<Stamp>)> = documents
                .iter()
                .map(|document| (document.clone(), store.stamp(document)))
                .collect();
            let held = self
                .checks
                .get(name)
                .filter(|checks| checks.path == *path && checks.outline == outline);
            // As they are, or being checked as they are.
            if held.is_some_and(|checks| {
                checks.documents == stamps
                    || checks
                        .running
                        .as_ref()
                        .is_some_and(|running| running.documents == stamps)
            }) {
                continue;
            }
            if let Some(why) = std::iter::once(path.as_path())
                .chain(documents.iter().map(PathBuf::as_path))
                .find_map(|path| readable(path).err())
            {
                self.checks.insert(
                    name.clone(),
                    Checks {
                        path: path.clone(),
                        outline,
                        documents: stamps,
                        running: None,
                    },
                );
                self.forget(host, name);
                events.push(Event::Rejected {
                    path: path.clone(),
                    error: Diagnostic::new(why),
                });
                continue;
            }
            // Off the frames once the input holds a check (the first is
            // taken in at once, so the scene never reads it absent).
            if self.background && held.is_some() && self.holds(name) {
                let (answer, text) = std::sync::mpsc::channel();
                let (store, root) = (std::sync::Arc::clone(store), path.clone());
                std::thread::spawn(move || {
                    let _ = answer.send(checked_text(store.as_ref(), &root, &documents, outline));
                });
                self.checks.get_mut(name).expect("held").running = Some(Running {
                    documents: stamps,
                    text,
                });
                continue;
            }
            let text = checked_text(store.as_ref(), path, &documents, outline);
            self.checks.insert(
                name.clone(),
                Checks {
                    path: path.clone(),
                    outline,
                    documents: stamps,
                    running: None,
                },
            );
            self.take_check(host, name, path, outline, &text, documents.len(), events);
        }
    }

    /// Holds a check's text as input `name`, said loaded (or rejected).
    #[allow(clippy::too_many_arguments)]
    fn take_check(
        &mut self,
        host: &mut Host,
        name: &str,
        path: &Path,
        outline: bool,
        text: &str,
        documents: usize,
        events: &mut Vec<Event>,
    ) {
        let lines = text.lines().filter(|line| !line.is_empty()).count();
        let found = if outline {
            "definition(s)"
        } else {
            "problem(s)"
        };
        let read = entelechy_formats::decode_bytes(text.as_bytes())
            .and_then(|(data, _)| host.adopt(data).map_err(|error| error.to_string()));
        match read {
            Ok(handle) => {
                self.hold(host, name, handle);
                events.push(Event::Loaded {
                    name: name.to_owned(),
                    path: path.to_path_buf(),
                    detail: format!("{lines} {found} in {documents} document(s)"),
                });
            }
            Err(error) => events.push(Event::Rejected {
                path: path.to_path_buf(),
                error: Diagnostic::new(error),
            }),
        }
    }

    /// Reads each `files` input's directory again when one of its files,
    /// or which there are, changed, and holds them packed
    /// ([`entelechy_formats::directory::pack`]); forgets undeclared ones.
    fn sync_files(
        &mut self,
        store: &dyn Store,
        files: &entelechy_scene::Files,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        host: &mut Host,
        events: &mut Vec<Event>,
    ) {
        let stale: Vec<String> = self
            .listed
            .keys()
            .filter(|name| !files.files.iter().any(|(wanted, _)| wanted == *name))
            .cloned()
            .collect();
        for name in stale {
            self.listed.remove(&name);
            self.forget(host, &name);
        }
        for (name, dir) in &files.files {
            let paths = files_under(store, dir);
            let stamps: Vec<(String, Option<Stamp>)> = paths
                .iter()
                .map(|path| (path.clone(), store.stamp(&dir.join(path))))
                .collect();
            if self
                .listed
                .get(name)
                .is_some_and(|listed| listed.dir == *dir && listed.files == stamps)
            {
                continue;
            }
            self.listed.insert(
                name.clone(),
                Listed {
                    dir: dir.clone(),
                    files: stamps,
                },
            );
            let read = std::iter::once(dir.clone())
                .chain(paths.iter().map(|path| dir.join(path)))
                .find_map(|path| readable(&path).err())
                .map_or_else(
                    || {
                        let mut room = room(host);
                        let mut read = Vec::with_capacity(paths.len());
                        for path in &paths {
                            let bytes = store
                                .read(&dir.join(path))
                                .map_err(|e| format!("{path}: {e}"))?;
                            room = room.checked_sub(bytes.len() + path.len() + 2).ok_or_else(
                                || {
                                    format!(
                                        "the files under {} are past the memory budget",
                                        dir.display()
                                    )
                                },
                            )?;
                            read.push((path.clone(), bytes));
                        }
                        Ok(read)
                    },
                    Err,
                )
                .and_then(|read| {
                    let size: usize = read.iter().map(|(_, bytes)| bytes.len()).sum();
                    let data = entelechy_formats::directory::pack(&read)?;
                    let handle = host.adopt(data).map_err(|error| error.to_string())?;
                    Ok((handle, size))
                });
            match read {
                Ok((handle, size)) => {
                    self.hold(host, name, handle);
                    events.push(Event::Loaded {
                        name: name.clone(),
                        path: dir.clone(),
                        detail: format!("{} file(s), {size} byte(s)", paths.len()),
                    });
                }
                Err(error) => {
                    self.forget(host, name);
                    events.push(Event::Rejected {
                        path: dir.clone(),
                        error: Diagnostic::new(error),
                    });
                }
            }
        }
    }

    /// Writes new or changed texts as `Rows<Int>` of character codes;
    /// forgets undeclared ones.
    fn sync_texts(&mut self, graph: Option<&Graph>, host: &mut Host, events: &mut Vec<Event>) {
        let wanted: Vec<(String, String)> =
            graph.map(|graph| graph.texts.clone()).unwrap_or_default();
        let stale: Vec<String> = self
            .texts
            .keys()
            .filter(|name| !wanted.iter().any(|(wanted, _)| wanted == *name))
            .cloned()
            .collect();
        for name in stale {
            self.texts.remove(&name);
            self.forget(host, &name);
        }
        for (name, text) in wanted {
            if self.texts.get(&name) == Some(&text) {
                continue;
            }
            let words: Vec<u32> = text.chars().map(|c| c as u32).collect();
            let chars = words.len();
            let data = FieldData::from_storage(
                entelechy_kernel::Schema::new(vec![entelechy_kernel::Scalar::Int]),
                entelechy_kernel::Domain::Rows(chars as u32),
                words,
            );
            match data.and_then(|data| host.adopt(data)) {
                Ok(handle) => {
                    self.hold(host, &name, handle);
                    self.texts.insert(name.clone(), text);
                    events.push(Event::Text { name, chars });
                }
                Err(error) => events.push(Event::SceneRejected(Diagnostic::new(format!(
                    "input `{name}`'s text: {error}"
                )))),
            }
        }
    }
}

/// A decoded Field re-encoded as the input's type says: an input
/// declared `Grid<Vec4 as unorm8>` holds bytes.
fn kept_as(data: FieldData, ty: &str) -> Result<FieldData, String> {
    let Some(name) = ty
        .split_once(" as ")
        .and_then(|(_, tail)| tail.split('>').next())
    else {
        return Ok(data);
    };
    let encoding = entelechy_kernel::Encoding::parse(name.trim())
        .ok_or_else(|| format!("`{name}` is not an encoding"))?;
    let schema = data
        .schema()
        .with_encoding(encoding)
        .map_err(|why| format!("what the file holds ({}) {why}", data.schema().describe()))?;
    FieldData::from_values(schema, data.domain(), &data.values()).map_err(|e| e.to_string())
}

/// What a `checks` input holds of `documents` under `root`: what the
/// language finds wrong in them, or what they define (`outline`).
fn checked_text(store: &dyn Store, root: &Path, documents: &[PathBuf], outline: bool) -> String {
    if outline {
        crate::check::outline_text(store, root, documents)
    } else {
        crate::check::problems_text(store, root, documents)
    }
}
