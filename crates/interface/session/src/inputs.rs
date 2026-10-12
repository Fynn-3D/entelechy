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

/// A `checks` input: the file or directory it checks, and each document
/// found there with its stamp when it was last checked.
struct Checks {
    path: PathBuf,
    documents: Vec<(PathBuf, Option<Stamp>)>,
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
    let mut want = entelechy_formats::file::HEADER_BYTES;
    let start = loop {
        let start = store.read_start(path, want).map_err(|e| e.to_string())?;
        if start.len() < want || entelechy_formats::header_whole(&start, path) {
            break start;
        }
        want = want.saturating_mul(2);
    };
    entelechy_formats::fits(&start, path, room)?;
    store.read(path).map_err(|e| e.to_string())
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
    /// Text inputs with their text.
    texts: BTreeMap<String, String>,
    /// Frame selection failures, reported by the next sync.
    frame_events: Vec<Event>,
}

impl Inputs {
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
    }

    /// Makes `handle` what input `name` holds, releasing what it held.
    pub(crate) fn hold(&mut self, host: &mut Host, name: &str, handle: Handle) {
        if let Some(previous) = self.handles.insert(name.to_owned(), handle) {
            let _ = host.release(previous);
        }
    }

    fn forget(&mut self, host: &mut Host, name: &str) {
        if let Some(handle) = self.handles.remove(name) {
            let _ = host.release(handle);
        }
    }

    /// Syncs inputs to what the scene declares: reads new or changed
    /// files, lists directories, writes texts, forgets undeclared inputs.
    pub(crate) fn sync(
        &mut self,
        store: &dyn Store,
        declared: &Declared,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        host: &mut Host,
        events: &mut Vec<Event>,
    ) {
        self.sync_assets(store, declared, readable, host, events);
        self.sync_sequences(store, declared.files, readable, host, events);
        self.sync_checks(store, declared.files, readable, host, events);
        self.sync_texts(declared.graph, host, events);
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
        host: &mut Host,
        events: &mut Vec<Event>,
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
            self.forget(host, &name);
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
                self.forget(host, &name);
                events.push(Event::Rejected {
                    path,
                    error: Diagnostic::new(why),
                });
                continue;
            }
            let read = read_within(store, &path, room(host));
            let extension = entelechy_formats::extension(&path);
            let noted = std::cell::RefCell::new(Vec::new());
            let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
            let tables = |table: &str| table_names(store, readable, &dir, table, &noted);
            let read = read.and_then(|found| match bytes {
                true => entelechy_formats::decode_bytes(&found),
                false if entelechy_formats::table::is_table(&extension) => read_table(
                    &found,
                    &extension,
                    &name,
                    table.as_ref(),
                    room(host),
                    &tables,
                ),
                false => entelechy_formats::decode(&found, &path, Some(&ty), room(host)),
            });
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
                    self.forget(host, &name);
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
            match host.adopt(data) {
                Ok(handle) => {
                    self.hold(host, &name, handle);
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
    pub(crate) fn select_frames(&mut self, store: &dyn Store, host: &mut Host, time: f32) {
        let names: Vec<String> = self.sequences.keys().cloned().collect();
        for name in names {
            let Some(sequence) = self.sequences.get(&name) else {
                continue;
            };
            if sequence.files.is_empty() {
                continue;
            }
            let count = sequence.files.len() as i64;
            let index = ((time * sequence.rate as f32).floor() as i64).rem_euclid(count) as usize;
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

    /// Checks each `checks` input's documents again when one of them, or
    /// which there are, changed, and holds what was found as text
    /// ([`crate::check::problems_text`]); forgets undeclared ones.
    fn sync_checks(
        &mut self,
        store: &dyn Store,
        files: &entelechy_scene::Files,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        host: &mut Host,
        events: &mut Vec<Event>,
    ) {
        let stale: Vec<String> = self
            .checks
            .keys()
            .filter(|name| !files.checks.iter().any(|(wanted, _)| wanted == *name))
            .cloned()
            .collect();
        for name in stale {
            self.checks.remove(&name);
            self.forget(host, &name);
        }
        for (name, path) in &files.checks {
            let documents = documents_under(store, path);
            let stamps: Vec<(PathBuf, Option<Stamp>)> = documents
                .iter()
                .map(|document| (document.clone(), store.stamp(document)))
                .collect();
            if self
                .checks
                .get(name)
                .is_some_and(|checks| checks.path == *path && checks.documents == stamps)
            {
                continue;
            }
            self.checks.insert(
                name.clone(),
                Checks {
                    path: path.clone(),
                    documents: stamps,
                },
            );
            if let Some(why) = std::iter::once(path.as_path())
                .chain(documents.iter().map(PathBuf::as_path))
                .find_map(|path| readable(path).err())
            {
                self.forget(host, name);
                events.push(Event::Rejected {
                    path: path.clone(),
                    error: Diagnostic::new(why),
                });
                continue;
            }
            let text = crate::check::problems_text(store, path, &documents);
            let problems = text.lines().filter(|line| !line.is_empty()).count();
            let read = entelechy_formats::decode_bytes(text.as_bytes())
                .and_then(|(data, _)| host.adopt(data).map_err(|error| error.to_string()));
            match read {
                Ok(handle) => {
                    self.hold(host, name, handle);
                    events.push(Event::Loaded {
                        name: name.clone(),
                        path: path.clone(),
                        detail: format!("{problems} problem(s) in {} document(s)", documents.len()),
                    });
                }
                Err(error) => events.push(Event::Rejected {
                    path: path.clone(),
                    error: Diagnostic::new(error),
                }),
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
