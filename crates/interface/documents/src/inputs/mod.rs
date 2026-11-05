//! The Fields a scene's inputs hold: a file (reread on change), a
//! directory frame chosen by time, a text as character codes, or a
//! caller binding. Inputs own their Host handles and release each one
//! they replace or forget. A data table (`.csv`, `.json`) is read as the
//! Kernel reading the input lays out its element, and read again when
//! that layout changes.
//!
//! Each kind of input, and each node the session feeds (font, files and
//! fields nodes), keeps what it read in a module of its own; this one
//! holds the inputs' Fields and syncs them all.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use entelechy_graph::{Binding, Graph, Source};
use entelechy_kernel::{FieldData, Handle, Host, Runtime, Schema};

use crate::store::Store;
use crate::{Diagnostic, Event};

mod assets;
mod checks;
mod field_nodes;
mod file_nodes;
mod fonts;
mod listed;
mod sequences;

use assets::Asset;
use checks::Checks;
use field_nodes::FieldNode;
use file_nodes::FileNode;
pub use file_nodes::paths_spelled;
use fonts::Font;
use listed::Listed;
use sequences::Sequence;

/// The element a data table input is read as: the name of each number it
/// holds and the layout, as the Kernel reading the input lays them out;
/// and how its port keeps them, narrow where it says so (`Int as u8`).
#[derive(Clone, Debug, PartialEq)]
pub struct TableElement {
    names: Vec<String>,
    schema: Schema,
    kept: Schema,
}

impl TableElement {
    /// Whether a Field of `schema` is what the Kernel reading the input
    /// takes.
    pub fn takes(&self, schema: &Schema) -> bool {
        &self.kept == schema
    }
}

/// What a scene declares of its inputs: its graph, the files and
/// directories they are read from, and the element each data table is
/// read as ([`table_elements`]).
pub struct Declared<'a> {
    /// The scene's graph, when one is built.
    pub graph: Option<&'a Graph>,
    /// The files and directories its inputs are read from.
    pub files: &'a entelechy_scene::Files,
    /// The element each data table is read as.
    pub tables: &'a BTreeMap<String, TableElement>,
    /// Files read as absent, whatever the disk holds: what a replay's
    /// scene saved for itself and has not written in this run.
    pub unread: &'a [std::path::PathBuf],
}

/// For each scene input a node reads, the element the first node reading
/// it (in the graph's order) lays it out as; an input no published Kernel
/// reads has none.
pub fn table_elements(graph: Option<&Graph>, runtime: &Runtime) -> BTreeMap<String, TableElement> {
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
                let kept = declared.ty.schema.clone();
                found.insert(
                    name.clone(),
                    TableElement {
                        names,
                        schema,
                        kept,
                    },
                );
            }
        }
    }
    found
}

/// The fewest bytes of an input's Host copy that are let go once the
/// input is on a device.
pub const COPY_LEAST: u64 = 1 << 20;

/// The words the Host has room for, which a decoded file may take.
fn room(host: &Host) -> usize {
    host.budget_words().saturating_sub(host.used_words())
}

/// Puts an input just read on the device, when there is one and its words
/// take [`COPY_LEAST`] bytes or more, and lets
/// go of its Host copy: a load then holds one big input's words at a time
/// beside its documents, not all of them. What reads it on the processor
/// brings the words back once. Answers whether its copy went.
fn put_on_device(runtime: &mut Runtime, handle: Handle) -> bool {
    runtime.has_device()
        && runtime
            .host()
            .field_bytes(handle)
            .is_ok_and(|bytes| bytes.host >= COPY_LEAST)
        && runtime.upload(handle).unwrap_or(false)
        && runtime.host_mut().let_go_of_copy(handle).is_ok()
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

/// The inputs of a scene and the Fields they hold.
#[derive(Default)]
pub struct Inputs {
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

impl Inputs {
    /// Makes checks after an input's first off the frames: each is taken
    /// in by the first sync after it is done, the text held until then. As
    /// the session's replacements, never in a replay or a run, which take
    /// every check in at once.
    pub fn set_background(&mut self, enabled: bool) {
        self.background = enabled;
    }

    /// The Field each input holds, by name.
    pub fn handles(&self) -> &BTreeMap<String, Handle> {
        &self.handles
    }

    /// The files and directories the inputs are read from.
    pub fn watched(&self) -> impl Iterator<Item = PathBuf> + '_ {
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
    pub fn hold(&mut self, host: &mut Host, name: &str, handle: Handle) {
        if let Some(previous) = self.handles.insert(name.to_owned(), handle) {
            let _ = host.release(previous);
        }
    }

    /// Whether `name` holds a Field.
    pub fn holds(&self, name: &str) -> bool {
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
    pub fn sync(
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
        self.sync_field_nodes(store.as_ref(), declared, runtime, events);
        self.sync_texts(declared.graph, runtime.host_mut(), events);
        on_device
    }

    /// Forgets what the frames gave the font and files nodes, as a scene
    /// just opened holds them: each font node the glyphs of no codes; each
    /// files node no paths, its directory listed again and its `opened`
    /// empty at the next sync; each fields node no paths.
    pub fn start_fed_over(&mut self, files: &entelechy_scene::Files, host: &mut Host) {
        self.file_nodes.clear();
        self.field_nodes_fed_over(host);
        self.fonts_fed_over(files, host);
    }

    /// Frame failures since the last call.
    pub fn take_frame_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.frame_events)
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
