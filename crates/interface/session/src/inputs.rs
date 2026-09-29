//! The Fields a scene's inputs hold: a file (reread on change), a
//! directory frame chosen by time, a text as character codes, or a
//! caller binding. Inputs own their Host handles and release each one
//! they replace or forget.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use orla_graph::Graph;
use orla_kernel::{FieldData, Handle, Host};

use crate::store::{Stamp, Store};
use crate::{Diagnostic, Event};

/// A file an input is read from.
struct Asset {
    path: PathBuf,
    stamp: Option<Stamp>,
}

/// A directory input: frame files in name order and the one held now.
struct Sequence {
    dir: PathBuf,
    ty: String,
    rate: u32,
    files: Vec<PathBuf>,
    current: Option<usize>,
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

/// The inputs of a scene and the Fields they hold.
#[derive(Default)]
pub(crate) struct Inputs {
    /// The Field each input holds, by name.
    handles: BTreeMap<String, Handle>,
    /// File inputs with their last-read stamps.
    assets: BTreeMap<String, Asset>,
    /// Directory inputs.
    sequences: BTreeMap<String, Sequence>,
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

    /// Syncs inputs to what `graph` declares: reads new or changed files,
    /// lists directories, writes texts, forgets undeclared inputs.
    pub(crate) fn sync(
        &mut self,
        store: &dyn Store,
        graph: Option<&Graph>,
        files: &orla_scene::Files,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        host: &mut Host,
        events: &mut Vec<Event>,
    ) {
        self.sync_assets(store, graph, files, readable, host, events);
        self.sync_sequences(store, files, readable, host, events);
        self.sync_texts(graph, host, events);
    }

    /// Frame failures since the last call.
    pub(crate) fn take_frame_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.frame_events)
    }

    /// Reads new or changed input files; forgets undeclared ones.
    fn sync_assets(
        &mut self,
        store: &dyn Store,
        graph: Option<&Graph>,
        files: &orla_scene::Files,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        host: &mut Host,
        events: &mut Vec<Event>,
    ) {
        let wanted: Vec<orla_scene::Asset> = files.assets.clone();
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
        for orla_scene::Asset {
            name,
            path,
            ty,
            bytes,
        } in wanted
        {
            let current = store.stamp(&path);
            if self
                .assets
                .get(&name)
                .is_some_and(|asset| asset.path == path && asset.stamp == current)
            {
                continue;
            }
            self.assets.insert(
                name.clone(),
                Asset {
                    path: path.clone(),
                    stamp: current,
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
            let read = store.read(&path).map_err(|e| e.to_string());
            let read = read.and_then(|found| match bytes {
                true => orla_formats::decode_bytes(&found),
                false => orla_formats::decode(&found, &path, Some(&ty)),
            });
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
        files: &orla_scene::Files,
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
            let loaded = store
                .read(&path)
                .map_err(|e| e.to_string())
                .and_then(|found| orla_formats::decode(&found, &path, Some(&ty)))
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
                orla_kernel::Schema::new(vec![orla_kernel::Scalar::Int]),
                orla_kernel::Domain::Rows(chars as u32),
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
    let encoding = orla_kernel::Encoding::parse(name.trim())
        .ok_or_else(|| format!("`{name}` is not an encoding"))?;
    let schema = data
        .schema()
        .with_encoding(encoding)
        .map_err(|why| format!("what the file holds ({}) {why}", data.schema().describe()))?;
    FieldData::from_values(schema, data.domain(), &data.values()).map_err(|e| e.to_string())
}
