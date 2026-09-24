//! The Fields a scene's inputs hold: a file's, read again when it
//! changes; a frame of a directory, chosen by the time; a text, written
//! as its characters' codes; or what the caller bound. These own their
//! handles in the Host and nothing else, and every handle they replace
//! or forget is released there.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use orla_graph::Graph;
use orla_runtime::{FieldData, Handle, Host};

use crate::stamp::{Stamp, stamp};
use crate::{Diagnostic, Event};

/// A file an input is read from.
struct Asset {
    path: PathBuf,
    stamp: Option<Stamp>,
}

/// An input read frame by frame from a directory: its files in name
/// order, and which of them the Host holds now.
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
fn frame_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
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
    /// The Field each input holds now, by name.
    handles: BTreeMap<String, Handle>,
    /// Inputs read from files, with the stamp each was last read at.
    assets: BTreeMap<String, Asset>,
    /// Inputs read frame by frame from directories.
    sequences: BTreeMap<String, Sequence>,
    /// Inputs written as text, with the text each holds.
    texts: BTreeMap<String, String>,
    /// What went wrong selecting a frame, reported by the next sync.
    frame_events: Vec<Event>,
    /// Whether reads are held to [`Inputs::readable`]: where an agent
    /// composes the scene and reads back what it holds.
    restricted: bool,
    /// Directories the person running the session allowed reads from,
    /// resolved.
    allowed: Vec<PathBuf>,
    /// The directories the scene names ([`crate::scene::Scene::read_from`]),
    /// resolved: each document reads beside itself, and a scene reads in
    /// what it uses.
    beside: Vec<PathBuf>,
}

impl Inputs {
    /// Holds reads to the documents' own directories and the ones
    /// [`Inputs::allow_reads`] allowed.
    pub(crate) fn restrict_reads(&mut self) {
        self.restricted = true;
    }

    /// Allows files in `dir` and under it to be read into inputs.
    pub(crate) fn allow_reads(&mut self, dir: &Path) -> Result<(), String> {
        let resolved = dir
            .canonicalize()
            .map_err(|e| format!("cannot allow reads from {}: {e}", dir.display()))?;
        if !resolved.is_dir() {
            return Err(format!(
                "cannot allow reads from {}: not a directory",
                dir.display()
            ));
        }
        self.allowed.push(resolved);
        Ok(())
    }

    /// Whether `path` may be read: it is inside an allowed directory once
    /// `..` and links are followed, so no spelling reaches past one.
    fn readable(&self, path: &Path) -> Result<(), String> {
        if !self.restricted {
            return Ok(());
        }
        let target = crate::writes::resolve(path)?;
        if crate::sources::in_a_package(&target)
            || self
                .beside
                .iter()
                .chain(&self.allowed)
                .any(|dir| target.starts_with(dir))
        {
            return Ok(());
        }
        Err(format!(
            "{} is outside the directories the scene's documents are in or use, every package, and every directory allowed with `--allow file.read`",
            target.display()
        ))
    }

    /// The Field each input holds now, by name.
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

    /// Brings every input in line with what `graph` declares: files that
    /// are new or changed are read, directories listed, texts written,
    /// and inputs the scene no longer declares forgotten.
    pub(crate) fn sync(
        &mut self,
        graph: Option<&Graph>,
        files: &orla_scene::Files,
        read_from: &[PathBuf],
        host: &mut Host,
        events: &mut Vec<Event>,
    ) {
        self.beside = read_from
            .iter()
            .filter_map(|dir| dir.canonicalize().ok())
            .collect();
        self.sync_assets(graph, files, host, events);
        self.sync_sequences(files, host, events);
        self.sync_texts(graph, host, events);
    }

    /// What went wrong bringing in a frame since the last call.
    pub(crate) fn take_frame_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.frame_events)
    }

    /// Reads every input's file that is new or changed since it was last
    /// read, and forgets inputs the scene no longer declares.
    fn sync_assets(
        &mut self,
        graph: Option<&Graph>,
        files: &orla_scene::Files,
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
            let current = stamp(&path);
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
            // Asked when the file is read, so a refusal is said once until
            // the file or the scene changes.
            if let Err(why) = self.readable(&path) {
                self.forget(host, &name);
                events.push(Event::Rejected {
                    path,
                    error: Diagnostic::new(why),
                });
                continue;
            }
            let read = if bytes {
                orla_formats::read_bytes(&path)
            } else {
                orla_formats::read(&path, Some(&ty))
            };
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
                // An input with somewhere else to read reads it while its
                // file is not there: that is a state, not a failure.
                Err(_) if !path.exists() && graph.is_some_and(|g| g.fallback(&name).is_some()) => {
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

    /// Lists every `frames` input's directory anew, and forgets sequences
    /// the scene no longer declares. A listing that changed starts over:
    /// the frame whose turn it is comes in at the next evaluation.
    fn sync_sequences(
        &mut self,
        files: &orla_scene::Files,
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
            let files = match self.readable(&dir).and_then(|()| frame_files(&dir)) {
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

    /// Brings in the frame whose turn it is at `time` for every `frames`
    /// input: frame `floor(time * rate)`, wrapping at the end. A frame
    /// that cannot be read is reported by the next sync and the input
    /// keeps the frame it had.
    pub(crate) fn select_frames(&mut self, host: &mut Host, time: f32) {
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
            let loaded = orla_formats::read(&path, Some(&ty)).and_then(|(data, _)| {
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

    /// Writes every input's text that is new or changed into the Host as
    /// `Rows<Int>` of its characters' codes, and forgets texts the scene no
    /// longer declares.
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
                orla_runtime::Schema::new(vec![orla_runtime::Scalar::Int]),
                orla_runtime::Domain::Rows(chars as u32),
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

/// A Field read from a file, kept as the input's type says: what a
/// file decodes to is whole words, and an input declared
/// `Grid<Vec4 as unorm8>` keeps it as bytes from the moment it lands.
fn kept_as(data: FieldData, ty: &str) -> Result<FieldData, String> {
    let Some(name) = ty
        .split_once(" as ")
        .and_then(|(_, tail)| tail.split('>').next())
    else {
        return Ok(data);
    };
    let encoding = orla_runtime::Encoding::parse(name.trim())
        .ok_or_else(|| format!("`{name}` is not an encoding"))?;
    let schema = data
        .schema()
        .with_encoding(encoding)
        .map_err(|why| format!("what the file holds ({}) {why}", data.schema().describe()))?;
    FieldData::from_values(schema, data.domain(), &data.values()).map_err(|e| e.to_string())
}
