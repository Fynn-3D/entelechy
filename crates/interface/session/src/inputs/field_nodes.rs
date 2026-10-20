//! Fields nodes: the Field files a node's paths name, read as the scene
//! asks and joined where they are.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use entelechy_kernel::{FieldData, Handle, Host, Runtime, Schema};

use super::{Declared, Inputs, room};
use crate::store::{Stamp, Store};
use crate::{Diagnostic, Event};

/// A `fields` node: its directory, the paths it was last given, each file
/// they name as it stands, the Field it last read its paths from, and
/// whether what it holds changed since its outputs were last made.
pub(super) struct FieldNode {
    pub(super) dir: PathBuf,
    paths: Vec<String>,
    files: BTreeMap<String, Chunk>,
    seen: Option<Handle>,
    changed: bool,
    /// The element as the Kernel reading the result lays it out, as the
    /// last sync found it.
    element: Option<Schema>,
    /// The result it last joined, which its `result` holds until the next
    /// is joined: [`made_in`] alone makes and holds one.
    joined: Option<Handle>,
}

impl FieldNode {
    /// Lets go of the Field of every file in.
    fn let_go(self, host: &mut Host) {
        for (_, chunk) in self.files {
            chunk.let_go(host);
        }
    }
}

/// A Field file a `fields` node names, with the stamp it was asked at.
enum Chunk {
    /// On its way.
    Coming(Box<dyn crate::store::Later>, Option<Stamp>),
    /// In, laid out as the node's elements ([`FieldNode::element`]).
    In(Piece, Option<Stamp>),
    /// Not in: no such file, or one refused and reported.
    Out(Option<Stamp>),
}

/// Where the elements of a file in are.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Piece {
    /// A Field of their own, on the device where the session has one, not
    /// yet joined.
    Own(Handle),
    /// How many there are from where, in the result the node last joined:
    /// joined, a file keeps no Field of its own.
    Joined(usize, usize),
}

impl Chunk {
    fn stamp(&self) -> Option<Stamp> {
        match self {
            Self::Coming(_, stamp) | Self::In(_, stamp) | Self::Out(stamp) => *stamp,
        }
    }

    /// Lets go of the Field of a file in not yet joined.
    fn let_go(self, host: &mut Host) {
        if let Self::In(Piece::Own(handle), _) = self {
            let _ = host.release(handle);
        }
    }
}

/// The layout a fields node joins its files in: the element as the Kernel
/// reading the result lays it out, else each file's scalars a word each.
fn joined_as(element: Option<&Schema>, scalars: &[entelechy_kernel::Scalar]) -> Schema {
    element
        .cloned()
        .unwrap_or_else(|| Schema::new(scalars.to_vec()))
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
/// `element` (where a Kernel reads it): its elements laid out as the node
/// joins them ([`joined_as`]), within `room` words.
fn taken(
    arrived: &dyn crate::store::Arrived,
    element: Option<&Schema>,
    room: usize,
) -> Result<FieldData, String> {
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
    let joined = joined_as(element, data.schema().scalars());
    if *data.schema() == joined {
        return Ok(data);
    }
    FieldData::from_values(joined, domain, &data.values()).map_err(|e| e.to_string())
}

/// A fields node's outputs from what it holds, those of `ports` its
/// scene reads: `result`, the elements of the files in, in the order
/// named, joined from where they are ([`Runtime::join`]), one zero
/// element where none is; `ranges`, a row a path, where its elements start
/// and how many there are, -1 and 0 for one not in (one such row where no
/// path is named). Joined, a file's own Field is let go and its elements
/// are found in the result.
fn made_in(
    node: &mut FieldNode,
    runtime: &mut Runtime,
    ports: &[&str],
) -> Vec<(&'static str, Result<Handle, String>)> {
    let host = runtime.host_mut();
    let schema = node.element.clone().or_else(|| {
        node.files.values().find_map(|chunk| match chunk {
            Chunk::In(Piece::Own(handle), _) => host.schema(*handle).ok().cloned(),
            Chunk::In(Piece::Joined(..), _) => node
                .joined
                .and_then(|joined| host.schema(joined).ok().cloned()),
            _ => None,
        })
    });
    let mut ranges = Vec::with_capacity(node.paths.len().max(1) * 2);
    let mut parts = Vec::new();
    let mut placed = Vec::new();
    let mut start = 0usize;
    for path in &node.paths {
        let part = match node.files.get(path) {
            Some(Chunk::In(Piece::Own(handle), _)) => host
                .domain(*handle)
                .ok()
                .map(|domain| (*handle, 0..domain.count())),
            Some(Chunk::In(Piece::Joined(from, count), _)) => {
                node.joined.map(|joined| (joined, *from..from + count))
            }
            _ => None,
        };
        match part {
            Some((handle, range)) => {
                let count = range.len();
                ranges.extend([start as u32, count as u32]);
                placed.push((path.clone(), start, count));
                parts.push((handle, range));
                start += count;
            }
            None => ranges.extend([-1i32 as u32, 0]),
        }
    }
    if ranges.is_empty() {
        ranges.extend([-1i32 as u32, 0]);
    }
    let mut made = Vec::new();
    if ports.contains(&"ranges") {
        made.push((
            "ranges",
            FieldData::from_storage(
                Schema::new(vec![entelechy_kernel::Scalar::Int; 2]),
                entelechy_kernel::Domain::Rows((ranges.len() / 2) as u32),
                ranges,
            )
            .and_then(|data| host.adopt(data))
            .map_err(|e| e.to_string()),
        ));
    }
    let Some(schema) = schema.filter(|_| ports.contains(&"result")) else {
        return made;
    };
    let result = if parts.is_empty() {
        host.create(schema, entelechy_kernel::Domain::Rows(1))
            .map_err(|e| e.to_string())
    } else {
        runtime.join(&parts, &schema).map_err(|e| e.to_string())
    };
    if let Ok(result) = result {
        for (path, start, count) in placed {
            if let Some(Chunk::In(piece, _)) = node.files.get_mut(&path)
                && let Piece::Own(handle) = std::mem::replace(piece, Piece::Joined(start, count))
            {
                let _ = runtime.host_mut().release(handle);
            }
        }
        node.joined = Some(result);
    }
    made.push(("result", result));
    made
}

/// A file's elements as a Field of a fields node's: put on the device
/// where the session has one and their words join there, the Host copy
/// let go, so the node keeps none of a file's words in memory.
fn kept_in(runtime: &mut Runtime, data: FieldData) -> Result<Handle, String> {
    let whole_words = data.schema().element_bytes().is_multiple_of(4);
    let handle = runtime.host_mut().adopt(data).map_err(|e| e.to_string())?;
    if whole_words && runtime.has_device() && runtime.upload(handle).unwrap_or(false) {
        let _ = runtime.host_mut().let_go_of_copy(handle);
    }
    Ok(handle)
}

impl Inputs {
    /// The Field of each file in that a fields node holds to join, by the
    /// node and the file's path: `near: b.field`.
    pub(crate) fn chunk_handles(&self) -> impl Iterator<Item = (String, Handle)> + '_ {
        self.field_nodes.iter().flat_map(|(node, known)| {
            known
                .files
                .iter()
                .filter_map(move |(path, chunk)| match chunk {
                    Chunk::In(Piece::Own(handle), _) => Some((format!("{node}: {path}"), *handle)),
                    _ => None,
                })
        })
    }

    /// Takes in the files each fields node named that came in, asks again
    /// for those that changed since they were read, and makes its outputs
    /// again where what it holds changed; forgets undeclared ones.
    pub(super) fn sync_field_nodes(
        &mut self,
        store: &dyn Store,
        declared: &Declared,
        runtime: &mut Runtime,
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
            if let Some(known) = self.field_nodes.remove(&node) {
                known.let_go(runtime.host_mut());
            }
            for port in entelechy_scene::FIELDS_OUTPUTS {
                self.forget(runtime.host_mut(), &format!("{node}.{port}"));
            }
        }
        let background = self.background;
        for node in &files.field_nodes {
            if self
                .field_nodes
                .get(&node.node)
                .is_none_or(|known| known.dir != node.dir)
            {
                let before = self.field_nodes.insert(
                    node.node.clone(),
                    FieldNode {
                        dir: node.dir.clone(),
                        paths: Vec::new(),
                        files: BTreeMap::new(),
                        seen: None,
                        changed: true,
                        element: None,
                        joined: None,
                    },
                );
                if let Some(before) = before {
                    before.let_go(runtime.host_mut());
                }
            }
            // The element as the Kernel reading the result keeps it.
            let element = node
                .outputs
                .iter()
                .find(|(port, _)| port == "result")
                .and_then(|(_, external)| declared.tables.get(external))
                .map(|table| table.kept.clone());
            let known = self.field_nodes.get_mut(&node.node).expect("just placed");
            let relaid = known.element != element;
            if relaid {
                known.element = element;
                known.changed = true;
            }
            // A file that changed since it was asked, or whose elements are
            // laid out anew, is asked again.
            for (path, chunk) in &mut known.files {
                let file = node.dir.join(path);
                let stamp = store.stamp(&file);
                if !matches!(chunk, Chunk::Coming(..)) && (relaid || stamp != chunk.stamp()) {
                    let asked = Chunk::Coming(ask(store, &file, background), stamp);
                    std::mem::replace(chunk, asked).let_go(runtime.host_mut());
                }
            }
            self.settle_field_node(node, runtime, events);
        }
    }

    /// Takes in the files fields node `node` named that came in, and makes
    /// its outputs again where what it holds changed.
    fn settle_field_node(
        &mut self,
        node: &entelechy_scene::FieldsNode,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) {
        let room = room(runtime.host());
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
                Ok(arrived) => match taken(arrived.as_ref(), known.element.as_ref(), room)
                    .and_then(|data| kept_in(runtime, data))
                {
                    Ok(handle) => Chunk::In(Piece::Own(handle), stamp),
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
        let ports: Vec<&str> = node.outputs.iter().map(|(port, _)| port.as_str()).collect();
        let made = made_in(known, runtime, &ports);
        for (port, handle) in made {
            let held = handle.map(|handle| {
                if let Some((_, external)) = node.outputs.iter().find(|(name, _)| name == port) {
                    self.hold(runtime.host_mut(), external, handle);
                }
            });
            if let Err(why) = held {
                events.push(Event::Rejected {
                    path: node.dir.clone(),
                    error: Diagnostic::new(format!("fields node `{}`: {why}", node.node)),
                });
            }
        }
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
        runtime: &mut Runtime,
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
        let gone: Vec<String> = known
            .files
            .keys()
            .filter(|path| !paths.contains(path))
            .cloned()
            .collect();
        for path in gone {
            if let Some(chunk) = known.files.remove(&path) {
                chunk.let_go(runtime.host_mut());
            }
        }
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
        self.settle_field_node(node, runtime, &mut events);
        self.frame_events.append(&mut events);
    }

    /// Lets go of every fields node, as a scene just opened has none.
    pub(super) fn field_nodes_fed_over(&mut self, host: &mut Host) {
        for (_, known) in std::mem::take(&mut self.field_nodes) {
            known.let_go(host);
        }
    }
}
