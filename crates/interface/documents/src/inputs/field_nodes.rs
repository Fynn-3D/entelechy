//! Fields nodes: the Field or compressed texture files a node's paths name,
//! read asynchronously off the frame loop and joined where they live —
//! either linearly as `Rows<E>` or as a 2D texture array `Volume<E>`.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use entelechy_kernel::{Domain, FieldData, Handle, Host, Runtime, Schema, Shape};

use super::{Declared, Inputs, room};
use crate::store::{Stamp, Store};
use crate::{Diagnostic, Event};

/// Frame-paced streaming upload budget (`8 MiB/frame`) used when background
/// streaming is active so bursty asset arrivals never stall interactive frame
/// pacing. At least one arrived file always settles per frame.
const STREAM_UPLOAD_BYTES_PER_FRAME: usize = 8 * 1024 * 1024;

/// A `fields` node: its directory, whether it joins into a 2D texture-array
/// `Volume` (otherwise `Rows`), the paths it was last given, each file they
/// name as it stands, the Field it last read its paths from, and whether what
/// it holds changed since its outputs were last made.
pub(super) struct FieldNode {
    pub(super) dir: PathBuf,
    is_volume: bool,
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
    /// Lets go of the Field of every file in, and cancels pending reads.
    fn let_go(self, host: &mut Host) {
        for (_, chunk) in self.files {
            chunk.let_go(host);
        }
    }
}

/// A file a `fields` node names, with the stamp it was asked at.
enum Chunk {
    /// On its way from the background I/O pool.
    Coming(Box<dyn crate::store::Later>, Option<Stamp>),
    /// Arrived from disk, waiting for its frame-paced upload budget slice.
    Ready(Box<dyn crate::store::Arrived>, Option<Stamp>),
    /// In, laid out as the node's elements ([`FieldNode::element`]).
    In(Piece, Option<Stamp>),
    /// Not in: no such file, or one refused and reported.
    Out(Option<Stamp>),
}

/// Where the elements (for `Rows`) or layers (for `Volume`) of a file in are.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Piece {
    /// A Field of their own, on the device where the session has one, not
    /// yet joined.
    Own(Handle),
    /// Start index and count (elements for `Rows`, layers for `Volume`) in
    /// the result the node last joined: joined, a file keeps no Field of its
    /// own.
    Joined(usize, usize),
}

impl Chunk {
    fn stamp(&self) -> Option<Stamp> {
        match self {
            Self::Coming(_, stamp)
            | Self::Ready(_, stamp)
            | Self::In(_, stamp)
            | Self::Out(stamp) => *stamp,
        }
    }

    /// Cancels a pending read or lets go of the Field of a file not yet joined.
    fn let_go(mut self, host: &mut Host) {
        match self {
            Self::Coming(ref mut later, _) => later.cancel(),
            Self::In(Piece::Own(handle), _) => {
                let _ = host.release(handle);
            }
            Self::Ready(..) | Self::In(Piece::Joined(..), _) | Self::Out(_) => {}
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

/// Decodes an arrived file for a `fields` node whose element is `element`
/// (where a Kernel reads it), preserving block-compressed textures directly
/// without CPU decompression when their layout already matches `joined_as`.
fn taken(
    file: &Path,
    arrived: &dyn crate::store::Arrived,
    element: Option<&Schema>,
    node_ty: &str,
    is_volume: bool,
    room: usize,
) -> Result<FieldData, String> {
    let ext = entelechy_formats::extension(file);
    let data = if ext == "field" || ext.is_empty() {
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
        entelechy_formats::file::from_words(schema, domain, words, bytes)
            .map_err(|e| e.to_string())?
    } else if is_volume && matches!(ext.as_str(), "ktx2" | "dds" | "png" | "ppm" | "pgm") {
        let bytes: Cow<'_, [u8]> = match arrived.as_bytes() {
            Some(slice) => Cow::Borrowed(slice),
            None => Cow::Owned(arrived.start(arrived.len())),
        };
        entelechy_formats::fits(&bytes, file, room)?;
        let want_grid = node_ty
            .strip_prefix("Volume<")
            .map(|rest| format!("Grid<{rest}"));
        let (data, _) = entelechy_formats::decode(&bytes, file, want_grid.as_deref(), room)?;
        if let Some(element) = element
            && data.schema().scalars() != element.scalars()
        {
            return Err(format!(
                "its elements are {}, where the node's are {element}",
                data.schema()
            ));
        }
        data
    } else if is_volume {
        return Err(format!(
            "unsupported file `.{ext}` for a Volume fields node (use .ktx2, .dds, .png, .ppm, .pgm, or .field)"
        ));
    } else {
        return Err(format!(
            "unsupported file `.{ext}` for a Rows fields node (use .field)"
        ));
    };

    if is_volume && !matches!(data.domain().shape(), Shape::Grid | Shape::Volume) {
        return Err(format!(
            "a Volume fields node requires 2D Grid or Volume files, not {}",
            data.domain().shape()
        ));
    }
    let joined = joined_as(element, data.schema().scalars());
    if *data.schema() == joined {
        return Ok(data);
    }
    FieldData::from_values(joined, data.domain(), &data.values()).map_err(|e| e.to_string())
}

/// A fields node's outputs from what it holds, those of `ports` its
/// scene reads:
/// - `result`: elements of loaded files concatenated in `paths` order (`Rows`),
///   or 2D texture layers stacked along Z (`Volume` / `Texture2DArray`), joined
///   GPU-to-GPU where resident ([`Runtime::join_domain`]); one zero element (or
///   1x1x1 layer) when none are loaded.
/// - `ranges`: one `IVec2` row per path: `(start, count)` in elements (`Rows`)
///   or layers (`Volume`), `(-1, 0)` for pending/absent paths.
fn made_in(
    node: &mut FieldNode,
    node_name: &str,
    runtime: &mut Runtime,
    ports: &[&str],
    events: &mut Vec<Event>,
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
    struct RejectedLayer {
        path: String,
        own_handle: Option<Handle>,
        got: (u32, u32),
        expected: (u32, u32),
    }
    let mut ranges = Vec::with_capacity(node.paths.len().max(1) * 2);
    let mut parts = Vec::new();
    let mut placed = Vec::new();
    let mut rejected: Vec<RejectedLayer> = Vec::new();
    let mut target_dims: Option<(u32, u32)> = None;
    let mut start = 0usize;

    for path in &node.paths {
        if node.is_volume {
            let entry = match node.files.get(path) {
                Some(Chunk::In(Piece::Own(handle), _)) => host.domain(*handle).ok().map(|d| {
                    (
                        *handle,
                        (d.width(), d.height()),
                        0..d.count(),
                        d.depth() as usize,
                        true,
                    )
                }),
                Some(Chunk::In(Piece::Joined(from_layer, layer_count), _)) => {
                    node.joined.and_then(|joined| {
                        let d = host.domain(joined).ok()?;
                        let layer_elems = (d.width() as usize) * (d.height() as usize);
                        let begin = *from_layer * layer_elems;
                        let end = (*from_layer + *layer_count) * layer_elems;
                        Some((
                            joined,
                            (d.width(), d.height()),
                            begin..end,
                            *layer_count,
                            false,
                        ))
                    })
                }
                _ => None,
            };
            match entry {
                Some((handle, dims, range, layers, is_own)) => {
                    let expected = *target_dims.get_or_insert(dims);
                    if dims != expected {
                        rejected.push(RejectedLayer {
                            path: path.clone(),
                            own_handle: is_own.then_some(handle),
                            got: dims,
                            expected,
                        });
                        ranges.extend([-1i32 as u32, 0]);
                        continue;
                    }
                    ranges.extend([start as u32, layers as u32]);
                    placed.push((path.clone(), start, layers));
                    parts.push((handle, range));
                    start += layers;
                }
                None => ranges.extend([-1i32 as u32, 0]),
            }
        } else {
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
    }

    for RejectedLayer {
        path,
        own_handle,
        got: (gw, gh),
        expected: (ew, eh),
    } in rejected
    {
        if let Some(handle) = own_handle {
            let _ = host.release(handle);
        }
        if let Some(chunk) = node.files.get_mut(&path) {
            let stamp = chunk.stamp();
            *chunk = Chunk::Out(stamp);
        }
        events.push(Event::Rejected {
            path: node.dir.join(&path),
            error: Diagnostic::new(format!(
                "fields node `{node_name}`: {path}: extent {gw}x{gh} does not match the node's layer size {ew}x{eh}"
            )),
        });
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
                Domain::Rows((ranges.len() / 2) as u32),
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
        let empty_domain = if node.is_volume {
            Domain::Volume(1, 1, 1)
        } else {
            Domain::Rows(1)
        };
        host.create(schema, empty_domain).map_err(|e| e.to_string())
    } else if node.is_volume {
        let (w, h) = target_dims.expect("non-empty volume parts set target_dims");
        let layers = u32::try_from(start).unwrap_or(u32::MAX);
        runtime
            .join_domain(&parts, &schema, Domain::Volume(w, h, layers))
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

/// A file's elements as a Field of a fields node's: put on the device where
/// the session has one and its blocks or words join there, the Host copy let
/// go, so the node keeps none of a file's bytes in Host RAM.
fn kept_in(runtime: &mut Runtime, data: FieldData) -> Result<Handle, String> {
    let compressed = data.blocks().is_some();
    let whole_words = data.schema().element_bytes().is_multiple_of(4);
    let handle = runtime.host_mut().adopt(data).map_err(|e| e.to_string())?;
    if (compressed || whole_words)
        && runtime.has_device()
        && runtime.upload(handle).unwrap_or(false)
    {
        let _ = runtime.host_mut().let_go_of_copy(handle);
    }
    Ok(handle)
}

impl Inputs {
    /// The Field of each file in that a fields node holds to join, by the
    /// node and the file's path: `near: b.field`.
    pub fn chunk_handles(&self) -> impl Iterator<Item = (String, Handle)> + '_ {
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
            let is_volume = node.ty.starts_with("Volume<");
            if self
                .field_nodes
                .get(&node.node)
                .is_none_or(|known| known.dir != node.dir || known.is_volume != is_volume)
            {
                let before = self.field_nodes.insert(
                    node.node.clone(),
                    FieldNode {
                        dir: node.dir.clone(),
                        is_volume,
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

    /// Takes in the files fields node `node` named that came in (respecting
    /// the per-frame upload budget when background streaming is enabled), and
    /// makes its outputs again where what it holds changed.
    fn settle_field_node(
        &mut self,
        node: &entelechy_scene::FieldsNode,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
    ) {
        let background = self.background;
        let room = room(runtime.host());
        let Some(known) = self.field_nodes.get_mut(&node.node) else {
            return;
        };
        // Step 1: non-blockingly drain any completed background reads into
        // `Chunk::Ready` so worker channels clear promptly.
        for (path, chunk) in &mut known.files {
            let Chunk::Coming(later, stamp) = chunk else {
                continue;
            };
            let Some(read) = later.poll() else {
                continue;
            };
            let stamp = *stamp;
            let file = node.dir.join(path);
            match read {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    known.changed = true;
                    *chunk = Chunk::Out(stamp);
                }
                Err(error) => {
                    known.changed = true;
                    events.push(Event::Rejected {
                        path: file,
                        error: Diagnostic::new(format!("fields node `{}`: {error}", node.node)),
                    });
                    *chunk = Chunk::Out(stamp);
                }
                Ok(arrived) => {
                    *chunk = Chunk::Ready(arrived, stamp);
                }
            }
        }
        // Step 2: decode and upload ready chunks within the per-frame budget
        // (always settling at least one file per tick; unbounded when
        // background mode is off so replays remain strictly deterministic).
        let mut settled_bytes = 0usize;
        let mut settled_count = 0usize;
        let is_volume = known.is_volume;
        let element = known.element.clone();
        for (path, chunk) in &mut known.files {
            if !matches!(chunk, Chunk::Ready(..)) {
                continue;
            }
            if background && settled_count > 0 && settled_bytes >= STREAM_UPLOAD_BYTES_PER_FRAME {
                break;
            }
            let Chunk::Ready(arrived, stamp) = std::mem::replace(chunk, Chunk::Out(None)) else {
                unreachable!();
            };
            let file_bytes = arrived.len();
            let file = node.dir.join(path);
            known.changed = true;
            *chunk = match taken(
                &file,
                arrived.as_ref(),
                element.as_ref(),
                &node.ty,
                is_volume,
                room,
            )
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
            };
            settled_bytes = settled_bytes.saturating_add(file_bytes);
            settled_count += 1;
        }
        if !known.changed {
            return;
        }
        known.changed = false;
        let ports: Vec<&str> = node.outputs.iter().map(|(port, _)| port.as_str()).collect();
        let made = made_in(known, &node.node, runtime, &ports, events);
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
    pub fn field_node_saw(&self, node: &entelechy_scene::FieldsNode, handle: Handle) -> bool {
        self.field_nodes
            .get(&node.node)
            .is_none_or(|known| known.seen == Some(handle))
    }

    /// Gives fields node `node` the paths `paths` spell (read from the
    /// Field `handle`): a file no longer named is let go (cancelling any
    /// in-flight background read), one newly named asked for, taken in by a
    /// sync once it is in. Refused whole, the paths before staying, where a
    /// path is not one under the directory or is not readable; reported by
    /// the next sync.
    pub fn give_field_paths(
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
