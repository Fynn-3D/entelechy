//! Bundled scene playback: opens a compiled [`Bundle`] directly into a
//! [`Session`] without source documents, compiler or asset codecs, and
//! builds a [`Bundle`] from a document-backed [`Session`] at export time.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use entelechy_graph::{Binding, Graph, Source};
use entelechy_kernel::{
    Domain, Expect, FieldData, Handle, Host, KernelId, Runtime, Scalar, Schema, Ticket, Value,
};

use super::{BEST, Backing, Session, TURN};
use crate::Limits;
use crate::bundle::{Bundle, BundledFont, glyphs_from_atlas};
use crate::memory::COPY_LEAST;

/// Runtime state of a session opened from a [`Bundle`].
pub(crate) struct BundledScene {
    pub(super) graph: Option<Arc<Graph>>,
    pub(super) pending: Option<PendingBundle>,
    pub(super) params: BTreeMap<String, Value>,
    pub(super) handles: BTreeMap<String, Handle>,
    pub(super) presented: Vec<(String, String)>,
    pub(super) feeds: Vec<(String, String)>,
    pub(super) fonts: Vec<BundledFontState>,
    pub(super) said: Vec<String>,
}

pub(super) struct PendingBundle {
    graph: Graph,
    inputs: Vec<(String, FieldData)>,
    tickets: Vec<Ticket>,
}

pub(super) struct BundledFontState {
    pub(super) atlas: BundledFont,
    codes: Vec<i32>,
    seen: Option<Handle>,
}

/// Replaces `name` in `handles`, releasing any previous Field in `host`.
pub(super) fn hold_in(
    handles: &mut BTreeMap<String, Handle>,
    host: &mut Host,
    name: &str,
    handle: Handle,
) {
    if let Some(old) = handles.insert(name.to_owned(), handle)
        && old != handle
    {
        let _ = host.release(old);
    }
}

fn put_on_device(runtime: &mut Runtime, handle: Handle) -> bool {
    runtime.has_device()
        && runtime
            .host()
            .field_bytes(handle)
            .is_ok_and(|bytes| bytes.host >= COPY_LEAST)
        && runtime.upload(handle).unwrap_or(false)
        && runtime.host_mut().let_go_of_copy(handle).is_ok()
}

fn shape_progress(graph: &Graph) -> (Vec<KernelId>, Vec<(usize, usize)>) {
    let place: BTreeMap<&entelechy_graph::NodeId, usize> = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(i, node)| (&node.id, i))
        .collect();
    let mut links = Vec::new();
    for (i, node) in graph.nodes.iter().enumerate() {
        for binding in node.inputs.values() {
            let source = match binding {
                Binding::Now(source) => source,
                Binding::Previous { source, .. } => source,
            };
            if let Source::Node { node: from, .. } = source
                && let Some(&from) = place.get(from)
            {
                links.push((i, from));
            }
        }
    }
    let nodes = graph.nodes.iter().map(|node| node.kernel.clone()).collect();
    (nodes, links)
}

impl Session {
    /// Opens a compiled [`Bundle`] on `backend` (`gpu`, `jit`, or `best`;
    /// `cpu` as well when the `documents` feature is enabled). When `turns`
    /// is true, Kernels prepare in turns across frames before the scene lands.
    pub fn open_bundle(
        bundle: Bundle,
        backend: &str,
        limits: Limits,
        wait: entelechy_gpu::Wait,
        turns: bool,
    ) -> Result<crate::LoadedBundle, String> {
        let mut validated = Vec::with_capacity(bundle.modules.len());
        for module in bundle.modules {
            validated.push(entelechy_ir::validate(module).map_err(|e| e.to_string())?);
        }
        let one = [backend];
        let named: &[&str] = match backend {
            "best" => BEST,
            _ => &one,
        };
        let mut passed_over = Vec::new();
        let mut last = String::new();
        for name in named {
            let built = match Self::build(name, limits, wait) {
                Ok(built) => built,
                Err(why) if named.len() > 1 => {
                    passed_over.push(format!("{name}: {why}"));
                    continue;
                }
                Err(why) => return Err(why),
            };
            if let Err(why) = validated
                .iter()
                .try_for_each(|m| built.runtime.granted(m).map_err(|e| e.to_string()))
            {
                last = why.clone();
                if named.len() > 1 {
                    passed_over.push(format!("{name}: {why}"));
                    continue;
                }
                return Err(why);
            }
            match Self::start_bundle(
                built.runtime,
                &validated,
                bundle.graph.clone(),
                bundle.params.clone(),
                bundle.inputs.clone(),
                bundle.presented.clone(),
                bundle.feeds.clone(),
                bundle.fonts.clone(),
                turns,
            ) {
                Ok(mut session) => {
                    let said = session.take_bundle_said();
                    return Ok(crate::LoadedBundle {
                        session,
                        display: built.display,
                        said,
                        passed_over,
                    });
                }
                Err(why) => {
                    last = why.clone();
                    if named.len() > 1 {
                        passed_over.push(format!("{name}: the bundle did not load: {why}"));
                    } else {
                        return Err(why);
                    }
                }
            }
        }
        Err(if last.is_empty() {
            format!("no backend opens: {}", passed_over.join("; "))
        } else {
            last
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn start_bundle(
        mut runtime: Runtime,
        validated: &[entelechy_ir::Validated],
        graph: Graph,
        params: BTreeMap<String, Value>,
        inputs: Vec<(String, FieldData)>,
        presented: Vec<(String, String)>,
        feeds: Vec<(String, String)>,
        fonts: Vec<BundledFont>,
        turns: bool,
    ) -> Result<Self, String> {
        let (nodes, links) = shape_progress(&graph);
        runtime.progress().shape(nodes, links);
        if turns {
            runtime.prepare_in_turns();
        }
        let tickets: Vec<Ticket> = if turns {
            validated
                .iter()
                .map(|m| runtime.prepare_later(m.clone(), String::new()))
                .collect()
        } else {
            let mut candidates = Vec::with_capacity(validated.len());
            for m in validated {
                let candidate = runtime
                    .prepare_module(m.clone(), "")
                    .map_err(|e| e.to_string())?;
                candidates.push((candidate, Expect::Absent));
            }
            runtime.publish_all(candidates).map_err(|e| e.to_string())?;
            Vec::new()
        };
        let fonts = fonts
            .into_iter()
            .map(|atlas| BundledFontState {
                atlas,
                codes: Vec::new(),
                seen: None,
            })
            .collect();
        let mut session = Self::from_backing(
            runtime,
            Backing::Bundled(Box::new(BundledScene {
                graph: None,
                pending: Some(PendingBundle {
                    graph,
                    inputs,
                    tickets,
                }),
                params,
                handles: BTreeMap::new(),
                presented,
                feeds,
                fonts,
                said: Vec::new(),
            })),
        );
        if turns {
            session.sync_bundle_turn()?;
        } else {
            session.land_pending_bundle()?;
        }
        Ok(session)
    }

    pub(super) fn take_bundle_said(&mut self) -> Vec<String> {
        match &mut self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(_) => Vec::new(),
            Backing::Bundled(b) => std::mem::take(&mut b.said),
        }
    }

    /// Syncs a frame's changes and returns lines to report.
    pub fn sync_lines(&mut self) -> Vec<String> {
        match &mut self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(_) => self
                .sync_on_frame()
                .into_iter()
                .map(|e| e.to_string())
                .collect(),
            Backing::Bundled(_) => {
                if let Err(why) = self.sync_bundle_turn()
                    && let Some(b) = self.backing.bundled_mut()
                {
                    b.said.push(why);
                }
                self.take_bundle_said()
            }
        }
    }

    /// Advances one turn of preparation and returns lines to report.
    pub fn sync_turn_lines(&mut self) -> Vec<String> {
        match &mut self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(_) => self
                .sync_a_turn()
                .into_iter()
                .map(|e| e.to_string())
                .collect(),
            Backing::Bundled(_) => {
                if let Err(why) = self.sync_bundle_turn()
                    && let Some(b) = self.backing.bundled_mut()
                {
                    b.said.push(why);
                }
                self.take_bundle_said()
            }
        }
    }

    pub(super) fn sync_bundle_turn(&mut self) -> Result<(), String> {
        let began = entelechy_kernel::clock::now();
        while self.runtime.prepare_turn(1) > 0 && entelechy_kernel::clock::since(began) < TURN {}

        let ready = match &self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(_) => false,
            Backing::Bundled(b) => b.pending.is_some() && self.runtime.prepare_queued() == 0,
        };
        if ready {
            self.land_pending_bundle()?;
        } else {
            self.sync_bundled_clipboard();
            self.finish_used();
        }
        Ok(())
    }

    fn land_pending_bundle(&mut self) -> Result<(), String> {
        let pending = match &mut self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(_) => return Ok(()),
            Backing::Bundled(b) => match b.pending.take() {
                Some(p) => p,
                None => return Ok(()),
            },
        };
        if !pending.tickets.is_empty() {
            let mut candidates = Vec::with_capacity(pending.tickets.len());
            for ticket in pending.tickets {
                let prepared = self
                    .runtime
                    .poll(ticket)
                    .ok_or_else(|| "a bundled Kernel is still preparing".to_owned())?
                    .map_err(|e| e.to_string())?;
                candidates.push((prepared, Expect::Absent));
            }
            self.runtime
                .publish_all(candidates)
                .map_err(|e| e.to_string())?;
        }
        let mut on_device = BTreeSet::new();
        if let Some(b) = self.backing.bundled_mut() {
            for (name, text) in &pending.graph.texts {
                let words: Vec<u32> = text.chars().map(|c| c as u32).collect();
                let data = FieldData::from_storage(
                    Schema::new(vec![Scalar::Int]),
                    Domain::Rows(words.len() as u32),
                    words,
                )
                .map_err(|e| format!("input `{name}`'s text: {e}"))?;
                let handle = self
                    .runtime
                    .host_mut()
                    .adopt(data)
                    .map_err(|e| format!("input `{name}`'s text: {e}"))?;
                hold_in(&mut b.handles, self.runtime.host_mut(), name, handle);
            }
            for (name, data) in pending.inputs {
                let handle = self
                    .runtime
                    .host_mut()
                    .adopt(data)
                    .map_err(|e| format!("input `{name}`: {e}"))?;
                if put_on_device(&mut self.runtime, handle) {
                    on_device.insert(handle);
                }
                hold_in(&mut b.handles, self.runtime.host_mut(), &name, handle);
            }
            for font in &b.fonts {
                let fields = glyphs_from_atlas(&font.atlas, &[])?;
                hold_font_fields(&mut b.handles, self.runtime.host_mut(), &font.atlas, fields)?;
            }
            b.graph = Some(Arc::new(pending.graph));
            b.said.push("scene loaded".to_owned());
        }
        self.copies_let_go.extend(on_device);
        self.sync_bundled_clipboard();
        self.finish_used();
        Ok(())
    }

    /// Overrides a graph parameter for later evaluations, in the shape of
    /// its declared type (`[[1, 0], [0, 1]]` for a `Mat2`); another layout
    /// is refused. Unknown names are kept for a later declaration.
    pub fn set_param(&mut self, name: &str, value: Value) -> Result<(), String> {
        match &mut self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(_) => {
                self.settle_lowering();
                let docs = self.backing.docs_mut();
                docs.scene.set_param(name, value)?;
                docs.history.changed(
                    crate::Change::Param {
                        name: name.to_owned(),
                    },
                    docs.origin.clone(),
                );
                Ok(())
            }
            Backing::Bundled(b) => {
                let g = b
                    .graph
                    .as_deref()
                    .or_else(|| b.pending.as_ref().map(|p| &p.graph));
                let value = match g.and_then(|g| g.params.iter().find(|p| p.name == name)) {
                    Some(param) => value.shaped_like(&param.default).ok_or_else(|| {
                        format!(
                            "param `{name}` is of type {}; `{value}` is not one",
                            param.ty
                        )
                    })?,
                    None => value,
                };
                if g.is_some_and(|g| g.times_params.iter().any(|p| p == name)) {
                    return Err(format!(
                        "param `{name}` sizes the graph and cannot be changed in a compiled bundle"
                    ));
                }
                b.params.insert(name.to_owned(), value);
                Ok(())
            }
        }
    }

    /// Puts `text` on the clipboard, as a person copying elsewhere would,
    /// and has the scene's `= clipboard` inputs hold it at once.
    pub fn place_clipboard(&mut self, text: &str) -> Result<(), String> {
        self.board.place(text)?;
        match &mut self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(_) => {
                let events = {
                    let docs = self.backing.docs_mut();
                    let clipboards = docs.scene.files().fed_by(crate::devices::CLIPBOARD);
                    let already = clipboards.iter().all(|name| docs.inputs.holds(name));
                    let inputs = &mut docs.inputs;
                    let said = self.board.sync(
                        &clipboards,
                        already,
                        self.runtime.host_mut(),
                        |host, name, h| {
                            inputs.hold(host, name, h);
                        },
                    );
                    said.into_iter()
                        .map(|why| crate::Event::Rejected {
                            path: std::path::PathBuf::from(crate::devices::CLIPBOARD),
                            error: crate::Diagnostic::new(why),
                        })
                        .collect::<Vec<_>>()
                };
                self.remember(&events);
                Ok(())
            }
            Backing::Bundled(_) => {
                self.sync_bundled_clipboard();
                Ok(())
            }
        }
    }

    fn sync_bundled_clipboard(&mut self) {
        let Some(b) = self.backing.bundled_mut() else {
            return;
        };
        if b.graph.is_none() {
            return;
        }
        let names: Vec<String> = b
            .feeds
            .iter()
            .filter(|(_, dev)| dev == crate::devices::CLIPBOARD)
            .map(|(inp, _)| inp.clone())
            .collect();
        let already = names.iter().all(|name| b.handles.contains_key(name));
        let handles = &mut b.handles;
        let said = self
            .board
            .sync(&names, already, self.runtime.host_mut(), |host, name, h| {
                hold_in(handles, host, name, h);
            });
        b.said
            .extend(said.into_iter().map(|why| format!("clipboard: {why}")));
    }

    pub(super) fn feed_bundled_fonts(&mut self) {
        let count = match &self.backing {
            #[cfg(feature = "documents")]
            Backing::Documents(_) => return,
            Backing::Bundled(b) => b.fonts.len(),
        };
        for idx in 0..count {
            let (source, seen) = match &self.backing {
                #[cfg(feature = "documents")]
                Backing::Documents(_) => return,
                Backing::Bundled(b) => (b.fonts[idx].atlas.codes.clone(), b.fonts[idx].seen),
            };
            let handle = match &source {
                Source::Node { node, port } => self.cached(node, port),
                Source::External(name) => self.backing.handles().get(name).copied(),
            };
            let Some(handle) = handle else {
                continue;
            };
            if seen == Some(handle) {
                continue;
            }
            let codes = self
                .runtime
                .materialize(handle)
                .map_err(|e| e.to_string())
                .and_then(|()| self.runtime.host().data(handle).map_err(|e| e.to_string()))
                .and_then(|data| {
                    if data.schema().scalars() == [Scalar::Int] {
                        Ok(data.storage().iter().map(|&w| w as i32).collect::<Vec<_>>())
                    } else {
                        Err("its `codes` are a `Rows<Int>` of characters".to_owned())
                    }
                });
            if let Some(b) = self.backing.bundled_mut() {
                let font = &mut b.fonts[idx];
                font.seen = Some(handle);
                match codes {
                    Ok(codes) if font.codes == codes => {}
                    Ok(codes) => {
                        match glyphs_from_atlas(&font.atlas, &codes).and_then(|fields| {
                            hold_font_fields(
                                &mut b.handles,
                                self.runtime.host_mut(),
                                &font.atlas,
                                fields,
                            )
                        }) {
                            Ok(()) => font.codes = codes,
                            Err(why) => b.said.push(format!("font `{}`: {why}", font.atlas.node)),
                        }
                    }
                    Err(why) => b.said.push(format!("font `{}`: {why}", font.atlas.node)),
                }
            }
        }
    }

    /// Packs the currently loaded document-backed scene into a self-contained
    /// [`Bundle`], calling `font_atlas(node, path)` to bake each `font` node's
    /// `[glyphs, segments, metrics]` atlas Fields.
    #[cfg(feature = "documents")]
    pub fn bundle_with_fonts(
        &mut self,
        mut font_atlas: impl FnMut(
            &str,
            &std::path::Path,
        ) -> Result<(FieldData, FieldData, FieldData), String>,
    ) -> Result<Bundle, String> {
        let mut graph = self
            .graph()
            .cloned()
            .ok_or_else(|| "no scene is loaded".to_owned())?;
        if !graph.times_rows.is_empty() {
            return Err(
                "a scene whose `times` reads a Field row cannot be packed into a static bundle"
                    .to_owned(),
            );
        }
        let files = self.backing.docs().scene.files().clone();
        if !files.sequences.is_empty() {
            return Err(
                "a scene with a frame sequence input cannot be exported as a single-file bundle"
                    .to_owned(),
            );
        }
        if !files.file_nodes.is_empty() || !files.field_nodes.is_empty() {
            return Err("a scene with dynamic `files` or `fields` directory nodes cannot be exported as a single-file bundle".to_owned());
        }

        let mut wanted_ids = BTreeSet::new();
        for node in &graph.nodes {
            wanted_ids.insert(node.kernel.clone());
            for binding in node.inputs.values() {
                if let Binding::Previous {
                    migration: Some(mig),
                    ..
                } = binding
                {
                    wanted_ids.insert(mig.clone());
                }
            }
        }
        let mut modules = Vec::new();
        for id in wanted_ids {
            let version = self
                .runtime
                .kernel(&id)
                .ok_or_else(|| format!("Kernel `{id}` is not published"))?;
            if let Some(validated) = version.module() {
                let mut m = validated.clone().into_module();
                for f in &mut m.functions {
                    f.bindings.clear();
                }
                m.entry.bindings.clear();
                if let Some(f) = &mut m.merge {
                    f.bindings.clear();
                }
                if let Some(f) = &mut m.init {
                    f.bindings.clear();
                }
                modules.push(m);
            }
        }

        for node in &mut graph.nodes {
            node.at = None;
        }
        for req in &mut graph.requirements {
            req.at = None;
        }

        let mut excluded: BTreeSet<String> = files
            .feeds
            .iter()
            .map(|feed| feed.input.clone())
            .chain(graph.texts.iter().map(|(name, _)| name.clone()))
            .collect();

        let mut fonts = Vec::with_capacity(files.fonts.len());
        for font in &files.fonts {
            let (glyphs, segments, metrics) = font_atlas(&font.node, &font.path)?;
            let port_ext = |port: &str| {
                font.outputs
                    .iter()
                    .find(|(p, _)| p == port)
                    .map(|(_, ext)| ext.clone())
                    .unwrap_or_default()
            };
            let outputs = [
                port_ext("glyphs"),
                port_ext("outlines"),
                port_ext("metrics"),
            ];
            for ext in &outputs {
                if !ext.is_empty() {
                    excluded.insert(ext.clone());
                }
            }
            fonts.push(BundledFont {
                node: font.node.clone(),
                codes: font.codes.clone(),
                outputs,
                glyphs,
                segments,
                metrics,
            });
        }

        let handle_entries: Vec<(String, Handle)> = self
            .backing
            .docs()
            .inputs
            .handles()
            .iter()
            .filter(|(name, _)| !excluded.contains(*name))
            .map(|(name, handle)| (name.clone(), *handle))
            .collect();
        let mut inputs = Vec::with_capacity(handle_entries.len());
        for (name, handle) in handle_entries {
            let data = self
                .field(handle)
                .map_err(|e| format!("input `{name}`: {e}"))?
                .clone();
            inputs.push((name, data));
        }

        let presented = files
            .presented
            .iter()
            .map(|p| (p.output.clone(), p.device.clone()))
            .collect();
        let feeds = files
            .feeds
            .iter()
            .map(|f| (f.input.clone(), f.device.clone()))
            .collect();
        let params = self.backing.docs().scene.params().clone();

        Ok(Bundle {
            modules,
            graph,
            params,
            inputs,
            presented,
            feeds,
            fonts,
        })
    }
}

fn hold_font_fields(
    handles: &mut BTreeMap<String, Handle>,
    host: &mut Host,
    atlas: &BundledFont,
    fields: (FieldData, FieldData, FieldData),
) -> Result<(), String> {
    for (ext, data) in [
        (&atlas.outputs[0], fields.0),
        (&atlas.outputs[1], fields.1),
        (&atlas.outputs[2], fields.2),
    ] {
        if ext.is_empty() {
            continue;
        }
        let handle = host.adopt(data).map_err(|e| e.to_string())?;
        hold_in(handles, host, ext, handle);
    }
    Ok(())
}
