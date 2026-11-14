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
#[cfg(feature = "documents")]
use crate::bundle::BundledAtlas;
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
        let Bundle {
            modules,
            graph,
            params,
            inputs,
            presented,
            feeds,
            fonts,
        } = bundle;
        let mut validated = Vec::with_capacity(modules.len());
        for module in modules {
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
            let mut runtime = built.runtime;
            let (nodes, links) = shape_progress(&graph);
            runtime.progress().shape(nodes, links);
            if !turns {
                let prepared = (|| -> Result<(), String> {
                    let mut candidates = Vec::with_capacity(validated.len());
                    for m in &validated {
                        let candidate = runtime
                            .prepare_module(m.clone(), "")
                            .map_err(|e| e.to_string())?;
                        candidates.push((candidate, Expect::Absent));
                    }
                    runtime
                        .publish_all(candidates)
                        .map(|_| ())
                        .map_err(|e| e.to_string())
                })();
                if let Err(why) = prepared {
                    last = why;
                    if named.len() > 1 {
                        passed_over.push(format!("{name}: the bundle did not load: {last}"));
                        continue;
                    }
                    return Err(last);
                }
            }
            let mut session = Self::start_bundle(
                runtime, &validated, graph, params, inputs, presented, feeds, fonts, turns,
            )?;
            let said = session.take_bundle_said();
            return Ok(crate::LoadedBundle {
                session,
                display: built.display,
                said,
                passed_over,
            });
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
        let fonts: Vec<BundledFontState> = fonts
            .into_iter()
            .map(|atlas| BundledFontState {
                atlas,
                codes: Vec::new(),
                seen: None,
            })
            .collect();
        let mut small_inputs = Vec::new();
        let mut large_inputs = Vec::new();
        for (name, data) in inputs {
            if (data.storage().len() as u64).saturating_mul(4) >= COPY_LEAST {
                large_inputs.push((name, data));
            } else {
                small_inputs.push((name, data));
            }
        }
        let mut handles = BTreeMap::new();
        let mut on_device = BTreeSet::new();
        for (name, text) in &graph.texts {
            let words: Vec<u32> = text.chars().map(|c| c as u32).collect();
            let data = FieldData::from_storage(
                Schema::new(vec![Scalar::Int]),
                Domain::Rows(words.len() as u32),
                words,
            )
            .map_err(|e| format!("input `{name}`'s text: {e}"))?;
            let handle = runtime
                .host_mut()
                .adopt(data)
                .map_err(|e| format!("input `{name}`'s text: {e}"))?;
            hold_in(&mut handles, runtime.host_mut(), name, handle);
        }
        for (name, data) in small_inputs {
            let handle = runtime
                .host_mut()
                .adopt(data)
                .map_err(|e| format!("input `{name}`: {e}"))?;
            hold_in(&mut handles, runtime.host_mut(), &name, handle);
        }
        for font in &fonts {
            let fields = glyphs_from_atlas(&font.atlas, &[])?;
            hold_font_fields(&mut handles, runtime.host_mut(), &font.atlas, fields)?;
        }
        // Upload and release large device-bound inputs in reverse allocation
        // order before queuing shader turns so top-of-heap storage merges back
        // into the linear allocator's wilderness without fragmentation holes.
        for (name, data) in large_inputs.into_iter().rev() {
            let handle = runtime
                .host_mut()
                .adopt(data)
                .map_err(|e| format!("input `{name}`: {e}"))?;
            if put_on_device(&mut runtime, handle) {
                on_device.insert(handle);
            }
            hold_in(&mut handles, runtime.host_mut(), &name, handle);
        }
        let tickets: Vec<Ticket> = if turns {
            runtime.prepare_in_turns();
            validated
                .iter()
                .map(|m| runtime.prepare_later(m.clone(), String::new()))
                .collect()
        } else {
            Vec::new()
        };
        let mut session = Self::from_backing(
            runtime,
            Backing::Bundled(Box::new(BundledScene {
                graph: None,
                pending: Some(PendingBundle { graph, tickets }),
                params,
                handles,
                presented,
                feeds,
                fonts,
                said: Vec::new(),
            })),
        );
        session.copies_let_go.extend(on_device);
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
        if let Some(b) = self.backing.bundled_mut() {
            b.graph = Some(Arc::new(pending.graph));
            b.said.push("scene loaded".to_owned());
        }
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
    /// [`Bundle`], calling `font_atlas(node, path, static_chars)` to bake each
    /// `font` node's atlas. `static_chars` is `Some` when every `font` node
    /// reading `path` computes its `codes` from static inputs and pure Kernels
    /// alone, naming the union of characters they can ever ask for.
    #[cfg(feature = "documents")]
    pub fn bundle_with_fonts(
        &mut self,
        mut font_atlas: impl FnMut(&str, &std::path::Path, Option<&str>) -> Result<BundledAtlas, String>,
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

        let static_chars = self.static_font_chars(&graph, &files);

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

        let mut fonts: Vec<BundledFont> = Vec::with_capacity(files.fonts.len());
        for font in &files.fonts {
            let inferred = static_chars.get(&font.path).map(String::as_str);
            let baked = font_atlas(&font.node, &font.path, inferred)?;
            let atlas = if let Some(prev) = fonts.iter().find(|f| *f.atlas == baked) {
                Arc::clone(&prev.atlas)
            } else {
                Arc::new(baked)
            };
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
                atlas,
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

    #[cfg(feature = "documents")]
    fn static_font_chars(
        &mut self,
        graph: &Graph,
        files: &entelechy_scene::Files,
    ) -> BTreeMap<std::path::PathBuf, String> {
        if files.fonts.is_empty() {
            return BTreeMap::new();
        }
        let dynamic_exts: BTreeSet<&str> = files
            .feeds
            .iter()
            .map(|f| f.input.as_str())
            .chain(
                files
                    .fonts
                    .iter()
                    .flat_map(|f| f.outputs.iter().map(|(_, ext)| ext.as_str())),
            )
            .collect();
        let fallbacks: BTreeMap<&str, &Source> = graph
            .fallbacks
            .iter()
            .map(|(n, s)| (n.as_str(), s))
            .collect();
        let node_map: BTreeMap<&entelechy_graph::NodeId, &entelechy_graph::Node> =
            graph.nodes.iter().map(|n| (&n.id, n)).collect();
        let held = self.backing.docs().inputs.handles();

        let mut needed_nodes = BTreeSet::new();
        let mut needed_exts = BTreeSet::new();
        let mut resolved_fonts: Vec<(usize, Source)> = Vec::new();
        let mut dynamic_paths: BTreeSet<std::path::PathBuf> = BTreeSet::new();

        for (idx, font) in files.fonts.iter().enumerate() {
            let mut local_nodes = BTreeSet::new();
            let mut local_exts = BTreeSet::new();
            let mut visiting = BTreeSet::new();
            let resolved = match &font.codes {
                Source::Node { node, port } => graph
                    .producer(node, port)
                    .map(|(n, p)| Source::Node { node: n, port: p }),
                Source::External(ext) => Some(Source::External(ext.clone())),
            };
            let ok = resolved.as_ref().is_some_and(|src| {
                is_static_source(
                    src,
                    graph,
                    &node_map,
                    &self.runtime,
                    held,
                    &dynamic_exts,
                    &fallbacks,
                    &mut visiting,
                    &mut local_nodes,
                    &mut local_exts,
                )
            });
            if ok && let Some(src) = resolved {
                needed_nodes.extend(local_nodes);
                needed_exts.extend(local_exts);
                resolved_fonts.push((idx, src));
            } else {
                dynamic_paths.insert(font.path.clone());
            }
        }
        resolved_fonts.retain(|(idx, _)| !dynamic_paths.contains(&files.fonts[*idx].path));
        if resolved_fonts.is_empty() {
            return BTreeMap::new();
        }

        let ext_handles: Vec<(String, Handle)> = needed_exts
            .into_iter()
            .filter_map(|name| held.get(&name).copied().map(|h| (name, h)))
            .collect();
        let mut ext_data = Vec::with_capacity(ext_handles.len());
        for (name, handle) in ext_handles {
            let Ok(data) = self.field(handle) else {
                return BTreeMap::new();
            };
            ext_data.push((name, data.clone()));
        }

        let backend: Box<dyn entelechy_kernel::Backend> = entelechy_jit::JitBackend::new()
            .map(|jit| Box::new(jit) as Box<dyn entelechy_kernel::Backend>)
            .unwrap_or_else(|_| Box::new(entelechy_cpu::CpuBackend));
        let mut temp_rt = Runtime::with_backend(self.runtime.host().budget_words(), backend);
        temp_rt.host_mut().set_work(self.runtime.host().work());

        let mut static_kernels = BTreeSet::new();
        for id in &needed_nodes {
            if let Some(node) = node_map.get(id) {
                static_kernels.insert(node.kernel.clone());
            }
        }
        let mut candidates = Vec::new();
        for kernel_id in static_kernels {
            let Some(version) = self.runtime.kernel(&kernel_id) else {
                return BTreeMap::new();
            };
            if let Some(validated) = version.module() {
                let Ok(prepared) = temp_rt.prepare_module(validated.clone(), "") else {
                    return BTreeMap::new();
                };
                candidates.push((prepared, Expect::Absent));
            }
        }
        if !candidates.is_empty() && temp_rt.publish_all(candidates).is_err() {
            return BTreeMap::new();
        }

        let mut temp_handles = BTreeMap::new();
        for (name, data) in ext_data {
            let Ok(h) = temp_rt.host_mut().adopt(data) else {
                return BTreeMap::new();
            };
            temp_handles.insert(name, h);
        }

        let mut node_outputs = Vec::new();
        for &(idx, ref src) in &resolved_fonts {
            let effective = resolve_fallback(src, &temp_handles, &fallbacks);
            if matches!(effective, Source::Node { .. }) {
                node_outputs.push((format!("f{idx}"), effective));
            }
        }

        let mut report_handles: BTreeMap<String, Handle> = BTreeMap::new();
        if !node_outputs.is_empty() {
            let nodes: Vec<entelechy_graph::Node> = graph
                .nodes
                .iter()
                .filter(|n| needed_nodes.contains(&n.id))
                .cloned()
                .collect();
            let kept_fallbacks = graph
                .fallbacks
                .iter()
                .filter(|(_, s)| match s {
                    Source::External(_) => true,
                    Source::Node { node, .. } => needed_nodes.contains(node),
                })
                .cloned()
                .collect();
            let sub = Graph {
                name: graph.name.clone(),
                externals: graph.externals.clone(),
                texts: Vec::new(),
                params: Vec::new(),
                requirements: Vec::new(),
                times_params: Vec::new(),
                times_rows: Vec::new(),
                nodes,
                outputs: node_outputs,
                fallbacks: kept_fallbacks,
                placed: Vec::new(),
                repeated: Vec::new(),
            };
            let mut eval = entelechy_graph::Evaluator::new();
            let Ok(report) = eval.evaluate(
                &sub,
                &mut temp_rt,
                &temp_handles,
                &entelechy_kernel::Context::default(),
                &BTreeMap::new(),
            ) else {
                return BTreeMap::new();
            };
            for (name, h) in report.outputs {
                report_handles.insert(name, h);
            }
        }

        let mut chars_by_path: BTreeMap<std::path::PathBuf, BTreeSet<char>> = BTreeMap::new();
        for (idx, src) in resolved_fonts {
            let effective = resolve_fallback(&src, &temp_handles, &fallbacks);
            let handle = match effective {
                Source::External(ext) => temp_handles.get(&ext).copied(),
                Source::Node { .. } => report_handles.get(&format!("f{idx}")).copied(),
            };
            let Some(handle) = handle else {
                return BTreeMap::new();
            };
            if temp_rt.materialize(handle).is_err() {
                return BTreeMap::new();
            }
            let Ok(data) = temp_rt.host().data(handle) else {
                return BTreeMap::new();
            };
            if data.schema().scalars() != [Scalar::Int] {
                return BTreeMap::new();
            }
            let set = chars_by_path
                .entry(files.fonts[idx].path.clone())
                .or_default();
            for &w in data.storage() {
                let code = w as i32;
                if code > 0
                    && let Some(c) = char::from_u32(code as u32)
                {
                    set.insert(c);
                }
            }
        }
        chars_by_path
            .into_iter()
            .map(|(path, set)| (path, set.into_iter().collect()))
            .collect()
    }
}

#[cfg(feature = "documents")]
fn resolve_fallback(
    src: &Source,
    held: &BTreeMap<String, Handle>,
    fallbacks: &BTreeMap<&str, &Source>,
) -> Source {
    let mut cur = src;
    let mut guard = 0usize;
    while let Source::External(ext) = cur {
        if held.contains_key(ext) || guard > 32 {
            break;
        }
        let Some(next) = fallbacks.get(ext.as_str()) else {
            break;
        };
        cur = next;
        guard += 1;
    }
    cur.clone()
}

#[cfg(feature = "documents")]
fn domain_expr_is_static(
    expr: &entelechy_graph::DomainExpr,
    kernel_params: &[entelechy_ir::Param],
) -> bool {
    use entelechy_graph::DomainExpr;
    match expr {
        DomainExpr::Int(_)
        | DomainExpr::Float(_)
        | DomainExpr::Count(_)
        | DomainExpr::Size(_)
        | DomainExpr::Value(..) => true,
        DomainExpr::Context(_) => false,
        DomainExpr::Name(name) => kernel_params.iter().any(|p| p.name == *name),
        DomainExpr::Vector(parts) | DomainExpr::Call(_, parts) => parts
            .iter()
            .all(|part| domain_expr_is_static(part, kernel_params)),
        DomainExpr::Neg(inner) => domain_expr_is_static(inner, kernel_params),
        DomainExpr::Binary(_, a, b) => {
            domain_expr_is_static(a, kernel_params) && domain_expr_is_static(b, kernel_params)
        }
    }
}

#[cfg(feature = "documents")]
#[allow(clippy::too_many_arguments)]
fn is_static_source(
    source: &Source,
    graph: &Graph,
    node_map: &BTreeMap<&entelechy_graph::NodeId, &entelechy_graph::Node>,
    runtime: &Runtime,
    held: &BTreeMap<String, Handle>,
    dynamic_exts: &BTreeSet<&str>,
    fallbacks: &BTreeMap<&str, &Source>,
    visiting: &mut BTreeSet<String>,
    needed_nodes: &mut BTreeSet<entelechy_graph::NodeId>,
    needed_exts: &mut BTreeSet<String>,
) -> bool {
    match source {
        Source::External(ext) => {
            if dynamic_exts.contains(ext.as_str()) {
                return false;
            }
            if held.contains_key(ext) {
                needed_exts.insert(ext.clone());
                return true;
            }
            if let Some(fb) = fallbacks.get(ext.as_str()) {
                let tag = format!("ext:{ext}");
                if !visiting.insert(tag.clone()) {
                    return false;
                }
                let ok = is_static_source(
                    fb,
                    graph,
                    node_map,
                    runtime,
                    held,
                    dynamic_exts,
                    fallbacks,
                    visiting,
                    needed_nodes,
                    needed_exts,
                );
                visiting.remove(&tag);
                return ok;
            }
            false
        }
        Source::Node { node, port } => {
            let Some((real_id, _)) = graph.producer(node, port) else {
                return false;
            };
            if needed_nodes.contains(&real_id) {
                return true;
            }
            let tag = format!("node:{real_id}");
            if !visiting.insert(tag.clone()) {
                return false;
            }
            let Some(n) = node_map.get(&real_id) else {
                visiting.remove(&tag);
                return false;
            };
            if n.is_async
                || !n.awake.is_empty()
                || n.params
                    .values()
                    .any(|v| matches!(v, entelechy_graph::ParamValue::Graph(_)))
            {
                visiting.remove(&tag);
                return false;
            }
            let Some(version) = runtime.kernel(&n.kernel) else {
                visiting.remove(&tag);
                return false;
            };
            let reqs = &version.manifest().requires;
            if reqs.contains(entelechy_abi::Capability::TimeRead)
                || reqs.contains(entelechy_abi::Capability::InputRead)
            {
                visiting.remove(&tag);
                return false;
            }
            if let Some(entelechy_graph::DomainSpec::Expr(expr)) = &n.domain
                && !domain_expr_is_static(expr, &version.manifest().params)
            {
                visiting.remove(&tag);
                return false;
            }
            for binding in n.inputs.values() {
                match binding {
                    Binding::Previous { .. } => {
                        visiting.remove(&tag);
                        return false;
                    }
                    Binding::Now(src) => {
                        if !is_static_source(
                            src,
                            graph,
                            node_map,
                            runtime,
                            held,
                            dynamic_exts,
                            fallbacks,
                            visiting,
                            needed_nodes,
                            needed_exts,
                        ) {
                            visiting.remove(&tag);
                            return false;
                        }
                    }
                }
            }
            visiting.remove(&tag);
            needed_nodes.insert(real_id);
            true
        }
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
