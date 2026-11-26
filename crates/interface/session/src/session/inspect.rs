//! Document-backed inspection, state mutation, dynamic file/row feeds, and
//! observation wrappers over [`crate::observe`].

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use entelechy_documents::{Event, SourcePlace};
use entelechy_graph::{EvaluateError, Report};
use entelechy_kernel::{Context, Handle, KernelId};

use super::{Backing, Session, no_scene};
use crate::{SessionError, Traced, observe};

impl Session {
    /// The declarations `kernel` shares with its document's other Kernels,
    /// as that document writes them (its `struct`s, `type`s and `fn`s),
    /// and the document; none for a Kernel no document publishes.
    pub fn shared_with(&self, kernel: &KernelId) -> Option<(PathBuf, String)> {
        use entelechy_language::StatementKind;
        let path = self.backing.docs().kernels.owner(kernel)?.to_path_buf();
        let text = self.store().read_text(&path).ok()?;
        let outline = entelechy_language::outline(&text).ok()?;
        let shared: Vec<&str> = outline
            .statements
            .iter()
            .filter(|s| {
                matches!(
                    s.kind,
                    StatementKind::Struct | StatementKind::Type | StatementKind::Fn
                )
            })
            .filter_map(|s| text.get(s.range.clone()))
            .collect();
        Some((path, shared.join("\n\n")))
    }

    /// Why `node`'s `port` has no value to read: not a node, placed as
    /// several, folded into its reader, or not run since it was placed.
    pub fn why_no_value(&self, node: &entelechy_graph::NodeId, port: &str) -> String {
        let (node, port) = self.producer(node, port);
        observe::why_no_value(
            self.graph(),
            &self.evaluator,
            self.failing.as_deref(),
            &node,
            &port,
        )
    }

    pub(super) fn place_by_rows(&mut self) {
        let resized = match &mut self.backing {
            Backing::Documents(docs) => std::mem::take(&mut docs.resized),
            Backing::Bundled(_) => false,
        };
        if resized {
            let said = self.sync_apart(false);
            self.backing.docs_mut().later.extend(said);
        }
    }

    pub(super) fn follow_rows(&mut self) {
        let rows = match &self.backing {
            Backing::Documents(docs) => docs.scene.times_rows(),
            Backing::Bundled(_) => return,
        };
        for (row, held) in rows {
            let source = entelechy_graph::NodeId(row.source.0.as_str().into());
            let read = match self.cached(&source, &row.source.1) {
                None => Ok(0),
                Some(handle) => self.row_count(handle, row.row),
            };
            let docs = self.backing.docs_mut();
            match read {
                Ok(count) if count == held => {}
                Ok(count) => {
                    docs.scene.set_count(&row.node, count);
                    docs.resized = true;
                }
                Err(why) => {
                    let said = format!(
                        "`{}`'s `times = value({}.{}, {})`: {why}",
                        row.node, row.source.0, row.source.1, row.row
                    );
                    let event = Event::SceneRejected(crate::Diagnostic::new(said));
                    if !docs.later.contains(&event) {
                        docs.later.push(event);
                    }
                }
            }
        }
    }

    fn row_count(&mut self, handle: Handle, row: u32) -> Result<u32, String> {
        let count = self
            .runtime
            .host()
            .domain(handle)
            .map_err(|error| error.to_string())?
            .count();
        if row as usize >= count {
            return Err(format!("it has {count} element(s)"));
        }
        let whole = self
            .runtime
            .host()
            .schema(handle)
            .map_err(|error| error.to_string())?
            .scalars()
            .first()
            .is_some_and(|scalar| *scalar == entelechy_kernel::Scalar::Int);
        if !whole {
            return Err("a count is an `Int`".into());
        }
        self.runtime
            .materialize(handle)
            .map_err(|error| error.to_string())?;
        let word = self
            .runtime
            .host()
            .element(handle, row)
            .map_err(|error| error.to_string())?[0];
        u32::try_from(word as i32).map_err(|_| format!("it holds {}, less than 0", word as i32))
    }

    pub(super) fn write_confirmed(&mut self) {
        let writes = match &self.backing {
            Backing::Documents(docs) => docs.scene.files().writes.clone(),
            Backing::Bundled(_) => return,
        };
        let named: Vec<(String, Result<Vec<String>, String>)> = writes
            .iter()
            .filter(|write| {
                entelechy_formats::table::is_table(&entelechy_formats::extension(&write.path))
            })
            .map(|write| {
                let output = entelechy_graph::NodeId(write.output.clone().into());
                (write.output.clone(), self.element_names(&output, "result"))
            })
            .collect();
        let names = |output: &str| {
            named.iter().find(|(name, _)| name == output).map_or_else(
                || Err(format!("no output `{output}`")),
                |(_, names)| names.clone(),
            )
        };
        let docs = self.backing.docs_mut();
        let store = Arc::clone(docs.scene.store());
        docs.writes.after(
            store.as_ref(),
            &writes,
            self.evaluator.confirmed(),
            &mut self.runtime,
            &names,
        );
    }

    /// Evaluates only what `demand` needs; see
    /// [`entelechy_graph::Evaluator::evaluate_for`].
    pub fn evaluate_for(
        &mut self,
        context: &Context,
        demand: &BTreeSet<entelechy_graph::NodeId>,
    ) -> Result<Report, EvaluateError> {
        self.before_tick(context.time, true);
        let mut demand = demand.clone();
        demand.extend(self.fed_sources());
        demand.extend(self.written_sources());
        let demand = &demand;
        self.last_context = *context;
        let graph = self.backing.shared_graph().cloned().ok_or_else(no_scene)?;
        let report = self
            .evaluator
            .evaluate_for(
                &graph,
                &mut self.runtime,
                self.backing.handles(),
                context,
                self.backing.params(),
                demand,
            )
            .map_err(|error| self.with_refusal(error));
        self.failing = report.as_ref().err().map(|error| self.told(error));
        let report = report?;
        self.after_tick_committed();
        self.written_after(&report);
        self.follow_rows();
        self.last_report = Some(report.clone());
        Ok(report)
    }

    /// Replaces the committed state of `node.port` with `words`; see
    /// [`entelechy_graph::Evaluator::write`].
    pub fn write(
        &mut self,
        node: &entelechy_graph::NodeId,
        port: &str,
        words: Vec<u32>,
    ) -> Result<(), EvaluateError> {
        self.repin();
        let graph = self.backing.shared_graph().cloned().ok_or_else(no_scene)?;
        self.evaluator
            .write(&graph, &mut self.runtime, node, port, words)?;
        if let Backing::Documents(docs) = &mut self.backing {
            docs.history.changed(
                crate::Change::Written {
                    node: node.to_string(),
                    port: port.to_owned(),
                },
                docs.origin.clone(),
            );
        }
        Ok(())
    }

    /// Forgets the committed state of the state port `node.port` alone;
    /// its `<~` readers restart on the next evaluation.
    pub fn reset_port(
        &mut self,
        node: &entelechy_graph::NodeId,
        port: &str,
    ) -> Result<(), EvaluateError> {
        self.repin();
        let graph = self.backing.shared_graph().cloned().ok_or_else(no_scene)?;
        self.evaluator
            .reset_port(&graph, &mut self.runtime, node, port)?;
        if let Backing::Documents(docs) = &mut self.backing {
            docs.history.changed(
                crate::Change::ResetPort {
                    node: node.to_string(),
                    port: port.to_owned(),
                },
                docs.origin.clone(),
            );
        }
        Ok(())
    }

    pub(super) fn feed_field_nodes(&mut self) {
        let Backing::Documents(docs) = &self.backing else {
            return;
        };
        let nodes = docs.scene.files().field_nodes.clone();
        if nodes.is_empty() {
            return;
        }
        let store = Arc::clone(docs.scene.store());
        let reads = Arc::clone(&docs.reads);
        let readable =
            |path: &Path| entelechy_documents::reads::lock(&reads).check(store.as_ref(), path);
        for node in &nodes {
            let handle = match &node.paths {
                entelechy_graph::Source::Node { node, port } => self.cached(node, port),
                entelechy_graph::Source::External(name) => {
                    self.backing.docs().inputs.handles().get(name).copied()
                }
            };
            let Some(handle) = handle else {
                continue;
            };
            if self.backing.docs().inputs.field_node_saw(node, handle) {
                continue;
            }
            let paths = self
                .runtime
                .materialize(handle)
                .map_err(|e| e.to_string())
                .and_then(|()| self.runtime.host().data(handle).map_err(|e| e.to_string()))
                .and_then(|data| {
                    if data.schema().scalars() == [entelechy_kernel::Scalar::Int] {
                        entelechy_documents::inputs::paths_spelled(data.storage())
                    } else {
                        Err(
                            "its `paths` are a `Rows<Int>` of each path's bytes, ended by a 0"
                                .to_owned(),
                        )
                    }
                });
            let docs = self.backing.docs_mut();
            docs.inputs.give_field_paths(
                store.as_ref(),
                &readable,
                node,
                handle,
                paths,
                &mut self.runtime,
            );
        }
    }

    pub(super) fn feed_file_nodes(&mut self) {
        let Backing::Documents(docs) = &self.backing else {
            return;
        };
        let nodes = docs.scene.files().file_nodes.clone();
        if nodes.is_empty() {
            return;
        }
        let store = Arc::clone(docs.scene.store());
        let reads = Arc::clone(&docs.reads);
        let readable =
            |path: &Path| entelechy_documents::reads::lock(&reads).check(store.as_ref(), path);
        for node in &nodes {
            let handle = match &node.paths {
                entelechy_graph::Source::Node { node, port } => self.cached(node, port),
                entelechy_graph::Source::External(name) => {
                    self.backing.docs().inputs.handles().get(name).copied()
                }
            };
            let Some(handle) = handle else {
                continue;
            };
            if self.backing.docs().inputs.file_node_saw(node, handle) {
                continue;
            }
            let paths = self
                .runtime
                .materialize(handle)
                .map_err(|e| e.to_string())
                .and_then(|()| self.runtime.host().data(handle).map_err(|e| e.to_string()))
                .and_then(|data| {
                    if data.schema().scalars() == [entelechy_kernel::Scalar::Int] {
                        entelechy_documents::inputs::paths_spelled(data.storage())
                    } else {
                        Err(
                            "its `paths` are a `Rows<Int>` of each path's bytes, ended by a 0"
                                .to_owned(),
                        )
                    }
                });
            let docs = self.backing.docs_mut();
            docs.inputs.give_paths(
                store.as_ref(),
                &readable,
                node,
                handle,
                paths,
                self.runtime.host_mut(),
            );
        }
    }

    pub(super) fn feed_document_fonts(&mut self) {
        let Backing::Documents(docs) = &self.backing else {
            return;
        };
        let fonts = docs.scene.files().fonts.clone();
        for font in &fonts {
            let handle = match &font.codes {
                entelechy_graph::Source::Node { node, port } => self.cached(node, port),
                entelechy_graph::Source::External(name) => {
                    self.backing.docs().inputs.handles().get(name).copied()
                }
            };
            let Some(handle) = handle else {
                continue;
            };
            if self.backing.docs().inputs.font_saw(font, handle) {
                continue;
            }
            let codes = self
                .runtime
                .materialize(handle)
                .map_err(|e| e.to_string())
                .and_then(|()| self.runtime.host().data(handle).map_err(|e| e.to_string()))
                .and_then(|data| {
                    if data.schema().scalars() == [entelechy_kernel::Scalar::Int] {
                        Ok(data.storage().iter().map(|&w| w as i32).collect())
                    } else {
                        Err("its `codes` are a `Rows<Int>` of characters".to_owned())
                    }
                });
            let docs = self.backing.docs_mut();
            docs.inputs
                .give_codes(font, handle, codes, self.runtime.host_mut());
        }
    }

    /// Writes an evaluated, pinned node's output as a file: `.field` for
    /// any Field, `.png`/`.ppm` for pixels (a `Grid<Float>` as gray),
    /// `.pgm` for a `Grid<Float>`, `.wav` at 48 kHz for rows of one or two
    /// floats, `.mid` for rows of notes, `.csv`/`.json` for rows as the
    /// data table a scene's input reads, a column for each number of the
    /// element the node's Kernel declares. Only where
    /// [`Session::allowed_write`] allows, whoever asks.
    pub fn save(
        &mut self,
        node: &entelechy_graph::NodeId,
        port: &str,
        path: &Path,
    ) -> Result<(), String> {
        let path = self.allowed_write(path)?;
        self.confirm().map_err(|error| error.to_string())?;
        let (producer, out) = self.producer(node, port);
        if self.evaluator.cached(&producer, &out).is_none() {
            return Err(self.why_no_value(node, port));
        }
        let docs = self.backing.docs();
        observe::save(
            docs.scene.store().as_ref(),
            docs.scene.graph(),
            &self.evaluator,
            &mut self.runtime,
            node,
            port,
            &path,
        )
    }

    /// How many elements a node's output holds.
    pub fn element_count(
        &self,
        node: &entelechy_graph::NodeId,
        port: &str,
    ) -> Result<usize, String> {
        let (node, port) = self.producer(node, port);
        observe::element_count(&self.evaluator, &self.runtime, &node, &port)
    }

    /// Runs one element of `node`'s Kernel on the interpreter and returns
    /// every binding: element `at` of a map, step `at[0]` of a scan. What
    /// the node reads is evaluated first under `context` with its sources
    /// pinned, the node itself not run, so a node that fails is traced up
    /// to its failure. The trace commits nothing and removes the pins it
    /// added.
    pub fn trace(
        &mut self,
        node: &entelechy_graph::NodeId,
        at: [u32; 3],
        context: &Context,
    ) -> Result<Traced, String> {
        let (target, version) = observe::trace_target(self.graph(), &self.runtime, node)?;
        let pinned: Vec<entelechy_graph::NodeId> = observe::trace_sources(&target)
            .into_iter()
            .filter(|source| !self.evaluator.pins().contains(source))
            .collect();
        for source in &pinned {
            self.evaluator.pin(source.clone());
        }
        if let Backing::Documents(docs) = &mut self.backing {
            let store = Arc::clone(docs.scene.store());
            docs.inputs
                .select_frames(store.as_ref(), self.runtime.host_mut(), context.time);
        }
        self.hold_progress();
        self.repin();
        let readers = BTreeSet::from([target.id.clone()]);
        let tried = match self.backing.shared_graph().cloned() {
            Some(graph) => self.evaluator.evaluate_trial_reading(
                &graph,
                &mut self.runtime,
                self.backing.handles(),
                context,
                self.backing.params(),
                &readers,
            ),
            None => Err(no_scene()),
        };
        let written = match &self.backing {
            Backing::Documents(docs) => docs
                .kernels
                .unoptimized(&target.kernel, version.source())
                .or_else(|| {
                    let kernel = docs.scene.resolve_source(version.source()).ok()?;
                    entelechy_documents::kernels::Written::compiled(&kernel, &target.kernel, None)
                }),
            Backing::Bundled(_) => None,
        };
        let traced = tried.map_err(|error| error.to_string()).and_then(|_| {
            let graph = self.backing.shared_graph().ok_or("no scene is loaded")?;
            observe::trace_element(
                graph,
                &self.evaluator,
                &mut self.runtime,
                self.backing.handles(),
                self.backing.params(),
                written.as_ref().map(|written| written.module()),
                (&target, &version),
                context,
                at,
            )
        });
        for source in pinned {
            self.evaluator.unpin(&source);
        }
        traced.map(|(trace, module)| Traced {
            place: trace
                .fault
                .as_ref()
                .zip(written.as_ref())
                .and_then(|(fault, written)| {
                    written.place_of(fault.function.as_deref(), fault.value?)
                }),
            trace,
            module,
            kernel: target.kernel.clone(),
        })
    }

    /// Where in its source an evaluation's failure at one element is
    /// written, found by tracing that element again; `None` for any other
    /// failure, or when the trace does not fail there (a group of fused
    /// nodes failing in one it absorbed) or no document owns the Kernel.
    pub fn failure_place(
        &mut self,
        error: &EvaluateError,
        context: &Context,
    ) -> Option<SourcePlace> {
        let EvaluateError::Execution { node, error } = error else {
            return None;
        };
        let (x, y) = match error {
            entelechy_kernel::ExecutionError::Arithmetic { x, y, .. }
            | entelechy_kernel::ExecutionError::Exhausted { x, y, .. } => (*x, *y),
            _ => return None,
        };
        let fused = self.evaluator.fused_parts(node);
        fused
            .iter()
            .chain(std::iter::once(node))
            .find_map(|traced| self.trace(traced, [x, y, 0], context).ok()?.place)
    }

    /// Traces the exact element that caused `error`, returning `(node_name, Traced)`
    /// with local bindings, calls, and fault source place when available.
    pub fn failure_trace(
        &mut self,
        error: &EvaluateError,
        context: &Context,
    ) -> Option<(String, Traced)> {
        let EvaluateError::Execution { node, error } = error else {
            return None;
        };
        let (x, y) = match error {
            entelechy_kernel::ExecutionError::Arithmetic { x, y, .. }
            | entelechy_kernel::ExecutionError::Exhausted { x, y, .. } => (*x, *y),
            _ => return None,
        };
        let fused = self.evaluator.fused_parts(node);
        for candidate in fused.iter().chain(std::iter::once(node)) {
            if let Ok(traced) = self.trace(candidate, [x, y, 0], context)
                && traced.trace.fault.is_some()
            {
                return Some((candidate.to_string(), traced));
            }
        }
        None
    }

    /// One element field of a node's output: its type and words. Named
    /// `node.port[index].field.field` ([`crate::observe::parse_target`]).
    pub fn element_field(
        &mut self,
        node: &entelechy_graph::NodeId,
        port: &str,
        index: u32,
        path: &[String],
    ) -> Result<(entelechy_ir::Type, Vec<u32>), String> {
        let (node, port) = self.producer(node, port);
        observe::element_field(
            self.backing.graph(),
            &self.evaluator,
            &mut self.runtime,
            (&node, &port),
            index,
            path,
        )
    }

    /// The name of each number in an element of what `node.port` names,
    /// in the order a read gives them: `hp`, `pos.x`; one number alone
    /// is `""`.
    pub fn element_names(
        &self,
        node: &entelechy_graph::NodeId,
        port: &str,
    ) -> Result<Vec<String>, String> {
        let (node, port) = self.producer(node, port);
        observe::element_names(self.backing.graph(), &self.runtime, (&node, &port))
    }

    /// The fields of the struct `path` names in `node.port`'s element, in
    /// order; none for a value that is not a struct.
    pub fn element_fields(
        &mut self,
        node: &entelechy_graph::NodeId,
        port: &str,
        path: &[String],
    ) -> Result<Vec<String>, String> {
        let (node, port) = self.producer(node, port);
        observe::element_fields(self.backing.graph(), &self.runtime, (&node, &port), path)
    }

    /// Writes a Field to a file in the format its extension names, as
    /// [`Session::save`] does, but for a data table, whose columns only a
    /// node's Kernel names. Only where [`Session::allowed_write`] allows,
    /// whoever asks.
    pub fn export(&mut self, handle: Handle, path: &Path) -> Result<(), SessionError> {
        let path = self.allowed_write(path).map_err(SessionError::Observe)?;
        let store = Arc::clone(self.backing.docs().scene.store());
        observe::write(store.as_ref(), &mut self.runtime, handle, &path, None).map_err(|why| {
            match why {
                observe::Written::Io(path, e) => SessionError::Io(path, e),
                observe::Written::Observe(why) => SessionError::Observe(why),
                other => SessionError::Unwritable(path.clone(), other.to_string()),
            }
        })
    }

    /// Reads one pixel of a cached image, materializing it if needed.
    pub fn read_pixel(
        &mut self,
        image: Handle,
        x: u32,
        y: u32,
    ) -> Result<entelechy_kernel::Pixel, SessionError> {
        observe::read_pixel(&mut self.runtime, image, x, y)
    }

    /// Searches available packages, Kernels, and preset graphs reachable from
    /// this session's store and scene directory without loading them.
    pub fn catalog(
        &self,
        query: &entelechy_documents::catalog::CatalogQuery,
    ) -> entelechy_documents::catalog::CatalogResult {
        entelechy_documents::catalog::query(self.store(), self.root().parent(), query)
    }
}
