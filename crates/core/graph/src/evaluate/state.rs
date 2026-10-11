//! State between evaluations: migration or restart on a layout change,
//! restore on failure, the commit, and writes from outside.

use super::*;

impl Evaluator {
    /// Before the nodes run: every `<~` port whose committed state lacks
    /// the port's shape and layout is carried over by its migration
    /// Kernel, or restarts from its initial source without one. The
    /// replaced state is kept for [`Evaluator::release_replaced`] or
    /// [`Evaluator::restore_replaced`]. A failed migration puts back what
    /// this call changed.
    pub(super) fn reconcile_state(
        &mut self,
        groups: &[Group],
        runtime: &mut Runtime,
        context: &Context,
    ) -> Result<(Migrated, Reset), EvaluateError> {
        let outcome = self.reconcile_into(groups, runtime, context);
        if outcome.is_err() {
            self.restore_replaced(runtime);
        }
        outcome
    }

    pub(super) fn reconcile_into(
        &mut self,
        groups: &[Group],
        runtime: &mut Runtime,
        context: &Context,
    ) -> Result<(Migrated, Reset), EvaluateError> {
        let mut migrated = Vec::new();
        let mut reset = Vec::new();
        for group in groups {
            let manifest = group.version.manifest();
            for (name, binding) in &group.node.inputs {
                let Binding::Previous {
                    source:
                        Source::Node {
                            node: producer,
                            port,
                        },
                    migration,
                    ..
                } = binding
                else {
                    continue;
                };
                let Some(input) = manifest.inputs.iter().find(|input| &input.name == name) else {
                    continue;
                };
                let key = (producer.clone(), port.clone());
                let Some(current) = self.ledger.committed(&key) else {
                    continue;
                };
                let shape = runtime.host().domain(current)?.shape();
                let schema = runtime.host().schema(current)?.clone();
                if shape == input.ty.shape && schema == input.ty.schema {
                    continue;
                }
                let mismatch = if shape != input.ty.shape {
                    format!("expects {} but the state is a {shape}", input.ty.shape)
                } else {
                    format!(
                        "expects an element of {} ({}) but the state is a Field of {} ({schema})",
                        input.ty.schema.describe(),
                        input.ty.schema,
                        schema.describe()
                    )
                };
                let carried = match migration {
                    Some(kernel) => {
                        let fail = |message: String| EvaluateError::Migration {
                            node: group.node.id.clone(),
                            port: name.clone(),
                            kernel: kernel.clone(),
                            message,
                        };
                        let version = runtime
                            .kernel(kernel)
                            .ok_or_else(|| fail("is not published".into()))?;
                        let carried = migrate(runtime, &version, current, context).map_err(fail)?;
                        let carried_shape = runtime.host().domain(carried)?.shape();
                        let carried_schema = runtime.host().schema(carried)?.clone();
                        if carried_shape != input.ty.shape || carried_schema != input.ty.schema {
                            release(runtime, carried);
                            return Err(fail(format!(
                                "gave a {carried_shape} of {carried_schema}, and the port {mismatch}"
                            )));
                        }
                        migrated.push((group.node.id.clone(), name.clone(), kernel.clone()));
                        Some(carried)
                    }
                    None => {
                        reset.push((group.node.id.clone(), name.clone(), mismatch));
                        None
                    }
                };
                self.ledger.commit_take(&key);
                self.replaced.push(Replaced {
                    key: key.clone(),
                    previous: Some(current),
                });
                if let Some(carried) = carried {
                    self.ledger.commit_put(key, carried);
                }
            }
        }
        Ok((migrated, reset))
    }

    /// Puts back the state [`Evaluator::reconcile_state`] replaced, last
    /// first, releasing what a migration carried in its place.
    pub(super) fn restore_replaced(&mut self, runtime: &mut Runtime) {
        // One at a time, last first, so a handle two ports swapped stays
        // held throughout.
        while let Some(Replaced { key, previous }) = self.replaced.pop() {
            let carried = match previous {
                Some(previous) => self.ledger.commit_put(key, previous),
                None => self.ledger.commit_take(&key),
            };
            if let Some(carried) = carried
                && Some(carried) != previous
                && !self.ledger.is_cached(carried)
                && !self.is_kept(carried)
            {
                release(runtime, carried);
            }
        }
    }

    /// Releases the state [`Evaluator::reconcile_state`] replaced, once
    /// the evaluation has committed.
    pub(super) fn release_replaced(&mut self, runtime: &mut Runtime) {
        while let Some(Replaced { previous, .. }) = self.replaced.pop() {
            if let Some(previous) = previous
                && !self.ledger.is_cached(previous)
                && !self.is_kept(previous)
            {
                release(runtime, previous);
            }
        }
    }

    /// Makes `states` the whole committed world, atomically; a port
    /// `states` does not name restarts from its initial source. One trial
    /// frame under `context` proves it (migrating as any evaluation does)
    /// and commits nothing; its results stay cached. When the trial fails
    /// or a name is not a state port, the world is unchanged and `states`
    /// are released. Runs off the frame are dropped either way.
    #[allow(clippy::too_many_arguments)]
    pub fn restore(
        &mut self,
        graph: &Graph,
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        context: &Context,
        params: &BTreeMap<String, Value>,
        states: Vec<(NodeId, String, Handle)>,
    ) -> Result<Report, EvaluateError> {
        self.confirm(runtime)?;
        let ports: BTreeSet<(NodeId, String)> = graph
            .nodes
            .iter()
            .flat_map(|reader| reader.inputs.values())
            .filter_map(|binding| match binding {
                Binding::Previous {
                    source: Source::Node { node, port },
                    ..
                } => Some((node.clone(), port.clone())),
                _ => None,
            })
            .collect();
        if let Some((node, port, _)) = states
            .iter()
            .find(|(node, port, _)| !ports.contains(&(node.clone(), port.clone())))
        {
            let refused = EvaluateError::NotState {
                node: node.clone(),
                port: port.clone(),
            };
            for (_, _, handle) in states {
                release(runtime, handle);
            }
            return Err(refused);
        }
        self.recover(runtime);
        let mut named = BTreeSet::new();
        if let Some((node, port, _)) = states
            .iter()
            .find(|(node, port, _)| !named.insert((node.clone(), port.clone())))
        {
            let refused = EvaluateError::Graph(GraphError::DuplicateName(format!("{node}.{port}")));
            for (_, _, handle) in states {
                release(runtime, handle);
            }
            return Err(refused);
        }
        for (_, running) in std::mem::take(&mut self.running) {
            runtime.discard_run(running.run);
        }
        // Every port is replaced; what stood there is kept for a failed
        // trial.
        let installed: BTreeMap<(NodeId, String), Handle> = states
            .into_iter()
            .map(|(node, port, handle)| ((node, port), handle))
            .collect();
        for key in &ports {
            let previous = self.ledger.commit_take(key);
            self.replaced.push(Replaced {
                key: key.clone(),
                previous,
            });
            if let Some(handle) = installed.get(key) {
                self.ledger.commit_put(key.clone(), *handle);
            }
        }
        self.trial = true;
        let proved = self
            .evaluate_with(graph, runtime, externals, context, params, Demand::Every)
            .and_then(reported);
        self.trial = false;
        // An evaluation that failed early leaves this replaced state to
        // the caller.
        if proved.is_err() && !self.replaced.is_empty() {
            self.restore_replaced(runtime);
        }
        proved
    }

    /// Undoes what a panicking evaluation left half done (a trial's flag,
    /// replaced state), so a caller that catches the panic finds the
    /// world as it was before.
    pub(super) fn recover(&mut self, runtime: &mut Runtime) {
        self.trial = false;
        if !self.replaced.is_empty() {
            self.restore_replaced(runtime);
        }
    }

    /// Makes every `<~` source's current output the committed state.
    pub(super) fn commit(
        &mut self,
        graph: &Graph,
        runtime: &mut Runtime,
    ) -> Result<Vec<(NodeId, String)>, EvaluateError> {
        let (advanced, replaced) = self.commit_holding(graph)?;
        for (_, previous) in replaced {
            if let Some(previous) = previous
                && !self.ledger.is_cached(previous)
                && !self.is_kept(previous)
            {
                release(runtime, previous);
            }
        }
        Ok(advanced)
    }

    /// Commits as [`Evaluator::commit`] does, letting go of nothing: the
    /// ports advanced, and what each held before.
    #[allow(clippy::type_complexity)]
    pub(super) fn commit_holding(
        &mut self,
        graph: &Graph,
    ) -> Result<
        (
            Vec<(NodeId, String)>,
            Vec<((NodeId, String), Option<Handle>)>,
        ),
        EvaluateError,
    > {
        let mut advanced = Vec::new();
        let mut replaced = Vec::new();
        // A set for the membership test; the list keeps binding order
        // for the answer.
        let mut already: BTreeSet<(NodeId, String)> = BTreeSet::new();
        // Every port's new state is found before any commits, so a
        // missing one fails the commit whole.
        let mut staged = Vec::new();
        for node in &graph.nodes {
            for binding in node.inputs.values() {
                let Binding::Previous {
                    source: Source::Node { node, port },
                    ..
                } = binding
                else {
                    continue;
                };
                let key = (node.clone(), port.clone());
                // A run off the frame keeps its state until it lands.
                if already.contains(&key) || self.running.contains_key(node) {
                    continue;
                }
                let current =
                    self.cached(node, port)
                        .ok_or_else(|| EvaluateError::UnknownOutput {
                            node: node.clone(),
                            port: port.clone(),
                            near: None,
                        })?;
                already.insert(key.clone());
                staged.push((key, current));
            }
        }
        for (key, current) in staged {
            let previous = self.ledger.commit_put(key.clone(), current);
            if previous != Some(current) {
                advanced.push(key.clone());
                replaced.push((key, previous));
            }
        }
        Ok((advanced, replaced))
    }

    /// Forgets the committed state of `node.port` alone; its `<~` readers
    /// read their initial sources next evaluation. Runs off the frame
    /// that depend on it are dropped. The port must be state.
    pub fn reset_port(
        &mut self,
        graph: &Graph,
        runtime: &mut Runtime,
        node: &NodeId,
        port: &str,
    ) -> Result<(), EvaluateError> {
        if !is_state(graph, node, port) {
            return Err(EvaluateError::NotState {
                node: node.clone(),
                port: port.to_owned(),
            });
        }
        self.recover(runtime);
        self.confirm(runtime)?;
        for id in downstream_of_state(graph, node, port) {
            if let Some(running) = self.running.remove(&id) {
                runtime.discard_run(running.run);
            }
        }
        self.forget_state(&(node.clone(), port.to_owned()), runtime);
        Ok(())
    }

    /// Takes one port's committed state, releasing the Field when nothing
    /// else holds it.
    pub(super) fn forget_state(&mut self, key: &(NodeId, String), runtime: &mut Runtime) {
        if let Some(handle) = self.ledger.commit_take(key)
            && !self.ledger.is_cached(handle)
            && !self.is_kept(handle)
        {
            release(runtime, handle);
        }
    }

    /// Replaces the committed state of `node.port` with `words`, in the
    /// port's current layout; the next evaluation's `<~` readers start
    /// from it. The port must be state and evaluated once.
    pub fn write(
        &mut self,
        graph: &Graph,
        runtime: &mut Runtime,
        node: &NodeId,
        port: &str,
        words: Vec<u32>,
    ) -> Result<(), EvaluateError> {
        if !is_state(graph, node, port) {
            return Err(EvaluateError::NotState {
                node: node.clone(),
                port: port.to_owned(),
            });
        }
        self.confirm(runtime)?;
        let key = (node.clone(), port.to_owned());
        let current = self
            .ledger
            .committed(&key)
            .or_else(|| self.cached(node, port))
            .ok_or_else(|| EvaluateError::UnknownOutput {
                node: node.clone(),
                port: port.to_owned(),
                near: None,
            })?;
        let domain = runtime.host().domain(current)?;
        let schema = runtime.host().schema(current)?.clone();
        // The Kernel's words, one per scalar, narrowed where the port
        // keeps them narrow.
        let expected = domain.count() * schema.value_words();
        if words.len() != expected {
            return Err(EvaluateError::WordCount {
                node: node.clone(),
                port: port.to_owned(),
                expected,
                found: words.len(),
            });
        }
        let data = FieldData::from_values(schema, domain, &words)?;
        let handle = runtime.host_mut().adopt(data)?;
        if let Some(previous) = self.ledger.commit_put(key, handle)
            && !self.ledger.is_cached(previous)
            && !self.is_kept(previous)
        {
            release(runtime, previous);
        }
        Ok(())
    }
}

/// Runs a migration Kernel (a one-input map, default parameters) once
/// over a state Field's extent; returns the new state.
pub(super) fn migrate(
    runtime: &mut Runtime,
    version: &Arc<KernelVersion>,
    state: Handle,
    context: &Context,
) -> Result<Handle, String> {
    let manifest = version.manifest();
    if manifest.mode != entelechy_ir::Mode::Map || manifest.inputs.len() != 1 {
        return Err("must be a map over one input, the old state, giving the new".to_owned());
    }
    let domain = runtime
        .host()
        .domain(state)
        .map_err(|error| error.to_string())?;
    // Its input takes the state in its committed layout.
    let schema = runtime
        .host()
        .schema(state)
        .map_err(|error| error.to_string())?;
    let takes = &manifest.inputs[0].ty;
    if takes.shape != domain.shape() || takes.schema != *schema {
        return Err(format!(
            "takes {} of {} ({}), but the state is a {} of {} ({schema})",
            takes.shape,
            takes.schema.describe(),
            takes.schema,
            domain.shape(),
            schema.describe()
        ));
    }
    let invocation = Invocation {
        inputs: vec![state],
        params: manifest
            .params
            .iter()
            .map(|param| param.default.clone())
            .collect(),
        domain,
        context: *context,
    };
    let mut outputs = runtime
        .execute(version, &invocation)
        .map_err(|error| format!("failed: {error}"))?;
    let carried = outputs.remove(0);
    for extra in outputs {
        release(runtime, extra);
    }
    Ok(carried)
}

/// Whether `node.port` is state: some `<~` binding reads it.
fn is_state(graph: &Graph, node: &NodeId, port: &str) -> bool {
    graph
        .nodes
        .iter()
        .any(|reader| reads_state(reader, node, port))
}

/// Whether `reader` reads the state of `node.port` through a `<~`.
fn reads_state(reader: &Node, node: &NodeId, port: &str) -> bool {
    reader.inputs.values().any(|binding| {
        matches!(binding, Binding::Previous { source: Source::Node { node: producer, port: read }, .. }
            if producer == node && read == port)
    })
}

/// The nodes whose outputs depend, within one evaluation, on the state
/// of `node.port`: its `<~` readers and everything that reads them now.
fn downstream_of_state(graph: &Graph, node: &NodeId, port: &str) -> BTreeSet<NodeId> {
    let mut reached = BTreeSet::new();
    let mut frontier: Vec<NodeId> = graph
        .nodes
        .iter()
        .filter(|reader| reads_state(reader, node, port))
        .map(|reader| reader.id.clone())
        .collect();
    // Who reads each node now, computed once.
    let mut readers: BTreeMap<&NodeId, Vec<&NodeId>> = BTreeMap::new();
    for reader in &graph.nodes {
        for binding in reader.inputs.values() {
            if let Binding::Now(Source::Node { node: producer, .. }) = binding {
                readers.entry(producer).or_default().push(&reader.id);
            }
        }
    }
    while let Some(id) = frontier.pop() {
        if !reached.insert(id.clone()) {
            continue;
        }
        for reader in readers.get(&id).into_iter().flatten() {
            if !reached.contains(*reader) {
                frontier.push((*reader).clone());
            }
        }
    }
    reached
}
