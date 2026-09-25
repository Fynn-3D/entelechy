//! State between evaluations: what a republished Kernel's new layout
//! does to the committed state (migrated, restarted, or put back when the
//! evaluation fails), the commit at the end, and writes from outside.

use super::*;

impl Evaluator {
    /// Before the nodes run: every `<~` port whose committed state no
    /// longer has the port's shape and layout (a republished Kernel
    /// changed what it carries) is carried over by the binding's
    /// migration Kernel, a map from the old elements to the new run once
    /// over the state; without one, the port restarts from its initial
    /// source. What was committed is kept, not released: the caller
    /// either lets it go once the evaluation commits
    /// ([`Evaluator::release_replaced`]) or puts it back when it fails
    /// ([`Evaluator::restore_replaced`]). A migration that fails, or gives
    /// the wrong layout, puts back what this call had already changed.
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
        // One at a time, last first, so what is still to be put back is
        // still held while the rest is: a handle two ports swapped is
        // never released between them.
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

    /// Lets go of the state [`Evaluator::reconcile_state`] replaced, once
    /// the evaluation has committed and nothing will put it back.
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

    /// Makes `states` the whole committed world, as one change: every
    /// `<~` port is given its state from `states`, and a port `states`
    /// does not name starts again from its initial source. The world is
    /// proved first: one frame is evaluated over it under `context`, a
    /// port whose layout the Kernel no longer takes migrated as any
    /// evaluation migrates it, and only when that frame succeeds does the
    /// world stand. The frame itself commits nothing, so the next
    /// evaluation is the one after the world `states` holds, and finds
    /// what the trial computed in the cache. When the frame fails, or a
    /// name is not a state port, the world is as it was and `states`
    /// are released. Runs off the frame are let go either way: they
    /// began in a world that is no longer the one restored from.
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
        // Every port the world has or will have is replaced: what stood
        // there kept, to be put back if the trial fails.
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
        let proved = self.evaluate_with(graph, runtime, externals, context, params, None);
        self.trial = false;
        // An evaluation that failed before it reached the state left what
        // this replaced for its caller to put back.
        if proved.is_err() && !self.replaced.is_empty() {
            self.restore_replaced(runtime);
        }
        proved
    }

    /// Undoes what an evaluation interrupted by a panic left half done,
    /// before the next: a trial's flag, and state it replaced and never
    /// put back or let go. A caller that catches a panic (the FFI) then
    /// finds the world as it was before the evaluation that panicked.
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
        let mut advanced = Vec::new();
        // What has advanced, for the test below: a list of them is a
        // walk of every one already advanced, once a binding, which a
        // graph of a thousand states pays a thousand times over. The
        // list stays for the answer, which is in the order the bindings
        // came.
        let mut already: BTreeSet<(NodeId, String)> = BTreeSet::new();
        // Every port's new state found first, and only then committed: a
        // port that has none fails the commit before any other advances.
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
                // A run off the frame has not given this evaluation its
                // value: the state stays until it lands.
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
                advanced.push(key);
                if let Some(previous) = previous
                    && !self.ledger.is_cached(previous)
                    && !self.is_kept(previous)
                {
                    release(runtime, previous);
                }
            }
        }
        Ok(advanced)
    }

    /// Forgets the committed state of `node.port` alone, so its `<~`
    /// readers read their initial sources again on the next evaluation
    /// while every other port keeps its state. Runs off the frame that
    /// read it, or read what was computed from it, are let go: one that
    /// landed after would commit the forgotten state again. Other runs go
    /// on. The port must be state; one not committed yet is left as it is.
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
        for id in downstream_of_state(graph, node, port) {
            if let Some(running) = self.running.remove(&id) {
                runtime.discard_run(running.run);
            }
        }
        self.forget_state(&(node.clone(), port.to_owned()), runtime);
        Ok(())
    }

    /// Takes one port's committed state, releasing the Field when nothing
    /// else holds it: a handle two ports share is released with the last.
    pub(super) fn forget_state(&mut self, key: &(NodeId, String), runtime: &mut Runtime) {
        if let Some(handle) = self.ledger.commit_take(key)
            && !self.ledger.is_cached(handle)
            && !self.is_kept(handle)
        {
            release(runtime, handle);
        }
    }

    /// Replaces the committed state of `node.port` with `words`, laid out
    /// as the port's current value is, so the next evaluation's `<~`
    /// readers start from it: how a Field comes in from outside, a saved
    /// file for instance. The port must be state, read by some `<~`
    /// binding, and must have been evaluated once, so its layout and
    /// extent are known.
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

/// Runs a migration Kernel once over a state Field: a map with that one
/// input, its parameters at their defaults, over the state's extent.
/// Returns the new state, or why the Kernel cannot be one.
pub(super) fn migrate(
    runtime: &mut Runtime,
    version: &Arc<KernelVersion>,
    state: Handle,
    context: &Context,
) -> Result<Handle, String> {
    let manifest = version.manifest();
    if manifest.mode != orla_ir::Mode::Map || manifest.inputs.len() != 1 {
        return Err("must be a map over one input, the old state, giving the new".to_owned());
    }
    let domain = runtime
        .host()
        .domain(state)
        .map_err(|error| error.to_string())?;
    // Its input must take the state as it is: the layout it was
    // committed with, not the one the port wants now.
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
    // Who reads each node now, found once rather than a walk of the
    // graph for every node reached.
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
