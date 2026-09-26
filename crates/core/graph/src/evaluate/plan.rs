//! What executes: every node as its own group, producers fused into
//! their sole pointwise consumers, a node placed many times run as one,
//! and each group's domain.

use super::*;

impl Evaluator {
    /// Decides what executes: every node as its own group, then producers
    /// folded into their sole pointwise consumers when fusion is on.
    pub(super) fn plan(
        &mut self,
        graph: &Graph,
        order: &[usize],
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        overrides: &BTreeMap<String, Value>,
        context: &Context,
    ) -> Result<Vec<Group>, EvaluateError> {
        let mut groups: Vec<Group> = Vec::with_capacity(order.len());
        for index in order {
            let node = &graph.nodes[*index];
            let version = runtime.kernel(&node.kernel).ok_or_else(|| {
                // Everything published, so a Kernel named
                // slightly wrong is named rather than searched
                // for.
                let wanted = node.kernel.to_string();
                let published: Vec<String> = runtime
                    .kernels()
                    .map(|version| version.id().to_string())
                    .collect();
                let near = orla_abi::names::nearest(&wanted, published.iter().map(String::as_str))
                    .map(str::to_owned);
                EvaluateError::UnknownKernel {
                    node: node.id.clone(),
                    kernel: node.kernel.clone(),
                    near,
                }
            })?;
            let parts = vec![(node.id.clone(), version.generation())];
            groups.push(Group {
                node: node.clone(),
                version,
                parts,
                absorbed: Vec::new(),
                repeat: 1,
                chain: None,
            });
        }
        if !self.fusion {
            self.fused.clear();
            self.forget_pending(runtime, |_| false);
            return Ok(groups);
        }
        // How many readers each output has: bindings and graph outputs.
        let mut readers: BTreeMap<(NodeId, String), usize> = BTreeMap::new();
        let mut count = |source: &Source| {
            if let Source::Node { node, port } = source {
                *readers.entry((node.clone(), port.clone())).or_default() += 1;
            }
        };
        for node in &graph.nodes {
            for binding in node.inputs.values() {
                match binding {
                    Binding::Now(source) => count(source),
                    Binding::Previous {
                        source, initial, ..
                    } => {
                        count(source);
                        count(initial);
                        // State sources must be materialized; count twice.
                        count(source);
                    }
                }
            }
        }
        for (_, source) in &graph.outputs {
            count(source);
        }
        let nodes = graph.index();
        let mut domains: BTreeMap<NodeId, Domain> = BTreeMap::new();
        let mut absorbed_into: BTreeMap<NodeId, usize> = BTreeMap::new();
        // Where each group is, by the node it is named for. Groups are
        // never added or removed while this runs, only rewired, so the
        // positions hold.
        let at: BTreeMap<NodeId, usize> = groups
            .iter()
            .enumerate()
            .map(|(position, group)| (group.node.id.clone(), position))
            .collect();
        for position in 0..groups.len() {
            // Planned domain of this group, for the equality check.
            let planned = self.planned_domain(
                graph,
                &nodes,
                &groups[position],
                runtime,
                externals,
                &domains,
                overrides,
                context,
            );
            // Each binding names a producer to try; one fed through
            // several ports is tried once, with all of them.
            let bindings: Vec<Binding> = groups[position].node.inputs.values().cloned().collect();
            for binding in bindings {
                let Binding::Now(Source::Node {
                    node: producer_id, ..
                }) = binding
                else {
                    continue;
                };
                if self.pinned.contains(&producer_id)
                    || absorbed_into.contains_key(&producer_id)
                    || groups[position].node.is_async
                    || nodes
                        .get(&producer_id)
                        .is_some_and(|at| graph.nodes[*at].is_async)
                {
                    continue;
                }
                let Some(producer_position) = at.get(&producer_id).copied() else {
                    continue;
                };
                let producer_domain = domains.get(&producer_id).copied();
                let consumer = &groups[position];
                // Every port of this consumer the producer feeds, as
                // (port, consumer input, producer output). The producer
                // fuses only when these are all its readers: a map of
                // several outputs goes in whole or not at all.
                let consumer_manifest = consumer.version.manifest();
                let producer_manifest = groups[producer_position].version.manifest();
                let mut links: Vec<(String, u32, u32)> = Vec::new();
                for (name, bound) in &consumer.node.inputs {
                    let Binding::Now(Source::Node { node, port: from }) = bound else {
                        continue;
                    };
                    if node != &producer_id {
                        continue;
                    }
                    let input = consumer_manifest
                        .inputs
                        .iter()
                        .position(|i| &i.name == name);
                    let output = producer_manifest
                        .outputs
                        .iter()
                        .position(|o| &o.name == from);
                    let (Some(input), Some(output)) = (input, output) else {
                        continue;
                    };
                    links.push((name.clone(), input as u32, output as u32));
                }
                let only_readers = producer_manifest
                    .outputs
                    .iter()
                    .enumerate()
                    .all(|(k, out)| {
                        let linked = links.iter().filter(|link| link.2 == k as u32).count();
                        let read = readers
                            .get(&(producer_id.clone(), out.name.clone()))
                            .copied()
                            .unwrap_or(0);
                        read == linked
                    });
                // A narrow output rounds what passes through it; fused, the
                // consumer would read the value before the rounding.
                let narrow = links.iter().any(|link| {
                    !producer_manifest.outputs[link.2 as usize]
                        .ty
                        .schema
                        .is_wide()
                });
                if links.is_empty() || !only_readers || narrow {
                    continue;
                }
                // The consumer's first input decides its extent when it
                // sets none; whether that input is one of these matters
                // below.
                let reads_first = links.iter().any(|link| link.1 == 0);
                // A consumer with no `domain` of its own runs over its
                // first input's extent, so when that input is the
                // producer the two are the same extent whatever it
                // turns out to be. That is how a node whose size a
                // Kernel decides this frame is still fused into: the
                // producer's expression moves onto the fused node,
                // under the names fusion gives its ports.
                let carried = match (producer_domain, planned) {
                    (Some(a), Some(b)) if a == b => None,
                    (None, None)
                        if consumer.node.domain.is_none()
                            && reads_first
                            && matches!(
                                groups[producer_position].node.domain,
                                Some(DomainSpec::Expr(_))
                            ) =>
                    {
                        let Some(DomainSpec::Expr(expr)) = &groups[producer_position].node.domain
                        else {
                            unreachable!("just matched")
                        };
                        Some(expr.clone())
                    }
                    _ => continue,
                };
                // Native operations have no IR to inline or be inlined into,
                // and a reduction neither maps over its output nor yields
                // one element per consumer element.
                let (Some(consumer_module), Some(producer_module)) = (
                    consumer.version.module(),
                    groups[producer_position].version.module(),
                ) else {
                    continue;
                };
                if consumer_module.manifest.mode != orla_ir::Mode::Map
                    || producer_module.manifest.mode != orla_ir::Mode::Map
                {
                    continue;
                }
                if !links
                    .iter()
                    .all(|link| orla_ir::samples_only(consumer_module, link.1))
                {
                    continue;
                }
                let producer = &groups[producer_position];
                let mut parts = consumer.parts.clone();
                parts.extend(producer.parts.iter().cloned());
                let pairs: Vec<(u32, u32)> = links.iter().map(|link| (link.1, link.2)).collect();
                let fuse = || {
                    orla_ir::fuse_outputs(
                        consumer_module,
                        &pairs,
                        producer_module,
                        &producer.node.id.0,
                    )
                };
                let Some(version) =
                    self.fused_version(runtime, &consumer.node.id, parts.clone(), fuse)?
                else {
                    continue;
                };
                // Rewire: the fused input disappears; the producer's inputs
                // and params join under `producer/name`.
                let producer = &groups[producer_position];
                let prefix = producer.node.id.0.clone();
                let mut inputs = consumer.node.inputs.clone();
                for (linked, _, _) in &links {
                    inputs.remove(linked);
                }
                for (name, binding) in &producer.node.inputs {
                    inputs.insert(format!("{prefix}/{name}"), binding.clone());
                }
                let mut params = consumer.node.params.clone();
                for (name, value) in &producer.node.params {
                    params.insert(format!("{prefix}/{name}"), value.clone());
                }
                let producer_params: Vec<String> = producer
                    .version
                    .manifest()
                    .params
                    .iter()
                    .map(|param| param.name.clone())
                    .collect();
                let mut absorbed = consumer.absorbed.clone();
                absorbed.push(producer_id.clone());
                absorbed.extend(producer.absorbed.iter().cloned());
                let group = &mut groups[position];
                group.version = version;
                group.parts = parts;
                group.absorbed = absorbed;
                group.node.inputs = inputs;
                group.node.params = params;
                if group.node.domain.is_none() {
                    group.node.domain = match (planned, carried) {
                        (Some(domain), _) => Some(DomainSpec::Literal(domain)),
                        (None, Some(expr)) => {
                            Some(DomainSpec::Expr(expr.renamed(&prefix, &producer_params)))
                        }
                        (None, None) => None,
                    };
                }
                absorbed_into.insert(producer_id, position);
            }
            if let Some(domain) = planned {
                domains.insert(groups[position].node.id.clone(), domain);
                // The plan reports what an expression resolved to.
                if matches!(groups[position].node.domain, Some(DomainSpec::Expr(_))) {
                    groups[position].node.domain = Some(DomainSpec::Literal(domain));
                }
            }
        }
        let executing: Vec<Group> = groups
            .into_iter()
            .filter(|group| !absorbed_into.contains_key(&group.node.id))
            .collect();
        let executing = self.repeated(executing, &readers);
        // Keep every group's version while its node is in the graph: a
        // chain is planned from its head each time, so the inner groups
        // (gray with source, under inv) are needed again next time.
        let present: BTreeSet<&NodeId> = graph.nodes.iter().map(|node| &node.id).collect();
        self.fused.retain(|node, _| present.contains(node));
        self.forget_pending(runtime, |node| present.contains(node));
        Ok(executing)
    }

    /// A node placed several times over its own output is one execution
    /// that repeats. `times = N` with a `carry` writes N nodes, each
    /// taking the one before it on one port and alike in every other
    /// way; nothing between them is read by anything else, so nothing
    /// between them has to be made. The last node keeps its name and
    /// its output, as a fused consumer does.
    pub(super) fn repeated(
        &self,
        groups: Vec<Group>,
        readers: &BTreeMap<(NodeId, String), usize>,
    ) -> Vec<Group> {
        let mut out: Vec<Group> = Vec::with_capacity(groups.len());
        for group in groups {
            let joined = match out.last() {
                Some(previous) => self.chained(previous, &group, readers),
                None => None,
            };
            let Some((input, output)) = joined else {
                out.push(group);
                continue;
            };
            let previous = out.pop().expect("a group to join");
            let mut absorbed = previous.absorbed.clone();
            absorbed.push(previous.node.id.clone());
            absorbed.extend(group.absorbed.iter().cloned());
            let mut parts = previous.parts.clone();
            parts.extend(group.parts.iter().cloned());
            // The run takes the first node's inputs and the last node's
            // name: what it gives is what the last one would have.
            let mut node = group.node.clone();
            node.inputs = previous.node.inputs.clone();
            out.push(Group {
                node,
                version: group.version,
                parts,
                absorbed,
                repeat: previous.repeat + 1,
                chain: Some((input, output)),
            });
        }
        out
    }

    /// Whether `next` is `previous` again, taking what it made: the same
    /// Kernel of the same generation, one input bound to the one before
    /// and every other binding and parameter the same, nothing else
    /// reading what passes between them, and nothing pinned, fused or
    /// off the frame. The ports it chains, as indices into the
    /// manifest's inputs and outputs.
    pub(super) fn chained(
        &self,
        previous: &Group,
        next: &Group,
        readers: &BTreeMap<(NodeId, String), usize>,
    ) -> Option<(usize, usize)> {
        // The one before is either a node of its own or a run of this
        // same chain already; what follows is always a plain node.
        let joinable = match previous.chain {
            None => previous.absorbed.is_empty(),
            Some(_) => true,
        };
        if !joinable
            || previous.node.kernel != next.node.kernel
            || previous.version.generation() != next.version.generation()
            || !next.absorbed.is_empty()
            || next.repeat != 1
            || next.chain.is_some()
            || previous.node.is_async
            || next.node.is_async
            || previous.node.params != next.node.params
            || previous.node.domain != next.node.domain
            || self.pinned.contains(&previous.node.id)
        {
            return None;
        }
        let manifest = next.version.manifest();
        // Only a map: a reduction, a scan and a scatter each mean
        // something of their own about what they run over.
        if manifest.mode != orla_ir::Mode::Map || next.version.module().is_none() {
            return None;
        }
        // Exactly one port differs, and it is bound to what the one
        // before it made.
        let mut chained = None;
        for (port, binding) in &next.node.inputs {
            if previous.node.inputs.get(port) == Some(binding) {
                continue;
            }
            let Binding::Now(Source::Node { node, port: from }) = binding else {
                return None;
            };
            if node != &previous.node.id || chained.is_some() {
                return None;
            }
            if readers.get(&(node.clone(), from.clone())).copied() != Some(1) {
                return None;
            }
            let input = manifest.inputs.iter().position(|i| &i.name == port)?;
            let output = manifest.outputs.iter().position(|o| &o.name == from)?;
            chained = Some((input, output));
        }
        // Every port of the one before must be a port of this one too.
        if previous.node.inputs.len() != next.node.inputs.len() {
            return None;
        }
        // A map of several outputs repeats only when the one it chains is
        // all anyone reads of the times before the last: the run keeps
        // the last time's outputs and nothing in between.
        let (_, carried) = chained?;
        let unread = manifest.outputs.iter().enumerate().all(|(k, port)| {
            k == carried
                || readers
                    .get(&(previous.node.id.clone(), port.name.clone()))
                    .is_none_or(|count| *count == 0)
        });
        if !unread {
            return None;
        }
        chained
    }

    /// The fused version for `parts`, if it is ready: kept from an earlier
    /// evaluation, prepared inline, or collected from the runtime's worker.
    /// `None` means it is still being prepared and the nodes run separately
    /// this time.
    pub(super) fn fused_version(
        &mut self,
        runtime: &mut Runtime,
        consumer: &NodeId,
        parts: Vec<(NodeId, Generation)>,
        fuse: impl FnOnce() -> Result<orla_ir::Validated, orla_ir::FuseError>,
    ) -> Result<Option<Arc<KernelVersion>>, EvaluateError> {
        if let Some(existing) = self.fused.get(consumer)
            && existing.parts == parts
        {
            return Ok(Some(Arc::clone(&existing.version)));
        }
        if self.declined.get(consumer) == Some(&parts) {
            return Ok(None);
        }
        let fusion = |message: String| EvaluateError::Fusion {
            node: consumer.clone(),
            message,
        };
        let ticket = match self.pending.get(consumer) {
            Some(pending) if pending.parts == parts => pending.ticket,
            other => {
                if let Some(superseded) = other {
                    runtime.discard(superseded.ticket);
                }
                // What fusion declines runs unfused: a producer whose
                // Field the consumer's port cannot take (the scene's
                // mistake, which the binding check goes on to name), or a
                // Kernel whose work is counted and held to its own
                // budget. Anything else a fusion refuses is this engine's
                // own fault and is reported.
                let module = match fuse() {
                    Ok(module) => module,
                    Err(error) if error.declines => {
                        self.declined.insert(consumer.clone(), parts);
                        return Ok(None);
                    }
                    Err(error) => return Err(fusion(error.to_string())),
                };
                let source = format!(
                    "// fused: {}\n",
                    parts
                        .iter()
                        .map(|(node, generation)| format!("{node}@{generation}"))
                        .collect::<Vec<_>>()
                        .join(" + ")
                );
                let ticket = runtime.prepare_later(module, source);
                self.pending.insert(
                    consumer.clone(),
                    Pending {
                        parts: parts.clone(),
                        ticket,
                    },
                );
                ticket
            }
        };
        let Some(result) = runtime.poll(ticket) else {
            return Ok(None);
        };
        self.pending.remove(consumer);
        let candidate = result.map_err(|error: PrepareError| fusion(error.to_string()))?;
        let version = runtime.detach(candidate, fused_id(consumer));
        self.fused.insert(
            consumer.clone(),
            Fused {
                parts,
                version: Arc::clone(&version),
            },
        );
        Ok(Some(version))
    }

    /// Drops pending preparations for nodes `keep` rejects.
    pub(super) fn forget_pending(&mut self, runtime: &mut Runtime, keep: impl Fn(&NodeId) -> bool) {
        let dropped: Vec<NodeId> = self
            .pending
            .keys()
            .filter(|node| !keep(node))
            .cloned()
            .collect();
        for node in dropped {
            if let Some(pending) = self.pending.remove(&node) {
                runtime.discard(pending.ticket);
            }
        }
    }

    /// The domain a group will execute over, when it can be known before
    /// execution: explicit, or the size of its first bound image.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn planned_domain(
        &self,
        graph: &Graph,
        nodes: &BTreeMap<&NodeId, usize>,
        group: &Group,
        runtime: &Runtime,
        externals: &BTreeMap<String, Handle>,
        domains: &BTreeMap<NodeId, Domain>,
        overrides: &BTreeMap<String, Value>,
        context: &Context,
    ) -> Option<Domain> {
        let manifest = group.version.manifest();
        if manifest.mode == orla_ir::Mode::Reduce {
            return Some(Domain::Rows(1));
        }
        let size = |handle: Handle| runtime.host().domain(handle).ok();
        let plain = |source: &Source| match source {
            Source::External(name) => externals.get(name).and_then(|h| size(*h)),
            Source::Node { node, .. } => domains.get(node).copied(),
        };
        // A memory port of a node has the memory's extent, not the node's
        // domain (a scan runs over its ticks, its grid memory is a grid):
        // what it committed last, else what its initial value is sized.
        let of_source = |source: &Source| match source {
            Source::Node { node, port } => {
                match nodes
                    .get(node)
                    .and_then(|at| graph.nodes[*at].inputs.get(port))
                {
                    Some(Binding::Previous { initial, .. }) => self
                        .ledger
                        .committed(&(node.clone(), port.clone()))
                        .and_then(size)
                        .or_else(|| plain(initial)),
                    _ => plain(source),
                }
            }
            external => plain(external),
        };
        let of_binding = |binding: &Binding| match binding {
            Binding::Now(source) => of_source(source),
            Binding::Previous {
                source: Source::Node { node, port },
                initial,
                ..
            } => self
                .ledger
                .committed(&(node.clone(), port.clone()))
                .and_then(size)
                .or_else(|| of_source(initial)),
            Binding::Previous { .. } => None,
        };
        if let orla_ir::Mode::Scatter { base, .. } = manifest.mode {
            let target = manifest.inputs.get(base as usize)?;
            return of_binding(group.node.inputs.get(&target.name)?);
        }
        match &group.node.domain {
            Some(DomainSpec::Literal(domain)) => Some(*domain),
            Some(DomainSpec::Expr(expr)) => {
                let port = |name: &str| group.node.inputs.get(name).and_then(of_binding);
                let param = |name: &str| {
                    let manifest_default = manifest
                        .params
                        .iter()
                        .find(|p| p.name == name)
                        .map(|p| p.default.clone());
                    match group.node.params.get(name) {
                        Some(ParamValue::Literal(value)) => Some(value.clone()),
                        Some(ParamValue::Graph(name)) => graph_param(graph, name, overrides),
                        None => manifest_default.or_else(|| graph_param(graph, name, overrides)),
                    }
                };
                // A domain that reads a number out of a Field is not
                // knowable before that Field exists, so a group whose
                // domain does is planned as unknown, and nothing fuses
                // into it.
                expr.resolve(&port, &param, &window_context(context), &|_, _| None)
                    .ok()
            }
            None => {
                let first = manifest.inputs.first()?;
                of_binding(group.node.inputs.get(&first.name)?)
            }
        }
    }
}

/// The id a fused group executes under: `fused.<node>@1` with the node
/// name made id-safe.
pub(super) fn fused_id(node: &NodeId) -> KernelId {
    let segment: String = node
        .0
        .chars()
        .map(|c| match c.to_ascii_lowercase() {
            c @ ('a'..='z' | '0'..='9' | '_') => c,
            _ => '_',
        })
        .collect();
    let segment = if segment.starts_with(|c: char| c.is_ascii_digit()) || segment.is_empty() {
        format!("n{segment}")
    } else {
        segment
    };
    KernelId::new(&format!("fused.{segment}"), 1).expect("sanitized segment forms a valid id")
}

/// A graph parameter's value: the caller's override, or its default.
pub(super) fn graph_param(
    graph: &Graph,
    name: &str,
    overrides: &BTreeMap<String, Value>,
) -> Option<Value> {
    overrides.get(name).cloned().or_else(|| {
        graph
            .params
            .iter()
            .find(|p| p.name == name)
            .map(|p| p.default.clone())
    })
}

/// `; did you mean `x`?` where there is one; see
/// [`orla_abi::names::nearest`], which found it.
pub(super) fn suggest(near: &Option<String>) -> String {
    orla_abi::names::suggestion(near.as_deref())
}
