//! What executes: every node as its own group, producers fused into
//! their sole pointwise consumers, a node placed many times run as one,
//! and each group's domain.

use super::*;

impl Evaluator {
    /// The order `graph` runs in, kept while the graph is the one before.
    /// A changed graph forgets the plan made for the one before, so a plan
    /// is reused only for the graph it was made from.
    pub(super) fn order_of(&mut self, graph: &Graph) -> Result<Arc<Vec<usize>>, GraphError> {
        if let Some((known, order)) = &self.shape
            && known == graph
        {
            return Ok(Arc::clone(order));
        }
        self.shape = None;
        self.last_plan = None;
        let order = Arc::new(graph.order()?);
        self.shape = Some((graph.clone(), Arc::clone(&order)));
        Ok(order)
    }

    /// Decides what executes: every node as its own group, then producers
    /// folded into their sole pointwise consumers when fusion is on.
    /// `order` is [`Evaluator::order_of`] this graph.
    pub(super) fn plan(
        &mut self,
        graph: &Graph,
        order: &[usize],
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        overrides: &BTreeMap<String, Value>,
        context: &Context,
    ) -> Result<Arc<Vec<Group>>, EvaluateError> {
        let mut versions = Vec::with_capacity(order.len());
        for index in order {
            let node = &graph.nodes[*index];
            let version = runtime.kernel(&node.kernel).ok_or_else(|| {
                // Everything published, for the suggestion.
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
            versions.push(version);
        }
        let nodes = graph.index();
        // What each node runs over and the extent it gives, in order: a
        // producer's extent is known before its readers are planned.
        let mut domains: BTreeMap<NodeId, Domain> = BTreeMap::new();
        let mut plans = Vec::with_capacity(order.len());
        for (index, version) in order.iter().zip(&versions) {
            let node = &graph.nodes[*index];
            let planned = self.planned(
                graph, &nodes, node, version, runtime, externals, &domains, overrides, context,
            );
            if let Some(extent) = planned.extent {
                domains.insert(node.id.clone(), extent);
            }
            plans.push(planned);
        }
        // The same graph, laws, extents, pins and fusion make the same
        // plan: the last one is reused, and fusion is decided once, not
        // each frame. The graph is the one the last plan was made from
        // ([`Evaluator::order_of`]); the rest is compared here.
        let parts: Vec<(Generation, Planned)> = versions
            .iter()
            .zip(&plans)
            .map(|(version, planned)| (version.generation(), planned.clone()))
            .collect();
        if let Some((last, executing)) = &self.last_plan
            && last.fusion == self.fusion
            && last.parts == parts
            && last.pinned == self.pinned
        {
            return Ok(Arc::clone(executing));
        }
        let key = PlanKey {
            parts,
            pinned: self.pinned.clone(),
            fusion: self.fusion,
        };
        let mut groups: Vec<Group> = order
            .iter()
            .zip(versions)
            .map(|(index, version)| {
                let node = &graph.nodes[*index];
                Group {
                    node: node.clone(),
                    parts: vec![(node.id.clone(), version.generation())],
                    version,
                    absorbed: Vec::new(),
                    repeat: 1,
                    chain: None,
                    fused: None,
                }
            })
            .collect();
        if !self.fusion {
            self.fused.clear();
            self.forget_pending(runtime, |_| false);
            let executing = Arc::new(groups);
            self.last_plan = Some((key, Arc::clone(&executing)));
            return Ok(executing);
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
        let mut absorbed_into: BTreeMap<NodeId, usize> = BTreeMap::new();
        // Each group's position by node; groups are only rewired here,
        // so positions hold.
        let at: BTreeMap<NodeId, usize> = groups
            .iter()
            .enumerate()
            .map(|(position, group)| (group.node.id.clone(), position))
            .collect();
        let originals = groups.clone();
        for position in 0..groups.len() {
            // The elements this group runs over and the extent its
            // readers see, where known before execution.
            let Planned { runs, extent } = plans[position].clone();
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
                // fuses only when these are all its readers.
                let consumer_manifest = consumer.manifest();
                let producer_manifest = groups[producer_position].manifest();
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
                // A narrow output rounds its values, so it stays
                // materialized.
                let narrow = links.iter().any(|link| {
                    !producer_manifest.outputs[link.2 as usize]
                        .ty
                        .schema
                        .is_wide()
                });
                if links.is_empty() || !only_readers || narrow {
                    continue;
                }
                // Whether the producer feeds the consumer's first input,
                // which sets its extent when it has no `domain`.
                let reads_first = links.iter().any(|link| link.1 == 0);
                // A consumer with no `domain` runs over its first input's
                // extent; when that input is the producer, the extents
                // match and the producer's expression moves onto the
                // fused node under the fused port names.
                let carried = match (producer_domain, runs) {
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
                // Only IR maps fuse: native operations have no IR, and a
                // reduction is not one element per consumer element.
                let (Some(consumer_module), Some(producer_module)) =
                    (consumer.module(), groups[producer_position].module())
                else {
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
                // The IR fuses here; only a group that executes is
                // prepared, once the plan is made.
                if self.declined.get(&consumer.node.id) == Some(&parts) {
                    continue;
                }
                let module = match orla_ir::fuse_outputs(
                    consumer_module,
                    &pairs,
                    producer_module,
                    &producer.node.id.0,
                ) {
                    Ok(module) => Arc::new(module),
                    Err(error) if error.declines => {
                        self.declined.insert(consumer.node.id.clone(), parts);
                        continue;
                    }
                    Err(error) => {
                        return Err(EvaluateError::Fusion {
                            node: consumer.node.id.clone(),
                            message: error.to_string(),
                        });
                    }
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
                    .manifest()
                    .params
                    .iter()
                    .map(|param| param.name.clone())
                    .collect();
                let mut absorbed = consumer.absorbed.clone();
                absorbed.push(producer_id.clone());
                absorbed.extend(producer.absorbed.iter().cloned());
                let group = &mut groups[position];
                group.fused = Some(module);
                group.parts = parts;
                group.absorbed = absorbed;
                group.node.inputs = inputs;
                group.node.params = params;
                if group.node.domain.is_none() {
                    group.node.domain = match (runs, carried) {
                        (Some(domain), _) => Some(DomainSpec::Literal(domain)),
                        (None, Some(expr)) => {
                            Some(DomainSpec::Expr(expr.renamed(&prefix, &producer_params)))
                        }
                        (None, None) => None,
                    };
                }
                absorbed_into.insert(producer_id, position);
                // What the producer fused is inside the consumer's now.
                groups[producer_position].fused = None;
            }
            if let Some(extent) = extent {
                domains.insert(groups[position].node.id.clone(), extent);
            }
            // The plan reports what an expression resolved to.
            if let Some(runs) = runs
                && matches!(groups[position].node.domain, Some(DomainSpec::Expr(_)))
            {
                groups[position].node.domain = Some(DomainSpec::Literal(runs));
            }
        }
        // Each fused group executing is prepared now; one still being
        // prepared runs unfused this frame, its members as they were.
        let owners: Vec<usize> = (0..originals.len())
            .map(|mut position| {
                while let Some(next) = absorbed_into.get(&originals[position].node.id) {
                    position = *next;
                }
                position
            })
            .collect();
        let owner = |position: usize| owners[position];
        let mut settled = true;
        let mut unfused = BTreeSet::new();
        for (position, group) in groups.iter_mut().enumerate() {
            if owner(position) != position {
                continue;
            }
            let Some(module) = group.fused.take() else {
                continue;
            };
            let id = group.node.id.clone();
            let parts = group.parts.clone();
            match self.fused_version(runtime, &id, parts, || Ok((*module).clone()))? {
                Some(version) => group.version = version,
                None => {
                    settled = false;
                    unfused.insert(position);
                }
            }
        }
        let mut groups: Vec<Option<Group>> = groups.into_iter().map(Some).collect();
        let executing: Vec<Group> = (0..groups.len())
            .filter_map(|position| {
                let at = owner(position);
                if unfused.contains(&at) {
                    Some(originals[position].clone())
                } else if at == position {
                    groups[position].take()
                } else {
                    None
                }
            })
            .collect();
        let executing = self.repeated(executing, &readers);
        // Only the fused versions this plan runs are kept.
        let kept: BTreeSet<NodeId> = (0..groups.len())
            .filter(|position| owner(*position) == *position)
            .map(|position| originals[position].node.id.clone())
            .collect();
        self.fused.retain(|node, _| kept.contains(node));
        self.forget_pending(runtime, |node| kept.contains(node));
        let executing = Arc::new(executing);
        self.last_plan = settled.then(|| (key, Arc::clone(&executing)));
        Ok(executing)
    }

    /// Runs `times = N` with a `carry` (N nodes, each chained to the one
    /// before on one port) as one repeated execution; intermediates are
    /// never made. The last node keeps its name and output.
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
            // The run so far is taken whole, not copied, so joining N
            // nodes costs N.
            let previous = out.pop().expect("a group to join");
            let mut absorbed = previous.absorbed;
            absorbed.push(previous.node.id);
            absorbed.extend(group.absorbed);
            let mut parts = previous.parts;
            parts.extend(group.parts);
            // The run takes the first node's inputs and the last's name.
            let mut node = group.node;
            node.inputs = previous.node.inputs;
            out.push(Group {
                node,
                version: group.version,
                parts,
                absorbed,
                repeat: previous.repeat + 1,
                chain: Some((input, output)),
                fused: None,
            });
        }
        out
    }

    /// The chained ports (manifest input and output indices) when `next`
    /// repeats `previous`: same Kernel and generation, one input bound to
    /// `previous`, everything else equal, no other readers in between,
    /// and nothing pinned, fused or off the frame.
    pub(super) fn chained(
        &self,
        previous: &Group,
        next: &Group,
        readers: &BTreeMap<(NodeId, String), usize>,
    ) -> Option<(usize, usize)> {
        // `previous` is a node or a run of this chain; `next` is a node.
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
        // Only a map repeats.
        if manifest.mode != orla_ir::Mode::Map || next.version.module().is_none() {
            return None;
        }
        // Exactly one port differs, bound to the previous output.
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
        // A map of several outputs repeats only when the chained output
        // is the only one read before the last time.
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

    /// The fused version for `parts` when ready (kept, prepared inline, or
    /// collected from the worker); `None` while it is still preparing.
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
                // A declined fusion (a Field the port cannot take, a
                // budgeted Kernel) runs unfused; any other refusal is an
                // engine error.
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

    /// What a group runs over and the extent its outputs have, each where
    /// it can be known before execution: it runs over its `domain`, or
    /// its first input's extent (a reduction over one row, as
    /// [`Evaluator::bind`] binds it), and its outputs' extent is
    /// where [`orla_kernel::extent`] says its mode takes it from.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn planned(
        &self,
        graph: &Graph,
        nodes: &BTreeMap<&NodeId, usize>,
        node: &Node,
        version: &KernelVersion,
        runtime: &Runtime,
        externals: &BTreeMap<String, Handle>,
        domains: &BTreeMap<NodeId, Domain>,
        overrides: &BTreeMap<String, Value>,
        context: &Context,
    ) -> Planned {
        let manifest = version.manifest();
        let size = |handle: Handle| runtime.host().domain(handle).ok();
        let plain = |source: &Source| match source {
            Source::External(name) => externals.get(name).and_then(|h| size(*h)),
            Source::Node { node, .. } => domains.get(node).copied(),
        };
        // A memory port has the memory's extent: its last commit, else
        // its initial value's.
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
        let input = |at: usize| {
            manifest
                .inputs
                .get(at)
                .and_then(|port| node.inputs.get(&port.name))
                .and_then(of_binding)
        };
        let runs = if manifest.mode == orla_ir::Mode::Reduce {
            Some(Domain::Rows(1))
        } else {
            self.planned_runs(graph, node, version, &of_binding, overrides, context)
                .or_else(|| node.domain.is_none().then(|| input(0)).flatten())
        };
        let extent = match orla_kernel::extent(manifest.mode) {
            orla_kernel::Extent::Invocation => runs,
            orla_kernel::Extent::OneRow => Some(Domain::Rows(1)),
            orla_kernel::Extent::Input(at) => input(at),
        };
        Planned { runs, extent }
    }

    /// The domain a group's `domain` resolves to before execution; `None`
    /// without one, or when it reads a Field's value.
    fn planned_runs(
        &self,
        graph: &Graph,
        node: &Node,
        version: &KernelVersion,
        of_binding: &dyn Fn(&Binding) -> Option<Domain>,
        overrides: &BTreeMap<String, Value>,
        context: &Context,
    ) -> Option<Domain> {
        let manifest = version.manifest();
        match &node.domain {
            Some(DomainSpec::Literal(domain)) => Some(*domain),
            Some(DomainSpec::Expr(expr)) => {
                let port = |name: &str| node.inputs.get(name).and_then(of_binding);
                let param = |name: &str| {
                    let manifest_default = manifest
                        .params
                        .iter()
                        .find(|p| p.name == name)
                        .map(|p| p.default.clone());
                    match node.params.get(name) {
                        Some(ParamValue::Literal(value)) => Some(value.clone()),
                        Some(ParamValue::Graph(name)) => graph_param(graph, name, overrides),
                        None => manifest_default.or_else(|| graph_param(graph, name, overrides)),
                    }
                };
                // A domain that reads a Field's value is unknown before
                // execution, so nothing fuses into it.
                expr.resolve(&port, &param, &window_context(context), &|_, _| None)
                    .ok()
            }
            None => None,
        }
    }
}

/// What a plan was made from besides the graph: each node's law and
/// planned extents in order, the pins, and whether fusion is on.
pub(super) struct PlanKey {
    parts: Vec<(Generation, Planned)>,
    pinned: BTreeSet<NodeId>,
    fusion: bool,
}

/// What the plan knows of a group before execution.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Planned {
    /// The elements it runs over.
    pub runs: Option<Domain>,
    /// The extent its outputs have, which its readers see.
    pub extent: Option<Domain>,
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

/// `; did you mean `x`?` for [`orla_abi::names::nearest`]'s match.
pub(super) fn suggest(near: &Option<String>) -> String {
    orla_abi::names::suggestion(near.as_deref())
}
