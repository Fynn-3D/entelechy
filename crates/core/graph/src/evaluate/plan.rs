//! What executes: every node as its own group, producers fused into
//! their sole consumers that read them pointwise or at an offset of their
//! element, readers that only count a fused producer given its extent, a
//! node placed many times run as one, and each group's domain.

use super::*;

/// The most nodes one fused group holds.
pub(super) const MAX_FUSED: usize = 64;
/// The most IR instructions a fused group may hold before further fusion
/// leaves the remaining producer separate, keeping device shader compilation
/// bounded.
const MAX_FUSED_INSTRUCTIONS: usize = 16_000;
/// The most struct words carried across branches a fused group may grow past,
/// matching the compiler's large-shader comfort bound (`5_000`).
const MAX_FUSED_CARRIED: u64 = 5_000;

impl Evaluator {
    /// The order `graph` runs in, kept while the graph is the one before.
    /// A changed graph forgets the plan made for the one before, so a plan
    /// is reused only for the graph it was made from.
    pub(super) fn order_of(
        &mut self,
        graph: &Graph,
        shared: Option<&Arc<Graph>>,
    ) -> Result<Arc<Vec<usize>>, GraphError> {
        if let Some((known, order)) = &self.shape {
            // Held, it is the one before; else it is compared whole, and
            // the same graph shared is held from here on.
            if shared.is_some_and(|shared| Arc::ptr_eq(shared, known)) {
                return Ok(Arc::clone(order));
            }
            if **known == *graph {
                let order = Arc::clone(order);
                if let Some(shared) = shared {
                    self.shape = Some((Arc::clone(shared), Arc::clone(&order)));
                }
                return Ok(order);
            }
        }
        self.shape = None;
        self.last_plan = None;
        self.planned_from = None;
        self.state_ports = None;
        self.state_keys = None;
        self.awake_nodes = None;
        self.frame_needed = None;
        self.state_needed = None;
        self.pruned = false;
        let order = Arc::new(graph.order()?);
        let held = shared.map_or_else(|| Arc::new(graph.clone()), Arc::clone);
        self.shape = Some((held, Arc::clone(&order)));
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
        // What a plan is made from besides the graph, each small: the last
        // plan stands while all of it is the same, checked in place before
        // anything is copied.
        self.ensure_state_ports(graph);
        let has_device = runtime.has_device();
        let hoist_trace = runtime.needs_host_trace_instances();
        if let Some((last, executing)) = &self.planned_from
            && last.kernels == runtime.revision()
            && last.window == context.window
            && last.scale == context.scale
            && last.fusion == self.fusion
            && last.has_device == has_device
            && last.hoist_trace == hoist_trace
            && last.pinned == self.pinned
            && last.overrides == *overrides
            && last.externals == *externals
            && self.state_extents_match(&last.states, runtime)
        {
            return Ok(Arc::clone(executing));
        }
        let states = self.state_extents(graph, runtime);
        let made = self.plan_anew(graph, order, runtime, externals, overrides, context)?;
        // A fused version still preparing joins a later plan: until none
        // is, each frame plans again.
        self.planned_from = self.pending.is_empty().then(|| {
            (
                PlanInputs {
                    kernels: runtime.revision(),
                    externals: externals.clone(),
                    overrides: overrides.clone(),
                    window: context.window,
                    scale: context.scale,
                    states,
                    pinned: self.pinned.clone(),
                    fusion: self.fusion,
                    has_device,
                    hoist_trace,
                },
                Arc::clone(&made),
            )
        });
        Ok(made)
    }

    fn ensure_state_ports(&mut self, graph: &Graph) {
        self.state_ports.get_or_insert_with(|| {
            let mut ports = Vec::new();
            for node in &graph.nodes {
                for (port, binding) in &node.inputs {
                    if let Binding::Previous {
                        source: Source::Node { node, port: from },
                        ..
                    } = binding
                    {
                        ports.push(((node.clone(), from.clone()), port.clone()));
                    }
                }
            }
            ports.sort();
            ports.dedup();
            ports
        });
    }

    fn state_extents_match(&self, expected: &[Option<Domain>], runtime: &Runtime) -> bool {
        let Some(ports) = &self.state_ports else {
            return false;
        };
        ports.len() == expected.len()
            && ports.iter().zip(expected).all(|((key, _), wanted)| {
                let current = self
                    .ledger
                    .committed(key)
                    .and_then(|handle| runtime.host().domain(handle).ok());
                current == *wanted
            })
    }

    /// The extent each port holding state has now, in order of node and
    /// port: a plan reads it for what reads the state.
    fn state_extents(&mut self, graph: &Graph, runtime: &Runtime) -> Vec<Option<Domain>> {
        self.ensure_state_ports(graph);
        self.state_ports
            .as_ref()
            .expect("initialized")
            .iter()
            .map(|(key, _)| {
                self.ledger
                    .committed(key)
                    .and_then(|handle| runtime.host().domain(handle).ok())
            })
            .collect()
    }

    /// Decides what executes, as [`Evaluator::plan`] says, for inputs it
    /// has not planned for.
    fn plan_anew(
        &mut self,
        graph: &Graph,
        order: &[usize],
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        overrides: &BTreeMap<String, Value>,
        context: &Context,
    ) -> Result<Arc<Vec<Group>>, EvaluateError> {
        let has_device = runtime.has_device();
        let hoist_trace = runtime.needs_host_trace_instances();
        let scheduled = if has_device {
            std::borrow::Cow::Owned(graph.schedule_for_device_for(order, hoist_trace))
        } else {
            std::borrow::Cow::Borrowed(order)
        };
        let order: &[usize] = &scheduled;
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
                let near =
                    entelechy_abi::names::nearest(&wanted, published.iter().map(String::as_str))
                        .map(str::to_owned);
                EvaluateError::UnknownKernel {
                    node: node.id.clone(),
                    kernel: node.kernel.clone(),
                    near,
                    refused: None,
                }
            })?;
            versions.push(version);
        }
        // Every node's binding its manifest decides, before any runs.
        for (index, version) in order.iter().zip(&versions) {
            let structs = version.module().map_or(&[][..], |module| &module.structs);
            binding_fault(graph, &graph.nodes[*index], version.manifest(), structs)?;
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
            && last.has_device == has_device
            && last.hoist_trace == hoist_trace
            && last.parts == parts
            && last.pinned == self.pinned
        {
            return Ok(Arc::clone(executing));
        }
        let key = PlanKey {
            parts,
            pinned: self.pinned.clone(),
            fusion: self.fusion,
            has_device,
            hoist_trace,
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
                    counted: Vec::new(),
                    apart: Vec::new(),
                    links: Vec::new(),
                    repeat: 1,
                    chain: None,
                    fused: None,
                    names: std::sync::OnceLock::new(),
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
        // The readers that ask an output only its count or size, each a map:
        // it can read the extent from params instead, so it holds no
        // fusion back.
        let mut counting: BTreeMap<(NodeId, String), Vec<(NodeId, String)>> = BTreeMap::new();
        for group in &groups {
            let Some(module) = group.module() else {
                continue;
            };
            if group.node.is_async
                || !group.node.awake.is_empty()
                || module.manifest.mode != entelechy_ir::Mode::Map
                || !module.manifest.memories.is_empty()
            {
                continue;
            }
            for (name, binding) in &group.node.inputs {
                let Binding::Now(Source::Node { node, port }) = binding else {
                    continue;
                };
                let input = module.manifest.inputs.iter().position(|i| &i.name == name);
                if input.is_some_and(|input| entelechy_ir::counts_only(module, input as u32)) {
                    counting
                        .entry((node.clone(), port.clone()))
                        .or_default()
                        .push((group.node.id.clone(), name.clone()));
                }
            }
        }
        let mut absorbed_into: BTreeMap<NodeId, usize> = BTreeMap::new();
        // A fused group and a reader counting what it absorbed: both run
        // their new versions, or neither does.
        let mut ties: Vec<(usize, usize)> = Vec::new();
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
                if absorbed_into.contains_key(&producer_id) {
                    continue;
                }
                // What keeps the two apart whatever the producer is read
                // as; said once this group is found its only reader.
                let consumer_id = groups[position].node.id.clone();
                let producer_node = nodes.get(&producer_id).map(|at| &graph.nodes[*at]);
                let held = if self.pinned.contains(&producer_id) {
                    Some("it is pinned".to_owned())
                } else if groups[position].node.is_async {
                    Some(format!("`{consumer_id}` runs async"))
                } else if !groups[position].node.awake.is_empty() {
                    Some(format!("`{consumer_id}` sleeps by `awake`"))
                } else if producer_node.is_some_and(|node| node.is_async) {
                    Some("it runs async".to_owned())
                } else if producer_node.is_some_and(|node| !node.awake.is_empty()) {
                    Some("it sleeps by `awake`".to_owned())
                } else {
                    None
                };
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
                // The readers that only count the producer outside this
                // group: each at the group it is in now, under the port it
                // reads through there, and the output it counts.
                let mut outside: Vec<(usize, String, u32)> = Vec::new();
                for (k, out) in producer_manifest.outputs.iter().enumerate() {
                    let key = (producer_id.clone(), out.name.clone());
                    for (reader, port) in counting.get(&key).into_iter().flatten() {
                        let mut owner = at[reader];
                        while let Some(next) = absorbed_into.get(&originals[owner].node.id) {
                            owner = *next;
                        }
                        if owner == position {
                            continue;
                        }
                        let name = if owner == at[reader] {
                            port.clone()
                        } else {
                            entelechy_ir::fused_name(&reader.0, port)
                        };
                        outside.push((owner, name, k as u32));
                    }
                }
                let only_readers = producer_manifest
                    .outputs
                    .iter()
                    .enumerate()
                    .all(|(k, out)| {
                        let linked = links.iter().filter(|link| link.2 == k as u32).count();
                        let counted = outside.iter().filter(|o| o.2 == k as u32).count();
                        let read = readers
                            .get(&(producer_id.clone(), out.name.clone()))
                            .copied()
                            .unwrap_or(0);
                        read == linked + counted
                    });
                if links.is_empty() || !only_readers {
                    continue;
                }
                // From here this group is the producer's only reader: what
                // keeps the two apart is said in its plan.
                if let Some(why) = held {
                    keep_apart(&mut groups[position], &producer_id, why);
                    continue;
                }
                // A narrow output rounds its values, so it stays
                // materialized.
                let narrow = links.iter().any(|link| {
                    !producer_manifest.outputs[link.2 as usize]
                        .ty
                        .schema
                        .is_wide()
                });
                if narrow {
                    let why = "it keeps its scalars narrow, rounded as they pass".to_owned();
                    keep_apart(&mut groups[position], &producer_id, why);
                    continue;
                }
                // Only IR maps fuse: native operations have no IR, and a
                // reduction is not one element per consumer element.
                let (Some(consumer_module), Some(producer_module)) =
                    (consumer.module(), groups[producer_position].module())
                else {
                    let native = if consumer.module().is_none() {
                        format!("`{consumer_id}`")
                    } else {
                        "it".to_owned()
                    };
                    let why = format!("{native} is a native operation, which has no IR to fuse");
                    keep_apart(&mut groups[position], &producer_id, why);
                    continue;
                };
                if consumer_module.manifest.mode != entelechy_ir::Mode::Map
                    || producer_module.manifest.mode != entelechy_ir::Mode::Map
                {
                    let why = "only maps fuse; a reduction or a scan runs on its own".to_owned();
                    keep_apart(&mut groups[position], &producer_id, why);
                    continue;
                }
                // Read at its own element, the producer runs first; read at
                // rows, the producer is called at each.
                let uses: Vec<entelechy_ir::InputUse> = links
                    .iter()
                    .map(|link| entelechy_ir::input_use(consumer_module, link.1))
                    .collect();
                let indexed = if uses.iter().all(|use_| !use_.addressed)
                    && uses.iter().any(|use_| use_.sampled)
                {
                    false
                } else if uses.iter().any(|use_| use_.addressed)
                    && links.iter().all(|link| {
                        entelechy_ir::row_reads(consumer_module, link.1).is_some()
                            || entelechy_ir::counts_only(consumer_module, link.1)
                    })
                {
                    // Read at one offset of the reader's element, no row
                    // is made twice. Read wherever the reader's work
                    // leads, the producer is called once for each asking:
                    // that costs less than storing it while the askings
                    // are no more than the rows it would make, both known
                    // before the frame, and the rows are more than a few.
                    let offsets = links.iter().all(|link| {
                        entelechy_ir::offset_read(consumer_module, link.1)
                            || entelechy_ir::counts_only(consumer_module, link.1)
                    });
                    if !offsets {
                        let places: u64 = links
                            .iter()
                            .filter_map(|link| entelechy_ir::row_reads(consumer_module, link.1))
                            .map(|places| places as u64)
                            .sum();
                        let asked = runs.map(|domain| domain.count() as u64 * places);
                        let made = producer_domain.map(|domain| domain.count() as u64);
                        match (asked, made) {
                            // A few rows cost nothing to store, and stored
                            // they stop what reads them where they come
                            // out as they were.
                            (Some(_), Some(made)) if made <= entelechy_kernel::FEW_ELEMENTS => {
                                let why = format!(
                                    "it makes {made} rows, too few for calling it to spare anything"
                                );
                                keep_apart(&mut groups[position], &producer_id, why);
                                continue;
                            }
                            (Some(asked), Some(made)) if asked <= made => {}
                            (Some(asked), Some(made)) => {
                                let why = format!(
                                    "`{consumer_id}` asks it for {asked} rows and it makes {made}: stored, each is made once"
                                );
                                keep_apart(&mut groups[position], &producer_id, why);
                                continue;
                            }
                            _ => {
                                let why = format!(
                                    "how many rows `{consumer_id}` asks it for, or how many it makes, is not known before the frame"
                                );
                                keep_apart(&mut groups[position], &producer_id, why);
                                continue;
                            }
                        }
                    }
                    true
                } else {
                    let why = format!(
                        "`{consumer_id}` reads it neither at its own element (`sample`) nor at rows its entry names outside any loop"
                    );
                    keep_apart(&mut groups[position], &producer_id, why);
                    continue;
                };
                // The producer's extent comes in by params wherever it is
                // read at an offset or counted, so it must be known now.
                let by_params =
                    indexed || !outside.is_empty() || uses.iter().any(|use_| use_.counted);
                let known = producer_domain.filter(|domain| domain.count() > 0);
                if by_params && known.is_none() {
                    let why = "its extent is not known before the frame".to_owned();
                    keep_apart(&mut groups[position], &producer_id, why);
                    continue;
                }
                // Whether the producer feeds the consumer's first input,
                // which sets its extent when it has no `domain`.
                let reads_first = links.iter().any(|link| link.1 == 0);
                // A consumer with no `domain` runs over its first input's
                // extent; when that input is the producer, the extents
                // match and the producer's expression moves onto the
                // fused node under the fused port names. A consumer that
                // reads at an offset runs over its own, known now.
                let carried = match (producer_domain, runs) {
                    (Some(_), _) if indexed => None,
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
                    _ => {
                        let why =
                            "the two do not run over one extent known before the frame".to_owned();
                        keep_apart(&mut groups[position], &producer_id, why);
                        continue;
                    }
                };
                // A consumer read at an offset keeps its own domain, its
                // counts of the fused ports read from the extent now.
                let linked: Vec<String> = links.iter().map(|link| link.0.clone()).collect();
                let indexed_domain = match known {
                    Some(extent) if indexed => {
                        match without_ports(&consumer.node.domain, runs, &linked, extent) {
                            Some(domain) => Some(domain),
                            None => {
                                let why = format!(
                                    "`{consumer_id}`'s domain cannot be said without the ports it would fuse"
                                );
                                keep_apart(&mut groups[position], &producer_id, why);
                                continue;
                            }
                        }
                    }
                    _ => None,
                };
                let producer = &groups[producer_position];
                // A fused group holds a bounded number of nodes, so a long
                // chain fuses into several, each compiled in bounded time.
                if consumer.parts.len() + producer.parts.len() > MAX_FUSED {
                    let why = format!("together they would hold more than {MAX_FUSED} nodes");
                    keep_apart(&mut groups[position], &producer_id, why);
                    continue;
                }
                let mut parts = consumer.parts.clone();
                parts.extend(producer.parts.iter().cloned());
                let mut counted = consumer.counted.clone();
                counted.extend(producer.counted.iter().cloned());
                let pairs: Vec<(u32, u32)> = links.iter().map(|link| (link.1, link.2)).collect();
                // Which ports were joined is part of what the fused version
                // is: the same members joined otherwise fuse anew.
                let mut fused_links = consumer.links.clone();
                fused_links.extend(producer.links.iter().cloned());
                fused_links.push((producer.node.id.clone(), pairs.clone()));
                let members = Members {
                    parts: parts.clone(),
                    counted: counted.clone(),
                    links: fused_links.clone(),
                };
                // The IR fuses here; only a group that executes is
                // prepared, once the plan is made.
                if let Some((_, why)) = self
                    .declined
                    .get(&consumer_id)
                    .filter(|(declined, _)| *declined == members)
                {
                    let why = why.clone();
                    keep_apart(&mut groups[position], &producer_id, why);
                    continue;
                }
                let fuse = if indexed {
                    entelechy_ir::fuse_indexed
                } else {
                    entelechy_ir::fuse_outputs
                };
                let module = match fuse(
                    consumer_module,
                    &pairs,
                    producer_module,
                    &producer.node.id.0,
                ) {
                    Ok(module) => {
                        let instructions = entelechy_ir::sizes(&module).total();
                        let carried_after = entelechy_ir::carried(&module).total;
                        let carried_before = entelechy_ir::carried(consumer_module).total;
                        if instructions > MAX_FUSED_INSTRUCTIONS {
                            let why = format!(
                                "together they would hold {instructions} instructions, past {MAX_FUSED_INSTRUCTIONS}"
                            );
                            self.declined
                                .insert(consumer_id.clone(), (members, why.clone()));
                            keep_apart(&mut groups[position], &producer_id, why);
                            continue;
                        }
                        if carried_after > MAX_FUSED_CARRIED && carried_after > carried_before {
                            let why = format!(
                                "together they would carry {carried_after} struct words through branches, past {MAX_FUSED_CARRIED}"
                            );
                            self.declined
                                .insert(consumer_id.clone(), (members, why.clone()));
                            keep_apart(&mut groups[position], &producer_id, why);
                            continue;
                        }
                        Arc::new(module)
                    }
                    Err(error) if error.declines => {
                        let why = error.to_string();
                        self.declined
                            .insert(consumer_id.clone(), (members, why.clone()));
                        keep_apart(&mut groups[position], &producer_id, why);
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
                    inputs.insert(entelechy_ir::fused_name(&prefix, name), binding.clone());
                }
                let mut params = consumer.node.params.clone();
                for (name, value) in &producer.node.params {
                    params.insert(entelechy_ir::fused_name(&prefix, name), value.clone());
                }
                let producer_params: Vec<String> = producer
                    .manifest()
                    .params
                    .iter()
                    .map(|param| param.name.clone())
                    .collect();
                // Each reader counting the producer elsewhere drops the
                // Field for the extent's params, its whole group at once.
                let mut recounted: BTreeMap<usize, Vec<(u32, String)>> = BTreeMap::new();
                let mut blocked = false;
                for (owner, name, _) in &outside {
                    let group = &groups[*owner];
                    let bound = matches!(
                        group.node.inputs.get(name),
                        Some(Binding::Now(Source::Node { node, .. })) if node == &producer_id
                    );
                    let input = group.manifest().inputs.iter().position(|i| &i.name == name);
                    match (bound, input, group.module()) {
                        (true, Some(input), Some(module))
                            if entelechy_ir::counts_only(module, input as u32) =>
                        {
                            recounted
                                .entry(*owner)
                                .or_default()
                                .push((input as u32, name.clone()));
                        }
                        _ => blocked = true,
                    }
                }
                // Each such reader's domain, its counts of the producer
                // read from the extent now.
                let mut reader_domains = BTreeMap::new();
                for (owner, inputs) in &recounted {
                    let gone: Vec<String> = inputs.iter().map(|(_, name)| name.clone()).collect();
                    let extent = known.expect("a counted producer's extent is known");
                    match without_ports(
                        &groups[*owner].node.domain,
                        plans[*owner].runs,
                        &gone,
                        extent,
                    ) {
                        Some(domain) => {
                            reader_domains.insert(*owner, domain);
                        }
                        None => blocked = true,
                    }
                }
                if blocked {
                    continue;
                }
                let mut readers_now = Vec::with_capacity(recounted.len());
                for (owner, inputs) in &recounted {
                    let group = &groups[*owner];
                    let module = group.module().expect("checked above");
                    let prefixed: Vec<(u32, String)> = inputs
                        .iter()
                        .map(|(input, _)| (*input, producer_id.0.to_string()))
                        .collect();
                    let module = entelechy_ir::counted(module, &prefixed).map_err(|error| {
                        EvaluateError::Fusion {
                            node: group.node.id.clone(),
                            message: error.to_string(),
                        }
                    })?;
                    readers_now.push((*owner, Arc::new(module)));
                }
                let given = known.map(extent_values);
                if let Some(given) = &given {
                    give_extent(&mut params, &module.manifest, &prefix, given);
                }
                let mut absorbed = consumer.absorbed.clone();
                absorbed.push(producer_id.clone());
                absorbed.extend(producer.absorbed.iter().cloned());
                // What the producer kept apart is now the group's to say.
                let mut apart = consumer.apart.clone();
                apart.extend(producer.apart.iter().cloned());
                let group = &mut groups[position];
                group.fused = Some(module);
                group.parts = parts;
                group.counted = counted;
                group.links = fused_links;
                group.absorbed = absorbed;
                group.apart = apart;
                group.node.inputs = inputs;
                group.node.params = params;
                if let Some(domain) = indexed_domain {
                    group.node.domain = domain;
                } else if group.node.domain.is_none() {
                    group.node.domain = match (runs, carried) {
                        (Some(domain), _) => Some(DomainSpec::Literal(domain)),
                        (None, Some(expr)) => {
                            Some(DomainSpec::Expr(expr.renamed(&prefix, &producer_params)))
                        }
                        (None, None) => None,
                    };
                }
                for (owner, module) in readers_now {
                    let reader = &mut groups[owner];
                    for (_, name) in &recounted[&owner] {
                        reader.node.inputs.remove(name);
                    }
                    let given = given
                        .as_ref()
                        .expect("a counted producer's extent is known");
                    give_extent(&mut reader.node.params, &module.manifest, &prefix, given);
                    reader.node.domain = reader_domains.remove(&owner).expect("found above");
                    reader.counted.push(producer_id.clone());
                    reader.fused = Some(module);
                    ties.push((position, owner));
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
            let members = group.members();
            let beside = Arc::clone(&group.version);
            match self.fused_version(runtime, &id, members, &beside, || Ok((*module).clone()))? {
                Some(version) => group.version = version,
                None => {
                    settled = false;
                    unfused.insert(position);
                }
            }
        }
        // A group still preparing holds back every group tied to it.
        let mut changed = true;
        while changed {
            changed = false;
            for (a, b) in &ties {
                let (a, b) = (owner(*a), owner(*b));
                if unfused.contains(&a) != unfused.contains(&b) {
                    unfused.insert(a);
                    unfused.insert(b);
                    changed = true;
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
            let mut counted = previous.counted;
            counted.extend(group.counted);
            let mut apart = previous.apart;
            apart.extend(group.apart);
            let mut links = previous.links;
            links.extend(group.links);
            out.push(Group {
                node,
                version: group.version,
                parts,
                absorbed,
                counted,
                apart,
                links,
                repeat: previous.repeat + 1,
                chain: Some((input, output)),
                fused: None,
                names: std::sync::OnceLock::new(),
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
            || previous.node.awake != next.node.awake
            || previous.node.params != next.node.params
            || previous.node.domain != next.node.domain
            || self.pinned.contains(&previous.node.id)
        {
            return None;
        }
        let manifest = next.version.manifest();
        // Only a map repeats.
        if manifest.mode != entelechy_ir::Mode::Map || next.version.module().is_none() {
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
        members: Members,
        beside: &KernelVersion,
        fuse: impl FnOnce() -> Result<entelechy_ir::Validated, entelechy_ir::FuseError>,
    ) -> Result<Option<Arc<KernelVersion>>, EvaluateError> {
        if let Some(existing) = self.fused.get(consumer)
            && existing.members == members
        {
            return Ok(Some(Arc::clone(&existing.version)));
        }
        if self
            .declined
            .get(consumer)
            .is_some_and(|(declined, _)| *declined == members)
        {
            return Ok(None);
        }
        let fusion = |message: String| EvaluateError::Fusion {
            node: consumer.clone(),
            message,
        };
        let ticket = match self.pending.get(consumer) {
            Some(pending) if pending.members == members => pending.ticket,
            other => {
                if let Some(superseded) = other {
                    runtime.discard(superseded.ticket);
                }
                // A declined fusion (a Field the port cannot take, a
                // budgeted Kernel) runs unfused, as does one the backend
                // refuses below; any other refusal is an engine error.
                let module = match fuse() {
                    Ok(module) => module,
                    Err(error) if error.declines => {
                        self.declined
                            .insert(consumer.clone(), (members, error.to_string()));
                        return Ok(None);
                    }
                    Err(error) => return Err(fusion(error.to_string())),
                };
                // The group is prepared, built and timed under the id it
                // runs under, not its consumer's.
                let mut module = module.into_module();
                module.manifest.id = fused_id(consumer);
                let module =
                    entelechy_ir::validate(module).map_err(|error| fusion(error.to_string()))?;
                let source = format!(
                    "// fused: {}\n",
                    members
                        .parts
                        .iter()
                        .map(|(node, generation)| format!("{node}@{generation}"))
                        .chain(
                            members
                                .counted
                                .iter()
                                .map(|node| format!("count of {node}"))
                        )
                        .collect::<Vec<_>>()
                        .join(" + ")
                );
                let ticket = runtime.prepare_later_beside(module, source, beside);
                self.pending.insert(
                    consumer.clone(),
                    Pending {
                        members: members.clone(),
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
        let candidate = match result {
            Ok(candidate) => candidate,
            // The backend took each member alone but refuses the group (a
            // processor's stack, a device's limits): it runs unfused, as a
            // fusion the IR declines does.
            Err(PrepareError::Backend(reason)) => {
                let why = format!("the backend refuses them together: {reason}");
                self.refused.push(Refused {
                    node: consumer.clone(),
                    members: members.parts.iter().map(|(node, _)| node.clone()).collect(),
                    reason,
                });
                self.declined.insert(consumer.clone(), (members, why));
                return Ok(None);
            }
            Err(error) => return Err(fusion(error.to_string())),
        };
        let version = runtime.detach(candidate, fused_id(consumer));
        self.fused.insert(
            consumer.clone(),
            Fused {
                members,
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
    /// where [`entelechy_kernel::extent`] says its mode takes it from.
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
        let runs = if manifest.mode == entelechy_ir::Mode::Reduce {
            Some(Domain::Rows(1))
        } else {
            self.planned_runs(graph, node, version, &of_binding, overrides, context)
                .or_else(|| node.domain.is_none().then(|| input(0)).flatten())
        };
        let extent = match entelechy_kernel::extent(manifest.mode) {
            entelechy_kernel::Extent::Invocation => runs,
            entelechy_kernel::Extent::OneRow => Some(Domain::Rows(1)),
            entelechy_kernel::Extent::Input(at) => input(at),
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

/// What a plan reads besides the graph, each small to compare: the
/// published Kernels' revision, the caller's Fields and parameters, the
/// window, the extent of each port holding state, the pins and whether
/// fusion is on. The same graph and the same inputs make the same plan.
#[derive(PartialEq)]
pub(super) struct PlanInputs {
    kernels: u64,
    externals: BTreeMap<String, Handle>,
    overrides: BTreeMap<String, Value>,
    window: [f32; 2],
    scale: f32,
    states: Vec<Option<Domain>>,
    pinned: BTreeSet<NodeId>,
    fusion: bool,
    has_device: bool,
    hoist_trace: bool,
}

/// What a plan was made from besides the graph: each node's law and
/// planned extents in order, the pins, and whether fusion is on.
pub(super) struct PlanKey {
    parts: Vec<(Generation, Planned)>,
    pinned: BTreeSet<NodeId>,
    fusion: bool,
    has_device: bool,
    hoist_trace: bool,
}

/// What the plan knows of a group before execution.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Planned {
    /// The elements it runs over.
    pub runs: Option<Domain>,
    /// The extent its outputs have, which its readers see.
    pub extent: Option<Domain>,
}

/// The values of a producer's extent params ([`entelechy_ir::extent_params`]):
/// its element count and its extent, as `count` and `size` give them.
fn extent_values(domain: Domain) -> [Value; 2] {
    let size = match domain {
        Domain::Rows(n) => Value::Int(n as i32),
        Domain::Grid(w, h) => Value::Vec2([w as f32, h as f32]),
        Domain::Volume(w, h, d) => Value::Vec3([w as f32, h as f32, d as f32]),
    };
    [Value::Int(domain.count() as i32), size]
}

/// A node's domain once its ports `gone` leave it, each bound to a
/// producer of extent `extent`: what it resolves to when known now, else
/// its expression with their counts and sizes read from that extent;
/// `None` when it reads a value of one, which only the Field holds.
fn without_ports(
    domain: &Option<DomainSpec>,
    runs: Option<Domain>,
    gone: &[String],
    extent: Domain,
) -> Option<Option<DomainSpec>> {
    if let Some(runs) = runs {
        return Some(Some(DomainSpec::Literal(runs)));
    }
    match domain {
        Some(DomainSpec::Expr(expr)) => {
            if expr
                .values_read()
                .iter()
                .any(|(port, _)| gone.contains(port))
            {
                return None;
            }
            let known = |port: &str| gone.iter().any(|g| g == port).then_some(extent);
            Some(Some(DomainSpec::Expr(expr.with_extents(&known))))
        }
        other => Some(other.clone()),
    }
}

/// Sets the extent params of the producer `prefix` that `manifest` reads.
fn give_extent(
    params: &mut BTreeMap<String, ParamValue>,
    manifest: &entelechy_kernel::Manifest,
    prefix: &str,
    values: &[Value; 2],
) {
    for (name, value) in entelechy_ir::extent_params(prefix).into_iter().zip(values) {
        if manifest.params.iter().any(|param| param.name == name) {
            params.insert(name, ParamValue::Literal(value.clone()));
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

/// `; did you mean `x`?` for [`entelechy_abi::names::nearest`]'s match.
pub(super) fn suggest(near: &Option<String>) -> String {
    entelechy_abi::names::suggestion(near.as_deref())
}

/// Says in `group`'s plan that `producer`, which only it reads, runs
/// apart from it, and why; once a producer.
fn keep_apart(group: &mut Group, producer: &NodeId, why: String) {
    if !group.apart.iter().any(|(node, _)| node == producer) {
        group.apart.push((producer.clone(), why));
    }
}
