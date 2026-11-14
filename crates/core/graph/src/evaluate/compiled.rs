//! Plan-time compiled slot bindings for each executing group and graph output,
//! turning per-frame string and map lookups into indexed slot access.

use super::bind::{Held as BoundHeld, measured, undeclared};
use super::*;

/// Pre-compiled binding descriptors for a plan's groups and graph outputs.
pub(super) struct CompiledPlan {
    /// Distinct external names referenced by any group input or graph output.
    pub(super) external_names: Vec<String>,
    /// Graph outputs resolved to external slots or producer output slots.
    pub(super) outputs: Vec<(String, CompiledSource)>,
    /// One compiled descriptor per group in `groups`.
    pub(super) groups: Vec<Result<CompiledGroup, EvaluateError>>,
}

/// Pre-resolved source of a Field: an external slot or a producer's output.
pub(super) enum CompiledSource {
    External {
        slot: usize,
        name: String,
    },
    Node {
        producer: NodeId,
        port: String,
        /// The producer group's shared output-name list, output slot index,
        /// and declared schema when `producer` is a group in the plan.
        known: Option<(Arc<Vec<String>>, usize, entelechy_abi::Schema)>,
    },
}

pub(super) struct CompiledGroup {
    generation: Generation,
    inputs: Vec<CompiledInput>,
    params: Vec<CompiledParam>,
    /// Graph parameter indices referenced by `params`, so a group whose slots
    /// have no caller override reuses `default_params` whole.
    graph_slots: Vec<usize>,
    /// Pre-converted parameter vector when every parameter is fixed or uses an
    /// unoverridden valid graph parameter default.
    default_params: Option<Vec<Value>>,
    domain: CompiledDomain,
    reads_clock: bool,
    reads_input: bool,
}

struct CompiledInput {
    port_name: String,
    ty: entelechy_abi::PortType,
    binding: CompiledBinding,
}

enum CompiledBinding {
    Now(CompiledSource),
    Previous {
        key: (NodeId, String),
        initial: CompiledSource,
    },
    Unbound,
    InvalidFeedback,
}

enum CompiledParam {
    Fixed(Value),
    Graph {
        slot: usize,
        param_name: String,
        ty: Type,
        expected: String,
        default: Result<Value, EvaluateError>,
    },
    Fault(EvaluateError),
}

enum CompiledDomain {
    Reduce,
    ReduceFault,
    Literal(Domain),
    FirstInput,
    Expr {
        expr: DomainExpr,
        measured: Result<(), EvaluateError>,
        /// Manifest input slots read by `value(port)` in `expr`.
        value_slots: Vec<Option<usize>>,
    },
}

/// Result of binding a compiled group: handles and cache key ready to check
/// before allocating an [`Invocation`], or cold producers to warm first.
pub(super) enum GroupBind {
    Ready { handles: Vec<Handle>, key: CacheKey },
    Keyed(CacheKey, Vec<NodeId>),
    Unkeyed(Vec<NodeId>),
}

impl CompiledPlan {
    pub(super) fn new(graph: &Graph, groups: &[Group], runs_at: &BTreeMap<NodeId, usize>) -> Self {
        let mut external_names: Vec<String> = Vec::new();
        let mut external_slots: BTreeMap<String, usize> = BTreeMap::new();
        let mut intern_external = |name: &str| -> usize {
            if let Some(&slot) = external_slots.get(name) {
                return slot;
            }
            let slot = external_names.len();
            external_names.push(name.to_owned());
            external_slots.insert(name.to_owned(), slot);
            slot
        };
        for group in groups {
            for binding in group.node.inputs.values() {
                match binding {
                    Binding::Now(Source::External(name))
                    | Binding::Previous {
                        initial: Source::External(name),
                        ..
                    } => {
                        intern_external(name);
                    }
                    _ => {}
                }
            }
        }
        for (_, source) in &graph.outputs {
            if let Source::External(name) = source {
                intern_external(name);
            }
        }

        let graph_param_slots: BTreeMap<&str, usize> = graph
            .params
            .iter()
            .enumerate()
            .map(|(slot, p)| (p.name.as_str(), slot))
            .collect();

        let compile_source = |source: &Source| -> CompiledSource {
            match source {
                Source::External(name) => CompiledSource::External {
                    slot: external_slots[name],
                    name: name.clone(),
                },
                Source::Node {
                    node: producer,
                    port,
                } => {
                    let known = runs_at.get(producer).and_then(|&pg| {
                        let prod_group = &groups[pg];
                        let manifest = prod_group.version.manifest();
                        let slot = manifest.outputs.iter().position(|o| &o.name == port)?;
                        Some((
                            output_names(prod_group),
                            slot,
                            manifest.outputs[slot].ty.schema.clone(),
                        ))
                    });
                    CompiledSource::Node {
                        producer: producer.clone(),
                        port: port.clone(),
                        known,
                    }
                }
            }
        };

        let outputs = graph
            .outputs
            .iter()
            .map(|(name, source)| (name.clone(), compile_source(source)))
            .collect();

        let compiled_groups = groups
            .iter()
            .map(|group| {
                let node = &group.node;
                let manifest = group.version.manifest();
                let structs = group
                    .version
                    .module()
                    .map_or(&[][..], |module| &module.structs);
                undeclared(node, manifest)?;

                let mut inputs = Vec::with_capacity(manifest.inputs.len());
                for input in &manifest.inputs {
                    let binding = match node.inputs.get(&input.name) {
                        None => CompiledBinding::Unbound,
                        Some(Binding::Now(source)) => CompiledBinding::Now(compile_source(source)),
                        Some(Binding::Previous {
                            source:
                                Source::Node {
                                    node: producer,
                                    port,
                                },
                            initial,
                            ..
                        }) => CompiledBinding::Previous {
                            key: (producer.clone(), port.clone()),
                            initial: compile_source(initial),
                        },
                        Some(Binding::Previous { .. }) => CompiledBinding::InvalidFeedback,
                    };
                    // Share the producer's `Schema` allocation when layouts
                    // agree, so runtime verification succeeds by pointer.
                    let ty = match &binding {
                        CompiledBinding::Now(CompiledSource::Node {
                            known: Some((_, _, prod_schema)),
                            ..
                        }) if *prod_schema == input.ty.schema => entelechy_abi::PortType {
                            shape: input.ty.shape,
                            element: input.ty.element,
                            schema: prod_schema.clone(),
                        },
                        _ => input.ty.clone(),
                    };
                    inputs.push(CompiledInput {
                        port_name: input.name.clone(),
                        ty,
                        binding,
                    });
                }

                let mut params = Vec::with_capacity(manifest.params.len());
                let mut graph_slots = Vec::new();
                let mut default_vec = Some(Vec::with_capacity(manifest.params.len()));
                for param in &manifest.params {
                    let compiled = match node.params.get(&param.name) {
                        None => {
                            if let Some(vec) = &mut default_vec {
                                vec.push(param.default.clone());
                            }
                            CompiledParam::Fixed(param.default.clone())
                        }
                        Some(ParamValue::Literal(value)) => {
                            match entelechy_ir::value_as(value, param.ty, structs) {
                                Some(converted) => {
                                    if let Some(vec) = &mut default_vec {
                                        vec.push(converted.clone());
                                    }
                                    CompiledParam::Fixed(converted)
                                }
                                None => {
                                    default_vec = None;
                                    CompiledParam::Fault(EvaluateError::ParamType {
                                        node: node.id.clone(),
                                        name: param.name.clone(),
                                        expected: entelechy_ir::type_name(param.ty, structs),
                                        actual: value.to_string(),
                                    })
                                }
                            }
                        }
                        Some(ParamValue::Graph(name)) => {
                            match graph_param_slots.get(name.as_str()).copied() {
                                None => {
                                    default_vec = None;
                                    let known: Vec<&str> =
                                        graph.params.iter().map(|p| p.name.as_str()).collect();
                                    CompiledParam::Fault(EvaluateError::UnknownPort {
                                        node: node.id.clone(),
                                        port: name.clone(),
                                        near: entelechy_abi::names::nearest(name, known)
                                            .map(str::to_owned),
                                    })
                                }
                                Some(slot) => {
                                    if !graph_slots.contains(&slot) {
                                        graph_slots.push(slot);
                                    }
                                    let expected = entelechy_ir::type_name(param.ty, structs);
                                    let raw_default = &graph.params[slot].default;
                                    let default =
                                        entelechy_ir::value_as(raw_default, param.ty, structs)
                                            .ok_or_else(|| EvaluateError::ParamType {
                                                node: node.id.clone(),
                                                name: param.name.clone(),
                                                expected: expected.clone(),
                                                actual: raw_default.to_string(),
                                            });
                                    match (&default, &mut default_vec) {
                                        (Ok(converted), Some(vec)) => vec.push(converted.clone()),
                                        _ => default_vec = None,
                                    }
                                    CompiledParam::Graph {
                                        slot,
                                        param_name: param.name.clone(),
                                        ty: param.ty,
                                        expected,
                                        default,
                                    }
                                }
                            }
                        }
                    };
                    params.push(compiled);
                }

                let domain = if manifest.mode == entelechy_ir::Mode::Reduce {
                    if node.domain.is_some() {
                        CompiledDomain::ReduceFault
                    } else {
                        CompiledDomain::Reduce
                    }
                } else {
                    match &node.domain {
                        Some(DomainSpec::Literal(domain)) => CompiledDomain::Literal(*domain),
                        Some(DomainSpec::Expr(expr)) => {
                            let value_slots = expr
                                .values_read()
                                .into_iter()
                                .map(|(port, _)| {
                                    manifest.inputs.iter().position(|i| i.name == port)
                                })
                                .collect();
                            CompiledDomain::Expr {
                                expr: expr.clone(),
                                measured: measured(node, manifest, expr),
                                value_slots,
                            }
                        }
                        None => CompiledDomain::FirstInput,
                    }
                };

                Ok(CompiledGroup {
                    generation: group.version.generation(),
                    inputs,
                    params,
                    graph_slots,
                    default_params: default_vec,
                    domain,
                    reads_clock: manifest.requires.contains(Capability::TimeRead),
                    reads_input: manifest.requires.contains(Capability::InputRead),
                })
            })
            .collect();

        Self {
            external_names,
            outputs,
            groups: compiled_groups,
        }
    }
}

impl Evaluator {
    /// Resolves a pre-compiled source against the frame's external slot table
    /// and the ledger.
    pub(super) fn resolve_compiled(
        &self,
        node_id: &NodeId,
        source: &CompiledSource,
        external_handles: &[Option<Handle>],
    ) -> Result<(BoundHeld, InputKey), EvaluateError> {
        match source {
            CompiledSource::External { slot, name } => {
                let handle =
                    external_handles[*slot].ok_or_else(|| EvaluateError::MissingExternal {
                        node: node_id.clone(),
                        name: name.clone(),
                    })?;
                Ok((BoundHeld::Field(handle), InputKey::External(handle)))
            }
            CompiledSource::Node {
                producer,
                port,
                known,
            } => {
                let unknown = || EvaluateError::UnknownOutput {
                    node: producer.clone(),
                    port: port.clone(),
                    near: None,
                };
                if let Some(cold) = self.ledger.cold(producer) {
                    let index = match known {
                        Some((names, slot, _)) if Arc::ptr_eq(&cold.output_names, names) => *slot,
                        _ => cold
                            .output_names
                            .iter()
                            .position(|name| name == port)
                            .ok_or_else(unknown)?,
                    };
                    let (domain, schema) = cold.laid[index].clone();
                    return Ok((
                        BoundHeld::Cold(producer.clone(), domain, schema),
                        InputKey::Node(producer.clone(), index, cold.epoch),
                    ));
                }
                let cached = self.ledger.entry(producer).ok_or_else(unknown)?;
                let index = match known {
                    Some((names, slot, _)) if Arc::ptr_eq(&cached.output_names, names) => *slot,
                    _ => cached
                        .output_names
                        .iter()
                        .position(|name| name == port)
                        .ok_or_else(unknown)?,
                };
                Ok((
                    BoundHeld::Field(cached.outputs[index]),
                    InputKey::Node(producer.clone(), index, cached.epoch),
                ))
            }
        }
    }

    /// Binds group `at` using its pre-compiled layout descriptor.
    pub(super) fn bind_group(
        &self,
        at: usize,
        layout: &Layout,
        scene: &Scene<'_>,
        runtime: &mut Runtime,
    ) -> Result<GroupBind, EvaluateError> {
        let external_handles = &scene.external_handles;
        let overridden_params = &scene.overridden_params;
        let group = &scene.groups[at];
        let node = &group.node;
        let compiled = layout.compiled.groups[at].as_ref().map_err(Clone::clone)?;
        let n_inputs = compiled.inputs.len();
        let needs_expr_domains = matches!(compiled.domain, CompiledDomain::Expr { .. });

        let mut handles = Vec::with_capacity(n_inputs);
        let mut input_keys = Vec::with_capacity(n_inputs);
        let mut cold: Vec<NodeId> = Vec::new();
        let mut first_domain: Option<Domain> = None;
        let mut expr_domains: Vec<(Option<Handle>, Domain)> = if needs_expr_domains {
            Vec::with_capacity(n_inputs)
        } else {
            Vec::new()
        };

        for (slot_idx, input) in compiled.inputs.iter().enumerate() {
            let (held, key) = match &input.binding {
                CompiledBinding::Now(source) => {
                    self.resolve_compiled(&node.id, source, external_handles)?
                }
                CompiledBinding::Previous { key, initial } => match self.ledger.committed(key) {
                    Some(handle) => (BoundHeld::Field(handle), InputKey::State(handle)),
                    None => match self.resolve_compiled(&node.id, initial, external_handles)? {
                        (BoundHeld::Field(handle), _) => {
                            (BoundHeld::Field(handle), InputKey::State(handle))
                        }
                        (BoundHeld::Cold(producer, ..), _) => {
                            return Ok(GroupBind::Unkeyed(vec![producer]));
                        }
                    },
                },
                CompiledBinding::Unbound => {
                    return Err(EvaluateError::UnboundInput {
                        node: node.id.clone(),
                        port: input.port_name.clone(),
                    });
                }
                CompiledBinding::InvalidFeedback => {
                    return Err(EvaluateError::Graph(GraphError::FeedbackNeedsNode {
                        node: node.id.clone(),
                        port: input.port_name.clone(),
                    }));
                }
            };

            let bound_domain = match &held {
                BoundHeld::Field(handle) => {
                    let domain = runtime.host().domain(*handle)?;
                    let schema = runtime.host().schema(*handle)?;
                    if (domain.shape() != input.ty.shape || *schema != input.ty.schema)
                        && let Some(message) = entelechy_kernel::input_type_fault(
                            &input.ty,
                            domain.shape(),
                            Some(schema),
                        )
                    {
                        return Err(EvaluateError::PortKind {
                            node: node.id.clone(),
                            port: input.port_name.clone(),
                            message,
                        });
                    }
                    handles.push(*handle);
                    if needs_expr_domains {
                        expr_domains.push((Some(*handle), domain));
                    }
                    domain
                }
                BoundHeld::Cold(producer, domain, schema) => {
                    if (domain.shape() != input.ty.shape || *schema != input.ty.schema)
                        && let Some(message) = entelechy_kernel::input_type_fault(
                            &input.ty,
                            domain.shape(),
                            Some(schema),
                        )
                    {
                        return Err(EvaluateError::PortKind {
                            node: node.id.clone(),
                            port: input.port_name.clone(),
                            message,
                        });
                    }
                    cold.push(producer.clone());
                    if needs_expr_domains {
                        expr_domains.push((None, *domain));
                    }
                    *domain
                }
            };
            if slot_idx == 0 {
                first_domain = Some(bound_domain);
            }
            input_keys.push(key);
        }

        let params =
            if let Some(defaults) = &compiled.default_params
                && compiled
                    .graph_slots
                    .iter()
                    .all(|&slot| overridden_params[slot].is_none())
            {
                defaults.clone()
            } else {
                let structs = group
                    .version
                    .module()
                    .map_or(&[][..], |module| &module.structs);
                let mut resolved = Vec::with_capacity(compiled.params.len());
                for param in &compiled.params {
                    match param {
                        CompiledParam::Fixed(value) => resolved.push(value.clone()),
                        CompiledParam::Graph {
                            slot,
                            param_name,
                            ty,
                            expected,
                            default,
                        } => match overridden_params[*slot] {
                            None => resolved.push(default.clone()?),
                            Some(raw) => {
                                let converted = entelechy_ir::value_as(raw, *ty, structs)
                                    .ok_or_else(|| EvaluateError::ParamType {
                                        node: node.id.clone(),
                                        name: param_name.clone(),
                                        expected: expected.clone(),
                                        actual: raw.to_string(),
                                    })?;
                                resolved.push(converted);
                            }
                        },
                        CompiledParam::Fault(error) => return Err(error.clone()),
                    }
                }
                resolved
            };

        let context = scene.context;
        let domain = match &compiled.domain {
            CompiledDomain::Reduce => Domain::Rows(1),
            CompiledDomain::ReduceFault => {
                return Err(EvaluateError::ReduceDomain(node.id.clone()));
            }
            CompiledDomain::Literal(domain) => *domain,
            CompiledDomain::FirstInput => {
                first_domain.ok_or_else(|| EvaluateError::NoDomain(node.id.clone()))?
            }
            CompiledDomain::Expr {
                expr,
                measured,
                value_slots,
            } => {
                measured.clone()?;
                let manifest = group.version.manifest();
                let mut read_cold = Vec::new();
                for slot in value_slots.iter().flatten() {
                    if expr_domains[*slot].0.is_none()
                        && let CompiledBinding::Now(CompiledSource::Node { producer, .. }) =
                            &compiled.inputs[*slot].binding
                    {
                        read_cold.push(producer.clone());
                    }
                }
                if !read_cold.is_empty() {
                    return Ok(GroupBind::Unkeyed(read_cold));
                }
                let fields: Vec<Option<Handle>> = expr_domains.iter().map(|(h, _)| *h).collect();
                let read = Self::domain_values(node, manifest, expr, runtime, &fields)?;
                let element = |name: &str, index: u32| read.get(&(name.to_owned(), index)).cloned();
                let port = |name: &str| {
                    manifest
                        .inputs
                        .iter()
                        .position(|input| input.name == name)
                        .map(|index| expr_domains[index].1)
                };
                let param = |name: &str| {
                    manifest
                        .params
                        .iter()
                        .position(|p| p.name == name)
                        .map(|index| params[index].clone())
                        .or_else(|| graph_param(scene.graph, name, scene.params))
                };
                expr.resolve(&port, &param, &window_context(context), &element)
                    .map_err(|e| EvaluateError::Domain {
                        node: node.id.clone(),
                        message: e.message,
                    })?
            }
        };

        let key = CacheKey {
            generation: compiled.generation,
            params,
            inputs: input_keys,
            domain,
            time: compiled.reads_clock.then_some(
                context.time.to_bits()
                    ^ u64::from(context.rate.to_bits()).rotate_left(7)
                    ^ u64::from(context.queued).rotate_left(39),
            ),
            input: compiled.reads_input.then_some(context.input_fingerprint()),
        };
        if !cold.is_empty() {
            return Ok(GroupBind::Keyed(key, cold));
        }
        Ok(GroupBind::Ready { handles, key })
    }
}
