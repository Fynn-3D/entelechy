//! A node's invocation (inputs, params, domain), checked against its
//! Kernel's declared ports.

use super::*;

/// A node that carries its own output back through an input other than
/// its first, and has no `domain`: it runs over its first input, so the
/// state it carries takes that input's size, not its own. The port that
/// carries it, the output carried, and the first input; `None` otherwise
/// (a reduction's output is one element whatever it reads; a memory
/// keeps its own size; a first input made from what the node carries, as
/// a newer state of it, is that state's size).
pub fn borrowed_domain(
    graph: &Graph,
    node: &Node,
    manifest: &orla_kernel::Manifest,
) -> Option<(String, String, String)> {
    if node.domain.is_some() || manifest.mode == orla_ir::Mode::Reduce {
        return None;
    }
    let own = |port: &str| match node.inputs.get(port) {
        Some(Binding::Previous {
            source: Source::Node { node: from, port },
            ..
        }) if *from == node.id => Some(port.clone()),
        _ => None,
    };
    let first = manifest.inputs.first()?;
    if own(&first.name).is_some() {
        return None;
    }
    if made_from_itself(graph, node, &first.name) {
        return None;
    }
    // A memory keeps its own size, whatever the domain.
    let memory = |port: &str| manifest.memories.iter().any(|m| m == port);
    manifest
        .inputs
        .iter()
        .skip(1)
        .filter(|input| !memory(&input.name))
        .find_map(|input| {
            own(&input.name).map(|output| (input.name.clone(), output, first.name.clone()))
        })
}

/// Whether `node`'s input `port` is made, through the nodes upstream of
/// it, from what `node` carried out of the last evaluation.
fn made_from_itself(graph: &Graph, node: &Node, port: &str) -> bool {
    let mut seen: Vec<&NodeId> = Vec::new();
    let mut open: Vec<&NodeId> = match node.inputs.get(port) {
        Some(Binding::Now(Source::Node { node: from, .. })) => vec![from],
        _ => return false,
    };
    while let Some(id) = open.pop() {
        if seen.contains(&id) {
            continue;
        }
        seen.push(id);
        let Some(upstream) = graph.node(id) else {
            continue;
        };
        for binding in upstream.inputs.values() {
            match binding {
                Binding::Previous {
                    source: Source::Node { node: from, .. },
                    ..
                } if *from == node.id => return true,
                Binding::Now(Source::Node { node: from, .. }) if *from != node.id => {
                    open.push(from)
                }
                _ => {}
            }
        }
    }
    false
}

/// What of a node's binding its Kernel's manifest alone decides, before
/// anything runs: every port it names is declared, every input is bound,
/// every parameter it sets is one the graph has or a value of the
/// parameter's type, a reduction takes no domain, and each `value(port)`
/// its domain reads names an input of numbers. [`Evaluator`] holds every
/// node to it as it binds; a check holds a scene to it unrun.
pub fn binding_fault(
    graph: &Graph,
    node: &Node,
    manifest: &orla_kernel::Manifest,
    structs: &[orla_ir::StructDef],
) -> Result<(), EvaluateError> {
    undeclared(node, manifest)?;
    for input in &manifest.inputs {
        if !node.inputs.contains_key(&input.name) {
            return Err(EvaluateError::UnboundInput {
                node: node.id.clone(),
                port: input.name.clone(),
            });
        }
    }
    for param in &manifest.params {
        param_value(graph, node, param, &BTreeMap::new(), structs)?;
    }
    if manifest.mode == orla_ir::Mode::Reduce && node.domain.is_some() {
        return Err(EvaluateError::ReduceDomain(node.id.clone()));
    }
    if let Some(DomainSpec::Expr(expr)) = &node.domain {
        for (port, _) in expr.values_read() {
            value_slot(node, manifest, &port)?;
        }
    }
    Ok(())
}

/// A port the node names that its Kernel does not declare.
fn undeclared(node: &Node, manifest: &orla_kernel::Manifest) -> Result<(), EvaluateError> {
    for port in node.inputs.keys().chain(node.params.keys()) {
        let declared = manifest.inputs.iter().any(|input| &input.name == port)
            || manifest.params.iter().any(|param| &param.name == port);
        if !declared {
            let known: Vec<&str> = manifest
                .inputs
                .iter()
                .map(|input| input.name.as_str())
                .chain(manifest.params.iter().map(|param| param.name.as_str()))
                .collect();
            return Err(EvaluateError::UnknownPort {
                node: node.id.clone(),
                port: port.clone(),
                near: orla_abi::names::nearest(port, known).map(str::to_owned),
            });
        }
    }
    Ok(())
}

/// The value a node gives parameter `param`, in the parameter's type (an
/// integer may stand for a float): its own, the graph's it names, or the
/// parameter's default.
fn param_value(
    graph: &Graph,
    node: &Node,
    param: &orla_ir::Param,
    overrides: &BTreeMap<String, Value>,
    structs: &[orla_ir::StructDef],
) -> Result<Value, EvaluateError> {
    let given = match node.params.get(&param.name) {
        None => None,
        Some(ParamValue::Literal(value)) => Some(value.clone()),
        Some(ParamValue::Graph(name)) => {
            Some(graph_param(graph, name, overrides).ok_or_else(|| {
                let known: Vec<&str> = graph.params.iter().map(|p| p.name.as_str()).collect();
                EvaluateError::UnknownPort {
                    node: node.id.clone(),
                    port: name.clone(),
                    near: orla_abi::names::nearest(name, known).map(str::to_owned),
                }
            })?)
        }
    };
    match given {
        None => Ok(param.default.clone()),
        Some(value) => {
            orla_ir::value_as(&value, param.ty, structs).ok_or_else(|| EvaluateError::ParamType {
                node: node.id.clone(),
                name: param.name.clone(),
                expected: orla_ir::type_name(param.ty, structs),
                actual: value.to_string(),
            })
        }
    }
}

/// The input slot `value(port)` reads: an input of the Kernel whose
/// elements are numbers.
fn value_slot(
    node: &Node,
    manifest: &orla_kernel::Manifest,
    port: &str,
) -> Result<usize, EvaluateError> {
    let fail = |message: String| EvaluateError::Domain {
        node: node.id.clone(),
        message,
    };
    let slot = manifest
        .inputs
        .iter()
        .position(|input| input.name == port)
        .ok_or_else(|| {
            fail(format!(
                "`value({port})`: `{port}` is not an input of this Kernel"
            ))
        })?;
    let element = manifest.inputs[slot].ty.element;
    if !matches!(element, Type::Int | Type::Float) {
        return Err(fail(format!(
            "`value({port})` reads a number, but `{port}` is a Field of {element}; give it a row of `Int` or `Float`"
        )));
    }
    Ok(slot)
}

impl Evaluator {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn bind(
        &self,
        graph: &Graph,
        node: &Node,
        version: &Arc<KernelVersion>,
        runtime: &mut Runtime,
        externals: &BTreeMap<String, Handle>,
        context: &Context,
        overrides: &BTreeMap<String, Value>,
    ) -> Result<(Invocation, CacheKey), EvaluateError> {
        let manifest = version.manifest();
        let structs = version.module().map_or(&[][..], |module| &module.structs);
        undeclared(node, manifest)?;
        let mut inputs = Vec::with_capacity(manifest.inputs.len());
        let mut input_keys = Vec::with_capacity(manifest.inputs.len());
        for input in &manifest.inputs {
            let binding =
                node.inputs
                    .get(&input.name)
                    .ok_or_else(|| EvaluateError::UnboundInput {
                        node: node.id.clone(),
                        port: input.name.clone(),
                    })?;
            let (handle, key) = match binding {
                Binding::Now(source) => self.resolve(node, source, externals)?,
                Binding::Previous {
                    source:
                        Source::Node {
                            node: producer,
                            port,
                        },
                    initial,
                    ..
                } => match self.ledger.committed(&(producer.clone(), port.clone())) {
                    Some(handle) => (handle, InputKey::State(handle)),
                    None => {
                        let (handle, _) = self.resolve(node, initial, externals)?;
                        (handle, InputKey::State(handle))
                    }
                },
                Binding::Previous { .. } => {
                    return Err(EvaluateError::Graph(GraphError::FeedbackNeedsNode {
                        node: node.id.clone(),
                        port: input.name.clone(),
                    }));
                }
            };
            // The bound resource must have the port's shape and layout,
            // however either is spelled.
            let bound_shape = runtime.host().domain(handle)?.shape();
            let bound_schema = runtime.host().schema(handle)?;
            let message = if input.ty.shape != bound_shape {
                Some(format!(
                    "expects {} but is bound to a {bound_shape}",
                    input.ty.shape
                ))
            } else if input.ty.schema != *bound_schema {
                Some(format!(
                    "expects an element of {} ({}) but is bound to a Field of {} ({bound_schema})",
                    input.ty.schema.describe(),
                    input.ty.schema,
                    bound_schema.describe()
                ))
            } else {
                None
            };
            if let Some(message) = message {
                return Err(EvaluateError::PortKind {
                    node: node.id.clone(),
                    port: input.name.clone(),
                    message,
                });
            }
            inputs.push(handle);
            input_keys.push(key);
        }
        let mut params = Vec::with_capacity(manifest.params.len());
        for param in &manifest.params {
            params.push(param_value(graph, node, param, overrides, structs)?);
        }
        let domain = if manifest.mode == orla_ir::Mode::Reduce {
            if node.domain.is_some() {
                return Err(EvaluateError::ReduceDomain(node.id.clone()));
            }
            Domain::Rows(1)
        } else {
            match (&node.domain, inputs.first()) {
                (Some(DomainSpec::Literal(domain)), _) => *domain,
                (Some(DomainSpec::Expr(expr)), _) => {
                    let read = Self::domain_values(node, manifest, expr, runtime, &inputs)?;
                    let element =
                        |name: &str, index: u32| read.get(&(name.to_owned(), index)).cloned();
                    let port = |name: &str| {
                        manifest
                            .inputs
                            .iter()
                            .position(|input| input.name == name)
                            .and_then(|index| runtime.host().domain(inputs[index]).ok())
                    };
                    let param = |name: &str| {
                        manifest
                            .params
                            .iter()
                            .position(|p| p.name == name)
                            .map(|index| params[index].clone())
                            .or_else(|| graph_param(graph, name, overrides))
                    };
                    expr.resolve(&port, &param, &window_context(context), &element)
                        .map_err(|e| EvaluateError::Domain {
                            node: node.id.clone(),
                            message: e.message,
                        })?
                }
                (None, Some(first)) => runtime.host().domain(*first)?,
                (None, None) => return Err(EvaluateError::NoDomain(node.id.clone())),
            }
        };
        let reads_clock = manifest.requires.contains(Capability::TimeRead);
        let reads_input = manifest.requires.contains(Capability::InputRead);
        let key = CacheKey {
            generation: version.generation(),
            params: params.clone(),
            inputs: input_keys,
            domain,
            time: reads_clock
                .then_some(context.time.to_bits() ^ context.rate.to_bits().rotate_left(7)),
            input: reads_input.then_some(context.input_fingerprint()),
        };
        Ok((
            Invocation {
                inputs,
                params,
                domain,
                context: *context,
            },
            key,
        ))
    }

    /// Reads every number a node's `domain` asks a Field for, bringing
    /// the Field to the Host. The port holds a row of at most
    /// [`VALUE_MAX_ELEMENTS`] scalars.
    pub(super) fn domain_values(
        node: &Node,
        manifest: &orla_kernel::Manifest,
        expr: &DomainExpr,
        runtime: &mut Runtime,
        inputs: &[Handle],
    ) -> Result<BTreeMap<(String, u32), Value>, EvaluateError> {
        let mut read = BTreeMap::new();
        for (port, index) in expr.values_read() {
            let fail = |message: String| EvaluateError::Domain {
                node: node.id.clone(),
                message,
            };
            let slot = value_slot(node, manifest, &port)?;
            let handle = *inputs
                .get(slot)
                .ok_or_else(|| fail(format!("`value({port})`: `{port}` is not a bound input")))?;
            let element = manifest.inputs[slot].ty.element;
            let count = runtime.host().domain(handle)?.count();
            if count > VALUE_MAX_ELEMENTS {
                return Err(fail(format!(
                    "`value({port})` brings `{port}` to the Host to read one number, so it holds counts, not data; this one has {count} elements and the most is {VALUE_MAX_ELEMENTS}"
                )));
            }
            if index as usize >= count {
                return Err(fail(format!(
                    "`value({port}, {index})`: `{port}` has {count} element(s)"
                )));
            }
            runtime
                .materialize(handle)
                .map_err(|e| fail(format!("`value({port})`: {e}")))?;
            let word = runtime.host().element(handle, index)?[0];
            read.insert(
                (port, index),
                match element {
                    Type::Int => Value::Int(word as i32),
                    _ => Value::Float(f32::from_bits(word)),
                },
            );
        }
        Ok(read)
    }

    /// Resolves a `Now` source to a handle and its cache key.
    pub(super) fn resolve(
        &self,
        node: &Node,
        source: &Source,
        externals: &BTreeMap<String, Handle>,
    ) -> Result<(Handle, InputKey), EvaluateError> {
        match source {
            Source::External(name) => {
                let handle =
                    *externals
                        .get(name)
                        .ok_or_else(|| EvaluateError::MissingExternal {
                            node: node.id.clone(),
                            name: name.clone(),
                        })?;
                Ok((handle, InputKey::External(handle)))
            }
            Source::Node {
                node: producer,
                port,
            } => {
                // The producer runs first, so a missing entry means it
                // never produced this port.
                let cached =
                    self.ledger
                        .entry(producer)
                        .ok_or_else(|| EvaluateError::UnknownOutput {
                            node: producer.clone(),
                            port: port.clone(),
                            near: None,
                        })?;
                let index = cached
                    .output_names
                    .iter()
                    .position(|name| name == port)
                    .ok_or_else(|| EvaluateError::UnknownOutput {
                        node: producer.clone(),
                        port: port.clone(),
                        near: None,
                    })?;
                Ok((
                    cached.outputs[index],
                    InputKey::Node(producer.clone(), index, cached.epoch),
                ))
            }
        }
    }
}
