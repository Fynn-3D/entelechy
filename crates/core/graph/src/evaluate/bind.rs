//! A node's invocation: its inputs from the cache, the committed state
//! and the externals, its params and its domain, checked against the
//! ports its Kernel declares.

use super::*;

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
            // The bound resource must have the shape and layout the port
            // declares; the spelling of either side does not matter.
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
            let given = match node.params.get(&param.name) {
                None => None,
                Some(ParamValue::Literal(value)) => Some(value.clone()),
                Some(ParamValue::Graph(name)) => {
                    Some(graph_param(graph, name, overrides).ok_or_else(|| {
                        let known: Vec<&str> =
                            graph.params.iter().map(|p| p.name.as_str()).collect();
                        EvaluateError::UnknownPort {
                            node: node.id.clone(),
                            port: name.clone(),
                            near: orla_abi::names::nearest(name, known).map(str::to_owned),
                        }
                    })?)
                }
            };
            // What was given, in the parameter's type: its layout, an
            // integer standing in for a float, whatever it was spelled as.
            let structs = version.module().map_or(&[][..], |module| &module.structs);
            let value = match given {
                None => param.default.clone(),
                Some(value) => orla_ir::value_as(&value, param.ty, structs).ok_or_else(|| {
                    EvaluateError::ParamType {
                        node: node.id.clone(),
                        name: param.name.clone(),
                        expected: orla_ir::type_name(param.ty, structs),
                        actual: value.to_string(),
                    }
                })?,
            };
            params.push(value);
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

    /// Reads every number a node's `domain` asks a Field for.
    ///
    /// This is the one domain term that is data rather than shape, so it
    /// is the one that brings a Field to the Host — a wait, where the
    /// Field is a device's. The port is held to a row of scalars and to
    /// [`VALUE_MAX_ELEMENTS`] of them so that the wait is for a count and
    /// never for a hidden download of a scene's geometry.
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
            let slot = manifest
                .inputs
                .iter()
                .position(|input| input.name == port)
                .ok_or_else(|| {
                    fail(format!(
                        "`value({port})`: `{port}` is not an input of this Kernel"
                    ))
                })?;
            let handle = *inputs
                .get(slot)
                .ok_or_else(|| fail(format!("`value({port})`: `{port}` is not a bound input")))?;
            let element = manifest.inputs[slot].ty.element;
            if !matches!(element, Type::Int | Type::Float) {
                return Err(fail(format!(
                    "`value({port})` reads a number, but `{port}` is a Field of {element}; give it a row of `Int` or `Float`"
                )));
            }
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
                // The producer precedes this node in topological order, so a
                // missing entry means it never produced this port.
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
