//! Kernel fusion: inlining a producer's per-pixel computation into a
//! consumer that only samples it at the current coordinate.
//!
//! `Y = C(P(X))` evaluated pixel by pixel never needs `P`'s image when `C`
//! reads it only through `sample`. The fused module computes `P`'s result
//! first and uses it where `C` sampled, so the intermediate image is never
//! written or read. The result is an ordinary module: it validates, caches,
//! and runs on any backend like a hand-written Kernel.

use std::collections::BTreeMap;
use std::fmt;

use crate::{
    Block, Capabilities, Function, Instruction, Manifest, Mode, Module, Op, Param, Port, PortType,
    Step, StructDef, Type, Validated, ValueId, validate,
};

/// Why two modules cannot fuse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FuseError {
    /// What went wrong.
    pub message: String,
    /// Whether the two simply do not fit: the producer writes a Field
    /// the consumer's port cannot take. That is the scene's mistake and
    /// not fusion's, and the binding check says it far better, so a
    /// planner declines to fuse and lets that happen. Everything else a
    /// fusion refuses is this crate's own fault and is reported.
    pub mismatch: bool,
}

impl FuseError {
    fn new(message: String) -> Self {
        Self {
            message,
            mismatch: false,
        }
    }
}

impl fmt::Display for FuseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for FuseError {}

/// How a module reads one of its inputs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InputUse {
    /// Read at the current coordinate.
    pub sampled: bool,
    /// Read at other coordinates or asked for its size.
    pub addressed: bool,
}

/// Reports how `module` reads `input`.
pub fn input_use(module: &Module, input: u32) -> InputUse {
    fn walk(instructions: &[Instruction], input: u32, use_: &mut InputUse) {
        for instruction in instructions {
            match &instruction.op {
                Op::Sample(i) if *i == input => use_.sampled = true,
                Op::Part {
                    input: i, at: None, ..
                } if *i == input => use_.sampled = true,
                Op::At(i, _)
                | Op::AtLinear(i, _)
                | Op::Count(i)
                | Op::Size(i)
                | Op::Part { input: i, .. }
                    if *i == input =>
                {
                    use_.addressed = true
                }
                Op::If {
                    then, otherwise, ..
                } => {
                    walk(&then.body, input, use_);
                    walk(&otherwise.body, input, use_);
                }
                Op::Match {
                    arms, otherwise, ..
                } => {
                    for (_, block) in arms {
                        walk(&block.body, input, use_);
                    }
                    walk(&otherwise.body, input, use_);
                }
                Op::Loop { until, body, .. } => {
                    if let Some(until) = until {
                        walk(&until.body, input, use_);
                    }
                    walk(&body.body, input, use_);
                }
                _ => {}
            }
        }
    }
    let mut use_ = InputUse::default();
    for function in module
        .functions
        .iter()
        .chain(std::iter::once(&module.entry))
    {
        walk(&function.body, input, &mut use_);
    }
    use_
}

/// Whether `module` reads `input` only by `sample`, so a producer of that
/// input can be fused into it.
pub fn samples_only(module: &Module, input: u32) -> bool {
    let use_ = input_use(module, input);
    use_.sampled && !use_.addressed
}

/// Fuses `producer` into `consumer` at the consumer's `input`. The fused
/// manifest keeps the consumer's other inputs and params, then the
/// producer's, renamed with `prefix` and a slash (`prefix/name`).
pub fn fuse(
    consumer: &Module,
    input: u32,
    producer: &Module,
    prefix: &str,
) -> Result<Validated, FuseError> {
    let consumer_manifest = &consumer.manifest;
    let producer_manifest = &producer.manifest;
    if input as usize >= consumer_manifest.inputs.len() {
        return Err(FuseError::new(format!("consumer has no input {input}")));
    }
    if !samples_only(consumer, input) {
        return Err(FuseError::new(format!(
            "consumer reads input `{}` by address or size, not only by sample",
            consumer_manifest.inputs[input as usize].name
        )));
    }
    if producer_manifest.outputs.len() != 1 {
        return Err(FuseError::new(
            "producer must have exactly one output".into(),
        ));
    }
    if consumer.manifest.mode != Mode::Map || producer.manifest.mode != Mode::Map {
        return Err(FuseError::new(
            "only map Kernels fuse; reductions and scans run on their own".into(),
        ));
    }
    // Ports match by structure: shape and layout. Element types are
    // struct indices local to each module and mean nothing across them.
    let (written, read) = (
        &producer_manifest.outputs[0].ty,
        &consumer_manifest.inputs[input as usize].ty,
    );
    if written.shape != read.shape || written.schema != read.schema {
        return Err(FuseError {
            message: format!("producer writes {written} but the consumer reads {read}"),
            mismatch: true,
        });
    }
    let consumer_inputs = consumer_manifest.inputs.len() as u32;
    let consumer_params = consumer_manifest.params.len() as u32;
    let consumer_functions = consumer.functions.len() as u32;
    let rename = |name: &str| format!("{prefix}/{name}");

    // The producer's structs join the consumer's table, and one the
    // consumer already has is not added twice. The dedup is what makes
    // the splice type: the consumer's sample of the fused input becomes
    // the producer's result, so a whole element the consumer passes
    // around — returned from one arm of an `if` beside one it builds
    // itself — must carry the same struct as the consumer's own. Struct
    // `i` may only contain structs before it, so one pass in order is
    // enough to map every producer struct into the fused table.
    let mut structs = consumer.structs.clone();
    let mut moved: Vec<u32> = Vec::with_capacity(producer.structs.len());
    for def in &producer.structs {
        let mapped = |ty: Type| match ty {
            Type::Struct(index) => Type::Struct(moved[index as usize]),
            other => other,
        };
        let fields: Vec<crate::Field> = def
            .fields
            .iter()
            .map(|field| crate::Field {
                name: field.name.clone(),
                ty: mapped(field.ty),
            })
            .collect();
        let element = def.element.map(mapped);
        match structs
            .iter()
            .position(|other| other.fields == fields && other.element == element)
        {
            Some(index) => moved.push(index as u32),
            None => {
                moved.push(structs.len() as u32);
                structs.push(StructDef {
                    name: rename(&def.name),
                    fields,
                    element,
                });
            }
        }
    }
    let shift_struct = |ty: Type| match ty {
        Type::Struct(index) => Type::Struct(moved[index as usize]),
        other => other,
    };
    let mut inputs: Vec<Port> = consumer_manifest
        .inputs
        .iter()
        .enumerate()
        .filter(|(index, _)| *index as u32 != input)
        .map(|(_, port)| port.clone())
        .collect();
    inputs.extend(producer_manifest.inputs.iter().map(|port| Port {
        name: rename(&port.name),
        ty: PortType {
            element: shift_struct(port.ty.element),
            shape: port.ty.shape,
            schema: port.ty.schema.clone(),
        },
    }));
    let mut params: Vec<Param> = consumer_manifest.params.clone();
    params.extend(producer_manifest.params.iter().map(|param| Param {
        name: rename(&param.name),
        default: param.default,
    }));
    let mut requires: Capabilities = consumer_manifest.requires.clone();
    for capability in producer_manifest.requires.iter() {
        requires.insert(capability);
    }
    let manifest = Manifest {
        id: consumer_manifest.id.clone(),
        mode: consumer_manifest.mode,
        requires,
        inputs,
        params,
        memories: Vec::new(),
        outputs: consumer_manifest.outputs.clone(),
    };

    // Consumer ops: inputs after the fused one move down by one.
    let consumer_ops = move |op: &Op| -> Op {
        let index = |i: u32| if i > input { i - 1 } else { i };
        match op {
            Op::Sample(i) => Op::Sample(index(*i)),
            Op::At(i, c) => Op::At(index(*i), *c),
            Op::AtLinear(i, c) => Op::AtLinear(index(*i), *c),
            Op::Count(i) => Op::Count(index(*i)),
            Op::Size(i) => Op::Size(index(*i)),
            Op::Part { input: i, at, path } => Op::Part {
                input: index(*i),
                at: *at,
                path: path.clone(),
            },
            Op::Put(i, a, v) => Op::Put(index(*i), *a, *v),
            other => other.clone(),
        }
    };
    // Producer ops: inputs, params, functions, and structs move after the
    // consumer's.
    let producer_ops = move |op: &Op| -> Op {
        let index = |i: u32| i + consumer_inputs - 1;
        match op {
            Op::Sample(i) => Op::Sample(index(*i)),
            Op::At(i, c) => Op::At(index(*i), *c),
            Op::AtLinear(i, c) => Op::AtLinear(index(*i), *c),
            Op::Count(i) => Op::Count(index(*i)),
            Op::Size(i) => Op::Size(index(*i)),
            Op::Part { input: i, at, path } => Op::Part {
                input: index(*i),
                at: *at,
                path: path.clone(),
            },
            Op::Param(k) => Op::Param(k + consumer_params),
            Op::Call(f, args) => Op::Call(f + consumer_functions, args.clone()),
            other => other.clone(),
        }
    };
    let identity = |ty: Type| ty;

    let mut functions: Vec<Function> = consumer
        .functions
        .iter()
        .map(|function| {
            let mut renumber = Renumber::new(&consumer_ops, &identity, None);
            renumber.function(function)
        })
        .collect();
    functions.extend(producer.functions.iter().map(|function| {
        let mut renumber = Renumber::new(&producer_ops, &shift_struct, None);
        // `function` already shifted the argument types along with the
        // instruction types; shifting them again would point past the
        // struct table.
        let mut fused = renumber.function(function);
        fused.name = rename(&function.name);
        fused
    }));

    // The entry: the producer's body first, then the consumer's with its
    // sample of the fused input replaced by the producer's result.
    let mut entry_renumber = Renumber::new(&producer_ops, &shift_struct, None);
    let mut body = entry_renumber.block(&producer.entry.body);
    let produced = entry_renumber.map(producer.entry.results[0]);
    let mut consumer_renumber = Renumber::new(&consumer_ops, &identity, Some((input, produced)));
    consumer_renumber.next = entry_renumber.next;
    // The path is walked against what the producer made, which is the
    // producer's own element type in the fused table's numbering.
    consumer_renumber.parts = Some((
        &structs,
        shift_struct(producer_manifest.outputs[0].element()),
    ));
    body.extend(consumer_renumber.block(&consumer.entry.body));
    let results = consumer
        .entry
        .results
        .iter()
        .map(|id| consumer_renumber.map(*id))
        .collect();
    let bindings = producer
        .entry
        .bindings
        .iter()
        .map(|(id, name)| {
            (
                entry_renumber.map(*id),
                format!("{}/{name}", producer.manifest.id),
            )
        })
        .chain(
            consumer
                .entry
                .bindings
                .iter()
                .map(|(id, name)| (consumer_renumber.map(*id), name.clone())),
        )
        .collect();
    let entry = Function {
        name: consumer.entry.name.clone(),
        args: Vec::new(),
        body,
        results,
        bindings,
    };
    validate(Module {
        manifest,
        structs,
        functions,
        entry,
        merge: None,
        init: None,
    })
    .map_err(|error| FuseError::new(format!("fused module is invalid: {error}")))
}

/// Rewrites a function body with fresh pre-order ids, applying `ops` to
/// every operation and `types` to every type, optionally replacing samples
/// of one input by an existing value.
struct Renumber<'a> {
    next: u32,
    ids: BTreeMap<u32, u32>,
    ops: &'a dyn Fn(&Op) -> Op,
    types: &'a dyn Fn(Type) -> Type,
    replace: Option<(u32, ValueId)>,
    /// What the replaced input's element is and the structs to read a
    /// path against: a part of a sample becomes the extracts and
    /// subscripts it stands for, since the value is then in hand.
    parts: Option<(&'a [StructDef], Type)>,
}

impl<'a> Renumber<'a> {
    fn new(
        ops: &'a dyn Fn(&Op) -> Op,
        types: &'a dyn Fn(Type) -> Type,
        replace: Option<(u32, ValueId)>,
    ) -> Self {
        Self {
            next: 0,
            ids: BTreeMap::new(),
            ops,
            types,
            replace,
            parts: None,
        }
    }

    fn map(&self, id: ValueId) -> ValueId {
        ValueId(*self.ids.get(&id.0).unwrap_or(&id.0))
    }

    fn function(&mut self, function: &Function) -> Function {
        let body = self.block(&function.body);
        Function {
            name: function.name.clone(),
            args: function.args.iter().map(|ty| (self.types)(*ty)).collect(),
            body,
            results: function.results.iter().map(|id| self.map(*id)).collect(),
            bindings: function
                .bindings
                .iter()
                .map(|(id, name)| (self.map(*id), name.clone()))
                .collect(),
        }
    }

    fn nested(&mut self, block: &Block) -> Block {
        let body = self.block(&block.body);
        Block {
            body,
            result: self.map(block.result),
        }
    }

    fn block(&mut self, instructions: &[Instruction]) -> Vec<Instruction> {
        let mut out = Vec::with_capacity(instructions.len());
        for instruction in instructions {
            if let (Some((input, replacement)), Op::Sample(i)) = (self.replace, &instruction.op)
                && *i == input
            {
                self.ids.insert(instruction.id.0, replacement.0);
                continue;
            }
            // A part of that sample: the producer's result is a value
            // here, so the way to the part is the extracts and
            // subscripts it stands for, and nothing is read.
            if let (
                Some((input, replacement)),
                Op::Part {
                    input: i,
                    at: None,
                    path,
                },
            ) = (self.replace, &instruction.op)
                && *i == input
                && let Some((structs, element)) = self.parts
            {
                let mut source = replacement;
                let mut ty = element;
                for step in path {
                    let (part, op) = match step {
                        Step::Field(k) => (
                            crate::part_at(structs, ty, *k as usize).map(|(_, ty)| ty),
                            Op::Extract(source, *k),
                        ),
                        Step::Index(index) => (
                            crate::elements_of(structs, ty).map(|(ty, _, _)| ty),
                            Op::Subscript(source, self.map(*index)),
                        ),
                    };
                    let Some(part) = part else { break };
                    let id = ValueId(self.next);
                    self.next += 1;
                    // The types walked are already the fused table's.
                    out.push(Instruction { id, ty: part, op });
                    source = id;
                    ty = part;
                }
                self.ids.insert(instruction.id.0, source.0);
                continue;
            }
            let id = ValueId(self.next);
            self.next += 1;
            self.ids.insert(instruction.id.0, id.0);
            let op = match &instruction.op {
                Op::If {
                    condition,
                    then,
                    otherwise,
                } => Op::If {
                    condition: self.map(*condition),
                    then: self.nested(then),
                    otherwise: self.nested(otherwise),
                },
                Op::Match {
                    scrutinee,
                    arms,
                    otherwise,
                } => Op::Match {
                    scrutinee: self.map(*scrutinee),
                    arms: arms
                        .iter()
                        .map(|(values, block)| (values.clone(), self.nested(block)))
                        .collect(),
                    otherwise: self.nested(otherwise),
                },
                Op::Loop {
                    count,
                    bound,
                    init,
                    until,
                    body,
                } => Op::Loop {
                    count: *count,
                    bound: bound.map(|bound| self.map(bound)),
                    init: self.map(*init),
                    until: until.as_ref().map(|block| self.nested(block)),
                    body: self.nested(body),
                },
                other => {
                    let mut op = (self.ops)(other);
                    self.map_operands(&mut op);
                    op
                }
            };
            out.push(Instruction {
                id,
                ty: (self.types)(instruction.ty),
                op,
            });
        }
        out
    }

    fn map_operands(&self, op: &mut Op) {
        match op {
            Op::Unary(_, a) => *a = self.map(*a),
            Op::Binary(_, a, b) => {
                *a = self.map(*a);
                *b = self.map(*b);
            }
            Op::Construct(parts) => {
                for part in parts {
                    *part = self.map(*part);
                }
            }
            Op::Extract(source, _) => *source = self.map(*source),
            Op::Subscript(array, index) => {
                *array = self.map(*array);
                *index = self.map(*index);
            }
            Op::Call(_, args) | Op::Intrinsic(_, args) => {
                for arg in args {
                    *arg = self.map(*arg);
                }
            }
            Op::At(_, coord) | Op::AtLinear(_, coord) => *coord = self.map(*coord),
            Op::Part { at, path, .. } => {
                if let Some(at) = at {
                    *at = self.map(*at);
                }
                for step in path {
                    if let Step::Index(index) = step {
                        *index = self.map(*index);
                    }
                }
            }
            Op::Put(_, address, value) => {
                *address = self.map(*address);
                *value = self.map(*value);
            }
            Op::Key(code) | Op::Button(code) | Op::Stick(code) => *code = self.map(*code),
            Op::Const(_)
            | Op::Param(_)
            | Op::Arg(_)
            | Op::Carried
            | Op::Index
            | Op::Coord
            | Op::Domain
            | Op::Time
            | Op::Rate
            | Op::Pointer
            | Op::PointerDown
            | Op::Wheel
            | Op::Window
            | Op::Scale
            | Op::Element
            | Op::Sample(_)
            | Op::Count(_)
            | Op::Size(_) => {}
            Op::If { .. } | Op::Match { .. } | Op::Loop { .. } => {
                unreachable!("blocks are rewritten separately")
            }
        }
    }
}
