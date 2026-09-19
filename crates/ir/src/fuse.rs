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
    StructDef, Type, Validated, ValueId, validate,
};

/// Why two modules cannot fuse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FuseError(pub String);

impl fmt::Display for FuseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
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
                Op::At(i, _) | Op::Count(i) | Op::Size(i) if *i == input => use_.addressed = true,
                Op::If {
                    then, otherwise, ..
                } => {
                    walk(&then.body, input, use_);
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
        return Err(FuseError(format!("consumer has no input {input}")));
    }
    if !samples_only(consumer, input) {
        return Err(FuseError(format!(
            "consumer reads input `{}` by address or size, not only by sample",
            consumer_manifest.inputs[input as usize].name
        )));
    }
    if producer_manifest.outputs.len() != 1 {
        return Err(FuseError("producer must have exactly one output".into()));
    }
    if consumer.manifest.mode != Mode::Map || producer.manifest.mode != Mode::Map {
        return Err(FuseError(
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
        return Err(FuseError(format!(
            "producer writes {written} but the consumer reads {read}"
        )));
    }
    let consumer_inputs = consumer_manifest.inputs.len() as u32;
    let consumer_params = consumer_manifest.params.len() as u32;
    let consumer_functions = consumer.functions.len() as u32;
    let consumer_structs = consumer.structs.len() as u32;
    let rename = |name: &str| format!("{prefix}/{name}");

    let shift_struct = |ty: Type| match ty {
        Type::Struct(index) => Type::Struct(index + consumer_structs),
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
        outputs: consumer_manifest.outputs.clone(),
    };

    let mut structs = consumer.structs.clone();
    structs.extend(producer.structs.iter().map(|def| {
        StructDef {
            name: rename(&def.name),
            fields: def
                .fields
                .iter()
                .map(|field| crate::Field {
                    name: field.name.clone(),
                    ty: shift_struct(field.ty),
                })
                .collect(),
        }
    }));

    // Consumer ops: inputs after the fused one move down by one.
    let consumer_ops = move |op: &Op| -> Op {
        let index = |i: u32| if i > input { i - 1 } else { i };
        match op {
            Op::Sample(i) => Op::Sample(index(*i)),
            Op::At(i, c) => Op::At(index(*i), *c),
            Op::Count(i) => Op::Count(index(*i)),
            Op::Size(i) => Op::Size(index(*i)),
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
            Op::Count(i) => Op::Count(index(*i)),
            Op::Size(i) => Op::Size(index(*i)),
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
    body.extend(consumer_renumber.block(&consumer.entry.body));
    let results = consumer
        .entry
        .results
        .iter()
        .map(|id| consumer_renumber.map(*id))
        .collect();
    let entry = Function {
        name: consumer.entry.name.clone(),
        args: Vec::new(),
        body,
        results,
    };
    validate(Module {
        manifest,
        structs,
        functions,
        entry,
        merge: None,
        init: None,
    })
    .map_err(|error| FuseError(format!("fused module is invalid: {error}")))
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
                Op::Loop {
                    count,
                    init,
                    until,
                    body,
                } => Op::Loop {
                    count: *count,
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
            Op::Call(_, args) | Op::Intrinsic(_, args) => {
                for arg in args {
                    *arg = self.map(*arg);
                }
            }
            Op::At(_, coord) => *coord = self.map(*coord),
            Op::Key(code) => *code = self.map(*code),
            Op::Const(_)
            | Op::Param(_)
            | Op::Arg(_)
            | Op::Carried
            | Op::Index
            | Op::Coord
            | Op::Domain
            | Op::Time
            | Op::Pointer
            | Op::PointerDown
            | Op::Element
            | Op::Sample(_)
            | Op::Count(_)
            | Op::Size(_) => {}
            Op::If { .. } | Op::Loop { .. } => unreachable!("blocks are rewritten separately"),
        }
    }
}
