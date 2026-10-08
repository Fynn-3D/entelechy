//! Kernel fusion: inlining a producer's per-element computation into a
//! consumer that samples it at the current coordinate, or reads it at an
//! offset of its own element.
//!
//! For `Y = C(P(X))`, the fused module computes `P`'s result first and uses
//! it where `C` sampled, or computes it as a function of the row where `C`
//! read, so the intermediate Field is never stored. A Kernel that only asks
//! a Field its count or size reads the extent from params instead. Each
//! result is an ordinary module that validates and runs on any backend.

use std::collections::BTreeMap;
use std::fmt;

use crate::{
    Block, Capabilities, Function, Instruction, Manifest, Mode, Module, Op, Param, Port, PortType,
    Step, StructDef, Type, Validated, Value, ValueId, validate,
};

/// Why two modules cannot fuse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FuseError {
    /// What went wrong.
    pub message: String,
    /// Whether the planner runs the two separately without reporting: the
    /// consumer's port cannot take the producer's Field (the binding check
    /// reports it), or a Kernel's work is counted. Any other refusal is a
    /// fault of this crate and is reported.
    pub declines: bool,
}

impl FuseError {
    fn new(message: String) -> Self {
        Self {
            message,
            declines: false,
        }
    }

    fn declined(message: String) -> Self {
        Self {
            message,
            declines: true,
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
    /// Read at other coordinates.
    pub addressed: bool,
    /// Asked for its count or size.
    pub counted: bool,
}

fn use_in(instructions: &[Instruction], input: u32, use_: &mut InputUse) {
    for instruction in instructions {
        match &instruction.op {
            Op::Sample(i) if *i == input => use_.sampled = true,
            Op::Part {
                input: i, at: None, ..
            } if *i == input => use_.sampled = true,
            Op::Count(i) | Op::Size(i) if *i == input => use_.counted = true,
            Op::At(i, _) | Op::AtLinear(i, _) | Op::Part { input: i, .. } if *i == input => {
                use_.addressed = true
            }
            op => {
                for block in op.blocks() {
                    use_in(&block.body, input, use_);
                }
            }
        }
    }
}

/// Reports how `module` reads `input`.
pub fn input_use(module: &Module, input: u32) -> InputUse {
    let mut use_ = InputUse::default();
    for function in module
        .functions
        .iter()
        .chain(std::iter::once(&module.entry))
    {
        use_in(&function.body, input, &mut use_);
    }
    use_
}

/// Whether `module` reads `input` only by `sample`, so its producer can
/// fuse into it.
pub fn samples_only(module: &Module, input: u32) -> bool {
    let use_ = input_use(module, input);
    use_.sampled && !use_.addressed && !use_.counted
}

/// Whether `module` asks `input` only its count or size, so the input can
/// leave it and the extent come in by params ([`counted`]).
pub fn counts_only(module: &Module, input: u32) -> bool {
    let use_ = input_use(module, input);
    use_.counted && !use_.sampled && !use_.addressed
}

/// Whether `module` reads `input`, a row, only at an offset of its own
/// element: `element()` plus or less a value every element shares, read
/// in the entry outside any loop and at that one address, so an element
/// reads at most one of the input's and two elements two. Besides, it may
/// ask the input's count or size. The input's producer can then fuse into
/// it as a function of the row ([`fuse_indexed`]).
pub fn offset_read(module: &Module, input: u32) -> bool {
    let rows = module
        .manifest
        .inputs
        .get(input as usize)
        .is_some_and(|port| port.shape() == crate::Shape::Rows);
    if !rows {
        return false;
    }
    // A function may ask the count or size, nothing more.
    let mut in_functions = InputUse::default();
    for function in &module.functions {
        use_in(&function.body, input, &mut in_functions);
    }
    if in_functions.sampled || in_functions.addressed {
        return false;
    }
    let mut reads = Reads {
        input,
        defs: BTreeMap::new(),
        addresses: Vec::new(),
        refused: false,
    };
    reads.walk(&module.entry.body, false);
    let Some(&address) = reads.addresses.first() else {
        return false;
    };
    if reads.refused || reads.addresses.iter().any(|other| *other != address) {
        return false;
    }
    reads.offset(address, &mut BTreeMap::new())
}

/// The entry's reads of one input, and every value's definition.
struct Reads<'a> {
    input: u32,
    defs: BTreeMap<u32, &'a Op>,
    addresses: Vec<ValueId>,
    /// Read otherwise than at an address, or at one inside a loop.
    refused: bool,
}

impl<'a> Reads<'a> {
    fn walk(&mut self, instructions: &'a [Instruction], looped: bool) {
        for instruction in instructions {
            let op = &instruction.op;
            self.defs.insert(instruction.id.0, op);
            match op {
                Op::Sample(i)
                | Op::AtLinear(i, _)
                | Op::Part {
                    input: i, at: None, ..
                } if *i == self.input => self.refused = true,
                Op::At(i, address)
                | Op::Part {
                    input: i,
                    at: Some(address),
                    ..
                } if *i == self.input => {
                    self.refused |= looped;
                    self.addresses.push(*address);
                }
                Op::Loop { until, body, .. } => {
                    if let Some(until) = until {
                        self.walk(&until.body, true);
                    }
                    self.walk(&body.body, true);
                }
                op => {
                    for block in op.blocks() {
                        self.walk(&block.body, looped);
                    }
                }
            }
        }
    }

    /// Whether `id` is `element()` plus or less a shared value.
    fn offset(&self, id: ValueId, shared: &mut BTreeMap<u32, bool>) -> bool {
        use crate::{BinaryOp, Intrinsic};
        match self.defs.get(&id.0) {
            Some(Op::Element) => true,
            Some(Op::Binary(BinaryOp::Add, a, b)) => self.sum(*a, *b, shared),
            Some(Op::Binary(BinaryOp::Sub, a, b)) => {
                self.offset(*a, shared) && self.shared(*b, shared)
            }
            Some(Op::Intrinsic(Intrinsic::AddWrap, args)) if args.len() == 2 => {
                self.sum(args[0], args[1], shared)
            }
            Some(Op::Intrinsic(Intrinsic::SubWrap, args)) if args.len() == 2 => {
                self.offset(args[0], shared) && self.shared(args[1], shared)
            }
            _ => false,
        }
    }

    /// Whether `a + b` is an offset: one of them is, the other shared.
    fn sum(&self, a: ValueId, b: ValueId, shared: &mut BTreeMap<u32, bool>) -> bool {
        (self.offset(a, shared) && self.shared(b, shared))
            || (self.shared(a, shared) && self.offset(b, shared))
    }

    /// Whether `id` is the same for every element, remembered by value.
    fn shared(&self, id: ValueId, known: &mut BTreeMap<u32, bool>) -> bool {
        if let Some(&shared) = known.get(&id.0) {
            return shared;
        }
        let shared = match self.defs.get(&id.0) {
            Some(
                Op::Const(_)
                | Op::Param(_)
                | Op::Count(_)
                | Op::Size(_)
                | Op::Domain
                | Op::Time
                | Op::Rate
                | Op::Window
                | Op::Scale
                | Op::Pads,
            ) => true,
            Some(Op::Unary(_, a) | Op::Extract(a, _)) => self.shared(*a, known),
            Some(Op::Binary(_, a, b)) => self.shared(*a, known) && self.shared(*b, known),
            Some(Op::Intrinsic(_, args) | Op::Construct(args)) => {
                args.iter().all(|arg| self.shared(*arg, known))
            }
            _ => false,
        };
        known.insert(id.0, shared);
        shared
    }
}

/// The params a Kernel reads a producer's extent from once that producer
/// is fused into it or is not stored for it: the element count, as
/// `count` gives it, and the extent, as `size` gives it. They are named
/// under the producer's node, as no written name can be; the planner sets
/// them.
pub fn extent_params(prefix: &str) -> [String; 2] {
    [format!("{prefix}/#count"), format!("{prefix}/#size")]
}

/// Whether `name` is one of [`extent_params`]'s.
fn is_extent_param(name: &str) -> bool {
    name.rsplit_once('/')
        .is_some_and(|(_, last)| last == "#count" || last == "#size")
}

/// The index of the param `name` of type `ty`, appended unless there.
fn extent_param(params: &mut Vec<Param>, name: &str, ty: Type) -> u32 {
    if let Some(index) = params.iter().position(|param| param.name == name) {
        return index as u32;
    }
    let default = match ty {
        Type::Vec2 => Value::Vec2([0.0; 2]),
        Type::Vec3 => Value::Vec3([0.0; 3]),
        _ => Value::Int(0),
    };
    params.push(Param {
        name: name.to_owned(),
        ty,
        default,
    });
    (params.len() - 1) as u32
}

/// The count and size params of a producer of `shape` under `prefix`.
fn extent_pair(params: &mut Vec<Param>, prefix: &str, shape: crate::Shape) -> (u32, u32) {
    use crate::ShapeIndex;
    let [count, size] = extent_params(prefix);
    (
        extent_param(params, &count, Type::Int),
        extent_param(params, &size, shape.index_type()),
    )
}

/// Drops the inputs `module` asks only the count or size of
/// ([`counts_only`]), each to read its producer's extent from the params
/// [`extent_params`] names under the prefix given with it, so nothing has
/// to be stored for it.
pub fn counted(module: &Module, inputs: &[(u32, String)]) -> Result<Validated, FuseError> {
    let manifest = &module.manifest;
    if module.merge.is_some() || module.init.is_some() || !manifest.memories.is_empty() {
        return Err(FuseError::new(
            "only a map without memories reads an extent from params".into(),
        ));
    }
    for (index, (input, _)) in inputs.iter().enumerate() {
        if *input as usize >= manifest.inputs.len() {
            return Err(FuseError::new(format!("the Kernel has no input {input}")));
        }
        if inputs[..index].iter().any(|(earlier, _)| earlier == input) {
            return Err(FuseError::new(format!("input {input} is counted twice")));
        }
        if !counts_only(module, *input) {
            return Err(FuseError::new(format!(
                "input `{}` is read, not only counted",
                manifest.inputs[*input as usize].name
            )));
        }
    }
    let mut params = manifest.params.clone();
    let extents: BTreeMap<u32, (u32, u32)> = inputs
        .iter()
        .map(|(input, prefix)| {
            let shape = manifest.inputs[*input as usize].shape();
            (*input, extent_pair(&mut params, prefix, shape))
        })
        .collect();
    let removed: Vec<u32> = inputs.iter().map(|(input, _)| *input).collect();
    let ops = |op: &Op| -> Op {
        match op {
            Op::Count(i) if extents.contains_key(i) => Op::Param(extents[i].0),
            Op::Size(i) if extents.contains_key(i) => Op::Param(extents[i].1),
            other => shift_inputs(other, |i| {
                i - removed.iter().filter(|gone| **gone < i).count() as u32
            }),
        }
    };
    let identity = |ty: Type| ty;
    let renumbered =
        |function: &Function| Renumber::new(&ops, &identity, BTreeMap::new()).function(function);
    let manifest = Manifest {
        inputs: manifest
            .inputs
            .iter()
            .enumerate()
            .filter(|(index, _)| !removed.contains(&(*index as u32)))
            .map(|(_, port)| port.clone())
            .collect(),
        params,
        ..manifest.clone()
    };
    validate(Module {
        manifest,
        structs: module.structs.clone(),
        functions: module.functions.iter().map(renumbered).collect(),
        entry: renumbered(&module.entry),
        merge: None,
        init: None,
    })
    .map_err(|error| FuseError::new(format!("counted module is invalid: {error}")))
}

/// `op` with each input it names renumbered by `index`.
fn shift_inputs(op: &Op, index: impl Fn(u32) -> u32) -> Op {
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
}

/// A name the producer brings into a fused Kernel: `prefix/name`, or the
/// name as it is when an earlier fusion already qualified it with its own
/// node's (no written name holds a `/`), so names stay short however long
/// a chain fuses and stay unique, each qualified once by its node.
pub fn fused_name(prefix: &str, name: &str) -> String {
    if name.contains('/') {
        name.to_owned()
    } else {
        format!("{prefix}/{name}")
    }
}

/// Fuses `producer` into `consumer` at the consumer's `input`, which reads
/// the producer's first output: [`fuse_outputs`] with one link.
pub fn fuse(
    consumer: &Module,
    input: u32,
    producer: &Module,
    prefix: &str,
) -> Result<Validated, FuseError> {
    fuse_outputs(consumer, &[(input, 0)], producer, prefix)
}

/// Fuses `producer` into `consumer`. Each link `(input, output)` binds a
/// consumer input to a producer output; every sample of that input becomes
/// the producer's value for it, and its count and size come in by the
/// params [`extent_params`] names. Unread producer outputs are computed and
/// dropped. The fused manifest keeps the consumer's other inputs and params,
/// then the producer's, renamed by [`fused_name`], then the extent's.
pub fn fuse_outputs(
    consumer: &Module,
    links: &[(u32, u32)],
    producer: &Module,
    prefix: &str,
) -> Result<Validated, FuseError> {
    fuse_links(consumer, links, producer, prefix, Reading::Pointwise)
}

/// Fuses `producer` into `consumer` as a function of the row: every read
/// of a linked input, at an offset of the consumer's element
/// ([`offset_read`]), becomes the producer's computation at that address,
/// clamped into the producer's extent as `at` clamps it; the input's count
/// and size come in by the params [`extent_params`] names, and so does the
/// producer's own extent. The manifest is [`fuse_outputs`]'s.
pub fn fuse_indexed(
    consumer: &Module,
    links: &[(u32, u32)],
    producer: &Module,
    prefix: &str,
) -> Result<Validated, FuseError> {
    fuse_links(consumer, links, producer, prefix, Reading::Indexed)
}

/// How a fused consumer reads its producer.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reading {
    /// At its own element: the producer's body runs first.
    Pointwise,
    /// At an offset of its element: the producer's body is a function of
    /// the row, called where the consumer reads.
    Indexed,
}

fn fuse_links(
    consumer: &Module,
    links: &[(u32, u32)],
    producer: &Module,
    prefix: &str,
    reading: Reading,
) -> Result<Validated, FuseError> {
    let consumer_manifest = &consumer.manifest;
    let producer_manifest = &producer.manifest;
    if links.is_empty() {
        return Err(FuseError::new("fusion needs a link".into()));
    }
    for (index, (input, output)) in links.iter().enumerate() {
        if *input as usize >= consumer_manifest.inputs.len() {
            return Err(FuseError::new(format!("consumer has no input {input}")));
        }
        if *output as usize >= producer_manifest.outputs.len() {
            return Err(FuseError::new(format!("producer has no output {output}")));
        }
        if links[..index].iter().any(|(earlier, _)| earlier == input) {
            return Err(FuseError::new(format!("input {input} is linked twice")));
        }
        let name = &consumer_manifest.inputs[*input as usize].name;
        match reading {
            Reading::Pointwise if input_use(consumer, *input).addressed => {
                return Err(FuseError::new(format!(
                    "consumer reads input `{name}` by address, not only by sample"
                )));
            }
            Reading::Indexed
                if !offset_read(consumer, *input) && !counts_only(consumer, *input) =>
            {
                return Err(FuseError::new(format!(
                    "consumer reads input `{name}` otherwise than at one offset of its element"
                )));
            }
            _ => {}
        }
    }
    // A Kernel whose work is counted keeps its own budget and run, so it
    // runs alone.
    if producer.counts_work() || consumer.counts_work() {
        return Err(FuseError::declined(
            "a Kernel whose work is counted runs on its own, held to its own budget".into(),
        ));
    }
    if consumer.manifest.mode != Mode::Map || producer.manifest.mode != Mode::Map {
        return Err(FuseError::new(
            "only map Kernels fuse; reductions and scans run on their own".into(),
        ));
    }
    // A narrow output rounds its values, which the consumer must read.
    if let Some((_, output)) = links.iter().find(|(_, output)| {
        !producer_manifest.outputs[*output as usize]
            .ty
            .schema
            .is_wide()
    }) {
        return Err(FuseError::new(format!(
            "output `{}` keeps its scalars narrow; what passes through it is rounded, and a fused read would not be",
            producer_manifest.outputs[*output as usize].name
        )));
    }
    // Ports match by shape and layout; struct indices are module-local.
    for (input, output) in links {
        let (written, read) = (
            &producer_manifest.outputs[*output as usize].ty,
            &consumer_manifest.inputs[*input as usize].ty,
        );
        if written.shape != read.shape || written.schema != read.schema {
            return Err(FuseError {
                message: format!("producer writes {written} but the consumer reads {read}"),
                declines: true,
            });
        }
    }
    // Linked inputs leave the consumer; later inputs move down to close up.
    let removed: Vec<u32> = links.iter().map(|(input, _)| *input).collect();
    let kept_inputs = (consumer_manifest.inputs.len() - removed.len()) as u32;
    let consumer_functions = consumer.functions.len() as u32;
    let rename = |name: &str| fused_name(prefix, name);

    // The producer's structs join the consumer's table once each, so a
    // spliced element carries the consumer's own struct and types match.
    // Structs refer only to earlier ones, so one ordered pass maps them.
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
        .filter(|(index, _)| !removed.contains(&(*index as u32)))
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
    // A producer's param keeps its place after the consumer's, but for an
    // extent both read already, which is one value.
    let mut params: Vec<Param> = consumer_manifest.params.clone();
    let mut producer_params = Vec::with_capacity(producer_manifest.params.len());
    for param in &producer_manifest.params {
        let name = rename(&param.name);
        let ty = shift_struct(param.ty);
        let index = if is_extent_param(&name) {
            extent_param(&mut params, &name, ty)
        } else {
            params.push(Param {
                name,
                ty,
                default: param.default.clone(),
            });
            (params.len() - 1) as u32
        };
        producer_params.push(index);
    }
    // The producer's extent, for a count or size of a linked input and for
    // the producer's own element in an indexed fusion.
    let counted_links = links
        .iter()
        .any(|(input, _)| input_use(consumer, *input).counted);
    let extent = (reading == Reading::Indexed || counted_links).then(|| {
        let shape = producer_manifest.outputs[links[0].1 as usize].shape();
        extent_pair(&mut params, prefix, shape)
    });
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
        // One element runs both, so it may run what both may.
        budget: consumer_manifest
            .budget
            .saturating_add(producer_manifest.budget)
            .min(crate::MAX_BUDGET),
    };

    // Consumer ops: inputs after a fused one move down, and a fused one's
    // count and size are the extent's params.
    let consumer_ops = |op: &Op| -> Op {
        match (op, extent) {
            (Op::Count(i), Some((count, _))) if removed.contains(i) => Op::Param(count),
            (Op::Size(i), Some((_, size))) if removed.contains(i) => Op::Param(size),
            _ => shift_inputs(op, |i| {
                i - removed.iter().filter(|gone| **gone < i).count() as u32
            }),
        }
    };
    // Producer ops: inputs, params, functions and structs follow the
    // consumer's.
    let producer_ops = |op: &Op| -> Op {
        match op {
            Op::Param(k) => Op::Param(producer_params[*k as usize]),
            Op::Call(f, args) => Op::Call(f + consumer_functions, args.clone()),
            other => shift_inputs(other, |i| i + kept_inputs),
        }
    };
    let identity = |ty: Type| ty;

    let mut functions: Vec<Function> = consumer
        .functions
        .iter()
        .map(|function| {
            let mut renumber = Renumber::new(&consumer_ops, &identity, BTreeMap::new());
            renumber.function(function)
        })
        .collect();
    functions.extend(producer.functions.iter().map(|function| {
        let mut renumber = Renumber::new(&producer_ops, &shift_struct, BTreeMap::new());
        // `function` has already shifted the argument types.
        let mut fused = renumber.function(function);
        fused.name = rename(&function.name);
        fused
    }));
    let producer_bindings = |renumber: &Renumber| -> Vec<(ValueId, String)> {
        producer
            .entry
            .bindings
            .iter()
            .map(|(id, name)| {
                (
                    renumber.map(*id),
                    format!("{}/{name}", producer.manifest.id),
                )
            })
            .collect()
    };

    let (body, results, bindings) = match reading {
        Reading::Pointwise => {
            // The producer's body first, then the consumer's with its
            // sample of the fused input replaced by the producer's result.
            let mut entry_renumber = Renumber::new(&producer_ops, &shift_struct, BTreeMap::new());
            let mut body = entry_renumber.block(&producer.entry.body);
            // Each linked input reads the producer's value for its output;
            // a path into it is walked against that output's element in the
            // fused table.
            let replace = links
                .iter()
                .map(|(input, output)| {
                    let produced = entry_renumber.map(producer.entry.results[*output as usize]);
                    let element =
                        shift_struct(producer_manifest.outputs[*output as usize].element());
                    (*input, (produced, element))
                })
                .collect();
            let mut consumer_renumber = Renumber::new(&consumer_ops, &identity, replace);
            consumer_renumber.next = entry_renumber.next;
            consumer_renumber.structs = Some(&structs);
            body.extend(consumer_renumber.block(&consumer.entry.body));
            let results = consumer_renumber.results(&consumer.entry.results);
            let mut bindings = producer_bindings(&entry_renumber);
            bindings.extend(consumer_renumber.bindings(&consumer.entry.bindings));
            (body, results, bindings)
        }
        Reading::Indexed => {
            let (count, size) = extent.expect("an indexed fusion reads the extent");
            // The producer's own extent is the param's, its element the row
            // it is called at.
            let row_ops = |op: &Op| -> Op {
                match op {
                    Op::Domain => Op::Param(size),
                    other => producer_ops(other),
                }
            };
            // One function per output read by address: the producer's
            // entry at a row, yielding that output.
            let mut rows: BTreeMap<u32, u32> = BTreeMap::new();
            let mut called = BTreeMap::new();
            for (input, output) in links {
                if !input_use(consumer, *input).addressed {
                    continue;
                }
                let function = match rows.get(output) {
                    Some(function) => *function,
                    None => {
                        let mut renumber = Renumber::new(&row_ops, &shift_struct, BTreeMap::new());
                        let row = ValueId(0);
                        renumber.next = 1;
                        renumber.own = Some(row);
                        let mut body = vec![Instruction {
                            id: row,
                            ty: Type::Int,
                            op: Op::Arg(0),
                        }];
                        body.extend(renumber.block(&producer.entry.body));
                        let output_port = &producer_manifest.outputs[*output as usize];
                        functions.push(Function {
                            name: format!("{prefix}/{} at a row", output_port.name),
                            args: vec![Type::Int],
                            body,
                            results: renumber.results(&[producer.entry.results[*output as usize]]),
                            bindings: producer_bindings(&renumber),
                        });
                        let function = (functions.len() - 1) as u32;
                        rows.insert(*output, function);
                        function
                    }
                };
                let element = shift_struct(producer_manifest.outputs[*output as usize].element());
                called.insert(*input, (function, element));
            }
            let mut consumer_renumber = Renumber::new(&consumer_ops, &identity, BTreeMap::new());
            consumer_renumber.called = called;
            consumer_renumber.count = count;
            consumer_renumber.structs = Some(&structs);
            let body = consumer_renumber.block(&consumer.entry.body);
            let results = consumer_renumber.results(&consumer.entry.results);
            let bindings = consumer_renumber.bindings(&consumer.entry.bindings);
            (body, results, bindings)
        }
    };
    let entry = Function {
        name: consumer.entry.name.clone(),
        args: Vec::new(),
        body,
        results,
        bindings,
    };
    let fused = validate(Module {
        manifest,
        structs,
        functions,
        entry,
        merge: None,
        init: None,
    })
    .map_err(|error| FuseError::new(format!("fused module is invalid: {error}")))?;
    // A fusion that would be counted where neither part is stays unfused.
    if fused.counts_work() {
        return Err(FuseError::declined(
            "together the two would run long enough to be counted, which neither is on its own"
                .into(),
        ));
    }
    Ok(fused)
}

/// Rewrites a function body with fresh pre-order ids, applying `ops` to
/// every operation and `types` to every type, and replacing samples of
/// linked inputs by values in hand and their reads by calls.
struct Renumber<'a> {
    next: u32,
    ids: BTreeMap<u32, u32>,
    ops: &'a dyn Fn(&Op) -> Op,
    types: &'a dyn Fn(Type) -> Type,
    /// Inputs whose samples become values in hand, with each value's
    /// element type for walking paths.
    replace: BTreeMap<u32, (ValueId, Type)>,
    /// Inputs whose reads become calls: the function computing the
    /// element at a row, and the element's type.
    called: BTreeMap<u32, (u32, Type)>,
    /// The param holding those inputs' count.
    count: u32,
    /// The calls made, by input and address, in each enclosing block.
    calls: Vec<BTreeMap<(u32, u32), ValueId>>,
    /// In a function of the row, the row: the element, and where its own
    /// samples read.
    own: Option<ValueId>,
    /// The structs such a path is read against.
    structs: Option<&'a [StructDef]>,
}

impl<'a> Renumber<'a> {
    fn new(
        ops: &'a dyn Fn(&Op) -> Op,
        types: &'a dyn Fn(Type) -> Type,
        replace: BTreeMap<u32, (ValueId, Type)>,
    ) -> Self {
        Self {
            next: 0,
            ids: BTreeMap::new(),
            ops,
            types,
            replace,
            called: BTreeMap::new(),
            count: 0,
            calls: vec![BTreeMap::new()],
            own: None,
            structs: None,
        }
    }

    fn map(&self, id: ValueId) -> ValueId {
        ValueId(*self.ids.get(&id.0).unwrap_or(&id.0))
    }

    fn results(&self, results: &[ValueId]) -> Vec<ValueId> {
        results.iter().map(|id| self.map(*id)).collect()
    }

    fn bindings(&self, bindings: &[(ValueId, String)]) -> Vec<(ValueId, String)> {
        bindings
            .iter()
            .map(|(id, name)| (self.map(*id), name.clone()))
            .collect()
    }

    fn function(&mut self, function: &Function) -> Function {
        let body = self.block(&function.body);
        Function {
            name: function.name.clone(),
            args: function.args.iter().map(|ty| (self.types)(*ty)).collect(),
            body,
            results: self.results(&function.results),
            bindings: self.bindings(&function.bindings),
        }
    }

    fn nested(&mut self, block: &Block) -> Block {
        self.calls.push(BTreeMap::new());
        let body = self.block(&block.body);
        self.calls.pop();
        Block {
            body,
            result: self.map(block.result),
        }
    }

    /// Appends an instruction with the next id.
    fn push(&mut self, out: &mut Vec<Instruction>, ty: Type, op: Op) -> ValueId {
        let id = ValueId(self.next);
        self.next += 1;
        out.push(Instruction { id, ty, op });
        id
    }

    /// The element of `input` at `address`, computed once per block it is
    /// read in: the address clamped into the extent, as `at` clamps it,
    /// and the producer called there.
    fn call(
        &mut self,
        out: &mut Vec<Instruction>,
        input: u32,
        address: ValueId,
        (function, element): (u32, Type),
    ) -> ValueId {
        let key = (input, address.0);
        if let Some(value) = self.calls.iter().rev().find_map(|scope| scope.get(&key)) {
            return *value;
        }
        use crate::{BinaryOp, Intrinsic};
        let count = self.push(out, Type::Int, Op::Param(self.count));
        let zero = self.push(out, Type::Int, Op::Const(Value::Int(0)));
        let one = self.push(out, Type::Int, Op::Const(Value::Int(1)));
        let last = self.push(out, Type::Int, Op::Binary(BinaryOp::Sub, count, one));
        let row = self.push(
            out,
            Type::Int,
            Op::Intrinsic(Intrinsic::Clamp, vec![address, zero, last]),
        );
        let value = self.push(out, element, Op::Call(function, vec![row]));
        self.calls
            .last_mut()
            .expect("a block's calls")
            .insert(key, value);
        value
    }

    /// A part of an element in hand: extracts and subscripts along `path`.
    fn part(
        &mut self,
        out: &mut Vec<Instruction>,
        mut source: ValueId,
        mut ty: Type,
        path: &[Step],
    ) -> ValueId {
        let structs = self
            .structs
            .expect("a path is walked against the fused structs");
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
            // The types walked are already the fused table's.
            source = self.push(out, part, op);
            ty = part;
        }
        source
    }

    fn block(&mut self, instructions: &[Instruction]) -> Vec<Instruction> {
        let mut out = Vec::with_capacity(instructions.len());
        for instruction in instructions {
            if let Op::Sample(i) = &instruction.op
                && let Some((replacement, _)) = self.replace.get(i)
            {
                self.ids.insert(instruction.id.0, replacement.0);
                continue;
            }
            // A part of that sample becomes extracts and subscripts of the
            // producer's value.
            if let Op::Part {
                input: i,
                at: None,
                path,
            } = &instruction.op
                && let Some(&(replacement, element)) = self.replace.get(i)
                && self.structs.is_some()
            {
                let source = self.part(&mut out, replacement, element, path);
                self.ids.insert(instruction.id.0, source.0);
                continue;
            }
            // A read of a called input is the producer's element there.
            let read = match &instruction.op {
                Op::At(i, address) => Some((*i, *address, None)),
                Op::Part {
                    input: i,
                    at: Some(address),
                    path,
                } => Some((*i, *address, Some(path))),
                _ => None,
            };
            if let Some((i, address, path)) = read
                && let Some(&called) = self.called.get(&i)
            {
                let address = self.map(address);
                let value = self.call(&mut out, i, address, called);
                let source = match path {
                    Some(path) => self.part(&mut out, value, called.1, path),
                    None => value,
                };
                self.ids.insert(instruction.id.0, source.0);
                continue;
            }
            // In a function of the row, the element is the row, and a
            // sample of the producer's own input reads there.
            if let Some(own) = self.own {
                let op = match &instruction.op {
                    Op::Element => {
                        self.ids.insert(instruction.id.0, own.0);
                        continue;
                    }
                    Op::Sample(i) => Some((self.ops)(&Op::At(*i, own))),
                    Op::Part {
                        input: i,
                        at: None,
                        path,
                    } => {
                        let mut path = path.clone();
                        for step in &mut path {
                            if let Step::Index(index) = step {
                                *index = self.map(*index);
                            }
                        }
                        Some((self.ops)(&Op::Part {
                            input: *i,
                            at: Some(own),
                            path,
                        }))
                    }
                    _ => None,
                };
                if let Some(op) = op {
                    let id = self.push(&mut out, (self.types)(instruction.ty), op);
                    self.ids.insert(instruction.id.0, id.0);
                    continue;
                }
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
            Op::Spend(counted) | Op::Emit(counted) => *counted = self.map(*counted),
            Op::Look => {}
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
            | Op::Pads
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
