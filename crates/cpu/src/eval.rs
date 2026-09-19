//! The IR interpreter.
//!
//! Values live in one flat stack. A plain value takes one slot; a struct
//! takes one slot per scalar field, nested structs flattened, so aggregate
//! operations are range copies within the stack and nothing allocates
//! per pixel.

use std::fmt;

use orla_abi::{Domain, Invocation, Scope, Type, Value};
use orla_ir::{
    BinaryOp, Block, Function, Instruction, Intrinsic, Module, Op, UnaryOp, Validated, ValueId,
};

use crate::ExecutionError;

/// Integer arithmetic that cannot produce a value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArithmeticError {
    /// Integer division or remainder by zero.
    DivisionByZero,
    /// Integer result does not fit in 32 bits.
    Overflow,
}

impl fmt::Display for ArithmeticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DivisionByZero => f.write_str("integer division by zero"),
            Self::Overflow => f.write_str("integer overflow"),
        }
    }
}

/// Slot layout of one struct type.
struct StructLayout {
    size: usize,
    field_offsets: Vec<usize>,
}

/// Slot layout of one function: where each argument and each value lives
/// relative to the frame.
struct FunctionLayout {
    arg_offsets: Vec<usize>,
    args_size: usize,
    /// Offset of each value id within the value area.
    offsets: Vec<usize>,
    /// Slot count of each value id.
    sizes: Vec<usize>,
    values_size: usize,
}

struct Layouts {
    structs: Vec<StructLayout>,
    functions: Vec<FunctionLayout>,
    entry: FunctionLayout,
    merge: Option<FunctionLayout>,
    init: Option<FunctionLayout>,
}

impl Layouts {
    fn new(module: &Module) -> Self {
        let mut structs: Vec<StructLayout> = Vec::with_capacity(module.structs.len());
        for def in &module.structs {
            let mut field_offsets = Vec::with_capacity(def.fields.len());
            let mut size = 0;
            for field in &def.fields {
                field_offsets.push(size);
                size += match field.ty {
                    Type::Struct(index) => structs[index as usize].size,
                    _ => 1,
                };
            }
            structs.push(StructLayout {
                size,
                field_offsets,
            });
        }
        let size_of = |ty: Type| match ty {
            Type::Struct(index) => structs[index as usize].size,
            _ => 1,
        };
        let function_layout = |function: &Function| {
            let mut arg_offsets = Vec::with_capacity(function.args.len());
            let mut args_size = 0;
            for arg in &function.args {
                arg_offsets.push(args_size);
                args_size += size_of(*arg);
            }
            let slots = function.slot_count();
            let mut offsets = vec![0; slots];
            let mut sizes = vec![1; slots];
            let mut next = 0;
            fn walk(
                instructions: &[Instruction],
                size_of: &dyn Fn(Type) -> usize,
                offsets: &mut [usize],
                sizes: &mut [usize],
                next: &mut usize,
            ) {
                for instruction in instructions {
                    let slot = instruction.id.0 as usize;
                    offsets[slot] = *next;
                    sizes[slot] = size_of(instruction.ty);
                    *next += sizes[slot];
                    match &instruction.op {
                        Op::If {
                            then, otherwise, ..
                        } => {
                            walk(&then.body, size_of, offsets, sizes, next);
                            walk(&otherwise.body, size_of, offsets, sizes, next);
                        }
                        Op::Loop { until, body, .. } => {
                            if let Some(until) = until {
                                walk(&until.body, size_of, offsets, sizes, next);
                            }
                            walk(&body.body, size_of, offsets, sizes, next);
                        }
                        _ => {}
                    }
                }
            }
            walk(
                &function.body,
                &size_of,
                &mut offsets,
                &mut sizes,
                &mut next,
            );
            FunctionLayout {
                arg_offsets,
                args_size,
                offsets,
                sizes,
                values_size: next,
            }
        };
        Self {
            functions: module.functions.iter().map(function_layout).collect(),
            entry: function_layout(&module.entry),
            merge: module.merge.as_ref().map(function_layout),
            init: module.init.as_ref().map(function_layout),
            structs,
        }
    }
}

/// Interpreter state for one thread. All frames live on one value stack:
/// a frame holds its arguments followed by the slots of every value of the
/// function, nested blocks included, so calls and loops allocate nothing.
pub(crate) struct Machine<'m> {
    module: &'m Validated,
    invocation: &'m Invocation,
    /// The domain the entry maps over: the output's, or the first input's
    /// for a reduction.
    domain: Domain,
    layouts: Layouts,
    stack: Vec<Value>,
}

impl<'m> Machine<'m> {
    pub(crate) fn new(module: &'m Validated, invocation: &'m Invocation, domain: Domain) -> Self {
        let layouts = Layouts::new(module);
        let stack = Vec::with_capacity(layouts.entry.values_size * 2);
        Self {
            module,
            invocation,
            domain,
            layouts,
            stack,
        }
    }

    /// Runs the entry point for one element and encodes its results into
    /// `out` as words, result after result.
    pub(crate) fn run_entry_words(
        &mut self,
        scope: &Scope<'_>,
        at: [u32; 3],
        out: &mut [u32],
    ) -> Result<(), ExecutionError> {
        self.stack.clear();
        let layout = &self.layouts.entry;
        self.stack.resize(layout.values_size, Value::Float(0.0));
        let entry = &self.module.entry;
        let mut frame = Frame {
            module: self.module,
            invocation: self.invocation,
            domain: self.domain,
            layouts: &self.layouts,
            layout,
            scope,
            at,
            base: 0,
            values: 0,
            loops: Vec::new(),
        };
        frame.instructions(&entry.body, &mut self.stack)?;
        let mut cursor = 0;
        for id in &entry.results {
            let ty = find_type(entry, *id).unwrap_or(Type::Float);
            cursor += encode(
                self.module,
                &self.layouts,
                ty,
                &self.stack[layout.offsets[id.0 as usize]..],
                &mut out[cursor..],
            );
        }
        Ok(())
    }

    /// Runs the entry for element `index` of a scan with the carried value
    /// `carried`, encoding the step's result (the next carried value) into
    /// `out`.
    pub(crate) fn run_scan_step(
        &mut self,
        scope: &Scope<'_>,
        index: u32,
        carried: &[u32],
        out: &mut [u32],
    ) -> Result<(), ExecutionError> {
        let layout = &self.layouts.entry;
        let entry = &self.module.entry;
        let ty = self.module.manifest.outputs[0].element();
        let carry_size = match ty {
            Type::Struct(index) => self.layouts.structs[index as usize].size,
            _ => 1,
        };
        self.stack.clear();
        self.stack
            .resize(carry_size + layout.values_size, Value::Float(0.0));
        decode(self.module, &self.layouts, ty, carried, &mut self.stack, 0);
        let mut frame = Frame {
            module: self.module,
            invocation: self.invocation,
            domain: self.domain,
            layouts: &self.layouts,
            layout,
            scope,
            at: [index, 0, 0],
            base: 0,
            values: carry_size,
            loops: vec![LoopState {
                carried: 0,
                index: index as i32,
            }],
        };
        frame.instructions(&entry.body, &mut self.stack)?;
        let result = entry.results[0];
        encode(
            self.module,
            &self.layouts,
            ty,
            &self.stack[carry_size + layout.offsets[result.0 as usize]..],
            out,
        );
        Ok(())
    }

    /// Runs the module's `init` and encodes the carried value into `out`.
    pub(crate) fn run_init_words(
        &mut self,
        scope: &Scope<'_>,
        out: &mut [u32],
    ) -> Result<(), ExecutionError> {
        let (init, layout) = match (&self.module.init, &self.layouts.init) {
            (Some(init), Some(layout)) => (init, layout),
            _ => unreachable!("only scan modules have an init"),
        };
        self.stack.clear();
        self.stack.resize(layout.values_size, Value::Float(0.0));
        let mut frame = Frame {
            module: self.module,
            invocation: self.invocation,
            domain: self.domain,
            layouts: &self.layouts,
            layout,
            scope,
            at: [0; 3],
            base: 0,
            values: 0,
            loops: Vec::new(),
        };
        frame.instructions(&init.body, &mut self.stack)?;
        let result = init.results[0];
        let ty = find_type(init, result).unwrap_or(Type::Float);
        encode(
            self.module,
            &self.layouts,
            ty,
            &self.stack[layout.offsets[result.0 as usize]..],
            out,
        );
        Ok(())
    }

    /// Runs the module's `merge` on two encoded values and encodes the
    /// result into `out`.
    pub(crate) fn run_merge_words(
        &mut self,
        scope: &Scope<'_>,
        a: &[u32],
        b: &[u32],
        out: &mut [u32],
    ) -> Result<(), ExecutionError> {
        let (merge, layout) = match (&self.module.merge, &self.layouts.merge) {
            (Some(merge), Some(layout)) => (merge, layout),
            _ => unreachable!("only reduce modules merge"),
        };
        self.stack.clear();
        self.stack
            .resize(layout.args_size + layout.values_size, Value::Float(0.0));
        let ty = merge.args[0];
        decode(
            self.module,
            &self.layouts,
            ty,
            a,
            &mut self.stack,
            layout.arg_offsets[0],
        );
        decode(
            self.module,
            &self.layouts,
            ty,
            b,
            &mut self.stack,
            layout.arg_offsets[1],
        );
        let mut frame = Frame {
            module: self.module,
            invocation: self.invocation,
            domain: self.domain,
            layouts: &self.layouts,
            layout,
            scope,
            at: [0; 3],
            base: 0,
            values: layout.args_size,
            loops: Vec::new(),
        };
        frame.instructions(&merge.body, &mut self.stack)?;
        let result = merge.results[0];
        let ty = find_type(merge, result).unwrap_or(Type::Float);
        encode(
            self.module,
            &self.layouts,
            ty,
            &self.stack[layout.args_size + layout.offsets[result.0 as usize]..],
            out,
        );
        Ok(())
    }
}

/// Encodes the value of type `ty` starting at `slots[0]` into words,
/// returning how many words were written.
fn encode(module: &Module, layouts: &Layouts, ty: Type, slots: &[Value], out: &mut [u32]) -> usize {
    match ty {
        Type::Struct(index) => {
            let def = &module.structs[index as usize];
            let layout = &layouts.structs[index as usize];
            let mut written = 0;
            for (field, offset) in def.fields.iter().zip(&layout.field_offsets) {
                written += encode(
                    module,
                    layouts,
                    field.ty,
                    &slots[*offset..],
                    &mut out[written..],
                );
            }
            written
        }
        _ => {
            let words: Vec<u32> = match slots[0] {
                Value::Float(v) => vec![v.to_bits()],
                Value::Int(v) => vec![v as u32],
                Value::Bool(v) => vec![v as u32],
                Value::Vec2(v) => v.iter().map(|c| c.to_bits()).collect(),
                Value::Vec3(v) => v.iter().map(|c| c.to_bits()).collect(),
                Value::Vec4(v) => v.iter().map(|c| c.to_bits()).collect(),
            };
            out[..words.len()].copy_from_slice(&words);
            words.len()
        }
    }
}

/// Decodes words into the slots of a value of type `ty` starting at
/// `stack[dst]`, returning how many words were consumed.
fn decode(
    module: &Module,
    layouts: &Layouts,
    ty: Type,
    words: &[u32],
    stack: &mut [Value],
    dst: usize,
) -> usize {
    match ty {
        Type::Struct(index) => {
            let def = &module.structs[index as usize];
            let layout = &layouts.structs[index as usize];
            let mut consumed = 0;
            for (field, offset) in def.fields.iter().zip(&layout.field_offsets) {
                consumed += decode(
                    module,
                    layouts,
                    field.ty,
                    &words[consumed..],
                    stack,
                    dst + offset,
                );
            }
            consumed
        }
        Type::Float => {
            stack[dst] = Value::Float(f32::from_bits(words[0]));
            1
        }
        Type::Int => {
            stack[dst] = Value::Int(words[0] as i32);
            1
        }
        Type::Bool => {
            stack[dst] = Value::Bool(words[0] != 0);
            1
        }
        Type::Vec2 => {
            stack[dst] = Value::Vec2([f32::from_bits(words[0]), f32::from_bits(words[1])]);
            2
        }
        Type::Vec3 => {
            stack[dst] = Value::Vec3([
                f32::from_bits(words[0]),
                f32::from_bits(words[1]),
                f32::from_bits(words[2]),
            ]);
            3
        }
        Type::Vec4 => {
            stack[dst] = Value::Vec4([
                f32::from_bits(words[0]),
                f32::from_bits(words[1]),
                f32::from_bits(words[2]),
                f32::from_bits(words[3]),
            ]);
            4
        }
    }
}

struct LoopState {
    /// Absolute slot where the carried value lives.
    carried: usize,
    index: i32,
}

/// Where an instruction's result comes from before it is stored.
enum Produced {
    Scalar(Value),
    /// A range starting at this absolute slot, `size` slots long.
    Range(usize),
    /// Already stored.
    Done,
}

/// One activation of a function.
struct Frame<'a, 'm> {
    module: &'m Validated,
    invocation: &'m Invocation,
    domain: Domain,
    layouts: &'a Layouts,
    layout: &'a FunctionLayout,
    scope: &'a Scope<'a>,
    at: [u32; 3],
    /// Where the arguments start.
    base: usize,
    /// Where the value area starts.
    values: usize,
    loops: Vec<LoopState>,
}

impl<'m> Frame<'_, 'm> {
    fn slot(&self, id: &ValueId) -> usize {
        self.values + self.layout.offsets[id.0 as usize]
    }

    fn size(&self, id: &ValueId) -> usize {
        self.layout.sizes[id.0 as usize]
    }

    fn instructions(
        &mut self,
        instructions: &'m [Instruction],
        stack: &mut Vec<Value>,
    ) -> Result<(), ExecutionError> {
        for instruction in instructions {
            let produced = self.instruction(instruction, stack)?;
            let destination = self.slot(&instruction.id);
            match produced {
                Produced::Scalar(value) => stack[destination] = value,
                Produced::Range(source) => {
                    if source != destination {
                        let size = self.size(&instruction.id);
                        stack.copy_within(source..source + size, destination);
                    }
                }
                Produced::Done => {}
            }
        }
        Ok(())
    }

    /// Runs a block and returns the absolute slot of its result.
    fn block(&mut self, block: &'m Block, stack: &mut Vec<Value>) -> Result<usize, ExecutionError> {
        self.instructions(&block.body, stack)?;
        Ok(self.slot(&block.result))
    }

    fn instruction(
        &mut self,
        instruction: &'m Instruction,
        stack: &mut Vec<Value>,
    ) -> Result<Produced, ExecutionError> {
        let [x, y, z] = self.at;
        let height = self.domain.height();
        let arithmetic = |error| ExecutionError::Arithmetic {
            x,
            y: z * height + y,
            error,
        };
        let at = |stack: &Vec<Value>, id: &ValueId| stack[self.slot(id)];
        Ok(Produced::Scalar(match &instruction.op {
            Op::Const(value) => *value,
            Op::Param(index) => self.invocation.params[*index as usize],
            Op::Arg(index) => {
                return Ok(Produced::Range(
                    self.base + self.layout.arg_offsets[*index as usize],
                ));
            }
            Op::Unary(UnaryOp::Neg, a) => negate(at(stack, a)).map_err(arithmetic)?,
            Op::Unary(UnaryOp::Not, a) => match at(stack, a) {
                Value::Bool(value) => Value::Bool(!value),
                _ => unreachable!("validation guarantees a Bool operand"),
            },
            Op::Binary(op, a, b) => binary(*op, at(stack, a), at(stack, b)).map_err(arithmetic)?,
            Op::Construct(parts) => {
                if let Type::Struct(_) = instruction.ty {
                    let destination = self.slot(&instruction.id);
                    let mut cursor = destination;
                    for part in parts {
                        let (source, size) = (self.slot(part), self.size(part));
                        stack.copy_within(source..source + size, cursor);
                        cursor += size;
                    }
                    return Ok(Produced::Done);
                }
                let mut components = [0.0; 4];
                for (slot, part) in components.iter_mut().zip(parts) {
                    *slot = float(at(stack, part));
                }
                from_parts(components, instruction.ty)
            }
            Op::Extract(source, component) => {
                let source_ty = self.type_of(source);
                if let Type::Struct(index) = source_ty {
                    let layout = &self.layouts.structs[index as usize];
                    return Ok(Produced::Range(
                        self.slot(source) + layout.field_offsets[*component as usize],
                    ));
                }
                Value::Float(components(at(stack, source))[*component as usize])
            }
            Op::Call(callee, call_args) => {
                let callee_index = *callee as usize;
                let callee = &self.module.functions[callee_index];
                let layout = &self.layouts.functions[callee_index];
                let frame_base = stack.len();
                stack.resize(
                    frame_base + layout.args_size + layout.values_size,
                    Value::Float(0.0),
                );
                for (arg, offset) in call_args.iter().zip(&layout.arg_offsets) {
                    let (source, size) = (self.slot(arg), self.size(arg));
                    stack.copy_within(source..source + size, frame_base + offset);
                }
                let mut frame = Frame {
                    module: self.module,
                    invocation: self.invocation,
                    domain: self.domain,
                    layouts: self.layouts,
                    layout,
                    scope: self.scope,
                    at: self.at,
                    base: frame_base,
                    values: frame_base + layout.args_size,
                    loops: Vec::new(),
                };
                frame.instructions(&callee.body, stack)?;
                let result = frame.slot(&callee.results[0]);
                let size = frame.size(&callee.results[0]);
                let destination = self.slot(&instruction.id);
                stack.copy_within(result..result + size, destination);
                stack.truncate(frame_base);
                return Ok(Produced::Done);
            }
            Op::Intrinsic(intrinsic, call_args) => {
                let mut args = [Value::Float(0.0); 3];
                for (slot, arg) in args.iter_mut().zip(call_args) {
                    *slot = at(stack, arg);
                }
                intrinsic_value(*intrinsic, &args[..call_args.len()])
            }
            Op::If {
                condition,
                then,
                otherwise,
            } => {
                let result = match at(stack, condition) {
                    Value::Bool(true) => self.block(then, stack)?,
                    Value::Bool(false) => self.block(otherwise, stack)?,
                    _ => unreachable!("validation guarantees a Bool condition"),
                };
                return Ok(Produced::Range(result));
            }
            Op::Loop {
                count,
                init,
                until,
                body,
            } => {
                let carried = self.slot(&instruction.id);
                let size = self.size(&instruction.id);
                let init_slot = self.slot(init);
                stack.copy_within(init_slot..init_slot + size, carried);
                self.loops.push(LoopState { carried, index: 0 });
                let outcome: Result<(), ExecutionError> = (|| {
                    for index in 0..*count {
                        self.loops.last_mut().expect("loop is open").index = index as i32;
                        if let Some(until) = until {
                            let done = self.block(until, stack)?;
                            if stack[done] == Value::Bool(true) {
                                break;
                            }
                        }
                        let next = self.block(body, stack)?;
                        stack.copy_within(next..next + size, carried);
                    }
                    Ok(())
                })();
                self.loops.pop();
                outcome?;
                return Ok(Produced::Done);
            }
            Op::Carried => {
                return Ok(Produced::Range(
                    self.loops
                        .last()
                        .expect("validation guarantees a loop")
                        .carried,
                ));
            }
            Op::Index => Value::Int(
                self.loops
                    .last()
                    .expect("validation guarantees a loop")
                    .index,
            ),
            Op::Element => {
                let domain = self.domain;
                let [x, y, z] = self.at;
                Value::Int(((z * domain.height() + y) * domain.width() + x) as i32)
            }
            Op::Count(input) => {
                let domain = self.scope.size(self.invocation.inputs[*input as usize])?;
                Value::Int(domain.count() as i32)
            }
            Op::Size(input) => extent(self.scope.size(self.invocation.inputs[*input as usize])?),
            Op::Sample(input) | Op::At(input, _) => {
                let handle = self.invocation.inputs[*input as usize];
                let domain = self.scope.size(handle)?;
                let address = match &instruction.op {
                    Op::At(_, index) => match at(stack, index) {
                        Value::Int(i) => [i as f32, 0.0, 0.0],
                        Value::Vec2([x, y]) => [x, y, 0.0],
                        Value::Vec3([x, y, z]) => [x, y, z],
                        _ => unreachable!("validation guarantees an address"),
                    },
                    _ => {
                        let [x, y, z] = self.at;
                        [x as f32, y as f32, z as f32]
                    }
                };
                let index = linear(domain, address);
                let words = self.scope.element(handle, index)?.to_vec();
                let destination = self.slot(&instruction.id);
                decode(
                    self.module,
                    self.layouts,
                    instruction.ty,
                    &words,
                    stack,
                    destination,
                );
                return Ok(Produced::Done);
            }
            Op::Coord => {
                let [x, y, z] = self.at;
                match self.domain {
                    Domain::Volume(..) => Value::Vec3([x as f32, y as f32, z as f32]),
                    _ => Value::Vec2([x as f32, y as f32]),
                }
            }
            Op::Domain => extent(self.domain),
            Op::Time => Value::Float(self.invocation.context.time),
            Op::Rate => Value::Float(self.invocation.context.rate),
            Op::Pointer => Value::Vec2(self.invocation.context.input.pointer),
            Op::PointerDown => Value::Bool(self.invocation.context.input.pointer_down),
            Op::Key(code) => {
                let Value::Int(code) = at(stack, code) else {
                    unreachable!("validation guarantees an Int key code")
                };
                Value::Bool(self.invocation.context.input.key(code))
            }
        }))
    }

    /// Type of a value in this frame's function, found by its slot size and
    /// the struct table: plain values are one slot, so only multi-slot
    /// values need the instruction looked up.
    fn type_of(&self, id: &ValueId) -> Type {
        if self.layout.sizes[id.0 as usize] == 1 && !self.is_single_slot_struct(id) {
            return Type::Float;
        }
        find_type(self.function(), *id).unwrap_or(Type::Float)
    }

    fn is_single_slot_struct(&self, id: &ValueId) -> bool {
        // A struct with exactly one scalar field also occupies one slot; look
        // it up only when the module declares such a struct at all.
        self.layouts.structs.iter().any(|layout| layout.size == 1)
            && matches!(find_type(self.function(), *id), Some(Type::Struct(_)))
    }

    fn function(&self) -> &'m Function {
        if std::ptr::eq(self.layout, &self.layouts.entry) {
            &self.module.entry
        } else if let (Some(merge), Some(layout)) = (&self.module.merge, &self.layouts.merge)
            && std::ptr::eq(self.layout, layout)
        {
            merge
        } else if let (Some(init), Some(layout)) = (&self.module.init, &self.layouts.init)
            && std::ptr::eq(self.layout, layout)
        {
            init
        } else {
            let index = self
                .layouts
                .functions
                .iter()
                .position(|layout| std::ptr::eq(layout, self.layout))
                .expect("frame layout belongs to the module");
            &self.module.functions[index]
        }
    }
}

fn find_type(function: &Function, id: ValueId) -> Option<Type> {
    fn find(instructions: &[Instruction], id: ValueId) -> Option<Type> {
        for instruction in instructions {
            if instruction.id == id {
                return Some(instruction.ty);
            }
            let nested = match &instruction.op {
                Op::If {
                    then, otherwise, ..
                } => find(&then.body, id).or_else(|| find(&otherwise.body, id)),
                Op::Loop { until, body, .. } => until
                    .as_ref()
                    .and_then(|block| find(&block.body, id))
                    .or_else(|| find(&body.body, id)),
                _ => None,
            };
            if nested.is_some() {
                return nested;
            }
        }
        None
    }
    find(&function.body, id)
}

/// The extent of a domain as the value its shape is addressed with.
fn extent(domain: Domain) -> Value {
    match domain {
        Domain::Rows(n) => Value::Int(n as i32),
        Domain::Grid(w, h) => Value::Vec2([w as f32, h as f32]),
        Domain::Volume(w, h, d) => Value::Vec3([w as f32, h as f32, d as f32]),
    }
}

/// The linear index of an address in a domain, each axis floored and
/// clamped into range (NaN counts as zero), so no read is ever outside.
fn linear(domain: Domain, address: [f32; 3]) -> u32 {
    let clamp = |value: f32, limit: u32| -> u32 {
        if value.is_nan() {
            0
        } else {
            value.floor().clamp(0.0, (limit - 1) as f32) as u32
        }
    };
    let x = clamp(address[0], domain.width());
    let y = clamp(address[1], domain.height());
    let z = clamp(address[2], domain.depth());
    (z * domain.height() + y) * domain.width() + x
}

fn float(value: Value) -> f32 {
    match value {
        Value::Float(v) => v,
        Value::Int(v) => v as f32,
        Value::Bool(v) => v as u8 as f32,
        Value::Vec2([x, _]) => x,
        Value::Vec3([x, ..]) => x,
        Value::Vec4([x, ..]) => x,
    }
}

/// Components padded to four; scalars broadcast.
fn components(value: Value) -> [f32; 4] {
    match value {
        Value::Float(v) => [v, v, v, v],
        Value::Int(v) => [v as f32; 4],
        Value::Bool(v) => [v as u8 as f32; 4],
        Value::Vec2([x, y]) => [x, y, 0.0, 0.0],
        Value::Vec3([x, y, z]) => [x, y, z, 0.0],
        Value::Vec4(v) => v,
    }
}

fn arity(value: Value) -> usize {
    value.ty().components().unwrap_or(1) as usize
}

fn from_parts(parts: [f32; 4], ty: Type) -> Value {
    match ty {
        Type::Vec2 => Value::Vec2([parts[0], parts[1]]),
        Type::Vec3 => Value::Vec3([parts[0], parts[1], parts[2]]),
        Type::Vec4 => Value::Vec4(parts),
        Type::Int => Value::Int(parts[0] as i32),
        Type::Bool => Value::Bool(parts[0] != 0.0),
        Type::Float | Type::Struct(_) => Value::Float(parts[0]),
    }
}

fn negate(value: Value) -> Result<Value, ArithmeticError> {
    Ok(match value {
        Value::Int(v) => Value::Int(v.checked_neg().ok_or(ArithmeticError::Overflow)?),
        other => map1(other, |v| -v),
    })
}

fn binary(op: BinaryOp, a: Value, b: Value) -> Result<Value, ArithmeticError> {
    match op {
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {}
        BinaryOp::And => {
            return Ok(Value::Bool(
                a == Value::Bool(true) && b == Value::Bool(true),
            ));
        }
        BinaryOp::Or => {
            return Ok(Value::Bool(
                a == Value::Bool(true) || b == Value::Bool(true),
            ));
        }
        BinaryOp::Equal => return Ok(Value::Bool(equal(a, b))),
        BinaryOp::NotEqual => return Ok(Value::Bool(!equal(a, b))),
        BinaryOp::Less | BinaryOp::LessEqual | BinaryOp::Greater | BinaryOp::GreaterEqual => {
            let ordering = match (a, b) {
                (Value::Int(a), Value::Int(b)) => a.partial_cmp(&b),
                (Value::Float(a), Value::Float(b)) => a.partial_cmp(&b),
                _ => unreachable!("validation guarantees scalar comparison operands"),
            };
            // NaN compares false, as in IEEE 754 and every GPU language.
            let holds = match (op, ordering) {
                (_, None) => false,
                (BinaryOp::Less, Some(o)) => o.is_lt(),
                (BinaryOp::LessEqual, Some(o)) => o.is_le(),
                (BinaryOp::Greater, Some(o)) => o.is_gt(),
                (BinaryOp::GreaterEqual, Some(o)) => o.is_ge(),
                _ => unreachable!(),
            };
            return Ok(Value::Bool(holds));
        }
    }
    if let (Value::Int(a), Value::Int(b)) = (a, b) {
        let result = match op {
            BinaryOp::Add => a.checked_add(b).ok_or(ArithmeticError::Overflow)?,
            BinaryOp::Sub => a.checked_sub(b).ok_or(ArithmeticError::Overflow)?,
            BinaryOp::Mul => a.checked_mul(b).ok_or(ArithmeticError::Overflow)?,
            BinaryOp::Div => {
                if b == 0 {
                    return Err(ArithmeticError::DivisionByZero);
                }
                a.checked_div(b).ok_or(ArithmeticError::Overflow)?
            }
            _ => unreachable!(),
        };
        return Ok(Value::Int(result));
    }
    let apply = |a: f32, b: f32| match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        _ => unreachable!(),
    };
    Ok(map2(a, b, apply))
}

fn equal(a: Value, b: Value) -> bool {
    match (a, b) {
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => a == b,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        _ => unreachable!("validation guarantees matching equality operands"),
    }
}

/// Applies `f` componentwise; the result takes the vector type of either
/// operand (scalars broadcast).
fn map2(a: Value, b: Value, f: impl Fn(f32, f32) -> f32) -> Value {
    let ty = if arity(a) >= arity(b) { a.ty() } else { b.ty() };
    let (a, b) = (components(a), components(b));
    from_parts(std::array::from_fn(|i| f(a[i], b[i])), ty)
}

fn map1(value: Value, f: impl Fn(f32) -> f32) -> Value {
    from_parts(components(value).map(f), value.ty())
}

fn dot(a: Value, b: Value) -> f32 {
    let (n, a, b) = (arity(a), components(a), components(b));
    (0..n).map(|i| a[i] * b[i]).sum()
}

fn intrinsic_value(intrinsic: Intrinsic, args: &[Value]) -> Value {
    match (intrinsic, args) {
        (Intrinsic::Floor, [a]) => map1(*a, f32::floor),
        (Intrinsic::Fract, [a]) => map1(*a, |v| v - v.floor()),
        (Intrinsic::Abs, [Value::Int(a)]) => Value::Int(a.wrapping_abs()),
        (Intrinsic::Abs, [a]) => map1(*a, f32::abs),
        (Intrinsic::Sqrt, [a]) => map1(*a, f32::sqrt),
        (Intrinsic::Sin, [a]) => map1(*a, f32::sin),
        (Intrinsic::Cos, [a]) => map1(*a, f32::cos),
        (Intrinsic::Exp, [a]) => map1(*a, f32::exp),
        (Intrinsic::Min, [Value::Int(a), Value::Int(b)]) => Value::Int((*a).min(*b)),
        (Intrinsic::Max, [Value::Int(a), Value::Int(b)]) => Value::Int((*a).max(*b)),
        (Intrinsic::Min, [a, b]) => map2(*a, *b, f32::min),
        (Intrinsic::Max, [a, b]) => map2(*a, *b, f32::max),
        (Intrinsic::Clamp, [Value::Int(v), Value::Int(low), Value::Int(high)]) => {
            Value::Int((*v).max(*low).min(*high))
        }
        (Intrinsic::Clamp, [v, low, high]) => {
            let low = map2(*v, *low, f32::max);
            map2(low, *high, f32::min)
        }
        (Intrinsic::Mix, [a, b, Value::Float(t)]) => map2(*a, *b, |a, b| a + (b - a) * t),
        (Intrinsic::ToFloat, [Value::Int(v)]) => Value::Float(*v as f32),
        (Intrinsic::ToFloat, [Value::Bool(v)]) => Value::Float(*v as u8 as f32),
        (Intrinsic::ToInt, [Value::Float(v)]) => Value::Int(*v as i32),
        (Intrinsic::ToInt, [Value::Bool(v)]) => Value::Int(*v as i32),
        (Intrinsic::Dot, [a, b]) => Value::Float(dot(*a, *b)),
        (Intrinsic::Cross, [a, b]) => {
            let (a, b) = (components(*a), components(*b));
            Value::Vec3([
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ])
        }
        (Intrinsic::Length, [a]) => Value::Float(dot(*a, *a).sqrt()),
        (Intrinsic::Normalize, [a]) => {
            let length = dot(*a, *a).sqrt();
            if length > 0.0 {
                map1(*a, |v| v / length)
            } else {
                map1(*a, |_| 0.0)
            }
        }
        (Intrinsic::Pow, [a, b]) => map2(*a, *b, f32::powf),
        (Intrinsic::Step, [edge, x]) => {
            map2(*edge, *x, |edge, x| if x >= edge { 1.0 } else { 0.0 })
        }
        (Intrinsic::Smoothstep, [low, high, x]) => {
            let t = map2(
                map2(*x, *low, |x, low| x - low),
                map2(*high, *low, |h, l| h - l),
                |a, b| (a / b).clamp(0.0, 1.0),
            );
            map1(t, |t| t * t * (3.0 - 2.0 * t))
        }
        (Intrinsic::Reflect, [incident, normal]) => {
            let scale = 2.0 * dot(*normal, *incident);
            map2(*incident, *normal, |i, n| i - scale * n)
        }
        (Intrinsic::Atan2, [Value::Float(y), Value::Float(x)]) => Value::Float(y.atan2(*x)),
        _ => unreachable!("validation guarantees intrinsic signatures"),
    }
}
