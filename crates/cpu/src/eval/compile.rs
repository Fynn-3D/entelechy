//! From validated IR to word code: a module compiled for one invocation,
//! its functions one by one, values given frame offsets as they are met.

use orla_abi::{Domain, Invocation, Scope, Type, Value};
use orla_ir::{BinaryOp, Block, Function, Instruction, Intrinsic, Op, UnaryOp, Validated, ValueId};

use crate::{ArithmeticError, ExecutionError};

use super::code::*;

/// An input the scope lent for the execution.
pub(super) struct InputView<'s> {
    pub(super) words: &'s [u32],
    pub(crate) domain: Domain,
    pub(super) stride: usize,
}

/// A module compiled for one invocation, shared by every thread that
/// runs it.
pub(crate) struct Program<'s> {
    pub(super) functions: Vec<Code>,
    pub(super) entry: Code,
    pub(super) merge: Option<Code>,
    pub(super) init: Option<Code>,
    /// Words of a scan's carried value at the front of the entry frame.
    pub(super) carry: usize,
    pub(super) inputs: Vec<InputView<'s>>,
    /// Which inputs are memories, written by the steps of a scan.
    pub(super) memories: Vec<bool>,
    pub(super) params: Vec<u32>,
    /// The domain the entry maps over: the output's, or the first input's
    /// for a reduction or a scan.
    pub(crate) domain: Domain,
    pub(super) context: orla_abi::Context,
}

/// What every function's compilation shares.
pub(super) struct Shared {
    pub(super) structs: Vec<StructLayout>,
    pub(super) functions: Vec<Code>,
    pub(super) param_at: Vec<usize>,
    /// The module's own struct table, which a path through an element
    /// is read against.
    pub(super) records: Vec<orla_ir::StructDef>,
    /// What one element of each input is.
    pub(super) elements: Vec<Type>,
}

impl Shared {
    fn words_of(&self, ty: Type) -> usize {
        match ty {
            Type::Struct(index) => self.structs[index as usize].words,
            Type::Vec2 => 2,
            Type::Vec3 => 3,
            Type::Vec4 => 4,
            Type::IVec2 => 2,
            Type::IVec3 => 3,
            Type::Mat2 => 4,
            Type::Mat3 => 9,
            Type::Mat4 => 16,
            Type::Float | Type::Int | Type::Bool => 1,
        }
    }
}

impl<'s> Program<'s> {
    /// Compiles `module` for `invocation`, borrowing its inputs from
    /// `scope`. Every Host access of the run happens here, through the
    /// scope's gate.
    pub(crate) fn new(
        module: &Validated,
        invocation: &Invocation,
        scope: &'s Scope<'_>,
        domain: Domain,
    ) -> Result<Self, ExecutionError> {
        let mut inputs = Vec::with_capacity(invocation.inputs.len());
        for handle in &invocation.inputs {
            inputs.push(InputView {
                words: scope.words(*handle)?,
                domain: scope.size(*handle)?,
                stride: scope.schema(*handle)?.words(),
            });
        }
        let mut params = Vec::new();
        let mut param_at = Vec::with_capacity(invocation.params.len());
        for value in &invocation.params {
            param_at.push(params.len());
            let (words, n) = value_words(*value);
            params.extend_from_slice(&words[..n]);
        }
        let mut structs: Vec<StructLayout> = Vec::with_capacity(module.structs.len());
        for def in &module.structs {
            let mut field_offsets = Vec::with_capacity(def.fields.len());
            let mut words = 0;
            for field in &def.fields {
                field_offsets.push(words);
                words += match field.ty {
                    Type::Struct(index) => structs[index as usize].words,
                    Type::Vec2 | Type::IVec2 => 2,
                    Type::Vec3 | Type::IVec3 => 3,
                    Type::Vec4 | Type::Mat2 => 4,
                    Type::Mat3 => 9,
                    Type::Mat4 => 16,
                    _ => 1,
                };
            }
            structs.push(StructLayout {
                words,
                field_offsets,
            });
        }
        let mut shared = Shared {
            structs,
            functions: Vec::with_capacity(module.functions.len()),
            param_at,
            records: module.structs.clone(),
            elements: module
                .manifest
                .inputs
                .iter()
                .map(|port| port.element())
                .collect(),
        };
        for function in &module.functions {
            let code = compile_function(&shared, function, None);
            shared.functions.push(code);
        }
        let carry_ty = (module.manifest.mode == orla_ir::Mode::Scan)
            .then(|| module.manifest.outputs[0].element());
        let carry = carry_ty.map_or(0, |ty| shared.words_of(ty));
        let entry = compile_function(&shared, &module.entry, carry_ty);
        let merge = module
            .merge
            .as_ref()
            .map(|merge| compile_function(&shared, merge, None));
        let init = module
            .init
            .as_ref()
            .map(|init| compile_function(&shared, init, None));
        let memories = module
            .manifest
            .inputs
            .iter()
            .map(|port| module.manifest.memories.contains(&port.name))
            .collect();
        Ok(Self {
            functions: shared.functions,
            entry,
            merge,
            init,
            carry,
            inputs,
            memories,
            params,
            domain,
            context: invocation.context,
        })
    }
}

/// The words of a parameter or constant, and how many are meaningful.
pub(super) fn value_words(value: Value) -> ([u32; 4], usize) {
    match value {
        Value::Float(v) => ([v.to_bits(), 0, 0, 0], 1),
        Value::Int(v) => ([v as u32, 0, 0, 0], 1),
        Value::Bool(v) => ([v as u32, 0, 0, 0], 1),
        Value::Vec2([x, y]) => ([x.to_bits(), y.to_bits(), 0, 0], 2),
        Value::Vec3([x, y, z]) => ([x.to_bits(), y.to_bits(), z.to_bits(), 0], 3),
        Value::Vec4(v) => (v.map(f32::to_bits), 4),
    }
}

/// Compiles one function. `carry` is the type of a scan's carried value,
/// laid at the front of the frame before the arguments.
pub(super) fn compile_function(shared: &Shared, function: &Function, carry: Option<Type>) -> Code {
    let slots = function.slot_count();
    let mut compiler = FunctionCompiler {
        shared,
        locs: vec![Loc { at: 0, n: 0 }; slots],
        types: vec![Type::Float; slots],
        next: 0,
        args: Vec::new(),
        loops: Vec::new(),
        carry: None,
        uses: use_counts(function, slots),
        constants: std::collections::HashMap::new(),
        pending_float: None,
        blocks: Vec::new(),
    };
    if let Some(ty) = carry {
        compiler.carry = Some(compiler.alloc(ty));
    }
    compiler.args = function.args.iter().map(|ty| compiler.alloc(*ty)).collect();
    let code = compiler.block(&function.body);
    let results = function.results.iter().map(|id| compiler.loc(id)).collect();
    Code {
        frame: compiler.next,
        args: compiler.args,
        results,
        code,
        locs: compiler.locs,
    }
}

/// A float operation as the compiler holds it for folding: what it
/// does, and where its two operands are with their strides and width.
pub(super) type FloatParts = (F2, usize, usize, usize, usize, usize);

pub(super) struct FunctionCompiler<'c> {
    pub(super) shared: &'c Shared,
    pub(super) locs: Vec<Loc>,
    pub(super) types: Vec<Type>,
    pub(super) next: usize,
    /// Where the function's arguments live.
    pub(super) args: Vec<Loc>,
    /// The carried value of each enclosing loop, innermost last.
    pub(super) loops: Vec<Loc>,
    /// A scan's carried value, read by `Carried` outside every loop.
    pub(super) carry: Option<Loc>,
    /// How many instructions read each value.
    pub(super) uses: Vec<u32>,
    /// The words of every value known when the function is compiled:
    /// literals and the structs, arrays and matrices built from them.
    pub(super) constants: std::collections::HashMap<u32, Vec<u32>>,
    /// The float operation the instruction being compiled emitted, which
    /// the next one may fold into itself.
    pub(super) pending_float: Option<(u32, FloatParts)>,
    /// The open blocks, innermost last: where each value's instruction
    /// sits in the block's code, and which of them a fold made dead.
    pub(super) blocks: Vec<BlockState>,
}

/// A block's bookkeeping while it compiles.
#[derive(Default)]
pub(super) struct BlockState {
    pub(super) positions: std::collections::HashMap<u32, usize>,
    pub(super) dead: Vec<usize>,
    /// The float operation this block emitted last, if the last
    /// instruction was one: the value it gives, its position, and its
    /// parts. Only the one just emitted folds into the next, since
    /// anything in between may write the slots it reads.
    pub(super) last_float: Option<(u32, usize, FloatParts)>,
}

/// How many operands name each value, nested blocks included.
pub(super) fn use_counts(function: &Function, slots: usize) -> Vec<u32> {
    fn walk(instructions: &[Instruction], counts: &mut Vec<u32>) {
        for instruction in instructions {
            let mut note = |id: &ValueId| {
                if let Some(count) = counts.get_mut(id.0 as usize) {
                    *count += 1;
                }
            };
            match &instruction.op {
                Op::Unary(_, a) => note(a),
                Op::Binary(_, a, b) => {
                    note(a);
                    note(b);
                }
                Op::Construct(parts) => parts.iter().for_each(&mut note),
                Op::Extract(source, _) => note(source),
                Op::Subscript(array, index) => {
                    note(array);
                    note(index);
                }
                Op::Call(_, args) | Op::Intrinsic(_, args) => args.iter().for_each(&mut note),
                Op::If {
                    condition,
                    then,
                    otherwise,
                } => {
                    note(condition);
                    note(&then.result);
                    note(&otherwise.result);
                }
                Op::Match {
                    scrutinee,
                    arms,
                    otherwise,
                } => {
                    note(scrutinee);
                    for (_, block) in arms {
                        note(&block.result);
                    }
                    note(&otherwise.result);
                }
                Op::Loop {
                    bound,
                    init,
                    until,
                    body,
                    ..
                } => {
                    if let Some(bound) = bound {
                        note(bound);
                    }
                    note(init);
                    if let Some(until) = until {
                        note(&until.result);
                    }
                    note(&body.result);
                }
                Op::At(_, a) | Op::AtLinear(_, a) | Op::Key(a) | Op::Button(a) => note(a),
                Op::Part { at, path, .. } => {
                    if let Some(at) = at {
                        note(at);
                    }
                    for step in path {
                        if let orla_ir::Step::Index(index) = step {
                            note(index);
                        }
                    }
                }
                Op::Put(_, a, v) => {
                    note(a);
                    note(v);
                }
                _ => {}
            }
            match &instruction.op {
                Op::If {
                    then, otherwise, ..
                } => {
                    walk(&then.body, counts);
                    walk(&otherwise.body, counts);
                }
                Op::Match {
                    arms, otherwise, ..
                } => {
                    for (_, block) in arms {
                        walk(&block.body, counts);
                    }
                    walk(&otherwise.body, counts);
                }
                Op::Loop { until, body, .. } => {
                    if let Some(until) = until {
                        walk(&until.body, counts);
                    }
                    walk(&body.body, counts);
                }
                _ => {}
            }
        }
    }
    let mut counts = vec![0; slots];
    walk(&function.body, &mut counts);
    // A result is read by the caller; a named value by a trace.
    let read_outside = function
        .results
        .iter()
        .chain(function.bindings.iter().map(|(id, _)| id));
    for id in read_outside {
        if let Some(count) = counts.get_mut(id.0 as usize) {
            *count += 1;
        }
    }
    counts
}

impl FunctionCompiler<'_> {
    fn alloc(&mut self, ty: Type) -> Loc {
        let loc = Loc {
            at: self.next,
            n: self.shared.words_of(ty),
        };
        self.next += loc.n;
        loc
    }

    fn loc(&self, id: &ValueId) -> Loc {
        self.locs[id.0 as usize]
    }

    fn ty(&self, id: &ValueId) -> Type {
        self.types[id.0 as usize]
    }

    /// The stride of an operand in an operation of width `n`: zero when a
    /// scalar broadcasts beside a vector.
    fn stride(&self, id: &ValueId, n: usize) -> usize {
        if self.loc(id).n == 1 && n > 1 { 0 } else { 1 }
    }

    /// Compiles a block.
    fn block(&mut self, instructions: &[Instruction]) -> Vec<Instr> {
        let mut code = Vec::with_capacity(instructions.len());
        self.blocks.push(BlockState::default());
        for instruction in instructions {
            let slot = instruction.id.0 as usize;
            self.types[slot] = instruction.ty;
            self.pending_float = None;
            if let Some(instr) = self.instruction(instruction) {
                code.push(instr);
                let pending = self.pending_float.take();
                if let Some(block) = self.blocks.last_mut() {
                    block.positions.insert(instruction.id.0, code.len() - 1);
                    // Only an operation the next instruction follows
                    // directly may fold into it.
                    block.last_float = pending.map(|(id, parts)| (id, code.len() - 1, parts));
                }
            } else if let Some(block) = self.blocks.last_mut() {
                block.last_float = None;
            }
            self.note_constant(instruction);
        }
        let block = self.blocks.pop().unwrap_or_default();
        if !block.dead.is_empty() {
            let mut dead = vec![false; code.len()];
            for position in block.dead {
                dead[position] = true;
            }
            let mut keep = dead.iter().map(|dead| !dead);
            code.retain(|_| keep.next().unwrap_or(true));
        }
        code
    }

    /// Compiles one instruction, returning its code, or nothing when the
    /// value is an alias of one that exists.
    fn instruction(&mut self, instruction: &Instruction) -> Option<Instr> {
        let slot = instruction.id.0 as usize;
        let ty = instruction.ty;
        let define = |this: &mut Self| {
            let loc = this.alloc(ty);
            this.locs[slot] = loc;
            loc
        };
        Some(match &instruction.op {
            Op::Const(value) => {
                let dst = define(self);
                let (words, n) = value_words(*value);
                Instr::Const {
                    dst: dst.at,
                    words,
                    n,
                }
            }
            Op::Param(index) => {
                let dst = define(self);
                Instr::Param {
                    dst: dst.at,
                    at: self.shared.param_at[*index as usize],
                    n: dst.n,
                }
            }
            Op::Arg(index) => {
                self.locs[slot] = self.args[*index as usize];
                return None;
            }
            Op::Unary(UnaryOp::Neg, a) => {
                let dst = define(self);
                let src = self.loc(a).at;
                if ty == Type::Int || ty.int_components().is_some() {
                    Instr::INeg {
                        dst: dst.at,
                        src,
                        n: dst.n,
                    }
                } else {
                    Instr::FNeg {
                        dst: dst.at,
                        src,
                        n: dst.n,
                    }
                }
            }
            Op::Unary(UnaryOp::Not, a) => {
                let dst = define(self);
                Instr::Not {
                    dst: dst.at,
                    src: self.loc(a).at,
                }
            }
            Op::Binary(op, a, b) => {
                let dst = define(self);
                let (ta, tb) = (self.ty(a), self.ty(b));
                let (la, lb) = (self.loc(a), self.loc(b));
                let product = match (op, ta.order(), tb.order()) {
                    (BinaryOp::Mul, Some(n), Some(_)) => Some((MatMul::MatMat, n)),
                    (BinaryOp::Mul, Some(n), None) if tb != Type::Float => {
                        Some((MatMul::MatVec, n))
                    }
                    (BinaryOp::Mul, None, Some(n)) if ta != Type::Float => {
                        Some((MatMul::VecMat, n))
                    }
                    _ => None,
                };
                if let Some((kind, n)) = product {
                    return Some(Instr::MatMul {
                        kind,
                        dst: dst.at,
                        a: la.at,
                        b: lb.at,
                        n: usize::from(n),
                    });
                }
                match op {
                    BinaryOp::BitAnd
                    | BinaryOp::BitOr
                    | BinaryOp::BitXor
                    | BinaryOp::Shl
                    | BinaryOp::Shr
                    | BinaryOp::Ushr => Instr::IBin {
                        op: *op,
                        dst: dst.at,
                        a: la.at,
                        sa: self.stride(a, dst.n),
                        b: lb.at,
                        sb: self.stride(b, dst.n),
                        n: dst.n,
                    },
                    BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Rem => {
                        if ta == Type::Int || ty.int_components().is_some() {
                            Instr::IBin {
                                op: *op,
                                dst: dst.at,
                                a: la.at,
                                sa: self.stride(a, dst.n),
                                b: lb.at,
                                sb: self.stride(b, dst.n),
                                n: dst.n,
                            }
                        } else {
                            let g: F2 = match op {
                                BinaryOp::Add => |a, b| a + b,
                                BinaryOp::Sub => |a, b| a - b,
                                BinaryOp::Mul => |a, b| a * b,
                                BinaryOp::Div => |a, b| a / b,
                                // Truncated, like `/`, and like WGSL's `%` on floats.
                                _ => |a, b| a % b,
                            };
                            let (sa, sb) = (self.stride(a, dst.n), self.stride(b, dst.n));
                            // An operand this block computed and nothing
                            // else reads folds in: two operations, one
                            // instruction, and no word through the frame.
                            let folded =
                                [(a, false), (b, true)]
                                    .into_iter()
                                    .find_map(|(operand, flip)| {
                                        let id = operand.0;
                                        if self.uses[id as usize] != 1 {
                                            return None;
                                        }
                                        let (last, position, parts) =
                                            self.blocks.last()?.last_float?;
                                        (last == id && parts.5 == dst.n)
                                            .then_some((flip, position, parts))
                                    });
                            match folded {
                                Some((flip, position, (f, ia, isa, ib, isb, _))) => {
                                    if let Some(block) = self.blocks.last_mut() {
                                        block.dead.push(position);
                                        block.last_float = None;
                                    }
                                    let (c, sc) = if flip { (la.at, sa) } else { (lb.at, sb) };
                                    Instr::FBin2 {
                                        f,
                                        g,
                                        flip,
                                        dst: dst.at,
                                        a: ia,
                                        sa: isa,
                                        b: ib,
                                        sb: isb,
                                        c,
                                        sc,
                                        n: dst.n,
                                    }
                                }
                                None => {
                                    self.pending_float =
                                        Some((instruction.id.0, (g, la.at, sa, lb.at, sb, dst.n)));
                                    Instr::FBin {
                                        f: g,
                                        dst: dst.at,
                                        a: la.at,
                                        sa,
                                        b: lb.at,
                                        sb,
                                        n: dst.n,
                                    }
                                }
                            }
                        }
                    }
                    BinaryOp::Less
                    | BinaryOp::LessEqual
                    | BinaryOp::Greater
                    | BinaryOp::GreaterEqual => {
                        if ta == Type::Int {
                            Instr::ICmp {
                                op: *op,
                                dst: dst.at,
                                a: la.at,
                                b: lb.at,
                            }
                        } else {
                            Instr::FCmp {
                                op: *op,
                                dst: dst.at,
                                a: la.at,
                                b: lb.at,
                            }
                        }
                    }
                    BinaryOp::Equal | BinaryOp::NotEqual => {
                        let negate = *op == BinaryOp::NotEqual;
                        if (matches!(ta, Type::Int | Type::Bool) || ta.int_components().is_some())
                            && tb == ta
                        {
                            Instr::IEq {
                                dst: dst.at,
                                a: la.at,
                                b: lb.at,
                                n: la.n,
                                negate,
                            }
                        } else {
                            Instr::FEq {
                                dst: dst.at,
                                a: la.at,
                                b: lb.at,
                                n: la.n,
                                negate,
                            }
                        }
                    }
                    BinaryOp::And => Instr::And {
                        dst: dst.at,
                        a: la.at,
                        b: lb.at,
                    },
                    BinaryOp::Or => Instr::Or {
                        dst: dst.at,
                        a: la.at,
                        b: lb.at,
                    },
                }
            }
            Op::Construct(parts)
                if (matches!(ty, Type::Struct(_))
                    || ty.order().is_some()
                    || ty.int_components().is_some())
                    && parts
                        .iter()
                        .all(|part| self.constants.contains_key(&part.0)) =>
            {
                // A table, known when the function compiles: its words in
                // one instruction, and the literals it was built from
                // dropped where nothing else reads them.
                let dst = define(self);
                let words: Vec<u32> = parts
                    .iter()
                    .flat_map(|part| self.constants[&part.0].clone())
                    .collect();
                if let Some(block) = self.blocks.last_mut() {
                    for part in parts {
                        if self.uses[part.0 as usize] == 1
                            && let Some(position) = block.positions.get(&part.0)
                        {
                            block.dead.push(*position);
                        }
                    }
                }
                Instr::Words { dst: dst.at, words }
            }
            Op::Construct(parts) => {
                let dst = define(self);
                if matches!(ty, Type::Struct(_))
                    || ty.order().is_some()
                    || ty.int_components().is_some()
                {
                    Instr::Gather {
                        dst: dst.at,
                        parts: parts.iter().map(|part| self.loc(part)).collect(),
                    }
                } else {
                    Instr::Vector {
                        dst: dst.at,
                        parts: parts
                            .iter()
                            .map(|part| {
                                let widen = match self.ty(part) {
                                    Type::Int | Type::Bool => Widen::Int,
                                    _ => Widen::Float,
                                };
                                (self.loc(part).at, widen)
                            })
                            .collect(),
                    }
                }
            }
            Op::Extract(source, component) => {
                let base = self.loc(source);
                let offset = match self.ty(source) {
                    Type::Struct(index) => {
                        self.shared.structs[index as usize].field_offsets[*component as usize]
                    }
                    other => *component as usize * other.order().map_or(1, usize::from),
                };
                self.locs[slot] = Loc {
                    at: base.at + offset,
                    n: self.shared.words_of(ty),
                };
                return None;
            }
            Op::Subscript(array, index) => {
                let base = self.loc(array);
                let index = self.loc(index);
                let count = match self.ty(array) {
                    Type::Struct(struct_index) => self.shared.structs[struct_index as usize]
                        .field_offsets
                        .len(),
                    other => other
                        .order()
                        .or(other.lanes().map(|(n, _)| n))
                        .map_or(1, usize::from),
                };
                let dst = define(self);
                Instr::Subscript {
                    dst: dst.at,
                    base: base.at,
                    index: index.at,
                    count,
                    n: dst.n,
                }
            }
            Op::Call(callee, call_args) => {
                let dst = define(self);
                Instr::Call {
                    function: *callee as usize,
                    args: call_args.iter().map(|arg| self.loc(arg)).collect(),
                    dst: dst.at,
                    n: dst.n,
                }
            }
            Op::Intrinsic(intrinsic, call_args) => {
                let dst = define(self);
                self.intrinsic(*intrinsic, call_args, dst)
            }
            Op::If {
                condition,
                then,
                otherwise,
            } => {
                let dst = define(self);
                let (then_code, then_result) = self.nested(then);
                let (otherwise_code, otherwise_result) = self.nested(otherwise);
                Instr::If {
                    condition: self.loc(condition).at,
                    then: then_code,
                    then_result,
                    otherwise: otherwise_code,
                    otherwise_result,
                    dst: dst.at,
                    n: dst.n,
                }
            }
            Op::Match {
                scrutinee,
                arms,
                otherwise,
            } => {
                let dst = define(self);
                let arms = arms
                    .iter()
                    .map(|(values, block)| {
                        let (code, result) = self.nested(block);
                        (values.clone(), code, result)
                    })
                    .collect();
                let (otherwise_code, otherwise_result) = self.nested(otherwise);
                Instr::Match {
                    scrutinee: self.loc(scrutinee).at,
                    arms,
                    otherwise: otherwise_code,
                    otherwise_result,
                    dst: dst.at,
                    n: dst.n,
                }
            }
            Op::Loop {
                count,
                bound,
                init,
                until,
                body,
            } => {
                let dst = define(self);
                self.loops.push(dst);
                let until = until.as_ref().map(|block| self.nested(block));
                let (body_code, body_result) = self.nested(body);
                self.loops.pop();
                Instr::Loop {
                    count: *count,
                    bound: bound.as_ref().map(|bound| self.loc(bound).at),
                    init: self.loc(init).at,
                    carried: dst.at,
                    n: dst.n,
                    until,
                    body: body_code,
                    body_result,
                }
            }
            Op::Carried => {
                self.locs[slot] = self
                    .loops
                    .last()
                    .copied()
                    .or(self.carry)
                    .expect("validation guarantees a loop or a scan");
                return None;
            }
            Op::Index => Instr::Index {
                dst: define(self).at,
            },
            Op::Coord => {
                let dst = define(self);
                Instr::Coord {
                    dst: dst.at,
                    n: dst.n,
                }
            }
            Op::Domain => Instr::Extent {
                dst: define(self).at,
            },
            Op::Time => Instr::Time {
                dst: define(self).at,
            },
            Op::Rate => Instr::Rate {
                dst: define(self).at,
            },
            Op::Pointer => Instr::Pointer {
                dst: define(self).at,
            },
            Op::PointerDown => Instr::PointerDown {
                dst: define(self).at,
            },
            Op::Button(code) => Instr::Button {
                dst: define(self).at,
                code: self.loc(code).at,
            },
            Op::Wheel => Instr::Wheel {
                dst: define(self).at,
            },
            Op::Window => Instr::Window {
                dst: define(self).at,
            },
            Op::Scale => Instr::Scale {
                dst: define(self).at,
            },
            Op::Key(code) => Instr::Key {
                dst: define(self).at,
                code: self.loc(code).at,
            },
            Op::Element => Instr::Element {
                dst: define(self).at,
            },
            Op::Count(input) => Instr::Count {
                dst: define(self).at,
                input: *input as usize,
            },
            Op::Size(input) => Instr::Size {
                dst: define(self).at,
                input: *input as usize,
            },
            Op::Sample(input) => {
                let dst = define(self);
                Instr::Sample {
                    dst: dst.at,
                    input: *input as usize,
                    n: dst.n,
                }
            }
            Op::At(input, index) => {
                let dst = define(self);
                Instr::At {
                    dst: dst.at,
                    input: *input as usize,
                    address: self.loc(index).at,
                    kind: self.address(index),
                    n: dst.n,
                }
            }
            Op::Part { input, at, path } => {
                let dst = define(self);
                let element = self.shared.elements[*input as usize];
                let layout = orla_ir::part_layout(&self.shared.records, element, path)
                    .expect("a path validation walked");
                let base = layout.base;
                let steps = layout
                    .steps
                    .iter()
                    .map(|(index, count, stride)| (self.loc(index).at, *count, *stride))
                    .collect();
                Instr::Part {
                    dst: dst.at,
                    input: *input as usize,
                    address: at.map(|at| (self.loc(&at).at, self.address(&at))),
                    base,
                    steps,
                    n: dst.n,
                }
            }
            Op::AtLinear(input, position) => {
                let dst = define(self);
                Instr::AtLinear {
                    dst: dst.at,
                    input: *input as usize,
                    position: self.loc(position).at,
                    kind: self.address(position),
                    n: dst.n,
                }
            }
            Op::Put(input, address, value) => {
                let dst = define(self);
                Instr::Put {
                    dst: dst.at,
                    input: *input as usize,
                    address: self.loc(address).at,
                    kind: self.address(address),
                    value: self.loc(value).at,
                    n: dst.n,
                }
            }
        })
    }

    /// Records a value's words when they are known at compile time: a
    /// literal, or a table built from known words.
    fn note_constant(&mut self, instruction: &Instruction) {
        let words = match &instruction.op {
            Op::Const(value) => {
                let (words, n) = value_words(*value);
                Some(words[..n].to_vec())
            }
            Op::Construct(parts)
                if (matches!(instruction.ty, Type::Struct(_))
                    || instruction.ty.order().is_some()
                    || instruction.ty.int_components().is_some())
                    && parts
                        .iter()
                        .all(|part| self.constants.contains_key(&part.0)) =>
            {
                Some(
                    parts
                        .iter()
                        .flat_map(|part| self.constants[&part.0].clone())
                        .collect(),
                )
            }
            _ => None,
        };
        if let Some(words) = words {
            self.constants.insert(instruction.id.0, words);
        }
    }

    /// Compiles a nested block, returning its code and where its result lives.
    fn nested(&mut self, block: &Block) -> (Vec<Instr>, usize) {
        let code = self.block(&block.body);
        (code, self.loc(&block.result).at)
    }

    fn address(&self, id: &ValueId) -> Address {
        match self.ty(id) {
            Type::Int => Address::Int,
            Type::Float => Address::Float,
            Type::Vec2 => Address::Vec2,
            Type::Vec3 => Address::Vec3,
            _ => unreachable!("validation guarantees an address"),
        }
    }

    fn intrinsic(&self, intrinsic: Intrinsic, args: &[ValueId], dst: Loc) -> Instr {
        let n = dst.n;
        let loc = |i: usize| self.loc(&args[i]).at;
        let is_int = |i: usize| {
            self.ty(&args[i]) == Type::Int || self.ty(&args[i]).int_components().is_some()
        };
        let imap1 = |f: fn(i32) -> i32| Instr::IMap1 {
            f,
            dst: dst.at,
            src: loc(0),
            n,
        };
        let imap2 = |f: fn(i32, i32) -> i32| Instr::IMap2 {
            f,
            dst: dst.at,
            a: loc(0),
            sa: self.stride(&args[0], n),
            b: loc(1),
            sb: self.stride(&args[1], n),
            n,
        };
        let map1 = |f: F1| Instr::Map1 {
            f,
            dst: dst.at,
            src: loc(0),
            n,
        };
        let map2 = |f: F2| Instr::Map2 {
            f,
            dst: dst.at,
            a: loc(0),
            sa: self.stride(&args[0], n),
            b: loc(1),
            sb: self.stride(&args[1], n),
            n,
        };
        let map3 = |f: F3| Instr::Map3 {
            f,
            dst: dst.at,
            a: loc(0),
            sa: self.stride(&args[0], n),
            b: loc(1),
            sb: self.stride(&args[1], n),
            c: loc(2),
            sc: self.stride(&args[2], n),
            n,
        };
        match intrinsic {
            Intrinsic::Transpose | Intrinsic::Determinant | Intrinsic::Inverse => Instr::Matrix {
                op: intrinsic,
                dst: dst.at,
                src: loc(0),
                n: usize::from(self.ty(&args[0]).order().unwrap_or(1)),
            },
            Intrinsic::Floor => map1(f32::floor),
            Intrinsic::Ceil => map1(f32::ceil),
            Intrinsic::Round => map1(f32::round_ties_even),
            Intrinsic::Sign if is_int(0) => imap1(i32::signum),
            Intrinsic::Sign => map1(|v| {
                if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }),
            Intrinsic::Fract => map1(|v| v - v.floor()),
            Intrinsic::Abs if is_int(0) => imap1(i32::wrapping_abs),
            Intrinsic::Abs => map1(f32::abs),
            Intrinsic::Sqrt => map1(f32::sqrt),
            Intrinsic::Sin => map1(f32::sin),
            Intrinsic::Cos => map1(f32::cos),
            Intrinsic::Tan => map1(f32::tan),
            Intrinsic::Asin => map1(f32::asin),
            Intrinsic::Acos => map1(f32::acos),
            Intrinsic::Atan => map1(f32::atan),
            Intrinsic::Exp => map1(f32::exp),
            Intrinsic::Log => map1(f32::ln),
            Intrinsic::Min if is_int(0) => imap2(i32::min),
            Intrinsic::Max if is_int(0) => imap2(i32::max),
            Intrinsic::Min => map2(f32::min),
            Intrinsic::Max => map2(f32::max),
            Intrinsic::AddWrap => imap2(i32::wrapping_add),
            Intrinsic::SubWrap => imap2(i32::wrapping_sub),
            Intrinsic::MulWrap => imap2(i32::wrapping_mul),
            Intrinsic::Clamp if is_int(0) => Instr::IClamp {
                dst: dst.at,
                v: loc(0),
                low: loc(1),
                sl: self.stride(&args[1], n),
                high: loc(2),
                sh: self.stride(&args[2], n),
                n,
            },
            Intrinsic::Clamp => map3(|v, low, high| v.max(low).min(high)),
            Intrinsic::Mix => map3(|a, b, t| a + (b - a) * t),
            Intrinsic::ToFloat => Instr::ToFloat {
                dst: dst.at,
                src: loc(0),
                n,
            },
            Intrinsic::ToInt if is_int(0) || self.ty(&args[0]) == Type::Bool => Instr::Move {
                dst: dst.at,
                src: loc(0),
                n: 1,
            },
            Intrinsic::ToInt => Instr::ToInt {
                dst: dst.at,
                src: loc(0),
                n,
            },
            Intrinsic::Dot => Instr::Dot {
                dst: dst.at,
                a: loc(0),
                b: loc(1),
                n: self.loc(&args[0]).n,
            },
            Intrinsic::Cross => Instr::Cross {
                dst: dst.at,
                a: loc(0),
                b: loc(1),
            },
            Intrinsic::Length => Instr::Length {
                dst: dst.at,
                a: loc(0),
                n: self.loc(&args[0]).n,
            },
            Intrinsic::Normalize => Instr::Normalize {
                dst: dst.at,
                a: loc(0),
                n,
            },
            Intrinsic::Pow => map2(f32::powf),
            Intrinsic::Step => map2(|edge, x| if x >= edge { 1.0 } else { 0.0 }),
            Intrinsic::Smoothstep => map3(|low, high, x| {
                let t = ((x - low) / (high - low)).clamp(0.0, 1.0);
                t * t * (3.0 - 2.0 * t)
            }),
            Intrinsic::Reflect => Instr::Reflect {
                dst: dst.at,
                incident: loc(0),
                normal: loc(1),
                n,
            },
            Intrinsic::Atan2 => Instr::Atan2 {
                dst: dst.at,
                y: loc(0),
                x: loc(1),
            },
            Intrinsic::Trunc => map1(f32::trunc),
            Intrinsic::Exp2 => map1(f32::exp2),
            Intrinsic::Log2 => map1(f32::log2),
            Intrinsic::Sinh => map1(f32::sinh),
            Intrinsic::Cosh => map1(f32::cosh),
            Intrinsic::Tanh => map1(f32::tanh),
            Intrinsic::Fma => map3(|a, b, c| a.mul_add(b, c)),
            Intrinsic::Popcount => imap1(|x| x.count_ones() as i32),
            Intrinsic::Clz => imap1(|x| x.leading_zeros() as i32),
            Intrinsic::Ctz => imap1(|x| x.trailing_zeros() as i32),
            Intrinsic::Rotl => imap2(|x, n| x.rotate_left((n & 31) as u32)),
            Intrinsic::Rotr => imap2(|x, n| x.rotate_right((n & 31) as u32)),
            Intrinsic::Ult => imap2(|a, b| ((a as u32) < (b as u32)) as i32),
            Intrinsic::Udiv => Instr::IMap2Checked {
                f: |a, b| {
                    if b == 0 {
                        Err(ArithmeticError::DivisionByZero)
                    } else {
                        Ok(((a as u32) / (b as u32)) as i32)
                    }
                },
                dst: dst.at,
                a: loc(0),
                b: loc(1),
            },
            Intrinsic::Urem => Instr::IMap2Checked {
                f: |a, b| {
                    if b == 0 {
                        Err(ArithmeticError::DivisionByZero)
                    } else {
                        Ok(((a as u32) % (b as u32)) as i32)
                    }
                },
                dst: dst.at,
                a: loc(0),
                b: loc(1),
            },
            Intrinsic::Mulhi => imap2(|a, b| ((a as i64 * b as i64) >> 32) as i32),
            Intrinsic::Umulhi => imap2(|a, b| ((a as u32 as u64 * b as u32 as u64) >> 32) as i32),
        }
    }
}
