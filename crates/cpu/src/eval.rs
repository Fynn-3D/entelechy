//! The IR interpreter.
//!
//! A module is compiled once per execution into word-addressed code and
//! that code runs once per element. Every value lives in a flat stack of
//! 32-bit words at an offset fixed at compile time, in the layout the ABI
//! gives elements: a scalar is one word, a vector one per component, a
//! struct its fields in order. Reading an argument, a struct field, a
//! vector component, or a loop's carried value is therefore an alias
//! resolved at compile time and costs nothing at run time; a Field read
//! is a copy of words out of a slice the scope lent once; and no
//! execution allocates per element.

use std::fmt;

use orla_abi::{Domain, Invocation, Scope, Type, Value};
use orla_ir::{BinaryOp, Block, Function, Instruction, Intrinsic, Op, UnaryOp, Validated, ValueId};

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

/// Where a value lives: a frame-relative word offset and a width in words.
#[derive(Clone, Copy, Debug)]
struct Loc {
    at: usize,
    n: usize,
}

/// Word layout of one struct type.
struct StructLayout {
    words: usize,
    field_offsets: Vec<usize>,
}

/// How a vector component is made from a part.
#[derive(Clone, Copy)]
enum Widen {
    /// Already a float word.
    Float,
    /// An `Int` or `Bool` word, converted.
    Int,
}

/// How an address is read for `at` and `at_linear`.
#[derive(Clone, Copy)]
enum Address {
    Int,
    Float,
    Vec2,
    Vec3,
}

type F1 = fn(f32) -> f32;
type F2 = fn(f32, f32) -> f32;
type F3 = fn(f32, f32, f32) -> f32;

/// One compiled instruction. Offsets are relative to the running frame;
/// `n` is a width in words; `sa`, `sb`, `sc` are operand strides, zero
/// for a scalar broadcast beside a vector.
enum Instr {
    Const {
        dst: usize,
        words: [u32; 4],
        n: usize,
    },
    Param {
        dst: usize,
        at: usize,
        n: usize,
    },
    Move {
        dst: usize,
        src: usize,
        n: usize,
    },
    Gather {
        dst: usize,
        parts: Vec<Loc>,
    },
    Vector {
        dst: usize,
        parts: Vec<(usize, Widen)>,
    },
    FNeg {
        dst: usize,
        src: usize,
        n: usize,
    },
    INeg {
        dst: usize,
        src: usize,
    },
    Not {
        dst: usize,
        src: usize,
    },
    FBin {
        f: F2,
        dst: usize,
        a: usize,
        sa: usize,
        b: usize,
        sb: usize,
        n: usize,
    },
    IBin {
        op: BinaryOp,
        dst: usize,
        a: usize,
        b: usize,
    },
    FCmp {
        op: BinaryOp,
        dst: usize,
        a: usize,
        b: usize,
    },
    ICmp {
        op: BinaryOp,
        dst: usize,
        a: usize,
        b: usize,
    },
    FEq {
        dst: usize,
        a: usize,
        b: usize,
        n: usize,
        negate: bool,
    },
    IEq {
        dst: usize,
        a: usize,
        b: usize,
        negate: bool,
    },
    And {
        dst: usize,
        a: usize,
        b: usize,
    },
    Or {
        dst: usize,
        a: usize,
        b: usize,
    },
    Map1 {
        f: F1,
        dst: usize,
        src: usize,
        n: usize,
    },
    Map2 {
        f: F2,
        dst: usize,
        a: usize,
        sa: usize,
        b: usize,
        sb: usize,
        n: usize,
    },
    Map3 {
        f: F3,
        dst: usize,
        a: usize,
        sa: usize,
        b: usize,
        sb: usize,
        c: usize,
        sc: usize,
        n: usize,
    },
    IMap1 {
        f: fn(i32) -> i32,
        dst: usize,
        src: usize,
    },
    IMap2 {
        f: fn(i32, i32) -> i32,
        dst: usize,
        a: usize,
        b: usize,
    },
    IClamp {
        dst: usize,
        v: usize,
        low: usize,
        high: usize,
    },
    ToFloat {
        dst: usize,
        src: usize,
    },
    ToInt {
        dst: usize,
        src: usize,
    },
    Dot {
        dst: usize,
        a: usize,
        b: usize,
        n: usize,
    },
    Cross {
        dst: usize,
        a: usize,
        b: usize,
    },
    Length {
        dst: usize,
        a: usize,
        n: usize,
    },
    Normalize {
        dst: usize,
        a: usize,
        n: usize,
    },
    Reflect {
        dst: usize,
        incident: usize,
        normal: usize,
        n: usize,
    },
    Atan2 {
        dst: usize,
        y: usize,
        x: usize,
    },
    If {
        condition: usize,
        then: Vec<Instr>,
        then_result: usize,
        otherwise: Vec<Instr>,
        otherwise_result: usize,
        dst: usize,
        n: usize,
    },
    Match {
        scrutinee: usize,
        /// Each arm's values, code and result; the first holding the
        /// value runs.
        arms: Vec<(Vec<i32>, Vec<Instr>, usize)>,
        otherwise: Vec<Instr>,
        otherwise_result: usize,
        dst: usize,
        n: usize,
    },
    Loop {
        count: u32,
        init: usize,
        carried: usize,
        n: usize,
        until: Option<(Vec<Instr>, usize)>,
        body: Vec<Instr>,
        body_result: usize,
    },
    Index {
        dst: usize,
    },
    Call {
        function: usize,
        args: Vec<Loc>,
        dst: usize,
        n: usize,
    },
    Coord {
        dst: usize,
        n: usize,
    },
    Extent {
        dst: usize,
    },
    Element {
        dst: usize,
    },
    Time {
        dst: usize,
    },
    Rate {
        dst: usize,
    },
    Pointer {
        dst: usize,
    },
    PointerDown {
        dst: usize,
    },
    Key {
        dst: usize,
        code: usize,
    },
    Button {
        dst: usize,
        code: usize,
    },
    Wheel {
        dst: usize,
    },
    Window {
        dst: usize,
    },
    Scale {
        dst: usize,
    },
    Count {
        dst: usize,
        input: usize,
    },
    Size {
        dst: usize,
        input: usize,
    },
    Sample {
        dst: usize,
        input: usize,
        n: usize,
    },
    At {
        dst: usize,
        input: usize,
        address: usize,
        kind: Address,
        n: usize,
    },
    AtLinear {
        dst: usize,
        input: usize,
        position: usize,
        kind: Address,
        n: usize,
    },
    Put {
        dst: usize,
        input: usize,
        address: usize,
        kind: Address,
        value: usize,
        n: usize,
    },
}

/// A compiled function: its frame size in words, where its arguments
/// and results live in the frame, and its code.
struct Code {
    frame: usize,
    args: Vec<Loc>,
    results: Vec<Loc>,
    code: Vec<Instr>,
    /// Where every value of the function lives in its frame, by id.
    locs: Vec<Loc>,
}

/// An input the scope lent for the execution.
struct InputView<'s> {
    words: &'s [u32],
    domain: Domain,
    stride: usize,
}

/// A module compiled for one invocation, shared by every thread that
/// runs it.
pub(crate) struct Program<'s> {
    functions: Vec<Code>,
    entry: Code,
    merge: Option<Code>,
    init: Option<Code>,
    /// Words of a scan's carried value at the front of the entry frame.
    carry: usize,
    inputs: Vec<InputView<'s>>,
    /// Which inputs are memories, written by the steps of a scan.
    memories: Vec<bool>,
    params: Vec<u32>,
    /// The domain the entry maps over: the output's, or the first input's
    /// for a reduction or a scan.
    pub(crate) domain: Domain,
    context: orla_abi::Context,
}

/// What every function's compilation shares.
struct Shared {
    structs: Vec<StructLayout>,
    functions: Vec<Code>,
    param_at: Vec<usize>,
}

impl Shared {
    fn words_of(&self, ty: Type) -> usize {
        match ty {
            Type::Struct(index) => self.structs[index as usize].words,
            Type::Vec2 => 2,
            Type::Vec3 => 3,
            Type::Vec4 => 4,
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
                    Type::Vec2 => 2,
                    Type::Vec3 => 3,
                    Type::Vec4 => 4,
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
fn value_words(value: Value) -> ([u32; 4], usize) {
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
fn compile_function(shared: &Shared, function: &Function, carry: Option<Type>) -> Code {
    let slots = function.slot_count();
    let mut compiler = FunctionCompiler {
        shared,
        locs: vec![Loc { at: 0, n: 0 }; slots],
        types: vec![Type::Float; slots],
        next: 0,
        args: Vec::new(),
        loops: Vec::new(),
        carry: None,
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

struct FunctionCompiler<'c> {
    shared: &'c Shared,
    locs: Vec<Loc>,
    types: Vec<Type>,
    next: usize,
    /// Where the function's arguments live.
    args: Vec<Loc>,
    /// The carried value of each enclosing loop, innermost last.
    loops: Vec<Loc>,
    /// A scan's carried value, read by `Carried` outside every loop.
    carry: Option<Loc>,
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
        for instruction in instructions {
            let slot = instruction.id.0 as usize;
            self.types[slot] = instruction.ty;
            if let Some(instr) = self.instruction(instruction) {
                code.push(instr);
            }
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
                if ty == Type::Int {
                    Instr::INeg { dst: dst.at, src }
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
                        b: lb.at,
                    },
                    BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Rem => {
                        if ta == Type::Int {
                            Instr::IBin {
                                op: *op,
                                dst: dst.at,
                                a: la.at,
                                b: lb.at,
                            }
                        } else {
                            let f: F2 = match op {
                                BinaryOp::Add => |a, b| a + b,
                                BinaryOp::Sub => |a, b| a - b,
                                BinaryOp::Mul => |a, b| a * b,
                                BinaryOp::Div => |a, b| a / b,
                                // Truncated, like `/`, and like WGSL's `%` on floats.
                                _ => |a, b| a % b,
                            };
                            Instr::FBin {
                                f,
                                dst: dst.at,
                                a: la.at,
                                sa: self.stride(a, dst.n),
                                b: lb.at,
                                sb: self.stride(b, dst.n),
                                n: dst.n,
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
                        if matches!(ta, Type::Int | Type::Bool) && tb == ta {
                            Instr::IEq {
                                dst: dst.at,
                                a: la.at,
                                b: lb.at,
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
            Op::Construct(parts) => {
                let dst = define(self);
                if let Type::Struct(_) = ty {
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
                    _ => *component as usize,
                };
                self.locs[slot] = Loc {
                    at: base.at + offset,
                    n: self.shared.words_of(ty),
                };
                return None;
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
        let is_int = |i: usize| self.ty(&args[i]) == Type::Int;
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
            Intrinsic::Floor => map1(f32::floor),
            Intrinsic::Ceil => map1(f32::ceil),
            Intrinsic::Round => map1(f32::round_ties_even),
            Intrinsic::Sign if is_int(0) => Instr::IMap1 {
                f: i32::signum,
                dst: dst.at,
                src: loc(0),
            },
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
            Intrinsic::Abs if is_int(0) => Instr::IMap1 {
                f: i32::wrapping_abs,
                dst: dst.at,
                src: loc(0),
            },
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
            Intrinsic::Min if is_int(0) => Instr::IMap2 {
                f: i32::min,
                dst: dst.at,
                a: loc(0),
                b: loc(1),
            },
            Intrinsic::Max if is_int(0) => Instr::IMap2 {
                f: i32::max,
                dst: dst.at,
                a: loc(0),
                b: loc(1),
            },
            Intrinsic::Min => map2(f32::min),
            Intrinsic::Max => map2(f32::max),
            Intrinsic::AddWrap => Instr::IMap2 {
                f: i32::wrapping_add,
                dst: dst.at,
                a: loc(0),
                b: loc(1),
            },
            Intrinsic::SubWrap => Instr::IMap2 {
                f: i32::wrapping_sub,
                dst: dst.at,
                a: loc(0),
                b: loc(1),
            },
            Intrinsic::MulWrap => Instr::IMap2 {
                f: i32::wrapping_mul,
                dst: dst.at,
                a: loc(0),
                b: loc(1),
            },
            Intrinsic::Clamp if is_int(0) => Instr::IClamp {
                dst: dst.at,
                v: loc(0),
                low: loc(1),
                high: loc(2),
            },
            Intrinsic::Clamp => map3(|v, low, high| v.max(low).min(high)),
            Intrinsic::Mix => map3(|a, b, t| a + (b - a) * t),
            Intrinsic::ToFloat => Instr::ToFloat {
                dst: dst.at,
                src: loc(0),
            },
            Intrinsic::ToInt if is_int(0) || self.ty(&args[0]) == Type::Bool => Instr::Move {
                dst: dst.at,
                src: loc(0),
                n: 1,
            },
            Intrinsic::ToInt => Instr::ToInt {
                dst: dst.at,
                src: loc(0),
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
        }
    }
}

/// Interpreter state for one thread: a stack of words holding every
/// frame, and the loop indices of the running element.
/// One call a traced run made: which function, and its frame as the
/// call left it.
pub(crate) struct CallRecord {
    pub function: usize,
    pub frame: Vec<u32>,
}

/// How many calls a trace records before it stops recording.
pub(crate) const CALL_RECORDS: usize = 256;

pub(crate) struct Machine<'p, 's> {
    program: &'p Program<'s>,
    /// The calls made while recording, for a trace; `None` when not.
    calls: Option<Vec<CallRecord>>,
    /// Whether more calls happened than were recorded.
    calls_truncated: bool,
    /// The memories' working copies, one per memory input: what the
    /// steps have written so far, and the outputs at the end.
    memories: Vec<Option<Vec<u32>>>,
    stack: Vec<u32>,
    /// Top of the stack: where the next call's frame begins.
    sp: usize,
    loops: Vec<i32>,
    at: [u32; 3],
    scratch: Vec<f32>,
}

fn f(word: u32) -> f32 {
    f32::from_bits(word)
}

impl<'p, 's> Machine<'p, 's> {
    pub(crate) fn new(program: &'p Program<'s>) -> Self {
        let memories = program
            .inputs
            .iter()
            .zip(&program.memories)
            .map(|(view, memory)| memory.then(|| view.words.to_vec()))
            .collect();
        Self {
            program,
            calls: None,
            calls_truncated: false,
            memories,
            stack: vec![0; program.entry.frame * 2],
            sp: 0,
            loops: Vec::new(),
            at: [0; 3],
            scratch: Vec::new(),
        }
    }

    /// Runs the entry point for one element and writes its results into
    /// `out` as words, result after result.
    pub(crate) fn run_entry_words(
        &mut self,
        at: [u32; 3],
        out: &mut [u32],
    ) -> Result<(), ExecutionError> {
        let program = self.program;
        let entry = &program.entry;
        self.at = at;
        self.loops.clear();
        self.begin(entry.frame);
        self.run(&entry.code, 0)?;
        let mut cursor = 0;
        for result in &entry.results {
            out[cursor..cursor + result.n]
                .copy_from_slice(&self.stack[result.at..result.at + result.n]);
            cursor += result.n;
        }
        Ok(())
    }

    /// Runs the entry for element `index` of a scan with the carried value
    /// `carried`, writing the step's result (the next carried value) into
    /// `out`.
    pub(crate) fn run_scan_step(
        &mut self,
        index: u32,
        carried: &[u32],
        out: &mut [u32],
    ) -> Result<(), ExecutionError> {
        let program = self.program;
        let entry = &program.entry;
        self.at = [index, 0, 0];
        self.loops.clear();
        self.loops.push(index as i32);
        self.begin(entry.frame);
        self.stack[..program.carry].copy_from_slice(carried);
        self.run(&entry.code, 0)?;
        let result = entry.results[0];
        out.copy_from_slice(&self.stack[result.at..result.at + result.n]);
        Ok(())
    }

    /// Runs the module's `init` and writes the carried value into `out`.
    pub(crate) fn run_init_words(&mut self, out: &mut [u32]) -> Result<(), ExecutionError> {
        let program = self.program;
        let init = program
            .init
            .as_ref()
            .expect("only scan modules have an init");
        self.at = [0; 3];
        self.loops.clear();
        self.begin(init.frame);
        self.run(&init.code, 0)?;
        let result = init.results[0];
        out.copy_from_slice(&self.stack[result.at..result.at + result.n]);
        Ok(())
    }

    /// Runs the module's `merge` on two encoded values and writes the
    /// result into `out`.
    pub(crate) fn run_merge_words(
        &mut self,
        a: &[u32],
        b: &[u32],
        out: &mut [u32],
    ) -> Result<(), ExecutionError> {
        let program = self.program;
        let merge = program.merge.as_ref().expect("only reduce modules merge");
        self.at = [0; 3];
        self.loops.clear();
        self.begin(merge.frame);
        let (la, lb) = (merge.args[0], merge.args[1]);
        self.stack[la.at..la.at + la.n].copy_from_slice(a);
        self.stack[lb.at..lb.at + lb.n].copy_from_slice(b);
        self.run(&merge.code, 0)?;
        let result = merge.results[0];
        out.copy_from_slice(&self.stack[result.at..result.at + result.n]);
        Ok(())
    }

    /// Starts recording the calls made from here on, for a trace.
    pub(crate) fn record_calls(&mut self) {
        self.calls = Some(Vec::new());
        self.calls_truncated = false;
    }

    /// The calls recorded, and whether any went unrecorded.
    pub(crate) fn take_calls(&mut self) -> (Vec<CallRecord>, bool) {
        (self.calls.take().unwrap_or_default(), self.calls_truncated)
    }

    /// The words of value `id` of function `function` in a recorded frame.
    pub(crate) fn function_value(
        &self,
        function: usize,
        frame: &[u32],
        id: orla_ir::ValueId,
    ) -> Vec<u32> {
        let loc = self.program.functions[function].locs[id.0 as usize];
        frame[loc.at..loc.at + loc.n].to_vec()
    }

    /// The words of the `index`th result of function `function` in a
    /// recorded frame.
    pub(crate) fn function_result(&self, function: usize, frame: &[u32], index: usize) -> Vec<u32> {
        let loc = self.program.functions[function].results[index];
        frame[loc.at..loc.at + loc.n].to_vec()
    }

    /// Zeros the entry's frame, so a trace of one step shows nothing
    /// left over from the steps before it.
    pub(crate) fn clear_entry_frame(&mut self) {
        let frame = self.program.entry.frame;
        if self.stack.len() < frame {
            self.stack.resize(frame, 0);
        }
        self.stack[..frame].fill(0);
    }

    /// The words of the entry's value `id`, as the last run of the entry
    /// left them in its frame (the frame begins at the stack's start):
    /// the value's last assignment, or zeros if its block never ran.
    pub(crate) fn entry_value(&self, id: orla_ir::ValueId) -> Vec<u32> {
        let loc = self.program.entry.locs[id.0 as usize];
        self.stack[loc.at..loc.at + loc.n].to_vec()
    }

    /// The words of the entry's `index`th result after a run.
    pub(crate) fn entry_result(&self, index: usize) -> Vec<u32> {
        let loc = self.program.entry.results[index];
        self.stack[loc.at..loc.at + loc.n].to_vec()
    }

    /// The memories as the steps left them, by input index.
    pub(crate) fn take_memories(&mut self) -> Vec<(usize, Vec<u32>)> {
        self.memories
            .iter_mut()
            .enumerate()
            .filter_map(|(index, memory)| memory.take().map(|words| (index, words)))
            .collect()
    }

    /// Opens the outermost frame of `words`.
    fn begin(&mut self, words: usize) {
        if self.stack.len() < words {
            self.stack.resize(words, 0);
        }
        self.sp = words;
    }

    fn arithmetic(&self, error: ArithmeticError) -> ExecutionError {
        let [x, y, z] = self.at;
        ExecutionError::Arithmetic {
            x,
            y: z * self.program.domain.height() + y,
            error,
        }
    }

    /// Runs `code` with its frame at `fp`.
    fn run(&mut self, code: &[Instr], fp: usize) -> Result<(), ExecutionError> {
        for instr in code {
            self.step(instr, fp)?;
        }
        Ok(())
    }

    fn step(&mut self, instr: &Instr, fp: usize) -> Result<(), ExecutionError> {
        let program = self.program;
        let s = &mut self.stack;
        match instr {
            Instr::Const { dst, words, n } => {
                s[fp + dst..fp + dst + n].copy_from_slice(&words[..*n]);
            }
            Instr::Param { dst, at, n } => {
                s[fp + dst..fp + dst + n].copy_from_slice(&program.params[*at..at + n]);
            }
            Instr::Move { dst, src, n } => {
                if src != dst {
                    s.copy_within(fp + src..fp + src + n, fp + dst);
                }
            }
            Instr::Gather { dst, parts } => {
                let mut cursor = fp + dst;
                for part in parts {
                    s.copy_within(fp + part.at..fp + part.at + part.n, cursor);
                    cursor += part.n;
                }
            }
            Instr::Vector { dst, parts } => {
                for (k, (src, widen)) in parts.iter().enumerate() {
                    let word = s[fp + src];
                    s[fp + dst + k] = match widen {
                        Widen::Float => word,
                        Widen::Int => (word as i32 as f32).to_bits(),
                    };
                }
            }
            Instr::FNeg { dst, src, n } => {
                for i in 0..*n {
                    s[fp + dst + i] = (-f(s[fp + src + i])).to_bits();
                }
            }
            Instr::INeg { dst, src } => {
                let v = (s[fp + src] as i32)
                    .checked_neg()
                    .ok_or(ArithmeticError::Overflow)
                    .map_err(|e| self.arithmetic(e))?;
                self.stack[fp + dst] = v as u32;
            }
            Instr::Not { dst, src } => s[fp + dst] = (s[fp + src] == 0) as u32,
            Instr::FBin {
                f: op,
                dst,
                a,
                sa,
                b,
                sb,
                n,
            } => {
                for i in 0..*n {
                    s[fp + dst + i] = op(f(s[fp + a + i * sa]), f(s[fp + b + i * sb])).to_bits();
                }
            }
            Instr::IBin { op, dst, a, b } => {
                let (a, b) = (s[fp + a] as i32, s[fp + b] as i32);
                let result = match op {
                    BinaryOp::Add => a.checked_add(b).ok_or(ArithmeticError::Overflow),
                    BinaryOp::Sub => a.checked_sub(b).ok_or(ArithmeticError::Overflow),
                    BinaryOp::Mul => a.checked_mul(b).ok_or(ArithmeticError::Overflow),
                    BinaryOp::Div if b == 0 => Err(ArithmeticError::DivisionByZero),
                    BinaryOp::Div => a.checked_div(b).ok_or(ArithmeticError::Overflow),
                    BinaryOp::Rem if b == 0 => Err(ArithmeticError::DivisionByZero),
                    BinaryOp::Rem => a.checked_rem(b).ok_or(ArithmeticError::Overflow),
                    BinaryOp::BitAnd => Ok(a & b),
                    BinaryOp::BitOr => Ok(a | b),
                    BinaryOp::BitXor => Ok(a ^ b),
                    // The count modulo 32, as the GPU masks it.
                    BinaryOp::Shl => Ok(a.wrapping_shl(b as u32)),
                    BinaryOp::Shr => Ok(a.wrapping_shr(b as u32)),
                    BinaryOp::Ushr => Ok((a as u32).wrapping_shr(b as u32) as i32),
                    _ => unreachable!("compiled from an arithmetic operator"),
                };
                let v = result.map_err(|e| self.arithmetic(e))?;
                self.stack[fp + dst] = v as u32;
            }
            Instr::FCmp { op, dst, a, b } => {
                let (a, b) = (f(s[fp + a]), f(s[fp + b]));
                // NaN compares false, as in IEEE 754 and every GPU language.
                let holds = match op {
                    BinaryOp::Less => a < b,
                    BinaryOp::LessEqual => a <= b,
                    BinaryOp::Greater => a > b,
                    _ => a >= b,
                };
                s[fp + dst] = holds as u32;
            }
            Instr::ICmp { op, dst, a, b } => {
                let (a, b) = (s[fp + a] as i32, s[fp + b] as i32);
                let holds = match op {
                    BinaryOp::Less => a < b,
                    BinaryOp::LessEqual => a <= b,
                    BinaryOp::Greater => a > b,
                    _ => a >= b,
                };
                s[fp + dst] = holds as u32;
            }
            Instr::FEq {
                dst,
                a,
                b,
                n,
                negate,
            } => {
                let equal = (0..*n).all(|i| f(s[fp + a + i]) == f(s[fp + b + i]));
                s[fp + dst] = (equal != *negate) as u32;
            }
            Instr::IEq { dst, a, b, negate } => {
                s[fp + dst] = ((s[fp + a] == s[fp + b]) != *negate) as u32;
            }
            Instr::And { dst, a, b } => s[fp + dst] = (s[fp + a] != 0 && s[fp + b] != 0) as u32,
            Instr::Or { dst, a, b } => s[fp + dst] = (s[fp + a] != 0 || s[fp + b] != 0) as u32,
            Instr::Map1 { f: op, dst, src, n } => {
                for i in 0..*n {
                    s[fp + dst + i] = op(f(s[fp + src + i])).to_bits();
                }
            }
            Instr::Map2 {
                f: op,
                dst,
                a,
                sa,
                b,
                sb,
                n,
            } => {
                for i in 0..*n {
                    s[fp + dst + i] = op(f(s[fp + a + i * sa]), f(s[fp + b + i * sb])).to_bits();
                }
            }
            Instr::Map3 {
                f: op,
                dst,
                a,
                sa,
                b,
                sb,
                c,
                sc,
                n,
            } => {
                for i in 0..*n {
                    s[fp + dst + i] = op(
                        f(s[fp + a + i * sa]),
                        f(s[fp + b + i * sb]),
                        f(s[fp + c + i * sc]),
                    )
                    .to_bits();
                }
            }
            Instr::IMap1 { f: op, dst, src } => s[fp + dst] = op(s[fp + src] as i32) as u32,
            Instr::IMap2 { f: op, dst, a, b } => {
                s[fp + dst] = op(s[fp + a] as i32, s[fp + b] as i32) as u32;
            }
            Instr::IClamp { dst, v, low, high } => {
                let (v, low, high) = (s[fp + v] as i32, s[fp + low] as i32, s[fp + high] as i32);
                s[fp + dst] = v.max(low).min(high) as u32;
            }
            Instr::ToFloat { dst, src } => s[fp + dst] = (s[fp + src] as i32 as f32).to_bits(),
            Instr::ToInt { dst, src } => s[fp + dst] = (f(s[fp + src]) as i32) as u32,
            Instr::Dot { dst, a, b, n } => {
                let v = (0..*n).fold(0.0f32, |sum, i| sum + f(s[fp + a + i]) * f(s[fp + b + i]));
                s[fp + dst] = v.to_bits();
            }
            Instr::Cross { dst, a, b } => {
                let a = [f(s[fp + a]), f(s[fp + a + 1]), f(s[fp + a + 2])];
                let b = [f(s[fp + b]), f(s[fp + b + 1]), f(s[fp + b + 2])];
                s[fp + dst] = (a[1] * b[2] - a[2] * b[1]).to_bits();
                s[fp + dst + 1] = (a[2] * b[0] - a[0] * b[2]).to_bits();
                s[fp + dst + 2] = (a[0] * b[1] - a[1] * b[0]).to_bits();
            }
            Instr::Length { dst, a, n } => {
                let v = (0..*n).fold(0.0f32, |sum, i| sum + f(s[fp + a + i]) * f(s[fp + a + i]));
                s[fp + dst] = v.sqrt().to_bits();
            }
            Instr::Normalize { dst, a, n } => {
                let length = (0..*n)
                    .fold(0.0f32, |sum, i| sum + f(s[fp + a + i]) * f(s[fp + a + i]))
                    .sqrt();
                for i in 0..*n {
                    let v = f(s[fp + a + i]);
                    s[fp + dst + i] = if length > 0.0 { v / length } else { 0.0 }.to_bits();
                }
            }
            Instr::Reflect {
                dst,
                incident,
                normal,
                n,
            } => {
                let scale = 2.0
                    * (0..*n).fold(0.0f32, |sum, i| {
                        sum + f(s[fp + normal + i]) * f(s[fp + incident + i])
                    });
                for i in 0..*n {
                    s[fp + dst + i] =
                        (f(s[fp + incident + i]) - scale * f(s[fp + normal + i])).to_bits();
                }
            }
            Instr::Atan2 { dst, y, x } => s[fp + dst] = f(s[fp + y]).atan2(f(s[fp + x])).to_bits(),
            Instr::If {
                condition,
                then,
                then_result,
                otherwise,
                otherwise_result,
                dst,
                n,
            } => {
                let result = if s[fp + condition] != 0 {
                    self.run(then, fp)?;
                    *then_result
                } else {
                    self.run(otherwise, fp)?;
                    *otherwise_result
                };
                if result != *dst {
                    self.stack
                        .copy_within(fp + result..fp + result + n, fp + dst);
                }
            }
            Instr::Match {
                scrutinee,
                arms,
                otherwise,
                otherwise_result,
                dst,
                n,
            } => {
                let value = s[fp + scrutinee] as i32;
                let (code, result) = arms
                    .iter()
                    .find(|(values, _, _)| values.contains(&value))
                    .map_or((otherwise, *otherwise_result), |(_, code, result)| {
                        (code, *result)
                    });
                self.run(code, fp)?;
                if result != *dst {
                    self.stack
                        .copy_within(fp + result..fp + result + n, fp + dst);
                }
            }
            Instr::Loop {
                count,
                init,
                carried,
                n,
                until,
                body,
                body_result,
            } => {
                s.copy_within(fp + init..fp + init + n, fp + carried);
                self.loops.push(0);
                for index in 0..*count {
                    *self.loops.last_mut().expect("loop is open") = index as i32;
                    if let Some((until, done)) = until {
                        self.run(until, fp)?;
                        if self.stack[fp + done] != 0 {
                            break;
                        }
                    }
                    self.run(body, fp)?;
                    self.stack
                        .copy_within(fp + body_result..fp + body_result + n, fp + carried);
                }
                self.loops.pop();
            }
            Instr::Index { dst } => {
                s[fp + dst] = *self.loops.last().expect("validation guarantees a loop") as u32;
            }
            Instr::Call {
                function,
                args,
                dst,
                n,
            } => {
                let callee = &program.functions[*function];
                let base = self.sp;
                if s.len() < base + callee.frame {
                    s.resize(base + callee.frame, 0);
                }
                for (arg, param) in args.iter().zip(&callee.args) {
                    s.copy_within(fp + arg.at..fp + arg.at + arg.n, base + param.at);
                }
                self.sp = base + callee.frame;
                self.run(&callee.code, base)?;
                if let Some(records) = &mut self.calls {
                    if records.len() < CALL_RECORDS {
                        records.push(CallRecord {
                            function: *function,
                            frame: self.stack[base..base + callee.frame].to_vec(),
                        });
                    } else {
                        self.calls_truncated = true;
                    }
                }
                let result = callee.results[0];
                self.stack
                    .copy_within(base + result.at..base + result.at + n, fp + dst);
                self.sp = base;
            }
            Instr::Coord { dst, n } => {
                for (i, axis) in self.at.iter().enumerate().take(*n) {
                    s[fp + dst + i] = (*axis as f32).to_bits();
                }
            }
            Instr::Extent { dst } => write_extent(s, fp + dst, program.domain),
            Instr::Element { dst } => {
                let [x, y, z] = self.at;
                let domain = program.domain;
                s[fp + dst] = (z * domain.height() + y) * domain.width() + x;
            }
            Instr::Time { dst } => s[fp + dst] = program.context.time.to_bits(),
            Instr::Rate { dst } => s[fp + dst] = program.context.rate.to_bits(),
            Instr::Pointer { dst } => {
                let [x, y] = program.context.input.pointer;
                s[fp + dst] = x.to_bits();
                s[fp + dst + 1] = y.to_bits();
            }
            Instr::PointerDown { dst } => s[fp + dst] = program.context.input.pointer_down as u32,
            Instr::Key { dst, code } => {
                s[fp + dst] = program.context.input.key(s[fp + code] as i32) as u32;
            }
            Instr::Button { dst, code } => {
                s[fp + dst] = program.context.input.button(s[fp + code] as i32) as u32;
            }
            Instr::Wheel { dst } => {
                let [x, y] = program.context.input.wheel;
                s[fp + dst] = x.to_bits();
                s[fp + dst + 1] = y.to_bits();
            }
            Instr::Window { dst } => {
                let [w, h] = program.context.window;
                s[fp + dst] = w.to_bits();
                s[fp + dst + 1] = h.to_bits();
            }
            Instr::Scale { dst } => s[fp + dst] = program.context.scale.to_bits(),
            Instr::Count { dst, input } => {
                s[fp + dst] = program.inputs[*input].domain.count() as u32;
            }
            Instr::Size { dst, input } => write_extent(s, fp + dst, program.inputs[*input].domain),
            Instr::Sample { dst, input, n } => {
                let view = &program.inputs[*input];
                let [x, y, z] = self.at;
                let index = linear(
                    view.domain,
                    [
                        x.min(view.domain.width() - 1),
                        y.min(view.domain.height() - 1),
                        z.min(view.domain.depth() - 1),
                    ],
                );
                let start = index * view.stride;
                let words = self.memories[*input].as_deref().unwrap_or(view.words);
                s[fp + dst..fp + dst + n].copy_from_slice(&words[start..start + n]);
            }
            Instr::At {
                dst,
                input,
                address,
                kind,
                n,
            } => {
                let view = &program.inputs[*input];
                let address = read_address(s, fp + address, *kind, 0.0);
                let index = linear(view.domain, clamp_address(view.domain, address));
                let start = index * view.stride;
                let words = self.memories[*input].as_deref().unwrap_or(view.words);
                s[fp + dst..fp + dst + n].copy_from_slice(&words[start..start + n]);
            }
            Instr::Put {
                dst,
                input,
                address,
                kind,
                value,
                n,
            } => {
                // The value written is the instruction's own value; an
                // address outside the memory is dropped, never clamped.
                let view = &program.inputs[*input];
                let written = fp + value;
                s.copy_within(written..written + n, fp + dst);
                let index = match kind {
                    Address::Int => {
                        let i = s[fp + address] as i32;
                        (i >= 0 && (i as usize) < view.domain.count()).then_some(i as usize)
                    }
                    _ => {
                        let address = read_address(s, fp + address, *kind, 0.0);
                        let words: Vec<u32> = address.iter().map(|v| v.to_bits()).collect();
                        crate::scatter_index(view.domain, &words)
                    }
                };
                if let (Some(index), Some(memory)) = (index, self.memories[*input].as_mut()) {
                    let start = index * view.stride;
                    memory[start..start + n].copy_from_slice(&s[written..written + n]);
                }
            }
            Instr::AtLinear {
                dst,
                input,
                position,
                kind,
                n,
            } => {
                let view = &program.inputs[*input];
                let address = read_address(s, fp + position, *kind, 0.5);
                let domain = view.domain;
                // Element centres sit at half-integers: the position is
                // moved by a half, split into the lower neighbour and a
                // weight, and each axis is clamped at the edges.
                let axis = |v: f32, limit: u32| -> (usize, usize, f32) {
                    let t = if v.is_nan() { 0.0 } else { v - 0.5 };
                    let lower = t.floor().clamp(0.0, (limit - 1) as f32);
                    let upper = (lower + 1.0).min((limit - 1) as f32);
                    let weight = (t - t.floor()).clamp(0.0, 1.0);
                    let weight = if t < 0.0 { 0.0 } else { weight };
                    (lower as usize, upper as usize, weight)
                };
                let (x0, x1, fx) = axis(address[0], domain.width());
                let (y0, y1, fy) = axis(address[1], domain.height());
                let (z0, z1, fz) = axis(address[2], domain.depth());
                self.scratch.clear();
                self.scratch.resize(*n, 0.0);
                let (width, height) = (domain.width() as usize, domain.height() as usize);
                for (corner, weight) in [
                    ((x0, y0, z0), (1.0 - fx) * (1.0 - fy) * (1.0 - fz)),
                    ((x1, y0, z0), fx * (1.0 - fy) * (1.0 - fz)),
                    ((x0, y1, z0), (1.0 - fx) * fy * (1.0 - fz)),
                    ((x1, y1, z0), fx * fy * (1.0 - fz)),
                    ((x0, y0, z1), (1.0 - fx) * (1.0 - fy) * fz),
                    ((x1, y0, z1), fx * (1.0 - fy) * fz),
                    ((x0, y1, z1), (1.0 - fx) * fy * fz),
                    ((x1, y1, z1), fx * fy * fz),
                ] {
                    if weight == 0.0 {
                        continue;
                    }
                    let (cx, cy, cz) = corner;
                    let start = ((cz * height + cy) * width + cx) * view.stride;
                    let words = self.memories[*input].as_deref().unwrap_or(view.words);
                    for (slot, word) in self.scratch.iter_mut().zip(&words[start..]) {
                        *slot += f(*word) * weight;
                    }
                }
                for (i, v) in self.scratch.iter().enumerate() {
                    self.stack[fp + dst + i] = v.to_bits();
                }
            }
        }
        Ok(())
    }
}

/// Writes a domain's extent as the value its shape is addressed with.
fn write_extent(stack: &mut [u32], at: usize, domain: Domain) {
    match domain {
        Domain::Rows(n) => stack[at] = n,
        Domain::Grid(w, h) => {
            stack[at] = (w as f32).to_bits();
            stack[at + 1] = (h as f32).to_bits();
        }
        Domain::Volume(w, h, d) => {
            stack[at] = (w as f32).to_bits();
            stack[at + 1] = (h as f32).to_bits();
            stack[at + 2] = (d as f32).to_bits();
        }
    }
}

/// Reads an address of the given kind; missing axes take `rest`.
fn read_address(stack: &[u32], at: usize, kind: Address, rest: f32) -> [f32; 3] {
    match kind {
        Address::Int => [stack[at] as i32 as f32, rest, rest],
        Address::Float => [f(stack[at]), rest, rest],
        Address::Vec2 => [f(stack[at]), f(stack[at + 1]), rest],
        Address::Vec3 => [f(stack[at]), f(stack[at + 1]), f(stack[at + 2])],
    }
}

/// Each axis floored and clamped into range (NaN counts as zero), so no
/// read is ever outside.
fn clamp_address(domain: Domain, address: [f32; 3]) -> [u32; 3] {
    let clamp = |value: f32, limit: u32| -> u32 {
        if value.is_nan() {
            0
        } else {
            value.floor().clamp(0.0, (limit - 1) as f32) as u32
        }
    };
    [
        clamp(address[0], domain.width()),
        clamp(address[1], domain.height()),
        clamp(address[2], domain.depth()),
    ]
}

/// The linear index of an in-range address.
fn linear(domain: Domain, [x, y, z]: [u32; 3]) -> usize {
    ((z * domain.height() + y) * domain.width() + x) as usize
}
