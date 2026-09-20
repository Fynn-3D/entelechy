//! Orla IR to Cranelift.
//!
//! Every IR value becomes its words, one Cranelift value each, `f32` or
//! `i32` (a `Bool` is an `i32` of 0 or 1), in the layout the ABI gives
//! elements, so a struct is its fields in order and a vector its
//! components. `if` and `loop` become blocks with block parameters; a
//! call passes the words; a map's entry takes the context, the element's
//! coordinates and a pointer for its output words; a scan's entry takes
//! the context, the step's index, a pointer to the carried words and a
//! pointer for the next, and its `init` writes the first carried words.
//! Memories are written through a helper. Integer arithmetic is
//! checked as the interpreter checks it, the first failure recorded in
//! the context. Field reads and the transcendental functions call the
//! helpers in the crate root, by symbol, so they round as the CPU does.

use std::sync::Arc;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    AbiParam, FuncRef, InstBuilder, MemFlags, Signature, StackSlot, StackSlotData, StackSlotKind,
    Value, types,
};
use cranelift_codegen::isa::TargetIsa;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module as _};
use orla_abi::Type;
use orla_ir::{BinaryOp, Function, Instruction, Intrinsic, Module, Op, UnaryOp, Validated};

use crate::offsets;

/// Error codes stored in the context, as the CPU backend reports them.
const DIVISION_BY_ZERO: i64 = 1;
const OVERFLOW: i64 = 2;

/// The imported helpers, by function id.
struct Imports {
    read: FuncId,
    read_rows: FuncId,
    sample: FuncId,
    linear: FuncId,
    key: FuncId,
    button: FuncId,
    put: FuncId,
    put_rows: FuncId,
    unary: Vec<(Intrinsic, FuncId)>,
    powf: FuncId,
    atan2f: FuncId,
    fmodf: FuncId,
}

/// A module's native code: the JIT module holding it, the entry's
/// address, and a scan's `init`.
pub struct Code {
    pub module: JITModule,
    pub entry: usize,
    pub init: Option<usize>,
}

/// What the function being translated is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// A function: the context, its result's slot, its arguments' words.
    Function,
    /// A map's entry: the context, `x`, `y`, `z`, the output slot.
    Map,
    /// A scan's entry: the context, the index, the carried words, the
    /// output slot.
    Scan,
}

/// Compiles the module.
pub fn compile(module: &Validated, isa: Arc<dyn TargetIsa>) -> Result<Code, String> {
    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    for (name, address) in crate::symbols() {
        builder.symbol(name, address);
    }
    let mut jit = JITModule::new(builder);
    let ptr = jit.target_config().pointer_type();
    let call_conv = jit.target_config().default_call_conv;
    let sig = move |params: &[types::Type], returns: &[types::Type]| {
        let mut signature = Signature::new(call_conv);
        signature.params = params.iter().map(|t| AbiParam::new(*t)).collect();
        signature.returns = returns.iter().map(|t| AbiParam::new(*t)).collect();
        signature
    };
    let declare = |jit: &mut JITModule, name: &str, signature: &Signature| {
        jit.declare_function(name, Linkage::Import, signature)
            .map_err(|error| error.to_string())
    };
    let (f, i) = (types::F32, types::I32);
    let read_sig = sig(&[ptr, i, f, f, f, ptr], &[]);
    let rows_sig = sig(&[ptr, i, i, ptr], &[]);
    let sample_sig = sig(&[ptr, i, i, i, i, ptr], &[]);
    let key_sig = sig(&[ptr, i], &[i]);
    let put_sig = sig(&[ptr, i, f, f, f, ptr], &[]);
    let put_rows_sig = sig(&[ptr, i, i, ptr], &[]);
    let unary_sig = sig(&[f], &[f]);
    let binary_sig = sig(&[f, f], &[f]);
    let imports = Imports {
        read: declare(&mut jit, "orla_jit_read", &read_sig)?,
        read_rows: declare(&mut jit, "orla_jit_read_rows", &rows_sig)?,
        sample: declare(&mut jit, "orla_jit_sample", &sample_sig)?,
        linear: declare(&mut jit, "orla_jit_linear", &read_sig)?,
        key: declare(&mut jit, "orla_jit_key", &key_sig)?,
        button: declare(&mut jit, "orla_jit_button", &key_sig)?,
        put: declare(&mut jit, "orla_jit_put", &put_sig)?,
        put_rows: declare(&mut jit, "orla_jit_put_rows", &put_rows_sig)?,
        unary: [
            (Intrinsic::Sin, "orla_jit_sinf"),
            (Intrinsic::Cos, "orla_jit_cosf"),
            (Intrinsic::Tan, "orla_jit_tanf"),
            (Intrinsic::Asin, "orla_jit_asinf"),
            (Intrinsic::Acos, "orla_jit_acosf"),
            (Intrinsic::Atan, "orla_jit_atanf"),
            (Intrinsic::Exp, "orla_jit_expf"),
            (Intrinsic::Log, "orla_jit_logf"),
        ]
        .into_iter()
        .map(|(intrinsic, name)| declare(&mut jit, name, &unary_sig).map(|id| (intrinsic, id)))
        .collect::<Result<Vec<_>, _>>()?,
        powf: declare(&mut jit, "orla_jit_powf", &binary_sig)?,
        atan2f: declare(&mut jit, "orla_jit_atan2f", &binary_sig)?,
        fmodf: declare(&mut jit, "orla_jit_fmodf", &binary_sig)?,
    };
    // Every function takes the context, a pointer for its result's words
    // (a struct or a vector is more words than registers return), then
    // its arguments' words; a map's entry takes the coordinates and the
    // output pointer instead, a scan's the index, the carried words'
    // pointer and the output pointer; a scan's `init` is a function
    // without arguments.
    let mut functions = Vec::with_capacity(module.functions.len());
    for (index, function) in module.functions.iter().enumerate() {
        let mut params = vec![ptr, ptr];
        for arg in &function.args {
            params.extend(word_types(module, *arg));
        }
        let signature = sig(&params, &[]);
        let id = jit
            .declare_function(&format!("f{index}"), Linkage::Local, &signature)
            .map_err(|error| error.to_string())?;
        functions.push((id, signature));
    }
    let scan = module.manifest.mode == orla_ir::Mode::Scan;
    let entry_signature = if scan {
        sig(&[ptr, i, ptr, ptr], &[])
    } else {
        sig(&[ptr, i, i, i, ptr], &[])
    };
    let entry_id = jit
        .declare_function("entry", Linkage::Local, &entry_signature)
        .map_err(|error| error.to_string())?;
    let init_signature = sig(&[ptr, ptr], &[]);
    let init_id = match &module.init {
        Some(_) if scan => Some(
            jit.declare_function("init", Linkage::Local, &init_signature)
                .map_err(|error| error.to_string())?,
        ),
        _ => None,
    };
    let max_stride = module
        .manifest
        .inputs
        .iter()
        .map(|port| port.ty.schema.words())
        .max()
        .unwrap_or(1)
        .max(1);
    let mut context = jit.make_context();
    let mut fbc = FunctionBuilderContext::new();
    for (index, function) in module.functions.iter().enumerate() {
        context.func.signature = functions[index].1.clone();
        {
            let builder = FunctionBuilder::new(&mut context.func, &mut fbc);
            Fx::new(
                module, &mut jit, builder, &imports, &functions, ptr, max_stride,
            )
            .function(function, Kind::Function);
        }
        jit.define_function(functions[index].0, &mut context)
            .map_err(|error| format!("{}: {error}", function.name))?;
        jit.clear_context(&mut context);
    }
    if let (Some(init_id), Some(init)) = (init_id, &module.init) {
        context.func.signature = init_signature;
        {
            let builder = FunctionBuilder::new(&mut context.func, &mut fbc);
            Fx::new(
                module, &mut jit, builder, &imports, &functions, ptr, max_stride,
            )
            .function(init, Kind::Function);
        }
        jit.define_function(init_id, &mut context)
            .map_err(|error| format!("init: {error}"))?;
        jit.clear_context(&mut context);
    }
    context.func.signature = entry_signature;
    {
        let builder = FunctionBuilder::new(&mut context.func, &mut fbc);
        Fx::new(
            module, &mut jit, builder, &imports, &functions, ptr, max_stride,
        )
        .function(&module.entry, if scan { Kind::Scan } else { Kind::Map });
    }
    jit.define_function(entry_id, &mut context)
        .map_err(|error| format!("entry: {error}"))?;
    jit.clear_context(&mut context);
    jit.finalize_definitions()
        .map_err(|error| error.to_string())?;
    let entry = jit.get_finalized_function(entry_id) as usize;
    let init = init_id.map(|id| jit.get_finalized_function(id) as usize);
    Ok(Code {
        module: jit,
        entry,
        init,
    })
}

/// The Cranelift type of each word of a value of `ty`.
fn word_types(module: &Module, ty: Type) -> Vec<types::Type> {
    match ty {
        Type::Float => vec![types::F32],
        Type::Int | Type::Bool => vec![types::I32],
        Type::Vec2 => vec![types::F32; 2],
        Type::Vec3 => vec![types::F32; 3],
        Type::Vec4 => vec![types::F32; 4],
        Type::Struct(index) => module.structs[index as usize]
            .fields
            .iter()
            .flat_map(|field| word_types(module, field.ty))
            .collect(),
    }
}

/// Word offset of a struct's field.
fn field_offset(module: &Module, index: u32, field: usize) -> usize {
    module.structs[index as usize].fields[..field]
        .iter()
        .map(|f| word_types(module, f.ty).len())
        .sum()
}

fn result_type(function: &Function) -> Type {
    fn find(instructions: &[Instruction], id: orla_ir::ValueId) -> Option<Type> {
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
    find(&function.body, function.results[0]).unwrap_or(Type::Float)
}

/// One function being translated.
struct Fx<'a, 'b> {
    module: &'a Module,
    b: FunctionBuilder<'b>,
    ptr: types::Type,
    /// The words of every value, by id.
    values: Vec<Vec<Value>>,
    types: Vec<Type>,
    /// The words of each argument.
    args: Vec<Vec<Value>>,
    /// Enclosing loops: the carried words and the index.
    loops: Vec<(Vec<Value>, Value)>,
    /// A scan entry's carried words and step index, read by `Carried`
    /// and `Index` outside every loop.
    scan: Option<(Vec<Value>, Value)>,
    ctx: Value,
    /// The entry's coordinates and output pointer.
    coords: Option<(Value, Value, Value, Value)>,
    calls: Vec<FuncRef>,
    read: FuncRef,
    read_rows: FuncRef,
    sample: FuncRef,
    linear: FuncRef,
    key: FuncRef,
    button: FuncRef,
    put: FuncRef,
    put_rows: FuncRef,
    unary: Vec<(Intrinsic, FuncRef)>,
    powf: FuncRef,
    atan2f: FuncRef,
    fmodf: FuncRef,
    scratch: StackSlot,
    flags: MemFlags,
}

impl<'a, 'b> Fx<'a, 'b> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        module: &'a Module,
        jit: &mut JITModule,
        mut b: FunctionBuilder<'b>,
        imports: &Imports,
        functions: &[(FuncId, Signature)],
        ptr: types::Type,
        max_stride: usize,
    ) -> Self {
        let calls = functions
            .iter()
            .map(|(id, _)| jit.declare_func_in_func(*id, b.func))
            .collect();
        let import = |jit: &mut JITModule, b: &mut FunctionBuilder<'b>, id: FuncId| {
            jit.declare_func_in_func(id, b.func)
        };
        let read = import(jit, &mut b, imports.read);
        let read_rows = import(jit, &mut b, imports.read_rows);
        let sample = import(jit, &mut b, imports.sample);
        let linear = import(jit, &mut b, imports.linear);
        let key = import(jit, &mut b, imports.key);
        let button = import(jit, &mut b, imports.button);
        let put = import(jit, &mut b, imports.put);
        let put_rows = import(jit, &mut b, imports.put_rows);
        let unary = imports
            .unary
            .iter()
            .map(|(intrinsic, id)| (*intrinsic, import(jit, &mut b, *id)))
            .collect();
        let powf = import(jit, &mut b, imports.powf);
        let atan2f = import(jit, &mut b, imports.atan2f);
        let fmodf = import(jit, &mut b, imports.fmodf);
        let scratch = b.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            (max_stride * 4) as u32,
            2,
        ));
        let entry_block = b.create_block();
        b.append_block_params_for_function_params(entry_block);
        b.switch_to_block(entry_block);
        b.seal_block(entry_block);
        let ctx = b.block_params(entry_block)[0];
        Self {
            module,
            b,
            ptr,
            values: Vec::new(),
            types: Vec::new(),
            args: Vec::new(),
            loops: Vec::new(),
            scan: None,
            ctx,
            coords: None,
            calls,
            read,
            read_rows,
            sample,
            linear,
            key,
            button,
            put,
            put_rows,
            unary,
            powf,
            atan2f,
            fmodf,
            scratch,
            flags: MemFlags::trusted(),
        }
    }

    /// Translates a whole function: its arguments from the entry block's
    /// parameters, its body, and its return.
    fn function(mut self, function: &Function, kind: Kind) {
        let slots = function.slot_count();
        self.values = vec![Vec::new(); slots];
        self.types = vec![Type::Float; slots];
        let block = self.b.current_block().expect("entry block");
        let params: Vec<Value> = self.b.block_params(block).to_vec();
        let out = match kind {
            Kind::Map => {
                self.coords = Some((params[1], params[2], params[3], params[4]));
                params[4]
            }
            Kind::Scan => {
                // The step's coordinates are its index, as the interpreter
                // has them; the carried words are loaded once.
                let zero = self.i32c(0);
                self.coords = Some((params[1], zero, zero, params[3]));
                let carried_types = word_types(self.module, result_type(function));
                let carried = carried_types
                    .iter()
                    .enumerate()
                    .map(|(k, t)| self.b.ins().load(*t, self.flags, params[2], (k * 4) as i32))
                    .collect();
                self.scan = Some((carried, params[1]));
                params[3]
            }
            Kind::Function => {
                let mut cursor = 2;
                for arg in &function.args {
                    let n = word_types(self.module, *arg).len();
                    self.args.push(params[cursor..cursor + n].to_vec());
                    cursor += n;
                }
                params[1]
            }
        };
        self.block(&function.body);
        let mut offset = 0i32;
        for result in &function.results {
            for word in self.values[result.0 as usize].clone() {
                self.b.ins().store(self.flags, word, out, offset);
                offset += 4;
            }
        }
        self.b.ins().return_(&[]);
        self.b.finalize();
    }

    fn block(&mut self, instructions: &[Instruction]) {
        for instruction in instructions {
            self.types[instruction.id.0 as usize] = instruction.ty;
            let words = self.instruction(instruction);
            self.values[instruction.id.0 as usize] = words;
        }
    }

    fn words(&self, id: &orla_ir::ValueId) -> Vec<Value> {
        self.values[id.0 as usize].clone()
    }

    fn ty(&self, id: &orla_ir::ValueId) -> Type {
        self.types[id.0 as usize]
    }

    fn f32c(&mut self, v: f32) -> Value {
        self.b.ins().f32const(v)
    }

    fn i32c(&mut self, v: i64) -> Value {
        self.b.ins().iconst(types::I32, v)
    }

    /// `Bool` word from a condition.
    fn word_of(&mut self, cond: Value) -> Value {
        self.b.ins().uextend(types::I32, cond)
    }

    /// Condition from a `Bool` word.
    fn cond_of(&mut self, word: Value) -> Value {
        self.b.ins().icmp_imm(IntCC::NotEqual, word, 0)
    }

    /// Records the first integer failure when `cond` holds.
    fn fail_if(&mut self, cond: Value, code: i64) {
        let current = self
            .b
            .ins()
            .load(types::I32, self.flags, self.ctx, offsets::ERROR);
        let none = self.b.ins().icmp_imm(IntCC::Equal, current, 0);
        let set = self.b.ins().band(none, cond);
        let code = self.i32c(code);
        let next = self.b.ins().select(set, code, current);
        self.b
            .ins()
            .store(self.flags, next, self.ctx, offsets::ERROR);
    }

    fn is_nan(&mut self, v: Value) -> Value {
        self.b.ins().fcmp(FloatCC::Unordered, v, v)
    }

    /// Rust's `f32::min`: the other operand when one is NaN.
    fn fmin(&mut self, a: Value, b: Value) -> Value {
        let less = self.b.ins().fcmp(FloatCC::LessThan, b, a);
        let picked = self.b.ins().select(less, b, a);
        let nan = self.is_nan(a);
        self.b.ins().select(nan, b, picked)
    }

    /// Rust's `f32::max`.
    fn fmax(&mut self, a: Value, b: Value) -> Value {
        let greater = self.b.ins().fcmp(FloatCC::GreaterThan, b, a);
        let picked = self.b.ins().select(greater, b, a);
        let nan = self.is_nan(a);
        self.b.ins().select(nan, b, picked)
    }

    /// Componentwise over two operands, a one-word operand broadcast.
    fn zip2(
        &mut self,
        a: &[Value],
        b: &[Value],
        mut f: impl FnMut(&mut Self, Value, Value) -> Value,
    ) -> Vec<Value> {
        let n = a.len().max(b.len());
        (0..n)
            .map(|k| {
                let x = if a.len() == 1 { a[0] } else { a[k] };
                let y = if b.len() == 1 { b[0] } else { b[k] };
                f(self, x, y)
            })
            .collect()
    }

    fn zip3(
        &mut self,
        a: &[Value],
        b: &[Value],
        c: &[Value],
        mut f: impl FnMut(&mut Self, Value, Value, Value) -> Value,
    ) -> Vec<Value> {
        let n = a.len().max(b.len()).max(c.len());
        (0..n)
            .map(|k| {
                let x = if a.len() == 1 { a[0] } else { a[k] };
                let y = if b.len() == 1 { b[0] } else { b[k] };
                let z = if c.len() == 1 { c[0] } else { c[k] };
                f(self, x, y, z)
            })
            .collect()
    }

    fn dot(&mut self, a: &[Value], b: &[Value]) -> Value {
        let mut sum = self.f32c(0.0);
        for (x, y) in a.iter().zip(b) {
            let product = self.b.ins().fmul(*x, *y);
            sum = self.b.ins().fadd(sum, product);
        }
        sum
    }

    /// Loads `n` words of the given types from the scratch slot.
    fn load_scratch(&mut self, tys: &[types::Type]) -> Vec<Value> {
        let base = self.b.ins().stack_addr(self.ptr, self.scratch, 0);
        tys.iter()
            .enumerate()
            .map(|(k, ty)| self.b.ins().load(*ty, self.flags, base, (k * 4) as i32))
            .collect()
    }

    fn scratch_addr(&mut self) -> Value {
        self.b.ins().stack_addr(self.ptr, self.scratch, 0)
    }

    /// The view of an input: its address.
    fn view(&mut self, input: u32) -> Value {
        let views = self
            .b
            .ins()
            .load(self.ptr, self.flags, self.ctx, offsets::VIEWS);
        self.b
            .ins()
            .iadd_imm(views, offsets::VIEW_SIZE * input as i64)
    }

    fn u32_to_f32(&mut self, v: Value) -> Value {
        self.b.ins().fcvt_from_uint(types::F32, v)
    }

    fn instruction(&mut self, instruction: &Instruction) -> Vec<Value> {
        let ty = instruction.ty;
        match &instruction.op {
            Op::Const(value) => {
                use orla_abi::Value as V;
                match value {
                    V::Float(v) => vec![self.f32c(*v)],
                    V::Int(v) => vec![self.i32c(*v as i64)],
                    V::Bool(v) => vec![self.i32c(*v as i64)],
                    V::Vec2(v) => v.iter().map(|c| self.f32c(*c)).collect(),
                    V::Vec3(v) => v.iter().map(|c| self.f32c(*c)).collect(),
                    V::Vec4(v) => v.iter().map(|c| self.f32c(*c)).collect(),
                }
            }
            Op::Param(index) => {
                let offset: usize = self.module.manifest.params[..*index as usize]
                    .iter()
                    .map(|p| word_types(self.module, p.default.ty()).len())
                    .sum();
                let base = self
                    .b
                    .ins()
                    .load(self.ptr, self.flags, self.ctx, offsets::PARAMS);
                word_types(self.module, ty)
                    .iter()
                    .enumerate()
                    .map(|(k, t)| {
                        self.b
                            .ins()
                            .load(*t, self.flags, base, ((offset + k) * 4) as i32)
                    })
                    .collect()
            }
            Op::Arg(index) => self.args[*index as usize].clone(),
            Op::Unary(UnaryOp::Neg, a) => {
                let words = self.words(a);
                if self.ty(a) == Type::Int {
                    let zero = self.i32c(0);
                    let (v, overflow) = self.b.ins().ssub_overflow(zero, words[0]);
                    self.fail_if(overflow, OVERFLOW);
                    vec![v]
                } else {
                    words.iter().map(|w| self.b.ins().fneg(*w)).collect()
                }
            }
            Op::Unary(UnaryOp::Not, a) => {
                let word = self.words(a)[0];
                let flipped = self.b.ins().icmp_imm(IntCC::Equal, word, 0);
                vec![self.word_of(flipped)]
            }
            Op::Binary(op, a, b) => self.binary(*op, a, b),
            Op::Construct(parts) => {
                if let Type::Struct(_) = ty {
                    parts.iter().flat_map(|part| self.words(part)).collect()
                } else {
                    parts
                        .iter()
                        .map(|part| {
                            let word = self.words(part)[0];
                            match self.ty(part) {
                                Type::Int | Type::Bool => {
                                    self.b.ins().fcvt_from_sint(types::F32, word)
                                }
                                _ => word,
                            }
                        })
                        .collect()
                }
            }
            Op::Extract(source, component) => {
                let words = self.words(source);
                match self.ty(source) {
                    Type::Struct(index) => {
                        let start = field_offset(self.module, index, *component as usize);
                        let n = word_types(self.module, ty).len();
                        words[start..start + n].to_vec()
                    }
                    _ => vec![words[*component as usize]],
                }
            }
            Op::Call(callee, args) => {
                // The callee writes its result's words to a slot of ours.
                let tys = word_types(self.module, ty);
                let slot = self.b.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    (tys.len().max(1) * 4) as u32,
                    2,
                ));
                let out = self.b.ins().stack_addr(self.ptr, slot, 0);
                let mut passed = vec![self.ctx, out];
                for arg in args {
                    passed.extend(self.words(arg));
                }
                self.b.ins().call(self.calls[*callee as usize], &passed);
                tys.iter()
                    .enumerate()
                    .map(|(k, t)| self.b.ins().load(*t, self.flags, out, (k * 4) as i32))
                    .collect()
            }
            Op::Intrinsic(intrinsic, args) => self.intrinsic(*intrinsic, args),
            Op::If {
                condition,
                then,
                otherwise,
            } => {
                let cond_word = self.words(condition)[0];
                let cond = self.cond_of(cond_word);
                let then_block = self.b.create_block();
                let else_block = self.b.create_block();
                let join = self.b.create_block();
                let result_types = word_types(self.module, ty);
                for t in &result_types {
                    self.b.append_block_param(join, *t);
                }
                self.b.ins().brif(cond, then_block, &[], else_block, &[]);
                self.b.switch_to_block(then_block);
                self.b.seal_block(then_block);
                self.block(&then.body);
                let then_words = self.words(&then.result);
                let then_args: Vec<_> = then_words.iter().map(|v| (*v).into()).collect();
                self.b.ins().jump(join, &then_args);
                self.b.switch_to_block(else_block);
                self.b.seal_block(else_block);
                self.block(&otherwise.body);
                let else_words = self.words(&otherwise.result);
                let else_args: Vec<_> = else_words.iter().map(|v| (*v).into()).collect();
                self.b.ins().jump(join, &else_args);
                self.b.switch_to_block(join);
                self.b.seal_block(join);
                self.b.block_params(join).to_vec()
            }
            Op::Loop {
                count,
                init,
                until,
                body,
            } => {
                let carried_types = word_types(self.module, ty);
                let header = self.b.create_block();
                let body_block = self.b.create_block();
                let exit = self.b.create_block();
                for t in &carried_types {
                    self.b.append_block_param(header, *t);
                    self.b.append_block_param(exit, *t);
                }
                self.b.append_block_param(header, types::I32);
                let init_words = self.words(init);
                let zero = self.i32c(0);
                let mut init_args: Vec<_> = init_words.iter().map(|v| (*v).into()).collect();
                init_args.push(zero.into());
                self.b.ins().jump(header, &init_args);
                self.b.switch_to_block(header);
                let header_params = self.b.block_params(header).to_vec();
                let carried = header_params[..carried_types.len()].to_vec();
                let index = header_params[carried_types.len()];
                self.loops.push((carried.clone(), index));
                let carried_args: Vec<_> = carried.iter().map(|v| (*v).into()).collect();
                if let Some(until) = until {
                    self.block(&until.body);
                    let done_word = self.words(&until.result)[0];
                    let done = self.cond_of(done_word);
                    self.b
                        .ins()
                        .brif(done, exit, &carried_args, body_block, &[]);
                } else {
                    self.b.ins().jump(body_block, &[]);
                }
                self.b.switch_to_block(body_block);
                self.b.seal_block(body_block);
                self.block(&body.body);
                let next = self.words(&body.result);
                let next_index = self.b.ins().iadd_imm(index, 1);
                let limit = self.i32c(*count as i64);
                let exhausted =
                    self.b
                        .ins()
                        .icmp(IntCC::UnsignedGreaterThanOrEqual, next_index, limit);
                let next_args: Vec<_> = next.iter().map(|v| (*v).into()).collect();
                let mut again: Vec<_> = next_args.clone();
                again.push(next_index.into());
                self.b
                    .ins()
                    .brif(exhausted, exit, &next_args, header, &again);
                self.b.seal_block(header);
                self.loops.pop();
                self.b.switch_to_block(exit);
                self.b.seal_block(exit);
                self.b.block_params(exit).to_vec()
            }
            Op::Carried => self
                .loops
                .last()
                .or(self.scan.as_ref())
                .expect("a loop or a scan")
                .0
                .clone(),
            Op::Index => vec![
                self.loops
                    .last()
                    .or(self.scan.as_ref())
                    .expect("a loop or a scan")
                    .1,
            ],
            Op::Coord => {
                let (x, y, z, _) = self.coords.expect("entry point");
                let mut words = vec![self.u32_to_f32(x), self.u32_to_f32(y)];
                if ty == Type::Vec3 {
                    words.push(self.u32_to_f32(z));
                }
                words
            }
            Op::Domain => {
                let w = self
                    .b
                    .ins()
                    .load(types::I32, self.flags, self.ctx, offsets::WIDTH);
                let h = self
                    .b
                    .ins()
                    .load(types::I32, self.flags, self.ctx, offsets::HEIGHT);
                let d = self
                    .b
                    .ins()
                    .load(types::I32, self.flags, self.ctx, offsets::DEPTH);
                match ty {
                    Type::Int => vec![w],
                    Type::Vec2 => vec![self.u32_to_f32(w), self.u32_to_f32(h)],
                    _ => vec![self.u32_to_f32(w), self.u32_to_f32(h), self.u32_to_f32(d)],
                }
            }
            Op::Element => {
                let (x, y, z, _) = self.coords.expect("entry point");
                let w = self
                    .b
                    .ins()
                    .load(types::I32, self.flags, self.ctx, offsets::WIDTH);
                let h = self
                    .b
                    .ins()
                    .load(types::I32, self.flags, self.ctx, offsets::HEIGHT);
                let zh = self.b.ins().imul(z, h);
                let row = self.b.ins().iadd(zh, y);
                let base = self.b.ins().imul(row, w);
                vec![self.b.ins().iadd(base, x)]
            }
            Op::Time => vec![
                self.b
                    .ins()
                    .load(types::F32, self.flags, self.ctx, offsets::TIME),
            ],
            Op::Rate => vec![
                self.b
                    .ins()
                    .load(types::F32, self.flags, self.ctx, offsets::RATE),
            ],
            Op::Pointer => vec![
                self.b
                    .ins()
                    .load(types::F32, self.flags, self.ctx, offsets::POINTER),
                self.b
                    .ins()
                    .load(types::F32, self.flags, self.ctx, offsets::POINTER + 4),
            ],
            Op::Wheel => vec![
                self.b
                    .ins()
                    .load(types::F32, self.flags, self.ctx, offsets::WHEEL),
                self.b
                    .ins()
                    .load(types::F32, self.flags, self.ctx, offsets::WHEEL + 4),
            ],
            Op::Window => vec![
                self.b
                    .ins()
                    .load(types::F32, self.flags, self.ctx, offsets::WINDOW),
                self.b
                    .ins()
                    .load(types::F32, self.flags, self.ctx, offsets::WINDOW + 4),
            ],
            Op::Scale => vec![
                self.b
                    .ins()
                    .load(types::F32, self.flags, self.ctx, offsets::SCALE),
            ],
            Op::PointerDown => {
                let zero = self.i32c(0);
                let call = self.b.ins().call(self.button, &[self.ctx, zero]);
                vec![self.b.inst_results(call)[0]]
            }
            Op::Button(code) => {
                let code = self.words(code)[0];
                let call = self.b.ins().call(self.button, &[self.ctx, code]);
                vec![self.b.inst_results(call)[0]]
            }
            Op::Key(code) => {
                let code = self.words(code)[0];
                let call = self.b.ins().call(self.key, &[self.ctx, code]);
                vec![self.b.inst_results(call)[0]]
            }
            Op::Count(input) => {
                let view = self.view(*input);
                let w = self
                    .b
                    .ins()
                    .load(types::I32, self.flags, view, offsets::VIEW_WIDTH);
                let h = self
                    .b
                    .ins()
                    .load(types::I32, self.flags, view, offsets::VIEW_HEIGHT);
                let d = self
                    .b
                    .ins()
                    .load(types::I32, self.flags, view, offsets::VIEW_DEPTH);
                let wh = self.b.ins().imul(w, h);
                vec![self.b.ins().imul(wh, d)]
            }
            Op::Size(input) => {
                let view = self.view(*input);
                let w = self
                    .b
                    .ins()
                    .load(types::I32, self.flags, view, offsets::VIEW_WIDTH);
                let h = self
                    .b
                    .ins()
                    .load(types::I32, self.flags, view, offsets::VIEW_HEIGHT);
                let d = self
                    .b
                    .ins()
                    .load(types::I32, self.flags, view, offsets::VIEW_DEPTH);
                match ty {
                    Type::Int => vec![w],
                    Type::Vec2 => vec![self.u32_to_f32(w), self.u32_to_f32(h)],
                    _ => vec![self.u32_to_f32(w), self.u32_to_f32(h), self.u32_to_f32(d)],
                }
            }
            Op::Sample(input) => {
                let (x, y, z, _) = self.coords.expect("entry point");
                let out = self.scratch_addr();
                let index = self.i32c(*input as i64);
                self.b
                    .ins()
                    .call(self.sample, &[self.ctx, index, x, y, z, out]);
                let tys = word_types(self.module, ty);
                self.load_scratch(&tys)
            }
            Op::At(input, address) => {
                let words = self.words(address);
                let out = self.scratch_addr();
                let index = self.i32c(*input as i64);
                match self.ty(address) {
                    Type::Int => {
                        self.b
                            .ins()
                            .call(self.read_rows, &[self.ctx, index, words[0], out]);
                    }
                    Type::Vec2 => {
                        let zero = self.f32c(0.0);
                        self.b
                            .ins()
                            .call(self.read, &[self.ctx, index, words[0], words[1], zero, out]);
                    }
                    _ => {
                        self.b.ins().call(
                            self.read,
                            &[self.ctx, index, words[0], words[1], words[2], out],
                        );
                    }
                }
                let tys = word_types(self.module, ty);
                self.load_scratch(&tys)
            }
            Op::AtLinear(input, position) => {
                let words = self.words(position);
                let out = self.scratch_addr();
                let index = self.i32c(*input as i64);
                let half = self.f32c(0.5);
                let (x, y, z) = match self.ty(position) {
                    Type::Float => (words[0], half, half),
                    Type::Vec2 => (words[0], words[1], half),
                    _ => (words[0], words[1], words[2]),
                };
                self.b
                    .ins()
                    .call(self.linear, &[self.ctx, index, x, y, z, out]);
                let tys = word_types(self.module, ty);
                self.load_scratch(&tys)
            }
            Op::Put(input, address, value) => {
                // The value's words go to the scratch slot; the helper
                // resolves the address as the interpreter does and drops a
                // write outside the memory. The instruction's value is the
                // value written.
                let words = self.words(value);
                let slot = self.scratch_addr();
                for (k, word) in words.iter().enumerate() {
                    self.b.ins().store(self.flags, *word, slot, (k * 4) as i32);
                }
                let index = self.i32c(*input as i64);
                let address_words = self.words(address);
                match self.ty(address) {
                    Type::Int => {
                        self.b
                            .ins()
                            .call(self.put_rows, &[self.ctx, index, address_words[0], slot]);
                    }
                    ty => {
                        let zero = self.f32c(0.0);
                        let (x, y, z) = match ty {
                            Type::Vec2 => (address_words[0], address_words[1], zero),
                            _ => (address_words[0], address_words[1], address_words[2]),
                        };
                        self.b
                            .ins()
                            .call(self.put, &[self.ctx, index, x, y, z, slot]);
                    }
                }
                words
            }
        }
    }

    fn binary(&mut self, op: BinaryOp, a: &orla_ir::ValueId, b: &orla_ir::ValueId) -> Vec<Value> {
        let (ta, tb) = (self.ty(a), self.ty(b));
        let (wa, wb) = (self.words(a), self.words(b));
        match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul if ta == Type::Int => {
                let (v, overflow) = match op {
                    BinaryOp::Add => self.b.ins().sadd_overflow(wa[0], wb[0]),
                    BinaryOp::Sub => self.b.ins().ssub_overflow(wa[0], wb[0]),
                    _ => self.b.ins().smul_overflow(wa[0], wb[0]),
                };
                self.fail_if(overflow, OVERFLOW);
                vec![v]
            }
            BinaryOp::Div | BinaryOp::Rem if ta == Type::Int => {
                // A zero divisor and MIN / -1 are the two failures; both
                // would trap, so the division runs on a safe divisor instead.
                let zero = self.b.ins().icmp_imm(IntCC::Equal, wb[0], 0);
                self.fail_if(zero, DIVISION_BY_ZERO);
                let is_min = self.b.ins().icmp_imm(IntCC::Equal, wa[0], i32::MIN as i64);
                let minus_one = self.b.ins().icmp_imm(IntCC::Equal, wb[0], -1);
                let overflow = self.b.ins().band(is_min, minus_one);
                self.fail_if(overflow, OVERFLOW);
                let unsafe_ = self.b.ins().bor(zero, overflow);
                let one = self.i32c(1);
                let divisor = self.b.ins().select(unsafe_, one, wb[0]);
                let v = if op == BinaryOp::Div {
                    self.b.ins().sdiv(wa[0], divisor)
                } else {
                    self.b.ins().srem(wa[0], divisor)
                };
                vec![v]
            }
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                self.zip2(&wa, &wb, |fx, x, y| match op {
                    BinaryOp::Add => fx.b.ins().fadd(x, y),
                    BinaryOp::Sub => fx.b.ins().fsub(x, y),
                    BinaryOp::Mul => fx.b.ins().fmul(x, y),
                    _ => fx.b.ins().fdiv(x, y),
                })
            }
            BinaryOp::Rem => self.zip2(&wa, &wb, |fx, x, y| {
                let call = fx.b.ins().call(fx.fmodf, &[x, y]);
                fx.b.inst_results(call)[0]
            }),
            BinaryOp::Less | BinaryOp::LessEqual | BinaryOp::Greater | BinaryOp::GreaterEqual => {
                let cond = if ta == Type::Int {
                    let cc = match op {
                        BinaryOp::Less => IntCC::SignedLessThan,
                        BinaryOp::LessEqual => IntCC::SignedLessThanOrEqual,
                        BinaryOp::Greater => IntCC::SignedGreaterThan,
                        _ => IntCC::SignedGreaterThanOrEqual,
                    };
                    self.b.ins().icmp(cc, wa[0], wb[0])
                } else {
                    let cc = match op {
                        BinaryOp::Less => FloatCC::LessThan,
                        BinaryOp::LessEqual => FloatCC::LessThanOrEqual,
                        BinaryOp::Greater => FloatCC::GreaterThan,
                        _ => FloatCC::GreaterThanOrEqual,
                    };
                    self.b.ins().fcmp(cc, wa[0], wb[0])
                };
                vec![self.word_of(cond)]
            }
            BinaryOp::Equal | BinaryOp::NotEqual => {
                let mut all: Option<Value> = None;
                for (x, y) in wa.iter().zip(&wb) {
                    let same = if matches!(ta, Type::Int | Type::Bool) && tb == ta {
                        self.b.ins().icmp(IntCC::Equal, *x, *y)
                    } else {
                        self.b.ins().fcmp(FloatCC::Equal, *x, *y)
                    };
                    all = Some(match all {
                        Some(prior) => self.b.ins().band(prior, same),
                        None => same,
                    });
                }
                let equal = all.expect("an operand has words");
                let word = self.word_of(equal);
                if op == BinaryOp::NotEqual {
                    vec![self.b.ins().bxor_imm(word, 1)]
                } else {
                    vec![word]
                }
            }
            BinaryOp::And => vec![self.b.ins().band(wa[0], wb[0])],
            BinaryOp::Or => vec![self.b.ins().bor(wa[0], wb[0])],
            BinaryOp::BitAnd => vec![self.b.ins().band(wa[0], wb[0])],
            BinaryOp::BitOr => vec![self.b.ins().bor(wa[0], wb[0])],
            BinaryOp::BitXor => vec![self.b.ins().bxor(wa[0], wb[0])],
            BinaryOp::Shl => {
                let count = self.b.ins().band_imm(wb[0], 31);
                vec![self.b.ins().ishl(wa[0], count)]
            }
            BinaryOp::Shr => {
                let count = self.b.ins().band_imm(wb[0], 31);
                vec![self.b.ins().sshr(wa[0], count)]
            }
            BinaryOp::Ushr => {
                let count = self.b.ins().band_imm(wb[0], 31);
                vec![self.b.ins().ushr(wa[0], count)]
            }
        }
    }

    fn intrinsic(&mut self, intrinsic: Intrinsic, args: &[orla_ir::ValueId]) -> Vec<Value> {
        let w: Vec<Vec<Value>> = args.iter().map(|a| self.words(a)).collect();
        let is_int = |fx: &Self, k: usize| fx.ty(&args[k]) == Type::Int;
        let map1 = |fx: &mut Self, f: &mut dyn FnMut(&mut Self, Value) -> Value| -> Vec<Value> {
            w[0].iter().map(|v| f(fx, *v)).collect()
        };
        match intrinsic {
            Intrinsic::Floor => map1(self, &mut |fx, v| fx.b.ins().floor(v)),
            Intrinsic::Ceil => map1(self, &mut |fx, v| fx.b.ins().ceil(v)),
            Intrinsic::Round => map1(self, &mut |fx, v| fx.b.ins().nearest(v)),
            Intrinsic::Sqrt => map1(self, &mut |fx, v| fx.b.ins().sqrt(v)),
            Intrinsic::Fract => map1(self, &mut |fx, v| {
                let f = fx.b.ins().floor(v);
                fx.b.ins().fsub(v, f)
            }),
            Intrinsic::Abs if is_int(self, 0) => vec![self.b.ins().iabs(w[0][0])],
            Intrinsic::Abs => map1(self, &mut |fx, v| fx.b.ins().fabs(v)),
            Intrinsic::Sign if is_int(self, 0) => {
                let v = w[0][0];
                let pos = self.b.ins().icmp_imm(IntCC::SignedGreaterThan, v, 0);
                let neg = self.b.ins().icmp_imm(IntCC::SignedLessThan, v, 0);
                let one = self.i32c(1);
                let minus = self.i32c(-1);
                let zero = self.i32c(0);
                let low = self.b.ins().select(neg, minus, zero);
                vec![self.b.ins().select(pos, one, low)]
            }
            Intrinsic::Sign => map1(self, &mut |fx, v| {
                let zero = fx.f32c(0.0);
                let pos = fx.b.ins().fcmp(FloatCC::GreaterThan, v, zero);
                let neg = fx.b.ins().fcmp(FloatCC::LessThan, v, zero);
                let one = fx.f32c(1.0);
                let minus = fx.f32c(-1.0);
                let low = fx.b.ins().select(neg, minus, zero);
                fx.b.ins().select(pos, one, low)
            }),
            Intrinsic::Sin
            | Intrinsic::Cos
            | Intrinsic::Tan
            | Intrinsic::Asin
            | Intrinsic::Acos
            | Intrinsic::Atan
            | Intrinsic::Exp
            | Intrinsic::Log => {
                let func = self
                    .unary
                    .iter()
                    .find(|(i, _)| *i == intrinsic)
                    .map(|(_, f)| *f)
                    .expect("declared");
                map1(self, &mut |fx, v| {
                    let call = fx.b.ins().call(func, &[v]);
                    fx.b.inst_results(call)[0]
                })
            }
            Intrinsic::Min if is_int(self, 0) => vec![self.b.ins().smin(w[0][0], w[1][0])],
            Intrinsic::Max if is_int(self, 0) => vec![self.b.ins().smax(w[0][0], w[1][0])],
            Intrinsic::Min => self.zip2(&w[0], &w[1], |fx, x, y| fx.fmin(x, y)),
            Intrinsic::Max => self.zip2(&w[0], &w[1], |fx, x, y| fx.fmax(x, y)),
            Intrinsic::Clamp if is_int(self, 0) => {
                let low = self.b.ins().smax(w[0][0], w[1][0]);
                vec![self.b.ins().smin(low, w[2][0])]
            }
            Intrinsic::Clamp => self.zip3(&w[0], &w[1], &w[2], |fx, v, low, high| {
                let floored = fx.fmax(v, low);
                fx.fmin(floored, high)
            }),
            Intrinsic::Mix => self.zip3(&w[0], &w[1], &w[2], |fx, a, b, t| {
                let span = fx.b.ins().fsub(b, a);
                let scaled = fx.b.ins().fmul(span, t);
                fx.b.ins().fadd(a, scaled)
            }),
            Intrinsic::ToFloat => vec![self.b.ins().fcvt_from_sint(types::F32, w[0][0])],
            Intrinsic::ToInt => {
                if matches!(self.ty(&args[0]), Type::Int | Type::Bool) {
                    vec![w[0][0]]
                } else {
                    vec![self.b.ins().fcvt_to_sint_sat(types::I32, w[0][0])]
                }
            }
            Intrinsic::Dot => vec![self.dot(&w[0], &w[1])],
            Intrinsic::Cross => {
                let (a, b) = (&w[0], &w[1]);
                let x1 = self.b.ins().fmul(a[1], b[2]);
                let x2 = self.b.ins().fmul(a[2], b[1]);
                let y1 = self.b.ins().fmul(a[2], b[0]);
                let y2 = self.b.ins().fmul(a[0], b[2]);
                let z1 = self.b.ins().fmul(a[0], b[1]);
                let z2 = self.b.ins().fmul(a[1], b[0]);
                vec![
                    self.b.ins().fsub(x1, x2),
                    self.b.ins().fsub(y1, y2),
                    self.b.ins().fsub(z1, z2),
                ]
            }
            Intrinsic::Length => {
                let d = self.dot(&w[0], &w[0]);
                vec![self.b.ins().sqrt(d)]
            }
            Intrinsic::Normalize => {
                let d = self.dot(&w[0], &w[0]);
                let length = self.b.ins().sqrt(d);
                let zero = self.f32c(0.0);
                let positive = self.b.ins().fcmp(FloatCC::GreaterThan, length, zero);
                w[0].iter()
                    .map(|v| {
                        let scaled = self.b.ins().fdiv(*v, length);
                        self.b.ins().select(positive, scaled, zero)
                    })
                    .collect()
            }
            Intrinsic::Pow => self.zip2(&w[0], &w[1], |fx, x, y| {
                let call = fx.b.ins().call(fx.powf, &[x, y]);
                fx.b.inst_results(call)[0]
            }),
            Intrinsic::Step => self.zip2(&w[0], &w[1], |fx, edge, x| {
                let ge = fx.b.ins().fcmp(FloatCC::GreaterThanOrEqual, x, edge);
                let one = fx.f32c(1.0);
                let zero = fx.f32c(0.0);
                fx.b.ins().select(ge, one, zero)
            }),
            Intrinsic::Smoothstep => self.zip3(&w[0], &w[1], &w[2], |fx, low, high, x| {
                let num = fx.b.ins().fsub(x, low);
                let den = fx.b.ins().fsub(high, low);
                let t = fx.b.ins().fdiv(num, den);
                let zero = fx.f32c(0.0);
                let one = fx.f32c(1.0);
                let below = fx.b.ins().fcmp(FloatCC::LessThan, t, zero);
                let t = fx.b.ins().select(below, zero, t);
                let above = fx.b.ins().fcmp(FloatCC::GreaterThan, t, one);
                let t = fx.b.ins().select(above, one, t);
                let two = fx.f32c(2.0);
                let three = fx.f32c(3.0);
                let tt = fx.b.ins().fmul(t, t);
                let twice = fx.b.ins().fmul(two, t);
                let rest = fx.b.ins().fsub(three, twice);
                fx.b.ins().fmul(tt, rest)
            }),
            Intrinsic::Reflect => {
                let d = self.dot(&w[1], &w[0]);
                let two = self.f32c(2.0);
                let scale = self.b.ins().fmul(two, d);
                self.zip2(&w[0], &w[1], |fx, i, n| {
                    let scaled = fx.b.ins().fmul(scale, n);
                    fx.b.ins().fsub(i, scaled)
                })
            }
            Intrinsic::Atan2 => {
                let call = self.b.ins().call(self.atan2f, &[w[0][0], w[1][0]]);
                vec![self.b.inst_results(call)[0]]
            }
        }
    }
}
