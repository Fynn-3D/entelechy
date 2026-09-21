//! Orla IR to Cranelift.
//!
//! Every IR value becomes its words, one Cranelift value each, `f32` or
//! `i32` (a `Bool` is an `i32` of 0 or 1), in the layout the ABI gives
//! elements, so a struct is its fields in order and a vector its
//! components. `if` and `loop` become blocks with block parameters; a
//! call passes the words; a map's entry takes the context, the element's
//! coordinates and a pointer for its output words; a scan's entry takes
//! the context, the step's index, a pointer to the carried words and a
//! pointer for the next, and its `init` writes the first carried words;
//! a reduction's `merge` takes its two values through pointers, so the
//! host can call it whatever their words. Memories are written through
//! a helper. Integer arithmetic is
//! checked as the interpreter checks it, the first failure recorded in
//! the context. Field reads and the transcendental functions call the
//! helpers in the crate root, by symbol, so they round as the CPU does.

use std::collections::HashMap;
use std::sync::Arc;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    AbiParam, BlockArg, BlockCall, FuncRef, InstBuilder, JumpTableData, MemFlags, Signature,
    StackSlot, StackSlotData, StackSlotKind, Value, types,
};
use cranelift_codegen::isa::TargetIsa;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, FuncId, Linkage, Module as _};
use orla_abi::Type;
use orla_ir::{BinaryOp, Function, Instruction, Intrinsic, Module, Op, UnaryOp, Validated};
use orla_rules::matrix::{self, Arith};

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
/// address, a scan's `init`, and a reduction's `merge`.
pub struct Code {
    pub module: JITModule,
    pub entry: usize,
    pub init: Option<usize>,
    pub merge: Option<usize>,
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
    /// A reduction's `merge`: the context, pointers to the two values'
    /// words, the output slot.
    Merge,
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
            (Intrinsic::Exp2, "orla_jit_exp2f"),
            (Intrinsic::Log2, "orla_jit_log2f"),
            (Intrinsic::Sinh, "orla_jit_sinhf"),
            (Intrinsic::Cosh, "orla_jit_coshf"),
            (Intrinsic::Tanh, "orla_jit_tanhf"),
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
    let merge_signature = sig(&[ptr, ptr, ptr, ptr], &[]);
    let merge_id = match &module.merge {
        Some(_) if module.manifest.mode == orla_ir::Mode::Reduce => Some(
            jit.declare_function("merge", Linkage::Local, &merge_signature)
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
    if let (Some(merge_id), Some(merge)) = (merge_id, &module.merge) {
        context.func.signature = merge_signature;
        {
            let builder = FunctionBuilder::new(&mut context.func, &mut fbc);
            Fx::new(
                module, &mut jit, builder, &imports, &functions, ptr, max_stride,
            )
            .function(merge, Kind::Merge);
        }
        jit.define_function(merge_id, &mut context)
            .map_err(|error| format!("merge: {error}"))?;
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
    let merge = merge_id.map(|id| jit.get_finalized_function(id) as usize);
    Ok(Code {
        module: jit,
        entry,
        init,
        merge,
    })
}

/// The matrix formulas' arithmetic, as instructions.
struct Ops<'a, 'b>(&'a mut FunctionBuilder<'b>);

impl Arith for Ops<'_, '_> {
    type V = Value;
    fn add(&mut self, a: Value, b: Value) -> Value {
        self.0.ins().fadd(a, b)
    }
    fn sub(&mut self, a: Value, b: Value) -> Value {
        self.0.ins().fsub(a, b)
    }
    fn mul(&mut self, a: Value, b: Value) -> Value {
        self.0.ins().fmul(a, b)
    }
    fn div(&mut self, a: Value, b: Value) -> Value {
        self.0.ins().fdiv(a, b)
    }
    fn neg(&mut self, a: Value) -> Value {
        self.0.ins().fneg(a)
    }
}

/// The Cranelift type of each word of a value of `ty`.
fn word_types(module: &Module, ty: Type) -> Vec<types::Type> {
    match ty {
        Type::Float => vec![types::F32],
        Type::Int | Type::Bool => vec![types::I32],
        Type::Vec2 => vec![types::F32; 2],
        Type::Vec3 => vec![types::F32; 3],
        Type::Vec4 => vec![types::F32; 4],
        Type::IVec2 => vec![types::I32; 2],
        Type::IVec3 => vec![types::I32; 3],
        Type::Mat2 => vec![types::F32; 4],
        Type::Mat3 => vec![types::F32; 9],
        Type::Mat4 => vec![types::F32; 16],
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
                Op::Match {
                    arms, otherwise, ..
                } => arms
                    .iter()
                    .find_map(|(_, block)| find(&block.body, id))
                    .or_else(|| find(&otherwise.body, id)),
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
    jit: &'a mut JITModule,
    b: FunctionBuilder<'b>,
    ptr: types::Type,
    /// The function's name, which names the tables it defines.
    name: String,
    /// The words of every value known when the Kernel is compiled, by
    /// id: literals, and vectors, structs and arrays built from them.
    constants: HashMap<u32, Vec<u32>>,
    /// Values that live in the module's read-only data, by id: the
    /// address of their words. An array picked out of a constant table
    /// is one, so a table of tables is indexed without a copy.
    pointers: HashMap<u32, Value>,
    /// The tables defined so far, by the id of the array they hold.
    tables: HashMap<u32, cranelift_codegen::ir::GlobalValue>,
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
        jit: &'a mut JITModule,
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
            jit,
            b,
            ptr,
            name: String::new(),
            constants: HashMap::new(),
            pointers: HashMap::new(),
            tables: HashMap::new(),
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
        self.name = function.name.clone();
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
            Kind::Merge => {
                // Each argument's words are loaded from its pointer.
                for (arg, pointer) in function.args.iter().zip([params[1], params[2]]) {
                    let words = word_types(self.module, *arg)
                        .iter()
                        .enumerate()
                        .map(|(k, t)| self.b.ins().load(*t, self.flags, pointer, (k * 4) as i32))
                        .collect();
                    self.args.push(words);
                }
                params[3]
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
            self.note_constant(instruction);
        }
    }

    /// Records a value's words when they are known at compile time: a
    /// literal, or a construction or extraction of known words.
    fn note_constant(&mut self, instruction: &Instruction) {
        use orla_abi::Value as V;
        let id = instruction.id.0;
        let words = match &instruction.op {
            Op::Const(value) => Some(match value {
                V::Float(v) => vec![v.to_bits()],
                V::Int(v) => vec![*v as u32],
                V::Bool(v) => vec![u32::from(*v)],
                V::Vec2(v) => v.iter().map(|c| c.to_bits()).collect(),
                V::Vec3(v) => v.iter().map(|c| c.to_bits()).collect(),
                V::Vec4(v) => v.iter().map(|c| c.to_bits()).collect(),
            }),
            // A float vector from Int literals widens them; the words
            // are then not the parts' words, so only all-float parts count.
            Op::Construct(parts)
                if instruction.ty.components().is_none()
                    || parts.iter().all(|part| self.ty(part) == Type::Float) =>
            {
                parts
                    .iter()
                    .map(|part| self.constants.get(&part.0).cloned())
                    .collect::<Option<Vec<_>>>()
                    .map(|parts| parts.concat())
            }
            Op::Extract(source, component) => self.constants.get(&source.0).and_then(|words| {
                let (start, n) = match self.ty(source) {
                    Type::Struct(index) => (
                        field_offset(self.module, index, *component as usize),
                        word_types(self.module, instruction.ty).len(),
                    ),
                    other if other.order().is_some() => {
                        let n = usize::from(other.order().unwrap_or(1));
                        (*component as usize * n, n)
                    }
                    _ => (*component as usize, 1),
                };
                words.get(start..start + n).map(<[u32]>::to_vec)
            }),
            _ => None,
        };
        if let Some(words) = words {
            self.constants.insert(id, words);
        }
    }

    /// The address of a constant array's words in the module's read-only
    /// data, defined once per array here, named after the function and
    /// the value.
    fn table(&mut self, id: u32) -> Value {
        let global = match self.tables.get(&id) {
            Some(global) => *global,
            None => {
                let words = self.constants.get(&id).cloned().unwrap_or_default();
                let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                let name = format!("{}.{id}", self.name);
                let data = self
                    .jit
                    .declare_data(&name, Linkage::Local, false, false)
                    .expect("a table's name is its own");
                let mut description = DataDescription::new();
                description.define(bytes.into_boxed_slice());
                self.jit
                    .define_data(data, &description)
                    .expect("a table is defined once");
                let global = self.jit.declare_data_in_func(data, self.b.func);
                self.tables.insert(id, global);
                global
            }
        };
        self.b.ins().global_value(self.ptr, global)
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
        let id = instruction.id;
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
                if self.ty(a) == Type::Int || self.ty(a).int_components().is_some() {
                    words
                        .iter()
                        .map(|w| {
                            let zero = self.i32c(0);
                            let (v, overflow) = self.b.ins().ssub_overflow(zero, *w);
                            self.fail_if(overflow, OVERFLOW);
                            v
                        })
                        .collect()
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
                if matches!(ty, Type::Struct(_))
                    || ty.order().is_some()
                    || ty.int_components().is_some()
                {
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
                    other if other.order().is_some() => {
                        let n = usize::from(other.order().unwrap_or(1));
                        let start = *component as usize * n;
                        words[start..start + n].to_vec()
                    }
                    _ => vec![words[*component as usize]],
                }
            }
            Op::Subscript(array, index) => {
                // The element is loaded at the clamped index times its
                // width from the array's words: in the module's read-only
                // data when the array is known at compile time (a table),
                // where a table's inner array already lives, or else in a
                // slot of this function's own, stored first.
                let count = match self.ty(array) {
                    Type::Struct(struct_index) => {
                        self.module.structs[struct_index as usize].fields.len()
                    }
                    other => other
                        .order()
                        .or(other.lanes().map(|(n, _)| n))
                        .map_or(1, usize::from),
                };
                let tys = word_types(self.module, ty);
                let n = tys.len();
                let (base, flags) = if let Some(pointer) = self.pointers.get(&array.0) {
                    (*pointer, self.flags.with_readonly())
                } else if self.constants.contains_key(&array.0)
                    && matches!(self.ty(array), Type::Struct(_))
                {
                    (self.table(array.0), self.flags.with_readonly())
                } else {
                    let words = self.words(array);
                    let slot = self.b.create_sized_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot,
                        (words.len().max(1) * 4) as u32,
                        2,
                    ));
                    for (k, word) in words.iter().enumerate() {
                        self.b.ins().stack_store(*word, slot, (k * 4) as i32);
                    }
                    (self.b.ins().stack_addr(self.ptr, slot, 0), self.flags)
                };
                let raw = self.words(index)[0];
                let zero = self.i32c(0);
                let last = self.i32c(count as i64 - 1);
                let low = self.b.ins().smax(raw, zero);
                let clamped = self.b.ins().smin(low, last);
                let scaled = self.b.ins().imul_imm(clamped, (n * 4) as i64);
                let offset = self.b.ins().uextend(self.ptr, scaled);
                let address = self.b.ins().iadd(base, offset);
                if flags.readonly()
                    && matches!(ty, Type::Struct(index) if self.module.structs[index as usize].is_array())
                {
                    self.pointers.insert(id.0, address);
                }
                tys.iter()
                    .enumerate()
                    .map(|(k, t)| self.b.ins().load(*t, flags, address, (k * 4) as i32))
                    .collect()
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
            Op::Match {
                scrutinee,
                arms,
                otherwise,
            } => {
                // A dense table when the values are close, else a chain of
                // compares; either way each arm is a block jumping to one
                // join with the result's words.
                let value = self.words(scrutinee)[0];
                let join = self.b.create_block();
                for t in &word_types(self.module, ty) {
                    self.b.append_block_param(join, *t);
                }
                let arm_blocks: Vec<_> = arms.iter().map(|_| self.b.create_block()).collect();
                let default_block = self.b.create_block();
                let all: Vec<(i32, usize)> = arms
                    .iter()
                    .enumerate()
                    .flat_map(|(index, (values, _))| values.iter().map(move |v| (*v, index)))
                    .collect();
                let bounds = all.iter().fold(None, |bounds: Option<(i32, i32)>, (v, _)| {
                    Some(bounds.map_or((*v, *v), |(lo, hi)| (lo.min(*v), hi.max(*v))))
                });
                let dense = bounds.is_some_and(|(lo, hi)| {
                    (hi as i64 - lo as i64 + 1) as usize <= all.len() * 4 + 16
                });
                if let (true, Some((lo, hi))) = (dense, bounds) {
                    let span = (hi as i64 - lo as i64 + 1) as usize;
                    let mut targets = vec![default_block; span];
                    for (v, index) in &all {
                        targets[(*v as i64 - lo as i64) as usize] = arm_blocks[*index];
                    }
                    let none: [BlockArg; 0] = [];
                    let default_call = self.b.func.dfg.block_call(default_block, &none);
                    let calls: Vec<BlockCall> = targets
                        .iter()
                        .map(|block| self.b.func.dfg.block_call(*block, &none))
                        .collect();
                    let table = self
                        .b
                        .create_jump_table(JumpTableData::new(default_call, &calls));
                    // The index is taken unsigned, so a value below the
                    // lowest wraps past the table and lands on the default.
                    let index = if lo == 0 {
                        value
                    } else {
                        self.b.ins().iadd_imm(value, -(lo as i64))
                    };
                    self.b.ins().br_table(index, table);
                } else {
                    for (v, index) in &all {
                        let next = self.b.create_block();
                        let hit = self.b.ins().icmp_imm(IntCC::Equal, value, *v as i64);
                        self.b.ins().brif(hit, arm_blocks[*index], &[], next, &[]);
                        self.b.switch_to_block(next);
                        self.b.seal_block(next);
                    }
                    self.b.ins().jump(default_block, &[]);
                }
                for (block, (_, arm)) in arm_blocks.iter().zip(arms) {
                    self.b.switch_to_block(*block);
                    self.b.seal_block(*block);
                    self.block(&arm.body);
                    let words = self.words(&arm.result);
                    let args: Vec<_> = words.iter().map(|v| (*v).into()).collect();
                    self.b.ins().jump(join, &args);
                }
                self.b.switch_to_block(default_block);
                self.b.seal_block(default_block);
                self.block(&otherwise.body);
                let words = self.words(&otherwise.result);
                let args: Vec<_> = words.iter().map(|v| (*v).into()).collect();
                self.b.ins().jump(join, &args);
                self.b.switch_to_block(join);
                self.b.seal_block(join);
                self.b.block_params(join).to_vec()
            }
            Op::Loop {
                count,
                bound,
                init,
                until,
                body,
            } => {
                // The run-time count, clamped to 0..=count, read before
                // the loop; the literal bound otherwise.
                let limit = match bound {
                    Some(bound) => {
                        let given = self.words(bound)[0];
                        let zero = self.i32c(0);
                        let most = self.i32c(*count as i64);
                        let floored = self.b.ins().smax(given, zero);
                        self.b.ins().smin(floored, most)
                    }
                    None => self.i32c(*count as i64),
                };
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
                let carried_args_from_init: Vec<_> = init_args.clone();
                init_args.push(zero.into());
                match bound {
                    // A run-time count may be zero: then the loop yields
                    // what it carries, as the interpreter's does.
                    Some(_) => {
                        let runs = self.b.ins().icmp(IntCC::SignedGreaterThan, limit, zero);
                        self.b
                            .ins()
                            .brif(runs, header, &init_args, exit, &carried_args_from_init);
                    }
                    None => {
                        self.b.ins().jump(header, &init_args);
                    }
                }
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
        let int_like =
            ta == Type::Int || ta.int_components().is_some() || tb.int_components().is_some();
        if op == BinaryOp::Mul && (ta.order().is_some() || tb.order().is_some()) {
            let mut ops = Ops(&mut self.b);
            return match (ta.order(), tb.order()) {
                (Some(n), Some(_)) => matrix::mul_mat_mat(&mut ops, usize::from(n), &wa, &wb),
                (Some(n), None) if tb != Type::Float => {
                    matrix::mul_mat_vec(&mut ops, usize::from(n), &wa, &wb)
                }
                (None, Some(n)) if ta != Type::Float => {
                    matrix::mul_vec_mat(&mut ops, usize::from(n), &wa, &wb)
                }
                _ => self.zip2(&wa, &wb, |fx, x, y| fx.b.ins().fmul(x, y)),
            };
        }
        match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul if int_like => {
                self.zip2(&wa, &wb, |fx, x, y| {
                    let (v, overflow) = match op {
                        BinaryOp::Add => fx.b.ins().sadd_overflow(x, y),
                        BinaryOp::Sub => fx.b.ins().ssub_overflow(x, y),
                        _ => fx.b.ins().smul_overflow(x, y),
                    };
                    fx.fail_if(overflow, OVERFLOW);
                    v
                })
            }
            BinaryOp::Div | BinaryOp::Rem if int_like => {
                // A zero divisor and MIN / -1 are the two failures; both
                // would trap, so the division runs on a safe divisor instead.
                self.zip2(&wa, &wb, |fx, x, y| {
                    let zero = fx.b.ins().icmp_imm(IntCC::Equal, y, 0);
                    fx.fail_if(zero, DIVISION_BY_ZERO);
                    let is_min = fx.b.ins().icmp_imm(IntCC::Equal, x, i32::MIN as i64);
                    let minus_one = fx.b.ins().icmp_imm(IntCC::Equal, y, -1);
                    let overflow = fx.b.ins().band(is_min, minus_one);
                    fx.fail_if(overflow, OVERFLOW);
                    let unsafe_ = fx.b.ins().bor(zero, overflow);
                    let one = fx.i32c(1);
                    let divisor = fx.b.ins().select(unsafe_, one, y);
                    if op == BinaryOp::Div {
                        fx.b.ins().sdiv(x, divisor)
                    } else {
                        fx.b.ins().srem(x, divisor)
                    }
                })
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
                    let same = if (matches!(ta, Type::Int | Type::Bool)
                        || ta.int_components().is_some())
                        && tb == ta
                    {
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
            BinaryOp::BitAnd => self.zip2(&wa, &wb, |fx, x, y| fx.b.ins().band(x, y)),
            BinaryOp::BitOr => self.zip2(&wa, &wb, |fx, x, y| fx.b.ins().bor(x, y)),
            BinaryOp::BitXor => self.zip2(&wa, &wb, |fx, x, y| fx.b.ins().bxor(x, y)),
            BinaryOp::Shl | BinaryOp::Shr | BinaryOp::Ushr => self.zip2(&wa, &wb, |fx, x, y| {
                let count = fx.b.ins().band_imm(y, 31);
                match op {
                    BinaryOp::Shl => fx.b.ins().ishl(x, count),
                    BinaryOp::Shr => fx.b.ins().sshr(x, count),
                    _ => fx.b.ins().ushr(x, count),
                }
            }),
        }
    }

    fn intrinsic(&mut self, intrinsic: Intrinsic, args: &[orla_ir::ValueId]) -> Vec<Value> {
        let w: Vec<Vec<Value>> = args.iter().map(|a| self.words(a)).collect();
        let is_int = |fx: &Self, k: usize| {
            fx.ty(&args[k]) == Type::Int || fx.ty(&args[k]).int_components().is_some()
        };
        let map1 = |fx: &mut Self, f: &mut dyn FnMut(&mut Self, Value) -> Value| -> Vec<Value> {
            w[0].iter().map(|v| f(fx, *v)).collect()
        };
        match intrinsic {
            Intrinsic::Transpose => {
                let n = usize::from(self.ty(&args[0]).order().unwrap_or(1));
                matrix::transpose(n, &w[0])
            }
            Intrinsic::Determinant => {
                let n = usize::from(self.ty(&args[0]).order().unwrap_or(1));
                vec![matrix::determinant(&mut Ops(&mut self.b), n, &w[0])]
            }
            Intrinsic::Inverse => {
                let n = usize::from(self.ty(&args[0]).order().unwrap_or(1));
                matrix::inverse(&mut Ops(&mut self.b), n, &w[0])
            }
            Intrinsic::Floor => map1(self, &mut |fx, v| fx.b.ins().floor(v)),
            Intrinsic::Ceil => map1(self, &mut |fx, v| fx.b.ins().ceil(v)),
            Intrinsic::Round => map1(self, &mut |fx, v| fx.b.ins().nearest(v)),
            Intrinsic::Sqrt => map1(self, &mut |fx, v| fx.b.ins().sqrt(v)),
            Intrinsic::Fract => map1(self, &mut |fx, v| {
                let f = fx.b.ins().floor(v);
                fx.b.ins().fsub(v, f)
            }),
            Intrinsic::Abs if is_int(self, 0) => map1(self, &mut |fx, v| fx.b.ins().iabs(v)),
            Intrinsic::Abs => map1(self, &mut |fx, v| fx.b.ins().fabs(v)),
            Intrinsic::Sign if is_int(self, 0) => map1(self, &mut |fx, v| {
                let pos = fx.b.ins().icmp_imm(IntCC::SignedGreaterThan, v, 0);
                let neg = fx.b.ins().icmp_imm(IntCC::SignedLessThan, v, 0);
                let one = fx.i32c(1);
                let minus = fx.i32c(-1);
                let zero = fx.i32c(0);
                let low = fx.b.ins().select(neg, minus, zero);
                fx.b.ins().select(pos, one, low)
            }),
            Intrinsic::Sign => map1(self, &mut |fx, v| {
                let zero = fx.f32c(0.0);
                let pos = fx.b.ins().fcmp(FloatCC::GreaterThan, v, zero);
                let neg = fx.b.ins().fcmp(FloatCC::LessThan, v, zero);
                let one = fx.f32c(1.0);
                let minus = fx.f32c(-1.0);
                let low = fx.b.ins().select(neg, minus, zero);
                fx.b.ins().select(pos, one, low)
            }),
            Intrinsic::Trunc => map1(self, &mut |fx, v| fx.b.ins().trunc(v)),
            Intrinsic::Fma => self.zip3(&w[0], &w[1], &w[2], |fx, a, b, c| fx.b.ins().fma(a, b, c)),
            Intrinsic::Popcount => vec![self.b.ins().popcnt(w[0][0])],
            Intrinsic::Clz => vec![self.b.ins().clz(w[0][0])],
            Intrinsic::Ctz => vec![self.b.ins().ctz(w[0][0])],
            Intrinsic::Rotl => {
                let count = self.b.ins().band_imm(w[1][0], 31);
                vec![self.b.ins().rotl(w[0][0], count)]
            }
            Intrinsic::Rotr => {
                let count = self.b.ins().band_imm(w[1][0], 31);
                vec![self.b.ins().rotr(w[0][0], count)]
            }
            Intrinsic::Ult => {
                let below = self.b.ins().icmp(IntCC::UnsignedLessThan, w[0][0], w[1][0]);
                vec![self.word_of(below)]
            }
            Intrinsic::Udiv | Intrinsic::Urem => {
                let zero = self.b.ins().icmp_imm(IntCC::Equal, w[1][0], 0);
                self.fail_if(zero, DIVISION_BY_ZERO);
                // A zero divisor is reported above; the division itself
                // must not trap, so it divides by one then.
                let one = self.i32c(1);
                let divisor = self.b.ins().select(zero, one, w[1][0]);
                if intrinsic == Intrinsic::Udiv {
                    vec![self.b.ins().udiv(w[0][0], divisor)]
                } else {
                    vec![self.b.ins().urem(w[0][0], divisor)]
                }
            }
            Intrinsic::Mulhi => vec![self.b.ins().smulhi(w[0][0], w[1][0])],
            Intrinsic::Umulhi => vec![self.b.ins().umulhi(w[0][0], w[1][0])],
            Intrinsic::Sin
            | Intrinsic::Cos
            | Intrinsic::Tan
            | Intrinsic::Asin
            | Intrinsic::Acos
            | Intrinsic::Atan
            | Intrinsic::Exp
            | Intrinsic::Log
            | Intrinsic::Exp2
            | Intrinsic::Log2
            | Intrinsic::Sinh
            | Intrinsic::Cosh
            | Intrinsic::Tanh => {
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
            Intrinsic::Min if is_int(self, 0) => {
                self.zip2(&w[0], &w[1], |fx, x, y| fx.b.ins().smin(x, y))
            }
            Intrinsic::Max if is_int(self, 0) => {
                self.zip2(&w[0], &w[1], |fx, x, y| fx.b.ins().smax(x, y))
            }
            Intrinsic::AddWrap => vec![self.b.ins().iadd(w[0][0], w[1][0])],
            Intrinsic::SubWrap => vec![self.b.ins().isub(w[0][0], w[1][0])],
            Intrinsic::MulWrap => vec![self.b.ins().imul(w[0][0], w[1][0])],
            Intrinsic::Min => self.zip2(&w[0], &w[1], |fx, x, y| fx.fmin(x, y)),
            Intrinsic::Max => self.zip2(&w[0], &w[1], |fx, x, y| fx.fmax(x, y)),
            Intrinsic::Clamp if is_int(self, 0) => {
                self.zip3(&w[0], &w[1], &w[2], |fx, v, low, high| {
                    let floored = fx.b.ins().smax(v, low);
                    fx.b.ins().smin(floored, high)
                })
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
            Intrinsic::ToFloat => map1(self, &mut |fx, v| fx.b.ins().fcvt_from_sint(types::F32, v)),
            Intrinsic::ToInt => {
                if matches!(self.ty(&args[0]), Type::Int | Type::Bool) {
                    vec![w[0][0]]
                } else {
                    map1(self, &mut |fx, v| {
                        fx.b.ins().fcvt_to_sint_sat(types::I32, v)
                    })
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
