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
    AbiParam, Block, BlockArg, BlockCall, FuncRef, InstBuilder, JumpTableData, MemFlags, Signature,
    StackSlot, StackSlotData, StackSlotKind, Value, types,
};
use cranelift_codegen::isa::TargetIsa;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, FuncId, Linkage, Module as _};
use orla_abi::{Encoding, Scalar, Type, encoding};
use orla_ir::{BinaryOp, Function, Instruction, Intrinsic, Module, Op, UnaryOp, Validated};

use crate::offsets;

/// Error codes stored in the context, as the CPU backend reports them.
const DIVISION_BY_ZERO: i64 = 1;
const OVERFLOW: i64 = 2;

/// The imported helpers, by function id.
struct Imports {
    read: FuncId,
    linear: FuncId,
    key: FuncId,
    button: FuncId,
    put: FuncId,
    put_rows: FuncId,
    unary: Vec<(Intrinsic, FuncId)>,
    powf: FuncId,
    atan2f: FuncId,
    fmodf: FuncId,
    half: FuncId,
    /// The same math on doubles.
    unary64: Vec<(Intrinsic, FuncId)>,
    pow: FuncId,
    atan2: FuncId,
    fmod: FuncId,
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
    /// A map's entry: the context, the run's first `x`, its `y` and
    /// `z`, how many elements it holds, and where their words go.
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
    // A port keeps every scalar one way, which is what a port type can
    // say; the code is generated for that one encoding.
    for port in module
        .manifest
        .inputs
        .iter()
        .chain(&module.manifest.outputs)
    {
        if port.ty.schema.uniform_encoding().is_none() && !port.ty.schema.is_wide() {
            return Err(format!(
                "`{}` keeps its scalars in more than one encoding ({}); a port keeps them one way",
                port.name, port.ty.schema
            ));
        }
    }
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
    let key_sig = sig(&[ptr, i], &[i]);
    let put_sig = sig(&[ptr, i, f, f, f, ptr], &[]);
    let put_rows_sig = sig(&[ptr, i, i, ptr], &[]);
    let unary_sig = sig(&[f], &[f]);
    let binary_sig = sig(&[f, f], &[f]);
    let unary64_sig = sig(&[types::F64], &[types::F64]);
    let binary64_sig = sig(&[types::F64, types::F64], &[types::F64]);
    let imports = Imports {
        read: declare(&mut jit, "orla_jit_read", &read_sig)?,
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
        half: declare(&mut jit, "orla_jit_half", &sig(&[i], &[i]))?,
        unary64: [
            (Intrinsic::Sin, "orla_jit_sin"),
            (Intrinsic::Cos, "orla_jit_cos"),
            (Intrinsic::Tan, "orla_jit_tan"),
            (Intrinsic::Asin, "orla_jit_asin"),
            (Intrinsic::Acos, "orla_jit_acos"),
            (Intrinsic::Atan, "orla_jit_atan"),
            (Intrinsic::Exp, "orla_jit_exp"),
            (Intrinsic::Log, "orla_jit_log"),
            (Intrinsic::Exp2, "orla_jit_exp2"),
            (Intrinsic::Log2, "orla_jit_log2"),
            (Intrinsic::Sinh, "orla_jit_sinh"),
            (Intrinsic::Cosh, "orla_jit_cosh"),
            (Intrinsic::Tanh, "orla_jit_tanh"),
        ]
        .into_iter()
        .map(|(intrinsic, name)| declare(&mut jit, name, &unary64_sig).map(|id| (intrinsic, id)))
        .collect::<Result<Vec<_>, _>>()?,
        pow: declare(&mut jit, "orla_jit_pow", &binary64_sig)?,
        atan2: declare(&mut jit, "orla_jit_atan2", &binary64_sig)?,
        fmod: declare(&mut jit, "orla_jit_fmod", &binary64_sig)?,
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
        // A map's entry walks a run of elements itself: the context, the
        // first `x`, the `y` and `z` they share, how many there are, and
        // where their words go.
        sig(&[ptr, i, i, i, i, ptr], &[])
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
        .map(|port| port.ty.schema.value_words())
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

/// The Cranelift type of each scalar of a value of `ty`, in layout
/// order: a 32-bit scalar is one word, and a `Double` or a `Long` one
/// 64-bit value that memory keeps as two words, the low one first
/// (little-endian, as every target the JIT builds for is).
fn word_types(module: &Module, ty: Type) -> Vec<types::Type> {
    match ty {
        Type::Struct(index) => module.structs[index as usize]
            .fields
            .iter()
            .flat_map(|field| word_types(module, field.ty))
            .collect(),
        other => {
            let (scalar, n) = other.shape().unwrap_or((Type::Float, 1));
            let lane = match scalar {
                Type::Float => types::F32,
                Type::Double => types::F64,
                Type::Long => types::I64,
                _ => types::I32,
            };
            vec![lane; usize::from(n)]
        }
    }
}

/// Where each scalar of `tys` begins in memory, in bytes.
fn byte_offsets(tys: &[types::Type]) -> Vec<i32> {
    let mut at = 0;
    tys.iter()
        .map(|t| {
            let here = at;
            at += t.bytes() as i32;
            here
        })
        .collect()
}

/// Bytes the scalars of `tys` take in memory.
fn byte_size(tys: &[types::Type]) -> usize {
    tys.iter().map(|t| t.bytes() as usize).sum()
}

/// Whether a type holds a 64-bit scalar anywhere.
fn holds_wide(module: &Module, ty: Type) -> bool {
    word_types(module, ty).iter().any(|t| t.bits() == 64)
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

/// An input's view as the run sees it: the parts that do not change
/// while the run walks its elements.
#[derive(Clone, Copy)]
struct ViewOnce {
    /// The first word of the Field.
    words: Value,
    /// Bytes an element takes.
    element: Value,
    /// The largest `x` the Field has.
    last_x: Value,
    /// The first element of the row this run is on, clamped into the
    /// Field as `sample` clamps.
    row: Value,
    /// The last element there is, for an address over rows.
    last: Value,
    /// Where this element's words are, carried from element to element
    /// rather than worked out again: `sample` walks the row one element
    /// at a time and stops at its end, which is the clamp.
    at: Option<Value>,
    /// The last element of the row, as an address: where `at` stops.
    end: Value,
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
    /// Where an array's words were put down to be indexed, and the
    /// block that put them there: a second index of the same array in
    /// that block reads what is already stored.
    spilled: HashMap<u32, (Block, Value)>,
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
    linear: FuncRef,
    key: FuncRef,
    button: FuncRef,
    put: FuncRef,
    put_rows: FuncRef,
    unary: Vec<(Intrinsic, FuncRef)>,
    powf: FuncRef,
    atan2f: FuncRef,
    fmodf: FuncRef,
    /// `orla_jit_half`, which widens a half float.
    half: FuncRef,
    unary64: Vec<(Intrinsic, FuncRef)>,
    pow: FuncRef,
    atan2: FuncRef,
    fmod: FuncRef,
    /// What every input's view says, read once before a map's run
    /// rather than at every element: the words, the bytes an element
    /// takes, the last `x` there is, and where the run's own row
    /// starts. A read is then one clamp, one multiply and one add.
    settled_views: Vec<Option<ViewOnce>>,
    scratch: StackSlot,
    flags: MemFlags,
    /// What the context and the views hold, which nothing writes while
    /// an execution runs: marked read-only so the loads a loop repeats
    /// are hoisted out of it rather than done again every element.
    settled: MemFlags,
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
        let half = import(jit, &mut b, imports.half);
        let unary64 = imports
            .unary64
            .iter()
            .map(|(intrinsic, id)| (*intrinsic, import(jit, &mut b, *id)))
            .collect();
        let pow = import(jit, &mut b, imports.pow);
        let atan2 = import(jit, &mut b, imports.atan2);
        let fmod = import(jit, &mut b, imports.fmod);
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
            spilled: HashMap::new(),
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
            linear,
            key,
            button,
            put,
            put_rows,
            unary,
            powf,
            atan2f,
            fmodf,
            half,
            unary64,
            pow,
            atan2,
            fmod,
            settled_views: Vec::new(),
            scratch,
            flags: MemFlags::trusted(),
            settled: MemFlags::trusted().with_readonly(),
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
        if kind == Kind::Map {
            return self.map_run(function, &params);
        }
        let out = match kind {
            Kind::Map => unreachable!("a map's entry walks its run"),
            Kind::Scan => {
                // The step's coordinates are its index, as the interpreter
                // has them; the carried words are loaded once.
                let zero = self.i32c(0);
                self.coords = Some((params[1], zero, zero, params[3]));
                let carried_types = word_types(self.module, result_type(function));
                let carried = self.loads(&carried_types, self.flags, params[2], 0);
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
                    let tys = word_types(self.module, *arg);
                    let words = self.loads(&tys, self.flags, pointer, 0);
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
                offset += self.b.func.dfg.value_type(word).bytes() as i32;
            }
        }
        self.b.ins().return_(&[]);
        self.b.finalize();
    }

    /// A map's entry: the body under a loop over the run of elements it
    /// was given, so the reads the body does not repeat are hoisted out
    /// of it and a million elements is one call rather than a million.
    fn map_run(mut self, function: &Function, params: &[Value]) {
        let (x0, y, z, count, out0) = (params[1], params[2], params[3], params[4], params[5]);
        self.settle_views(y, z);
        let header = self.b.create_block();
        let body_block = self.b.create_block();
        let failed = self.b.create_block();
        let exit = self.b.create_block();
        self.b.append_block_param(header, types::I32);
        self.b.append_block_param(header, self.ptr);
        self.b.append_block_param(failed, types::I32);
        // Where each input's first element is, for the run's first `x`.
        let mut first: Vec<BlockArg> = vec![self.i32c(0).into(), out0.into()];
        let settled = self.settled_views.clone();
        for view in settled.iter().flatten() {
            self.b.append_block_param(header, self.ptr);
            let cx = self.b.ins().umin(x0, view.last_x);
            let index = self.b.ins().iadd(view.row, cx);
            let bytes = self.b.ins().imul(index, view.element);
            let offset = self.b.ins().uextend(self.ptr, bytes);
            let at = self.b.ins().iadd(view.words, offset);
            first.push(at.into());
        }
        self.b.ins().jump(header, &first);
        self.b.switch_to_block(header);
        let header_params = self.b.block_params(header).to_vec();
        let taken = header_params[0];
        let out = header_params[1];
        for (slot, at) in self
            .settled_views
            .iter_mut()
            .flatten()
            .zip(header_params[2..].iter())
        {
            slot.at = Some(*at);
        }
        let more = self.b.ins().icmp(IntCC::UnsignedLessThan, taken, count);
        self.b.ins().brif(more, body_block, &[], exit, &[]);
        self.b.switch_to_block(body_block);
        self.b.seal_block(body_block);
        let x = self.b.ins().iadd(x0, taken);
        self.coords = Some((x, y, z, out));
        self.block(&function.body);
        let mut offset = 0i32;
        for result in &function.results {
            for word in self.values[result.0 as usize].clone() {
                self.b.ins().store(self.flags, word, out, offset);
                offset += self.b.func.dfg.value_type(word).bytes() as i32;
            }
        }
        // A failure stops the run where it happened, so the element the
        // caller names is the element that failed.
        let error = self
            .b
            .ins()
            .load(types::I32, self.flags, self.ctx, offsets::ERROR);
        let stop = self.b.ins().icmp_imm(IntCC::NotEqual, error, 0);
        let next = self.b.ins().iadd_imm(taken, 1);
        // What the stores just wrote is what one element takes.
        let step = self.b.ins().iadd_imm(out, i64::from(offset));
        // Every input walks on one element, and stops at the end of its
        // row, which is what clamping an `x` past the edge comes to.
        let views = self.settled_views.clone();
        let mut again: Vec<BlockArg> = vec![next.into(), step.into()];
        for view in views.iter().flatten() {
            let at = view.at.expect("the run carries every input's place");
            let element = self.b.ins().uextend(self.ptr, view.element);
            let moved = self.b.ins().iadd(at, element);
            let held = self.b.ins().umin(moved, view.end);
            again.push(held.into());
        }
        self.b.ins().brif(stop, failed, &[x.into()], header, &again);
        self.b.seal_block(header);
        self.b.switch_to_block(failed);
        self.b.seal_block(failed);
        let at = self.b.block_params(failed)[0];
        self.b
            .ins()
            .store(self.flags, at, self.ctx, offsets::ERROR_AT);
        self.b.ins().jump(exit, &[]);
        self.b.switch_to_block(exit);
        self.b.seal_block(exit);
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
        let id = instruction.id.0;
        // What a table keeps is words, and a 64-bit scalar's lane is not
        // one: such a value is built in registers instead.
        if holds_wide(self.module, instruction.ty) || instruction_reads_wide(self, &instruction.op)
        {
            return;
        }
        let words = match &instruction.op {
            Op::Const(value) => Some(value.words()),
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

    /// `float`, `int`, `double` or `long` where a 64-bit kind is on
    /// either side, componentwise.
    fn conversion(
        &mut self,
        intrinsic: Intrinsic,
        args: &[orla_ir::ValueId],
        words: &[Value],
    ) -> Vec<Value> {
        let from = self.ty(&args[0]).scalar().unwrap_or(Type::Float);
        let to = match intrinsic {
            Intrinsic::ToFloat => types::F32,
            Intrinsic::ToInt => types::I32,
            Intrinsic::ToDouble => types::F64,
            _ => types::I64,
        };
        words.iter().map(|v| self.convert(*v, from, to)).collect()
    }

    /// A scalar of kind `from` as one of Cranelift type `to`: a float
    /// truncates toward zero and saturates (a NaN is zero), a long wraps
    /// to an int, and a `Bool` is 0 or 1.
    fn convert(&mut self, v: Value, from: Type, to: types::Type) -> Value {
        let have = self.b.func.dfg.value_type(v);
        let float = matches!(from, Type::Float | Type::Double);
        match (float, to.is_float()) {
            (true, true) if have == to => v,
            (true, true) if to == types::F64 => self.b.ins().fpromote(to, v),
            (true, true) => self.b.ins().fdemote(to, v),
            (true, false) => self.b.ins().fcvt_to_sint_sat(to, v),
            (false, true) => self.b.ins().fcvt_from_sint(to, v),
            (false, false) if have == to => v,
            (false, false) if to.bits() > have.bits() => self.b.ins().sextend(to, v),
            (false, false) => self.b.ins().ireduce(to, v),
        }
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

    /// Loads scalars of the given types from the scratch slot.
    fn load_scratch(&mut self, tys: &[types::Type]) -> Vec<Value> {
        let base = self.b.ins().stack_addr(self.ptr, self.scratch, 0);
        self.loads(tys, self.flags, base, 0)
    }

    /// Scalars of the given types laid end to end at `base + offset`.
    fn loads(
        &mut self,
        tys: &[types::Type],
        flags: MemFlags,
        base: Value,
        offset: i32,
    ) -> Vec<Value> {
        tys.iter()
            .zip(byte_offsets(tys))
            .map(|(t, at)| self.b.ins().load(*t, flags, base, offset + at))
            .collect()
    }

    /// Stores values end to end at `base`, each at its own size.
    fn stores(&mut self, words: &[Value], flags: MemFlags, base: Value) {
        let mut at = 0i32;
        for word in words {
            self.b.ins().store(flags, *word, base, at);
            at += self.b.func.dfg.value_type(*word).bytes() as i32;
        }
    }

    /// A float constant of `like`'s width.
    fn fconst(&mut self, like: Value, v: f64) -> Value {
        if self.b.func.dfg.value_type(like) == types::F64 {
            self.b.ins().f64const(v)
        } else {
            self.b.ins().f32const(v as f32)
        }
    }

    /// An integer constant of `like`'s width.
    fn iconst(&mut self, like: Value, v: i64) -> Value {
        let t = self.b.func.dfg.value_type(like);
        self.b.ins().iconst(t, v)
    }

    fn scratch_addr(&mut self) -> Value {
        self.b.ins().stack_addr(self.ptr, self.scratch, 0)
    }

    /// The view of an input: its address.
    /// Reads what every input's view says before the run begins: the
    /// words, the bytes an element takes, and the row this run is on.
    /// A run stays on one row, so the row's first element is the same
    /// for every element of it and is worked out here.
    fn settle_views(&mut self, y: Value, z: Value) {
        let inputs = self.module.manifest.inputs.len();
        let mut settled = Vec::with_capacity(inputs);
        for input in 0..inputs as u32 {
            let view = self.view(input);
            let field =
                |fx: &mut Self, offset: i32| fx.b.ins().load(types::I32, fx.settled, view, offset);
            let words = self
                .b
                .ins()
                .load(self.ptr, self.settled, view, offsets::VIEW_WORDS);
            let width = field(self, offsets::VIEW_WIDTH);
            let height = field(self, offsets::VIEW_HEIGHT);
            let depth = field(self, offsets::VIEW_DEPTH);
            let element = field(self, offsets::VIEW_BYTES);
            let last_x = self.b.ins().iadd_imm(width, -1);
            let last_y = self.b.ins().iadd_imm(height, -1);
            let last_z = self.b.ins().iadd_imm(depth, -1);
            let cy = self.b.ins().umin(y, last_y);
            let cz = self.b.ins().umin(z, last_z);
            let layer = self.b.ins().imul(cz, height);
            let row = self.b.ins().iadd(layer, cy);
            let row = self.b.ins().imul(row, width);
            let rows = self.b.ins().imul(width, height);
            let count = self.b.ins().imul(rows, depth);
            let last = self.b.ins().iadd_imm(count, -1);
            let end_index = self.b.ins().iadd(row, last_x);
            let end_bytes = self.b.ins().imul(end_index, element);
            let end_offset = self.b.ins().uextend(self.ptr, end_bytes);
            let end = self.b.ins().iadd(words, end_offset);
            settled.push(Some(ViewOnce {
                words,
                element,
                last_x,
                row,
                last,
                at: None,
                end,
            }));
        }
        self.settled_views = settled;
    }

    /// What the run read about an input before it began, if it did.
    fn settled_view(&self, input: u32) -> Option<ViewOnce> {
        self.settled_views.get(input as usize).copied().flatten()
    }

    /// The words of one element of an input, loaded where the helper
    /// that copies them would have been called: the extent and the
    /// stride come from the view, and the element is clamped into the
    /// Field exactly as `orla_jit_sample` and `orla_jit_read_rows` do.
    /// A read is a handful of loads, and the loop around it keeps the
    /// view's own loads in registers.
    fn element_words(
        &mut self,
        input: u32,
        index: Value,
        ty: Type,
        tys: &[types::Type],
    ) -> Vec<Value> {
        let (words, element) = match self.settled_view(input) {
            Some(view) => (view.words, view.element),
            None => {
                let view = self.view(input);
                let words = self
                    .b
                    .ins()
                    .load(self.ptr, self.settled, view, offsets::VIEW_WORDS);
                let bytes = self
                    .b
                    .ins()
                    .load(types::I32, self.settled, view, offsets::VIEW_BYTES);
                (words, bytes)
            }
        };
        let bytes = self.b.ins().imul(index, element);
        let offset = self.b.ins().uextend(self.ptr, bytes);
        let at = self.b.ins().iadd(words, offset);
        self.field_loads(input, at, ty, tys)
    }

    /// The Kernel's words of a read of `input` at `base`, one per scalar
    /// of `ty`: whole-word loads when the input keeps its scalars wide,
    /// and otherwise each scalar loaded at its size and widened by its
    /// encoding's rule — the Host's, so the words are the interpreter's.
    fn field_loads(
        &mut self,
        input: u32,
        base: Value,
        ty: Type,
        tys: &[types::Type],
    ) -> Vec<Value> {
        let schema = &self.module.manifest.inputs[input as usize].ty.schema;
        let encoding = schema.uniform_encoding().unwrap_or(Encoding::Word);
        if encoding == Encoding::Word {
            return self.loads(tys, self.flags, base, 0);
        }
        let kinds: Vec<Scalar> = orla_ir::schema_of(ty, &self.module.structs)
            .map(|schema| schema.scalars().to_vec())
            .unwrap_or_default();
        let size = encoding.bytes() as i32;
        (0..tys.len())
            .map(|k| {
                let word = self.narrow_load(encoding, base, k as i32 * size);
                if kinds.get(k) == Some(&Scalar::Bool) {
                    let set = self.b.ins().icmp_imm(IntCC::NotEqual, word, 0);
                    self.b.ins().uextend(types::I32, set)
                } else {
                    word
                }
            })
            .collect()
    }

    /// One narrow scalar at `base + offset`, widened to the Kernel's word.
    fn narrow_load(&mut self, encoding: Encoding, base: Value, offset: i32) -> Value {
        let (i, f, flags) = (types::I32, types::F32, self.flags);
        let unit = |fx: &mut Self, raw: Value, signed: bool, scale: f32| {
            let x = if signed {
                fx.b.ins().fcvt_from_sint(f, raw)
            } else {
                fx.b.ins().fcvt_from_uint(f, raw)
            };
            let scale = fx.f32c(scale);
            fx.b.ins().fmul(x, scale)
        };
        match encoding {
            Encoding::U8 => self.b.ins().uload8(i, flags, base, offset),
            Encoding::I8 => self.b.ins().sload8(i, flags, base, offset),
            Encoding::U16 => self.b.ins().uload16(i, flags, base, offset),
            Encoding::I16 => self.b.ins().sload16(i, flags, base, offset),
            Encoding::Unorm8 => {
                let raw = self.b.ins().uload8(i, flags, base, offset);
                unit(self, raw, false, encoding::UNORM8)
            }
            Encoding::Unorm16 => {
                let raw = self.b.ins().uload16(i, flags, base, offset);
                unit(self, raw, false, encoding::UNORM16)
            }
            Encoding::Snorm8 => {
                let raw = self.b.ins().sload8(i, flags, base, offset);
                let x = unit(self, raw, true, encoding::SNORM8);
                let floor = self.f32c(-1.0);
                self.b.ins().fmax(x, floor)
            }
            Encoding::F16 => {
                let raw = self.b.ins().uload16(i, flags, base, offset);
                let call = self.b.ins().call(self.half, &[raw]);
                let bits = self.b.inst_results(call)[0];
                self.b.ins().bitcast(f, MemFlags::new(), bits)
            }
            Encoding::Word => self.b.ins().load(i, flags, base, offset),
        }
    }

    /// A run of words inside an element of an input: the element's own
    /// words, `within` bytes in. What a field or an array element of a
    /// read comes to, without the rest of the element.
    fn part_words(
        &mut self,
        input: u32,
        index: Value,
        within: Value,
        ty: Type,
        tys: &[types::Type],
    ) -> Vec<Value> {
        let (words, element) = match self.settled_view(input) {
            Some(view) => (view.words, view.element),
            None => {
                let view = self.view(input);
                let words = self
                    .b
                    .ins()
                    .load(self.ptr, self.settled, view, offsets::VIEW_WORDS);
                let bytes = self
                    .b
                    .ins()
                    .load(types::I32, self.settled, view, offsets::VIEW_BYTES);
                (words, bytes)
            }
        };
        let bytes = self.b.ins().imul(index, element);
        let total = self.b.ins().iadd(bytes, within);
        let offset = self.b.ins().uextend(self.ptr, total);
        let at = self.b.ins().iadd(words, offset);
        self.field_loads(input, at, ty, tys)
    }

    /// Where a path through an element leads, in bytes from its start at
    /// `unit` bytes a scalar: the words known when the Kernel compiled,
    /// then each index clamped into what it indexes and taken as its own
    /// width. The last word is held inside the element, whatever the
    /// indices say. The unit is what the element is read from: the
    /// input's own memory, or its widened words in the scratch slot.
    fn part_offset(
        &mut self,
        input: u32,
        path: &[orla_ir::Step],
        words: usize,
        unit: i64,
    ) -> Value {
        let element = self.module.manifest.inputs[input as usize].element();
        let layout = orla_ir::part_layout(&self.module.structs, element, path)
            .expect("a path validation walked");
        let mut offset = self.i32c(layout.base as i64);
        for (index, count, stride) in layout.steps {
            let at = self.words(&index)[0];
            let zero = self.i32c(0);
            let last = self.i32c(count as i64 - 1);
            let floored = self.b.ins().smax(at, zero);
            let clamped = self.b.ins().smin(floored, last);
            let scaled = self.b.ins().imul_imm(clamped, stride as i64);
            offset = self.b.ins().iadd(offset, scaled);
        }
        let stride = self.module.manifest.inputs[input as usize]
            .ty
            .schema
            .value_words();
        let last = self.i32c(stride.saturating_sub(words) as i64);
        let held = self.b.ins().smin(offset, last);
        let zero = self.i32c(0);
        let inside = self.b.ins().smax(held, zero);
        self.b.ins().imul_imm(inside, unit)
    }

    /// This element's own place in an input, clamped into it: what
    /// `sample` reads.
    fn sample_index(&mut self, input: u32, x: Value, y: Value, z: Value) -> Value {
        // The row is the same for every element of a run, so what is
        // left here is one clamp and one add.
        if let Some(view) = self.settled_view(input) {
            let cx = self.b.ins().umin(x, view.last_x);
            return self.b.ins().iadd(view.row, cx);
        }
        let view = self.view(input);
        let extent =
            |fx: &mut Self, offset: i32| fx.b.ins().load(types::I32, fx.flags, view, offset);
        let width = extent(self, offsets::VIEW_WIDTH);
        let height = extent(self, offsets::VIEW_HEIGHT);
        let depth = extent(self, offsets::VIEW_DEPTH);
        let last = |fx: &mut Self, size: Value, at: Value| {
            let end = fx.b.ins().iadd_imm(size, -1);
            fx.b.ins().umin(at, end)
        };
        let cx = last(self, width, x);
        let cy = last(self, height, y);
        let cz = last(self, depth, z);
        let layer = self.b.ins().imul(cz, height);
        let row = self.b.ins().iadd(layer, cy);
        let scaled = self.b.ins().imul(row, width);
        self.b.ins().iadd(scaled, cx)
    }

    /// An `Int` address into a Field, clamped to its elements: what `at`
    /// reads over rows.
    fn rows_index(&mut self, input: u32, address: Value) -> Value {
        if let Some(view) = self.settled_view(input) {
            let zero = self.i32c(0);
            let floored = self.b.ins().smax(address, zero);
            return self.b.ins().smin(floored, view.last);
        }
        let view = self.view(input);
        let width = self
            .b
            .ins()
            .load(types::I32, self.settled, view, offsets::VIEW_WIDTH);
        let height = self
            .b
            .ins()
            .load(types::I32, self.settled, view, offsets::VIEW_HEIGHT);
        let depth = self
            .b
            .ins()
            .load(types::I32, self.settled, view, offsets::VIEW_DEPTH);
        let rows = self.b.ins().imul(width, height);
        let count = self.b.ins().imul(rows, depth);
        let last = self.b.ins().iadd_imm(count, -1);
        let zero = self.i32c(0);
        let floored = self.b.ins().smax(address, zero);
        self.b.ins().smin(floored, last)
    }

    fn view(&mut self, input: u32) -> Value {
        let views = self
            .b
            .ins()
            .load(self.ptr, self.settled, self.ctx, offsets::VIEWS);
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
                // A constant is its scalars, each as the type lays it out.
                let _ = ty;
                value
                    .numbers()
                    .into_iter()
                    .map(|number| match number {
                        orla_ir::Number::Float(v) => self.f32c(v),
                        orla_ir::Number::Int(v) => self.i32c(i64::from(v)),
                        orla_ir::Number::Bool(v) => self.i32c(i64::from(v)),
                        orla_ir::Number::Double(v) => self.b.ins().f64const(v),
                        orla_ir::Number::Long(v) => self.b.ins().iconst(types::I64, v),
                    })
                    .collect()
            }
            Op::Param(index) => {
                let offset: usize = self.module.manifest.params[..*index as usize]
                    .iter()
                    .map(|p| byte_size(&word_types(self.module, p.ty)))
                    .sum();
                let base = self
                    .b
                    .ins()
                    .load(self.ptr, self.settled, self.ctx, offsets::PARAMS);
                let tys = word_types(self.module, ty);
                self.loads(&tys, self.flags, base, offset as i32)
            }
            Op::Arg(index) => self.args[*index as usize].clone(),
            Op::Unary(UnaryOp::Neg, a) => {
                let words = self.words(a);
                if self.ty(a).is_int_kind() {
                    words
                        .iter()
                        .map(|w| {
                            let zero = self.iconst(*w, 0);
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
                // A float vector's components may widen from integers;
                // anything else is its parts' scalars end to end.
                if ty.components().is_none() {
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
                let (base, flags) = if let Some(pointer) = self.pointers.get(&array.0) {
                    (*pointer, self.flags.with_readonly())
                } else if self.constants.contains_key(&array.0)
                    && matches!(self.ty(array), Type::Struct(_))
                {
                    (self.table(array.0), self.flags.with_readonly())
                } else if let Some((block, address)) = self.spilled.get(&array.0).copied()
                    && Some(block) == self.b.current_block()
                {
                    // The same array indexed again where it was put
                    // down: one slot holds it, however many places are
                    // read out of it. Only in the block that stored it,
                    // which is the block every later read is in when it
                    // can use what was stored.
                    (address, self.flags)
                } else {
                    let words = self.words(array);
                    let array_types = word_types(self.module, self.ty(array));
                    let slot = self.b.create_sized_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot,
                        byte_size(&array_types).max(4) as u32,
                        3,
                    ));
                    for (word, at) in words.iter().zip(byte_offsets(&array_types)) {
                        self.b.ins().stack_store(*word, slot, at);
                    }
                    let address = self.b.ins().stack_addr(self.ptr, slot, 0);
                    if let Some(block) = self.b.current_block() {
                        self.spilled.insert(array.0, (block, address));
                    }
                    (address, self.flags)
                };
                let raw = self.words(index)[0];
                let zero = self.i32c(0);
                let last = self.i32c(count as i64 - 1);
                let low = self.b.ins().smax(raw, zero);
                let clamped = self.b.ins().smin(low, last);
                let scaled = self.b.ins().imul_imm(clamped, byte_size(&tys) as i64);
                let offset = self.b.ins().uextend(self.ptr, scaled);
                let address = self.b.ins().iadd(base, offset);
                if flags.readonly()
                    && matches!(ty, Type::Struct(index) if self.module.structs[index as usize].is_array())
                {
                    self.pointers.insert(id.0, address);
                }
                self.loads(&tys, flags, address, 0)
            }
            Op::Call(callee, args) => {
                // The callee writes its result's words to a slot of ours.
                let tys = word_types(self.module, ty);
                let slot = self.b.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    byte_size(&tys).max(4) as u32,
                    3,
                ));
                let out = self.b.ins().stack_addr(self.ptr, slot, 0);
                let mut passed = vec![self.ctx, out];
                for arg in args {
                    passed.extend(self.words(arg));
                }
                self.b.ins().call(self.calls[*callee as usize], &passed);
                self.loads(&tys, self.flags, out, 0)
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
                    .load(types::I32, self.settled, self.ctx, offsets::WIDTH);
                let h = self
                    .b
                    .ins()
                    .load(types::I32, self.settled, self.ctx, offsets::HEIGHT);
                let d = self
                    .b
                    .ins()
                    .load(types::I32, self.settled, self.ctx, offsets::DEPTH);
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
                    .load(types::I32, self.settled, self.ctx, offsets::WIDTH);
                let h = self
                    .b
                    .ins()
                    .load(types::I32, self.settled, self.ctx, offsets::HEIGHT);
                let zh = self.b.ins().imul(z, h);
                let row = self.b.ins().iadd(zh, y);
                let base = self.b.ins().imul(row, w);
                vec![self.b.ins().iadd(base, x)]
            }
            Op::Time => vec![
                self.b
                    .ins()
                    .load(types::F32, self.settled, self.ctx, offsets::TIME),
            ],
            Op::Rate => vec![
                self.b
                    .ins()
                    .load(types::F32, self.settled, self.ctx, offsets::RATE),
            ],
            Op::Pointer => vec![
                self.b
                    .ins()
                    .load(types::F32, self.settled, self.ctx, offsets::POINTER),
                self.b
                    .ins()
                    .load(types::F32, self.settled, self.ctx, offsets::POINTER + 4),
            ],
            Op::Stick(which) => {
                // Three pairs in the context, the number held inside
                // them: an eight-byte step from the first.
                let which = self.words(which)[0];
                let zero = self.i32c(0);
                let last = self.i32c(2);
                let floored = self.b.ins().smax(which, zero);
                let held = self.b.ins().smin(floored, last);
                let off = self.b.ins().imul_imm(held, 8);
                let wide = self.b.ins().sextend(self.ptr, off);
                let base = self.b.ins().iadd_imm(self.ctx, offsets::STICKS as i64);
                let at = self.b.ins().iadd(base, wide);
                // A number outside the three is a pair of zeros.
                let inside = self.b.ins().icmp(IntCC::Equal, which, held);
                let x = self.b.ins().load(types::F32, self.settled, at, 0);
                let y = self.b.ins().load(types::F32, self.settled, at, 4);
                let none = self.f32c(0.0);
                vec![
                    self.b.ins().select(inside, x, none),
                    self.b.ins().select(inside, y, none),
                ]
            }
            Op::Wheel => vec![
                self.b
                    .ins()
                    .load(types::F32, self.settled, self.ctx, offsets::WHEEL),
                self.b
                    .ins()
                    .load(types::F32, self.settled, self.ctx, offsets::WHEEL + 4),
            ],
            Op::Window => vec![
                self.b
                    .ins()
                    .load(types::F32, self.settled, self.ctx, offsets::WINDOW),
                self.b
                    .ins()
                    .load(types::F32, self.settled, self.ctx, offsets::WINDOW + 4),
            ],
            Op::Scale => {
                vec![
                    self.b
                        .ins()
                        .load(types::F32, self.settled, self.ctx, offsets::SCALE),
                ]
            }
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
                    .load(types::I32, self.settled, view, offsets::VIEW_WIDTH);
                let h = self
                    .b
                    .ins()
                    .load(types::I32, self.settled, view, offsets::VIEW_HEIGHT);
                let d = self
                    .b
                    .ins()
                    .load(types::I32, self.settled, view, offsets::VIEW_DEPTH);
                let wh = self.b.ins().imul(w, h);
                vec![self.b.ins().imul(wh, d)]
            }
            Op::Size(input) => {
                let view = self.view(*input);
                let w = self
                    .b
                    .ins()
                    .load(types::I32, self.settled, view, offsets::VIEW_WIDTH);
                let h = self
                    .b
                    .ins()
                    .load(types::I32, self.settled, view, offsets::VIEW_HEIGHT);
                let d = self
                    .b
                    .ins()
                    .load(types::I32, self.settled, view, offsets::VIEW_DEPTH);
                match ty {
                    Type::Int => vec![w],
                    Type::Vec2 => vec![self.u32_to_f32(w), self.u32_to_f32(h)],
                    _ => vec![self.u32_to_f32(w), self.u32_to_f32(h), self.u32_to_f32(d)],
                }
            }
            Op::Sample(input) => {
                let tys = word_types(self.module, ty);
                // The run carries where this element is in every input,
                // so a `sample` is the loads and nothing else.
                if let Some(at) = self.settled_view(*input).and_then(|view| view.at) {
                    return self.field_loads(*input, at, ty, &tys);
                }
                let (x, y, z, _) = self.coords.expect("entry point");
                let index = self.sample_index(*input, x, y, z);
                self.element_words(*input, index, ty, &tys)
            }
            Op::At(input, address) => {
                let words = self.words(address);
                let out = self.scratch_addr();
                let index = self.i32c(*input as i64);
                match self.ty(address) {
                    Type::Int => {
                        let at = self.rows_index(*input, words[0]);
                        let tys = word_types(self.module, ty);
                        return self.element_words(*input, at, ty, &tys);
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
            Op::Part { input, at, path } => {
                let tys = word_types(self.module, ty);
                // Read from the input's own memory, a scalar is its
                // encoding's size; read from the scratch slot, a word.
                let unit = self.module.manifest.inputs[*input as usize]
                    .ty
                    .schema
                    .uniform_encoding()
                    .map_or(4, |encoding| encoding.bytes() as i64);
                // This element's own, where there is no address: the
                // run already carries where it is in every input, so
                // the part is the loads at its offset and nothing else.
                let Some(at) = at else {
                    let within = self.part_offset(*input, path, byte_size(&tys) / 4, unit);
                    if let Some(base) = self.settled_view(*input).and_then(|view| view.at) {
                        let offset = self.b.ins().uextend(self.ptr, within);
                        let base = self.b.ins().iadd(base, offset);
                        return self.field_loads(*input, base, ty, &tys);
                    }
                    let (x, y, z, _) = self.coords.expect("entry point");
                    let index = self.sample_index(*input, x, y, z);
                    return self.part_words(*input, index, within, ty, &tys);
                };
                let address = self.words(at);
                if matches!(self.ty(at), Type::Int) {
                    let within = self.part_offset(*input, path, byte_size(&tys) / 4, unit);
                    let index = self.rows_index(*input, address[0]);
                    return self.part_words(*input, index, within, ty, &tys);
                }
                // Over a grid or a volume the element is read as it
                // always was, and the part is taken out of what came
                // back: the address is the Host's to work out.
                let out = self.scratch_addr();
                let index = self.i32c(*input as i64);
                match self.ty(at) {
                    Type::Vec2 => {
                        let zero = self.f32c(0.0);
                        self.b.ins().call(
                            self.read,
                            &[self.ctx, index, address[0], address[1], zero, out],
                        );
                    }
                    _ => {
                        self.b.ins().call(
                            self.read,
                            &[self.ctx, index, address[0], address[1], address[2], out],
                        );
                    }
                }
                let within = self.part_offset(*input, path, byte_size(&tys) / 4, 4);
                let bytes = self.b.ins().uextend(self.ptr, within);
                let base = self.b.ins().iadd(out, bytes);
                self.loads(&tys, self.flags, base, 0)
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
                self.stores(&words, self.flags, slot);
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
        let int_like = ta.is_int_kind() || tb.is_int_kind();
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
                    let min = if fx.b.func.dfg.value_type(x) == types::I64 {
                        i64::MIN
                    } else {
                        i64::from(i32::MIN)
                    };
                    let is_min = fx.b.ins().icmp_imm(IntCC::Equal, x, min);
                    let minus_one = fx.b.ins().icmp_imm(IntCC::Equal, y, -1);
                    let overflow = fx.b.ins().band(is_min, minus_one);
                    fx.fail_if(overflow, OVERFLOW);
                    let unsafe_ = fx.b.ins().bor(zero, overflow);
                    let one = fx.iconst(y, 1);
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
                let fmod = if fx.b.func.dfg.value_type(x) == types::F64 {
                    fx.fmod
                } else {
                    fx.fmodf
                };
                let call = fx.b.ins().call(fmod, &[x, y]);
                fx.b.inst_results(call)[0]
            }),
            BinaryOp::Less | BinaryOp::LessEqual | BinaryOp::Greater | BinaryOp::GreaterEqual => {
                let cond = if ta.is_int_kind() {
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
                    let same = if (ta.is_int_kind() || ta == Type::Bool) && tb == ta {
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
                // The count modulo the width: 32, or 64 for a long.
                let bits = i64::from(fx.b.func.dfg.value_type(x).bits());
                let count = fx.b.ins().band_imm(y, bits - 1);
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
        let is_int = |fx: &Self, k: usize| fx.ty(&args[k]).is_int_kind();
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
            Intrinsic::Abs if is_int(self, 0) => map1(self, &mut |fx, v| fx.b.ins().iabs(v)),
            Intrinsic::Abs => map1(self, &mut |fx, v| fx.b.ins().fabs(v)),
            Intrinsic::Sign if is_int(self, 0) => map1(self, &mut |fx, v| {
                let pos = fx.b.ins().icmp_imm(IntCC::SignedGreaterThan, v, 0);
                let neg = fx.b.ins().icmp_imm(IntCC::SignedLessThan, v, 0);
                let one = fx.iconst(v, 1);
                let minus = fx.iconst(v, -1);
                let zero = fx.iconst(v, 0);
                let low = fx.b.ins().select(neg, minus, zero);
                fx.b.ins().select(pos, one, low)
            }),
            Intrinsic::Sign => map1(self, &mut |fx, v| {
                let zero = fx.fconst(v, 0.0);
                let pos = fx.b.ins().fcmp(FloatCC::GreaterThan, v, zero);
                let neg = fx.b.ins().fcmp(FloatCC::LessThan, v, zero);
                let one = fx.fconst(v, 1.0);
                let minus = fx.fconst(v, -1.0);
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
                let table = if self.ty(&args[0]).is_64bit() {
                    &self.unary64
                } else {
                    &self.unary
                };
                let func = table
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
            Intrinsic::ToFloat | Intrinsic::ToInt if self.ty(&args[0]).is_64bit() => {
                self.conversion(intrinsic, args, &w[0])
            }
            Intrinsic::ToDouble | Intrinsic::ToLong => self.conversion(intrinsic, args, &w[0]),
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
            Intrinsic::Pow => self.zip2(&w[0], &w[1], |fx, x, y| {
                let pow = if fx.b.func.dfg.value_type(x) == types::F64 {
                    fx.pow
                } else {
                    fx.powf
                };
                let call = fx.b.ins().call(pow, &[x, y]);
                fx.b.inst_results(call)[0]
            }),
            Intrinsic::Step => self.zip2(&w[0], &w[1], |fx, edge, x| {
                let ge = fx.b.ins().fcmp(FloatCC::GreaterThanOrEqual, x, edge);
                let one = fx.fconst(x, 1.0);
                let zero = fx.fconst(x, 0.0);
                fx.b.ins().select(ge, one, zero)
            }),
            Intrinsic::Smoothstep => self.zip3(&w[0], &w[1], &w[2], |fx, low, high, x| {
                let num = fx.b.ins().fsub(x, low);
                let den = fx.b.ins().fsub(high, low);
                let t = fx.b.ins().fdiv(num, den);
                let zero = fx.fconst(t, 0.0);
                let one = fx.fconst(t, 1.0);
                // The same clamp `clamp` is: a comparison lets a NaN
                // through and these do not, and a Kernel that writes
                // `clamp` and one that writes `smoothstep` must not
                // disagree about that.
                let t = fx.fmax(t, zero);
                let t = fx.fmin(t, one);
                let two = fx.fconst(t, 2.0);
                let three = fx.fconst(t, 3.0);
                let tt = fx.b.ins().fmul(t, t);
                let twice = fx.b.ins().fmul(two, t);
                let rest = fx.b.ins().fsub(three, twice);
                fx.b.ins().fmul(tt, rest)
            }),
            Intrinsic::Atan2 => {
                let atan2 = if self.ty(&args[0]) == Type::Double {
                    self.atan2
                } else {
                    self.atan2f
                };
                let call = self.b.ins().call(atan2, &[w[0][0], w[1][0]]);
                vec![self.b.inst_results(call)[0]]
            }
        }
    }
}

/// Whether an extraction reads out of a value holding a 64-bit scalar,
/// whose words a table would misplace.
fn instruction_reads_wide(fx: &Fx<'_, '_>, op: &Op) -> bool {
    match op {
        Op::Extract(source, _) => holds_wide(fx.module, fx.ty(source)),
        _ => false,
    }
}
