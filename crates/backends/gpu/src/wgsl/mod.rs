//! WGSL emission from validated Orla IR.
//!
//! A value slot is a WGSL `let`, or a `var` for `if` and `loop` results;
//! IR blocks are WGSL blocks. Resources are storage buffers of `u32`
//! words, one per scalar, plus uniform arrays for context and parameters.

use std::fmt::Write;

use orla_abi::{Encoding, Scalar, encoding};
use orla_ir::{
    BinaryOp, Block, Combine, Function, Instruction, Intrinsic, Mode, Module, Op, Shape, Type,
    UnaryOp, Value, ValueId,
};

mod emitter;
mod map;
mod prefix;
mod prelude;
mod reduce;
mod scatter;
mod storage;

use emitter::*;
pub use map::*;
pub use prefix::*;
use prelude::*;
pub use reduce::*;
pub use scatter::*;
pub use storage::*;

/// Binding index of the context uniform.
pub const CONTEXT_BINDING: u32 = 0;
/// Binding index of the parameter uniform.
pub const PARAMS_BINDING: u32 = 1;
/// Binding index of the output storage buffer.
pub const OUTPUT_BINDING: u32 = 2;
/// Binding index of the first input storage buffer; input `i` is at
/// `INPUT_BINDING + i`.
pub const INPUT_BINDING: u32 = 3;

/// Binding index of a failing pass's record. Word 0 is the first
/// failure: all ones less `element * 4 + code`, the greatest kept, so
/// it names the first element in row-major order; zero is none. Word 1
/// counts blocks of work taken, word 2 is the most allowed
/// ([`orla_rules::Run`]).
pub const FAILURE_BINDING: u32 = 90;

/// The deepest brace nesting WGSL compiles.
pub const BRACE_LIMIT: usize = 127;

/// An upper bound on the module shader's brace depth: a function body
/// is one, an `if` or a loop one more, a `match` two. An `if` emitted as
/// a `select` still counts.
pub fn brace_depth(module: &Module) -> usize {
    fn nest(instructions: &[Instruction]) -> usize {
        instructions
            .iter()
            .map(|instruction| match &instruction.op {
                Op::If {
                    then, otherwise, ..
                } => 1 + nest(&then.body).max(nest(&otherwise.body)),
                Op::Loop { until, body, .. } => {
                    1 + nest(&body.body).max(until.as_ref().map_or(0, |until| nest(&until.body)))
                }
                Op::Match {
                    arms, otherwise, ..
                } => {
                    2 + arms
                        .iter()
                        .map(|(_, arm)| nest(&arm.body))
                        .fold(nest(&otherwise.body), usize::max)
                }
                _ => 0,
            })
            .max()
            .unwrap_or(0)
    }
    module
        .functions
        .iter()
        .chain([&module.entry])
        .chain(module.merge.iter())
        .chain(module.init.iter())
        .map(|function| 1 + nest(&function.body))
        .max()
        .unwrap_or(1)
}

/// Binding index of the buffer that packs inputs; see [`packed_inputs`].
pub const PACKED_BINDING: u32 = 91;

/// `(uniforms, storage)` buffers the module's widest pass binds unpacked:
/// context and parameters, every input, every written output, the
/// failure record where arithmetic can fail, an emitting scatter's target.
pub fn bindings(module: &Module) -> (usize, usize) {
    let inputs = module.manifest.inputs.len();
    let failure = usize::from(can_fail(module));
    // A reduction's widest pass is its fold: a third uniform (the level's
    // count) and the level it reads beside the one it writes.
    if module.manifest.mode == Mode::Reduce {
        return (3, inputs + 2 + failure);
    }
    let storage =
        inputs + written_outputs(module) + failure + usize::from(!emitted_pairs(module).is_empty());
    (2, storage)
}

/// The inputs packed into one buffer so the module binds at most `most`
/// buffers and `most_storage` storage buffers: the fewest that fit, last
/// read-only inputs first (a memory keeps its own). Packing `k` inputs
/// saves `k - 1` bindings. Empty when the module fits, or when packing
/// every read-only input is still too wide.
pub fn packed_inputs(module: &Module, most: usize, most_storage: usize) -> Vec<usize> {
    let (uniforms, storage) = bindings(module);
    let over = (uniforms + storage)
        .saturating_sub(most)
        .max(storage.saturating_sub(most_storage));
    if over == 0 {
        return Vec::new();
    }
    let candidates: Vec<usize> = (0..module.manifest.inputs.len())
        .rev()
        .filter(|&k| {
            !module
                .manifest
                .memories
                .contains(&module.manifest.inputs[k].name)
        })
        .collect();
    if candidates.len() < over + 1 {
        return Vec::new();
    }
    let mut packed: Vec<usize> = candidates[..over + 1].to_vec();
    packed.sort_unstable();
    packed
}

/// `(buffer, offset prefix)` input `index` reads from: its own buffer, or
/// the pack at the offset the parameters carry, in the input's load unit
/// (words, or bytes where it is narrow).
fn input_source(
    module: &Module,
    packed: &[usize],
    index: usize,
    encoding: Encoding,
) -> (String, String) {
    let Some(slot) = packed.iter().position(|&k| k == index) else {
        return (format!("in{index}"), String::new());
    };
    let first = param_slots(module).1;
    let offset = format!("params[{}u][{}u]", first + slot / 4, slot % 4);
    let offset = if encoding == Encoding::Word {
        offset
    } else {
        format!("{offset} * 4u")
    };
    ("inputs_packed".to_owned(), format!("{offset} + "))
}

/// Bytes of the failure record, rounded to sixteen: the first failure,
/// blocks taken, their limit, the remainder short of a block, and the
/// first element whose whole sum overflows ([`sums_whole`]).
pub const FAILURE_BYTES: u64 = 32;

/// The failure record as declared by a pass that only records a sum.
pub(crate) const FAILURE_STRUCT: &str = "struct Failure { word: atomic<u32>, blocks: atomic<u32>, limit: u32, left: atomic<u32>, total: atomic<u32> }";

/// Binding index of an emitting scatter's high words, beside its target.
pub const EMIT_HIGHS_BINDING: u32 = 92;

/// Whether a scatter or prefix sums integers whole: each total is a word
/// and a high word, checked once every value is in, and the first
/// element whose total overflows is recorded.
pub fn sums_whole(module: &Module) -> bool {
    let (scalars, combine) = match module.manifest.mode {
        Mode::Scatter { base, combine } => (
            module.manifest.inputs[base as usize].ty.schema.scalars(),
            combine,
        ),
        Mode::Prefix { combine } => (module.manifest.outputs[0].ty.schema.scalars(), combine),
        _ => return false,
    };
    combine == Combine::Sum && scalars.iter().any(|s| matches!(s, orla_abi::Scalar::Int))
}

/// Whether a whole sum keeps its high words in a separate buffer: a
/// prefix, and a parallel scatter. An ordered scatter keeps them as it
/// folds.
pub fn highs_beside(module: &Module) -> bool {
    sums_whole(module)
        && (matches!(module.manifest.mode, Mode::Prefix { .. })
            || matches!(scatter_plan(module), ScatterPlan::Parallel { .. }))
}

/// A failure's code in the word: an overflow.
pub const FAILED_OVERFLOW: u32 = 1;
/// A failure's code in the word: a division by zero.
pub const FAILED_DIVISION: u32 = 2;
/// A failure's code in the word: more loop iterations than the budget.
pub const FAILED_BUDGET: u32 = 3;

/// Whether the module's integer arithmetic can fail (`+ - *` overflow,
/// `/ %` `udiv` `urem` by zero, `-` of the least integer) or its work is
/// accounted. Only such a module's passes record failures.
pub fn can_fail(module: &Module) -> bool {
    // An accounted module fails when its loops exceed the budget.
    module.accounted()
        || module
            .functions
            .iter()
            .chain([&module.entry])
            .chain(module.merge.iter())
            .chain(module.init.iter())
            .any(|function| function.body.iter().any(fails_within))
}

/// Whether an instruction or any nested one can fail.
fn fails_within(instruction: &Instruction) -> bool {
    let nested = |blocks: &[&Block]| {
        blocks
            .iter()
            .any(|block| block.body.iter().any(fails_within))
    };
    let blocks = instruction.op.blocks();
    if blocks.is_empty() {
        fails(&instruction.op, instruction.ty)
    } else {
        nested(&blocks)
    }
}

/// Whether one operation of result type `ty` can fail.
fn fails(op: &Op, ty: Type) -> bool {
    match op {
        Op::Binary(
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem,
            ..,
        )
        | Op::Unary(UnaryOp::Neg, _) => ty.is_int_kind(),
        Op::Intrinsic(Intrinsic::Udiv | Intrinsic::Urem, _) => true,
        _ => false,
    }
}

/// Outputs the main pass writes: every output of a map, each in its own
/// buffer; one for other modes.
pub fn written_outputs(module: &Module) -> usize {
    match module.manifest.mode {
        Mode::Map => module.manifest.outputs.len().max(1),
        _ => 1,
    }
}

/// Binding index of output `k`: the first at [`OUTPUT_BINDING`], the
/// others after the inputs.
pub fn output_binding(module: &Module, k: usize) -> u32 {
    if k == 0 {
        OUTPUT_BINDING
    } else {
        INPUT_BINDING + (module.manifest.inputs.len() + k - 1) as u32
    }
}

/// Workgroup edge length in the first two dimensions.
pub const WORKGROUP: u32 = 8;
/// Context slots before the per-input extents: `[width, height, time,
/// button bits]`, `[pointer.x, pointer.y, depth, rate]`, two slots of key
/// bits, `[wheel.x, wheel.y, window.x, window.y]`, and `[scale, fold, 0,
/// 0]`, where `fold` is the elements per unit of `gid.y` for a long
/// `Rows` domain dispatched as a grid (else zero). Each input then has
/// `[width, height, depth, 0]`. Extents, the fold and the key bits are
/// the bits of `u32`s, read with `bitcast`, so every extent is exact.
pub const CONTEXT_HEAD: usize = 8;

/// The context word holding 1, which the shader compiler cannot fold.
/// Multiplied into an integer dividend, it makes a narrowed value (such
/// as `x >>> 16`) a whole `i32` for division, which Metal miscompiles.
pub const OPAQUE_ONE: &str = "ctx[5].z";

/// The most workgroups in one dispatch dimension.
pub const MAX_GROUPS: u32 = 65535;

/// `(x, y)` workgroups for a one-dimensional pass of `groups`: rows of at
/// most `MAX_GROUPS`. The pass's index is `gid.x + gid.y * span`, span
/// being workgroup size times `MAX_GROUPS`; its bounds check drops the
/// threads past the end.
pub fn folded(groups: u32) -> (u32, u32) {
    if groups <= MAX_GROUPS {
        (groups, 1)
    } else {
        (MAX_GROUPS, groups.div_ceil(MAX_GROUPS))
    }
}

/// A map's dispatch over `domain`: workgroups in x, y, z, and the fold
/// (elements per unit of `gid.y`) when rows exceed one dimension.
pub fn dispatch(domain: orla_abi::Domain) -> (u32, u32, u32, u32) {
    let groups_x = domain.width().div_ceil(WORKGROUP);
    if domain.shape() == orla_abi::Shape::Rows && groups_x > MAX_GROUPS {
        let fold = MAX_GROUPS * WORKGROUP;
        let strips = domain.width().div_ceil(fold);
        (MAX_GROUPS, strips.div_ceil(WORKGROUP), 1, fold)
    } else {
        (
            groups_x,
            domain.height().div_ceil(WORKGROUP),
            domain.depth(),
            0,
        )
    }
}

/// The shader's name for helper function `index`. Source names stay out
/// of the shader; `fn_<digits>` matches no WGSL reserved or predeclared
/// word.
fn function_name(index: usize) -> String {
    format!("fn_{index}")
}

/// The shader's name for struct field `k`, by the same rule.
fn member_name(k: usize) -> String {
    format!("m_{k}")
}

/// The path to field `k` of a struct value: a member or an array index.
fn field_path(module: &Module, index: u32, k: usize) -> String {
    if module.structs[index as usize].is_array() {
        format!("[{k}]")
    } else {
        format!(".{}", member_name(k))
    }
}

/// The declaration of struct `index`: a WGSL struct, or an array alias;
/// both build with `S{index}(...)`.
fn struct_decl(module: &Module, index: usize) -> String {
    let def = &module.structs[index];
    match def.element {
        Some(element) => format!(
            "alias S{index} = array<{}, {}>;",
            wgsl_type(element),
            def.fields.len()
        ),
        None => {
            let fields = def
                .fields
                .iter()
                .enumerate()
                .map(|(k, field)| format!("{}: {}", member_name(k), wgsl_type(field.ty)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("struct S{index} {{ {fields} }}")
        }
    }
}

fn wgsl_type(ty: Type) -> String {
    match ty {
        Type::Scalar(kind) => scalar_type(kind).into(),
        Type::Vector(kind, n) => format!("vec{n}<{}>", scalar_type(kind)),
        Type::Matrix(kind, n) => format!("mat{n}x{n}<{}>", scalar_type(kind)),
        Type::Struct(index) => format!("S{index}"),
    }
}

/// The WGSL scalar a kind is; a `Double` never reaches a shader.
fn scalar_type(kind: Scalar) -> &'static str {
    match kind {
        Scalar::Float => "f32",
        Scalar::Int => "i32",
        Scalar::Bool => "bool",
        Scalar::Long => "i64",
        Scalar::Double => unreachable!("the GPU refuses a module holding a Double"),
    }
}

/// A matrix of `f(0), f(1), ...` over its columns, for vector-only
/// builtins.
fn per_column(ty: Type, f: impl Fn(usize) -> String) -> String {
    let n = usize::from(ty.order().unwrap_or(0));
    let columns = (0..n).map(f).collect::<Vec<_>>().join(", ");
    format!("{}({columns})", wgsl_type(ty))
}

fn literal(value: &Value) -> String {
    // WGSL rejects non-finite constant expressions: a NaN or infinity is
    // written as its bits.
    let f = |v: f32| {
        if v.is_finite() {
            format!("{v:?}")
        } else {
            format!("bitcast<f32>({:#010x}u)", v.to_bits())
        }
    };
    let ints = |v: &[i32]| {
        v.iter()
            .map(|i| literal(&Value::Int(*i)))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let floats = |v: &[f32]| v.iter().map(|x| f(*x)).collect::<Vec<_>>().join(", ");
    let longs = |v: &[i64]| {
        v.iter()
            .map(|l| long_literal(*l))
            .collect::<Vec<_>>()
            .join(", ")
    };
    match value {
        Value::Float(v) => f(*v),
        // The most negative int is written as its bits.
        Value::Int(i32::MIN) => "bitcast<i32>(0x80000000u)".to_owned(),
        Value::Int(v) => format!("{v}i"),
        Value::Bool(v) => v.to_string(),
        Value::Vec2([x, y]) => format!("vec2<f32>({}, {})", f(*x), f(*y)),
        Value::Vec3([x, y, z]) => format!("vec3<f32>({}, {}, {})", f(*x), f(*y), f(*z)),
        Value::Vec4([x, y, z, w]) => {
            format!("vec4<f32>({}, {}, {}, {})", f(*x), f(*y), f(*z), f(*w))
        }
        Value::IVec2(v) => format!("vec2<i32>({})", ints(v)),
        Value::IVec3(v) => format!("vec3<i32>({})", ints(v)),
        Value::IVec4(v) => format!("vec4<i32>({})", ints(v)),
        Value::Mat2(v) => format!("mat2x2<f32>({})", floats(v)),
        Value::Mat3(v) => format!("mat3x3<f32>({})", floats(v)),
        Value::Mat4(v) => format!("mat4x4<f32>({})", floats(v)),
        Value::Long(v) => long_literal(*v),
        Value::LVec2(v) => format!("vec2<i64>({})", longs(v)),
        Value::LVec3(v) => format!("vec3<i64>({})", longs(v)),
        // Records are built by `Construct`; the validator refuses
        // record constants.
        Value::Record(_) => unreachable!("a constant of the IR is never a record"),
        other => unreachable!("the GPU refuses a module holding {other}"),
    }
}

/// A `Long` literal; the most negative is written as `-MAX - 1`.
fn long_literal(v: i64) -> String {
    if v == i64::MIN {
        "(-9223372036854775807li - 1li)".to_owned()
    } else {
        format!("{v}li")
    }
}

/// The call that keeps an element's remainder short of a block, in an
/// accounted module.
fn keep_left(module: &Module) -> &'static str {
    if module.accounted() {
        "keep_left();"
    } else {
        ""
    }
}

/// The call at a pass's end that settles its elements' remainders.
fn settle_left(module: &Module) -> &'static str {
    if module.accounted() {
        "settle_left();"
    } else {
        ""
    }
}
