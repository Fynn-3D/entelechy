//! WGSL emission from validated Orla IR.
//!
//! Every value slot becomes a WGSL `let` (straight-line values) or `var`
//! (results of `if` and `loop`). Blocks map to WGSL blocks, so IR scoping
//! and WGSL scoping coincide. Resources are storage buffers of `u32` words,
//! one per scalar, plus two small uniform arrays for context and parameters.

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

/// Binding index of the record a pass that can fail keeps. Its first
/// word is the first failure: all ones less the element times four plus
/// the failure's code, the greatest kept, so the first element in
/// row-major order is the one named, as the reference names it; zero is
/// none. Its second counts the blocks of work the run has taken, and its
/// third is the most it may take ([`orla_rules::Run`]).
pub const FAILURE_BINDING: u32 = 90;

/// How deeply WGSL lets braces nest: a shader past it does not compile.
pub const BRACE_LIMIT: usize = 127;

/// How deeply the module's shader nests its braces, at most: a function's
/// body is one, an `if` one more for its arms, a loop one more for its
/// body and its `until`, a `match` two, for the switch and its cases.
/// An `if` the emitter turns into a `select` opens none, but is counted
/// all the same, so this is never less than what is emitted.
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

/// Binding index of the inputs a pass packs into one buffer when it
/// binds more than the device does at once; see [`packed_inputs`].
pub const PACKED_BINDING: u32 = 91;

/// Buffers a module's widest pass binds when nothing is packed: its two
/// uniforms (the context and the parameters), every input, every output
/// it writes, the failure record where its arithmetic can fail, and an
/// emitting scatter's target.
pub fn bindings(module: &Module) -> (usize, usize) {
    let inputs = module.manifest.inputs.len();
    let failure = usize::from(can_fail(module));
    // A reduction's fold pass is its widest: a third uniform (the level's
    // count) and the level it reads beside the one it writes.
    if module.manifest.mode == Mode::Reduce {
        return (3, inputs + 2 + failure);
    }
    let storage =
        inputs + written_outputs(module) + failure + usize::from(!emitted_pairs(module).is_empty());
    (2, storage)
}

/// The inputs a module packs into one buffer so it binds no more than
/// `most` buffers in all and `most_storage` storage buffers: none when it
/// fits, else the fewest that do, the last read-only inputs first (a
/// memory is written, so it keeps its own). Packing `k` inputs binds
/// `k - 1` fewer buffers, since the pack is one. Empty too when packing
/// every read-only input would not be enough, which leaves the module
/// wider than the device.
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

/// Where the words of input `index` are read from: its own buffer, or
/// the pack at the offset the parameters carry for it, in the unit the
/// input's loads count in (words, or bytes where it is kept narrow).
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

/// The bytes of the failure record: three words, and one to round it.
pub const FAILURE_BYTES: u64 = 16;

/// A failure's code in the word: an overflow.
pub const FAILED_OVERFLOW: u32 = 1;
/// A failure's code in the word: a division by zero.
pub const FAILED_DIVISION: u32 = 2;
/// A failure's code in the word: more loop iterations than the budget.
pub const FAILED_BUDGET: u32 = 3;

/// Whether any function of the module does integer arithmetic that can
/// fail: `+ - *` past the ends of the range, `/ %` by zero, `-` of the
/// least integer, `udiv` and `urem` by zero. Such a module's passes
/// record their first failure; any other pays nothing for the checking.
pub fn can_fail(module: &Module) -> bool {
    // A module whose work is accounted fails where its looks say.
    module.accounted()
        || module
            .functions
            .iter()
            .chain([&module.entry])
            .chain(module.merge.iter())
            .chain(module.init.iter())
            .any(|function| function.body.iter().any(fails_within))
}

/// Whether an instruction, or any inside it, can fail.
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

/// How many outputs the main pass writes: a map's every output, each to a
/// buffer of its own; one for the other modes, whose single output is
/// what their later passes work on.
pub fn written_outputs(module: &Module) -> usize {
    match module.manifest.mode {
        Mode::Map => module.manifest.outputs.len().max(1),
        _ => 1,
    }
}

/// Binding index of output `k`: the first at [`OUTPUT_BINDING`], the
/// others after the inputs, so no input moves when a map has several.
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
/// 0]`, where `fold` is the elements each unit of `gid.y` stands for
/// when a long `Rows` domain is dispatched as a grid of workgroups (zero
/// otherwise). Each input then has `[width, height, depth, 0]`.
pub const CONTEXT_HEAD: usize = 8;

/// The context word holding 1, as an expression: a value the shader
/// compiler cannot fold, multiplied into an integer dividend so that a
/// value it narrowed (the top sixteen bits of a word, say) is a whole
/// `i32` again when it divides. Metal divided such a narrowed value by
/// a constant wrongly (`(x >>> 16) % 7` came out 8192 for a zero).
pub const OPAQUE_ONE: &str = "ctx[5].z";

/// The most workgroups one dispatch dimension takes.
pub const MAX_GROUPS: u32 = 65535;

/// The workgroups a one-dimensional pass of `groups` dispatches: one row
/// while they fit in a dimension, rows of `MAX_GROUPS` beyond. A pass
/// reads its index as `gid.x + gid.y * span`, the span being its
/// workgroup size times `MAX_GROUPS`, and its bounds check turns away
/// the threads past the end of the last row.
pub fn folded(groups: u32) -> (u32, u32) {
    if groups <= MAX_GROUPS {
        (groups, 1)
    } else {
        (MAX_GROUPS, groups.div_ceil(MAX_GROUPS))
    }
}

/// How a map over `domain` is dispatched: the workgroups in x, y and z,
/// and the fold (elements per unit of `gid.y`) when rows are too many
/// for one dimension.
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

/// The shader's name for helper function `index` of a module. A name the
/// source wrote never reaches the shader, and a name WGSL has for itself
/// must not either: `f32` is its type, and a 33rd helper once took that
/// name. No word WGSL reserves or predeclares is letters, an underscore
/// and digits, so no number of helpers meets one.
fn function_name(index: usize) -> String {
    format!("fn_{index}")
}

/// The shader's name for field `k` of a struct, by the same rule.
fn member_name(k: usize) -> String {
    format!("m_{k}")
}

/// The path to field `k` of a struct value: a named field, or an
/// array's element.
fn field_path(module: &Module, index: u32, k: usize) -> String {
    if module.structs[index as usize].is_array() {
        format!("[{k}]")
    } else {
        format!(".{}", member_name(k))
    }
}

/// The declaration of struct `index`: a WGSL struct, or an alias of an
/// array, which builds with the same `S{index}(...)` constructor.
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

/// `f(m[0]), f(m[1]), ...` over a matrix's columns, for the builtins
/// WGSL has for vectors but not for matrices.
fn per_column(ty: Type, f: impl Fn(usize) -> String) -> String {
    let n = usize::from(ty.order().unwrap_or(0));
    let columns = (0..n).map(f).collect::<Vec<_>>().join(", ");
    format!("{}({columns})", wgsl_type(ty))
}

fn literal(value: &Value) -> String {
    // WGSL evaluates an expression of constants when it compiles, and
    // refuses one that is not finite: a NaN or an infinity the optimizer
    // folded is written as its bits.
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
        // WGSL has no literal for the most negative int; its bits do.
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
        Value::Mat2(v) => format!("mat2x2<f32>({})", floats(v)),
        Value::Mat3(v) => format!("mat3x3<f32>({})", floats(v)),
        Value::Mat4(v) => format!("mat4x4<f32>({})", floats(v)),
        Value::Long(v) => long_literal(*v),
        Value::LVec2(v) => format!("vec2<i64>({})", longs(v)),
        Value::LVec3(v) => format!("vec3<i64>({})", longs(v)),
        // The validator refuses a constant record: a struct or an array
        // is built by `Construct`.
        Value::Record(_) => unreachable!("a constant of the IR is never a record"),
        other => unreachable!("the GPU refuses a module holding {other}"),
    }
}

/// A `Long` as WGSL writes one; the most negative has no literal, and
/// is one below the next.
fn long_literal(v: i64) -> String {
    if v == i64::MIN {
        "(-9223372036854775807li - 1li)".to_owned()
    } else {
        format!("{v}li")
    }
}

/// A pass's call that keeps what an element ran short of a block, when
/// the module counts its work.
fn keep_left(module: &Module) -> &'static str {
    if module.accounted() {
        "keep_left();"
    } else {
        ""
    }
}

/// A pass's call that hands the run what its elements kept, at its end.
fn settle_left(module: &Module) -> &'static str {
    if module.accounted() {
        "settle_left();"
    } else {
        ""
    }
}
