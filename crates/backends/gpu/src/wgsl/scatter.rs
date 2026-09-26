//! A scatter's passes: the pairs an element sends, how the target
//! takes them (atomically, by winners, by sorting keys), and the fold
//! that combines what landed on one place.

use super::*;

/// Binding index of the scatter pass's pairs, the map pass's output.
pub const SCATTER_PAIRS_BINDING: u32 = 1;
/// Binding index of the scatter pass's target, a copy of the base input.
pub const SCATTER_OUT_BINDING: u32 = 2;
/// Binding index of a whole sum's high words, one beside each word of
/// the target ([`sums_whole`]).
pub const SCATTER_HIGHS_BINDING: u32 = 6;

/// The high words a whole sum keeps beside its target, bound at `binding`
/// as `name`.
pub(crate) fn highs_binding(binding: u32, name: &str) -> String {
    format!("@group(0) @binding({binding}) var<storage, read_write> {name}: array<atomic<i32>>;\n")
}

/// The number of elements of a scatter's target, from the context.
fn target_count(module: &Module) -> String {
    let Mode::Scatter { base, .. } = module.manifest.mode else {
        return "0u".to_owned();
    };
    let slot = CONTEXT_HEAD + base as usize;
    format!("(u32(ctx[{slot}].x) * u32(ctx[{slot}].y) * u32(ctx[{slot}].z))")
}

/// The pass that sets a whole sum's high words from the base, before any
/// value is added: each the sign of the word below it, so a word and its
/// high word are the element's 64-bit total.
pub fn emit_scatter_highs(module: &Module) -> String {
    let Mode::Scatter { base, .. } = module.manifest.mode else {
        return String::new();
    };
    let stride = module.manifest.inputs[base as usize]
        .ty
        .schema
        .value_words();
    let context_len = CONTEXT_HEAD + module.manifest.inputs.len();
    format!(
        "@group(0) @binding({CONTEXT_BINDING}) var<uniform> ctx: array<vec4<f32>, {context_len}>;
@group(0) @binding({SCATTER_PAIRS_BINDING}) var<storage, read> base: array<u32>;
@group(0) @binding({SCATTER_HIGHS_BINDING}) var<storage, read_write> highs: array<i32>;
@compute @workgroup_size({SCATTER_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x + gid.y * {SCATTER_SPAN}u;
    if (i >= {} * {stride}u) {{
        return;
    }}
    highs[i] = select(0i, -1i, bitcast<i32>(base[i]) < 0i);
}}
",
        target_count(module)
    )
}

/// The pass that checks a whole sum once every value is in: an element
/// whose 64-bit total is no `Int` is recorded, the least one kept.
pub fn emit_scatter_totals(module: &Module) -> String {
    let Mode::Scatter { base, .. } = module.manifest.mode else {
        return String::new();
    };
    let stride = module.manifest.inputs[base as usize]
        .ty
        .schema
        .value_words();
    let context_len = CONTEXT_HEAD + module.manifest.inputs.len();
    format!(
        "@group(0) @binding({CONTEXT_BINDING}) var<uniform> ctx: array<vec4<f32>, {context_len}>;
@group(0) @binding({SCATTER_OUT_BINDING}) var<storage, read> out0: array<u32>;
@group(0) @binding({SCATTER_HIGHS_BINDING}) var<storage, read> highs: array<i32>;
{FAILURE_STRUCT}
@group(0) @binding({FAILURE_BINDING}) var<storage, read_write> failure: Failure;
@compute @workgroup_size({SCATTER_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let at = gid.x + gid.y * {SCATTER_SPAN}u;
    if (at >= {}) {{
        return;
    }}
    for (var k = 0u; k < {stride}u; k = k + 1u) {{
        let low = out0[at * {stride}u + k];
        let high = highs[at * {stride}u + k];
        if (!((high == 0i && low < 0x80000000u) || (high == -1i && low >= 0x80000000u))) {{
            atomicMax(&failure.total, 0xFFFFFFFFu - at);
            return;
        }}
    }}
}}
",
        target_count(module)
    )
}

/// The combine of two values of a type, as a WGSL expression: fieldwise
/// through structs, WGSL's own operators and builtins below.
pub(crate) fn combine_expr(
    module: &Module,
    ty: Type,
    combine: Combine,
    a: &str,
    b: &str,
) -> String {
    match ty {
        Type::Struct(index) => {
            let fields = module.structs[index as usize]
                .fields
                .iter()
                .enumerate()
                .map(|(k, field)| {
                    combine_expr(
                        module,
                        field.ty,
                        combine,
                        &format!("{a}{}", field_path(module, index, k)),
                        &format!("{b}{}", field_path(module, index, k)),
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("S{index}({fields})")
        }
        _ if ty.order().is_some() => match combine {
            Combine::Sum => format!("({a} + {b})"),
            Combine::Min => per_column(ty, |c| format!("min({a}[{c}], {b}[{c}])")),
            Combine::Max => per_column(ty, |c| format!("max({a}[{c}], {b}[{c}])")),
            Combine::Last => b.to_owned(),
        },
        _ => match combine {
            Combine::Sum => format!("({a} + {b})"),
            Combine::Min => format!("min({a}, {b})"),
            Combine::Max => format!("max({a}, {b})"),
            Combine::Last => b.to_owned(),
        },
    }
}

/// Binding index of the scatter pass's winners, for `last`: one word per
/// target element, the greatest source index plus one, zero for none.
pub const SCATTER_WINNER_BINDING: u32 = 3;
/// Workgroup length of the parallel scatter passes.
pub const SCATTER_WORKGROUP: u32 = 64;
/// Threads a row of scatter workgroups covers; see [`folded`].
pub const SCATTER_SPAN: u32 = MAX_GROUPS * SCATTER_WORKGROUP;

/// How a scatter's combine runs on the GPU: order-free rules apply the
/// pairs in parallel with atomics; a float sum, whose rounding is the
/// element order, sorts the pairs by target and source and folds each
/// target's run in that order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScatterPlan {
    /// The pairs sorted by address then source index, then one invocation
    /// per target element folding its run in element order.
    Sorted,
    /// One invocation per pair, atomics on the target; `last` records the
    /// winning source and gathers afterwards; `keys` says float words are
    /// turned into ordered keys around the pass.
    Parallel {
        /// The rule is `last`.
        last: bool,
        /// Float words need keys for `min` and `max`.
        keys: bool,
    },
}

/// Whether a scatter lands `Long`s under `min` or `max`: every scalar of
/// its target a `Long`, taken whole by 64-bit atomics on their keys (the
/// sign bit flipped, so the unsigned order is the signed one).
pub fn long_extremes(module: &Module) -> bool {
    let Mode::Scatter { base, combine } = module.manifest.mode else {
        return false;
    };
    let scalars = module.manifest.inputs[base as usize].ty.schema.scalars();
    matches!(combine, Combine::Min | Combine::Max)
        && !scalars.is_empty()
        && scalars.iter().all(|s| matches!(s, orla_abi::Scalar::Long))
}

/// The atomic array a scatter's target is bound as: words, or whole
/// longs where [`long_extremes`] lands them.
pub(crate) fn atomic_target(module: &Module) -> &'static str {
    if long_extremes(module) {
        "array<atomic<u64>>"
    } else {
        "array<atomic<i32>>"
    }
}

/// The plan for a scatter module.
pub fn scatter_plan(module: &Module) -> ScatterPlan {
    let Mode::Scatter { base, combine } = module.manifest.mode else {
        return ScatterPlan::Sorted;
    };
    let scalars = module.manifest.inputs[base as usize].ty.schema.scalars();
    let floats = scalars.iter().any(|s| matches!(s, orla_abi::Scalar::Float));
    match combine {
        Combine::Sum if floats => ScatterPlan::Sorted,
        Combine::Sum => ScatterPlan::Parallel {
            last: false,
            keys: false,
        },
        Combine::Min | Combine::Max => ScatterPlan::Parallel {
            last: false,
            keys: floats || long_extremes(module),
        },
        Combine::Last => ScatterPlan::Parallel {
            last: true,
            keys: false,
        },
    }
}

/// The words of a value as WGSL `u32` expressions, one per word in layout
/// order, each with its scalar type.
pub(crate) fn value_words(module: &Module, ty: Type, path: &str, out: &mut Vec<(String, Type)>) {
    let mut scalars = Vec::new();
    value_scalars(module, ty, path, &mut scalars);
    for (expr, kind) in scalars {
        for word in stored_words(kind, &expr) {
            out.push((word, Type::Scalar(kind)));
        }
    }
}

/// The scalars of a value as WGSL expressions, in layout order.
pub(crate) fn value_scalars(
    module: &Module,
    ty: Type,
    path: &str,
    out: &mut Vec<(String, Scalar)>,
) {
    match ty {
        Type::Struct(index) => {
            for (k, field) in module.structs[index as usize].fields.iter().enumerate() {
                let path = format!("{path}{}", field_path(module, index, k));
                value_scalars(module, field.ty, &path, out);
            }
        }
        other => out.extend(scalar_parts(other, path)),
    }
}

/// The header the scatter passes share: the context and the pairs, the
/// target when `target` says so (as atomics when `Some(true)`), and any
/// further bindings in `extra`; then the structs, the float key, and the
/// pair loader.
pub(crate) fn scatter_header(module: &Module, target: Option<bool>, extra: &str) -> String {
    let manifest = &module.manifest;
    let context_len = CONTEXT_HEAD + manifest.inputs.len();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "@group(0) @binding({CONTEXT_BINDING}) var<uniform> ctx: array<vec4<f32>, {context_len}>;"
    );
    let _ = writeln!(
        out,
        "@group(0) @binding({SCATTER_PAIRS_BINDING}) var<storage, read> pairs: array<u32>;"
    );
    if let Some(atomic) = target {
        let ty = if atomic {
            atomic_target(module)
        } else {
            "array<u32>"
        };
        let _ = writeln!(
            out,
            "@group(0) @binding({SCATTER_OUT_BINDING}) var<storage, read_write> out0: {ty};"
        );
    }
    out.push_str(extra);
    for index in 0..module.structs.len() {
        let _ = writeln!(out, "{}", struct_decl(module, index));
    }
    // Pairs of longs are loaded, and landed, through the long helpers.
    if long_extremes(module) {
        out.push_str(LONG_WORDS);
    }
    // The total order over floats as keys, an involution on the bits:
    // every negative below every positive, NaN beyond the infinities.
    let _ = writeln!(
        out,
        "fn key_of(b: i32) -> i32 {{ return select(b ^ 0x7fffffff, b, b >= 0); }}"
    );
    let pair = module.entry.result_type();
    let pair_stride = type_words(module, pair);
    let (expr, _) = load_expr(module, pair, "pairs", "b", 0, Encoding::Word);
    let _ = writeln!(
        out,
        "fn load_pair(index: u32) -> {} {{
    let b = index * {pair_stride}u;
    return {expr};
}}",
        wgsl_type(pair)
    );
    out
}

/// The winners of `last`, one word per target element.
pub(crate) fn winners_binding(read_only: bool) -> String {
    let (access, ty) = if read_only {
        ("read", "array<u32>")
    } else {
        ("read_write", "array<atomic<u32>>")
    };
    format!("@group(0) @binding({SCATTER_WINNER_BINDING}) var<storage, {access}> winner: {ty};\n")
}

/// Binding index of the sort's keys: one target index per source, the
/// greatest value for a source that lands outside or pads the sort.
pub const SCATTER_SORT_KEYS_BINDING: u32 = 3;
/// Binding index of the sort's order: the source index each key came from.
pub const SCATTER_SORT_ORDER_BINDING: u32 = 4;
/// Binding index of the sort's parameters, a `vec4<u32>` of the padded
/// count and the step's `k` and `j`, one record per dispatch at a dynamic
/// offset.
pub const SCATTER_SORT_PARAMS_BINDING: u32 = 5;

/// The sort's bindings.
pub(crate) fn sort_bindings(read_only: bool) -> String {
    let access = if read_only { "read" } else { "read_write" };
    format!(
        "@group(0) @binding({SCATTER_SORT_KEYS_BINDING}) var<storage, {access}> keys: array<u32>;
@group(0) @binding({SCATTER_SORT_ORDER_BINDING}) var<storage, {access}> order: array<u32>;
@group(0) @binding({SCATTER_SORT_PARAMS_BINDING}) var<uniform> sort: vec4<u32>;
"
    )
}

/// The lines that turn a pair's address into `at`, leaving the invocation
/// with `exit` when it falls outside the target.
pub(crate) fn scatter_address(module: &Module, exit: &str) -> String {
    let Mode::Scatter { base, .. } = module.manifest.mode else {
        return String::new();
    };
    let target = &module.manifest.inputs[base as usize];
    let slot = CONTEXT_HEAD + base as usize;
    // A pair is its address, then its value.
    let address = format!("p.{}", member_name(0));
    match target.shape() {
        // Named `row`, not `i`: the pass's own `i` is the source index.
        Shape::Rows => format!(
            "let row = {address};
    if (row < 0i || row >= i32(ctx[{slot}].x)) {{ {exit} }}
    let at = u32(row);"
        ),
        Shape::Grid => format!(
            "let x = floor({address}.x);
    let y = floor({address}.y);
    if (!(x >= 0.0 && x < ctx[{slot}].x && y >= 0.0 && y < ctx[{slot}].y)) {{ {exit} }}
    let at = u32(y) * u32(ctx[{slot}].x) + u32(x);"
        ),
        Shape::Volume => format!(
            "let x = floor({address}.x);
    let y = floor({address}.y);
    let z = floor({address}.z);
    if (!(x >= 0.0 && x < ctx[{slot}].x && y >= 0.0 && y < ctx[{slot}].y && z >= 0.0 && z < ctx[{slot}].z)) {{ {exit} }}
    let at = (u32(z) * u32(ctx[{slot}].y) + u32(y)) * u32(ctx[{slot}].x) + u32(x);"
        ),
    }
}

/// Emits the parallel apply pass: one invocation per pair, combining its
/// value into the target with atomics (`sum` of ints, `min` and `max` on
/// ordered keys) or recording the greatest source index per element
/// (`last`). Order-free, so the result is the sequential one.
pub fn emit_scatter_parallel(module: &Module) -> String {
    let Mode::Scatter { base, combine } = module.manifest.mode else {
        return String::new();
    };
    let target = &module.manifest.inputs[base as usize];
    let element = target.element();
    let stride = target.ty.schema.value_words();
    let mut extra = winners_binding(false);
    if highs_beside(module) {
        extra.push_str(&highs_binding(SCATTER_HIGHS_BINDING, "highs"));
    }
    let mut out = scatter_header(module, Some(true), &extra);
    let lines = match combine {
        Combine::Last => vec!["    atomicMax(&winner[at], i + 1u);".to_owned()],
        _ => atomic_lines(module, element, combine, "out0", "highs"),
    };
    let _ = writeln!(
        out,
        "
@compute @workgroup_size({SCATTER_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let n = u32(ctx[0].x) * u32(ctx[0].y) * u32(ctx[1].z);
    let i = gid.x + gid.y * {SCATTER_SPAN}u;
    if (i >= n) {{
        return;
    }}
    let p = load_pair(i);
    {}
    let b = at * {stride}u;
{}
}}",
        scatter_address(module, "return;"),
        lines.join("\n")
    );
    out
}

/// The atomics that combine a pair's value `p.value` into the element of
/// `target` at word `b`, under an order-free rule: `sum` adds, `min` and
/// `max` compare, floats as their ordered keys, longs whole as theirs.
pub(crate) fn atomic_lines(
    module: &Module,
    element: Type,
    combine: Combine,
    target: &str,
    highs: &str,
) -> Vec<String> {
    let op = match combine {
        Combine::Sum => "atomicAdd",
        Combine::Min => "atomicMin",
        _ => "atomicMax",
    };
    let path = format!("p.{}", member_name(1));
    if long_extremes(module) {
        let mut scalars = Vec::new();
        value_scalars(module, element, &path, &mut scalars);
        return scalars
            .iter()
            .enumerate()
            .map(|(k, (expr, _))| {
                format!(
                    "    {op}(&{target}[b / 2u + {k}u], bitcast<u64>({expr}) ^ 0x8000000000000000lu);"
                )
            })
            .collect();
    }
    let mut words = Vec::new();
    value_words(module, element, &path, &mut words);
    // A whole sum: the word added atomically, and what it carried out of
    // the word, with the value's sign, added to the high word beside it.
    if highs_beside(module) {
        return words
            .iter()
            .enumerate()
            .map(|(k, (expr, _))| {
                format!(
                    "    {{
        let v = bitcast<u32>({expr});
        let old = bitcast<u32>(atomicAdd(&{target}[b + {k}u], bitcast<i32>(v)));
        let carry = select(0i, 1i, old + v < old);
        let high = carry + select(0i, -1i, bitcast<i32>(v) < 0i);
        if (high != 0i) {{
            atomicAdd(&{highs}[b + {k}u], high);
        }}
    }}"
                )
            })
            .collect();
    }
    words
        .iter()
        .enumerate()
        .map(|(k, (expr, ty))| {
            let key = if *ty == Type::Float && combine != Combine::Sum {
                format!("key_of(bitcast<i32>({expr}))")
            } else {
                format!("bitcast<i32>({expr})")
            };
            format!("    {op}(&{target}[b + {k}u], {key});")
        })
        .collect()
}

/// Binding index of the target a scatter's `emit` combines into as it
/// runs: the base copied, float words as keys, bound as atomics.
pub const EMIT_BINDING: u32 = 89;

/// The pair types a scatter's entry emits, each once; none when it
/// emits nothing.
pub fn emitted_pairs(module: &Module) -> Vec<Type> {
    fn walk(instructions: &[Instruction], found: &mut Vec<Type>) {
        for instruction in instructions {
            match &instruction.op {
                Op::Emit(_) if !found.contains(&instruction.ty) => found.push(instruction.ty),
                op => {
                    for block in op.blocks() {
                        walk(&block.body, found);
                    }
                }
            }
        }
    }
    let mut found = Vec::new();
    walk(&module.entry.body, &mut found);
    found
}

/// The target binding and one function a pair type that `emit` gives
/// the target at once: the address checked as the apply pass checks it,
/// then the value's words combined with atomics.
pub(crate) fn emit_functions(module: &Module, out: &mut String) {
    let Mode::Scatter { base, combine } = module.manifest.mode else {
        return;
    };
    let mut pairs = emitted_pairs(module);
    if pairs.is_empty() {
        return;
    }
    // The pair the entry returns is landed the same way.
    let returned = module.entry.result_type();
    if !pairs.contains(&returned) {
        pairs.push(returned);
    }
    let target = &module.manifest.inputs[base as usize];
    let stride = target.ty.schema.value_words();
    let _ = writeln!(
        out,
        "@group(0) @binding({EMIT_BINDING}) var<storage, read_write> emitted: {};
fn key_of(b: i32) -> i32 {{ return select(b ^ 0x7fffffff, b, b >= 0); }}",
        atomic_target(module)
    );
    if highs_beside(module) {
        out.push_str(&highs_binding(EMIT_HIGHS_BINDING, "emitted_highs"));
    }
    for pair in pairs {
        let _ = writeln!(
            out,
            "fn {}(p: {}) {{
    {}
    let b = at * {stride}u;
{}
}}",
            emit_name(pair),
            wgsl_type(pair),
            scatter_address(module, "return;"),
            atomic_lines(
                module,
                target.element(),
                combine,
                "emitted",
                "emitted_highs"
            )
            .join("\n")
        );
    }
}

/// The name of the function that emits a pair of a type.
pub(crate) fn emit_name(pair: Type) -> String {
    match pair {
        Type::Struct(index) => format!("emit_s{index}"),
        _ => "emit_pair".to_owned(),
    }
}

/// Emits the gather pass of `last`: one invocation per target element,
/// taking the winning pair's value when there is one.
pub fn emit_scatter_gather(module: &Module) -> String {
    let Mode::Scatter { base, .. } = module.manifest.mode else {
        return String::new();
    };
    let target = &module.manifest.inputs[base as usize];
    let stride = target.ty.schema.value_words();
    let slot = CONTEXT_HEAD + base as usize;
    let mut out = scatter_header(module, Some(false), &winners_binding(true));
    let mut words = Vec::new();
    value_words(
        module,
        target.element(),
        &format!("p.{}", member_name(1)),
        &mut words,
    );
    let stores = words
        .iter()
        .enumerate()
        .map(|(k, (expr, _))| format!("    out0[b + {k}u] = {expr};"))
        .collect::<Vec<_>>()
        .join("\n");
    let _ = writeln!(
        out,
        "
@compute @workgroup_size({SCATTER_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let count = u32(ctx[{slot}].x) * u32(ctx[{slot}].y) * u32(ctx[{slot}].z);
    let e = gid.x + gid.y * {SCATTER_SPAN}u;
    if (e >= count) {{
        return;
    }}
    let w = winner[e];
    if (w == 0u) {{
        return;
    }}
    let p = load_pair(w - 1u);
    let b = e * {stride}u;
{stores}
}}"
    );
    out
}

/// Emits the pass that turns the target's float words into ordered keys
/// or back, around a parallel `min` or `max`: in place, or, `from_base`,
/// read from the base bound where the pairs are and written whole into
/// the target, which then needs no copy of the base first.
pub fn emit_scatter_keys(module: &Module, from_base: bool) -> String {
    let Mode::Scatter { base, .. } = module.manifest.mode else {
        return String::new();
    };
    let target = &module.manifest.inputs[base as usize];
    let stride = target.ty.schema.value_words();
    let slot = CONTEXT_HEAD + base as usize;
    let mut out = scatter_header(module, Some(false), "");
    // The key transform is an involution, so one function serves both ways.
    // A float's key is `key_of` of its word; a long's is its sign bit
    // flipped, in its high word.
    // The schema lists a scalar per word, a long twice.
    let from = if from_base { "pairs" } else { "out0" };
    let mut lines = Vec::new();
    let scalars = target.ty.schema.scalars();
    let mut k = 0;
    while let Some(scalar) = scalars.get(k) {
        match scalar {
            orla_abi::Scalar::Float => lines.push(format!(
                "    out0[b + {k}u] = bitcast<u32>(key_of(bitcast<i32>({from}[b + {k}u])));"
            )),
            orla_abi::Scalar::Long => {
                if from_base {
                    lines.push(format!("    out0[b + {k}u] = pairs[b + {k}u];"));
                }
                lines.push(format!(
                    "    out0[b + {high}u] = {from}[b + {high}u] ^ 0x80000000u;",
                    high = k + 1
                ));
            }
            _ if from_base => {
                for word in k..k + scalar.words() {
                    lines.push(format!("    out0[b + {word}u] = pairs[b + {word}u];"));
                }
            }
            _ => {}
        }
        k += scalar.words();
    }
    let lines = lines.join("\n");
    let _ = writeln!(
        out,
        "
@compute @workgroup_size({SCATTER_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let count = u32(ctx[{slot}].x) * u32(ctx[{slot}].y) * u32(ctx[{slot}].z);
    let e = gid.x + gid.y * {SCATTER_SPAN}u;
    if (e >= count) {{
        return;
    }}
    let b = e * {stride}u;
{lines}
}}"
    );
    out
}

/// Emits the sort's first pass: one invocation per slot of the padded
/// sort, writing a source's target index as its key and its index as its
/// order, the greatest key for a source outside the target or a pad.
pub fn emit_scatter_sort_keys(module: &Module) -> String {
    let mut out = scatter_header(module, None, &sort_bindings(false));
    let _ = writeln!(
        out,
        "
@compute @workgroup_size({SCATTER_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x + gid.y * {SCATTER_SPAN}u;
    if (i >= sort.x) {{
        return;
    }}
    let n = u32(ctx[0].x) * u32(ctx[0].y) * u32(ctx[1].z);
    order[i] = i;
    if (i >= n) {{
        keys[i] = 0xffffffffu;
        return;
    }}
    let p = load_pair(i);
    {}
    keys[i] = at;
}}",
        scatter_address(module, "keys[i] = 0xffffffffu; return;")
    );
    out
}

/// Emits one step of the bitonic sort over (key, order): the same shader
/// for every `k` and `j`, which the parameters carry.
pub fn emit_scatter_sort_step() -> String {
    format!(
        "{}
@compute @workgroup_size({SCATTER_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x + gid.y * {SCATTER_SPAN}u;
    let partner = i ^ sort.z;
    if (i >= sort.x || partner <= i) {{
        return;
    }}
    let ascending = (i & sort.y) == 0u;
    let ka = keys[i];
    let kb = keys[partner];
    let oa = order[i];
    let ob = order[partner];
    let a_greater = ka > kb || (ka == kb && oa > ob);
    if (ascending == a_greater) {{
        keys[i] = kb;
        keys[partner] = ka;
        order[i] = ob;
        order[partner] = oa;
    }}
}}",
        sort_bindings(false)
    )
}

/// Emits the fold: one invocation per target element finds its run in the
/// sorted keys and adds the pairs' values to the element in source order,
/// exactly as the CPU walks them.
pub fn emit_scatter_fold(module: &Module) -> String {
    let Mode::Scatter { base, combine } = module.manifest.mode else {
        return String::new();
    };
    let target = &module.manifest.inputs[base as usize];
    let element = target.element();
    let stride = target.ty.schema.value_words();
    let slot = CONTEXT_HEAD + base as usize;
    let mut bindings = sort_bindings(true);
    // Its integers folded whole: a high word each, carried as it folds,
    // and the first element whose total does not fit recorded.
    let mut integers = Vec::new();
    if sums_whole(module) {
        let _ = writeln!(
            bindings,
            "{FAILURE_STRUCT}\n@group(0) @binding({FAILURE_BINDING}) var<storage, read_write> failure: Failure;"
        );
        let mut scalars = Vec::new();
        value_scalars(module, element, "", &mut scalars);
        integers = scalars
            .into_iter()
            .filter(|(_, kind)| *kind == Scalar::Int)
            .map(|(path, _)| path)
            .collect();
    }
    let highs_start: String = integers
        .iter()
        .enumerate()
        .map(|(k, path)| format!("    var high{k} = select(0i, -1i, acc{path} < 0i);\n"))
        .collect();
    let highs_step: String = integers
        .iter()
        .enumerate()
        .map(|(k, path)| {
            format!(
                "        {{ let a = bitcast<u32>(acc{path}); let b = bitcast<u32>(v{path}); high{k} = high{k} + select(0i, 1i, a + b < a) + select(0i, -1i, v{path} < 0i); }}\n"
            )
        })
        .collect();
    let highs_check: String = integers
        .iter()
        .enumerate()
        .map(|(k, path)| {
            format!(
                "    if (!((high{k} == 0i && acc{path} >= 0i) || (high{k} == -1i && acc{path} < 0i))) {{ atomicMax(&failure.total, 0xFFFFFFFFu - e); }}\n"
            )
        })
        .collect();
    let mut out = scatter_header(module, Some(false), &bindings);
    let (expr, _) = load_expr(module, element, "out0", "b", 0, Encoding::Word);
    let _ = writeln!(
        out,
        "fn load_out0(index: u32) -> {} {{
    let b = index * {stride}u;
    return {expr};
}}",
        wgsl_type(element)
    );
    let mut stores = Vec::new();
    store_lines(module, element, "v", 0, &mut stores);
    let _ = writeln!(
        out,
        "fn store_out0(index: u32, v: {}) {{
    let b = index * {stride}u;
{}
}}",
        wgsl_type(element),
        stores.join("\n")
    );
    let _ = writeln!(
        out,
        "fn combine(a: {ty}, b: {ty}) -> {ty} {{ return {}; }}",
        combine_expr(module, element, combine, "a", "b"),
        ty = wgsl_type(element)
    );
    let _ = writeln!(
        out,
        "
@compute @workgroup_size({SCATTER_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let count = u32(ctx[{slot}].x) * u32(ctx[{slot}].y) * u32(ctx[{slot}].z);
    let e = gid.x + gid.y * {SCATTER_SPAN}u;
    if (e >= count) {{
        return;
    }}
    let n = sort.x;
    var lo = 0u;
    var hi = n;
    while (lo < hi) {{
        let mid = (lo + hi) / 2u;
        if (keys[mid] < e) {{
            lo = mid + 1u;
        }} else {{
            hi = mid;
        }}
    }}
    if (lo >= n || keys[lo] != e) {{
        return;
    }}
    var acc = load_out0(e);
{highs_start}    var pos = lo;
    loop {{
        if (pos >= n || keys[pos] != e) {{
            break;
        }}
        let v = load_pair(order[pos]).{value};
{highs_step}        acc = combine(acc, v);
        pos = pos + 1u;
    }}
{highs_check}    store_out0(e, acc);
}}",
        value = member_name(1),
    );
    out
}
