//! A scan run as a prefix over its elements: the per-block sums, their
//! prefix, and the pass that applies it.

use super::*;

/// The prefix passes' bindings: the plan, the output being folded in
/// place, and one element a block.
pub const PREFIX_PLAN_BINDING: u32 = 0;
/// The output buffer the fold walks in place.
pub const PREFIX_OUT_BINDING: u32 = 1;
/// One element a block: each block's total, then those scanned.
pub const PREFIX_SUMS_BINDING: u32 = 2;
/// Binding index of a whole sum's high words beside the output.
pub const PREFIX_OUT_HIGHS_BINDING: u32 = 3;
/// Binding index of a whole sum's high words beside the blocks' totals.
pub const PREFIX_SUMS_HIGHS_BINDING: u32 = 4;
/// Invocations a workgroup in the prefix passes.
pub const PREFIX_WORKGROUP: u32 = 64;
/// Threads a row of prefix workgroups covers; see [`folded`].
pub const PREFIX_SPAN: u32 = MAX_GROUPS * PREFIX_WORKGROUP;

/// One of the three passes a prefix fold takes on a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixPass {
    /// One invocation a block: walk the block in place, leave its total.
    Blocks,
    /// One invocation: scan the blocks' totals.
    Tops,
    /// One invocation an element: take the block's offset.
    Offsets,
}

/// How two words meet under `combine`, by the word's scalar — the same
/// rule as `orla_rules::combine_words`, which is what holds the device
/// to the reference's words. An integer `sum` is never folded here: it
/// is kept whole ([`sums_whole`]).
pub(crate) fn prefix_word(combine: Combine, scalar: orla_abi::Scalar, a: &str, b: &str) -> String {
    let float = matches!(scalar, orla_abi::Scalar::Float);
    match (combine, float) {
        (Combine::Sum, false) => {
            format!("bitcast<u32>(bitcast<i32>({a}) + bitcast<i32>({b}))")
        }
        (Combine::Sum, true) => {
            format!("bitcast<u32>(bitcast<f32>({a}) + bitcast<f32>({b}))")
        }
        // Floats meet by their bits' total order, as they do in a
        // scatter, so a not-a-number lands where the reference puts it.
        (Combine::Min, true) => {
            format!("select({a}, {b}, key_of(bitcast<i32>({b})) < key_of(bitcast<i32>({a})))")
        }
        (Combine::Max, true) => {
            format!("select({a}, {b}, key_of(bitcast<i32>({b})) > key_of(bitcast<i32>({a})))")
        }
        (Combine::Min, false) => {
            format!("bitcast<u32>(min(bitcast<i32>({a}), bitcast<i32>({b})))")
        }
        (Combine::Max, false) => {
            format!("bitcast<u32>(max(bitcast<i32>({a}), bitcast<i32>({b})))")
        }
        (Combine::Last, _) => b.to_owned(),
    }
}

/// Emits one of a prefix fold's three passes. The output is walked in
/// place as words, so the element may be anything the layout holds, and
/// nothing here needs shared memory or a barrier: a block is one
/// invocation's walk, the blocks' totals are another's, and the third
/// hands every element its block's offset.
pub fn emit_prefix(module: &Module, pass: PrefixPass) -> String {
    let Mode::Prefix { combine } = module.manifest.mode else {
        return String::new();
    };
    if sums_whole(module) {
        return emit_prefix_whole(module, pass);
    }
    let port = &module.manifest.outputs[0];
    let scalars = port.ty.schema.scalars().to_vec();
    let stride = scalars.len().max(1);
    let mut out = String::new();
    let _ = writeln!(
        out,
        "@group(0) @binding({PREFIX_PLAN_BINDING}) var<uniform> plan: vec4<u32>;
@group(0) @binding({PREFIX_OUT_BINDING}) var<storage, read_write> out0: array<u32>;
@group(0) @binding({PREFIX_SUMS_BINDING}) var<storage, read_write> sums: array<u32>;
const STRIDE: u32 = {stride}u;
fn key_of(b: i32) -> i32 {{ return select(b ^ 0x7fffffff, b, b >= 0); }}"
    );
    let fold = (0..stride)
        .map(|k| {
            format!(
                "    r[{k}u] = {};",
                prefix_word(
                    combine,
                    scalars[k],
                    &format!("a[{k}u]"),
                    &format!("b[{k}u]")
                )
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let load = |from: &str| {
        (0..stride)
            .map(|k| format!("    v[{k}u] = {from}[b + {k}u];"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let store = |into: &str| {
        (0..stride)
            .map(|k| format!("    {into}[b + {k}u] = v[{k}u];"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let _ = writeln!(
        out,
        "alias Element = array<u32, {stride}u>;
fn fold(a: Element, b: Element) -> Element {{
    var r: Element;
{fold}
    return r;
}}
fn load_out(index: u32) -> Element {{
    let b = index * STRIDE;
    var v: Element;
{}
    return v;
}}
fn store_out(index: u32, v: Element) {{
    let b = index * STRIDE;
{}
}}
fn load_sum(index: u32) -> Element {{
    let b = index * STRIDE;
    var v: Element;
{}
    return v;
}}
fn store_sum(index: u32, v: Element) {{
    let b = index * STRIDE;
{}
}}",
        load("out0"),
        store("out0"),
        load("sums"),
        store("sums")
    );
    // plan.x: elements, plan.y: elements a block, plan.z: blocks.
    let body = match pass {
        PrefixPass::Blocks => format!(
            "
@compute @workgroup_size({PREFIX_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let block = gid.x + gid.y * {PREFIX_SPAN}u;
    if (block >= plan.z) {{
        return;
    }}
    let start = block * plan.y;
    let end = min(start + plan.y, plan.x);
    var running = load_out(start);
    for (var i = start + 1u; i < end; i = i + 1u) {{
        running = fold(running, load_out(i));
        store_out(i, running);
    }}
    store_sum(block, running);
}}"
        ),
        PrefixPass::Tops => "
@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }
    var running = load_sum(0u);
    for (var i = 1u; i < plan.z; i = i + 1u) {
        running = fold(running, load_sum(i));
        store_sum(i, running);
    }
}"
        .to_owned(),
        PrefixPass::Offsets => format!(
            "
@compute @workgroup_size({PREFIX_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x + gid.y * {PREFIX_SPAN}u;
    if (i >= plan.x) {{
        return;
    }}
    let block = i / plan.y;
    if (block == 0u) {{
        return;
    }}
    store_out(i, fold(load_sum(block - 1u), load_out(i)));
}}"
        ),
    };
    out.push_str(&body);
    out.push('\n');
    out
}

/// A prefix `sum` of integers, each running total whole: a word and a
/// high word beside it, 64 bits, walked and offset as any prefix is, and
/// every element's total checked once it is final. An element whose
/// total is no `Int` is recorded, the least one kept: the first prefix
/// that does not fit, which is where the reference's walk fails.
fn emit_prefix_whole(module: &Module, pass: PrefixPass) -> String {
    let stride = module.manifest.outputs[0].ty.schema.value_words().max(1);
    let mut out = format!(
        "@group(0) @binding({PREFIX_PLAN_BINDING}) var<uniform> plan: vec4<u32>;
@group(0) @binding({PREFIX_OUT_BINDING}) var<storage, read_write> out0: array<u32>;
@group(0) @binding({PREFIX_SUMS_BINDING}) var<storage, read_write> sums: array<u32>;
@group(0) @binding({PREFIX_OUT_HIGHS_BINDING}) var<storage, read_write> out_highs: array<i32>;
@group(0) @binding({PREFIX_SUMS_HIGHS_BINDING}) var<storage, read_write> sum_highs: array<i32>;
{FAILURE_STRUCT}
@group(0) @binding({FAILURE_BINDING}) var<storage, read_write> failure: Failure;
const STRIDE: u32 = {stride}u;
alias Words = array<u32, {stride}u>;
alias Highs = array<i32, {stride}u>;
struct Whole {{ low: Words, high: Highs }}
// The high words of words as they are: each its word's sign.
fn widened(low: Words) -> Whole {{
    var w: Whole;
    w.low = low;
    for (var k = 0u; k < STRIDE; k = k + 1u) {{
        w.high[k] = select(0i, -1i, bitcast<i32>(low[k]) < 0i);
    }}
    return w;
}}
fn add(a: Whole, b: Whole) -> Whole {{
    var r: Whole;
    for (var k = 0u; k < STRIDE; k = k + 1u) {{
        r.low[k] = a.low[k] + b.low[k];
        r.high[k] = a.high[k] + b.high[k] + select(0i, 1i, r.low[k] < a.low[k]);
    }}
    return r;
}}
fn fits(w: Whole) -> bool {{
    for (var k = 0u; k < STRIDE; k = k + 1u) {{
        let low = w.low[k];
        let high = w.high[k];
        if (!((high == 0i && low < 0x80000000u) || (high == -1i && low >= 0x80000000u))) {{
            return false;
        }}
    }}
    return true;
}}
fn load_out(index: u32) -> Whole {{
    var w: Whole;
    for (var k = 0u; k < STRIDE; k = k + 1u) {{
        w.low[k] = out0[index * STRIDE + k];
        w.high[k] = out_highs[index * STRIDE + k];
    }}
    return w;
}}
fn store_out(index: u32, w: Whole) {{
    for (var k = 0u; k < STRIDE; k = k + 1u) {{
        out0[index * STRIDE + k] = w.low[k];
        out_highs[index * STRIDE + k] = w.high[k];
    }}
}}
fn load_sum(index: u32) -> Whole {{
    var w: Whole;
    for (var k = 0u; k < STRIDE; k = k + 1u) {{
        w.low[k] = sums[index * STRIDE + k];
        w.high[k] = sum_highs[index * STRIDE + k];
    }}
    return w;
}}
fn store_sum(index: u32, w: Whole) {{
    for (var k = 0u; k < STRIDE; k = k + 1u) {{
        sums[index * STRIDE + k] = w.low[k];
        sum_highs[index * STRIDE + k] = w.high[k];
    }}
}}
fn value(index: u32) -> Whole {{
    var low: Words;
    for (var k = 0u; k < STRIDE; k = k + 1u) {{
        low[k] = out0[index * STRIDE + k];
    }}
    return widened(low);
}}
"
    );
    // plan.x: elements, plan.y: elements a block, plan.z: blocks.
    let body = match pass {
        PrefixPass::Blocks => format!(
            "
@compute @workgroup_size({PREFIX_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let block = gid.x + gid.y * {PREFIX_SPAN}u;
    if (block >= plan.z) {{
        return;
    }}
    let start = block * plan.y;
    let end = min(start + plan.y, plan.x);
    var running = value(start);
    store_out(start, running);
    for (var i = start + 1u; i < end; i = i + 1u) {{
        running = add(running, value(i));
        store_out(i, running);
    }}
    store_sum(block, running);
}}"
        ),
        PrefixPass::Tops => "
@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }
    var running = load_sum(0u);
    for (var i = 1u; i < plan.z; i = i + 1u) {
        running = add(running, load_sum(i));
        store_sum(i, running);
    }
}"
        .to_owned(),
        PrefixPass::Offsets => format!(
            "
@compute @workgroup_size({PREFIX_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x + gid.y * {PREFIX_SPAN}u;
    if (i >= plan.x) {{
        return;
    }}
    let block = i / plan.y;
    var total = load_out(i);
    if (block > 0u) {{
        total = add(load_sum(block - 1u), total);
        store_out(i, total);
    }}
    if (!fits(total)) {{
        atomicMax(&failure.total, 0xFFFFFFFFu - i);
    }}
}}"
        ),
    };
    out.push_str(&body);
    out.push('\n');
    out
}
