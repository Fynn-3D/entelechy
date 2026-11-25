//! A hierarchical, coalesced parallel prefix fold over blocks of 256 elements.
//!
//! Each workgroup of [`PREFIX_BLOCK`] invocations scans one contiguous block in
//! place and writes its total to `sums[block]`. Higher levels scan `sums` with
//! the exact same shader until one block remains, then [`PrefixPass::Offsets`]
//! propagates block totals back down the hierarchy.
//!
//! When hardware subgroups are available (`SubgroupCaps::wave_parallel_256`),
//! intra-wave scans run register-to-register via `subgroupInclusiveAdd` (float
//! sum) or `subgroupShuffleUp` (min/max/last and 64-bit whole integer sums),
//! leaving only 64–128 bytes of cross-wave workgroup storage and 2 barriers per
//! block. Without subgroups, the exact same 256-lane hierarchy runs through a
//! 1–2 KiB per-channel workgroup shared-memory Hillis-Steele scan.

use super::*;

/// Binding index of the prefix plan uniform (`[count, blocks, 0, flags]`).
pub const PREFIX_PLAN_BINDING: u32 = 0;
/// Binding index of the output, folded in place.
pub const PREFIX_OUT_BINDING: u32 = 1;
/// Binding index of the block totals, one element per block.
pub const PREFIX_SUMS_BINDING: u32 = 2;
/// Binding index of a whole sum's high words beside the output.
pub const PREFIX_OUT_HIGHS_BINDING: u32 = 3;
/// Binding index of a whole sum's high words beside the blocks' totals.
pub const PREFIX_SUMS_HIGHS_BINDING: u32 = 4;
/// Elements scanned by one workgroup in [`PrefixPass::Blocks`], and
/// invocations per workgroup in [`PrefixPass::Offsets`].
pub const PREFIX_BLOCK: u32 = 256;
/// Invocations per workgroup in the prefix passes.
pub const PREFIX_WORKGROUP: u32 = PREFIX_BLOCK;
/// Threads a row of prefix workgroups covers; see [`folded`].
pub const PREFIX_SPAN: u32 = MAX_GROUPS * PREFIX_WORKGROUP;
/// `plan.w` bit: level 0 of a whole integer sum widens raw 32-bit words from
/// `out0` instead of reading existing `(out0, out_highs)` pairs.
pub const PREFIX_FLAG_WIDEN: u32 = 1;
/// `plan.w` bit: final pass of a whole integer sum checks every element's
/// 64-bit total against 32-bit signed range and records the least overflow.
pub const PREFIX_FLAG_CHECK_FITS: u32 = 2;

/// One of a hierarchical prefix fold's two passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixPass {
    /// One workgroup per block of [`PREFIX_BLOCK`] elements: scan the block in
    /// place and write its final element to `sums[block]`.
    Blocks,
    /// One invocation per element: fold its block's preceding total from
    /// `sums[block - 1]` into `out0[i]` (and verify 32-bit range on level 0 of
    /// a whole integer sum).
    Offsets,
}

/// Two words combined under `combine` by their scalar, the rule of
/// `entelechy_rules::combine_words`. An integer `sum` is kept whole instead
/// ([`sums_whole`]).
pub(crate) fn prefix_word(
    combine: Combine,
    scalar: entelechy_abi::Scalar,
    a: &str,
    b: &str,
) -> String {
    let float = matches!(scalar, entelechy_abi::Scalar::Float);
    match (combine, float) {
        (Combine::Sum, false) => {
            format!("bitcast<u32>(bitcast<i32>({a}) + bitcast<i32>({b}))")
        }
        (Combine::Sum, true) => {
            format!("bitcast<u32>(bitcast<f32>({a}) + bitcast<f32>({b}))")
        }
        // Floats compare by their bits' total order, placing NaN as the
        // reference does.
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

/// Emits one of a prefix fold's passes without subgroup instructions.
pub fn emit_prefix(module: &Module, pass: PrefixPass) -> String {
    emit_prefix_with_subgroups(module, pass, SubgroupCaps::default())
}

/// Emits one of a prefix fold's passes using hardware subgroups when
/// `subgroups.wave_parallel_256()` holds.
pub fn emit_prefix_with_subgroups(
    module: &Module,
    pass: PrefixPass,
    subgroups: SubgroupCaps,
) -> String {
    let Mode::Prefix { combine } = module.manifest.mode else {
        return String::new();
    };
    let use_sg = subgroups.wave_parallel_256() && pass == PrefixPass::Blocks;
    let sg_directive = if use_sg { subgroups.directive() } else { "" };
    if sums_whole(module) {
        return emit_prefix_whole(module, pass, use_sg, sg_directive);
    }
    let port = &module.manifest.outputs[0];
    let scalars = port.ty.schema.scalars().to_vec();
    let stride = scalars.len().max(1);
    let mut out = String::new();
    out.push_str(sg_directive);
    let _ = writeln!(
        out,
        "@group(0) @binding({PREFIX_PLAN_BINDING}) var<uniform> plan: vec4<u32>;
@group(0) @binding({PREFIX_OUT_BINDING}) var<storage, read_write> out0: array<u32>;
@group(0) @binding({PREFIX_SUMS_BINDING}) var<storage, read_write> sums: array<u32>;
const STRIDE: u32 = {stride}u;
fn key_of(b: i32) -> i32 {{ return select(b ^ 0x7fffffff, b, b >= 0); }}"
    );
    let uniform_scalar = scalars
        .first()
        .copied()
        .filter(|first| scalars.iter().all(|s| s == first));
    match pass {
        PrefixPass::Blocks => {
            if use_sg {
                emit_blocks_subgroup(&mut out, combine, &scalars, uniform_scalar);
            } else {
                emit_blocks_workgroup(&mut out, combine, &scalars, uniform_scalar);
            }
        }
        PrefixPass::Offsets => {
            emit_offsets(&mut out, combine, &scalars, uniform_scalar);
        }
    }
    out
}

fn emit_channel_scan_subgroup(
    out: &mut String,
    combine: Combine,
    scalar: entelechy_abi::Scalar,
    k_expr: &str,
    trailing_barrier: bool,
) {
    let is_float_sum = combine == Combine::Sum && matches!(scalar, entelechy_abi::Scalar::Float);
    if is_float_sum {
        let _ = writeln!(
            out,
            "    {{
        var v: f32 = 0.0;
        if (in_bounds) {{ v = bitcast<f32>(out0[i * STRIDE + {k_expr}]); }}
        v = subgroupInclusiveAdd(v);
        if (sg_id == sg_size - 1u) {{ wave_scratch[wave] = bitcast<u32>(v); }}
        workgroupBarrier();
        if (num_waves <= sg_size) {{
            if (wave == 0u) {{
                var wv: f32 = 0.0;
                if (sg_id < num_waves) {{ wv = bitcast<f32>(wave_scratch[sg_id]); }}
                let w_inc = subgroupInclusiveAdd(wv);
                if (sg_id < num_waves) {{ wave_scratch[sg_id] = bitcast<u32>(w_inc); }}
            }}
        }} else if (lid == 0u) {{
            var acc: f32 = bitcast<f32>(wave_scratch[0u]);
            for (var w: u32 = 1u; w < num_waves; w = w + 1u) {{
                acc = acc + bitcast<f32>(wave_scratch[w]);
                wave_scratch[w] = bitcast<u32>(acc);
            }}
        }}
        workgroupBarrier();
        if (wave > 0u) {{ v = bitcast<f32>(wave_scratch[wave - 1u]) + v; }}
        let res = bitcast<u32>(v);
        if (in_bounds) {{ out0[i * STRIDE + {k_expr}] = res; }}
        if (lid == last_lid) {{ sums[block * STRIDE + {k_expr}] = res; }}
    }}"
        );
    } else {
        let fold_inner = prefix_word(combine, scalar, "prev", "cur");
        let fold_wave = prefix_word(combine, scalar, "wprev", "wcur");
        let fold_serial = prefix_word(combine, scalar, "acc", "wave_scratch[w]");
        let fold_add = prefix_word(combine, scalar, "wave_scratch[wave - 1u]", "cur");
        let _ = writeln!(
            out,
            "    {{
        var cur: u32 = 0u;
        if (in_bounds) {{ cur = out0[i * STRIDE + {k_expr}]; }}
        var d: u32 = 1u;
        loop {{
            if (d >= sg_size) {{ break; }}
            let prev = subgroupShuffleUp(cur, d);
            if (sg_id >= d) {{ cur = {fold_inner}; }}
            d = d << 1u;
        }}
        if (sg_id == sg_size - 1u) {{ wave_scratch[wave] = cur; }}
        workgroupBarrier();
        if (num_waves <= sg_size) {{
            if (wave == 0u) {{
                var wcur: u32 = 0u;
                if (sg_id < num_waves) {{ wcur = wave_scratch[sg_id]; }}
                var wd: u32 = 1u;
                loop {{
                    if (wd >= num_waves) {{ break; }}
                    let wprev = subgroupShuffleUp(wcur, wd);
                    if (sg_id >= wd && sg_id < num_waves) {{ wcur = {fold_wave}; }}
                    wd = wd << 1u;
                }}
                if (sg_id < num_waves) {{ wave_scratch[sg_id] = wcur; }}
            }}
        }} else if (lid == 0u) {{
            var acc: u32 = wave_scratch[0u];
            for (var w: u32 = 1u; w < num_waves; w = w + 1u) {{
                acc = {fold_serial};
                wave_scratch[w] = acc;
            }}
        }}
        workgroupBarrier();
        if (wave > 0u) {{ cur = {fold_add}; }}
        if (in_bounds) {{ out0[i * STRIDE + {k_expr}] = cur; }}
        if (lid == last_lid) {{ sums[block * STRIDE + {k_expr}] = cur; }}
    }}"
        );
    }
    if trailing_barrier {
        let _ = writeln!(out, "    workgroupBarrier();");
    }
}

fn emit_blocks_subgroup(
    out: &mut String,
    combine: Combine,
    scalars: &[entelechy_abi::Scalar],
    uniform_scalar: Option<entelechy_abi::Scalar>,
) {
    let _ = writeln!(
        out,
        "var<workgroup> wave_scratch: array<u32, 64>;
@compute @workgroup_size({PREFIX_BLOCK}, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {{
    let block = wid.x + wid.y * {MAX_GROUPS}u;
    if (block >= plan.y) {{
        return;
    }}
    let base = block * {PREFIX_BLOCK}u;
    let i = base + lid;
    let in_bounds = i < plan.x;
    let last_lid = min({PREFIX_BLOCK}u, plan.x - base) - 1u;
    let wave = lid / sg_size;
    let num_waves = {PREFIX_BLOCK}u / sg_size;"
    );
    if let Some(scalar) = uniform_scalar {
        if scalars.len() <= 1 {
            emit_channel_scan_subgroup(out, combine, scalar, "0u", false);
        } else {
            let _ = writeln!(out, "    for (var k: u32 = 0u; k < STRIDE; k = k + 1u) {{");
            emit_channel_scan_subgroup(out, combine, scalar, "k", true);
            let _ = writeln!(out, "    }}");
        }
    } else {
        for (k, &scalar) in scalars.iter().enumerate() {
            let need_barrier = k + 1 < scalars.len();
            emit_channel_scan_subgroup(out, combine, scalar, &format!("{k}u"), need_barrier);
        }
    }
    let _ = writeln!(out, "}}");
}

fn emit_channel_scan_workgroup(
    out: &mut String,
    combine: Combine,
    scalar: entelechy_abi::Scalar,
    k_expr: &str,
) {
    let fold_expr = prefix_word(combine, scalar, "wg_scratch[lid - d]", "cur");
    let _ = writeln!(
        out,
        "    {{
        var cur: u32 = 0u;
        if (in_bounds) {{ cur = out0[i * STRIDE + {k_expr}]; }}
        wg_scratch[lid] = cur;
        workgroupBarrier();
        for (var d: u32 = 1u; d < {PREFIX_BLOCK}u; d = d << 1u) {{
            var next_val = cur;
            if (lid >= d) {{
                next_val = {fold_expr};
            }}
            workgroupBarrier();
            cur = next_val;
            wg_scratch[lid] = cur;
            workgroupBarrier();
        }}
        if (in_bounds) {{ out0[i * STRIDE + {k_expr}] = cur; }}
        if (lid == last_lid) {{ sums[block * STRIDE + {k_expr}] = cur; }}
    }}"
    );
}

fn emit_blocks_workgroup(
    out: &mut String,
    combine: Combine,
    scalars: &[entelechy_abi::Scalar],
    uniform_scalar: Option<entelechy_abi::Scalar>,
) {
    let _ = writeln!(
        out,
        "var<workgroup> wg_scratch: array<u32, {PREFIX_BLOCK}>;
@compute @workgroup_size({PREFIX_BLOCK}, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {{
    let block = wid.x + wid.y * {MAX_GROUPS}u;
    if (block >= plan.y) {{
        return;
    }}
    let base = block * {PREFIX_BLOCK}u;
    let i = base + lid;
    let in_bounds = i < plan.x;
    let last_lid = min({PREFIX_BLOCK}u, plan.x - base) - 1u;"
    );
    if let Some(scalar) = uniform_scalar {
        if scalars.len() <= 1 {
            emit_channel_scan_workgroup(out, combine, scalar, "0u");
        } else {
            let _ = writeln!(out, "    for (var k: u32 = 0u; k < STRIDE; k = k + 1u) {{");
            emit_channel_scan_workgroup(out, combine, scalar, "k");
            let _ = writeln!(out, "    }}");
        }
    } else {
        for (k, &scalar) in scalars.iter().enumerate() {
            emit_channel_scan_workgroup(out, combine, scalar, &format!("{k}u"));
        }
    }
    let _ = writeln!(out, "}}");
}

fn emit_offsets(
    out: &mut String,
    combine: Combine,
    scalars: &[entelechy_abi::Scalar],
    uniform_scalar: Option<entelechy_abi::Scalar>,
) {
    let _ = writeln!(
        out,
        "@compute @workgroup_size({PREFIX_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x + gid.y * {PREFIX_SPAN}u;
    if (i >= plan.x) {{
        return;
    }}
    let block = i / {PREFIX_BLOCK}u;
    if (block == 0u) {{
        return;
    }}
    let off = (block - 1u) * STRIDE;
    let dst = i * STRIDE;"
    );
    if let Some(scalar) = uniform_scalar {
        let fold = prefix_word(combine, scalar, "sums[off + k]", "out0[dst + k]");
        let _ = writeln!(
            out,
            "    for (var k: u32 = 0u; k < STRIDE; k = k + 1u) {{
        out0[dst + k] = {fold};
    }}"
        );
    } else {
        for (k, &scalar) in scalars.iter().enumerate() {
            let fold = prefix_word(
                combine,
                scalar,
                &format!("sums[off + {k}u]"),
                &format!("out0[dst + {k}u]"),
            );
            let _ = writeln!(out, "    out0[dst + {k}u] = {fold};");
        }
    }
    let _ = writeln!(out, "}}");
}

/// A prefix integer `sum` with 64-bit running totals (word plus high
/// word), each checked once final. The least element whose total
/// overflows an `Int` is recorded, where the reference's walk fails.
fn emit_prefix_whole(
    module: &Module,
    pass: PrefixPass,
    use_sg: bool,
    sg_directive: &str,
) -> String {
    let scalars = module.manifest.outputs[0].ty.schema.scalars().to_vec();
    let stride = scalars.len().max(1);
    let all_int = scalars
        .iter()
        .all(|s| matches!(s, entelechy_abi::Scalar::Int));
    let mut out = String::new();
    out.push_str(sg_directive);
    let _ = writeln!(
        out,
        "@group(0) @binding({PREFIX_PLAN_BINDING}) var<uniform> plan: vec4<u32>;
@group(0) @binding({PREFIX_OUT_BINDING}) var<storage, read_write> out0: array<u32>;
@group(0) @binding({PREFIX_SUMS_BINDING}) var<storage, read_write> sums: array<u32>;
@group(0) @binding({PREFIX_OUT_HIGHS_BINDING}) var<storage, read_write> out_highs: array<i32>;
@group(0) @binding({PREFIX_SUMS_HIGHS_BINDING}) var<storage, read_write> sum_highs: array<i32>;
{FAILURE_STRUCT}
@group(0) @binding({FAILURE_BINDING}) var<storage, read_write> failure: Failure;
const STRIDE: u32 = {stride}u;
fn fits_word(low: u32, high: i32) -> bool {{
    return (high == 0i && low < 0x80000000u) || (high == -1i && low >= 0x80000000u);
}}"
    );
    match pass {
        PrefixPass::Blocks if use_sg => {
            emit_whole_blocks_subgroup(&mut out, &scalars, all_int);
        }
        PrefixPass::Blocks => {
            emit_whole_blocks_workgroup(&mut out, &scalars, all_int);
        }
        PrefixPass::Offsets => {
            emit_whole_offsets(&mut out, &scalars, all_int);
        }
    }
    out
}

fn emit_whole_channel_subgroup(
    out: &mut String,
    scalar: entelechy_abi::Scalar,
    k_expr: &str,
    trailing_barrier: bool,
) {
    if matches!(scalar, entelechy_abi::Scalar::Float) {
        let _ = writeln!(
            out,
            "    {{
        var v: f32 = 0.0;
        if (in_bounds) {{ v = bitcast<f32>(out0[i * STRIDE + {k_expr}]); }}
        v = subgroupInclusiveAdd(v);
        if (sg_id == sg_size - 1u) {{ wave_lo[wave] = bitcast<u32>(v); }}
        workgroupBarrier();
        if (num_waves <= sg_size) {{
            if (wave == 0u) {{
                var wv: f32 = 0.0;
                if (sg_id < num_waves) {{ wv = bitcast<f32>(wave_lo[sg_id]); }}
                let w_inc = subgroupInclusiveAdd(wv);
                if (sg_id < num_waves) {{ wave_lo[sg_id] = bitcast<u32>(w_inc); }}
            }}
        }} else if (lid == 0u) {{
            var acc: f32 = bitcast<f32>(wave_lo[0u]);
            for (var w: u32 = 1u; w < num_waves; w = w + 1u) {{
                acc = acc + bitcast<f32>(wave_lo[w]);
                wave_lo[w] = bitcast<u32>(acc);
            }}
        }}
        workgroupBarrier();
        if (wave > 0u) {{ v = bitcast<f32>(wave_lo[wave - 1u]) + v; }}
        let res = bitcast<u32>(v);
        if (in_bounds) {{ out0[i * STRIDE + {k_expr}] = res; }}
        if (lid == last_lid) {{ sums[block * STRIDE + {k_expr}] = res; }}
    }}"
        );
    } else {
        let _ = writeln!(
            out,
            "    {{
        var cur_lo: u32 = 0u;
        var cur_hi: i32 = 0i;
        if (in_bounds) {{
            cur_lo = out0[i * STRIDE + {k_expr}];
            if ((plan.w & {PREFIX_FLAG_WIDEN}u) != 0u) {{
                cur_hi = select(0i, -1i, bitcast<i32>(cur_lo) < 0i);
            }} else {{
                cur_hi = out_highs[i * STRIDE + {k_expr}];
            }}
        }}
        var d: u32 = 1u;
        loop {{
            if (d >= sg_size) {{ break; }}
            let prev_lo = subgroupShuffleUp(cur_lo, d);
            let prev_hi = subgroupShuffleUp(cur_hi, d);
            if (sg_id >= d) {{
                let sum_lo = prev_lo + cur_lo;
                cur_hi = prev_hi + cur_hi + select(0i, 1i, sum_lo < prev_lo);
                cur_lo = sum_lo;
            }}
            d = d << 1u;
        }}
        if (sg_id == sg_size - 1u) {{
            wave_lo[wave] = cur_lo;
            wave_hi[wave] = cur_hi;
        }}
        workgroupBarrier();
        if (num_waves <= sg_size) {{
            if (wave == 0u) {{
                var w_lo: u32 = 0u;
                var w_hi: i32 = 0i;
                if (sg_id < num_waves) {{
                    w_lo = wave_lo[sg_id];
                    w_hi = wave_hi[sg_id];
                }}
                var wd: u32 = 1u;
                loop {{
                    if (wd >= num_waves) {{ break; }}
                    let p_lo = subgroupShuffleUp(w_lo, wd);
                    let p_hi = subgroupShuffleUp(w_hi, wd);
                    if (sg_id >= wd && sg_id < num_waves) {{
                        let s_lo = p_lo + w_lo;
                        w_hi = p_hi + w_hi + select(0i, 1i, s_lo < p_lo);
                        w_lo = s_lo;
                    }}
                    wd = wd << 1u;
                }}
                if (sg_id < num_waves) {{
                    wave_lo[sg_id] = w_lo;
                    wave_hi[sg_id] = w_hi;
                }}
            }}
        }} else if (lid == 0u) {{
            var acc_lo: u32 = wave_lo[0u];
            var acc_hi: i32 = wave_hi[0u];
            for (var w: u32 = 1u; w < num_waves; w = w + 1u) {{
                let w_lo = wave_lo[w];
                let w_hi = wave_hi[w];
                let s_lo = acc_lo + w_lo;
                acc_hi = acc_hi + w_hi + select(0i, 1i, s_lo < acc_lo);
                acc_lo = s_lo;
                wave_lo[w] = acc_lo;
                wave_hi[w] = acc_hi;
            }}
        }}
        workgroupBarrier();
        if (wave > 0u) {{
            let off_lo = wave_lo[wave - 1u];
            let off_hi = wave_hi[wave - 1u];
            let sum_lo = off_lo + cur_lo;
            cur_hi = off_hi + cur_hi + select(0i, 1i, sum_lo < off_lo);
            cur_lo = sum_lo;
        }}
        if (in_bounds) {{
            out0[i * STRIDE + {k_expr}] = cur_lo;
            out_highs[i * STRIDE + {k_expr}] = cur_hi;
            if (!fits_word(cur_lo, cur_hi)) {{ all_fit = false; }}
        }}
        if (lid == last_lid) {{
            sums[block * STRIDE + {k_expr}] = cur_lo;
            sum_highs[block * STRIDE + {k_expr}] = cur_hi;
        }}
    }}"
        );
    }
    if trailing_barrier {
        let _ = writeln!(out, "    workgroupBarrier();");
    }
}

fn emit_whole_blocks_subgroup(out: &mut String, scalars: &[entelechy_abi::Scalar], all_int: bool) {
    let _ = writeln!(
        out,
        "var<workgroup> wave_lo: array<u32, 64>;
var<workgroup> wave_hi: array<i32, 64>;
@compute @workgroup_size({PREFIX_BLOCK}, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {{
    let block = wid.x + wid.y * {MAX_GROUPS}u;
    if (block >= plan.y) {{
        return;
    }}
    let base = block * {PREFIX_BLOCK}u;
    let i = base + lid;
    let in_bounds = i < plan.x;
    let last_lid = min({PREFIX_BLOCK}u, plan.x - base) - 1u;
    let wave = lid / sg_size;
    let num_waves = {PREFIX_BLOCK}u / sg_size;
    var all_fit: bool = true;"
    );
    if all_int {
        if scalars.len() <= 1 {
            emit_whole_channel_subgroup(out, entelechy_abi::Scalar::Int, "0u", false);
        } else {
            let _ = writeln!(out, "    for (var k: u32 = 0u; k < STRIDE; k = k + 1u) {{");
            emit_whole_channel_subgroup(out, entelechy_abi::Scalar::Int, "k", true);
            let _ = writeln!(out, "    }}");
        }
    } else {
        for (k, &scalar) in scalars.iter().enumerate() {
            let need_barrier = k + 1 < scalars.len();
            emit_whole_channel_subgroup(out, scalar, &format!("{k}u"), need_barrier);
        }
    }
    let _ = writeln!(
        out,
        "    if ((plan.w & {PREFIX_FLAG_CHECK_FITS}u) != 0u && in_bounds && !all_fit) {{
        atomicMax(&failure.total, 0xFFFFFFFFu - i);
    }}
}}"
    );
}

fn emit_whole_channel_workgroup(out: &mut String, scalar: entelechy_abi::Scalar, k_expr: &str) {
    if matches!(scalar, entelechy_abi::Scalar::Float) {
        let _ = writeln!(
            out,
            "    {{
        var cur: u32 = 0u;
        if (in_bounds) {{ cur = out0[i * STRIDE + {k_expr}]; }}
        wg_lo[lid] = cur;
        workgroupBarrier();
        for (var d: u32 = 1u; d < {PREFIX_BLOCK}u; d = d << 1u) {{
            var next_val = cur;
            if (lid >= d) {{
                next_val = bitcast<u32>(bitcast<f32>(wg_lo[lid - d]) + bitcast<f32>(cur));
            }}
            workgroupBarrier();
            cur = next_val;
            wg_lo[lid] = cur;
            workgroupBarrier();
        }}
        if (in_bounds) {{ out0[i * STRIDE + {k_expr}] = cur; }}
        if (lid == last_lid) {{ sums[block * STRIDE + {k_expr}] = cur; }}
    }}"
        );
    } else {
        let _ = writeln!(
            out,
            "    {{
        var cur_lo: u32 = 0u;
        var cur_hi: i32 = 0i;
        if (in_bounds) {{
            cur_lo = out0[i * STRIDE + {k_expr}];
            if ((plan.w & {PREFIX_FLAG_WIDEN}u) != 0u) {{
                cur_hi = select(0i, -1i, bitcast<i32>(cur_lo) < 0i);
            }} else {{
                cur_hi = out_highs[i * STRIDE + {k_expr}];
            }}
        }}
        wg_lo[lid] = cur_lo;
        wg_hi[lid] = cur_hi;
        workgroupBarrier();
        for (var d: u32 = 1u; d < {PREFIX_BLOCK}u; d = d << 1u) {{
            var next_lo = cur_lo;
            var next_hi = cur_hi;
            if (lid >= d) {{
                let p_lo = wg_lo[lid - d];
                let p_hi = wg_hi[lid - d];
                next_lo = p_lo + cur_lo;
                next_hi = p_hi + cur_hi + select(0i, 1i, next_lo < p_lo);
            }}
            workgroupBarrier();
            cur_lo = next_lo;
            cur_hi = next_hi;
            wg_lo[lid] = cur_lo;
            wg_hi[lid] = cur_hi;
            workgroupBarrier();
        }}
        if (in_bounds) {{
            out0[i * STRIDE + {k_expr}] = cur_lo;
            out_highs[i * STRIDE + {k_expr}] = cur_hi;
            if (!fits_word(cur_lo, cur_hi)) {{ all_fit = false; }}
        }}
        if (lid == last_lid) {{
            sums[block * STRIDE + {k_expr}] = cur_lo;
            sum_highs[block * STRIDE + {k_expr}] = cur_hi;
        }}
    }}"
        );
    }
}

fn emit_whole_blocks_workgroup(out: &mut String, scalars: &[entelechy_abi::Scalar], all_int: bool) {
    let _ = writeln!(
        out,
        "var<workgroup> wg_lo: array<u32, {PREFIX_BLOCK}>;
var<workgroup> wg_hi: array<i32, {PREFIX_BLOCK}>;
@compute @workgroup_size({PREFIX_BLOCK}, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {{
    let block = wid.x + wid.y * {MAX_GROUPS}u;
    if (block >= plan.y) {{
        return;
    }}
    let base = block * {PREFIX_BLOCK}u;
    let i = base + lid;
    let in_bounds = i < plan.x;
    let last_lid = min({PREFIX_BLOCK}u, plan.x - base) - 1u;
    var all_fit: bool = true;"
    );
    if all_int {
        if scalars.len() <= 1 {
            emit_whole_channel_workgroup(out, entelechy_abi::Scalar::Int, "0u");
        } else {
            let _ = writeln!(out, "    for (var k: u32 = 0u; k < STRIDE; k = k + 1u) {{");
            emit_whole_channel_workgroup(out, entelechy_abi::Scalar::Int, "k");
            let _ = writeln!(out, "    }}");
        }
    } else {
        for (k, &scalar) in scalars.iter().enumerate() {
            emit_whole_channel_workgroup(out, scalar, &format!("{k}u"));
        }
    }
    let _ = writeln!(
        out,
        "    if ((plan.w & {PREFIX_FLAG_CHECK_FITS}u) != 0u && in_bounds && !all_fit) {{
        atomicMax(&failure.total, 0xFFFFFFFFu - i);
    }}
}}"
    );
}

fn emit_whole_offsets(out: &mut String, scalars: &[entelechy_abi::Scalar], all_int: bool) {
    let _ = writeln!(
        out,
        "@compute @workgroup_size({PREFIX_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x + gid.y * {PREFIX_SPAN}u;
    if (i >= plan.x) {{
        return;
    }}
    let block = i / {PREFIX_BLOCK}u;
    let dst = i * STRIDE;
    let check = (plan.w & {PREFIX_FLAG_CHECK_FITS}u) != 0u;
    var all_fit: bool = true;"
    );
    if all_int {
        let _ = writeln!(
            out,
            "    if (block > 0u) {{
        let off = (block - 1u) * STRIDE;
        for (var k: u32 = 0u; k < STRIDE; k = k + 1u) {{
            let a_lo = sums[off + k];
            let a_hi = sum_highs[off + k];
            let b_lo = out0[dst + k];
            let b_hi = out_highs[dst + k];
            let s_lo = a_lo + b_lo;
            let s_hi = a_hi + b_hi + select(0i, 1i, s_lo < a_lo);
            out0[dst + k] = s_lo;
            out_highs[dst + k] = s_hi;
            if (!fits_word(s_lo, s_hi)) {{ all_fit = false; }}
        }}
    }} else if (check) {{
        for (var k: u32 = 0u; k < STRIDE; k = k + 1u) {{
            if (!fits_word(out0[dst + k], out_highs[dst + k])) {{ all_fit = false; }}
        }}
    }}"
        );
    } else {
        for (k, &scalar) in scalars.iter().enumerate() {
            if matches!(scalar, entelechy_abi::Scalar::Float) {
                let _ = writeln!(
                    out,
                    "    if (block > 0u) {{
        let off = (block - 1u) * STRIDE;
        out0[dst + {k}u] = bitcast<u32>(bitcast<f32>(sums[off + {k}u]) + bitcast<f32>(out0[dst + {k}u]));
    }}"
                );
            } else {
                let _ = writeln!(
                    out,
                    "    if (block > 0u) {{
        let off = (block - 1u) * STRIDE;
        let a_lo = sums[off + {k}u];
        let a_hi = sum_highs[off + {k}u];
        let b_lo = out0[dst + {k}u];
        let b_hi = out_highs[dst + {k}u];
        let s_lo = a_lo + b_lo;
        let s_hi = a_hi + b_hi + select(0i, 1i, s_lo < a_lo);
        out0[dst + {k}u] = s_lo;
        out_highs[dst + {k}u] = s_hi;
        if (!fits_word(s_lo, s_hi)) {{ all_fit = false; }}
    }} else if (check) {{
        if (!fits_word(out0[dst + {k}u], out_highs[dst + {k}u])) {{ all_fit = false; }}
    }}"
                );
            }
        }
    }
    let _ = writeln!(
        out,
        "    if (check && !all_fit) {{
        atomicMax(&failure.total, 0xFFFFFFFFu - i);
    }}
}}"
    );
}
