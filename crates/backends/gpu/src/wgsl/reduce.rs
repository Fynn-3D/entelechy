//! A reduction's fold pass, run level by level until one element is left,
//! preserving the exact pairwise binary tree of [`entelechy_rules::reduce`].
//!
//! When a 256-lane workgroup fits the element type, one workgroup folds 256
//! elements (8 tree levels) into one output element with coalesced global
//! loads:
//! - [`ReduceStrategy::Subgroup256`]: folds the first `log2(sg_size)` levels
//!   register-to-register via `subgroupShuffleDown`, then reduces the at most
//!   16 wave roots in Wave 0 (1 barrier, tiny shared memory).
//! - [`ReduceStrategy::Workgroup256`]: folds all 8 levels in workgroup shared
//!   memory without subgroup extensions.
//! - [`ReduceStrategy::Serial16`]: fallback for oversized structs exceeding
//!   workgroup shared memory (`entelechy_rules::reduce::FOLD` elements per
//!   invocation).

use super::*;

/// Workgroup length of the serial fallback reduce step.
pub const REDUCE_WORKGROUP: u32 = 64;
/// Threads a row of serial reduce workgroups covers; see [`folded`].
pub const REDUCE_SPAN: u32 = MAX_GROUPS * REDUCE_WORKGROUP;
/// Elements folded per workgroup by the parallel 8-level reduce passes.
pub const REDUCE_BLOCK: u32 = 256;

/// Parallelism strategy selected for a reduction module on a given device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceStrategy {
    /// 256-thread workgroup folding 8 pairwise levels via `subgroupShuffleDown`
    /// plus a 16-slot cross-wave shared buffer.
    Subgroup256,
    /// 256-thread workgroup folding 8 pairwise levels in workgroup shared memory.
    Workgroup256,
    /// Fallback for giant structs: each invocation folds [`entelechy_rules::reduce::FOLD`]
    /// elements in a private array.
    Serial16,
}

impl ReduceStrategy {
    /// Number of input elements folded into one output element per pass.
    pub const fn fold_factor(self) -> usize {
        match self {
            Self::Subgroup256 | Self::Workgroup256 => REDUCE_BLOCK as usize,
            Self::Serial16 => entelechy_rules::reduce::FOLD,
        }
    }
}

/// Chooses the fastest reduction fold strategy that fits the device's
/// `subgroups` capabilities and `max_wg_storage` workgroup memory budget.
pub fn reduce_strategy(
    module: &Module,
    subgroups: SubgroupCaps,
    max_wg_storage: u64,
) -> ReduceStrategy {
    let Some(merge) = &module.merge else {
        return ReduceStrategy::Serial16;
    };
    let element = merge.args[0];
    let (bytes, align) = super::map::layout(module, element);
    let slot = bytes.next_multiple_of(align);
    let overhead: u64 = if module.accounted() { 16 } else { 0 };
    let wave_slots = u64::from(REDUCE_BLOCK / subgroups.min_size.max(4)).max(1);
    if subgroups.wave_parallel_256()
        && type_words(module, element) <= SUBGROUP_SHUFFLE_MAX_WORDS
        && slot.saturating_mul(wave_slots).saturating_add(overhead) <= max_wg_storage
    {
        return ReduceStrategy::Subgroup256;
    }
    if slot
        .saturating_mul(u64::from(REDUCE_BLOCK))
        .saturating_add(overhead)
        <= max_wg_storage / 2
    {
        return ReduceStrategy::Workgroup256;
    }
    ReduceStrategy::Serial16
}

/// The fold pass with the `packed` inputs read from one buffer.
pub fn emit_reduce_packed(module: &Module, packed: &[usize]) -> String {
    emit_reduce_packed_longs(module, packed, Longs::Native)
}

/// [`emit_reduce_packed`] with the module's Longs computed as `longs` says.
pub fn emit_reduce_packed_longs(module: &Module, packed: &[usize], longs: Longs) -> String {
    let groups = if packed.is_empty() {
        Vec::new()
    } else {
        vec![packed.to_vec()]
    };
    emit_reduce_packed_groups_longs(module, &groups, longs)
}

/// Emits the reduce fold pass reading packed inputs from `packs` using the
/// serial fallback strategy.
pub fn emit_reduce_packed_groups_longs(
    module: &Module,
    packs: &[Vec<usize>],
    longs: Longs,
) -> String {
    emit_reduce_with_strategy(
        module,
        packs,
        longs,
        ReduceStrategy::Serial16,
        SubgroupCaps::default(),
    )
}

/// Emits the reduce fold pass reading packed inputs from `packs` using the
/// selected [`ReduceStrategy`] and `subgroups` capabilities.
pub fn emit_reduce_with_strategy(
    module: &Module,
    packs: &[Vec<usize>],
    longs: Longs,
    strategy: ReduceStrategy,
    subgroups: SubgroupCaps,
) -> String {
    emit_reduce_with_caps(module, packs, longs, strategy, subgroups, false)
}

/// Emits the reduce fold pass reading packed inputs from `packs` with full
/// hardware capabilities (`subgroups` and `shader_f16`).
pub fn emit_reduce_with_caps(
    module: &Module,
    packs: &[Vec<usize>],
    longs: Longs,
    strategy: ReduceStrategy,
    subgroups: SubgroupCaps,
    shader_f16: bool,
) -> String {
    debug_assert_eq!(module.manifest.mode, Mode::Reduce);
    let Some(merge) = &module.merge else {
        return String::new();
    };
    let element_ty = merge.args[0];
    let stride = module.manifest.outputs[0].ty.schema.value_words();
    let (count_binding, in_binding, out_binding) = reduce_bindings(module);
    let mut out = String::new();
    if shader_uses_f16(module, false, packs, &[], shader_f16) {
        out.push_str("enable f16;\n");
    }
    if strategy == ReduceStrategy::Subgroup256 {
        out.push_str(subgroups.directive());
    }
    // The map shader's context, params and inputs, which `merge` may read.
    emit_preamble(
        &mut out,
        module,
        false,
        packs,
        &[],
        longs,
        false,
        shader_f16,
    );
    if strategy == ReduceStrategy::Subgroup256 && longs == Longs::Native && module.holds(Type::Long)
    {
        out.push_str(SUBGROUP_I64_HELPERS);
    }
    let _ = writeln!(
        out,
        "@group(0) @binding({count_binding}) var<uniform> level: vec4<u32>;"
    );
    let _ = writeln!(
        out,
        "@group(0) @binding({in_binding}) var<storage, read> fold_in: array<u32>;"
    );
    let _ = writeln!(
        out,
        "@group(0) @binding({out_binding}) var<storage, read_write> fold_out: array<u32>;"
    );
    let (expr, _) = load_expr(module, element_ty, "fold_in", "b", 0, Encoding::Word);
    let _ = writeln!(
        out,
        "fn load_level(index: u32) -> {} {{
    let b = index * {stride}u;
    return {expr};
}}",
        wgsl_type(element_ty)
    );
    let mut stores = Vec::new();
    store_lines(module, element_ty, "v", 0, &mut stores);
    let body = stores.join("\n").replace("out0[", "fold_out[");
    let _ = writeln!(
        out,
        "fn store_level(index: u32, v: {}) {{
    let b = index * {stride}u;
{body}
}}",
        wgsl_type(element_ty)
    );
    for (index, function) in module.functions.iter().enumerate() {
        emit_function(
            &mut out,
            module,
            function,
            &function_name(index),
            false,
            packs,
            &[],
            longs,
            false,
            shader_f16,
        );
        out.push('\n');
    }
    emit_function(
        &mut out,
        module,
        merge,
        "merge",
        false,
        packs,
        &[],
        longs,
        false,
        shader_f16,
    );
    let element = wgsl_type(element_ty);
    match strategy {
        ReduceStrategy::Subgroup256 => {
            let shuffle_val = subgroup_shuffle_expr(
                module,
                element_ty,
                "val",
                "d",
                "subgroupShuffleDown",
                longs == Longs::Pairs,
            );
            let shuffle_wval = subgroup_shuffle_expr(
                module,
                element_ty,
                "wval",
                "wd",
                "subgroupShuffleDown",
                longs == Longs::Pairs,
            );
            let settle_waves = if module.accounted() {
                "    let wave_owed = subgroupAdd(owed);
    if (sg_id == 0u) {
        owed = wave_owed;
        settle_left();
    }"
            } else {
                ""
            };
            let wave_slots = (REDUCE_BLOCK / subgroups.min_size.max(4)).max(1);
            let _ = writeln!(
                out,
                "
var<workgroup> wave_vals: array<{element}, {wave_slots}>;
@compute @workgroup_size({REDUCE_BLOCK}, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {{
    let n = level.x;
    current = 0u;
    let block = wid.x + wid.y * {MAX_GROUPS}u;
    let start = block * {REDUCE_BLOCK}u;
    if (start >= n) {{
        return;
    }}
    let run = min({REDUCE_BLOCK}u, n - start);
    var val: {element};
    if (lid < run) {{
        val = load_level(start + lid);
    }}
    var d: u32 = 1u;
    loop {{
        if (d >= sg_size) {{ break; }}
        let partner = {shuffle_val};
        if ((lid & (2u * d - 1u)) == 0u && lid + d < run) {{
            begin_work();
            val = merge(val, partner);
            {keep}
        }}
        d = d << 1u;
    }}
    let wave = lid / sg_size;
    let num_waves = {REDUCE_BLOCK}u / sg_size;
    let active_waves = (run + sg_size - 1u) / sg_size;
    if (sg_id == 0u && wave < active_waves) {{
        wave_vals[wave] = val;
    }}
    workgroupBarrier();
    if (num_waves <= sg_size) {{
        if (wave == 0u) {{
            var wval: {element};
            if (sg_id < active_waves) {{
                wval = wave_vals[sg_id];
            }}
            var wd: u32 = 1u;
            loop {{
                if (wd >= num_waves) {{ break; }}
                let wpartner = {shuffle_wval};
                if ((sg_id & (2u * wd - 1u)) == 0u && sg_id + wd < active_waves) {{
                    begin_work();
                    wval = merge(wval, wpartner);
                    {keep}
                }}
                wd = wd << 1u;
            }}
            if (sg_id == 0u) {{
                store_level(block, wval);
            }}
        }}
    }} else {{
        var wd: u32 = 1u;
        loop {{
            if (wd >= num_waves) {{ break; }}
            if (lid < active_waves && (lid & (2u * wd - 1u)) == 0u && lid + wd < active_waves) {{
                begin_work();
                wave_vals[lid] = merge(wave_vals[lid], wave_vals[lid + wd]);
                {keep}
            }}
            workgroupBarrier();
            wd = wd << 1u;
        }}
        if (lid == 0u) {{
            store_level(block, wave_vals[0]);
        }}
    }}
{settle_waves}
}}",
                keep = keep_left(module),
            );
        }
        ReduceStrategy::Workgroup256 => {
            let settle_wg = if module.accounted() {
                "    atomicAdd(&leftover, owed);
    workgroupBarrier();
    if (lid == 0u) {
        owed = atomicLoad(&leftover);
        settle_left();
    }"
            } else {
                ""
            };
            let _ = writeln!(
                out,
                "
var<workgroup> wg_vals: array<{element}, {REDUCE_BLOCK}>;
@compute @workgroup_size({REDUCE_BLOCK}, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {{
    let n = level.x;
    current = 0u;
    let block = wid.x + wid.y * {MAX_GROUPS}u;
    let start = block * {REDUCE_BLOCK}u;
    if (start >= n) {{
        return;
    }}
    let run = min({REDUCE_BLOCK}u, n - start);
    if (lid < run) {{
        wg_vals[lid] = load_level(start + lid);
    }}
    workgroupBarrier();
    var d: u32 = 1u;
    loop {{
        if (d >= {REDUCE_BLOCK}u) {{ break; }}
        if ((lid & (2u * d - 1u)) == 0u && lid + d < run) {{
            begin_work();
            wg_vals[lid] = merge(wg_vals[lid], wg_vals[lid + d]);
            {keep}
        }}
        workgroupBarrier();
        d = d << 1u;
    }}
    if (lid == 0u) {{
        store_level(block, wg_vals[0]);
    }}
{settle_wg}
}}",
                keep = keep_left(module),
            );
        }
        ReduceStrategy::Serial16 => {
            let fold = entelechy_rules::reduce::FOLD;
            let _ = writeln!(
                out,
                "
@compute @workgroup_size({REDUCE_WORKGROUP}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let n = level.x;
    // A merge that fails fails at the first element, as the reference's.
    current = 0u;
    let index = gid.x + gid.y * {REDUCE_SPAN}u;
    let start = index * {fold}u;
    if (start >= n) {{
        return;
    }}
    // The run this element folds, then that run folded by the tree: a
    // level of pairs, an odd last one carried up, until one is left.
    var held: array<{element}, {fold}>;
    var run = min({fold}u, n - start);
    var at = 0u;
    loop {{
        if (at >= run) {{ break; }}
        held[at] = load_level(start + at);
        at = at + 1u;
    }}
    loop {{
        if (run <= 1u) {{ break; }}
        let pairs = run / 2u;
        var k = 0u;
        loop {{
            if (k >= pairs) {{ break; }}
            begin_work();
            held[k] = merge(held[2u * k], held[2u * k + 1u]);
            {keep}
            k = k + 1u;
        }}
        if ((run & 1u) == 1u) {{ held[pairs] = held[run - 1u]; }}
        run = (run + 1u) / 2u;
    }}
    {settle}
    store_level(index, held[0]);
}}",
                keep = keep_left(module),
                settle = settle_left(module),
            );
        }
    }
    out
}

/// The fold pass's bindings after the inputs: level count, level read,
/// level written.
pub fn reduce_bindings(module: &Module) -> (u32, u32, u32) {
    let base = INPUT_BINDING + module.manifest.inputs.len() as u32;
    (base, base + 1, base + 2)
}
