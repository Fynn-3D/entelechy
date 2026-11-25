//! A reduction's fold pass, run level by level until one element is left.

use super::*;

/// Workgroup length of the reduce step.
pub const REDUCE_WORKGROUP: u32 = 64;
/// Threads a row of reduce workgroups covers; see [`folded`].
pub const REDUCE_SPAN: u32 = MAX_GROUPS * REDUCE_WORKGROUP;

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

/// Emits the reduce fold pass reading packed inputs from `packs`.
pub fn emit_reduce_packed_groups_longs(
    module: &Module,
    packs: &[Vec<usize>],
    longs: Longs,
) -> String {
    debug_assert_eq!(module.manifest.mode, Mode::Reduce);
    let Some(merge) = &module.merge else {
        return String::new();
    };
    let element = merge.args[0];
    let stride = module.manifest.outputs[0].ty.schema.value_words();
    let (count_binding, in_binding, out_binding) = reduce_bindings(module);
    let mut out = String::new();
    // The map shader's context, params and inputs, which `merge` may read.
    emit_preamble(&mut out, module, false, packs, &[], longs);
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
    let (expr, _) = load_expr(module, element, "fold_in", "b", 0, Encoding::Word);
    let _ = writeln!(
        out,
        "fn load_level(index: u32) -> {} {{
    let b = index * {stride}u;
    return {expr};
}}",
        wgsl_type(element)
    );
    let mut stores = Vec::new();
    store_lines(module, element, "v", 0, &mut stores);
    let body = stores.join("\n").replace("out0[", "fold_out[");
    let _ = writeln!(
        out,
        "fn store_level(index: u32, v: {}) {{
    let b = index * {stride}u;
{body}
}}",
        wgsl_type(element)
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
        );
        out.push('\n');
    }
    emit_function(&mut out, module, merge, "merge", false, packs, &[], longs);
    let fold = entelechy_rules::reduce::FOLD;
    let element = wgsl_type(element);
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
    out
}

/// The fold pass's bindings after the inputs: level count, level read,
/// level written.
pub fn reduce_bindings(module: &Module) -> (u32, u32, u32) {
    let base = INPUT_BINDING + module.manifest.inputs.len() as u32;
    (base, base + 1, base + 2)
}
