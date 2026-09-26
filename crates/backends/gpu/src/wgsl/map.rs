//! A map's entry point: one element per thread, written in place.

use super::*;

/// Emits a complete compute shader for the module.
pub fn emit(module: &Module) -> String {
    emit_packed(module, &[])
}

/// Emits the module's shader with the `packed` inputs read from one
/// buffer; see [`packed_inputs`].
pub fn emit_packed(module: &Module, packed: &[usize]) -> String {
    let manifest = &module.manifest;
    let mut out = String::new();
    emit_preamble(&mut out, module, true, packed);
    // One store per written output; a scatter stores the entry's pairs.
    let scattering = matches!(manifest.mode, Mode::Scatter { .. });
    for (k, port) in manifest
        .outputs
        .iter()
        .enumerate()
        .take(written_outputs(module))
    {
        let element = if scattering {
            module.entry.result_type()
        } else {
            port.element()
        };
        let body = if !scattering && !port.ty.schema.is_wide() {
            narrow_store_body(module, k, element)
        } else {
            let stride = type_words(module, element);
            let mut stores = Vec::new();
            store_lines(module, element, "v", 0, &mut stores);
            format!(
                "    let b = index * {stride}u;\n{}",
                stores.join("\n").replace("out0[", &format!("out{k}["))
            )
        };
        let _ = writeln!(
            out,
            "fn store_out{k}(index: u32, v: {}) {{\n{body}\n}}",
            wgsl_type(element)
        );
    }
    emit_functions(module, &mut out);
    for (index, function) in module.functions.iter().enumerate() {
        emit_function(
            &mut out,
            module,
            function,
            &function_name(index),
            false,
            packed,
        );
        out.push('\n');
    }
    emit_function(&mut out, module, &module.entry, "entry", true, packed);
    if let Some(merge) = &module.merge {
        emit_function(&mut out, module, merge, "merge", false, packed);
    }
    if let Some(init) = &module.init {
        emit_function(&mut out, module, init, "init", false, packed);
        // Ordered work: one invocation walks the rows, carrying the value.
        let _ = writeln!(
            out,
            "
@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x != 0u) {{
        return;
    }}
    let n = u32(ctx[0].x);
    var carried: {} = init();
    for (var i: u32 = 0u; i < n; i = i + 1u) {{
        storageBarrier();
        current = i;
        begin_work();
        carried = entry(vec3<f32>(f32(i), 0.0, 0.0), i, carried);
        store_out0(i, carried);
        {keep}
    }}
    {settle}
}}",
            wgsl_type(
                init.results
                    .first()
                    .map_or(Type::Float, |r| init.type_of(*r).unwrap_or(Type::Float))
            ),
            keep = keep_left(module),
            settle = settle_left(module),
        );
        return out;
    }
    // A map of several outputs stores them from the entry. An emitting
    // scatter combines its returned pair into the target at once, as its
    // emitted ones: its rule is order-free.
    let landing = if emitted_pairs(module).is_empty() {
        None
    } else {
        Some(format!(
            "{}(entry(coord, element));",
            emit_name(module.entry.result_type())
        ))
    };
    let store = if let Some(landing) = &landing {
        landing.as_str()
    } else if written_outputs(module) > 1 {
        "entry(coord, element);"
    } else {
        "store_out0(element, entry(coord, element));"
    };
    if module.accounted() {
        // An accounted module runs every invocation to one barrier. An
        // element runs while the run is within the Host's bound, adds its
        // remainder short of a block to the workgroup's sum, and one
        // invocation settles the sum.
        let _ = writeln!(
            out,
            "
@compute @workgroup_size({WORKGROUP}, {WORKGROUP}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {{
    let w = u32(ctx[0].x);
    let h = u32(ctx[0].y);
    let d = u32(ctx[1].z);
    let fold = u32(ctx[5].y);
    var x = gid.x;
    var y = gid.y;
    if (fold != 0u) {{
        x = gid.x + gid.y * fold;
        y = 0u;
    }}
    if (x < w && y < h && gid.z < d && atomicLoad(&failure.blocks) <= failure.limit) {{
        let element = (gid.z * h + y) * w + x;
        current = element;
        begin_work();
        let coord = vec3<f32>(f32(x), f32(y), f32(gid.z));
        {store}
        atomicAdd(&leftover, work & {mask}u);
    }}
    workgroupBarrier();
    if (lid == 0u) {{
        owed = atomicLoad(&leftover);
        settle_left();
    }}
}}",
            mask = orla_ir::WORK_BLOCK - 1,
        );
        return out;
    }
    let _ = writeln!(
        out,
        "
@compute @workgroup_size({WORKGROUP}, {WORKGROUP}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let w = u32(ctx[0].x);
    let h = u32(ctx[0].y);
    let d = u32(ctx[1].z);
    let fold = u32(ctx[5].y);
    var x = gid.x;
    var y = gid.y;
    if (fold != 0u) {{
        x = gid.x + gid.y * fold;
        y = 0u;
    }}
    if (x >= w || y >= h || gid.z >= d) {{
        return;
    }}
    let element = (gid.z * h + y) * w + x;
    current = element;
    begin_work();
    let coord = vec3<f32>(f32(x), f32(y), f32(gid.z));
    {store}
}}"
    );
    out
}

/// Words a value of a type takes in a buffer.
pub fn type_words(module: &Module, ty: Type) -> usize {
    match ty {
        Type::Struct(index) => module.structs[index as usize]
            .fields
            .iter()
            .map(|field| type_words(module, field.ty))
            .sum(),
        other => other.words().unwrap_or(1),
    }
}
