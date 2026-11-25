//! A map's entry point: one element per thread, written in place.

use super::*;

/// Emits a complete compute shader for the module.
pub fn emit(module: &Module) -> String {
    emit_packed(module, &[])
}

/// Emits the module's shader with the `packed` inputs read from one
/// buffer; see [`packed_inputs`].
pub fn emit_packed(module: &Module, packed: &[usize]) -> String {
    emit_packed_longs(module, packed, Longs::Native)
}

/// [`emit_packed`] with the module's Longs computed as `longs` says; a
/// scatter landing its Longs by halves lands the high ones
/// ([`by_halves`]).
pub fn emit_packed_longs(module: &Module, packed: &[usize], longs: Longs) -> String {
    emit_packed_half(module, packed, longs, first_half(module, longs))
}

/// [`emit_packed_longs`], an emitting scatter landing `half`.
pub fn emit_packed_half(module: &Module, packed: &[usize], longs: Longs, half: Half) -> String {
    let groups = if packed.is_empty() {
        Vec::new()
    } else {
        vec![packed.to_vec()]
    };
    emit_packed_groups_half(module, &groups, longs, half)
}

/// Emits the module's shader with multiple `packs` input groups and `longs` mode.
pub fn emit_packed_groups_longs(module: &Module, packs: &[Vec<usize>], longs: Longs) -> String {
    emit_packed_groups_half(module, packs, longs, first_half(module, longs))
}

/// [`emit_packed_groups_longs`], an emitting scatter landing `half`.
pub fn emit_packed_groups_half(
    module: &Module,
    packs: &[Vec<usize>],
    longs: Longs,
    half: Half,
) -> String {
    emit_full(module, packs, &[], longs, half)
}

/// Emits the module's compute shader with packed input groups, hardware
/// `textured` inputs, `longs` mode, and scatter `half`.
pub fn emit_full(
    module: &Module,
    packs: &[Vec<usize>],
    textured: &[usize],
    longs: Longs,
    half: Half,
) -> String {
    emit_full_filtered(module, packs, textured, longs, half, true)
}

/// Emits the module's compute shader with packed input groups, hardware
/// `textured` inputs, `longs` mode, scatter `half`, and `float32_filterable`
/// capability flag for hardware filtering samplers.
pub fn emit_full_filtered(
    module: &Module,
    packs: &[Vec<usize>],
    textured: &[usize],
    longs: Longs,
    half: Half,
    float32_filterable: bool,
) -> String {
    emit_full_with_subgroups(
        module,
        packs,
        textured,
        longs,
        half,
        float32_filterable,
        SubgroupCaps::default(),
    )
}

/// Whether `module` is a `group N` Kernel that runs its exchanges in a single
/// hardware subgroup via `subgroupShuffle` (zero workgroup memory, zero
/// barriers) given `subgroups`.
pub fn uses_subgroup_group(module: &Module, subgroups: SubgroupCaps) -> bool {
    let Mode::Group { size } = module.manifest.mode else {
        return false;
    };
    if !subgroups.fits_single_wave(size) {
        return false;
    }
    let mut met = Vec::new();
    exchanged(&module.entry.body, &mut met);
    met.into_iter()
        .all(|ty| type_words(module, ty) <= SUBGROUP_SHUFFLE_MAX_WORDS)
}

/// Emits the module's compute shader with packed input groups, hardware
/// `textured` inputs, `longs` mode, scatter `half`, `float32_filterable`, and
/// hardware `subgroups` capabilities.
#[allow(clippy::too_many_arguments)]
pub fn emit_full_with_subgroups(
    module: &Module,
    packs: &[Vec<usize>],
    textured: &[usize],
    longs: Longs,
    half: Half,
    float32_filterable: bool,
    subgroups: SubgroupCaps,
) -> String {
    let manifest = &module.manifest;
    let subgroup_group = uses_subgroup_group(module, subgroups);
    let mut out = String::new();
    if subgroup_group {
        out.push_str(subgroups.directive());
    }
    emit_preamble(
        &mut out,
        module,
        true,
        packs,
        textured,
        longs,
        float32_filterable,
    );
    if subgroup_group && longs == Longs::Native && module.holds(Type::Long) {
        out.push_str(SUBGROUP_I64_HELPERS);
    }
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
    emit_functions(module, half, &mut out);
    for (index, function) in module.functions.iter().enumerate() {
        emit_function(
            &mut out,
            module,
            function,
            &function_name(index),
            false,
            packs,
            textured,
            longs,
            false,
        );
        out.push('\n');
    }
    emit_function(
        &mut out,
        module,
        &module.entry,
        "entry",
        true,
        packs,
        textured,
        longs,
        subgroup_group,
    );
    if let Some(merge) = &module.merge {
        emit_function(
            &mut out, module, merge, "merge", false, packs, textured, longs, false,
        );
    }
    if let Some(init) = &module.init {
        emit_function(
            &mut out, module, init, "init", false, packs, textured, longs, false,
        );
        // Ordered work: one invocation walks the rows, carrying the value.
        let _ = writeln!(
            out,
            "
@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x != 0u) {{
        return;
    }}
    let n = bitcast<u32>(ctx[0].x);
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
    if let Mode::Group { size } = manifest.mode {
        if subgroup_group {
            emit_group_main_subgroup(&mut out, module, size);
        } else {
            emit_group_main(&mut out, module, size);
        }
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
    let w = bitcast<u32>(ctx[0].x);
    let h = bitcast<u32>(ctx[0].y);
    let d = bitcast<u32>(ctx[1].z);
    let fold = bitcast<u32>(ctx[5].y);
    let slice = bitcast<u32>(ctx[5].w);
    var x = gid.x;
    var y = gid.y;
    var z = gid.z;
    if (fold != 0u) {{
        x = gid.x + gid.y * fold + slice;
        y = 0u;
    }} else if (d > 1u) {{
        z = gid.z + slice;
    }} else {{
        y = gid.y + slice;
    }}
    if (x < w && y < h && z < d && atomicLoad(&failure.blocks) <= failure.limit) {{
        let element = (z * h + y) * w + x;
        current = element;
        begin_work();
        let coord = vec3<f32>(f32(x), f32(y), f32(z));
        {store}
        atomicAdd(&leftover, work & {mask}u);
    }}
    workgroupBarrier();
    if (lid == 0u) {{
        owed = atomicLoad(&leftover);
        settle_left();
    }}
}}",
            mask = entelechy_ir::WORK_BLOCK - 1,
        );
        return out;
    }
    let _ = writeln!(
        out,
        "
@compute @workgroup_size({WORKGROUP}, {WORKGROUP}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let w = bitcast<u32>(ctx[0].x);
    let h = bitcast<u32>(ctx[0].y);
    let d = bitcast<u32>(ctx[1].z);
    let fold = bitcast<u32>(ctx[5].y);
    let slice = bitcast<u32>(ctx[5].w);
    var x = gid.x;
    var y = gid.y;
    var z = gid.z;
    if (fold != 0u) {{
        x = gid.x + gid.y * fold + slice;
        y = 0u;
    }} else if (d > 1u) {{
        z = gid.z + slice;
    }} else {{
        y = gid.y + slice;
    }}
    if (x >= w || y >= h || z >= d) {{
        return;
    }}
    let element = (z * h + y) * w + x;
    current = element;
    begin_work();
    let coord = vec3<f32>(f32(x), f32(y), f32(z));
    {store}
}}"
    );
    out
}

/// A group's entry point: a workgroup a group, a lane an invocation, each
/// meeting array declared once for the types exchanged. Every lane runs
/// the entry, the lanes past the domain's end too (their exchanges are
/// met by all), and only the lanes inside store.
fn emit_group_main(out: &mut String, module: &Module, size: u32) {
    // Whether a lane of the group has failed, as the lanes last met.
    let _ = writeln!(
        out,
        "var<private> group_lane: u32;\nvar<private> group_inside: bool;\nvar<workgroup> group_failed: u32;"
    );
    let mut met: Vec<Type> = Vec::new();
    exchanged(&module.entry.body, &mut met);
    for ty in met {
        let _ = writeln!(
            out,
            "var<workgroup> {}: array<{}, {size}>;",
            super::emitter::meet_name(ty),
            wgsl_type(ty)
        );
    }
    let head = format!(
        "
@compute @workgroup_size({size}, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {{
    let w = bitcast<u32>(ctx[0].x);
    let h = bitcast<u32>(ctx[0].y);
    let d = bitcast<u32>(ctx[1].z);
    let element = (wid.x + wid.y * {span}u) * {size}u + lid;
    group_lane = lid;
    group_inside = element < w * h * d;
    current = element;
    let row = element / w;
    let coord = vec3<f32>(f32(element % w), f32(row % h), f32(row / h));",
        span = MAX_GROUPS,
    );
    // One output comes back from the entry; several the entry stores
    // itself, each where the lane is inside.
    let store = if written_outputs(module) > 1 {
        "entry(coord, element);"
    } else {
        "let v = entry(coord, element);
        if (group_inside) {
            store_out0(element, v);
        }"
    };
    if !module.accounted() {
        let _ = writeln!(
            out,
            "{head}
    begin_work();
    {store}
}}"
        );
        return;
    }
    // A group whose work is counted opens together, while the run is within
    // the Host's bound, as one lane read it. A look its lanes meet at (in
    // the `until` of a loop holding an exchange) checks the lane's own
    // work, a lane past its budget failing there, and stops every lane
    // together once the group knew of a failure where its lanes last met:
    // each lane runs the stretch it is in to its end, as the reference
    // does, so the failure named is the lowest in the first stretch that
    // failed, and none waits at an exchange the others never reach. The
    // flag is written at exchanges and read between two barriers. Every
    // lane adds its remainder short of a block, the lanes past the
    // domain's end too, and one settles the sum.
    let _ = writeln!(
        out,
        "var<workgroup> group_open: u32;
fn group_look() -> bool {{
    _ = look();
    let stop = workgroupUniformLoad(&group_failed) != 0u;
    workgroupBarrier();
    return stop;
}}
{head}
    if (lid == 0u) {{
        group_open = u32(atomicLoad(&failure.blocks) <= failure.limit);
    }}
    if (workgroupUniformLoad(&group_open) != 0u) {{
        begin_work();
        {store}
        atomicAdd(&leftover, work & {mask}u);
    }}
    workgroupBarrier();
    if (lid == 0u) {{
        owed = atomicLoad(&leftover);
        settle_left();
    }}
}}",
        mask = entelechy_ir::WORK_BLOCK - 1,
    );
}

/// A group's entry point when `size` fits inside one hardware subgroup:
/// every exchange runs register-to-register via `subgroupShuffle`, failures
/// propagate via `subgroupAny`, and accounted groups gate and settle via
/// `subgroupBroadcastFirst` and `subgroupAdd` — zero workgroup memory and
/// zero `workgroupBarrier` calls.
fn emit_group_main_subgroup(out: &mut String, module: &Module, size: u32) {
    let _ = writeln!(
        out,
        "var<private> group_lane: u32;\nvar<private> group_inside: bool;\nvar<private> group_failed: bool;"
    );
    let head = format!(
        "
@compute @workgroup_size({size}, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(subgroup_invocation_id) sid: u32) {{
    let w = bitcast<u32>(ctx[0].x);
    let h = bitcast<u32>(ctx[0].y);
    let d = bitcast<u32>(ctx[1].z);
    let element = (wid.x + wid.y * {span}u) * {size}u + sid;
    group_lane = sid;
    group_inside = element < w * h * d;
    group_failed = false;
    current = element;
    let row = element / w;
    let coord = vec3<f32>(f32(element % w), f32(row % h), f32(row / h));",
        span = MAX_GROUPS,
    );
    let store = if written_outputs(module) > 1 {
        "entry(coord, element);"
    } else {
        "let v = entry(coord, element);
        if (group_inside) {
            store_out0(element, v);
        }"
    };
    if !module.accounted() {
        let _ = writeln!(
            out,
            "{head}
    begin_work();
    {store}
}}"
        );
        return;
    }
    let _ = writeln!(
        out,
        "fn group_look() -> bool {{
    _ = look();
    return group_failed;
}}
{head}
    var open_u: u32 = 0u;
    if (sid == 0u) {{
        open_u = u32(atomicLoad(&failure.blocks) <= failure.limit);
    }}
    if (subgroupBroadcastFirst(open_u) != 0u) {{
        begin_work();
        {store}
        let total_left = subgroupAdd(work & {mask}u);
        if (sid == 0u) {{
            owed = total_left;
            settle_left();
        }}
    }}
}}",
        mask = entelechy_ir::WORK_BLOCK - 1,
    );
}

/// The types of the values exchanged in `body`, each once, in order.
fn exchanged(body: &[entelechy_ir::Instruction], out: &mut Vec<Type>) {
    for instruction in body {
        if let Op::Exchange(..) = instruction.op
            && !out.contains(&instruction.ty)
        {
            out.push(instruction.ty);
        }
        for block in instruction.op.blocks() {
            exchanged(&block.body, out);
        }
    }
}

/// The bytes a group's workgroup holds at once: an array of a value for
/// each lane, of every type its lanes exchange, laid out as the shader
/// lays them, and the flags its lanes meet on.
pub fn group_memory(module: &Module) -> u64 {
    let Mode::Group { size } = module.manifest.mode else {
        return 0;
    };
    let mut met: Vec<Type> = Vec::new();
    exchanged(&module.entry.body, &mut met);
    let arrays: u64 = met
        .into_iter()
        .map(|ty| {
            let (bytes, align) = layout(module, ty);
            bytes.next_multiple_of(align) * u64::from(size)
        })
        .sum();
    // The flag of a failure met, and a counted group's gate and sum.
    let flags = if module.accounted() { 12 } else { 4 };
    arrays + flags
}

/// The bytes a group's workgroup holds at once on a device with `subgroups`:
/// zero when the group runs register-only via `subgroupShuffle`, otherwise
/// [`group_memory`].
pub fn group_memory_for(module: &Module, subgroups: SubgroupCaps) -> u64 {
    if uses_subgroup_group(module, subgroups) {
        0
    } else {
        group_memory(module)
    }
}

/// A value's size and alignment in a shader's memory, by WGSL's layout:
/// a three-vector aligned as a four, a struct to its widest field.
pub(crate) fn layout(module: &Module, ty: Type) -> (u64, u64) {
    let scalar = |kind: entelechy_abi::Scalar| -> u64 {
        if matches!(
            kind,
            entelechy_abi::Scalar::Long | entelechy_abi::Scalar::Double
        ) {
            8
        } else {
            4
        }
    };
    let vector = |kind, n: u8| {
        let c = scalar(kind);
        match n {
            2 => (2 * c, 2 * c),
            3 => (3 * c, 4 * c),
            _ => (4 * c, 4 * c),
        }
    };
    match ty {
        Type::Scalar(kind) => (scalar(kind), scalar(kind)),
        Type::Vector(kind, n) => vector(kind, n),
        Type::Matrix(kind, n) => {
            let (column, align) = vector(kind, n);
            (u64::from(n) * column.next_multiple_of(align), align)
        }
        Type::Struct(index) => {
            let mut offset = 0u64;
            let mut widest = 1u64;
            for field in &module.structs[index as usize].fields {
                let (bytes, align) = layout(module, field.ty);
                offset = offset.next_multiple_of(align) + bytes;
                widest = widest.max(align);
            }
            (offset.next_multiple_of(widest), widest)
        }
    }
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
