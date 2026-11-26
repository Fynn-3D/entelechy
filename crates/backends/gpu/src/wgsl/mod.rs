//! WGSL emission from validated Entelechy IR.
//!
//! A value slot is a WGSL `let`, or a `var` for `if` and `loop` results;
//! IR blocks are WGSL blocks. Resources are storage buffers of `u32`
//! words, one per scalar, plus uniform arrays for context and parameters.

use std::fmt::Write;

use entelechy_abi::{Encoding, Scalar, encoding};
use entelechy_ir::{
    BinaryOp, Block, Combine, Function, Instruction, Intrinsic, Mode, Module, Op, Shape, Type,
    UnaryOp, Value, ValueId,
};

mod emitter;
mod map;
mod prefix;
mod prelude;
mod prune;
mod reduce;
mod scatter;
mod storage;

use emitter::*;
pub use map::*;
pub use prefix::*;
use prelude::*;
pub use prune::pruned;
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
/// ([`entelechy_rules::Run`]).
pub const FAILURE_BINDING: u32 = 90;

/// Binding index of the linear ClampToEdge hardware sampler (`anisotropy = 1`).
pub const SAMPLER_CLAMP_BINDING: u32 = 97;
/// Binding index of the linear Repeat (X/Y) / ClampToEdge (Z) hardware sampler (`anisotropy = 1`).
pub const SAMPLER_REPEAT_BINDING: u32 = 98;
/// Binding index of the 16x anisotropic ClampToEdge hardware sampler.
pub const SAMPLER_ANISO_CLAMP_BINDING: u32 = 99;
/// Binding index of the 16x anisotropic Repeat (X/Y) / ClampToEdge (Z) hardware sampler.
pub const SAMPLER_ANISO_REPEAT_BINDING: u32 = 100;

/// Hardware subgroup (wave / SIMD-group) capabilities reported by a wgpu
/// adapter. When `enabled` is false, every WGSL emitter uses pure workgroup
/// shared-memory or scalar fallbacks with no subgroup intrinsics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SubgroupCaps {
    /// Whether the device enabled `wgpu::Features::SUBGROUP`.
    pub enabled: bool,
    /// Minimum subgroup lane count guaranteed on this adapter.
    pub min_size: u32,
    /// Maximum subgroup lane count possible on this adapter.
    pub max_size: u32,
    /// Whether the target WGSL frontend requires `enable subgroups;` (true on
    /// browser WebGPU; false on native Naga 30, which gates subgroup builtins
    /// by `wgpu::Features::SUBGROUP` and rejects the `enable` directive).
    pub enable_directive: bool,
}

impl SubgroupCaps {
    /// Leading WGSL enable directive when required by the target frontend.
    pub const fn directive(self) -> &'static str {
        if self.enable_directive {
            "enable subgroups;\n"
        } else {
            ""
        }
    }

    /// Whether a `group N` Kernel of `size` lanes fits inside one hardware
    /// subgroup, so `exchange` runs register-to-register via `subgroupShuffle`
    /// with zero workgroup memory and zero barriers.
    pub const fn fits_single_wave(self, size: u32) -> bool {
        self.enabled && size > 0 && size <= self.min_size
    }

    /// Whether every subgroup on this device is guaranteed to be 32 lanes wide
    /// (Apple Silicon Metal, NVIDIA), allowing 32-bit `subgroupBallot` masks
    /// to map 1:1 to `u32` bitmask words with zero workgroup atomics.
    pub const fn exact_wave_32(self) -> bool {
        self.enabled && self.min_size == 32 && self.max_size == 32
    }

    /// Whether a 256-lane workgroup decomposes into at most 64 power-of-two
    /// hardware waves (`4 <= min_size <= max_size <= 256`).
    pub const fn wave_parallel_256(self) -> bool {
        self.enabled
            && self.min_size >= 4
            && self.max_size <= 256
            && self.min_size <= self.max_size
            && self.min_size.is_power_of_two()
            && self.max_size.is_power_of_two()
    }
}

/// Maximum scalar words per exchanged value for register-only `subgroupShuffle`
/// in `group N` Kernels; wider arrays stay in workgroup shared memory to avoid
/// excessive register pressure.
pub(crate) const SUBGROUP_SHUFFLE_MAX_WORDS: usize = 16;

/// WGSL helpers splitting 64-bit `i64` across a `vec2<u32>` subgroup shuffle so
/// drivers whose hardware shuffle intrinsics only take 32-bit words (such as
/// Apple Metal) compile cleanly.
pub(crate) const SUBGROUP_I64_HELPERS: &str = "
fn sg_shuffle_i64(v: i64, l: u32) -> i64 {
    let u = bitcast<u64>(v);
    let p = subgroupShuffle(vec2<u32>(u32(u), u32(u >> 32u)), l);
    return bitcast<i64>(u64(p.x) | (u64(p.y) << 32u));
}
fn sg_shuffle_down_i64(v: i64, d: u32) -> i64 {
    let u = bitcast<u64>(v);
    let p = subgroupShuffleDown(vec2<u32>(u32(u), u32(u >> 32u)), d);
    return bitcast<i64>(u64(p.x) | (u64(p.y) << 32u));
}
";

/// Emits a WGSL subgroup shuffle expression (`subgroupShuffle` or
/// `subgroupShuffleDown`) for any Entelechy type, decomposing booleans,
/// native 64-bit `Long`s, matrices, and structs into 32-bit hardware shuffles.
pub(crate) fn subgroup_shuffle_expr(
    module: &Module,
    ty: Type,
    expr: &str,
    lane: &str,
    op: &str,
    pairs: bool,
) -> String {
    let i64_fn = if op == "subgroupShuffleDown" {
        "sg_shuffle_down_i64"
    } else {
        "sg_shuffle_i64"
    };
    match ty {
        Type::Scalar(Scalar::Float | Scalar::Int) => format!("{op}({expr}, {lane})"),
        Type::Scalar(Scalar::Bool) => format!("({op}(select(0u, 1u, {expr}), {lane}) != 0u)"),
        Type::Scalar(Scalar::Long) => {
            if pairs {
                format!("{op}({expr}, {lane})")
            } else {
                format!("{i64_fn}({expr}, {lane})")
            }
        }
        Type::Scalar(Scalar::Double) => unreachable!("the GPU refuses a module holding a Double"),
        Type::Vector(Scalar::Float | Scalar::Int, _) => format!("{op}({expr}, {lane})"),
        Type::Vector(Scalar::Bool, n) => format!(
            "({op}(select(vec{n}<u32>(0u), vec{n}<u32>(1u), {expr}), {lane}) != vec{n}<u32>(0u))"
        ),
        Type::Vector(Scalar::Long, n) => {
            let comps = ["x", "y", "z", "w"];
            let parts = comps
                .iter()
                .take(usize::from(n))
                .map(|c| format!("{i64_fn}({expr}.{c}, {lane})"))
                .collect::<Vec<_>>()
                .join(", ");
            format!("vec{n}<i64>({parts})")
        }
        Type::Vector(Scalar::Double, _) => {
            unreachable!("the GPU refuses a module holding a Double")
        }
        Type::Matrix(kind, n) => {
            let col_ty = Type::Vector(kind, n);
            let cols = (0..n)
                .map(|c| {
                    subgroup_shuffle_expr(module, col_ty, &format!("{expr}[{c}u]"), lane, op, pairs)
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("mat{n}x{n}<{}>({cols})", scalar_type(kind))
        }
        Type::Struct(index) => {
            let fields = module.structs[index as usize]
                .fields
                .iter()
                .enumerate()
                .map(|(k, field)| {
                    subgroup_shuffle_expr(
                        module,
                        field.ty,
                        &format!("{expr}{}", field_path(module, index, k)),
                        lane,
                        op,
                        pairs,
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("S{index}({fields})")
        }
    }
}

/// Which hardware `wgpu::Sampler`s a module's WGSL shader declares and binds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SamplerSet {
    /// Linear ClampToEdge trilinear sampler (`SAMPLER_CLAMP_BINDING`).
    pub clamp: bool,
    /// Linear Repeat (X/Y) / ClampToEdge (Z) trilinear sampler (`SAMPLER_REPEAT_BINDING`).
    pub repeat: bool,
    /// 16x anisotropic ClampToEdge sampler (`SAMPLER_ANISO_CLAMP_BINDING`).
    pub aniso_clamp: bool,
    /// 16x anisotropic Repeat (X/Y) / ClampToEdge (Z) sampler (`SAMPLER_ANISO_REPEAT_BINDING`).
    pub aniso_repeat: bool,
}

/// Summary of `Op::AtFiltered` uses across a module's functions.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FilteredUses {
    /// Whether the module contains any `Op::AtFiltered` instruction.
    pub any: bool,
    /// Inputs sampled via 2-arg `at_repeat(field, pos)` (mip 0).
    pub repeat_2arg: Vec<usize>,
    /// Inputs sampled via 3-arg `at_linear(field, pos, lod)`.
    pub clamp_lod: Vec<usize>,
    /// Inputs sampled via 3-arg `at_repeat(field, pos, lod)`.
    pub repeat_lod: Vec<usize>,
    /// Inputs sampled via 4-arg `at_linear(field, pos, ddx, ddy)`.
    pub clamp_grad: Vec<usize>,
    /// Inputs sampled via 4-arg `at_repeat(field, pos, ddx, ddy)`.
    pub repeat_grad: Vec<usize>,
    /// Inputs requiring full hardware mipmap chains (`lod` or `grad`).
    pub mipmapped: Vec<usize>,
    /// `Volume` inputs sampled with 2D gradients (`grad`), which must bind as
    /// `texture_2d_array<f32>` so WGSL `textureSampleGrad` accepts `vec2<f32>`.
    pub grad_volumes: Vec<usize>,
}

/// Scans `module` for all `Op::AtFiltered` instructions across entry and helper functions.
pub fn filtered_uses(module: &Module) -> FilteredUses {
    #[derive(Default)]
    struct Acc {
        any: bool,
        repeat_2arg: std::collections::BTreeSet<usize>,
        clamp_lod: std::collections::BTreeSet<usize>,
        repeat_lod: std::collections::BTreeSet<usize>,
        clamp_grad: std::collections::BTreeSet<usize>,
        repeat_grad: std::collections::BTreeSet<usize>,
        mipmapped: std::collections::BTreeSet<usize>,
        grad_volumes: std::collections::BTreeSet<usize>,
    }

    fn walk(body: &[Instruction], manifest: &entelechy_ir::Manifest, acc: &mut Acc) {
        for instr in body {
            if let Op::AtFiltered {
                input,
                repeat,
                lod,
                grad,
                ..
            } = &instr.op
            {
                acc.any = true;
                let idx = *input as usize;
                if grad.is_some() {
                    acc.mipmapped.insert(idx);
                    if *repeat {
                        acc.repeat_grad.insert(idx);
                    } else {
                        acc.clamp_grad.insert(idx);
                    }
                    if manifest
                        .inputs
                        .get(idx)
                        .is_some_and(|p| p.shape() == Shape::Volume)
                    {
                        acc.grad_volumes.insert(idx);
                    }
                } else if lod.is_some() {
                    acc.mipmapped.insert(idx);
                    if *repeat {
                        acc.repeat_lod.insert(idx);
                    } else {
                        acc.clamp_lod.insert(idx);
                    }
                } else if *repeat {
                    acc.repeat_2arg.insert(idx);
                }
            }
            for block in instr.op.blocks() {
                walk(&block.body, manifest, acc);
            }
        }
    }

    let mut acc = Acc::default();
    for func in module
        .functions
        .iter()
        .chain([&module.entry])
        .chain(module.merge.iter())
        .chain(module.init.iter())
    {
        walk(&func.body, &module.manifest, &mut acc);
    }

    FilteredUses {
        any: acc.any,
        repeat_2arg: acc.repeat_2arg.into_iter().collect(),
        clamp_lod: acc.clamp_lod.into_iter().collect(),
        repeat_lod: acc.repeat_lod.into_iter().collect(),
        clamp_grad: acc.clamp_grad.into_iter().collect(),
        repeat_grad: acc.repeat_grad.into_iter().collect(),
        mipmapped: acc.mipmapped.into_iter().collect(),
        grad_volumes: acc.grad_volumes.into_iter().collect(),
    }
}

/// Whether textured input `index` supports hardware filtering samplers on a
/// device with `float32_filterable`.
pub fn input_is_filterable(module: &Module, index: usize, float32_filterable: bool) -> bool {
    input_encoding(module, index) != Encoding::Word || float32_filterable
}

/// Whether `Volume` input `index` binds as `texture_2d_array<f32>` (true for
/// `Unorm8` streamed stacks and any `Volume` sampled with 2D gradients) rather
/// than `texture_3d<f32>`.
pub fn input_is_2d_array(module: &Module, index: usize, uses: &FilteredUses) -> bool {
    module.manifest.inputs[index].shape() == Shape::Volume
        && (input_encoding(module, index) == Encoding::Unorm8 || uses.grad_volumes.contains(&index))
}

/// Hardware samplers required by `module` given `textured` inputs and
/// `float32_filterable`.
pub fn samplers_for(module: &Module, textured: &[usize], float32_filterable: bool) -> SamplerSet {
    let uses = filtered_uses(module);
    let mut set = SamplerSet::default();
    for &index in textured {
        if !input_is_filterable(module, index, float32_filterable) {
            continue;
        }
        if uses.clamp_lod.contains(&index) {
            set.clamp = true;
        }
        if uses.repeat_lod.contains(&index) || uses.repeat_2arg.contains(&index) {
            set.repeat = true;
        }
        if uses.clamp_grad.contains(&index) {
            set.aniso_clamp = true;
        }
        if uses.repeat_grad.contains(&index) {
            set.aniso_repeat = true;
        }
    }
    set
}

/// How a shader computes a `Long`: as the device's own 64-bit integer, or,
/// on a device without one (a page's WebGPU), as a pair of 32-bit words
/// with every operation a helper ([`pairs_take`] says which modules can).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Longs {
    /// The device's own 64-bit integer, `i64`.
    Native,
    /// Two 32-bit words, low first, every operation a helper.
    Pairs,
}

/// Whether a module runs with its Longs in pairs of words: a map, a group, a
/// reduction or a scatter of Longs under `min` or `max` ([`Half`]) whose
/// Longs are all scalars (no Long vector or matrix), whose
/// `match` never chooses by a Long, and whose intrinsics on Longs are the
/// conversions, `sign`, `abs`, `min`, `max`, `clamp` and the wrapping
/// arithmetic. Anything else still goes to the ordered backend.
pub fn pairs_take(module: &Module) -> bool {
    // A scatter of Longs under `min` or `max` lands them by halves.
    if !(matches!(
        module.manifest.mode,
        Mode::Map | Mode::Group { .. } | Mode::Reduce
    ) || long_extremes(module))
    {
        return false;
    }
    let long_lanes = |ty: Type| {
        matches!(
            ty,
            Type::Vector(Scalar::Long, _) | Type::Matrix(Scalar::Long, _)
        )
    };
    let manifest = &module.manifest;
    if manifest
        .inputs
        .iter()
        .chain(&manifest.outputs)
        .any(|port| long_lanes(port.element()))
        || manifest.params.iter().any(|param| long_lanes(param.ty))
        || module
            .structs
            .iter()
            .any(|def| def.fields.iter().any(|field| long_lanes(field.ty)))
    {
        return false;
    }
    fn takes(function: &Function, long_lanes: &dyn Fn(Type) -> bool) -> bool {
        let mut types = std::collections::HashMap::new();
        fn gather(body: &[Instruction], out: &mut std::collections::HashMap<u32, Type>) {
            for instruction in body {
                out.insert(instruction.id.0, instruction.ty);
                for block in instruction.op.blocks() {
                    gather(&block.body, out);
                }
            }
        }
        gather(&function.body, &mut types);
        let long = |id: &ValueId| types.get(&id.0) == Some(&Type::Long);
        fn walk(
            body: &[Instruction],
            long: &dyn Fn(&ValueId) -> bool,
            long_lanes: &dyn Fn(Type) -> bool,
        ) -> bool {
            body.iter().all(|instruction| {
                let fits = !long_lanes(instruction.ty)
                    && match &instruction.op {
                        Op::Match { scrutinee, .. } => !long(scrutinee),
                        Op::Intrinsic(intrinsic, args) => {
                            !(instruction.ty == Type::Long || args.iter().any(long))
                                || matches!(
                                    intrinsic,
                                    Intrinsic::ToFloat
                                        | Intrinsic::ToInt
                                        | Intrinsic::ToLong
                                        | Intrinsic::Sign
                                        | Intrinsic::Abs
                                        | Intrinsic::Min
                                        | Intrinsic::Max
                                        | Intrinsic::Clamp
                                        | Intrinsic::AddWrap
                                        | Intrinsic::SubWrap
                                        | Intrinsic::MulWrap
                                )
                        }
                        _ => true,
                    };
                fits && instruction
                    .op
                    .blocks()
                    .iter()
                    .all(|block| walk(&block.body, long, long_lanes))
            })
        }
        !function.args.iter().any(|ty| long_lanes(*ty)) && walk(&function.body, &long, long_lanes)
    }
    module
        .functions
        .iter()
        .chain([&module.entry])
        .chain(module.merge.iter())
        .chain(module.init.iter())
        .all(|function| takes(function, &long_lanes))
}

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
                Op::Each { body, .. } => 1 + nest(&body.body).max(2),
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

/// Binding index of the first buffer that packs inputs; see [`packed_inputs`].
pub const PACKED_BINDING: u32 = 91;
/// Binding index of the second packed-input buffer (`inputs_packed_1`);
/// further pack buffers follow consecutively up to [`MAX_PACK_BUFFERS`].
pub const EXTRA_PACKED_BINDING: u32 = 93;
/// Maximum number of packed-input storage buffers a wide Kernel splits its
/// packed read-only inputs across so each stays well under WebGPU's `128 MiB`
/// `maxStorageBufferBindingSize` ceiling.
pub const MAX_PACK_BUFFERS: usize = 4;

/// Binding index of packed-input buffer `pack` (`0..MAX_PACK_BUFFERS`).
pub const fn packed_binding(pack: usize) -> u32 {
    if pack == 0 {
        PACKED_BINDING
    } else {
        EXTRA_PACKED_BINDING + (pack as u32 - 1)
    }
}

/// Whether `binding` is one of the packed-input storage bindings.
pub const fn is_packed_binding(binding: u32) -> bool {
    binding == PACKED_BINDING
        || (binding >= EXTRA_PACKED_BINDING
            && binding < EXTRA_PACKED_BINDING + (MAX_PACK_BUFFERS as u32 - 1))
}

pub(crate) fn packed_buffer_name(pack: usize) -> String {
    if pack == 0 {
        "inputs_packed".to_owned()
    } else {
        format!("inputs_packed_{pack}")
    }
}

/// `(uniforms, storage)` buffers the module's widest pass binds unpacked:
/// context and parameters, every input, every written output, the
/// failure record where arithmetic can fail, an emitting scatter's target
/// and its high words where it sums integers whole.
pub fn bindings(module: &Module) -> (usize, usize) {
    let inputs = module.manifest.inputs.len();
    let failure = usize::from(can_fail(module));
    // A reduction's widest pass is its fold: a third uniform (the level's
    // count) and the level it reads beside the one it writes.
    if module.manifest.mode == Mode::Reduce {
        return (3, inputs + 2 + failure);
    }
    let emitting =
        usize::from(!emitted_pairs(module).is_empty()) * (1 + usize::from(highs_beside(module)));
    let storage = inputs + written_outputs(module) + failure + emitting;
    (2, storage)
}

/// Read-only `Grid` and `Volume` inputs eligible for hardware 2D/3D texture
/// mirroring (`texture_2d<f32>` / `texture_3d<f32>`), capped at `max_textures`.
///
/// Selects read-only non-memory `Float`/`Vec2`/`Vec4` inputs stored in a
/// hardware-native float/normalized encoding (`Word`, `F16`, `Unorm8`, `Snorm8`)
/// that are never read via `Op::Part`, prioritizing workloads where hardware
/// Morton tiling and 1-cycle unpack outweigh buffer-to-texture mirroring:
/// - Any 3D `Volume` read via `Op::AtLinear` or `Op::At`.
/// - Any narrow-encoded (`F16`, `Unorm8`, `Snorm8`) 2D `Grid` read via
///   `Op::AtLinear` or inside a loop via `Op::At`.
/// - Any wide (`Word`) 2D `Grid` sampled via `Op::AtLinear` inside a loop
///   (including through helper calls).
pub fn textured_inputs(module: &Module, max_textures: usize) -> Vec<usize> {
    textured_inputs_for_limits(module, max_textures, usize::MAX, usize::MAX)
}

/// Like [`textured_inputs`], also promoting non-loop wide 2D `Grid` `AtLinear`
/// inputs into free texture slots whenever doing so keeps storage bindings
/// within `(max_buffers, max_storage)` without packing copies.
pub fn textured_inputs_for_limits(
    module: &Module,
    max_textures: usize,
    max_buffers: usize,
    max_storage: usize,
) -> Vec<usize> {
    if max_textures == 0 || !matches!(module.manifest.mode, Mode::Map | Mode::Group { .. }) {
        return Vec::new();
    }
    #[derive(Default)]
    struct FuncReads {
        linear: std::collections::BTreeSet<usize>,
        mipmapped: std::collections::BTreeSet<usize>,
        at: std::collections::BTreeSet<usize>,
        loop_linear: std::collections::BTreeSet<usize>,
        loop_at: std::collections::BTreeSet<usize>,
        part: std::collections::BTreeSet<usize>,
        calls: std::collections::BTreeSet<usize>,
        loop_calls: std::collections::BTreeSet<usize>,
    }
    fn walk(body: &[Instruction], in_loop: bool, out: &mut FuncReads) {
        for instr in body {
            match &instr.op {
                Op::AtLinear(idx, _) => {
                    out.linear.insert(*idx as usize);
                    if in_loop {
                        out.loop_linear.insert(*idx as usize);
                    }
                }
                Op::AtFiltered {
                    input, lod, grad, ..
                } => {
                    let idx = *input as usize;
                    out.linear.insert(idx);
                    if lod.is_some() || grad.is_some() {
                        out.mipmapped.insert(idx);
                    }
                    if in_loop {
                        out.loop_linear.insert(idx);
                    }
                }
                Op::At(idx, _) => {
                    out.at.insert(*idx as usize);
                    if in_loop {
                        out.loop_at.insert(*idx as usize);
                    }
                }
                Op::Part { input, .. } => {
                    out.part.insert(*input as usize);
                }
                Op::Call(callee, _) => {
                    out.calls.insert(*callee as usize);
                    if in_loop {
                        out.loop_calls.insert(*callee as usize);
                    }
                }
                _ => {}
            }
            let child_loop = in_loop || matches!(instr.op, Op::Loop { .. } | Op::Each { .. });
            for block in instr.op.blocks() {
                walk(&block.body, child_loop, out);
            }
        }
    }
    let fn_count = module.functions.len();
    let mut funcs: Vec<FuncReads> = (0..=fn_count).map(|_| FuncReads::default()).collect();
    for (idx, func) in module.functions.iter().enumerate() {
        walk(&func.body, false, &mut funcs[idx]);
    }
    walk(&module.entry.body, false, &mut funcs[fn_count]);

    // Propagate loop context through helper calls (skipping `#tile_r*`
    // functions, which lower to 1-cycle workgroup SRAM lookups inside loops).
    let mut in_loop_fn = vec![false; fn_count];
    let mut stack: Vec<usize> = funcs
        .iter()
        .flat_map(|f| f.loop_calls.iter().copied())
        .filter(|&c| {
            c < fn_count && entelechy_ir::tile_fn_radius(&module.functions[c].name).is_none()
        })
        .collect();
    while let Some(c) = stack.pop() {
        if !in_loop_fn[c] {
            in_loop_fn[c] = true;
            for &next in &funcs[c].calls {
                if next < fn_count && !in_loop_fn[next] {
                    stack.push(next);
                }
            }
        }
    }

    let mut linear_used: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    let mut mipmapped_used: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    let mut at_used: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    let mut loop_linear: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    let mut loop_at: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    let mut part_used: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for (idx, f) in funcs.iter().enumerate() {
        linear_used.extend(f.linear.iter().copied());
        mipmapped_used.extend(f.mipmapped.iter().copied());
        at_used.extend(f.at.iter().copied());
        loop_linear.extend(f.loop_linear.iter().copied());
        loop_at.extend(f.loop_at.iter().copied());
        part_used.extend(f.part.iter().copied());
        if idx < fn_count && in_loop_fn[idx] {
            loop_linear.extend(f.linear.iter().copied());
            loop_at.extend(f.at.iter().copied());
        }
    }

    let mut mipmapped = Vec::new();
    let mut volumes = Vec::new();
    let mut primary_grids = Vec::new();
    let mut spill_grids = Vec::new();
    for (index, port) in module.manifest.inputs.iter().enumerate() {
        if module.manifest.memories.contains(&port.name) || part_used.contains(&index) {
            continue;
        }
        if !matches!(port.element(), Type::Float | Type::Vec2 | Type::Vec4) {
            continue;
        }
        let enc = input_encoding(module, index);
        if !matches!(
            enc,
            Encoding::Word | Encoding::F16 | Encoding::Unorm8 | Encoding::Snorm8
        ) {
            continue;
        }
        if mipmapped_used.contains(&index) && matches!(port.shape(), Shape::Grid | Shape::Volume) {
            mipmapped.push(index);
            continue;
        }
        let narrow = enc != Encoding::Word;
        match port.shape() {
            Shape::Volume if linear_used.contains(&index) || at_used.contains(&index) => {
                volumes.push(index);
            }
            Shape::Grid
                if (enc == Encoding::Unorm8
                    && (linear_used.contains(&index) || at_used.contains(&index)))
                    || (narrow && (linear_used.contains(&index) || loop_at.contains(&index)))
                    || loop_linear.contains(&index) =>
            {
                primary_grids.push(index);
            }
            Shape::Grid if linear_used.contains(&index) => {
                spill_grids.push(index);
            }
            _ => {}
        }
    }
    let mut chosen: Vec<usize> = mipmapped
        .into_iter()
        .chain(volumes)
        .chain(primary_grids)
        .take(max_textures)
        .collect();
    if chosen.len() < max_textures && !spill_grids.is_empty() {
        let (uniforms, total_storage) = bindings(module);
        let remaining_storage = total_storage.saturating_sub(chosen.len());
        let over = (uniforms + remaining_storage)
            .saturating_sub(max_buffers)
            .max(remaining_storage.saturating_sub(max_storage));
        if over > 0 {
            let room = (max_textures - chosen.len()).min(over);
            chosen.extend(spill_grids.into_iter().take(room));
        }
    }
    chosen.sort_unstable();
    chosen
}

/// Subset of `textured` inputs that are plain non-mipmapped 2D `Grid`s read
/// only outside loops. Static (`ONCE` / Host-uploaded) Grids benefit from
/// mirroring once, whereas freshly-produced per-frame dynamic Grids avoid a
/// redundant full-grid `RowPad + copy_buffer_to_texture` copy every frame by
/// sampling directly from L2-cached storage buffers.
pub fn plain_2d_linear_inputs(module: &Module, textured: &[usize]) -> Vec<usize> {
    if textured.is_empty() || !matches!(module.manifest.mode, Mode::Map | Mode::Group { .. }) {
        return Vec::new();
    }
    #[derive(Default)]
    struct Reads {
        mipmapped: std::collections::BTreeSet<usize>,
        loop_read: std::collections::BTreeSet<usize>,
        calls: std::collections::BTreeSet<usize>,
        loop_calls: std::collections::BTreeSet<usize>,
    }
    fn walk(body: &[Instruction], in_loop: bool, out: &mut Reads) {
        for instr in body {
            match &instr.op {
                Op::AtLinear(idx, _) | Op::At(idx, _) => {
                    if in_loop {
                        out.loop_read.insert(*idx as usize);
                    }
                }
                Op::AtFiltered {
                    input, lod, grad, ..
                } => {
                    let idx = *input as usize;
                    if lod.is_some() || grad.is_some() {
                        out.mipmapped.insert(idx);
                    }
                    if in_loop {
                        out.loop_read.insert(idx);
                    }
                }
                Op::Call(callee, _) => {
                    out.calls.insert(*callee as usize);
                    if in_loop {
                        out.loop_calls.insert(*callee as usize);
                    }
                }
                _ => {}
            }
            let child_loop = in_loop || matches!(instr.op, Op::Loop { .. } | Op::Each { .. });
            for block in instr.op.blocks() {
                walk(&block.body, child_loop, out);
            }
        }
    }
    let fn_count = module.functions.len();
    let mut funcs: Vec<Reads> = (0..=fn_count).map(|_| Reads::default()).collect();
    for (idx, func) in module.functions.iter().enumerate() {
        walk(&func.body, false, &mut funcs[idx]);
    }
    walk(&module.entry.body, false, &mut funcs[fn_count]);
    let mut in_loop_fn = vec![false; fn_count];
    let mut stack: Vec<usize> = funcs
        .iter()
        .flat_map(|f| f.loop_calls.iter().copied())
        .filter(|&c| {
            c < fn_count && entelechy_ir::tile_fn_radius(&module.functions[c].name).is_none()
        })
        .collect();
    while let Some(c) = stack.pop() {
        if !in_loop_fn[c] {
            in_loop_fn[c] = true;
            for &next in &funcs[c].calls {
                if next < fn_count
                    && !in_loop_fn[next]
                    && entelechy_ir::tile_fn_radius(&module.functions[next].name).is_none()
                {
                    stack.push(next);
                }
            }
        }
    }
    let mut mipmapped = std::collections::BTreeSet::new();
    let mut loop_read = std::collections::BTreeSet::new();
    for (idx, f) in funcs.iter().enumerate() {
        mipmapped.extend(f.mipmapped.iter().copied());
        loop_read.extend(f.loop_read.iter().copied());
        if idx < fn_count && in_loop_fn[idx] {
            // Propagate any read inside a loop-called helper.
            fn walk_any(body: &[Instruction], out: &mut std::collections::BTreeSet<usize>) {
                for instr in body {
                    match &instr.op {
                        Op::AtLinear(i, _) | Op::At(i, _) | Op::AtFiltered { input: i, .. } => {
                            out.insert(*i as usize);
                        }
                        _ => {}
                    }
                    for block in instr.op.blocks() {
                        walk_any(&block.body, out);
                    }
                }
            }
            walk_any(&module.functions[idx].body, &mut loop_read);
        }
    }
    textured
        .iter()
        .copied()
        .filter(|&idx| {
            module.manifest.inputs[idx].shape() == Shape::Grid
                && !mipmapped.contains(&idx)
                && !loop_read.contains(&idx)
        })
        .collect()
}

fn packing_candidates(module: &Module, textured: &[usize]) -> (Vec<usize>, usize) {
    let inputs = &module.manifest.inputs;
    let read_only =
        |k: &usize| !module.manifest.memories.contains(&inputs[*k].name) && !textured.contains(k);
    let rows = |k: &usize| inputs[*k].shape() == Shape::Rows;
    let rows_cands: Vec<usize> = (0..inputs.len())
        .rev()
        .filter(|k| read_only(k) && rows(k))
        .collect();
    let rows_count = rows_cands.len();
    let mut wide_cands: Vec<usize> = (0..inputs.len())
        .filter(|k| read_only(k) && !rows(k))
        .collect();
    // Pack narrower element schemas (`Float` / `Int` grids and volumes) before
    // wide `Vec4` / struct textures so multi-megabyte maps stay resident in
    // their own zero-copy bindings whenever slots allow.
    wide_cands.sort_by_key(|&k| (inputs[k].ty.schema.value_words(), std::cmp::Reverse(k)));
    let mut candidates = rows_cands;
    candidates.extend(wide_cands);
    (candidates, rows_count)
}

/// The inputs packed into one buffer so the module binds at most `most`
/// buffers and `most_storage` storage buffers: the fewest that fit, read-
/// only inputs only (a memory keeps its own), rows before grids and
/// volumes, narrower elements first, and the last declared first within each.
/// Packing `k` inputs saves `k - 1` bindings. Empty when the module fits,
/// or when packing every read-only input is still too wide.
pub fn packed_inputs(module: &Module, most: usize, most_storage: usize) -> Vec<usize> {
    let (uniforms, storage) = bindings(module);
    let over = (uniforms + storage)
        .saturating_sub(most)
        .max(storage.saturating_sub(most_storage));
    if over == 0 {
        return Vec::new();
    }
    let (candidates, _) = packing_candidates(module, &[]);
    if candidates.len() < over + 1 {
        return Vec::new();
    }
    let mut packed: Vec<usize> = candidates[..over + 1].to_vec();
    packed.sort_unstable();
    packed
}

/// Packed read-only input groups (`1..=MAX_PACK_BUFFERS` buffers) so `module`
/// fits within `(most, most_storage)` while spreading packed grids and volumes
/// across multiple buffers to stay under WebGPU's `128 MiB` per-binding limit.
pub fn packed_input_groups(module: &Module, most: usize, most_storage: usize) -> Vec<Vec<usize>> {
    packed_input_groups_with_textures(module, most, most_storage, &[])
}

/// Like [`packed_input_groups`], excluding inputs bound as hardware textures
/// (`textured`) from storage-buffer counts and pack candidates.
pub fn packed_input_groups_with_textures(
    module: &Module,
    most: usize,
    most_storage: usize,
    textured: &[usize],
) -> Vec<Vec<usize>> {
    let (uniforms, total_storage) = bindings(module);
    let storage = total_storage.saturating_sub(textured.len());
    let over = (uniforms + storage)
        .saturating_sub(most)
        .max(storage.saturating_sub(most_storage));
    if over == 0 {
        return Vec::new();
    }
    let inputs = &module.manifest.inputs;
    let (candidates, rows_count) = packing_candidates(module, textured);
    if candidates.len() < over + 1 {
        return Vec::new();
    }
    let buffer_inputs = inputs.len().saturating_sub(textured.len());
    let fixed_storage = storage.saturating_sub(buffer_inputs);
    let avail_slots = most_storage
        .saturating_sub(fixed_storage)
        .min(most.saturating_sub(uniforms + fixed_storage));
    let max_packs = MAX_PACK_BUFFERS
        .min(avail_slots)
        .min(candidates.len().saturating_sub(over));
    let mut pack_count = 1usize;
    while pack_count < max_packs {
        let total = over + pack_count;
        let grids = total.saturating_sub(rows_count);
        if grids <= pack_count * 3 {
            break;
        }
        pack_count += 1;
    }
    if pack_count <= 1 {
        let mut single = candidates[..over + 1].to_vec();
        single.sort_unstable();
        return vec![single];
    }
    let chosen = &candidates[..over + pack_count];
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); pack_count];
    let mut wide_chosen = Vec::new();
    for &idx in chosen {
        if inputs[idx].shape() == Shape::Rows {
            groups[0].push(idx);
        } else {
            wide_chosen.push(idx);
        }
    }
    wide_chosen.sort_by_key(|&k| (inputs[k].ty.schema.value_words(), k));
    for (slot, idx) in wide_chosen.into_iter().enumerate() {
        groups[slot % pack_count].push(idx);
    }
    for group in &mut groups {
        group.sort_unstable();
    }
    groups.retain(|group| !group.is_empty());
    groups
}

/// `(buffer, offset prefix)` input `index` reads from: its own buffer, or
/// its assigned pack buffer at the offset the parameters carry, in the input's
/// load unit (words, or bytes where it is narrow).
fn input_source(
    module: &Module,
    packs: &[Vec<usize>],
    index: usize,
    encoding: Encoding,
) -> (String, String) {
    let mut base_slot = 0usize;
    for (pack_idx, group) in packs.iter().enumerate() {
        if let Some(pos) = group.iter().position(|&k| k == index) {
            let slot = base_slot + pos;
            let first = param_slots(module).1;
            let offset = format!("params[{}u][{}u]", first + slot / 4, slot % 4);
            let offset = if encoding == Encoding::Word {
                offset
            } else {
                format!("{offset} * 4u")
            };
            return (packed_buffer_name(pack_idx), format!("{offset} + "));
        }
        base_slot += group.len();
    }
    (format!("in{index}"), String::new())
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
    combine == Combine::Sum
        && scalars
            .iter()
            .any(|s| matches!(s, entelechy_abi::Scalar::Int))
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

/// Outputs the main pass writes: every output of a map or a group, each
/// in its own buffer; one for other modes.
pub fn written_outputs(module: &Module) -> usize {
    match module.manifest.mode {
        Mode::Map | Mode::Group { .. } => module.manifest.outputs.len().max(1),
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
/// bits, `[wheel.x, wheel.y, window.x, window.y]`, `[scale, fold, 1, slice]`,
/// where `fold` is the elements per unit of `gid.y` for a `Rows` domain,
/// which is dispatched as a grid (else zero), and `slice` is the base
/// element, row or depth offset when a large map run is sliced across
/// submissions (else zero), the pad's two slots
/// (`[left.x, left.y, right.x, right.y]`, `[triggers.x, triggers.y, pads,
/// queued]`), and eight slots of the characters typed, four a slot. Each input
/// then has `[width, height, depth, 0]`. Extents, the fold, the slice, the key bits,
/// the samples queued and the characters are the bits of `u32`s, read with `bitcast`, so
/// every one is exact.
pub const CONTEXT_HEAD: usize = 16;

// `typed_at` in the prelude reads eight slots of four characters.
const _: () = assert!(entelechy_abi::TYPED == 32);

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

/// Like [`dispatch`], honoring `module`'s map workgroup size (`16x16` for
/// tiled stencil maps, `8x8` otherwise).
pub fn dispatch_for(module: &Module, domain: entelechy_abi::Domain) -> (u32, u32, u32, u32) {
    let wg = map_workgroup_size(module);
    if wg == WORKGROUP || domain.shape() != entelechy_abi::Shape::Grid {
        dispatch(domain)
    } else {
        (
            domain.width().div_ceil(wg),
            domain.height().div_ceil(wg),
            domain.depth(),
            0,
        )
    }
}

/// Like [`map_slices`], honoring `module`'s map workgroup size (`16x16` for
/// tiled stencil maps, `8x8` otherwise).
pub fn map_slices_for(
    module: &Module,
    domain: entelechy_abi::Domain,
    max_elements: usize,
) -> Vec<(u32, u32, u32, u32, u32)> {
    let wg = map_workgroup_size(module);
    if wg == WORKGROUP || domain.shape() != entelechy_abi::Shape::Grid {
        return map_slices(domain, max_elements);
    }
    let total = domain.count();
    if total == 0 || total <= max_elements {
        let (gx, gy, gz, fold) = dispatch_for(module, domain);
        return vec![(gx, gy, gz, fold, 0)];
    }
    let gx = domain.width().div_ceil(wg);
    let gy_total = domain.height().div_ceil(wg);
    let band = (domain.width().max(1) as usize).saturating_mul(wg as usize);
    let yg_step = (max_elements / band).max(1) as u32;
    let mut slices = Vec::with_capacity(gy_total.div_ceil(yg_step) as usize);
    let mut yg = 0u32;
    while yg < gy_total {
        let count = (gy_total - yg).min(yg_step);
        slices.push((gx, count, domain.depth().max(1), 0, yg * wg));
        yg = yg.saturating_add(yg_step);
    }
    slices
}

/// A map's dispatch over `domain`: workgroups in x, y, z, and the fold
/// (elements per unit of `gid.y`) for rows. Rows fill whole workgroups:
/// strips `fold` wide, a workgroup's rows on consecutive strips.
pub fn dispatch(domain: entelechy_abi::Domain) -> (u32, u32, u32, u32) {
    if domain.shape() == entelechy_abi::Shape::Rows {
        let groups_x = domain
            .width()
            .div_ceil(WORKGROUP * WORKGROUP)
            .clamp(1, MAX_GROUPS);
        let fold = groups_x * WORKGROUP;
        let strips = domain.width().div_ceil(fold);
        (groups_x, strips.div_ceil(WORKGROUP), 1, fold)
    } else {
        (
            domain.width().div_ceil(WORKGROUP),
            domain.height().div_ceil(WORKGROUP),
            domain.depth(),
            0,
        )
    }
}

/// Dispatches `(groups_x, groups_y, groups_z, fold, slice)` covering `domain`
/// with at most `max_elements` threads per slice (rounded up to one workgroup
/// row or depth layer), bijective up to the domain bounds.
pub fn map_slices(
    domain: entelechy_abi::Domain,
    max_elements: usize,
) -> Vec<(u32, u32, u32, u32, u32)> {
    let total = domain.count();
    if total == 0 || total <= max_elements {
        let (gx, gy, gz, fold) = dispatch(domain);
        return vec![(gx, gy, gz, fold, 0)];
    }
    match domain.shape() {
        entelechy_abi::Shape::Rows => {
            let per_group = (WORKGROUP * WORKGROUP) as usize;
            let groups_x = (max_elements / per_group).clamp(1, MAX_GROUPS as usize) as u32;
            let yg_step = (max_elements / (groups_x as usize * per_group)).max(1) as u32;
            let step = groups_x * WORKGROUP * WORKGROUP * yg_step;
            let width = domain.width();
            let mut slices = Vec::with_capacity(width.div_ceil(step) as usize);
            let mut at = 0u32;
            while at < width {
                let left = width - at;
                let gx = left.div_ceil(WORKGROUP * WORKGROUP).clamp(1, groups_x);
                let chunk_fold = gx * WORKGROUP;
                let gy = left
                    .min(step)
                    .div_ceil(chunk_fold)
                    .div_ceil(WORKGROUP)
                    .max(1);
                slices.push((gx, gy, 1, chunk_fold, at));
                at = at.saturating_add(step);
            }
            slices
        }
        entelechy_abi::Shape::Volume if domain.depth() > 1 => {
            let gx = domain.width().div_ceil(WORKGROUP);
            let gy = domain.height().div_ceil(WORKGROUP);
            let layer =
                (domain.width().max(1) as usize).saturating_mul(domain.height().max(1) as usize);
            let step = (max_elements / layer).max(1) as u32;
            let depth = domain.depth();
            let mut slices = Vec::with_capacity(depth.div_ceil(step) as usize);
            let mut z = 0u32;
            while z < depth {
                let count = (depth - z).min(step);
                slices.push((gx, gy, count, 0, z));
                z = z.saturating_add(step);
            }
            slices
        }
        _ => {
            let gx = domain.width().div_ceil(WORKGROUP);
            let gy_total = domain.height().div_ceil(WORKGROUP);
            let band = (domain.width().max(1) as usize).saturating_mul(WORKGROUP as usize);
            let yg_step = (max_elements / band).max(1) as u32;
            let mut slices = Vec::with_capacity(gy_total.div_ceil(yg_step) as usize);
            let mut yg = 0u32;
            while yg < gy_total {
                let count = (gy_total - yg).min(yg_step);
                slices.push((gx, count, domain.depth().max(1), 0, yg * WORKGROUP));
                yg = yg.saturating_add(yg_step);
            }
            slices
        }
    }
}

/// A group's dispatch over `domain`: a workgroup a group of `size` lanes,
/// rows of at most [`MAX_GROUPS`] workgroups ([`folded`]).
pub fn group_dispatch(domain: entelechy_abi::Domain, size: u32) -> (u32, u32, u32, u32) {
    let groups = u32::try_from(domain.count().div_ceil(size as usize)).unwrap_or(u32::MAX);
    let (x, y) = folded(groups);
    (x, y, 1, 0)
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
    // WGSL rejects non-finite constant expressions, and a strict compiler
    // (a page's) a decimal past the largest finite f32, which the shortest
    // decimal of a float near it is: those are written as their bits.
    let f = |v: f32| {
        if v.is_finite() && v.abs() < 1.0e38 {
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
        Value::Mat3(v) => format!("mat3x3<f32>({})", floats(&v[..])),
        Value::Mat4(v) => format!("mat4x4<f32>({})", floats(&v[..])),
        Value::Long(v) => long_literal(*v),
        Value::LVec2(v) => format!("vec2<i64>({})", longs(v)),
        Value::LVec3(v) => format!("vec3<i64>({})", longs(&v[..])),
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
