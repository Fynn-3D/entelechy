//! The GPU side of native `rasterize`, with the contract of
//! [`entelechy_rules::raster`], as a visibility buffer drawn two ways:
//!
//! 1. A compute pass sorts the triangles. One the rule would not draw (a
//!    corner not usable, no area, no pixel center in its box) is dropped.
//!    One whose box is a few pixels across is drawn there and then, by the
//!    rule itself: each pixel center it covers takes the least key of its
//!    depth and its row, one 64-bit atomic apiece, where the device has
//!    them. The rest are kept in row order, as an index buffer and the
//!    rows they came from: a block of triangles marks its own, one pass
//!    sums the blocks, and each kept triangle writes itself at its block's
//!    offset and its place among the marks.
//! 2. The device's rasterizer draws the kept ones into a depth target and
//!    one integer target, the kept triangle's place: indexed where a
//!    fragment knows its primitive, so a vertex shared by many triangles is
//!    shaded once; otherwise a vertex pulled for each triangle that uses it.
//! 3. A compute pass takes, at each pixel, the nearer of the two, the
//!    earlier row of two at one depth, and gives its depth and attributes
//!    by the rule's own sums at the pixel center, into the Host's word
//!    layout as a resident grid.
//!
//! A tile-binning rasterizer spends as much on a triangle of one pixel as
//! on one of many, so the small ones, which most of a dense mesh is, are
//! drawn as the rule draws them. Coverage of the large ones is the
//! device's; what a covered pixel holds is always the rule's.

use std::sync::OnceLock;

use entelechy_abi::{Handle, Invocation, Scope};
use entelechy_rules::raster::{SURFACE_WORDS, check_rasterize, surface_schema};
use entelechy_rules::{ExecutionError, NativeOp};

use crate::failure::Scopes;
use crate::raster_hiz::{
    self, HIZ_TRI_THRESHOLD, HizPipelines, build_hiz_pipelines, encode_hiz_pyramid, hiz_dims,
    hiz_level_payloads, take_hiz_texture,
};
use crate::{DeviceImage, WgpuBackend};

/// What every stage reads: the canvas, the counts, the cutout, how small a
/// triangle the indirect micro-triangle pass draws itself, and Hi-Z state.
const SETUP_WGSL: &str = "
struct Setup {
    size: vec2<f32>,
    vertex_count: u32,
    triangle_count: u32,
    blocks_x: u32,
    block_count: u32,
    mask_width: u32,
    mask_height: u32,
    cutout: f32,
    masked: u32,
    small: u32,
    hiz_mips: u32,
    occ_only: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};
@group(0) @binding(0) var<uniform> setup: Setup;
@group(0) @binding(1) var<storage, read> vertices: array<u32>;

// A vertex's clip position.
fn clip_of(v: u32) -> vec4<f32> {
    let base = v * 9u;
    return vec4<f32>(
        bitcast<f32>(vertices[base]),
        bitcast<f32>(vertices[base + 1u]),
        bitcast<f32>(vertices[base + 2u]),
        bitcast<f32>(vertices[base + 3u]));
}

// A corner in pixels, as the rule places it: `x`, `y` from the top left,
// `z` after the divide, and `1 / w`.
fn screen(v: u32) -> vec4<f32> {
    let c = clip_of(v);
    let inv_w = 1.0 / c.w;
    return vec4<f32>(
        (c.x * inv_w * 0.5 + 0.5) * setup.size.x,
        (0.5 - c.y * inv_w * 0.5) * setup.size.y,
        c.z * inv_w,
        inv_w);
}

fn edge(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> f32 {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}
";

/// The triangle rows, for the stages that read them.
const TRIANGLES_WGSL: &str = "
@group(0) @binding(2) var<storage, read> triangles: array<u32>;

// A triangle's corner, clamped into the vertex rows as the rule clamps it.
fn corner(t: u32, k: u32) -> u32 {
    let raw = bitcast<i32>(triangles[t * 3u + k]);
    return u32(clamp(raw, 0i, i32(setup.vertex_count) - 1i));
}
";

/// Hardware-filtered 2D tiled texture mask sampling (when `FLOAT32_FILTERABLE`
/// is supported): single hardware TMU bilinear tap sharing the immutable
/// `DeviceImage` texture mirror with shading passes.
const MASK_FILTERED_WGSL: &str = "
@group(0) @binding({mask}) var mask_tex: texture_2d<f32>;
@group(0) @binding({sampler}) var mask_sampler: sampler;

fn mask_alpha(uv: vec2<f32>) -> f32 {
    return textureSampleLevel(mask_tex, mask_sampler, uv, 0.0).a;
}

fn uv_of(v: u32) -> vec2<f32> {
    return vec2<f32>(bitcast<f32>(vertices[v * 9u + 7u]), bitcast<f32>(vertices[v * 9u + 8u]));
}
";

/// 2D tiled texture mask sampling fallback without `FLOAT32_FILTERABLE`:
/// manual 4-tap bilinear over the 2D tiled hardware texture cache.
const MASK_LOAD_WGSL: &str = "
@group(0) @binding({mask}) var mask_tex: texture_2d<f32>;

fn texel(x: i32, y: i32, w: i32, h: i32) -> f32 {
    let cx = clamp(x, 0, max(w - 1, 0));
    let cy = clamp(y, 0, max(h - 1, 0));
    return textureLoad(mask_tex, vec2<i32>(cx, cy), 0).a;
}

// The mask's alpha where `at_linear` would read it: the four texels
// around the place, blended, clamped at the edges, as the rule reads it.
fn mask_alpha(uv: vec2<f32>) -> f32 {
    let w = i32(setup.mask_width);
    let h = i32(setup.mask_height);
    let sx = uv.x * f32(w) - 0.5;
    let sy = uv.y * f32(h) - 0.5;
    let x0 = i32(floor(sx));
    let y0 = i32(floor(sy));
    let fx = sx - floor(sx);
    let fy = sy - floor(sy);
    let at = texel(x0, y0, w, h);
    let bx = texel(x0 + 1, y0, w, h);
    let ay = texel(x0, y0 + 1, w, h);
    let by = texel(x0 + 1, y0 + 1, w, h);
    let top = at + (bx - at) * fx;
    let bottom = ay + (by - ay) * fx;
    return top + (bottom - top) * fy;
}

fn uv_of(v: u32) -> vec2<f32> {
    return vec2<f32>(bitcast<f32>(vertices[v * 9u + 7u]), bitcast<f32>(vertices[v * 9u + 8u]));
}
";

/// Step 1: common bindings, `usable`, conservative Hi-Z culling, and `sort`
/// shared across subgroup and fallback compaction passes.
/// Keeps storage buffers at 8 (core WebGPU baseline) plus 1 sampled Hi-Z texture.
const KEEP_COMMON_WGSL: &str = "
@group(0) @binding(3) var<storage, read_write> marks: array<u32>;
@group(0) @binding(4) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> indices: array<u32>;
@group(0) @binding(6) var<storage, read_write> rows: array<u32>;
@group(0) @binding(7) var<storage, read_write> args: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> small_rows: array<u32>;
@group(0) @binding(9) var hiz: texture_2d<f32>;

const BLOCK: u32 = 256u;

// A vertex the rule can use: every clip component finite and w above
// zero. The test is on the words, where a NaN and an infinity are the
// ones exponent, so no fast maths can fold it away.
fn usable(v: u32) -> bool {
    let base = v * 9u;
    let open = 0x7F800000u;
    return (vertices[base] & open) != open && (vertices[base + 1u] & open) != open
        && (vertices[base + 2u] & open) != open && (vertices[base + 3u] & open) != open
        && bitcast<f32>(vertices[base + 3u]) > 0.0;
}

fn finite_f32(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7F800000u) != 0x7F800000u;
}

// Conservative MAX-Z pyramid test: returns true only when the triangle's
// minimum vertex depth is strictly behind every pixel covered by its screen
// bbox (with safety margin so coplanar earlier-row-wins tie-breaks and
// hardware/software interpolation differences are never violated).
fn hiz_occluded(ux0: u32, uy0: u32, ux1: u32, uy1: u32, min_z: f32) -> bool {
    if setup.hiz_mips == 0u || !(min_z > 0.0) {
        return false;
    }
    let span = max(ux1 - ux0, uy1 - uy0);
    var mip = 0u;
    if span > 2u {
        mip = min((32u - countLeadingZeros(span - 1u)) - 1u, setup.hiz_mips - 1u);
    }
    let shift = mip + 1u;
    let dim = textureDimensions(hiz, i32(mip));
    let tx0 = min(ux0 >> shift, dim.x - 1u);
    let tx1 = min(ux1 >> shift, dim.x - 1u);
    let ty0 = min(uy0 >> shift, dim.y - 1u);
    let ty1 = min(uy1 >> shift, dim.y - 1u);
    var hiz_max = 0.0;
    for (var ty = ty0; ty <= ty1; ty++) {
        for (var tx = tx0; tx <= tx1; tx++) {
            let d = textureLoad(hiz, vec2<i32>(i32(tx), i32(ty)), i32(mip)).x;
            hiz_max = max(hiz_max, d);
        }
    }
    return min_z > hiz_max + 2e-5;
}

// Macro-occluder predicate shared by Phase 1 (`occ_only = 1`) and Phase 3
// (`hiz_mips > 0`) so every triangle belongs to exactly one phase (zero redraw):
// hardware-rasterized triangles (`>= SMALL` along at least one axis) covering at
// least a 16x16 pixel tile (>= 128 px^2) draw once in Phase 1 directly into
// `places` + `depth`, seeding both the MAX-Z Hi-Z pyramid and Phase 4 early-Z.
fn is_occ(twice_area: f32, x0: f32, y0: f32, x1: f32, y1: f32) -> bool {
    let not_small = (x1 - x0) >= f32(setup.small) || (y1 - y0) >= f32(setup.small);
    return not_small && twice_area >= 256.0 && (x1 - x0) >= 15.0 && (y1 - y0) >= 15.0;
}

// How a triangle is drawn: 0 not at all (unusable, degenerate, offscreen,
// frustum-Z, already drawn in Phase 1, or Hi-Z occluded), 1 by the device's
// rasterizer, 2 by the indirect micro-triangle pass.
fn sort(t: u32) -> u32 {
    let ia = corner(t, 0u);
    let ib = corner(t, 1u);
    let ic = corner(t, 2u);
    if !(usable(ia) && usable(ib) && usable(ic)) {
        return 0u;
    }
    let a = screen(ia);
    let b = screen(ib);
    let c = screen(ic);
    let twice_area = abs(edge(a.xy, b.xy, c.xy));
    if !(twice_area > 0.0) {
        return 0u;
    }
    let x0 = max(ceil(min(min(a.x, b.x), c.x) - 0.5), 0.0);
    let y0 = max(ceil(min(min(a.y, b.y), c.y) - 0.5), 0.0);
    let x1 = min(floor(max(max(a.x, b.x), c.x) - 0.5), setup.size.x - 1.0);
    let y1 = min(floor(max(max(a.y, b.y), c.y) - 0.5), setup.size.y - 1.0);
    if !(x1 >= x0 && y1 >= y0) {
        return 0u;
    }
    let finite_z = finite_f32(a.z) && finite_f32(b.z) && finite_f32(c.z);
    if finite_z {
        let min_z = min(a.z, min(b.z, c.z));
        let max_z = max(a.z, max(b.z, c.z));
        if max_z < -1e-5 || min_z > 1.0 + 1e-5 {
            return 0u;
        }
        if setup.occ_only != 0u {
            return select(0u, 1u, is_occ(twice_area, x0, y0, x1, y1));
        }
        if setup.hiz_mips != 0u && is_occ(twice_area, x0, y0, x1, y1) {
            return 0u;
        }
        if hiz_occluded(u32(x0), u32(y0), u32(x1), u32(y1), min_z) {
            return 0u;
        }
    } else if setup.occ_only != 0u {
        return 0u;
    }
    let small = f32(setup.small);
    if x1 - x0 < small && y1 - y0 < small {
        return 2u;
    }
    return 1u;
}

fn write_indirect_args(total_kept: u32) {
    // Words 0..5: DrawIndexedIndirectArgs / DrawIndirectArgs (byte offset 0).
    atomicStore(&args[0u], total_kept * 3u);
    atomicStore(&args[1u], 1u);
    atomicStore(&args[2u], 0u);
    atomicStore(&args[3u], 0u);
    atomicStore(&args[4u], 0u);
    // Words 8..11: DispatchIndirectArgs for indirect micro-triangles (byte offset 32).
    // Word 11: compacted small_count; word 12: dispatch X stride.
    let sc = atomicLoad(&args[11u]);
    let groups = (sc + 63u) / 64u;
    let gx = min(groups, 65535u);
    let safe_gx = max(gx, 1u);
    let gy = select(1u, (groups + safe_gx - 1u) / safe_gx, groups > 0u);
    atomicStore(&args[8u], gx);
    atomicStore(&args[9u], gy);
    atomicStore(&args[10u], 1u);
    atomicStore(&args[12u], gx);
}

var<workgroup> place_marks: array<u32, 8>;

// Each kept triangle writes its corners and its row at its place, caching the
// block's 8 mark words once in workgroup memory.
@compute @workgroup_size(256)
fn place(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>) {
    let block = group.x + group.y * setup.blocks_x;
    if local < 8u && block < setup.block_count {
        place_marks[local] = marks[block * 8u + local];
    }
    workgroupBarrier();
    let t = block * BLOCK + local;
    if block >= setup.block_count || t >= setup.triangle_count {
        return;
    }
    let word = place_marks[local / 32u];
    let bit = local % 32u;
    if ((word >> bit) & 1u) == 0u {
        return;
    }
    var before = countOneBits(word & ((1u << bit) - 1u));
    for (var w = 0u; w < local / 32u; w++) {
        before += countOneBits(place_marks[w]);
    }
    let at = offsets[block] + before;
    indices[at * 3u] = corner(t, 0u);
    indices[at * 3u + 1u] = corner(t, 1u);
    indices[at * 3u + 2u] = corner(t, 2u);
    rows[at] = t;
}
";

/// Step 1 `mark` + `sum` for 32-lane hardware subgroups (Apple Metal, NVIDIA):
/// `subgroupBallot` produces each 32-bit mark word and compacts surviving
/// micro-triangles with zero workgroup atomics and 1 global atomic per block.
const KEEP_SG32_WGSL: &str = "
var<workgroup> wave_counts: array<u32, 8>;
var<workgroup> wave_small: array<u32, 8>;
var<workgroup> wave_small_off: array<u32, 8>;
var<workgroup> wg_small_base: u32;

@compute @workgroup_size(256)
fn mark(
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
) {
    let block = group.x + group.y * setup.blocks_x;
    let t = block * BLOCK + local;
    var cls = 0u;
    if block < setup.block_count && t < setup.triangle_count {
        cls = sort(t);
    }
    let keep = cls == 1u;
    let is_small = cls == 2u;
    let ballot = subgroupBallot(keep).x;
    let s_ballot = subgroupBallot(is_small).x;
    let wave = local / 32u;
    if sg_id == 0u {
        if block < setup.block_count {
            marks[block * 8u + wave] = ballot;
        }
        wave_counts[wave] = countOneBits(ballot);
        wave_small[wave] = countOneBits(s_ballot);
    }
    workgroupBarrier();
    if wave == 0u {
        var c = 0u;
        var sc = 0u;
        if sg_id < 8u {
            c = wave_counts[sg_id];
            sc = wave_small[sg_id];
        }
        let total = subgroupAdd(c);
        let s_inc = subgroupInclusiveAdd(sc);
        if sg_id < 8u {
            wave_small_off[sg_id] = s_inc - sc;
        }
        if sg_id == 0u && block < setup.block_count {
            offsets[block] = total;
        }
        if sg_id == 7u {
            if s_inc > 0u {
                wg_small_base = atomicAdd(&args[11u], s_inc);
            } else {
                wg_small_base = 0u;
            }
        }
    }
    workgroupBarrier();
    if is_small {
        let lane_rank = countOneBits(s_ballot & ((1u << sg_id) - 1u));
        small_rows[wg_small_base + wave_small_off[wave] + lane_rank] = t;
    }
}

var<workgroup> wave_sums: array<u32, 64>;

@compute @workgroup_size(256)
fn sum(
    @builtin(local_invocation_index) local: u32,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let n = setup.block_count;
    let per = (n + 255u) / 256u;
    let first = min(local * per, n);
    let last = min(first + per, n);
    var own = 0u;
    for (var i = first; i < last; i++) {
        own += offsets[i];
    }
    let wave = local / sg_size;
    let num_waves = 256u / sg_size;
    var inc = subgroupInclusiveAdd(own);
    if sg_id == sg_size - 1u {
        wave_sums[wave] = inc;
    }
    workgroupBarrier();
    if num_waves <= sg_size {
        if wave == 0u {
            var w = 0u;
            if sg_id < num_waves {
                w = wave_sums[sg_id];
            }
            let w_inc = subgroupInclusiveAdd(w);
            if sg_id < num_waves {
                wave_sums[sg_id] = w_inc;
            }
        }
    } else if local == 0u {
        var acc = wave_sums[0u];
        for (var w = 1u; w < num_waves; w++) {
            acc += wave_sums[w];
            wave_sums[w] = acc;
        }
    }
    workgroupBarrier();
    if wave > 0u {
        inc += wave_sums[wave - 1u];
    }
    var at = inc - own;
    for (var i = first; i < last; i++) {
        let count = offsets[i];
        offsets[i] = at;
        at += count;
    }
    if local == 255u {
        write_indirect_args(inc);
    }
}
";

/// Step 1 `mark` + `sum` for general power-of-two hardware subgroups (16/32/64
/// lanes, e.g. AMD Wave64 / Intel Xe Wave16): wave-aggregated `kept` counts and
/// 2-barrier `subgroupInclusiveAdd` block scan.
const KEEP_SG_GENERAL_WGSL: &str = "
var<workgroup> marked: array<atomic<u32>, 8>;
var<workgroup> kept: atomic<u32>;
var<workgroup> small_cnt: atomic<u32>;
var<workgroup> small_base: u32;

@compute @workgroup_size(256)
fn mark(
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
) {
    if local < 8u {
        atomicStore(&marked[local], 0u);
    }
    if local == 0u {
        atomicStore(&kept, 0u);
        atomicStore(&small_cnt, 0u);
    }
    workgroupBarrier();
    let block = group.x + group.y * setup.blocks_x;
    let t = block * BLOCK + local;
    var cls = 0u;
    if block < setup.block_count && t < setup.triangle_count {
        cls = sort(t);
    }
    let keep = cls == 1u;
    let is_small = cls == 2u;
    if keep {
        atomicOr(&marked[local / 32u], 1u << (local % 32u));
    }
    var local_rank = 0u;
    if is_small {
        local_rank = atomicAdd(&small_cnt, 1u);
    }
    let wave_kept = subgroupAdd(u32(keep));
    if sg_id == 0u && wave_kept > 0u {
        atomicAdd(&kept, wave_kept);
    }
    workgroupBarrier();
    if local == 0u {
        let sc = atomicLoad(&small_cnt);
        if sc > 0u {
            small_base = atomicAdd(&args[11u], sc);
        } else {
            small_base = 0u;
        }
    }
    if block < setup.block_count {
        if local < 8u {
            marks[block * 8u + local] = atomicLoad(&marked[local]);
        }
        if local == 0u {
            offsets[block] = atomicLoad(&kept);
        }
    }
    workgroupBarrier();
    if is_small {
        small_rows[small_base + local_rank] = t;
    }
}

var<workgroup> wave_sums: array<u32, 64>;

@compute @workgroup_size(256)
fn sum(
    @builtin(local_invocation_index) local: u32,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let n = setup.block_count;
    let per = (n + 255u) / 256u;
    let first = min(local * per, n);
    let last = min(first + per, n);
    var own = 0u;
    for (var i = first; i < last; i++) {
        own += offsets[i];
    }
    let wave = local / sg_size;
    let num_waves = 256u / sg_size;
    var inc = subgroupInclusiveAdd(own);
    if sg_id == sg_size - 1u {
        wave_sums[wave] = inc;
    }
    workgroupBarrier();
    if num_waves <= sg_size {
        if wave == 0u {
            var w = 0u;
            if sg_id < num_waves {
                w = wave_sums[sg_id];
            }
            let w_inc = subgroupInclusiveAdd(w);
            if sg_id < num_waves {
                wave_sums[sg_id] = w_inc;
            }
        }
    } else if local == 0u {
        var acc = wave_sums[0u];
        for (var w = 1u; w < num_waves; w++) {
            acc += wave_sums[w];
            wave_sums[w] = acc;
        }
    }
    workgroupBarrier();
    if wave > 0u {
        inc += wave_sums[wave - 1u];
    }
    var at = inc - own;
    for (var i = first; i < last; i++) {
        let count = offsets[i];
        offsets[i] = at;
        at += count;
    }
    if local == 255u {
        write_indirect_args(inc);
    }
}
";

/// Step 1 `mark` + `sum` fallback without subgroup extensions.
const KEEP_FALLBACK_WGSL: &str = "
var<workgroup> marked: array<atomic<u32>, 8>;
var<workgroup> kept: atomic<u32>;
var<workgroup> small_cnt: atomic<u32>;
var<workgroup> small_base: u32;

// Each block of triangles marks the ones the device draws, a bit each,
// counts them, and compacts surviving micro-triangles for indirect dispatch.
@compute @workgroup_size(256)
fn mark(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>) {
    if local < 8u {
        atomicStore(&marked[local], 0u);
    }
    if local == 0u {
        atomicStore(&kept, 0u);
        atomicStore(&small_cnt, 0u);
    }
    workgroupBarrier();
    let block = group.x + group.y * setup.blocks_x;
    let t = block * BLOCK + local;
    var cls = 0u;
    if block < setup.block_count && t < setup.triangle_count {
        cls = sort(t);
    }
    if cls == 1u {
        atomicOr(&marked[local / 32u], 1u << (local % 32u));
        atomicAdd(&kept, 1u);
    }
    var local_rank = 0u;
    if cls == 2u {
        local_rank = atomicAdd(&small_cnt, 1u);
    }
    workgroupBarrier();
    if local == 0u {
        let sc = atomicLoad(&small_cnt);
        if sc > 0u {
            small_base = atomicAdd(&args[11u], sc);
        } else {
            small_base = 0u;
        }
    }
    if block < setup.block_count {
        if local < 8u {
            marks[block * 8u + local] = atomicLoad(&marked[local]);
        }
        if local == 0u {
            offsets[block] = atomicLoad(&kept);
        }
    }
    workgroupBarrier();
    if cls == 2u {
        small_rows[small_base + local_rank] = t;
    }
}

var<workgroup> sums: array<u32, 256>;

// Every block's count becomes where its triangles begin, and the draw is
// sized by the whole: one workgroup, each invocation a run of blocks.
@compute @workgroup_size(256)
fn sum(@builtin(local_invocation_index) local: u32) {
    let n = setup.block_count;
    let per = (n + 255u) / 256u;
    let first = min(local * per, n);
    let last = min(first + per, n);
    var own = 0u;
    for (var i = first; i < last; i++) {
        own += offsets[i];
    }
    sums[local] = own;
    workgroupBarrier();
    for (var d = 1u; d < 256u; d <<= 1u) {
        var before = 0u;
        if local >= d {
            before = sums[local - d];
        }
        workgroupBarrier();
        sums[local] += before;
        workgroupBarrier();
    }
    var at = sums[local] - own;
    for (var i = first; i < last; i++) {
        let count = offsets[i];
        offsets[i] = at;
        at += count;
    }
    if local == 255u {
        write_indirect_args(sums[255]);
    }
}
";

/// GPU-driven indirect micro-triangle pass (`dispatch_workgroups_indirect`),
/// compiled only where the device supports 64-bit atomics: draws compacted
/// non-occluded `< SMALL` triangles into `keys` with full SIMD occupancy.
const DRAW_SMALL_WGSL: &str = "
@group(0) @binding(3) var<storage, read> small_rows: array<u32>;
@group(0) @binding(4) var<storage, read> args: array<u32>;
@group(0) @binding(5) var<storage, read_write> keys: array<atomic<u64>>;

fn draw_here(t: u32, ia: u32, ib_: u32, ic_: u32, a: vec4<f32>, b_: vec4<f32>, c_: vec4<f32>, box: vec4<u32>) {
    var ib = ib_;
    var ic = ic_;
    var b = b_;
    var c = c_;
    // Wound as the rule winds it, so the edge functions are positive inside.
    if edge(a.xy, b.xy, c.xy) < 0.0 {
        b = c_;
        c = b_;
        ib = ic_;
        ic = ib_;
    }
    let area = edge(a.xy, b.xy, c.xy);
    let width = u32(setup.size.x);
    for (var y = box.y; y <= box.w; y++) {
        for (var x = box.x; x <= box.z; x++) {
            let p = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);
            let w0 = edge(b.xy, c.xy, p);
            let w1 = edge(c.xy, a.xy, p);
            let w2 = edge(a.xy, b.xy, p);
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }
            let l0 = w0 / area;
            let l1 = w1 / area;
            let l2 = w2 / area;
            let z = l0 * a.z + l1 * b.z + l2 * c.z;
            if !(z >= 0.0 && z < 1.0) {
                continue;
            }
            if setup.masked != 0u {
                let inv_w = l0 * a.w + l1 * b.w + l2 * c.w;
                let uv = (l0 * uv_of(ia) * a.w + l1 * uv_of(ib) * b.w + l2 * uv_of(ic) * c.w) / inv_w;
                if mask_alpha(uv) < setup.cutout {
                    continue;
                }
            }
            let key = (u64(bitcast<u32>(z)) << 32u) | u64(t);
            atomicMax(&keys[y * width + x], ~key);
        }
    }
}

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let small_count = args[11u];
    let stride = args[12u];
    let idx = (group.x + group.y * stride) * 64u + local;
    if idx >= small_count {
        return;
    }
    let t = small_rows[idx];
    let ia = corner(t, 0u);
    let ib = corner(t, 1u);
    let ic = corner(t, 2u);
    let a = screen(ia);
    let b = screen(ib);
    let c = screen(ic);
    let x0 = max(ceil(min(min(a.x, b.x), c.x) - 0.5), 0.0);
    let y0 = max(ceil(min(min(a.y, b.y), c.y) - 0.5), 0.0);
    let x1 = min(floor(max(max(a.x, b.x), c.x) - 0.5), setup.size.x - 1.0);
    let y1 = min(floor(max(max(a.y, b.y), c.y) - 0.5), setup.size.y - 1.0);
    draw_here(t, ia, ib, ic, a, b, c, vec4<u32>(u32(x0), u32(y0), u32(x1), u32(y1)));
}
";

/// Step 2 where a fragment knows its primitive: the kept triangles drawn
/// indexed, writing their global triangle row + 1 directly into `places` so
/// Phase 1 (occluders) and Phase 4 (surviving non-occluders) compose cleanly
/// in a single visibility buffer with zero double-drawing.
const INDEXED_WGSL: &str = "
@group(0) @binding(3) var<storage, read> draw_rows: array<u32>;

struct Cut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) v: u32) -> @builtin(position) vec4<f32> {
    return clip_of(v);
}

@vertex
fn vs_cut(@builtin(vertex_index) v: u32) -> Cut {
    var out: Cut;
    out.clip = clip_of(v);
    out.uv = uv_of(v);
    return out;
}

// Global triangle row + 1, so a cleared zero is none.
@fragment
fn fs(@builtin(primitive_index) p: u32) -> @location(0) u32 {
    return draw_rows[p] + 1u;
}

@fragment
fn fs_cut(in: Cut, @builtin(primitive_index) p: u32) -> @location(0) u32 {
    if mask_alpha(in.uv) < setup.cutout {
        discard;
    }
    return draw_rows[p] + 1u;
}
";

/// Step 2 where it does not: each triangle's corners pulled from the kept
/// index buffer, its global triangle row carried flat.
const PULLED_WGSL: &str = "
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> draw_rows: array<u32>;

struct Plain {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) row: u32,
};

struct Cut {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) row: u32,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) at: u32) -> Plain {
    var out: Plain;
    out.clip = clip_of(indices[at]);
    out.row = draw_rows[at / 3u];
    return out;
}

@vertex
fn vs_cut(@builtin(vertex_index) at: u32) -> Cut {
    let v = indices[at];
    var out: Cut;
    out.clip = clip_of(v);
    out.row = draw_rows[at / 3u];
    out.uv = uv_of(v);
    return out;
}

@fragment
fn fs(in: Plain) -> @location(0) u32 {
    return in.row + 1u;
}

@fragment
fn fs_cut(in: Cut) -> @location(0) u32 {
    if mask_alpha(in.uv) < setup.cutout {
        discard;
    }
    return in.row + 1u;
}
";

/// Step 3: each pixel's surface, by the rule's sums.
const RESOLVE_WGSL: &str = "
@group(0) @binding(3) var<storage, read> rows: array<u32>;
@group(0) @binding(4) var places: texture_2d<u32>;
@group(0) @binding(5) var<storage, read_write> out: array<u32>;
@group(0) @binding(6) var<storage, read> keys: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read> before: array<vec2<u32>>;

struct Fragment {
    z: f32,
    normal: vec3<f32>,
    uv: vec2<f32>,
};

fn word_of(v: u32, k: u32) -> f32 {
    return bitcast<f32>(vertices[v * 9u + k]);
}

// Triangle `t`'s fragment at the pixel center `p`, as the rule gives it.
fn fragment(t: u32, p: vec2<f32>) -> Fragment {
    let ia = corner(t, 0u);
    var ib = corner(t, 1u);
    var ic = corner(t, 2u);
    let a = screen(ia);
    var b = screen(ib);
    var c = screen(ic);
    // Wound as the rule winds it, so the edge functions are positive inside.
    if edge(a.xy, b.xy, c.xy) < 0.0 {
        let swap = b;
        b = c;
        c = swap;
        let row = ib;
        ib = ic;
        ic = row;
    }
    let area = edge(a.xy, b.xy, c.xy);
    let l0 = edge(b.xy, c.xy, p) / area;
    let l1 = edge(c.xy, a.xy, p) / area;
    let l2 = edge(a.xy, b.xy, p) / area;
    let inv_w = l0 * a.w + l1 * b.w + l2 * c.w;
    var values: array<f32, 5>;
    for (var k = 0u; k < 5u; k++) {
        values[k] = (l0 * word_of(ia, k + 4u) * a.w + l1 * word_of(ib, k + 4u) * b.w
            + l2 * word_of(ic, k + 4u) * c.w) / inv_w;
    }
    return Fragment(
        l0 * a.z + l1 * b.z + l2 * c.z,
        vec3<f32>(values[0], values[1], values[2]),
        vec2<f32>(values[3], values[4]));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = u32(setup.size.x);
    let height = u32(setup.size.y);
    if id.x >= width || id.y >= height {
        return;
    }
    let pixel = id.y * width + id.x;
    let p = vec2<f32>(f32(id.x) + 0.5, f32(id.y) + 0.5);
    let place = textureLoad(places, vec2<i32>(i32(id.x), i32(id.y)), 0).r;
    var row = 0u;
    var found = false;
    var f: Fragment;
    if place != 0u {
        row = place - 1u;
        f = fragment(row, p);
        found = true;
    }
    if setup.small != 0u {
        let key = keys[pixel];
        if (key.x | key.y) != 0u {
            let here = ~key.x;
            let z = bitcast<f32>(~key.y);
            if !found || z < f.z || (z == f.z && here < row) {
                row = here;
                f = fragment(row, p);
                found = true;
            }
        }
    }
    let dst = pixel * 7u;
    if !found {
        out[dst] = bitcast<u32>(1.0);
        for (var k = 1u; k < 6u; k++) {
            out[dst + k] = 0u;
        }
        out[dst + 6u] = bitcast<u32>(-1i);
        return;
    }
    out[dst] = bitcast<u32>(f.z);
    out[dst + 1u] = bitcast<u32>(f.normal.x);
    out[dst + 2u] = bitcast<u32>(f.normal.y);
    out[dst + 3u] = bitcast<u32>(f.normal.z);
    out[dst + 4u] = bitcast<u32>(f.uv.x);
    out[dst + 5u] = bitcast<u32>(f.uv.y);
    out[dst + 6u] = row;
}

// Triangle `t`'s depth at the pixel center `p`, as `fragment` gives it.
fn depth_of(t: u32, p: vec2<f32>) -> f32 {
    let a = screen(corner(t, 0u));
    var b = screen(corner(t, 1u));
    var c = screen(corner(t, 2u));
    if edge(a.xy, b.xy, c.xy) < 0.0 {
        let swap = b;
        b = c;
        c = swap;
    }
    let area = edge(a.xy, b.xy, c.xy);
    return edge(b.xy, c.xy, p) / area * a.z + edge(c.xy, a.xy, p) / area * b.z
        + edge(a.xy, b.xy, p) / area * c.z;
}

// The cover: at each pixel the least, as a signed 64-bit key, of the key
// before it and the nearest fragment's, its depth's bits above and its
// row below, a key two words, the low first.
@compute @workgroup_size(8, 8, 1)
fn cover(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = u32(setup.size.x);
    let height = u32(setup.size.y);
    if id.x >= width || id.y >= height {
        return;
    }
    let pixel = id.y * width + id.x;
    let p = vec2<f32>(f32(id.x) + 0.5, f32(id.y) + 0.5);
    let place = textureLoad(places, vec2<i32>(i32(id.x), i32(id.y)), 0).r;
    var row = 0u;
    var z = 0.0;
    var found = false;
    if place != 0u {
        row = place - 1u;
        z = depth_of(row, p);
        found = true;
    }
    if setup.small != 0u {
        let key = keys[pixel];
        if (key.x | key.y) != 0u {
            let here = ~key.x;
            let near = bitcast<f32>(~key.y);
            if !found || near < z || (near == z && here < row) {
                row = here;
                z = near;
                found = true;
            }
        }
    }
    var least = before[pixel];
    if found {
        let high = bitcast<u32>(z);
        if bitcast<i32>(high) < bitcast<i32>(least.y) || (high == least.y && row < least.x) {
            least = vec2<u32>(row, high);
        }
    }
    out[pixel * 2u] = least.x;
    out[pixel * 2u + 1u] = least.y;
}
";

/// The widest box, in pixel centers, of a triangle the sorting pass draws
/// itself rather than handing to the device's rasterizer.
const SMALL: u32 = 8;
/// Triangles a block of the keeping pass marks.
const BLOCK: u32 = 256;
/// The most workgroups a dispatch takes along one axis.
const MOST_GROUPS: u32 = 65_535;

/// The pipelines, built once per backend.
pub(crate) struct RasterPipelines {
    keep_layout: wgpu::BindGroupLayout,
    mark: wgpu::ComputePipeline,
    sum: wgpu::ComputePipeline,
    place: wgpu::ComputePipeline,
    draw_small_layout: Option<wgpu::BindGroupLayout>,
    draw_small: Option<wgpu::ComputePipeline>,
    draw_layout: wgpu::BindGroupLayout,
    draw: wgpu::RenderPipeline,
    draw_cut: wgpu::RenderPipeline,
    hiz: HizPipelines,
    resolve_layout: wgpu::BindGroupLayout,
    resolve: wgpu::ComputePipeline,
    /// The resolving pass of `native.cover`: keys, not a surface.
    cover: wgpu::ComputePipeline,
    /// A single unread 1x1 texture bound when a call carries no cutout mask.
    empty_mask_view: wgpu::TextureView,
    /// Whether the draw is indexed: see [`GpuBackend::primitive_index`].
    indexed: bool,
    /// Whether the indirect micro-triangle pass draws small triangles itself:
    /// where the device lands 64-bit atomics.
    here: bool,
    /// Whether `mask_tex` uses hardware bilinear filtering (`FLOAT32_FILTERABLE`).
    mask_filtered: bool,
}

const PLACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

fn build(
    device: &wgpu::Device,
    wait: crate::Wait,
    indexed: bool,
    here: bool,
    subgroups: crate::wgsl::SubgroupCaps,
    mask_filtered: bool,
) -> Result<RasterPipelines, String> {
    let guard = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = |label: &str, parts: &[&str]| {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(parts.concat().into()),
        })
    };
    let storage = |binding: u32, visibility: wgpu::ShaderStages, read_only: bool| {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    };
    let uniform = |visibility: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| {
        let group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries,
        });
        let pipeline = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(&group)],
            immediate_size: 0,
        });
        (group, pipeline)
    };
    let compute =
        |label: &str, layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
    let c = wgpu::ShaderStages::COMPUTE;

    let (keep_enable, keep_impl) = if subgroups.exact_wave_32() {
        (subgroups.directive(), KEEP_SG32_WGSL)
    } else if subgroups.wave_parallel_256() {
        (subgroups.directive(), KEEP_SG_GENERAL_WGSL)
    } else {
        ("", KEEP_FALLBACK_WGSL)
    };
    let keep_module = module(
        "entelechy rasterize keep",
        &[
            keep_enable,
            SETUP_WGSL,
            TRIANGLES_WGSL,
            KEEP_COMMON_WGSL,
            keep_impl,
        ],
    );
    let (keep_layout, keep_pipeline_layout) = layout(
        "entelechy rasterize keep",
        &[
            uniform(c),
            storage(1, c, true),
            storage(2, c, true),
            storage(3, c, false),
            storage(4, c, false),
            storage(5, c, false),
            storage(6, c, false),
            storage(7, c, false),
            storage(8, c, false),
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: c,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    );
    let mark = compute(
        "entelechy rasterize mark",
        &keep_pipeline_layout,
        &keep_module,
        "mark",
    );
    let sum = compute(
        "entelechy rasterize sum",
        &keep_pipeline_layout,
        &keep_module,
        "sum",
    );
    let place = compute(
        "entelechy rasterize place",
        &keep_pipeline_layout,
        &keep_module,
        "place",
    );

    let mask_wgsl = |mask_binding: &str, sampler_binding: &str| {
        if mask_filtered {
            MASK_FILTERED_WGSL
                .replace("{mask}", mask_binding)
                .replace("{sampler}", sampler_binding)
        } else {
            MASK_LOAD_WGSL.replace("{mask}", mask_binding)
        }
    };
    let mask_tex_entry =
        |binding: u32, visibility: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float {
                    filterable: mask_filtered,
                },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
    let mask_sampler_entry =
        |binding: u32, visibility: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };

    let (draw_small_layout, draw_small) = if here {
        let small_mask = mask_wgsl("6", "7");
        let small_mod = module(
            "entelechy rasterize draw_small",
            &[SETUP_WGSL, TRIANGLES_WGSL, &small_mask, DRAW_SMALL_WGSL],
        );
        let mut small_entries = vec![
            uniform(c),
            storage(1, c, true),
            storage(2, c, true),
            storage(3, c, true),
            storage(4, c, true),
            storage(5, c, false),
            mask_tex_entry(6, c),
        ];
        if mask_filtered {
            small_entries.push(mask_sampler_entry(7, c));
        }
        let (sl, spl) = layout("entelechy rasterize draw_small", &small_entries);
        let sp = compute("entelechy rasterize draw_small", &spl, &small_mod, "main");
        (Some(sl), Some(sp))
    } else {
        (None, None)
    };

    let feed = if indexed { INDEXED_WGSL } else { PULLED_WGSL };
    // An `enable` directive comes first in a module.
    let enable = if indexed {
        "enable primitive_index;\n"
    } else {
        ""
    };
    let draw_mask = mask_wgsl("4", "5");
    let draw_module = module(
        "entelechy rasterize draw",
        &[enable, SETUP_WGSL, &draw_mask, feed],
    );
    let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
    let mut draw_entries = vec![
        uniform(vf),
        storage(1, wgpu::ShaderStages::VERTEX_FRAGMENT, true),
        storage(2, wgpu::ShaderStages::VERTEX, true),
        storage(3, vf, true),
        mask_tex_entry(4, wgpu::ShaderStages::FRAGMENT),
    ];
    if mask_filtered {
        draw_entries.push(mask_sampler_entry(5, wgpu::ShaderStages::FRAGMENT));
    }
    let (draw_layout, draw_pipeline_layout) = layout("entelechy rasterize draw", &draw_entries);
    let color_targets = [Some(wgpu::ColorTargetState {
        format: PLACE_FORMAT,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    })];
    let render = |vertex: &str, fragment: &str| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("entelechy rasterize draw"),
            layout: Some(&draw_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &draw_module,
                entry_point: Some(vertex),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &draw_module,
                entry_point: Some(fragment),
                compilation_options: Default::default(),
                targets: &color_targets,
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    let draw = render("vs", "fs");
    let draw_cut = render("vs_cut", "fs_cut");
    let hiz = build_hiz_pipelines(device);

    let resolve_module = module(
        "entelechy rasterize resolve",
        &[SETUP_WGSL, TRIANGLES_WGSL, RESOLVE_WGSL],
    );
    let (resolve_layout, resolve_pipeline_layout) = layout(
        "entelechy rasterize resolve",
        &[
            uniform(c),
            storage(1, c, true),
            storage(2, c, true),
            storage(3, c, true),
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: c,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            storage(5, c, false),
            storage(6, c, true),
            storage(7, c, true),
        ],
    );
    let resolve = compute(
        "entelechy rasterize resolve",
        &resolve_pipeline_layout,
        &resolve_module,
        "main",
    );
    let cover = compute(
        "entelechy cover resolve",
        &resolve_pipeline_layout,
        &resolve_module,
        "cover",
    );
    let empty_mask_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rasterize empty mask"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let empty_mask_view = empty_mask_tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2),
        ..Default::default()
    });
    if let Some(error) = wait.block_on(guard.pop()) {
        return Err(format!(
            "rasterize pipeline: {}",
            crate::device_error(&error)
        ));
    }
    Ok(RasterPipelines {
        keep_layout,
        mark,
        sum,
        place,
        draw_small_layout,
        draw_small,
        draw_layout,
        draw,
        draw_cut,
        hiz,
        resolve_layout,
        resolve,
        cover,
        empty_mask_view,
        indexed,
        here,
        mask_filtered,
    })
}

/// A target of this shape from the backend's pile, written as an
/// attachment and read in place by the resolving pass.
fn target(
    gpu: &WgpuBackend,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    gpu.targets().take(
        &gpu.device,
        label,
        width,
        height,
        format,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    )
}

/// Rasterizes on the device into a new resident surface grid.
pub(crate) fn rasterize(
    gpu: &WgpuBackend,
    pipelines: &OnceLock<Result<RasterPipelines, String>>,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
    op: NativeOp,
) -> Result<Vec<Handle>, ExecutionError> {
    // The cover's keys before it, which it reads and leaves its least over.
    let mut before = None;
    let (vertices, triangles, mask, alpha) = match op {
        NativeOp::Cover => {
            let (vertices, triangles, keys) =
                entelechy_rules::raster::check_cover(invocation, scope)?;
            before = Some(keys);
            (vertices, triangles, None, 0.0)
        }
        NativeOp::RasterizeCutout => {
            let (vertices, triangles, mask, alpha) =
                entelechy_rules::raster::check_rasterize_cutout(invocation, scope)?;
            (vertices, triangles, Some(mask), alpha)
        }
        NativeOp::RasterizeCutoutCanvas => {
            let (vertices, triangles, mask, alpha) =
                entelechy_rules::raster::check_rasterize_cutout_canvas(invocation, scope)?;
            (vertices, triangles, Some(mask), alpha)
        }
        NativeOp::RasterizeCanvas => {
            let (vertices, triangles) =
                entelechy_rules::raster::check_rasterize_canvas(invocation, scope)?;
            (vertices, triangles, None, 0.0)
        }
        NativeOp::Rasterize => {
            let (vertices, triangles) = check_rasterize(invocation, scope)?;
            (vertices, triangles, None, 0.0)
        }
        NativeOp::Trace | NativeOp::TraceInstances => {
            return Err(ExecutionError::Device(
                "rasterizer called for trace op".into(),
            ));
        }
    };
    let pipelines = match pipelines.get_or_init(|| {
        build(
            &gpu.device,
            gpu.wait,
            gpu.primitive_index,
            gpu.int64_atomics,
            gpu.subgroups,
            gpu.float32_filterable,
        )
    }) {
        Ok(pipelines) => pipelines,
        Err(error) => return Err(ExecutionError::Device(error.clone())),
    };
    let _guard = gpu
        .lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let domain = invocation.domain;
    let (width, height) = (domain.width(), domain.height());
    let vertex_count = scope.size(vertices)?.count() as u32;
    // With no vertex rows nothing is drawn: no triangle has a corner.
    let triangle_count = if vertex_count == 0 {
        0
    } else {
        scope.size(triangles)?.count() as u32
    };
    let vertex_buffer = gpu.bind_input(scope, vertices)?;
    let triangle_buffer = gpu.bind_input(scope, triangles)?;

    // The mask is always bound as a 2D tiled hardware texture sharing the
    // immutable `DeviceImage` texture mirror with shading passes (zero extra
    // copies after first use); with none it is a 1x1 dummy texture left unread.
    let mask_size = match mask {
        Some(mask) => scope.size(mask)?,
        None => entelechy_abi::Domain::Grid(1, 1),
    };
    let (mask_view, mask_src_buf, mask_mirror_plan) = if let Some(mask_h) = mask {
        let buf = gpu.bind_input(scope, mask_h)?;
        let image = scope
            .resident(mask_h)?
            .and_then(|dev| dev.downcast_ref::<DeviceImage>())
            .ok_or_else(|| {
                ExecutionError::Device("rasterize mask is not resident on the GPU".to_owned())
            })?;
        let mask_port = entelechy_ir::Port {
            name: "mask".into(),
            ty: entelechy_ir::PortType {
                element: entelechy_ir::Type::Vec4,
                shape: entelechy_abi::Shape::Grid,
                schema: entelechy_rules::raster::mask_schema(),
            },
        };
        let (view, plan) =
            gpu.plan_texture_mirror(image, mask_h, &mask_port, mask_size, 0, false, false, None)?;
        (view, Some(buf), plan)
    } else {
        (pipelines.empty_mask_view.clone(), None, None)
    };
    let block_count = triangle_count.div_ceil(BLOCK);
    let blocks_x = block_count.clamp(1, MOST_GROUPS);
    let blocks_y = block_count.div_ceil(blocks_x).max(1);
    let use_hiz = triangle_count >= HIZ_TRI_THRESHOLD;
    let (_, _, hiz_mip_count) = if use_hiz {
        hiz_dims(width, height)
    } else {
        (1, 1, 0)
    };
    let encode_setup_bytes = |small: u32, hiz_mips: u32, occ_only: u32| -> [u8; 64] {
        let words: [u32; 16] = [
            (width as f32).to_bits(),
            (height as f32).to_bits(),
            vertex_count,
            triangle_count,
            blocks_x,
            block_count,
            mask_size.width(),
            mask_size.height(),
            alpha.to_bits(),
            u32::from(mask.is_some()),
            small,
            hiz_mips,
            occ_only,
            0,
            0,
            0,
        ];
        let mut bytes = [0u8; 64];
        for (slot, word) in bytes.chunks_exact_mut(4).zip(words) {
            slot.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    };
    let small_limit = if pipelines.here { SMALL } else { 0 };
    let main_setup_bytes = encode_setup_bytes(small_limit, hiz_mip_count, 0);
    let occ_setup_bytes = encode_setup_bytes(small_limit, 0, 1);
    let hiz_payloads = if use_hiz {
        hiz_level_payloads(width, height, before.is_some())
    } else {
        Vec::new()
    };
    // The keys the indirect micro-triangle pass draws into, a pixel each,
    // cleared to none. Check page pacing and bind `before` BEFORE allocating
    // from the batch uniform slab so no flush can happen after allocating slices.
    let pixels = u64::from(width) * u64::from(height);
    let est_ops = u64::from(triangle_count)
        .saturating_mul(64)
        .saturating_add(pixels.saturating_mul(16));
    if gpu.wait.parks_with_host()
        && !gpu.measured()
        && gpu.page_queue.full(est_ops, gpu.page_pace_ops_budget())
    {
        gpu.batch.flush()?;
        gpu.drain_page()?;
    }
    // The keys before the cover; the surface's pass binds the small
    // triangles' keys there, unread.
    let before_buffer = match before {
        Some(keys) => Some(gpu.bind_input(scope, keys)?),
        None => None,
    };
    let batched = gpu.batch.can_batch(gpu.measured(), &gpu.frames);
    if !batched {
        gpu.batch.flush()?;
    }
    let mut all_uniform_payloads: Vec<&[u8]> = Vec::with_capacity(3 + hiz_payloads.len());
    all_uniform_payloads.push(&main_setup_bytes);
    if use_hiz {
        all_uniform_payloads.push(&occ_setup_bytes);
        for p in &hiz_payloads {
            all_uniform_payloads.push(p.as_slice());
        }
    }
    let mask_pad_idx = mask_mirror_plan
        .as_ref()
        .and_then(|plan| plan.pad_params.as_ref())
        .map(|pad| {
            let idx = all_uniform_payloads.len();
            all_uniform_payloads.push(pad.as_slice());
            idx
        });
    let mut unbatched_uniforms = Vec::new();
    let uniform_slices = if batched {
        gpu.batch.alloc_uniforms(&all_uniform_payloads)?
    } else {
        let mut slices = Vec::with_capacity(all_uniform_payloads.len());
        for p in &all_uniform_payloads {
            let buf = gpu.take_uniform(p);
            unbatched_uniforms.push(buf.clone());
            slices.push(crate::batch::UniformSlice::entire(buf));
        }
        slices
    };
    let setup_slice = &uniform_slices[0];
    let occ_setup_slice = use_hiz.then(|| &uniform_slices[1]);
    let hiz_level_slices = if use_hiz {
        &uniform_slices[2..2 + hiz_payloads.len()]
    } else {
        &[]
    };
    let spare = gpu.spare();
    let mut extra_scratch: Vec<wgpu::Buffer> = Vec::new();
    let mut pending_mask_mirror: Option<(
        Handle,
        crate::texture::TextureMirror,
        crate::texture::PendingMirrorCopy,
    )> = None;
    if let Some(plan) = mask_mirror_plan {
        let padded = if let Some(pad_idx) = mask_pad_idx {
            let byte_size =
                (u64::from(plan.padded_row_bytes) * u64::from(plan.height) * u64::from(plan.depth))
                    .max(16)
                    .div_ceil(16)
                    * 16;
            let scratch_buf = spare.take(
                &gpu.device,
                byte_size,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            extra_scratch.push(scratch_buf.clone());
            Some((
                scratch_buf,
                uniform_slices[pad_idx].clone(),
                plan.padded_row_bytes,
                plan.fold_x,
            ))
        } else {
            None
        };
        let copy = crate::texture::PendingMirrorCopy {
            src: mask_src_buf.expect("mask source buffer"),
            dst: plan.mirror.texture.clone(),
            view_dim: plan.mirror.view_dim,
            width: plan.width,
            height: plan.height,
            depth: plan.depth,
            row_bytes: plan.row_bytes,
            padded,
            mip_scratch: None,
            mip_levels: Vec::new(),
        };
        pending_mask_mirror = Some((plan.handle, plan.mirror, copy));
    }

    // The keeping pass's buffers, from the pile and back to it.
    let scratch = |words: u64, usage: wgpu::BufferUsages| {
        spare.take(
            &gpu.device,
            (words.max(4) * 4).div_ceil(16) * 16,
            usage | wgpu::BufferUsages::STORAGE,
        )
    };
    let marks = scratch(u64::from(block_count) * 8, wgpu::BufferUsages::empty());
    let offsets = scratch(u64::from(block_count), wgpu::BufferUsages::empty());
    let indices = scratch(u64::from(triangle_count) * 3, wgpu::BufferUsages::INDEX);
    let rows = scratch(u64::from(triangle_count), wgpu::BufferUsages::empty());
    let args = scratch(
        16,
        wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
    );
    let small_rows = scratch(
        if pipelines.here {
            u64::from(triangle_count)
        } else {
            4
        },
        wgpu::BufferUsages::empty(),
    );
    let keys = scratch(
        if pipelines.here { pixels * 2 } else { 4 },
        wgpu::BufferUsages::COPY_DST,
    );

    fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding,
            resource: buffer.as_entire_binding(),
        }
    }
    let places = target(gpu, "surface places", width, height, PLACE_FORMAT);
    let depth = target(gpu, "surface depth", width, height, DEPTH_FORMAT);
    let places_view = places.create_view(&Default::default());
    let depth_view = depth.create_view(&Default::default());
    let hiz_tex = use_hiz.then(|| take_hiz_texture(gpu, width, height));
    let hiz_full_view = hiz_tex.as_ref().map(|tex| {
        tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("surface hiz full"),
            format: Some(raster_hiz::HIZ_FORMAT),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(hiz_mip_count),
            base_array_layer: 0,
            array_layer_count: Some(1),
            usage: None,
        })
    });
    let keep_hiz_view = hiz_full_view.as_ref().unwrap_or(&pipelines.hiz.empty_view);

    let occ_keep_group = occ_setup_slice.map(|occ_u| {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entelechy rasterize occ keep"),
            layout: &pipelines.keep_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: occ_u.binding(),
                },
                entry(1, &vertex_buffer),
                entry(2, &triangle_buffer),
                entry(3, &marks),
                entry(4, &offsets),
                entry(5, &indices),
                entry(6, &rows),
                entry(7, &args),
                entry(8, &small_rows),
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&pipelines.hiz.empty_view),
                },
            ],
        })
    });
    let keep_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("entelechy rasterize keep"),
        layout: &pipelines.keep_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: setup_slice.binding(),
            },
            entry(1, &vertex_buffer),
            entry(2, &triangle_buffer),
            entry(3, &marks),
            entry(4, &offsets),
            entry(5, &indices),
            entry(6, &rows),
            entry(7, &args),
            entry(8, &small_rows),
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::TextureView(keep_hiz_view),
            },
        ],
    });
    let draw_small_group = pipelines.draw_small_layout.as_ref().map(|sl| {
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: setup_slice.binding(),
            },
            entry(1, &vertex_buffer),
            entry(2, &triangle_buffer),
            entry(3, &small_rows),
            entry(4, &args),
            entry(5, &keys),
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&mask_view),
            },
        ];
        if pipelines.mask_filtered {
            entries.push(wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::Sampler(&gpu.sampler_clamp),
            });
        }
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entelechy rasterize draw_small"),
            layout: sl,
            entries: &entries,
        })
    });
    let occ_draw_group = occ_setup_slice.map(|occ_u| {
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: occ_u.binding(),
            },
            entry(1, &vertex_buffer),
            entry(2, &indices),
            entry(3, &rows),
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&mask_view),
            },
        ];
        if pipelines.mask_filtered {
            entries.push(wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(&gpu.sampler_clamp),
            });
        }
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entelechy rasterize occ draw"),
            layout: &pipelines.draw_layout,
            entries: &entries,
        })
    });
    let draw_group = {
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: setup_slice.binding(),
            },
            entry(1, &vertex_buffer),
            entry(2, &indices),
            entry(3, &rows),
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&mask_view),
            },
        ];
        if pipelines.mask_filtered {
            entries.push(wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(&gpu.sampler_clamp),
            });
        }
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entelechy rasterize draw"),
            layout: &pipelines.draw_layout,
            entries: &entries,
        })
    };

    // A surface, or two words a key.
    let (schema, words) = match before {
        Some(_) => (entelechy_rules::raster::canvas_schema(), 2),
        None => (surface_schema(), SURFACE_WORDS as u64),
    };
    let output_size = width as u64 * height as u64 * words * 4;
    let output_buffer = spare.take(
        &gpu.device,
        output_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let resolve_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("entelechy rasterize resolve"),
        layout: &pipelines.resolve_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: setup_slice.binding(),
            },
            entry(1, &vertex_buffer),
            entry(2, &triangle_buffer),
            entry(3, &rows),
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&places_view),
            },
            entry(5, &output_buffer),
            entry(6, &keys),
            entry(7, before_buffer.as_ref().unwrap_or(&keys)),
        ],
    });

    // The set is made before the encoder so the opening timestamp is its
    // first command: the first pass opens it, the last closes it.
    let timing_set = gpu.timing_set(2);
    let mut encoder = if batched {
        gpu.batch.take_or_open()
    } else {
        gpu.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("entelechy rasterize"),
            })
    };
    let drawing = triangle_count > 0;
    if let Some((mask_h, mirror, copy)) = pending_mask_mirror {
        gpu.encode_mirror_copy(&mut encoder, &copy)?;
        if let Some(image) = scope
            .resident(mask_h)?
            .and_then(|dev| dev.downcast_ref::<DeviceImage>())
        {
            *image.mirror.lock().unwrap_or_else(|p| p.into_inner()) = Some(mirror);
        }
    }

    // Phase 1 + Phase 2 (zero-redraw 2-phase Hi-Z, opaque and alpha-cutout):
    // draw macro-occluders ONCE directly into `places` (global row + 1) and
    // `depth`, then build the conservative MAX-Z Hi-Z pyramid. Transparent
    // cutout holes leave depth at 1.0, so MAX-Z over any tile containing holes
    // stays 1.0 and never falsely culls.
    let draw_pipeline = if mask.is_some() {
        &pipelines.draw_cut
    } else {
        &pipelines.draw
    };
    if use_hiz {
        encoder.clear_buffer(&args, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("entelechy rasterize occ"),
                timestamp_writes: timing_set
                    .as_ref()
                    .map(|timing| timing.compute(Some(0), None)),
            });
            pass.set_bind_group(0, occ_keep_group.as_ref().expect("occ keep"), &[]);
            pass.set_pipeline(&pipelines.mark);
            pass.dispatch_workgroups(blocks_x, blocks_y, 1);
            pass.set_pipeline(&pipelines.sum);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&pipelines.place);
            pass.dispatch_workgroups(blocks_x, blocks_y, 1);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("entelechy rasterize occ draw"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &places_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(draw_pipeline);
            pass.set_bind_group(0, occ_draw_group.as_ref().expect("occ draw"), &[]);
            if pipelines.indexed {
                pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed_indirect(&args, 0);
            } else {
                pass.draw_indirect(&args, 0);
            }
        }
        encode_hiz_pyramid(
            &gpu.device,
            &pipelines.hiz,
            &mut encoder,
            &depth_view,
            before_buffer.as_ref().unwrap_or(&keys),
            hiz_tex.as_ref().expect("hiz texture"),
            width,
            height,
            hiz_level_slices,
        );
    }

    // Phase 3: visibility culling of non-occluders + compaction + indirect micro-triangles.
    if pipelines.here {
        encoder.clear_buffer(&keys, 0, None);
    }
    if drawing {
        encoder.clear_buffer(&args, 0, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("entelechy rasterize keep"),
            timestamp_writes: (!use_hiz)
                .then_some(())
                .and(timing_set.as_ref())
                .map(|timing| timing.compute(Some(0), None)),
        });
        pass.set_bind_group(0, &keep_group, &[]);
        pass.set_pipeline(&pipelines.mark);
        pass.dispatch_workgroups(blocks_x, blocks_y, 1);
        pass.set_pipeline(&pipelines.sum);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&pipelines.place);
        pass.dispatch_workgroups(blocks_x, blocks_y, 1);
        if let (Some(ds_pipe), Some(ds_group)) = (&pipelines.draw_small, &draw_small_group) {
            pass.set_pipeline(ds_pipe);
            pass.set_bind_group(0, ds_group, &[]);
            pass.dispatch_workgroups_indirect(&args, 32);
        }
    }
    // Phase 4: hardware visibility rasterization of surviving non-occluders in
    // exact row order, loading Phase 1's visibility + depth buffers when Hi-Z
    // ran so no macro-occluder is ever shaded twice and hardware early-Z stays warm.
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("entelechy rasterize draw"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &places_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if use_hiz {
                        wgpu::LoadOp::Load
                    } else {
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: if use_hiz {
                        wgpu::LoadOp::Load
                    } else {
                        wgpu::LoadOp::Clear(1.0)
                    },
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: (!drawing)
                .then_some(())
                .and(timing_set.as_ref())
                .map(|timing| timing.render(Some(0), None)),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if drawing {
            pass.set_pipeline(draw_pipeline);
            pass.set_bind_group(0, &draw_group, &[]);
            if pipelines.indexed {
                pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed_indirect(&args, 0);
            } else {
                pass.draw_indirect(&args, 0);
            }
        }
    }
    // Phase 5: resolve surface / cover keys.
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("entelechy rasterize resolve"),
            timestamp_writes: timing_set
                .as_ref()
                .map(|timing| timing.compute(None, Some(1))),
        });
        pass.set_pipeline(if before.is_some() {
            &pipelines.cover
        } else {
            &pipelines.resolve
        });
        pass.set_bind_group(0, &resolve_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }
    let timing = gpu.resolve_timing(&mut encoder, timing_set.as_ref(), 2);
    let mut keep_textures = vec![places, depth];
    if let Some(hiz) = hiz_tex {
        keep_textures.push(hiz);
    }
    if batched {
        gpu.frames.note_batched_run();
        let page_budget = gpu
            .wait
            .parks_with_host()
            .then(|| gpu.page_pace_ops_budget());
        let mut keep_buffers = vec![marks, offsets, indices, rows, args, small_rows, keys];
        keep_buffers.extend(extra_scratch);
        gpu.batch
            .finish_node(encoder, est_ops, keep_buffers, keep_textures, page_budget)?;
        if gpu.wait.parks_with_host() && gpu.page_queue.full(0, gpu.page_pace_ops_budget()) {
            gpu.batch.flush()?;
            gpu.drain_page()?;
        }
    } else {
        let scopes = Scopes::push(&gpu.device);
        let submission = gpu.submit(encoder.finish());
        if gpu.wait.parks_with_host() && !gpu.measured() {
            gpu.page_queue.note(est_ops);
            if gpu.page_queue.full(0, gpu.page_pace_ops_budget()) {
                gpu.drain_page()?;
            }
        }
        gpu.settle_measured(submission, timing);
        if gpu.wait.parks_with_host() && !gpu.measured() {
            gpu.defer_check(Box::new(scopes.pop_later(gpu.wait, gpu.spent.clone())));
        } else if let Err(error) = scopes.pop(gpu.wait, "") {
            return Err(ExecutionError::Device(format!(
                "rasterize: {}",
                error.said()
            )));
        }
        for texture in keep_textures {
            gpu.targets().keep(texture);
        }
        for buffer in unbatched_uniforms
            .into_iter()
            .chain([marks, offsets, indices, rows, args, small_rows, keys])
            .chain(extra_scratch)
        {
            spare.keep(buffer, None);
        }
    }
    let output = scope.adopt_resident(
        schema,
        domain,
        Box::new(DeviceImage::new(output_buffer, Some(gpu.spare()), None)),
    )?;
    Ok(vec![output])
}
