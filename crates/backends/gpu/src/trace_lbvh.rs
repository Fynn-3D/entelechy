//! Full-GPU zero-readback LBVH builder and 1-pass in-place dynamic BVH refitter
//! for portable GPU Compute BVH ray tracing (WebGPU, non-RT adapters, and
//! `GpuBackend::without_ray_query()`).
//!
//! Pipeline (100% portable WGSL 1.0, `<= 6` storage buffers per pass, zero CPU
//! readback, zero 64-bit atomics, zero subgroup requirements):
//! 1. **Centroid Bounds** (`lbvh_clear_meta` + `lbvh_prim_bounds`): computes per-triangle
//!    centroids and global scene centroid AABB via order-preserving IEEE-754 sortable u32
//!    atomics.
//! 2. **30-bit Morton Codes** (`lbvh_morton`): maps centroids onto a `1024^3` Z-order curve.
//! 3. **8-Pass 4-Bit Bitmask Stable GPU Radix Sort** (`radix_hist` / `radix_scan_blocks` /
//!    `radix_scan_buckets` / `radix_scatter`): sorts `(morton, tri_idx)` pairs completely on
//!    GPU using 512 B workgroup bitmasks and `O(1)` `countOneBits` local ranking.
//! 4. **Karras 2012 Parallel Binary Radix Tree** (`lbvh_hierarchy` + `lbvh_classify_tiers` +
//!    `lbvh_finalize_args`): constructs exact `2N - 1` internal + leaf nodes directly in the
//!    32-byte `BvhNode` flat layout (encoding explicit Karras right-child indices with bit 31
//!    set in word 7 so `BVH_TRACE_WGSL` traverses both SAH and LBVH trees transparently),
//!    and partitions internal nodes into maximal `<= 64`-leaf Tier-1 subtrees plus an upper
//!    tree list.
//! 5. **Race-Free 2-Tier Bottom-Up AABB Refit** (`lbvh_refit_leaves` + indirect
//!    `lbvh_refit_tier1` + indirect `lbvh_refit_upper`):
//!    - Computes every leaf AABB in parallel (degenerate/NaN triangles get an inverted empty
//!      AABB `[1e30..-1e30]`, so dynamically appearing/disappearing triangles work seamlessly
//!      even on refit frames!).
//!    - Computes all maximal `<= 64`-leaf subtrees via iterative post-order traversal with
//!      zero cross-workgroup atomics (`dispatch_workgroups_indirect`).
//!    - Finishes the tiny upper tree in one 256-thread workgroup using two-phase
//!      `workgroupBarrier()` synchronization (100% data-race-free under WebGPU's relaxed
//!      memory model).
//!
//! On dynamic deforming meshes (same `tris_id`, new `verts_id`), frames `1..31` skip steps
//! 1–4 completely and execute **only step 5** in place on the GPU, rebuilding full Morton
//! topology every 32 animated frames to preserve spatial BVH quality.

use crate::wgsl;

const WORKGROUP_64: u32 = 64;
const RADIX_BLOCK: u32 = 256;
const MOST_GROUPS: u32 = 65_535;
const RADIX_PASSES: u32 = 8;

/// Number of dynamic in-place refit frames before refreshing Morton topology.
pub(crate) const LBVH_REFIT_PERIOD: u32 = 32;

const LBVH_COMMON_WGSL: &str = "
struct LbvhSetup {
    v_count: u32,
    t_count: u32,
    num_blocks: u32,
    blocks_x: u32,
    shift: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

fn is_finite_bits(w: u32) -> bool {
    return (w & 0x7F800000u) != 0x7F800000u;
}

fn float_to_sortable_u32(v: f32) -> u32 {
    let b = bitcast<u32>(v);
    let mask = select(0x80000000u, 0xFFFFFFFFu, (b & 0x80000000u) != 0u);
    return b ^ mask;
}

fn sortable_u32_to_float(u: u32) -> f32 {
    let mask = select(0xFFFFFFFFu, 0x80000000u, (u & 0x80000000u) != 0u);
    return bitcast<f32>(u ^ mask);
}
";

const LBVH_BOUNDS_WGSL: &str = "
@group(0) @binding(0) var<uniform> setup: LbvhSetup;
@group(0) @binding(1) var<storage, read> vertices: array<u32>;
@group(0) @binding(2) var<storage, read> triangles: array<u32>;
@group(0) @binding(3) var<storage, read_write> centroids: array<u32>;
// meta layout (16 u32 words):
// [0..2]: atomic min sortable xyz, [3..5]: atomic max sortable xyz,
// [6]: valid_count, [7]: tier1_count, [8]: upper_count, [9..15]: reserved
@group(0) @binding(4) var<storage, read_write> lbvh_meta: array<atomic<u32>>;

fn finite_vert(idx: u32) -> bool {
    let b = idx * 3u;
    return is_finite_bits(vertices[b])
        && is_finite_bits(vertices[b + 1u])
        && is_finite_bits(vertices[b + 2u]);
}

fn load_vert(idx: u32) -> vec3<f32> {
    let b = idx * 3u;
    return vec3<f32>(
        bitcast<f32>(vertices[b]),
        bitcast<f32>(vertices[b + 1u]),
        bitcast<f32>(vertices[b + 2u]),
    );
}

@compute @workgroup_size(1, 1, 1)
fn clear_meta() {
    let inf_pos = float_to_sortable_u32(3.4e38);
    let inf_neg = float_to_sortable_u32(-3.4e38);
    atomicStore(&lbvh_meta[0u], inf_pos);
    atomicStore(&lbvh_meta[1u], inf_pos);
    atomicStore(&lbvh_meta[2u], inf_pos);
    atomicStore(&lbvh_meta[3u], inf_neg);
    atomicStore(&lbvh_meta[4u], inf_neg);
    atomicStore(&lbvh_meta[5u], inf_neg);
    atomicStore(&lbvh_meta[6u], 0u);
    atomicStore(&lbvh_meta[7u], 0u);
    atomicStore(&lbvh_meta[8u], 0u);
}

@compute @workgroup_size(64, 1, 1)
fn prim_bounds(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let t = gid.x + gid.y * (nwg.x * 64u);
    if t >= setup.t_count {
        return;
    }
    let max_v = i32(setup.v_count) - 1i;
    let tb = t * 3u;
    let ia = u32(clamp(bitcast<i32>(triangles[tb]), 0i, max_v));
    let ib = u32(clamp(bitcast<i32>(triangles[tb + 1u]), 0i, max_v));
    let ic = u32(clamp(bitcast<i32>(triangles[tb + 2u]), 0i, max_v));
    var ok = finite_vert(ia) && finite_vert(ib) && finite_vert(ic);
    var c = vec3<f32>(0.0, 0.0, 0.0);
    if ok {
        let v0 = load_vert(ia);
        let v1 = load_vert(ib);
        let v2 = load_vert(ic);
        let cr = cross(v1 - v0, v2 - v0);
        let len2 = dot(cr, cr);
        ok = len2 > 0.0 && is_finite_bits(bitcast<u32>(len2));
        if ok {
            let bmin = min(min(v0, v1), v2);
            let bmax = max(max(v0, v1), v2);
            c = (bmin + bmax) * 0.5;
            ok = is_finite_bits(bitcast<u32>(c.x))
                && is_finite_bits(bitcast<u32>(c.y))
                && is_finite_bits(bitcast<u32>(c.z));
        }
    }
    let cb = t * 4u;
    if ok {
        centroids[cb] = bitcast<u32>(c.x);
        centroids[cb + 1u] = bitcast<u32>(c.y);
        centroids[cb + 2u] = bitcast<u32>(c.z);
        centroids[cb + 3u] = 1u;
        let sx = float_to_sortable_u32(c.x);
        let sy = float_to_sortable_u32(c.y);
        let sz = float_to_sortable_u32(c.z);
        atomicMin(&lbvh_meta[0u], sx);
        atomicMin(&lbvh_meta[1u], sy);
        atomicMin(&lbvh_meta[2u], sz);
        atomicMax(&lbvh_meta[3u], sx);
        atomicMax(&lbvh_meta[4u], sy);
        atomicMax(&lbvh_meta[5u], sz);
        atomicAdd(&lbvh_meta[6u], 1u);
    } else {
        centroids[cb] = 0u;
        centroids[cb + 1u] = 0u;
        centroids[cb + 2u] = 0u;
        centroids[cb + 3u] = 0u;
    }
}
";

const LBVH_MORTON_WGSL: &str = "
@group(0) @binding(0) var<uniform> setup: LbvhSetup;
@group(0) @binding(1) var<storage, read> centroids: array<u32>;
@group(0) @binding(2) var<storage, read> lbvh_meta: array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> vals_out: array<u32>;

fn expand_bits10(v_in: u32) -> u32 {
    var v = v_in & 0x3FFu;
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

@compute @workgroup_size(64, 1, 1)
fn morton(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let t = gid.x + gid.y * (nwg.x * 64u);
    if t >= setup.t_count {
        return;
    }
    vals_out[t] = t;
    let cb = t * 4u;
    if centroids[cb + 3u] == 0u || lbvh_meta[6u] == 0u {
        keys_out[t] = 0u;
        return;
    }
    let cmin = vec3<f32>(
        sortable_u32_to_float(lbvh_meta[0u]),
        sortable_u32_to_float(lbvh_meta[1u]),
        sortable_u32_to_float(lbvh_meta[2u]),
    );
    let cmax = vec3<f32>(
        sortable_u32_to_float(lbvh_meta[3u]),
        sortable_u32_to_float(lbvh_meta[4u]),
        sortable_u32_to_float(lbvh_meta[5u]),
    );
    let ext = max(cmax - cmin, vec3<f32>(1e-20, 1e-20, 1e-20));
    let c = vec3<f32>(
        bitcast<f32>(centroids[cb]),
        bitcast<f32>(centroids[cb + 1u]),
        bitcast<f32>(centroids[cb + 2u]),
    );
    let norm = clamp((c - cmin) / ext, vec3<f32>(0.0), vec3<f32>(1.0));
    let ix = u32(min(norm.x * 1024.0, 1023.0));
    let iy = u32(min(norm.y * 1024.0, 1023.0));
    let iz = u32(min(norm.z * 1024.0, 1023.0));
    keys_out[t] = expand_bits10(ix) | (expand_bits10(iy) << 1u) | (expand_bits10(iz) << 2u);
}
";

const LBVH_RADIX_WGSL: &str = "
@group(0) @binding(0) var<uniform> setup: LbvhSetup;
@group(0) @binding(1) var<storage, read> keys_in: array<u32>;
@group(0) @binding(2) var<storage, read> vals_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> vals_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> block_hist: array<u32>;
@group(0) @binding(6) var<storage, read_write> bucket_buf: array<u32>;

var<workgroup> wg_hist: array<atomic<u32>, 16>;

@compute @workgroup_size(256, 1, 1)
fn radix_hist(
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    if local < 16u {
        atomicStore(&wg_hist[local], 0u);
    }
    workgroupBarrier();
    let block = group.x + group.y * setup.blocks_x;
    let idx = block * 256u + local;
    if block < setup.num_blocks && idx < setup.t_count {
        let d = (keys_in[idx] >> setup.shift) & 15u;
        atomicAdd(&wg_hist[d], 1u);
    }
    workgroupBarrier();
    if local < 16u && block < setup.num_blocks {
        block_hist[local * setup.num_blocks + block] = atomicLoad(&wg_hist[local]);
    }
}

var<workgroup> wg_scan: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn radix_scan_blocks(
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let bucket = group.x;
    let nb = setup.num_blocks;
    var running = 0u;
    for (var base = 0u; base < nb; base += 256u) {
        let b_idx = base + local;
        let val = select(0u, block_hist[bucket * nb + b_idx], b_idx < nb);
        wg_scan[local] = val;
        workgroupBarrier();
        for (var d = 1u; d < 256u; d <<= 1u) {
            var add = 0u;
            if local >= d {
                add = wg_scan[local - d];
            }
            workgroupBarrier();
            wg_scan[local] += add;
            workgroupBarrier();
        }
        let excl = wg_scan[local] - val + running;
        if b_idx < nb {
            block_hist[bucket * nb + b_idx] = excl;
        }
        running += wg_scan[255u];
        workgroupBarrier();
    }
    if local == 0u {
        bucket_buf[bucket] = running;
    }
}

var<workgroup> wg_bscan: array<u32, 16>;

@compute @workgroup_size(16, 1, 1)
fn radix_scan_buckets(@builtin(local_invocation_index) local: u32) {
    let v = bucket_buf[local];
    wg_bscan[local] = v;
    workgroupBarrier();
    for (var d = 1u; d < 16u; d <<= 1u) {
        var add = 0u;
        if local >= d {
            add = wg_bscan[local - d];
        }
        workgroupBarrier();
        wg_bscan[local] += add;
        workgroupBarrier();
    }
    bucket_buf[16u + local] = wg_bscan[local] - v;
}

var<workgroup> wg_bits: array<atomic<u32>, 128>;

@compute @workgroup_size(256, 1, 1)
fn radix_scatter(
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    if local < 128u {
        atomicStore(&wg_bits[local], 0u);
    }
    workgroupBarrier();
    let block = group.x + group.y * setup.blocks_x;
    let idx = block * 256u + local;
    let valid = block < setup.num_blocks && idx < setup.t_count;
    var key = 0u;
    var val = 0u;
    var digit = 0u;
    if valid {
        key = keys_in[idx];
        val = vals_in[idx];
        digit = (key >> setup.shift) & 15u;
        atomicOr(&wg_bits[digit * 8u + (local >> 5u)], 1u << (local & 31u));
    }
    workgroupBarrier();
    if valid {
        let word_idx = local >> 5u;
        let bit_idx = local & 31u;
        let mask_below = select((1u << bit_idx) - 1u, 0u, bit_idx == 0u);
        var rank = countOneBits(atomicLoad(&wg_bits[digit * 8u + word_idx]) & mask_below);
        for (var w = 0u; w < word_idx; w += 1u) {
            rank += countOneBits(atomicLoad(&wg_bits[digit * 8u + w]));
        }
        let dst = bucket_buf[16u + digit] + block_hist[digit * setup.num_blocks + block] + rank;
        keys_out[dst] = key;
        vals_out[dst] = val;
    }
}
";

const LBVH_HIERARCHY_WGSL: &str = "
@group(0) @binding(0) var<uniform> setup: LbvhSetup;
@group(0) @binding(1) var<storage, read> keys_sorted: array<u32>;
@group(0) @binding(2) var<storage, read_write> bvh_nodes: array<u32>;
@group(0) @binding(3) var<storage, read_write> parents: array<u32>;
@group(0) @binding(4) var<storage, read_write> sub_sizes: array<u32>;
@group(0) @binding(5) var<storage, read_write> mesh_desc: array<u32>;

fn clz_delta(i: i32, j: i32, n: i32) -> i32 {
    if j < 0 || j >= n {
        return -1;
    }
    let ki = keys_sorted[u32(i)];
    let kj = keys_sorted[u32(j)];
    if ki == kj {
        return 32 + i32(countLeadingZeros(u32(i) ^ u32(j)));
    }
    return i32(countLeadingZeros(ki ^ kj));
}

@compute @workgroup_size(64, 1, 1)
fn hierarchy(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.x + gid.y * (nwg.x * 64u);
    let n_u = setup.t_count;
    if idx == 0u {
        mesh_desc[0u] = 0u;
        mesh_desc[1u] = setup.v_count;
        mesh_desc[2u] = 0u;
        mesh_desc[3u] = 0u;
        mesh_desc[4u] = 0u;
        mesh_desc[5u] = select(2u * n_u - 1u, n_u, n_u <= 1u);
        mesh_desc[6u] = 0u;
        mesh_desc[7u] = 0u;
        if n_u >= 1u {
            parents[0u] = 0xFFFFFFFFu;
        }
    }
    if n_u <= 1u || idx >= n_u - 1u {
        return;
    }
    let i = i32(idx);
    let n = i32(n_u);
    let d_right = clz_delta(i, i + 1, n);
    let d_left = clz_delta(i, i - 1, n);
    let d = select(-1, 1, d_right > d_left);
    let delta_min = clz_delta(i, i - d, n);
    var lmax: i32 = 2;
    while clz_delta(i, i + lmax * d, n) > delta_min {
        lmax = lmax * 2;
    }
    var l: i32 = 0;
    var t_step: i32 = lmax / 2;
    while t_step >= 1 {
        if clz_delta(i, i + (l + t_step) * d, n) > delta_min {
            l = l + t_step;
        }
        t_step = t_step / 2;
    }
    let j = i + l * d;
    let first = min(i, j);
    let last = max(i, j);
    let delta_node = clz_delta(i, j, n);
    var s: i32 = 0;
    var step: i32 = l;
    loop {
        step = (step + 1) / 2;
        if clz_delta(i, i + (s + step) * d, n) > delta_node {
            s = s + step;
        }
        if step <= 1 {
            break;
        }
    }
    let gamma = i + s * d + min(d, 0);
    let num_internal = n_u - 1u;
    let left = select(u32(gamma), num_internal + u32(gamma), gamma == first);
    let right = select(u32(gamma + 1), num_internal + u32(gamma + 1), gamma + 1 == last);

    let nb = idx * 8u;
    bvh_nodes[nb + 3u] = left;
    // Set bit 31 on word 7 to encode explicit Karras right-child index while
    // preserving compatibility with SAH BVH nodes (`w7 == 0` -> `left + 1`).
    bvh_nodes[nb + 7u] = right | 0x80000000u;
    parents[left] = idx;
    parents[right] = idx;
    sub_sizes[idx] = u32(last - first + 1);
}
";

const LBVH_TIERS_WGSL: &str = "
@group(0) @binding(0) var<uniform> setup: LbvhSetup;
@group(0) @binding(1) var<storage, read> parents: array<u32>;
@group(0) @binding(2) var<storage, read> sub_sizes: array<u32>;
@group(0) @binding(3) var<storage, read_write> tier1_roots: array<u32>;
@group(0) @binding(4) var<storage, read_write> upper_nodes: array<u32>;
@group(0) @binding(5) var<storage, read_write> lbvh_meta: array<atomic<u32>>;
// Indirect dispatch arguments: [0..2] = tier1_refit (gx, gy, gz), [3..5] = upper_refit (gx, gy, gz)
@group(0) @binding(6) var<storage, read_write> indirect_args: array<u32>;

const TIER1_MAX_LEAVES: u32 = 64u;

@compute @workgroup_size(64, 1, 1)
fn classify_tiers(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.x + gid.y * (nwg.x * 64u);
    if setup.t_count <= 1u || idx >= setup.t_count - 1u {
        return;
    }
    let sz = sub_sizes[idx];
    if sz <= TIER1_MAX_LEAVES {
        let p = parents[idx];
        if p == 0xFFFFFFFFu || sub_sizes[p] > TIER1_MAX_LEAVES {
            let slot = atomicAdd(&lbvh_meta[7u], 1u);
            tier1_roots[slot] = idx;
        }
    } else {
        let slot = atomicAdd(&lbvh_meta[8u], 1u);
        upper_nodes[slot] = idx;
    }
}

@compute @workgroup_size(1, 1, 1)
fn finalize_tier_args() {
    let t1 = atomicLoad(&lbvh_meta[7u]);
    let t1_groups = (t1 + 63u) / 64u;
    let gx = min(t1_groups, 65535u);
    let safe_gx = max(gx, 1u);
    let gy = select(1u, (t1_groups + safe_gx - 1u) / safe_gx, t1_groups > 0u);
    indirect_args[0u] = gx;
    indirect_args[1u] = gy;
    indirect_args[2u] = 1u;

    let up = atomicLoad(&lbvh_meta[8u]);
    indirect_args[3u] = select(0u, 1u, up > 0u);
    indirect_args[4u] = 1u;
    indirect_args[5u] = 1u;
}
";

const LBVH_REFIT_LEAVES_WGSL: &str = "
@group(0) @binding(0) var<uniform> setup: LbvhSetup;
@group(0) @binding(1) var<storage, read> vertices: array<u32>;
@group(0) @binding(2) var<storage, read> triangles: array<u32>;
@group(0) @binding(3) var<storage, read> ordered: array<u32>;
@group(0) @binding(4) var<storage, read_write> bvh_nodes: array<u32>;

fn finite_vert(idx: u32) -> bool {
    let b = idx * 3u;
    return is_finite_bits(vertices[b])
        && is_finite_bits(vertices[b + 1u])
        && is_finite_bits(vertices[b + 2u]);
}

fn load_vert(idx: u32) -> vec3<f32> {
    let b = idx * 3u;
    return vec3<f32>(
        bitcast<f32>(vertices[b]),
        bitcast<f32>(vertices[b + 1u]),
        bitcast<f32>(vertices[b + 2u]),
    );
}

@compute @workgroup_size(64, 1, 1)
fn refit_leaves(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let leaf_i = gid.x + gid.y * (nwg.x * 64u);
    if leaf_i >= setup.t_count {
        return;
    }
    let t = ordered[leaf_i];
    let max_v = i32(setup.v_count) - 1i;
    let tb = t * 3u;
    let ia = u32(clamp(bitcast<i32>(triangles[tb]), 0i, max_v));
    let ib = u32(clamp(bitcast<i32>(triangles[tb + 1u]), 0i, max_v));
    let ic = u32(clamp(bitcast<i32>(triangles[tb + 2u]), 0i, max_v));
    var bmin = vec3<f32>(1e30, 1e30, 1e30);
    var bmax = vec3<f32>(-1e30, -1e30, -1e30);
    if finite_vert(ia) && finite_vert(ib) && finite_vert(ic) {
        let v0 = load_vert(ia);
        let v1 = load_vert(ib);
        let v2 = load_vert(ic);
        let cr = cross(v1 - v0, v2 - v0);
        let len2 = dot(cr, cr);
        if len2 > 0.0 && is_finite_bits(bitcast<u32>(len2)) {
            bmin = min(min(v0, v1), v2);
            bmax = max(max(v0, v1), v2);
        }
    }
    let num_internal = select(0u, setup.t_count - 1u, setup.t_count >= 2u);
    let nb = (num_internal + leaf_i) * 8u;
    bvh_nodes[nb] = bitcast<u32>(bmin.x);
    bvh_nodes[nb + 1u] = bitcast<u32>(bmin.y);
    bvh_nodes[nb + 2u] = bitcast<u32>(bmin.z);
    bvh_nodes[nb + 3u] = leaf_i;
    bvh_nodes[nb + 4u] = bitcast<u32>(bmax.x);
    bvh_nodes[nb + 5u] = bitcast<u32>(bmax.y);
    bvh_nodes[nb + 6u] = bitcast<u32>(bmax.z);
    bvh_nodes[nb + 7u] = 1u;
}
";

const LBVH_REFIT_TREE_WGSL: &str = "
@group(0) @binding(0) var<uniform> setup: LbvhSetup;
@group(0) @binding(1) var<storage, read> list_nodes: array<u32>;
@group(0) @binding(2) var<storage, read> sub_sizes: array<u32>;
@group(0) @binding(3) var<storage, read_write> bvh_nodes: array<u32>;
@group(0) @binding(4) var<storage, read_write> ready: array<u32>;
@group(0) @binding(5) var<storage, read_write> pending: array<u32>;
@group(0) @binding(6) var<storage, read> lbvh_meta: array<u32>;

const TIER1_MAX_LEAVES: u32 = 64u;

fn merge_node(n: u32, lc: u32, rc: u32) {
    let lb = lc * 8u;
    let rb = rc * 8u;
    let lmin = vec3<f32>(bitcast<f32>(bvh_nodes[lb]), bitcast<f32>(bvh_nodes[lb + 1u]), bitcast<f32>(bvh_nodes[lb + 2u]));
    let lmax = vec3<f32>(bitcast<f32>(bvh_nodes[lb + 4u]), bitcast<f32>(bvh_nodes[lb + 5u]), bitcast<f32>(bvh_nodes[lb + 6u]));
    let rmin = vec3<f32>(bitcast<f32>(bvh_nodes[rb]), bitcast<f32>(bvh_nodes[rb + 1u]), bitcast<f32>(bvh_nodes[rb + 2u]));
    let rmax = vec3<f32>(bitcast<f32>(bvh_nodes[rb + 4u]), bitcast<f32>(bvh_nodes[rb + 5u]), bitcast<f32>(bvh_nodes[rb + 6u]));
    let bmin = min(lmin, rmin);
    let bmax = max(lmax, rmax);
    let nb = n * 8u;
    bvh_nodes[nb] = bitcast<u32>(bmin.x);
    bvh_nodes[nb + 1u] = bitcast<u32>(bmin.y);
    bvh_nodes[nb + 2u] = bitcast<u32>(bmin.z);
    bvh_nodes[nb + 4u] = bitcast<u32>(bmax.x);
    bvh_nodes[nb + 5u] = bitcast<u32>(bmax.y);
    bvh_nodes[nb + 6u] = bitcast<u32>(bmax.z);
}

// Each thread owns one independent maximal `<= 64`-leaf subtree and computes all
// internal node AABBs in strict iterative post-order with zero atomics.
@compute @workgroup_size(64, 1, 1)
fn refit_tier1(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let idx = gid.x + gid.y * (nwg.x * 64u);
    if idx >= lbvh_meta[7u] || setup.t_count <= 1u {
        return;
    }
    let num_internal = setup.t_count - 1u;
    let root = list_nodes[idx];
    var stack: array<u32, 68>;
    var st_arr: array<u32, 68>;
    stack[0u] = root;
    st_arr[0u] = 0u;
    var sp = 1u;
    while sp > 0u {
        sp -= 1u;
        let n = stack[sp];
        let st = st_arr[sp];
        let nb = n * 8u;
        let lc = bvh_nodes[nb + 3u];
        let rc = bvh_nodes[nb + 7u] & 0x7FFFFFFFu;
        if st == 0u {
            stack[sp] = n;
            st_arr[sp] = 1u;
            sp += 1u;
            if rc < num_internal {
                stack[sp] = rc;
                st_arr[sp] = 0u;
                sp += 1u;
            }
            if lc < num_internal {
                stack[sp] = lc;
                st_arr[sp] = 0u;
                sp += 1u;
            }
        } else {
            merge_node(n, lc, rc);
        }
    }
}

fn child_is_ready(c: u32, num_internal: u32) -> bool {
    if c >= num_internal {
        return true;
    }
    if sub_sizes[c] <= TIER1_MAX_LEAVES {
        return true;
    }
    return ready[c] != 0u;
}

var<workgroup> wg_done: u32;

// Single 256-thread workgroup finishes all upper-tree nodes (`sub_size > 64`)
// using strict two-phase `workgroupBarrier()` + `workgroupUniformLoad`
// synchronization (100% data-race-free and Tint uniformity compliant).
@compute @workgroup_size(256, 1, 1)
fn refit_upper(@builtin(local_invocation_index) local: u32) {
    let upper_count = select(0u, lbvh_meta[8u], setup.t_count > 1u);
    let num_internal = select(0u, setup.t_count - 1u, setup.t_count > 1u);
    for (var k = local; k < upper_count; k += 256u) {
        ready[list_nodes[k]] = 0u;
        pending[k] = 0u;
    }
    if local == 0u {
        wg_done = select(0u, 1u, upper_count == 0u);
    }
    workgroupBarrier();
    // Maximum height of a 30-bit Morton + 32-bit index Karras radix tree is <= 62.
    for (var iter = 0u; iter < 96u; iter += 1u) {
        if workgroupUniformLoad(&wg_done) != 0u {
            break;
        }
        for (var k = local; k < upper_count; k += 256u) {
            let n = list_nodes[k];
            if ready[n] == 0u {
                let nb = n * 8u;
                let lc = bvh_nodes[nb + 3u];
                let rc = bvh_nodes[nb + 7u] & 0x7FFFFFFFu;
                if child_is_ready(lc, num_internal) && child_is_ready(rc, num_internal) {
                    merge_node(n, lc, rc);
                    pending[k] = 1u;
                }
            }
        }
        workgroupBarrier();
        for (var k = local; k < upper_count; k += 256u) {
            if pending[k] != 0u {
                ready[list_nodes[k]] = 1u;
                pending[k] = 0u;
            }
        }
        workgroupBarrier();
        if local == 0u {
            wg_done = ready[0u];
        }
        workgroupBarrier();
    }
}
";

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Compiled compute pipelines for Full-GPU LBVH construction and in-place refit.
pub(crate) struct LbvhPipelines {
    bounds_layout: wgpu::BindGroupLayout,
    clear_meta: wgpu::ComputePipeline,
    prim_bounds: wgpu::ComputePipeline,
    morton_layout: wgpu::BindGroupLayout,
    morton: wgpu::ComputePipeline,
    radix_layout: wgpu::BindGroupLayout,
    radix_hist: wgpu::ComputePipeline,
    radix_scan_blocks: wgpu::ComputePipeline,
    radix_scan_buckets: wgpu::ComputePipeline,
    radix_scatter: wgpu::ComputePipeline,
    hierarchy_layout: wgpu::BindGroupLayout,
    hierarchy: wgpu::ComputePipeline,
    tiers_layout: wgpu::BindGroupLayout,
    classify_tiers: wgpu::ComputePipeline,
    finalize_tier_args: wgpu::ComputePipeline,
    refit_leaves_layout: wgpu::BindGroupLayout,
    refit_leaves: wgpu::ComputePipeline,
    refit_tree_layout: wgpu::BindGroupLayout,
    refit_tier1: wgpu::ComputePipeline,
    refit_upper: wgpu::ComputePipeline,
}

impl LbvhPipelines {
    pub(crate) fn build(device: &wgpu::Device) -> Self {
        let make = |label: &str,
                    body: &str,
                    entries: &[wgpu::BindGroupLayoutEntry],
                    entry_points: &[&str]| {
            let src = format!("{LBVH_COMMON_WGSL}\n{body}");
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipes: Vec<wgpu::ComputePipeline> = entry_points
                .iter()
                .map(|ep| {
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(ep),
                        layout: Some(&pl),
                        module: &module,
                        entry_point: Some(ep),
                        compilation_options: Default::default(),
                        cache: None,
                    })
                })
                .collect();
            (layout, pipes)
        };

        let (bounds_layout, mut bounds_p) = make(
            "entelechy lbvh bounds",
            LBVH_BOUNDS_WGSL,
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
            ],
            &["clear_meta", "prim_bounds"],
        );
        let prim_bounds = bounds_p.pop().expect("prim_bounds");
        let clear_meta = bounds_p.pop().expect("clear_meta");

        let (morton_layout, mut morton_p) = make(
            "entelechy lbvh morton",
            LBVH_MORTON_WGSL,
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
            ],
            &["morton"],
        );
        let morton = morton_p.pop().expect("morton");

        let (radix_layout, mut radix_p) = make(
            "entelechy lbvh radix",
            LBVH_RADIX_WGSL,
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
            ],
            &[
                "radix_hist",
                "radix_scan_blocks",
                "radix_scan_buckets",
                "radix_scatter",
            ],
        );
        let radix_scatter = radix_p.pop().expect("radix_scatter");
        let radix_scan_buckets = radix_p.pop().expect("radix_scan_buckets");
        let radix_scan_blocks = radix_p.pop().expect("radix_scan_blocks");
        let radix_hist = radix_p.pop().expect("radix_hist");

        let (hierarchy_layout, mut hier_p) = make(
            "entelechy lbvh hierarchy",
            LBVH_HIERARCHY_WGSL,
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
            ],
            &["hierarchy"],
        );
        let hierarchy = hier_p.pop().expect("hierarchy");

        let (tiers_layout, mut tiers_p) = make(
            "entelechy lbvh tiers",
            LBVH_TIERS_WGSL,
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
            ],
            &["classify_tiers", "finalize_tier_args"],
        );
        let finalize_tier_args = tiers_p.pop().expect("finalize_tier_args");
        let classify_tiers = tiers_p.pop().expect("classify_tiers");

        let (refit_leaves_layout, mut rl_p) = make(
            "entelechy lbvh refit leaves",
            LBVH_REFIT_LEAVES_WGSL,
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
            ],
            &["refit_leaves"],
        );
        let refit_leaves = rl_p.pop().expect("refit_leaves");

        let (refit_tree_layout, mut rt_p) = make(
            "entelechy lbvh refit tree",
            LBVH_REFIT_TREE_WGSL,
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, true),
            ],
            &["refit_tier1", "refit_upper"],
        );
        let refit_upper = rt_p.pop().expect("refit_upper");
        let refit_tier1 = rt_p.pop().expect("refit_tier1");

        Self {
            bounds_layout,
            clear_meta,
            prim_bounds,
            morton_layout,
            morton,
            radix_layout,
            radix_hist,
            radix_scan_blocks,
            radix_scan_buckets,
            radix_scatter,
            hierarchy_layout,
            hierarchy,
            tiers_layout,
            classify_tiers,
            finalize_tier_args,
            refit_leaves_layout,
            refit_leaves,
            refit_tree_layout,
            refit_tier1,
            refit_upper,
        }
    }
}

/// Persistent GPU-resident LBVH topology and node buffers for one triangle mesh.
/// Supports both full GPU construction and zero-allocation 1-pass dynamic AABB refit.
#[derive(Clone)]
pub(crate) struct GpuLbvhMesh {
    pub(crate) nodes: wgpu::Buffer,
    pub(crate) ordered: wgpu::Buffer,
    pub(crate) mesh_desc: wgpu::Buffer,
    pub(crate) t_count: u32,
    // Persistent topology buffers reused across dynamic refit frames:
    sub_sizes: wgpu::Buffer,
    tier1_roots: wgpu::Buffer,
    upper_nodes: wgpu::Buffer,
    ready: wgpu::Buffer,
    pending: wgpu::Buffer,
    meta: wgpu::Buffer,
    indirect_args: wgpu::Buffer,
    pub(crate) built_verts_id: u64,
    pub(crate) refits_since_build: u32,
    pub(crate) last_eval: u64,
}

fn alloc_storage(
    device: &wgpu::Device,
    label: &str,
    words: u64,
    extra: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let size = (words.max(4) * 4).div_ceil(16) * 16;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | extra,
        mapped_at_creation: false,
    })
}

pub(crate) fn encode_setup(
    v_count: u32,
    t_count: u32,
    num_blocks: u32,
    blocks_x: u32,
    shift: u32,
) -> [u8; 32] {
    let words = [v_count, t_count, num_blocks, blocks_x, shift, 0, 0, 0];
    let mut bytes = [0u8; 32];
    for (slot, w) in bytes.chunks_exact_mut(4).zip(words) {
        slot.copy_from_slice(&w.to_le_bytes());
    }
    bytes
}

/// Returns the 8 uniform payloads (`shift = 0, 4, ..., 28`) required by
/// [`GpuLbvhMesh::encode_full_build`] so callers can allocate them atomically
/// from the batch uniform arena before opening the command encoder.
pub(crate) fn full_build_uniform_payloads(v_count: u32, t_count: u32) -> [[u8; 32]; 8] {
    let num_blocks = t_count.div_ceil(RADIX_BLOCK).max(1);
    let blocks_x = num_blocks.clamp(1, MOST_GROUPS);
    std::array::from_fn(|p| encode_setup(v_count, t_count, num_blocks, blocks_x, (p as u32) * 4))
}

/// Returns the single uniform payload required by [`GpuLbvhMesh::encode_refit`].
pub(crate) fn refit_uniform_payload(v_count: u32, t_count: u32) -> [u8; 32] {
    encode_setup(v_count, t_count, 1, 1, 0)
}

impl GpuLbvhMesh {
    /// Allocates persistent GPU LBVH buffers sized for `t_count` triangles.
    pub(crate) fn allocate(device: &wgpu::Device, t_count: u32, eval_now: u64) -> Self {
        let t = u64::from(t_count.max(1));
        let node_cap = if t_count <= 1 { 1 } else { 2 * t - 1 };
        Self {
            nodes: alloc_storage(
                device,
                "entelechy lbvh nodes",
                node_cap * 8,
                wgpu::BufferUsages::empty(),
            ),
            ordered: alloc_storage(
                device,
                "entelechy lbvh ordered",
                t,
                wgpu::BufferUsages::empty(),
            ),
            mesh_desc: alloc_storage(
                device,
                "entelechy lbvh desc",
                8,
                wgpu::BufferUsages::empty(),
            ),
            t_count,
            sub_sizes: alloc_storage(
                device,
                "entelechy lbvh sub_sizes",
                t,
                wgpu::BufferUsages::empty(),
            ),
            tier1_roots: alloc_storage(
                device,
                "entelechy lbvh tier1",
                t,
                wgpu::BufferUsages::empty(),
            ),
            upper_nodes: alloc_storage(
                device,
                "entelechy lbvh upper",
                t,
                wgpu::BufferUsages::empty(),
            ),
            ready: alloc_storage(
                device,
                "entelechy lbvh ready",
                t,
                wgpu::BufferUsages::empty(),
            ),
            pending: alloc_storage(
                device,
                "entelechy lbvh pending",
                t,
                wgpu::BufferUsages::empty(),
            ),
            meta: alloc_storage(
                device,
                "entelechy lbvh meta",
                16,
                wgpu::BufferUsages::empty(),
            ),
            indirect_args: alloc_storage(
                device,
                "entelechy lbvh indirect",
                8,
                wgpu::BufferUsages::INDIRECT,
            ),
            built_verts_id: 0,
            refits_since_build: 0,
            last_eval: eval_now,
        }
    }

    /// Encodes a complete zero-readback GPU LBVH build (`bounds -> morton -> 8-pass radix sort ->
    /// Karras hierarchy -> tier classification -> 2-tier refit`) into `encoder`.
    /// `shift_uniforms` must hold the 8 slices corresponding to [`full_build_uniform_payloads`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_full_build(
        &mut self,
        device: &wgpu::Device,
        spare: &crate::pool::Spare,
        pipelines: &LbvhPipelines,
        encoder: &mut wgpu::CommandEncoder,
        verts_buf: &wgpu::Buffer,
        tris_buf: &wgpu::Buffer,
        shift_uniforms: &[crate::batch::UniformSlice],
        verts_id: u64,
        eval_now: u64,
        keep_buffers: &mut Vec<wgpu::Buffer>,
        timing_set: Option<&crate::timing::Timing>,
        opened_ts: &mut bool,
    ) {
        let t_count = self.t_count;
        let num_blocks = t_count.div_ceil(RADIX_BLOCK).max(1);
        let blocks_x = num_blocks.clamp(1, MOST_GROUPS);
        let blocks_y = num_blocks.div_ceil(blocks_x).max(1);

        let scratch = |words: u64| {
            spare.take(
                device,
                (words.max(4) * 4).div_ceil(16) * 16,
                wgpu::BufferUsages::STORAGE,
            )
        };
        let t = u64::from(t_count.max(1));
        let centroids = scratch(t * 4);
        let keys_a = scratch(t);
        let keys_b = scratch(t);
        let vals_b = scratch(t);
        let block_hist = scratch(u64::from(num_blocks) * 16);
        let bucket_buf = scratch(32);
        let parents = scratch(2 * t);

        let base_u = &shift_uniforms[0];

        let bounds_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entelechy lbvh bounds"),
            layout: &pipelines.bounds_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: base_u.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: verts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tris_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: centroids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.meta.as_entire_binding(),
                },
            ],
        });
        let morton_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entelechy lbvh morton"),
            layout: &pipelines.morton_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: base_u.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: centroids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: keys_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.ordered.as_entire_binding(),
                },
            ],
        });
        let mut radix_bgs = Vec::with_capacity(RADIX_PASSES as usize);
        for (p, su) in shift_uniforms.iter().enumerate() {
            let even = p % 2 == 0;
            let (kin, vin, kout, vout) = if even {
                (&keys_a, &self.ordered, &keys_b, &vals_b)
            } else {
                (&keys_b, &vals_b, &keys_a, &self.ordered)
            };
            radix_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("entelechy lbvh radix"),
                layout: &pipelines.radix_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: su.binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: kin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: vin.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: kout.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: vout.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: block_hist.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: bucket_buf.as_entire_binding(),
                    },
                ],
            }));
        }
        let hier_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entelechy lbvh hierarchy"),
            layout: &pipelines.hierarchy_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: base_u.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: keys_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: parents.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.sub_sizes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.mesh_desc.as_entire_binding(),
                },
            ],
        });
        let tiers_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entelechy lbvh tiers"),
            layout: &pipelines.tiers_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: base_u.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: parents.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.sub_sizes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.tier1_roots.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.upper_nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.indirect_args.as_entire_binding(),
                },
            ],
        });

        let (tx, ty) = wgsl::folded(t_count.div_ceil(WORKGROUP_64).max(1));
        {
            let open_here = !*opened_ts;
            if open_here && timing_set.is_some() {
                *opened_ts = true;
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("entelechy lbvh build"),
                timestamp_writes: open_here
                    .then_some(())
                    .and(timing_set)
                    .map(|t| t.compute(Some(0), None)),
            });
            pass.set_bind_group(0, &bounds_bg, &[]);
            pass.set_pipeline(&pipelines.clear_meta);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&pipelines.prim_bounds);
            pass.dispatch_workgroups(tx, ty, 1);

            pass.set_bind_group(0, &morton_bg, &[]);
            pass.set_pipeline(&pipelines.morton);
            pass.dispatch_workgroups(tx, ty, 1);

            if t_count > 1 {
                for bg in &radix_bgs {
                    pass.set_bind_group(0, bg, &[]);
                    pass.set_pipeline(&pipelines.radix_hist);
                    pass.dispatch_workgroups(blocks_x, blocks_y, 1);
                    pass.set_pipeline(&pipelines.radix_scan_blocks);
                    pass.dispatch_workgroups(16, 1, 1);
                    pass.set_pipeline(&pipelines.radix_scan_buckets);
                    pass.dispatch_workgroups(1, 1, 1);
                    pass.set_pipeline(&pipelines.radix_scatter);
                    pass.dispatch_workgroups(blocks_x, blocks_y, 1);
                }
            }

            pass.set_bind_group(0, &hier_bg, &[]);
            pass.set_pipeline(&pipelines.hierarchy);
            pass.dispatch_workgroups(tx, ty, 1);

            pass.set_bind_group(0, &tiers_bg, &[]);
            pass.set_pipeline(&pipelines.classify_tiers);
            pass.dispatch_workgroups(tx, ty, 1);
            pass.set_pipeline(&pipelines.finalize_tier_args);
            pass.dispatch_workgroups(1, 1, 1);
        }

        self.encode_refit_with_uniform(
            device, pipelines, encoder, verts_buf, tris_buf, base_u, verts_id, eval_now, None,
            opened_ts,
        );
        self.refits_since_build = 0;

        keep_buffers.extend([
            centroids, keys_a, keys_b, vals_b, block_hist, bucket_buf, parents,
        ]);
    }

    /// Encodes a fast zero-allocation 1-pass dynamic AABB refit over existing LBVH topology.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_refit(
        &mut self,
        device: &wgpu::Device,
        pipelines: &LbvhPipelines,
        encoder: &mut wgpu::CommandEncoder,
        verts_buf: &wgpu::Buffer,
        tris_buf: &wgpu::Buffer,
        base_u: &crate::batch::UniformSlice,
        verts_id: u64,
        eval_now: u64,
        timing_set: Option<&crate::timing::Timing>,
        opened_ts: &mut bool,
    ) {
        self.encode_refit_with_uniform(
            device, pipelines, encoder, verts_buf, tris_buf, base_u, verts_id, eval_now,
            timing_set, opened_ts,
        );
        self.refits_since_build = self.refits_since_build.saturating_add(1);
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_refit_with_uniform(
        &mut self,
        device: &wgpu::Device,
        pipelines: &LbvhPipelines,
        encoder: &mut wgpu::CommandEncoder,
        verts_buf: &wgpu::Buffer,
        tris_buf: &wgpu::Buffer,
        base_u: &crate::batch::UniformSlice,
        verts_id: u64,
        eval_now: u64,
        timing_set: Option<&crate::timing::Timing>,
        opened_ts: &mut bool,
    ) {
        let leaves_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entelechy lbvh refit leaves"),
            layout: &pipelines.refit_leaves_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: base_u.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: verts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tris_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.ordered.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.nodes.as_entire_binding(),
                },
            ],
        });
        let tier1_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entelechy lbvh refit tier1"),
            layout: &pipelines.refit_tree_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: base_u.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.tier1_roots.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.sub_sizes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.ready.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.pending.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.meta.as_entire_binding(),
                },
            ],
        });
        let upper_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entelechy lbvh refit upper"),
            layout: &pipelines.refit_tree_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: base_u.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.upper_nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.sub_sizes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.ready.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.pending.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.meta.as_entire_binding(),
                },
            ],
        });
        let (tx, ty) = wgsl::folded(self.t_count.div_ceil(WORKGROUP_64).max(1));
        {
            let open_here = !*opened_ts;
            if open_here && timing_set.is_some() {
                *opened_ts = true;
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("entelechy lbvh refit"),
                timestamp_writes: open_here
                    .then_some(())
                    .and(timing_set)
                    .map(|t| t.compute(Some(0), None)),
            });
            pass.set_pipeline(&pipelines.refit_leaves);
            pass.set_bind_group(0, &leaves_bg, &[]);
            pass.dispatch_workgroups(tx, ty, 1);

            if self.t_count > 1 {
                pass.set_pipeline(&pipelines.refit_tier1);
                pass.set_bind_group(0, &tier1_bg, &[]);
                pass.dispatch_workgroups_indirect(&self.indirect_args, 0);

                pass.set_pipeline(&pipelines.refit_upper);
                pass.set_bind_group(0, &upper_bg, &[]);
                pass.dispatch_workgroups_indirect(&self.indirect_args, 12);
            }
        }
        self.built_verts_id = verts_id;
        self.last_eval = eval_now;
    }
}
