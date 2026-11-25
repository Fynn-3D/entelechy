//! GPU execution of native ray tracing (`native.trace@1` and `native.trace@2`).
//!
//! Two paths share one contract ([`entelechy_rules::trace`]):
//! 1. **Hardware RT Core** (`#[cfg(not(target_arch = "wasm32"))]`, when
//!    `wgpu::Features::EXPERIMENTAL_RAY_QUERY` is enabled): builds and caches
//!    hardware BLASes (sanitizing vertices/indices on GPU with zero CPU
//!    readback for single meshes) and TLASes keyed by immutable `DeviceImage`
//!    IDs, then runs an `enable wgpu_ray_query;` compute pass.
//! 2. **GPU Compute BVH Fallback** (WebGPU, non-RT adapters, or
//!    `GpuBackend::without_ray_query()`): builds and caches flat 32-byte
//!    SAH-binned BVHs (per mesh plus a world-space TLAS BVH for instanced
//!    scenes) on first encounter of immutable geometry IDs, then traverses them
//!    in a portable WGSL compute shader with zero ray-query extensions.

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};

use entelechy_abi::{Domain, Handle, Invocation, Scope};
use entelechy_rules::trace::{
    self, BuiltBvh, HIT_WORDS, POSITION_WORDS, TRIANGLE_WORDS, check_trace, check_trace_instances,
    hit_schema,
};
#[cfg(not(target_arch = "wasm32"))]
use entelechy_rules::trace::{INSTANCE_WORDS, MeshSlice};
use entelechy_rules::{ExecutionError, NativeOp};

use crate::batch::UniformSlice;
use crate::failure::Scopes;
use crate::trace_lbvh::{self, GpuLbvhMesh, LBVH_REFIT_PERIOD, LbvhPipelines};
use crate::{DeviceImage, WgpuBackend, wgsl};

const TRACE_WORKGROUP: u32 = 64;
const COMPACT_RAY_THRESHOLD: u32 = 256;
const _: () = assert!(wgsl::MAX_GROUPS * TRACE_WORKGROUP == 4_194_240);
const CACHE_MAX_ENTRIES: usize = 32;
const CACHE_MAX_AGE: u64 = 16;

/// GPU-driven active-ray stream compaction shader shared by both Hardware RT
/// and Compute BVH paths: writes MISS for dead/invalid rays (`tmax <= tmin` or
/// non-finite) and packs live ray indices into `active_rays[1..]` (with count
/// in `active_rays[0]`) plus `[gx, gy, 1]` into `indirect_args` for
/// `dispatch_workgroups_indirect`.
const COMPACT_RAYS_WGSL: &str = "
struct CompactSetup {
    ray_count: u32,
    flags: u32,
    pad0: u32,
    pad1: u32,
};
@group(0) @binding(0) var<uniform> setup: CompactSetup;
@group(0) @binding(1) var<storage, read> rays: array<u32>;
@group(0) @binding(2) var<storage, read_write> hits: array<u32>;
// active_rays[0] = active_count, active_rays[1 + slot] = original ray index r
@group(0) @binding(3) var<storage, read_write> active_rays: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> indirect_args: array<u32>;

fn is_finite_bits(w: u32) -> bool {
    return (w & 0x7F800000u) != 0x7F800000u;
}

fn is_nan_bits(w: u32) -> bool {
    return (w & 0x7FFFFFFFu) > 0x7F800000u;
}

fn write_miss(r: u32) {
    let b = r * 7u;
    hits[b] = 0xBF800000u;
    hits[b + 1u] = 0u;
    hits[b + 2u] = 0u;
    hits[b + 3u] = 0xFFFFFFFFu;
    hits[b + 4u] = 0xFFFFFFFFu;
    hits[b + 5u] = 0u;
    hits[b + 6u] = 0u;
}

@compute @workgroup_size(1, 1, 1)
fn clear_args() {
    atomicStore(&active_rays[0u], 0u);
}

var<workgroup> wg_scan: array<u32, 64>;
var<workgroup> wg_base: u32;

@compute @workgroup_size(64, 1, 1)
fn compact_rays(
    @builtin(local_invocation_index) local: u32,
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let r = gid.x + gid.y * (nwg.x * 64u);
    var alive = 0u;
    if r < setup.ray_count {
        let cull_both = (setup.flags & 6u) == 6u;
        let rb = r * 8u;
        let ox_w = rays[rb];
        let oy_w = rays[rb + 1u];
        let oz_w = rays[rb + 2u];
        let tmin_w = rays[rb + 3u];
        let dx_w = rays[rb + 4u];
        let dy_w = rays[rb + 5u];
        let dz_w = rays[rb + 6u];
        let tmax_w = rays[rb + 7u];
        let valid_f = !cull_both
            && is_finite_bits(ox_w) && is_finite_bits(oy_w) && is_finite_bits(oz_w)
            && is_finite_bits(dx_w) && is_finite_bits(dy_w) && is_finite_bits(dz_w)
            && is_finite_bits(tmin_w) && !is_nan_bits(tmax_w);
        if valid_f {
            let tmin = bitcast<f32>(tmin_w);
            let tmax = bitcast<f32>(tmax_w);
            if tmin >= 0.0 && tmax > tmin {
                alive = 1u;
            }
        }
        if alive == 0u {
            write_miss(r);
        }
    }
    wg_scan[local] = alive;
    workgroupBarrier();
    for (var d = 1u; d < 64u; d <<= 1u) {
        var add = 0u;
        if local >= d {
            add = wg_scan[local - d];
        }
        workgroupBarrier();
        wg_scan[local] += add;
        workgroupBarrier();
    }
    if local == 63u {
        let total = wg_scan[63u];
        if total > 0u {
            wg_base = atomicAdd(&active_rays[0u], total);
        } else {
            wg_base = 0u;
        }
    }
    workgroupBarrier();
    if alive != 0u {
        let rank = wg_scan[local] - 1u;
        atomicStore(&active_rays[1u + wg_base + rank], r);
    }
}

@compute @workgroup_size(1, 1, 1)
fn finalize_args() {
    let active_cnt = atomicLoad(&active_rays[0u]);
    let groups = (active_cnt + 63u) / 64u;
    let gx = min(groups, 65535u);
    let safe_gx = max(gx, 1u);
    let gy = select(1u, (groups + safe_gx - 1u) / safe_gx, groups > 0u);
    indirect_args[0u] = gx;
    indirect_args[1u] = gy;
    indirect_args[2u] = 1u;
}
";

/// Clear-to-miss shader used whenever a scene has zero triangles or zero active
/// instances so no empty acceleration structure is ever submitted to the device.
const CLEAR_MISS_WGSL: &str = "
struct ClearSetup {
    ray_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};
@group(0) @binding(0) var<uniform> setup: ClearSetup;
@group(0) @binding(1) var<storage, read_write> hits: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let r = gid.x + gid.y * (nwg.x * 64u);
    if r >= setup.ray_count {
        return;
    }
    let b = r * 7u;
    hits[b] = 0xBF800000u; // -1.0f32
    hits[b + 1u] = 0u;
    hits[b + 2u] = 0u;
    hits[b + 3u] = 0xFFFFFFFFu;
    hits[b + 4u] = 0xFFFFFFFFu;
    hits[b + 5u] = 0u;
    hits[b + 6u] = 0u;
}
";

/// GPU compute pass that copies/sanitizes triangle vertex positions and clamped
/// non-degenerate triangle indices into `BLAS_INPUT` staging buffers so
/// hardware BLAS construction never reads out of bounds or builds NaN bounds.
#[cfg(not(target_arch = "wasm32"))]
const SANITIZE_BLAS_WGSL: &str = "
struct SanitizeSetup {
    vertex_offset: u32,
    vertex_count: u32,
    triangle_offset: u32,
    triangle_count: u32,
};
@group(0) @binding(0) var<uniform> setup: SanitizeSetup;
@group(0) @binding(1) var<storage, read> src_verts: array<u32>;
@group(0) @binding(2) var<storage, read> src_tris: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst_verts: array<u32>;
@group(0) @binding(4) var<storage, read_write> dst_tris: array<u32>;

fn is_finite_bits(w: u32) -> bool {
    return (w & 0x7F800000u) != 0x7F800000u;
}

fn finite_vertex(local_v: u32) -> bool {
    let base = (setup.vertex_offset + local_v) * 3u;
    return is_finite_bits(src_verts[base])
        && is_finite_bits(src_verts[base + 1u])
        && is_finite_bits(src_verts[base + 2u]);
}

fn load_vertex(local_v: u32) -> vec3<f32> {
    let base = (setup.vertex_offset + local_v) * 3u;
    return vec3<f32>(
        bitcast<f32>(src_verts[base]),
        bitcast<f32>(src_verts[base + 1u]),
        bitcast<f32>(src_verts[base + 2u]),
    );
}

@compute @workgroup_size(64, 1, 1)
fn sanitize_verts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x + gid.y * 4194240u;
    if v >= setup.vertex_count {
        return;
    }
    let sb = (setup.vertex_offset + v) * 3u;
    let db = v * 3u;
    if finite_vertex(v) {
        dst_verts[db] = src_verts[sb];
        dst_verts[db + 1u] = src_verts[sb + 1u];
        dst_verts[db + 2u] = src_verts[sb + 2u];
    } else {
        dst_verts[db] = 0u;
        dst_verts[db + 1u] = 0u;
        dst_verts[db + 2u] = 0u;
    }
}

@compute @workgroup_size(64, 1, 1)
fn sanitize_tris(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x + gid.y * 4194240u;
    if t >= setup.triangle_count {
        return;
    }
    let sb = (setup.triangle_offset + t) * 3u;
    let db = t * 3u;
    let max_v = i32(setup.vertex_count) - 1i;
    let ia = u32(clamp(bitcast<i32>(src_tris[sb]), 0i, max_v));
    let ib = u32(clamp(bitcast<i32>(src_tris[sb + 1u]), 0i, max_v));
    let ic = u32(clamp(bitcast<i32>(src_tris[sb + 2u]), 0i, max_v));
    var ok = finite_vertex(ia) && finite_vertex(ib) && finite_vertex(ic);
    if ok {
        let a = load_vertex(ia);
        let b = load_vertex(ib);
        let c = load_vertex(ic);
        let n = cross(b - a, c - a);
        let len2 = dot(n, n);
        ok = len2 > 0.0 && is_finite_bits(bitcast<u32>(len2));
    }
    if ok {
        dst_tris[db] = ia;
        dst_tris[db + 1u] = ib;
        dst_tris[db + 2u] = ic;
    } else {
        // Degenerate triangle never intersects any ray.
        dst_tris[db] = 0u;
        dst_tris[db + 1u] = 0u;
        dst_tris[db + 2u] = 0u;
    }
}
";

/// Hardware RT Core compute shader (`enable wgpu_ray_query;`).
/// Computes exact CCW `front` orientation and optional face culling directly
/// against the hit triangle's geometric normal (scaled by instance determinant
/// sign) so Metal, Vulkan, DX12, WebGPU BVH, and CPU/JIT agree identically.
#[cfg(not(target_arch = "wasm32"))]
const HW_TRACE_WGSL: &str = "
enable wgpu_ray_query;

struct Setup {
    ray_count: u32,
    vertex_count: u32,
    triangle_count: u32,
    flags: u32,
    mask: u32,
    instanced: u32,
    mesh_count: u32,
    indirect: u32,
};

@group(0) @binding(0) var<uniform> setup: Setup;
@group(0) @binding(1) var tlas: acceleration_structure;
@group(0) @binding(2) var<storage, read> rays: array<u32>;
@group(0) @binding(3) var<storage, read_write> hits: array<u32>;
@group(0) @binding(4) var<storage, read> vertices: array<u32>;
@group(0) @binding(5) var<storage, read> triangles: array<u32>;
@group(0) @binding(6) var<storage, read> meshes: array<u32>;
@group(0) @binding(7) var<storage, read> instances: array<u32>;
@group(0) @binding(8) var<storage, read> active_rays: array<u32>;

fn is_finite_bits(w: u32) -> bool {
    return (w & 0x7F800000u) != 0x7F800000u;
}

fn is_nan_bits(w: u32) -> bool {
    return (w & 0x7FFFFFFFu) > 0x7F800000u;
}

fn write_miss(r: u32) {
    let b = r * 7u;
    hits[b] = 0xBF800000u;
    hits[b + 1u] = 0u;
    hits[b + 2u] = 0u;
    hits[b + 3u] = 0xFFFFFFFFu;
    hits[b + 4u] = 0xFFFFFFFFu;
    hits[b + 5u] = 0u;
    hits[b + 6u] = 0u;
}

fn load_vert(idx: u32) -> vec3<f32> {
    let b = idx * 3u;
    return vec3<f32>(
        bitcast<f32>(vertices[b]),
        bitcast<f32>(vertices[b + 1u]),
        bitcast<f32>(vertices[b + 2u]),
    );
}

fn hit_is_front(inst_row: u32, tri_local: u32, world_dir: vec3<f32>) -> bool {
    var v_off = 0u;
    var v_cnt = setup.vertex_count;
    var t_off = 0u;
    if setup.instanced != 0u {
        let ib = inst_row * 15u;
        let mesh_idx = bitcast<u32>(instances[ib + 12u]);
        let mb = mesh_idx * 4u;
        v_off = meshes[mb];
        v_cnt = meshes[mb + 1u];
        t_off = meshes[mb + 2u];
    }
    let max_v = i32(v_cnt) - 1i;
    let tb = (t_off + tri_local) * 3u;
    let ia = v_off + u32(clamp(bitcast<i32>(triangles[tb]), 0i, max_v));
    let ib = v_off + u32(clamp(bitcast<i32>(triangles[tb + 1u]), 0i, max_v));
    let ic = v_off + u32(clamp(bitcast<i32>(triangles[tb + 2u]), 0i, max_v));
    let v0 = load_vert(ia);
    let v1 = load_vert(ib);
    let v2 = load_vert(ic);
    if setup.instanced == 0u {
        return dot(world_dir, cross(v1 - v0, v2 - v0)) < 0.0;
    }
    let inb = inst_row * 15u;
    let r0 = vec3<f32>(
        bitcast<f32>(instances[inb]),
        bitcast<f32>(instances[inb + 1u]),
        bitcast<f32>(instances[inb + 2u]),
    );
    let r1 = vec3<f32>(
        bitcast<f32>(instances[inb + 4u]),
        bitcast<f32>(instances[inb + 5u]),
        bitcast<f32>(instances[inb + 6u]),
    );
    let r2 = vec3<f32>(
        bitcast<f32>(instances[inb + 8u]),
        bitcast<f32>(instances[inb + 9u]),
        bitcast<f32>(instances[inb + 10u]),
    );
    let e1_obj = v1 - v0;
    let e2_obj = v2 - v0;
    let e1_w = vec3<f32>(dot(r0, e1_obj), dot(r1, e1_obj), dot(r2, e1_obj));
    let e2_w = vec3<f32>(dot(r0, e2_obj), dot(r1, e2_obj), dot(r2, e2_obj));
    return dot(world_dir, cross(e1_w, e2_w)) < 0.0;
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let slot = gid.x + gid.y * (nwg.x * 64u);
    var r = slot;
    if setup.indirect != 0u {
        let active_cnt = active_rays[0u];
        if slot >= active_cnt {
            return;
        }
        r = active_rays[1u + slot];
    } else if r >= setup.ray_count {
        return;
    }
    let rb = r * 8u;
    let ox_w = rays[rb];
    let oy_w = rays[rb + 1u];
    let oz_w = rays[rb + 2u];
    let tmin_w = rays[rb + 3u];
    let dx_w = rays[rb + 4u];
    let dy_w = rays[rb + 5u];
    let dz_w = rays[rb + 6u];
    let tmax_w = rays[rb + 7u];
    let valid_f = is_finite_bits(ox_w) && is_finite_bits(oy_w) && is_finite_bits(oz_w)
        && is_finite_bits(dx_w) && is_finite_bits(dy_w) && is_finite_bits(dz_w)
        && is_finite_bits(tmin_w) && !is_nan_bits(tmax_w);
    if !valid_f {
        write_miss(r);
        return;
    }
    let origin = vec3<f32>(bitcast<f32>(ox_w), bitcast<f32>(oy_w), bitcast<f32>(oz_w));
    let tmin = bitcast<f32>(tmin_w);
    let dir = vec3<f32>(bitcast<f32>(dx_w), bitcast<f32>(dy_w), bitcast<f32>(dz_w));
    let tmax = bitcast<f32>(tmax_w);
    if !(tmin >= 0.0 && tmax > tmin) {
        write_miss(r);
        return;
    }
    let tmax_clamped = min(tmax, 3.4e38);
    let any_hit = (setup.flags & 1u) != 0u;
    let cull_back = (setup.flags & 2u) != 0u;
    let cull_front = (setup.flags & 4u) != 0u;
    if cull_back && cull_front {
        write_miss(r);
        return;
    }
    var rq: ray_query;
    // NOTE: Never name the RayDesc local `desc`: Naga 30's MSL writer hardcodes
    // `RayDesc desc = <arg>;`, so a local named `desc` self-shadows as uninitialized!
    if cull_back || cull_front {
        let ray_desc = RayDesc(
            RAY_FLAG_FORCE_NO_OPAQUE,
            setup.mask,
            tmin,
            tmax_clamped,
            origin,
            dir,
        );
        rayQueryInitialize(&rq, tlas, ray_desc);
        while rayQueryProceed(&rq) {
            let cand = rayQueryGetCandidateIntersection(&rq);
            if cand.kind == RAY_QUERY_INTERSECTION_TRIANGLE {
                let inst_row = select(0u, cand.instance_custom_data, setup.instanced != 0u);
                let front = hit_is_front(inst_row, cand.primitive_index, dir);
                let keep = select(front, !front, cull_front);
                if keep {
                    rayQueryConfirmIntersection(&rq);
                    if any_hit {
                        rayQueryTerminate(&rq);
                    }
                }
            }
        }
    } else {
        var rq_flags = RAY_FLAG_FORCE_OPAQUE;
        if any_hit {
            rq_flags = rq_flags | RAY_FLAG_TERMINATE_ON_FIRST_HIT;
        }
        let ray_desc = RayDesc(
            rq_flags,
            setup.mask,
            tmin,
            tmax_clamped,
            origin,
            dir,
        );
        rayQueryInitialize(&rq, tlas, ray_desc);
        rayQueryProceed(&rq);
    }
    let hit = rayQueryGetCommittedIntersection(&rq);
    if hit.kind != RAY_QUERY_INTERSECTION_TRIANGLE {
        write_miss(r);
        return;
    }
    let tri = hit.primitive_index;
    let inst_row = select(0u, hit.instance_custom_data, setup.instanced != 0u);
    let custom = select(0u, instances[inst_row * 15u + 13u], setup.instanced != 0u);
    let front = hit_is_front(inst_row, tri, dir);
    let b = r * 7u;
    hits[b] = bitcast<u32>(hit.t);
    hits[b + 1u] = bitcast<u32>(hit.barycentrics.x);
    hits[b + 2u] = bitcast<u32>(hit.barycentrics.y);
    hits[b + 3u] = tri;
    hits[b + 4u] = inst_row;
    hits[b + 5u] = custom;
    hits[b + 6u] = u32(front);
}
";

/// Portable GPU Compute BVH ray-tracing shader (WebGPU / non-RT adapters /
/// `without_ray_query()`), traversing the exact flat 32-byte SAH BVHs produced
/// by [`entelechy_rules::trace`].
const BVH_TRACE_WGSL: &str = "
struct Setup {
    ray_count: u32,
    vertex_count: u32,
    triangle_count: u32,
    flags: u32,
    mask: u32,
    instanced: u32,
    tlas_nodes: u32,
    indirect: u32,
};

// Packed per-mesh descriptor for GPU Compute BVH:
// [v_off, v_cnt, t_off, node_base, ord_base, node_count, 0, 0]
// Packed per-instance descriptor for GPU Compute BVH (24 words):
// [inv_row0 (4), inv_row1 (4), inv_row2 (4), world_min (3), row (1), world_max (3), mesh (1), custom (1), mask (1), 0, 0]
@group(0) @binding(0) var<uniform> setup: Setup;
@group(0) @binding(1) var<storage, read> rays: array<u32>;
@group(0) @binding(2) var<storage, read_write> hits: array<u32>;
@group(0) @binding(3) var<storage, read> vertices: array<u32>;
@group(0) @binding(4) var<storage, read> triangles: array<u32>;
@group(0) @binding(5) var<storage, read> bvh_nodes: array<u32>;
@group(0) @binding(6) var<storage, read> bvh_ordered: array<u32>;
@group(0) @binding(7) var<storage, read> mesh_descs: array<u32>;
@group(0) @binding(8) var<storage, read> tlas_nodes: array<u32>;
@group(0) @binding(9) var<storage, read> tlas_ordered: array<u32>;
@group(0) @binding(10) var<storage, read> inst_descs: array<u32>;

fn is_finite_bits(w: u32) -> bool {
    return (w & 0x7F800000u) != 0x7F800000u;
}

fn is_nan_bits(w: u32) -> bool {
    return (w & 0x7FFFFFFFu) > 0x7F800000u;
}

fn write_miss(r: u32) {
    let b = r * 7u;
    hits[b] = 0xBF800000u;
    hits[b + 1u] = 0u;
    hits[b + 2u] = 0u;
    hits[b + 3u] = 0xFFFFFFFFu;
    hits[b + 4u] = 0xFFFFFFFFu;
    hits[b + 5u] = 0u;
    hits[b + 6u] = 0u;
}

fn inv_dir_of(d: vec3<f32>) -> vec3<f32> {
    let ix = select(1.0 / d.x, select(-1e30, 1e30, d.x >= 0.0), d.x == 0.0);
    let iy = select(1.0 / d.y, select(-1e30, 1e30, d.y >= 0.0), d.y == 0.0);
    let iz = select(1.0 / d.z, select(-1e30, 1e30, d.z >= 0.0), d.z == 0.0);
    return vec3<f32>(ix, iy, iz);
}

fn slab_hit(origin: vec3<f32>, inv_d: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>, tmin: f32, tmax: f32) -> vec2<f32> {
    if bmin.x > bmax.x {
        return vec2<f32>(0.0, 0.0);
    }
    let t0 = (bmin - origin) * inv_d;
    let t1 = (bmax - origin) * inv_d;
    let lo = min(t0, t1);
    let hi = max(t0, t1);
    let near = max(max(max(lo.x, lo.y), lo.z), tmin);
    let far = min(min(min(hi.x, hi.y), hi.z), tmax);
    return vec2<f32>(near, select(0.0, 1.0, far >= near));
}

fn load_vert(idx: u32) -> vec3<f32> {
    let b = idx * 3u;
    return vec3<f32>(
        bitcast<f32>(vertices[b]),
        bitcast<f32>(vertices[b + 1u]),
        bitcast<f32>(vertices[b + 2u]),
    );
}

struct Best {
    t: f32,
    u: f32,
    v: f32,
    tri: i32,
    inst: i32,
    custom: u32,
    front: u32,
};

fn traverse_blas(
    node_base: u32,
    ord_base: u32,
    node_count: u32,
    v_off: u32,
    v_cnt: u32,
    t_off: u32,
    origin: vec3<f32>,
    dir: vec3<f32>,
    tmin: f32,
    inst_row: i32,
    custom: u32,
    flip_winding: bool,
    best_in: Best,
) -> Best {
    var best = best_in;
    if node_count == 0u {
        return best;
    }
    let any_hit = (setup.flags & 1u) != 0u;
    let cull_back = (setup.flags & 2u) != 0u;
    let cull_front = (setup.flags & 4u) != 0u;
    let inv_d = inv_dir_of(dir);
    let root_b = node_base * 8u;
    let rmin = vec3<f32>(bitcast<f32>(bvh_nodes[root_b]), bitcast<f32>(bvh_nodes[root_b + 1u]), bitcast<f32>(bvh_nodes[root_b + 2u]));
    let rmax = vec3<f32>(bitcast<f32>(bvh_nodes[root_b + 4u]), bitcast<f32>(bvh_nodes[root_b + 5u]), bitcast<f32>(bvh_nodes[root_b + 6u]));
    if slab_hit(origin, inv_d, rmin, rmax, tmin, best.t).y < 0.5 {
        return best;
    }
    var stack: array<u32, 64>;
    stack[0] = 0u;
    var sp = 1u;
    let max_v = i32(v_cnt) - 1i;
    while sp > 0u {
        sp = sp - 1u;
        let nb = (node_base + stack[sp]) * 8u;
        let left_or_first = bvh_nodes[nb + 3u];
        let count = bvh_nodes[nb + 7u];
        if count > 0u && (count & 0x80000000u) == 0u {
            for (var k = 0u; k < count; k = k + 1u) {
                let local_t = bvh_ordered[ord_base + left_or_first + k];
                let tb = (t_off + local_t) * 3u;
                let ia = v_off + u32(clamp(bitcast<i32>(triangles[tb]), 0i, max_v));
                let ib = v_off + u32(clamp(bitcast<i32>(triangles[tb + 1u]), 0i, max_v));
                let ic = v_off + u32(clamp(bitcast<i32>(triangles[tb + 2u]), 0i, max_v));
                let v0 = load_vert(ia);
                let v1 = load_vert(ib);
                let v2 = load_vert(ic);
                let e1 = v1 - v0;
                let e2 = v2 - v0;
                let h = cross(dir, e2);
                let det = dot(e1, h);
                if det == 0.0 || !is_finite_bits(bitcast<u32>(det)) {
                    continue;
                }
                let front = (det > 0.0) != flip_winding;
                if (cull_back && !front) || (cull_front && front) {
                    continue;
                }
                let inv_det = 1.0 / det;
                let s = origin - v0;
                let u = dot(s, h) * inv_det;
                if !(u >= 0.0 && u <= 1.0) {
                    continue;
                }
                let q = cross(s, e1);
                let v = dot(dir, q) * inv_det;
                if !(v >= 0.0 && u + v <= 1.0) {
                    continue;
                }
                let t = dot(e2, q) * inv_det;
                if !(t >= tmin && t <= best.t) {
                    continue;
                }
                let tri_i = i32(local_t);
                let closer = best.tri < 0i
                    || t < best.t
                    || (t == best.t && (inst_row < best.inst || (inst_row == best.inst && tri_i < best.tri)));
                if closer {
                    best.t = t;
                    best.u = u;
                    best.v = v;
                    best.tri = tri_i;
                    best.inst = inst_row;
                    best.custom = custom;
                    best.front = u32(front);
                    if any_hit {
                        return best;
                    }
                }
            }
        } else {
            let l_idx = left_or_first;
            let r_idx = select(l_idx + 1u, count & 0x7FFFFFFFu, (count & 0x80000000u) != 0u);
            let lb = (node_base + l_idx) * 8u;
            let rb_n = (node_base + r_idx) * 8u;
            let lh = slab_hit(
                origin,
                inv_d,
                vec3<f32>(bitcast<f32>(bvh_nodes[lb]), bitcast<f32>(bvh_nodes[lb + 1u]), bitcast<f32>(bvh_nodes[lb + 2u])),
                vec3<f32>(bitcast<f32>(bvh_nodes[lb + 4u]), bitcast<f32>(bvh_nodes[lb + 5u]), bitcast<f32>(bvh_nodes[lb + 6u])),
                tmin,
                best.t,
            );
            let rh = slab_hit(
                origin,
                inv_d,
                vec3<f32>(bitcast<f32>(bvh_nodes[rb_n]), bitcast<f32>(bvh_nodes[rb_n + 1u]), bitcast<f32>(bvh_nodes[rb_n + 2u])),
                vec3<f32>(bitcast<f32>(bvh_nodes[rb_n + 4u]), bitcast<f32>(bvh_nodes[rb_n + 5u]), bitcast<f32>(bvh_nodes[rb_n + 6u])),
                tmin,
                best.t,
            );
            let hl = lh.y > 0.5;
            let hr = rh.y > 0.5;
            if hl && hr {
                if sp + 2u <= 64u {
                    if lh.x <= rh.x {
                        stack[sp] = r_idx;
                        stack[sp + 1u] = l_idx;
                    } else {
                        stack[sp] = l_idx;
                        stack[sp + 1u] = r_idx;
                    }
                    sp = sp + 2u;
                }
            } else if hl {
                if sp < 64u {
                    stack[sp] = l_idx;
                    sp = sp + 1u;
                }
            } else if hr {
                if sp < 64u {
                    stack[sp] = r_idx;
                    sp = sp + 1u;
                }
            }
        }
    }
    return best;
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let slot = gid.x + gid.y * (nwg.x * 64u);
    var r = slot;
    if setup.instanced == 0u && setup.indirect != 0u {
        let active_cnt = tlas_nodes[0u];
        if slot >= active_cnt {
            return;
        }
        r = tlas_nodes[1u + slot];
    } else if r >= setup.ray_count {
        return;
    }
    let rb = r * 8u;
    let ox_w = rays[rb];
    let oy_w = rays[rb + 1u];
    let oz_w = rays[rb + 2u];
    let tmin_w = rays[rb + 3u];
    let dx_w = rays[rb + 4u];
    let dy_w = rays[rb + 5u];
    let dz_w = rays[rb + 6u];
    let tmax_w = rays[rb + 7u];
    let valid_f = is_finite_bits(ox_w) && is_finite_bits(oy_w) && is_finite_bits(oz_w)
        && is_finite_bits(dx_w) && is_finite_bits(dy_w) && is_finite_bits(dz_w)
        && is_finite_bits(tmin_w) && !is_nan_bits(tmax_w);
    if !valid_f {
        write_miss(r);
        return;
    }
    let origin = vec3<f32>(bitcast<f32>(ox_w), bitcast<f32>(oy_w), bitcast<f32>(oz_w));
    let tmin = bitcast<f32>(tmin_w);
    let dir = vec3<f32>(bitcast<f32>(dx_w), bitcast<f32>(dy_w), bitcast<f32>(dz_w));
    let tmax = bitcast<f32>(tmax_w);
    if !(tmin >= 0.0 && tmax > tmin) {
        write_miss(r);
        return;
    }
    var best = Best(tmax, 0.0, 0.0, -1i, -1i, 0u, 0u);
    if setup.instanced == 0u {
        let node_count = mesh_descs[5u];
        best = traverse_blas(
            0u,
            0u,
            node_count,
            0u,
            setup.vertex_count,
            0u,
            origin,
            dir,
            tmin,
            0i,
            0u,
            false,
            best,
        );
    } else if setup.tlas_nodes > 0u && setup.mask != 0u {
        let any_hit = (setup.flags & 1u) != 0u;
        let inv_d = inv_dir_of(dir);
        let rmin = vec3<f32>(bitcast<f32>(tlas_nodes[0u]), bitcast<f32>(tlas_nodes[1u]), bitcast<f32>(tlas_nodes[2u]));
        let rmax = vec3<f32>(bitcast<f32>(tlas_nodes[4u]), bitcast<f32>(tlas_nodes[5u]), bitcast<f32>(tlas_nodes[6u]));
        if slab_hit(origin, inv_d, rmin, rmax, tmin, best.t).y > 0.5 {
            var tstack: array<u32, 64>;
            tstack[0] = 0u;
            var tsp = 1u;
            while tsp > 0u {
                tsp = tsp - 1u;
                let nb = tstack[tsp] * 8u;
                let left_or_first = tlas_nodes[nb + 3u];
                let count = tlas_nodes[nb + 7u];
                if count > 0u && (count & 0x80000000u) == 0u {
                    for (var k = 0u; k < count; k = k + 1u) {
                        let pidx = tlas_ordered[left_or_first + k];
                        let ib = pidx * 24u;
                        let imask = inst_descs[ib + 21u];
                        if (imask & setup.mask) == 0u {
                            continue;
                        }
                        let wmin = vec3<f32>(bitcast<f32>(inst_descs[ib + 12u]), bitcast<f32>(inst_descs[ib + 13u]), bitcast<f32>(inst_descs[ib + 14u]));
                        let wmax = vec3<f32>(bitcast<f32>(inst_descs[ib + 16u]), bitcast<f32>(inst_descs[ib + 17u]), bitcast<f32>(inst_descs[ib + 18u]));
                        if slab_hit(origin, inv_d, wmin, wmax, tmin, best.t).y < 0.5 {
                            continue;
                        }
                        let i0 = vec4<f32>(bitcast<f32>(inst_descs[ib]), bitcast<f32>(inst_descs[ib + 1u]), bitcast<f32>(inst_descs[ib + 2u]), bitcast<f32>(inst_descs[ib + 3u]));
                        let i1 = vec4<f32>(bitcast<f32>(inst_descs[ib + 4u]), bitcast<f32>(inst_descs[ib + 5u]), bitcast<f32>(inst_descs[ib + 6u]), bitcast<f32>(inst_descs[ib + 7u]));
                        let i2 = vec4<f32>(bitcast<f32>(inst_descs[ib + 8u]), bitcast<f32>(inst_descs[ib + 9u]), bitcast<f32>(inst_descs[ib + 10u]), bitcast<f32>(inst_descs[ib + 11u]));
                        let obj_o = vec3<f32>(dot(i0.xyz, origin) + i0.w, dot(i1.xyz, origin) + i1.w, dot(i2.xyz, origin) + i2.w);
                        let obj_d = vec3<f32>(dot(i0.xyz, dir), dot(i1.xyz, dir), dot(i2.xyz, dir));
                        let inst_row = i32(inst_descs[ib + 15u]);
                        let mesh_idx = inst_descs[ib + 19u];
                        let custom = inst_descs[ib + 20u];
                        let flip_winding = inst_descs[ib + 22u] != 0u;
                        let mb = mesh_idx * 8u;
                        best = traverse_blas(
                            mesh_descs[mb + 3u],
                            mesh_descs[mb + 4u],
                            mesh_descs[mb + 5u],
                            mesh_descs[mb],
                            mesh_descs[mb + 1u],
                            mesh_descs[mb + 2u],
                            obj_o,
                            obj_d,
                            tmin,
                            inst_row,
                            custom,
                            flip_winding,
                            best,
                        );
                        if any_hit && best.tri >= 0i {
                            tsp = 0u;
                            break;
                        }
                    }
                } else {
                    let l_idx = left_or_first;
                    let r_idx = select(l_idx + 1u, count & 0x7FFFFFFFu, (count & 0x80000000u) != 0u);
                    let lb = l_idx * 8u;
                    let rb_n = r_idx * 8u;
                    let lh = slab_hit(
                        origin,
                        inv_d,
                        vec3<f32>(bitcast<f32>(tlas_nodes[lb]), bitcast<f32>(tlas_nodes[lb + 1u]), bitcast<f32>(tlas_nodes[lb + 2u])),
                        vec3<f32>(bitcast<f32>(tlas_nodes[lb + 4u]), bitcast<f32>(tlas_nodes[lb + 5u]), bitcast<f32>(tlas_nodes[lb + 6u])),
                        tmin,
                        best.t,
                    );
                    let rh = slab_hit(
                        origin,
                        inv_d,
                        vec3<f32>(bitcast<f32>(tlas_nodes[rb_n]), bitcast<f32>(tlas_nodes[rb_n + 1u]), bitcast<f32>(tlas_nodes[rb_n + 2u])),
                        vec3<f32>(bitcast<f32>(tlas_nodes[rb_n + 4u]), bitcast<f32>(tlas_nodes[rb_n + 5u]), bitcast<f32>(tlas_nodes[rb_n + 6u])),
                        tmin,
                        best.t,
                    );
                    let hl = lh.y > 0.5;
                    let hr = rh.y > 0.5;
                    if hl && hr {
                        if tsp + 2u <= 64u {
                            if lh.x <= rh.x {
                                tstack[tsp] = r_idx;
                                tstack[tsp + 1u] = l_idx;
                            } else {
                                tstack[tsp] = l_idx;
                                tstack[tsp + 1u] = r_idx;
                            }
                            tsp = tsp + 2u;
                        }
                    } else if hl {
                        if tsp < 64u {
                            tstack[tsp] = l_idx;
                            tsp = tsp + 1u;
                        }
                    } else if hr {
                        if tsp < 64u {
                            tstack[tsp] = r_idx;
                            tsp = tsp + 1u;
                        }
                    }
                }
            }
        }
    }
    if best.tri < 0i {
        write_miss(r);
        return;
    }
    let b = r * 7u;
    hits[b] = bitcast<u32>(best.t);
    hits[b + 1u] = bitcast<u32>(best.u);
    hits[b + 2u] = bitcast<u32>(best.v);
    hits[b + 3u] = u32(best.tri);
    hits[b + 4u] = u32(best.inst);
    hits[b + 5u] = best.custom;
    hits[b + 6u] = best.front;
}
";

/// Pipelines for GPU-driven active-ray compaction + `dispatch_workgroups_indirect`.
struct CompactPipelines {
    layout: wgpu::BindGroupLayout,
    clear_args: wgpu::ComputePipeline,
    compact_rays: wgpu::ComputePipeline,
    finalize_args: wgpu::ComputePipeline,
}

/// Pipelines for the Hardware RT Core path (`enable wgpu_ray_query;`).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct HwTracePipelines {
    sanitize_layout: wgpu::BindGroupLayout,
    sanitize_verts: wgpu::ComputePipeline,
    sanitize_tris: wgpu::ComputePipeline,
    compact: CompactPipelines,
    trace_layout: wgpu::BindGroupLayout,
    trace: wgpu::ComputePipeline,
    clear_layout: wgpu::BindGroupLayout,
    clear: wgpu::ComputePipeline,
    empty_buf: wgpu::Buffer,
}

/// Pipelines for the portable GPU Compute BVH + Full-GPU LBVH path.
pub(crate) struct BvhTracePipelines {
    lbvh: LbvhPipelines,
    compact: CompactPipelines,
    trace_layout: wgpu::BindGroupLayout,
    trace: wgpu::ComputePipeline,
    clear_layout: wgpu::BindGroupLayout,
    clear: wgpu::ComputePipeline,
    empty_buf: wgpu::Buffer,
}

#[cfg(not(target_arch = "wasm32"))]
struct CachedSingleHw {
    _blas: Arc<wgpu::Blas>,
    tlas: Arc<wgpu::Tlas>,
    last_eval: u64,
}

/// Pre-allocated double-buffered dynamic BLAS/TLAS ring (`PREFER_FAST_BUILD | ALLOW_UPDATE`,
/// `PreferUpdate`) reused across animated frames with zero GPU buffer/AS allocations.
#[cfg(not(target_arch = "wasm32"))]
struct DynamicHwSlot {
    clean_verts: wgpu::Buffer,
    clean_tris: wgpu::Buffer,
    blas: Arc<wgpu::Blas>,
    tlas: Arc<wgpu::Tlas>,
    size_desc: wgpu::BlasTriangleGeometrySizeDescriptor,
    built_verts_id: u64,
}

#[cfg(not(target_arch = "wasm32"))]
struct DynamicHwBlas {
    slots: [DynamicHwSlot; 2],
    cursor: usize,
    last_eval: u64,
}

#[cfg(not(target_arch = "wasm32"))]
struct CachedMeshSetHw {
    blases: Vec<Option<Arc<wgpu::Blas>>>,
    last_eval: u64,
}

#[cfg(not(target_arch = "wasm32"))]
struct CachedInstancedHw {
    tlas: Option<Arc<wgpu::Tlas>>,
    last_eval: u64,
}

struct CachedMeshSetBvh {
    nodes: wgpu::Buffer,
    ordered: wgpu::Buffer,
    mesh_descs: wgpu::Buffer,
    bounds: Vec<Option<([f32; 3], [f32; 3])>>,
    last_eval: u64,
}

struct CachedInstancedBvh {
    tlas_nodes: wgpu::Buffer,
    tlas_ordered: wgpu::Buffer,
    inst_descs: wgpu::Buffer,
    tlas_node_count: u32,
    last_eval: u64,
}

#[derive(Default)]
struct CacheInner {
    #[cfg(not(target_arch = "wasm32"))]
    hw_single: HashMap<(u64, u64), CachedSingleHw>,
    #[cfg(not(target_arch = "wasm32"))]
    hw_seen_tris: HashMap<u64, u64>,
    #[cfg(not(target_arch = "wasm32"))]
    hw_dynamic: HashMap<(u64, u32, u32), DynamicHwBlas>,
    #[cfg(not(target_arch = "wasm32"))]
    hw_meshes: HashMap<(u64, u64, u64), CachedMeshSetHw>,
    #[cfg(not(target_arch = "wasm32"))]
    hw_instanced: HashMap<(u64, u64, u64, u64), CachedInstancedHw>,
    lbvh_single: HashMap<(u64, u32, u32), GpuLbvhMesh>,
    bvh_meshes: HashMap<(u64, u64, u64), CachedMeshSetBvh>,
    bvh_instanced: HashMap<(u64, u64, u64, u64), CachedInstancedBvh>,
}

/// Evicts the single oldest entry strictly older than `eval_now` so resources
/// recorded into the currently open frame batch are never dropped mid-frame.
fn evict_oldest_before<K: Eq + std::hash::Hash + Clone, V>(
    map: &mut HashMap<K, V>,
    eval_now: u64,
    last_eval: impl Fn(&V) -> u64,
) {
    if map.len() < CACHE_MAX_ENTRIES {
        return;
    }
    let victim = map
        .iter()
        .filter(|(_, v)| last_eval(v) < eval_now)
        .min_by_key(|(_, v)| last_eval(v))
        .map(|(k, _)| k.clone());
    if let Some(k) = victim {
        map.remove(&k);
    }
}

/// Bounded per-device cache of hardware BLAS/TLAS acceleration structures and
/// GPU Compute LBVH/BVH buffers, keyed by immutable [`DeviceImage::id`]s and aged
/// once per graph evaluation.
#[derive(Default)]
pub(crate) struct TraceCache {
    eval: std::sync::atomic::AtomicU64,
    inner: Mutex<CacheInner>,
}

impl TraceCache {
    pub(crate) fn evaluation_begun(&self) {
        let now = self.eval.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            inner
                .hw_single
                .retain(|_, v| now.saturating_sub(v.last_eval) <= CACHE_MAX_AGE);
            inner
                .hw_dynamic
                .retain(|_, v| now.saturating_sub(v.last_eval) <= CACHE_MAX_AGE);
            inner
                .hw_meshes
                .retain(|_, v| now.saturating_sub(v.last_eval) <= CACHE_MAX_AGE);
            inner
                .hw_instanced
                .retain(|_, v| now.saturating_sub(v.last_eval) <= CACHE_MAX_AGE);
            if inner.hw_seen_tris.len() > CACHE_MAX_ENTRIES * 4 {
                inner.hw_seen_tris.clear();
            }
        }
        inner
            .lbvh_single
            .retain(|_, v| now.saturating_sub(v.last_eval) <= CACHE_MAX_AGE);
        inner
            .bvh_meshes
            .retain(|_, v| now.saturating_sub(v.last_eval) <= CACHE_MAX_AGE);
        inner
            .bvh_instanced
            .retain(|_, v| now.saturating_sub(v.last_eval) <= CACHE_MAX_AGE);
    }

    fn current_eval(&self) -> u64 {
        self.eval.load(std::sync::atomic::Ordering::Relaxed)
    }
}

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

fn make_compact_pipelines(device: &wgpu::Device) -> CompactPipelines {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("entelechy trace compact"),
        source: wgpu::ShaderSource::Wgsl(COMPACT_RAYS_WGSL.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("entelechy trace compact"),
        entries: &[
            uniform_entry(0),
            storage_entry(1, true),
            storage_entry(2, false),
            storage_entry(3, false),
            storage_entry(4, false),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("entelechy trace compact"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let mk = |ep: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(ep),
            layout: Some(&pl),
            module: &module,
            entry_point: Some(ep),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    CompactPipelines {
        layout,
        clear_args: mk("clear_args"),
        compact_rays: mk("compact_rays"),
        finalize_args: mk("finalize_args"),
    }
}

fn make_clear_pipeline(device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("entelechy trace clear"),
        source: wgpu::ShaderSource::Wgsl(CLEAR_MISS_WGSL.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("entelechy trace clear"),
        entries: &[uniform_entry(0), storage_entry(1, false)],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("entelechy trace clear"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("entelechy trace clear"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (layout, pipeline)
}

#[cfg(not(target_arch = "wasm32"))]
fn build_hw_pipelines(
    device: &wgpu::Device,
    wait: crate::Wait,
) -> Result<HwTracePipelines, String> {
    let guard = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let san_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("entelechy trace sanitize"),
        source: wgpu::ShaderSource::Wgsl(SANITIZE_BLAS_WGSL.into()),
    });
    let sanitize_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("entelechy trace sanitize"),
        entries: &[
            uniform_entry(0),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            storage_entry(4, false),
        ],
    });
    let san_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("entelechy trace sanitize"),
        bind_group_layouts: &[Some(&sanitize_layout)],
        immediate_size: 0,
    });
    let sanitize_verts = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("entelechy trace sanitize verts"),
        layout: Some(&san_pl),
        module: &san_mod,
        entry_point: Some("sanitize_verts"),
        compilation_options: Default::default(),
        cache: None,
    });
    let sanitize_tris = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("entelechy trace sanitize tris"),
        layout: Some(&san_pl),
        module: &san_mod,
        entry_point: Some("sanitize_tris"),
        compilation_options: Default::default(),
        cache: None,
    });

    let trace_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("entelechy trace hw"),
        source: wgpu::ShaderSource::Wgsl(HW_TRACE_WGSL.into()),
    });
    let trace_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("entelechy trace hw"),
        entries: &[
            uniform_entry(0),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::AccelerationStructure {
                    vertex_return: false,
                },
                count: None,
            },
            storage_entry(2, true),
            storage_entry(3, false),
            storage_entry(4, true),
            storage_entry(5, true),
            storage_entry(6, true),
            storage_entry(7, true),
            storage_entry(8, true),
        ],
    });
    let trace_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("entelechy trace hw"),
        bind_group_layouts: &[Some(&trace_layout)],
        immediate_size: 0,
    });
    let trace = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("entelechy trace hw"),
        layout: Some(&trace_pl),
        module: &trace_mod,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let compact = make_compact_pipelines(device);
    let (clear_layout, clear) = make_clear_pipeline(device);
    let empty_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("entelechy trace empty"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    if let Some(error) = wait.block_on(guard.pop()) {
        return Err(format!(
            "trace hw pipeline: {}",
            crate::device_error(&error)
        ));
    }
    Ok(HwTracePipelines {
        sanitize_layout,
        sanitize_verts,
        sanitize_tris,
        compact,
        trace_layout,
        trace,
        clear_layout,
        clear,
        empty_buf,
    })
}

fn build_bvh_pipelines(
    device: &wgpu::Device,
    wait: crate::Wait,
) -> Result<BvhTracePipelines, String> {
    let guard = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let lbvh = LbvhPipelines::build(device);
    let compact = make_compact_pipelines(device);
    let trace_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("entelechy trace bvh"),
        source: wgpu::ShaderSource::Wgsl(BVH_TRACE_WGSL.into()),
    });
    let trace_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("entelechy trace bvh"),
        entries: &[
            uniform_entry(0),
            storage_entry(1, true),
            storage_entry(2, false),
            storage_entry(3, true),
            storage_entry(4, true),
            storage_entry(5, true),
            storage_entry(6, true),
            storage_entry(7, true),
            storage_entry(8, true),
            storage_entry(9, true),
            storage_entry(10, true),
        ],
    });
    let trace_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("entelechy trace bvh"),
        bind_group_layouts: &[Some(&trace_layout)],
        immediate_size: 0,
    });
    let trace = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("entelechy trace bvh"),
        layout: Some(&trace_pl),
        module: &trace_mod,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let (clear_layout, clear) = make_clear_pipeline(device);
    let empty_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("entelechy trace bvh empty"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    if let Some(error) = wait.block_on(guard.pop()) {
        return Err(format!(
            "trace bvh pipeline: {}",
            crate::device_error(&error)
        ));
    }
    Ok(BvhTracePipelines {
        lbvh,
        compact,
        trace_layout,
        trace,
        clear_layout,
        clear,
        empty_buf,
    })
}

fn image_id(scope: &Scope<'_>, handle: Handle) -> Option<u64> {
    scope
        .resident(handle)
        .ok()
        .flatten()
        .and_then(|any| any.downcast_ref::<DeviceImage>())
        .map(|dev| dev.id)
}

/// Reads words of a Host-backed or GPU-resident input handle (downloading from
/// the device only if Host memory does not already hold it).
fn read_words(
    gpu: &WgpuBackend,
    scope: &Scope<'_>,
    handle: Handle,
) -> Result<Vec<u32>, ExecutionError> {
    if let Ok(values) = scope.values(handle) {
        return Ok(values.into_owned());
    }
    let byte_len = scope
        .schema(handle)?
        .storage_words(scope.size(handle)?.count())
        * 4;
    let dev = scope
        .resident(handle)?
        .ok_or_else(|| ExecutionError::Device("trace input is neither host nor resident".into()))?;
    gpu.download_unlocked(dev, byte_len)
}

fn upload_words_buf(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    words: &[u32],
) -> wgpu::Buffer {
    let byte_len = ((words.len() * 4) as u64).max(16).div_ceil(16) * 16;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !words.is_empty() {
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for &w in words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        queue.write_buffer(&buf, 0, &bytes);
    }
    buf
}

fn dispatch_clear_misses(
    gpu: &WgpuBackend,
    clear_layout: &wgpu::BindGroupLayout,
    clear_pipeline: &wgpu::ComputePipeline,
    ray_count: u32,
    domain: Domain,
    scope: &mut Scope<'_>,
) -> Result<Vec<Handle>, ExecutionError> {
    let out_bytes = (ray_count as u64 * HIT_WORDS as u64 * 4).max(16);
    let output_buffer = gpu.spare().take(
        &gpu.device,
        out_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let mut setup_bytes = [0u8; 16];
    setup_bytes[..4].copy_from_slice(&ray_count.to_le_bytes());
    let setup_buf = gpu.take_uniform(&setup_bytes);
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("entelechy trace clear"),
        layout: clear_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: setup_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });
    gpu.batch.flush()?;
    let scopes = Scopes::push(&gpu.device);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("entelechy trace clear"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("entelechy trace clear"),
            timestamp_writes: None,
        });
        pass.set_pipeline(clear_pipeline);
        pass.set_bind_group(0, &group, &[]);
        let (gx, gy) = wgsl::folded(ray_count.div_ceil(TRACE_WORKGROUP));
        pass.dispatch_workgroups(gx, gy, 1);
    }
    gpu.submit(encoder.finish());
    gpu.spare().keep(setup_buf, None);
    if gpu._waiter.is_none() {
        let _ = gpu.device.poll(wgpu::PollType::Poll);
    }
    scopes
        .pop(gpu.wait, "")
        .map_err(|error| ExecutionError::Device(error.said()))?;
    let dev = Box::new(DeviceImage::new(
        output_buffer,
        Some(gpu.spare().clone()),
        None,
    ));
    Ok(vec![scope.adopt_resident(hit_schema(), domain, dev)?])
}

/// Entry point from [`WgpuBackend::execute_native`] for `NativeOp::Trace` and
/// `NativeOp::TraceInstances`.
pub(crate) fn trace(
    gpu: &WgpuBackend,
    hw_lock: &OnceLock<Result<HwTraceSlot, String>>,
    bvh_lock: &OnceLock<Result<BvhTracePipelines, String>>,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
    op: NativeOp,
) -> Result<Vec<Handle>, ExecutionError> {
    #[cfg(not(target_arch = "wasm32"))]
    if gpu.ray_query {
        return trace_hw(gpu, hw_lock, invocation, scope, op);
    }
    #[cfg(target_arch = "wasm32")]
    let _ = hw_lock;
    trace_bvh(gpu, bvh_lock, invocation, scope, op)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) type HwTraceSlot = HwTracePipelines;
#[cfg(target_arch = "wasm32")]
pub(crate) type HwTraceSlot = ();

fn alloc_uniform_slices(
    gpu: &WgpuBackend,
    batched: bool,
    payloads: &[&[u8]],
    temp_buffers: &mut Vec<wgpu::Buffer>,
) -> Result<Vec<UniformSlice>, ExecutionError> {
    if payloads.is_empty() {
        return Ok(Vec::new());
    }
    if batched {
        gpu.batch.alloc_uniforms(payloads)
    } else {
        let mut slices = Vec::with_capacity(payloads.len());
        for bytes in payloads {
            let buf = gpu.take_uniform(bytes);
            temp_buffers.push(buf.clone());
            slices.push(UniformSlice::entire(buf));
        }
        Ok(slices)
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_compact_rays(
    gpu: &WgpuBackend,
    compact: &CompactPipelines,
    encoder: &mut wgpu::CommandEncoder,
    compact_u: &UniformSlice,
    rays_buf: &wgpu::Buffer,
    output_buffer: &wgpu::Buffer,
    ray_count: u32,
    timing_set: Option<&crate::timing::Timing>,
    opened_ts: &mut bool,
    temp_buffers: &mut Vec<wgpu::Buffer>,
) -> (wgpu::Buffer, wgpu::Buffer) {
    let active_bytes = ((u64::from(ray_count) + 1) * 4).max(16).div_ceil(16) * 16;
    let active_rays = gpu
        .spare()
        .take(&gpu.device, active_bytes, wgpu::BufferUsages::STORAGE);
    let indirect_args = gpu.spare().take(
        &gpu.device,
        16,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
    );
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("entelechy trace compact"),
        layout: &compact.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: compact_u.binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rays_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: active_rays.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: indirect_args.as_entire_binding(),
            },
        ],
    });
    let ts_open = if !*opened_ts {
        *opened_ts = true;
        timing_set.map(|t| t.compute(Some(0), None))
    } else {
        None
    };
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("entelechy trace compact"),
            timestamp_writes: ts_open,
        });
        pass.set_bind_group(0, &group, &[]);
        pass.set_pipeline(&compact.clear_args);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&compact.compact_rays);
        let (gx, gy) = wgsl::folded(ray_count.div_ceil(TRACE_WORKGROUP).max(1));
        pass.dispatch_workgroups(gx, gy, 1);
        pass.set_pipeline(&compact.finalize_args);
        pass.dispatch_workgroups(1, 1, 1);
    }
    temp_buffers.push(active_rays.clone());
    temp_buffers.push(indirect_args.clone());
    (active_rays, indirect_args)
}

#[allow(clippy::too_many_arguments)]
fn finish_trace_encoder(
    gpu: &WgpuBackend,
    mut encoder: wgpu::CommandEncoder,
    batched: bool,
    scopes: Option<Scopes>,
    timing_set: Option<crate::timing::Timing>,
    est_ops: u64,
    temp_buffers: Vec<wgpu::Buffer>,
) -> Result<(), ExecutionError> {
    let timing = gpu.resolve_timing(&mut encoder, timing_set.as_ref(), 2);
    if batched {
        gpu.frames.note_batched_run();
        let page_budget = gpu
            .wait
            .parks_with_host()
            .then(|| gpu.page_pace_ops_budget());
        gpu.batch
            .finish_node(encoder, est_ops, temp_buffers, Vec::new(), page_budget)?;
        if gpu.wait.parks_with_host() && gpu.page_queue.full(0, gpu.page_pace_ops_budget()) {
            gpu.batch.flush()?;
            gpu.drain_page()?;
        }
    } else {
        let submission = gpu.submit(encoder.finish());
        if gpu.wait.parks_with_host() && !gpu.measured() {
            gpu.page_queue.note(est_ops);
            if gpu.page_queue.full(0, gpu.page_pace_ops_budget()) {
                gpu.drain_page()?;
            }
        }
        gpu.settle_measured(submission, timing);
        if let Some(scopes) = scopes {
            if gpu.wait.parks_with_host() && !gpu.measured() {
                gpu.defer_check(Box::new(scopes.pop_later(gpu.wait, gpu.spent.clone())));
            } else if let Err(error) = scopes.pop(gpu.wait, "") {
                return Err(ExecutionError::Device(format!("trace: {}", error.said())));
            }
        }
        for buf in temp_buffers {
            gpu.spare().keep(buf, None);
        }
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn sanitize_setup_bytes(slice: MeshSlice) -> [u8; 16] {
    let setup: [u32; 4] = [
        slice.vertex_offset as u32,
        slice.vertex_count as u32,
        slice.triangle_offset as u32,
        slice.triangle_count as u32,
    ];
    let mut bytes = [0u8; 16];
    for (slot, w) in bytes.chunks_exact_mut(4).zip(setup) {
        slot.copy_from_slice(&w.to_le_bytes());
    }
    bytes
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn encode_sanitize_into(
    device: &wgpu::Device,
    pipelines: &HwTracePipelines,
    encoder: &mut wgpu::CommandEncoder,
    setup_u: &UniformSlice,
    src_verts: &wgpu::Buffer,
    src_tris: &wgpu::Buffer,
    clean_verts: &wgpu::Buffer,
    clean_tris: &wgpu::Buffer,
    v_cnt: u32,
    t_cnt: u32,
    timing_set: Option<&crate::timing::Timing>,
    opened_ts: &mut bool,
) {
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("entelechy trace sanitize"),
        layout: &pipelines.sanitize_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: setup_u.binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: src_verts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: src_tris.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: clean_verts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: clean_tris.as_entire_binding(),
            },
        ],
    });
    let ts_open = if !*opened_ts {
        *opened_ts = true;
        timing_set.map(|t| t.compute(Some(0), None))
    } else {
        None
    };
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("entelechy trace sanitize"),
        timestamp_writes: ts_open,
    });
    pass.set_bind_group(0, &group, &[]);
    pass.set_pipeline(&pipelines.sanitize_verts);
    let (vx, vy) = wgsl::folded(v_cnt.div_ceil(TRACE_WORKGROUP));
    pass.dispatch_workgroups(vx, vy, 1);
    pass.set_pipeline(&pipelines.sanitize_tris);
    let (tx, ty) = wgsl::folded(t_cnt.div_ceil(TRACE_WORKGROUP));
    pass.dispatch_workgroups(tx, ty, 1);
}

#[cfg(not(target_arch = "wasm32"))]
fn alloc_dynamic_hw_slot(device: &wgpu::Device, v_cnt: u32, t_cnt: u32) -> DynamicHwSlot {
    let clean_verts = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("entelechy dyn blas verts"),
        size: (u64::from(v_cnt) * 12).max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
        mapped_at_creation: false,
    });
    let clean_tris = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("entelechy dyn blas tris"),
        size: (u64::from(t_cnt) * 12).max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
        mapped_at_creation: false,
    });
    let size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: v_cnt,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(t_cnt * 3),
        flags: wgpu::AccelerationStructureGeometryFlags::empty(),
    };
    let blas = Arc::new(device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("entelechy dyn blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD
                | wgpu::AccelerationStructureFlags::ALLOW_UPDATE,
            update_mode: wgpu::AccelerationStructureUpdateMode::PreferUpdate,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![size_desc.clone()],
        },
    ));
    let mut tlas_raw = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("entelechy dyn tlas"),
        max_instances: 1,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    });
    let identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    tlas_raw[0] = Some(wgpu::TlasInstance::new(&blas, identity, 0, 0xFF));
    DynamicHwSlot {
        clean_verts,
        clean_tris,
        blas,
        tlas: Arc::new(tlas_raw),
        size_desc,
        built_verts_id: 0,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn alloc_static_blas(
    device: &wgpu::Device,
    slice: MeshSlice,
) -> Option<(
    Arc<wgpu::Blas>,
    wgpu::BlasTriangleGeometrySizeDescriptor,
    wgpu::Buffer,
    wgpu::Buffer,
)> {
    if slice.vertex_count == 0 || slice.triangle_count == 0 {
        return None;
    }
    let v_cnt = slice.vertex_count as u32;
    let t_cnt = slice.triangle_count as u32;
    let clean_verts = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("entelechy blas verts"),
        size: (u64::from(v_cnt) * 12).max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
        mapped_at_creation: false,
    });
    let clean_tris = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("entelechy blas tris"),
        size: (u64::from(t_cnt) * 12).max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
        mapped_at_creation: false,
    });
    let size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: v_cnt,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(t_cnt * 3),
        flags: wgpu::AccelerationStructureGeometryFlags::empty(),
    };
    let blas = Arc::new(device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("entelechy blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![size_desc.clone()],
        },
    ));
    Some((blas, size_desc, clean_verts, clean_tris))
}

#[cfg(not(target_arch = "wasm32"))]
enum SingleHwPlan {
    Cached(Arc<wgpu::Tlas>),
    RebuildDynamic { slot_idx: usize },
    PromoteDynamic,
    BuildStatic,
}

#[cfg(not(target_arch = "wasm32"))]
fn trace_hw(
    gpu: &WgpuBackend,
    pipelines_lock: &OnceLock<Result<HwTracePipelines, String>>,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
    op: NativeOp,
) -> Result<Vec<Handle>, ExecutionError> {
    let pipelines = match pipelines_lock.get_or_init(|| build_hw_pipelines(&gpu.device, gpu.wait)) {
        Ok(p) => p,
        Err(err) => return Err(ExecutionError::Device(err.clone())),
    };
    let _guard = gpu
        .lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let domain = invocation.domain;
    let ray_count = domain.count() as u32;
    let eval_now = gpu.trace_cache.current_eval();

    let (rays_h, verts_h, tris_h, meshes_h, insts_h, flags, mask, instanced) = match op {
        NativeOp::Trace => {
            let inp = check_trace(invocation, scope)?;
            (
                inp.rays,
                inp.vertices,
                inp.triangles,
                None,
                None,
                inp.flags,
                0xFFu8,
                false,
            )
        }
        NativeOp::TraceInstances => {
            let inp = check_trace_instances(invocation, scope)?;
            (
                inp.rays,
                inp.vertices,
                inp.triangles,
                Some(inp.meshes),
                Some(inp.instances),
                inp.flags,
                inp.mask,
                true,
            )
        }
        _ => unreachable!("only Trace and TraceInstances enter trace::trace"),
    };

    let v_count = scope.size(verts_h)?.count() as u32;
    let t_count = scope.size(tris_h)?.count() as u32;
    if v_count == 0 || t_count == 0 || mask == 0 {
        return dispatch_clear_misses(
            gpu,
            &pipelines.clear_layout,
            &pipelines.clear,
            ray_count,
            domain,
            scope,
        );
    }

    let rays_buf = gpu.bind_input(scope, rays_h)?;
    let verts_buf = gpu.bind_input(scope, verts_h)?;
    let tris_buf = gpu.bind_input(scope, tris_h)?;
    let verts_id = image_id(scope, verts_h).unwrap_or(0);
    let tris_id = image_id(scope, tris_h).unwrap_or(0);

    // Phase 1: Resolve cache & any Host/GPU reads BEFORE allocating batch uniforms or
    // opening the shared command encoder.
    let mut single_plan = None;
    let mut inst_cached_tlas: Option<Option<Arc<wgpu::Tlas>>> = None;
    let mut inst_pending_builds: Vec<(
        MeshSlice,
        Arc<wgpu::Blas>,
        wgpu::BlasTriangleGeometrySizeDescriptor,
        wgpu::Buffer,
        wgpu::Buffer,
    )> = Vec::new();
    let mut inst_new_tlas: Option<Arc<wgpu::Tlas>> = None;
    let mut meshes_buf = pipelines.empty_buf.clone();
    let mut insts_buf = pipelines.empty_buf.clone();
    let mut mesh_count = 0u32;

    if !instanced {
        let dyn_key = (tris_id, v_count, t_count);
        let static_key = (verts_id, tris_id);
        let mut cache = gpu
            .trace_cache
            .inner
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if let Some(dyn_entry) = cache.hw_dynamic.get_mut(&dyn_key) {
            dyn_entry.last_eval = eval_now;
            if let Some(slot) = dyn_entry
                .slots
                .iter()
                .find(|s| s.built_verts_id == verts_id)
            {
                single_plan = Some(SingleHwPlan::Cached(Arc::clone(&slot.tlas)));
            } else {
                let slot_idx = dyn_entry.cursor & 1;
                dyn_entry.cursor = dyn_entry.cursor.wrapping_add(1);
                single_plan = Some(SingleHwPlan::RebuildDynamic { slot_idx });
            }
        } else if let Some(entry) = cache.hw_single.get_mut(&static_key) {
            entry.last_eval = eval_now;
            single_plan = Some(SingleHwPlan::Cached(Arc::clone(&entry.tlas)));
        } else if tris_id != 0
            && cache
                .hw_seen_tris
                .get(&tris_id)
                .is_some_and(|&prev_v| prev_v != verts_id)
        {
            single_plan = Some(SingleHwPlan::PromoteDynamic);
        } else {
            if tris_id != 0 {
                cache.hw_seen_tris.insert(tris_id, verts_id);
            }
            single_plan = Some(SingleHwPlan::BuildStatic);
        }
    } else {
        let mh = meshes_h.expect("instanced trace has meshes");
        let ih = insts_h.expect("instanced trace has instances");
        mesh_count = scope.size(mh)?.count() as u32;
        let i_count = scope.size(ih)?.count() as u32;
        if mesh_count == 0 || i_count == 0 {
            return dispatch_clear_misses(
                gpu,
                &pipelines.clear_layout,
                &pipelines.clear,
                ray_count,
                domain,
                scope,
            );
        }
        meshes_buf = gpu.bind_input(scope, mh)?;
        insts_buf = gpu.bind_input(scope, ih)?;
        let meshes_id = image_id(scope, mh).unwrap_or(0);
        let insts_id = image_id(scope, ih).unwrap_or(0);
        let tlas_key = (verts_id, tris_id, meshes_id, insts_id);
        let cached = {
            let mut cache = gpu
                .trace_cache
                .inner
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            cache.hw_instanced.get_mut(&tlas_key).map(|entry| {
                entry.last_eval = eval_now;
                entry.tlas.clone()
            })
        };
        if let Some(tlas_opt) = cached {
            inst_cached_tlas = Some(tlas_opt);
        } else {
            let mesh_key = (verts_id, tris_id, meshes_id);
            let cached_meshes = {
                let mut cache = gpu
                    .trace_cache
                    .inner
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                cache.hw_meshes.get_mut(&mesh_key).map(|entry| {
                    entry.last_eval = eval_now;
                    entry.blases.clone()
                })
            };
            let blases = if let Some(blases) = cached_meshes {
                blases
            } else {
                let mesh_words = read_words(gpu, scope, mh)?;
                let slices = trace::parse_meshes(&mesh_words, v_count as usize, t_count as usize);
                let mut blases = Vec::with_capacity(slices.len());
                for &slice in &slices {
                    if let Some((blas, size_desc, clean_v, clean_t)) =
                        alloc_static_blas(&gpu.device, slice)
                    {
                        blases.push(Some(Arc::clone(&blas)));
                        inst_pending_builds.push((slice, blas, size_desc, clean_v, clean_t));
                    } else {
                        blases.push(None);
                    }
                }
                let mut cache = gpu
                    .trace_cache
                    .inner
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                evict_oldest_before(&mut cache.hw_meshes, eval_now, |v| v.last_eval);
                cache.hw_meshes.insert(
                    mesh_key,
                    CachedMeshSetHw {
                        blases: blases.clone(),
                        last_eval: eval_now,
                    },
                );
                blases
            };

            let inst_words = read_words(gpu, scope, ih)?;
            let mut active_instances: Vec<wgpu::TlasInstance> = Vec::new();
            let total_insts = inst_words.len() / INSTANCE_WORDS;
            for row in 0..total_insts {
                let base = row * INSTANCE_WORDS;
                let r0 = [
                    f32::from_bits(inst_words[base]),
                    f32::from_bits(inst_words[base + 1]),
                    f32::from_bits(inst_words[base + 2]),
                    f32::from_bits(inst_words[base + 3]),
                ];
                let r1 = [
                    f32::from_bits(inst_words[base + 4]),
                    f32::from_bits(inst_words[base + 5]),
                    f32::from_bits(inst_words[base + 6]),
                    f32::from_bits(inst_words[base + 7]),
                ];
                let r2 = [
                    f32::from_bits(inst_words[base + 8]),
                    f32::from_bits(inst_words[base + 9]),
                    f32::from_bits(inst_words[base + 10]),
                    f32::from_bits(inst_words[base + 11]),
                ];
                let mesh_i = inst_words[base + 12] as i32;
                let imask = (inst_words[base + 14] as i32 & 0xFF) as u8;
                if imask == 0 || mesh_i < 0 || (mesh_i as usize) >= blases.len() {
                    continue;
                }
                let Some(ref blas) = blases[mesh_i as usize] else {
                    continue;
                };
                if trace::invert_affine_3x4(r0, r1, r2).is_none() {
                    continue;
                }
                let transform = [
                    r0[0], r0[1], r0[2], r0[3], r1[0], r1[1], r1[2], r1[3], r2[0], r2[1], r2[2],
                    r2[3],
                ];
                active_instances.push(wgpu::TlasInstance::new(
                    blas,
                    transform,
                    (row as u32) & 0x00FF_FFFF,
                    imask,
                ));
            }
            if !active_instances.is_empty() {
                let mut tlas_raw = gpu.device.create_tlas(&wgpu::CreateTlasDescriptor {
                    label: Some("entelechy trace instanced tlas"),
                    max_instances: active_instances.len() as u32,
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                });
                for (slot, inst) in active_instances.into_iter().enumerate() {
                    tlas_raw[slot] = Some(inst);
                }
                inst_new_tlas = Some(Arc::new(tlas_raw));
            }
            let mut cache = gpu
                .trace_cache
                .inner
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            evict_oldest_before(&mut cache.hw_instanced, eval_now, |v| v.last_eval);
            cache.hw_instanced.insert(
                tlas_key,
                CachedInstancedHw {
                    tlas: inst_new_tlas.clone(),
                    last_eval: eval_now,
                },
            );
        }
    }

    // Handle zero-active-instance case after flushing any newly built BLASes.
    if instanced && inst_cached_tlas.as_ref().is_some_and(Option::is_none) {
        return dispatch_clear_misses(
            gpu,
            &pipelines.clear_layout,
            &pipelines.clear,
            ray_count,
            domain,
            scope,
        );
    }
    let has_tlas = !instanced || inst_cached_tlas.is_some() || inst_new_tlas.is_some();

    let use_compact = has_tlas && ray_count >= COMPACT_RAY_THRESHOLD && (flags & 6) != 6;
    let est_ops = u64::from(ray_count).saturating_mul(64);
    let batched = gpu.batch.can_batch(gpu.measured(), &gpu.frames);
    if !batched {
        gpu.batch.flush()?;
    }

    // Phase 2: Prepare all uniform payloads atomically before taking the encoder.
    let single_slice = MeshSlice {
        vertex_offset: 0,
        vertex_count: v_count as usize,
        triangle_offset: 0,
        triangle_count: t_count as usize,
    };
    let mut sanitize_payloads: Vec<[u8; 16]> = Vec::new();
    if let Some(plan) = &single_plan {
        if !matches!(plan, SingleHwPlan::Cached(_)) {
            sanitize_payloads.push(sanitize_setup_bytes(single_slice));
        }
    } else {
        for (slice, _, _, _, _) in &inst_pending_builds {
            sanitize_payloads.push(sanitize_setup_bytes(*slice));
        }
    }
    let mut compact_payload = [0u8; 16];
    compact_payload[..4].copy_from_slice(&ray_count.to_le_bytes());
    compact_payload[4..8].copy_from_slice(&(flags as u32).to_le_bytes());
    let trace_setup: [u32; 8] = [
        ray_count,
        v_count,
        t_count,
        flags as u32,
        u32::from(mask),
        u32::from(instanced),
        mesh_count,
        u32::from(use_compact),
    ];
    let mut trace_payload = [0u8; 32];
    for (slot, w) in trace_payload.chunks_exact_mut(4).zip(trace_setup) {
        slot.copy_from_slice(&w.to_le_bytes());
    }

    let mut all_payloads: Vec<&[u8]> = sanitize_payloads.iter().map(|b| b.as_slice()).collect();
    if use_compact {
        all_payloads.push(&compact_payload);
    }
    if has_tlas {
        all_payloads.push(&trace_payload);
    }
    let mut temp_buffers = Vec::new();
    let uniform_slices = alloc_uniform_slices(gpu, batched, &all_payloads, &mut temp_buffers)?;
    let san_count = sanitize_payloads.len();
    let compact_u = use_compact.then(|| &uniform_slices[san_count]);
    let trace_u = has_tlas.then(|| &uniform_slices[san_count + usize::from(use_compact)]);

    let scopes = (!batched).then(|| Scopes::push(&gpu.device));
    let timing_set = has_tlas.then(|| gpu.timing_set(2)).flatten();
    let mut opened_ts = false;
    let mut encoder = if batched {
        gpu.batch.take_or_open()
    } else {
        gpu.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("entelechy trace hw"),
            })
    };

    // Phase 3: Encode BLAS/TLAS sanitize + acceleration structure build.
    let final_tlas: Option<Arc<wgpu::Tlas>> = if let Some(plan) = single_plan {
        match plan {
            SingleHwPlan::Cached(tlas) => Some(tlas),
            SingleHwPlan::RebuildDynamic { slot_idx } => {
                let dyn_key = (tris_id, v_count, t_count);
                let mut cache = gpu
                    .trace_cache
                    .inner
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                let dyn_entry = cache
                    .hw_dynamic
                    .get_mut(&dyn_key)
                    .expect("dynamic BLAS slot exists");
                let slot = &mut dyn_entry.slots[slot_idx];
                encode_sanitize_into(
                    &gpu.device,
                    pipelines,
                    &mut encoder,
                    &uniform_slices[0],
                    &verts_buf,
                    &tris_buf,
                    &slot.clean_verts,
                    &slot.clean_tris,
                    v_count,
                    t_count,
                    timing_set.as_ref(),
                    &mut opened_ts,
                );
                let blas_entry = wgpu::BlasBuildEntry {
                    blas: &slot.blas,
                    geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                        wgpu::BlasTriangleGeometry {
                            size: &slot.size_desc,
                            vertex_buffer: &slot.clean_verts,
                            first_vertex: 0,
                            vertex_stride: 12,
                            index_buffer: Some(&slot.clean_tris),
                            first_index: Some(0),
                            transform_buffer: None,
                            transform_buffer_offset: None,
                        },
                    ]),
                };
                encoder.build_acceleration_structures(
                    std::iter::once(&blas_entry),
                    std::iter::once(&*slot.tlas),
                );
                slot.built_verts_id = verts_id;
                dyn_entry.last_eval = eval_now;
                Some(Arc::clone(&slot.tlas))
            }
            SingleHwPlan::PromoteDynamic => {
                let slot0 = alloc_dynamic_hw_slot(&gpu.device, v_count, t_count);
                let slot1 = alloc_dynamic_hw_slot(&gpu.device, v_count, t_count);
                encode_sanitize_into(
                    &gpu.device,
                    pipelines,
                    &mut encoder,
                    &uniform_slices[0],
                    &verts_buf,
                    &tris_buf,
                    &slot0.clean_verts,
                    &slot0.clean_tris,
                    v_count,
                    t_count,
                    timing_set.as_ref(),
                    &mut opened_ts,
                );
                let blas_entry = wgpu::BlasBuildEntry {
                    blas: &slot0.blas,
                    geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                        wgpu::BlasTriangleGeometry {
                            size: &slot0.size_desc,
                            vertex_buffer: &slot0.clean_verts,
                            first_vertex: 0,
                            vertex_stride: 12,
                            index_buffer: Some(&slot0.clean_tris),
                            first_index: Some(0),
                            transform_buffer: None,
                            transform_buffer_offset: None,
                        },
                    ]),
                };
                encoder.build_acceleration_structures(
                    std::iter::once(&blas_entry),
                    std::iter::once(&*slot0.tlas),
                );
                let tlas = Arc::clone(&slot0.tlas);
                let mut s0 = slot0;
                s0.built_verts_id = verts_id;
                let dyn_key = (tris_id, v_count, t_count);
                let mut cache = gpu
                    .trace_cache
                    .inner
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                evict_oldest_before(&mut cache.hw_dynamic, eval_now, |v| v.last_eval);
                cache.hw_dynamic.insert(
                    dyn_key,
                    DynamicHwBlas {
                        slots: [s0, slot1],
                        cursor: 1,
                        last_eval: eval_now,
                    },
                );
                Some(tlas)
            }
            SingleHwPlan::BuildStatic => {
                let (blas, size_desc, clean_v, clean_t) =
                    alloc_static_blas(&gpu.device, single_slice)
                        .expect("non-empty mesh produces BLAS");
                encode_sanitize_into(
                    &gpu.device,
                    pipelines,
                    &mut encoder,
                    &uniform_slices[0],
                    &verts_buf,
                    &tris_buf,
                    &clean_v,
                    &clean_t,
                    v_count,
                    t_count,
                    timing_set.as_ref(),
                    &mut opened_ts,
                );
                let mut tlas_raw = gpu.device.create_tlas(&wgpu::CreateTlasDescriptor {
                    label: Some("entelechy trace single tlas"),
                    max_instances: 1,
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                });
                let identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
                tlas_raw[0] = Some(wgpu::TlasInstance::new(&blas, identity, 0, 0xFF));
                let blas_entry = wgpu::BlasBuildEntry {
                    blas: &blas,
                    geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                        wgpu::BlasTriangleGeometry {
                            size: &size_desc,
                            vertex_buffer: &clean_v,
                            first_vertex: 0,
                            vertex_stride: 12,
                            index_buffer: Some(&clean_t),
                            first_index: Some(0),
                            transform_buffer: None,
                            transform_buffer_offset: None,
                        },
                    ]),
                };
                encoder.build_acceleration_structures(
                    std::iter::once(&blas_entry),
                    std::iter::once(&tlas_raw),
                );
                let tlas = Arc::new(tlas_raw);
                let mut cache = gpu
                    .trace_cache
                    .inner
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                evict_oldest_before(&mut cache.hw_single, eval_now, |v| v.last_eval);
                cache.hw_single.insert(
                    (verts_id, tris_id),
                    CachedSingleHw {
                        _blas: blas,
                        tlas: Arc::clone(&tlas),
                        last_eval: eval_now,
                    },
                );
                Some(tlas)
            }
        }
    } else if let Some(cached) = inst_cached_tlas {
        cached
    } else {
        for (idx, (slice, _, _, clean_v, clean_t)) in inst_pending_builds.iter().enumerate() {
            encode_sanitize_into(
                &gpu.device,
                pipelines,
                &mut encoder,
                &uniform_slices[idx],
                &verts_buf,
                &tris_buf,
                clean_v,
                clean_t,
                slice.vertex_count as u32,
                slice.triangle_count as u32,
                timing_set.as_ref(),
                &mut opened_ts,
            );
        }
        let entries: Vec<wgpu::BlasBuildEntry<'_>> = inst_pending_builds
            .iter()
            .map(
                |(_, blas, size_desc, clean_v, clean_t)| wgpu::BlasBuildEntry {
                    blas,
                    geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                        wgpu::BlasTriangleGeometry {
                            size: size_desc,
                            vertex_buffer: clean_v,
                            first_vertex: 0,
                            vertex_stride: 12,
                            index_buffer: Some(clean_t),
                            first_index: Some(0),
                            transform_buffer: None,
                            transform_buffer_offset: None,
                        },
                    ]),
                },
            )
            .collect();
        if let Some(ref tlas) = inst_new_tlas {
            encoder.build_acceleration_structures(entries.iter(), std::iter::once(&**tlas));
        } else if !entries.is_empty() {
            encoder.build_acceleration_structures(entries.iter(), std::iter::empty());
        }
        inst_new_tlas
    };

    let Some(tlas) = final_tlas else {
        finish_trace_encoder(gpu, encoder, batched, scopes, None, 0, temp_buffers)?;
        return dispatch_clear_misses(
            gpu,
            &pipelines.clear_layout,
            &pipelines.clear,
            ray_count,
            domain,
            scope,
        );
    };

    // Phase 4: Optional active-ray stream compaction + indirect hardware RT trace!
    let out_bytes = (ray_count as u64 * HIT_WORDS as u64 * 4).max(16);
    let output_buffer = gpu.spare().take(
        &gpu.device,
        out_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let (active_rays_buf, indirect_args_opt) = if let Some(cu) = compact_u {
        let (act, ind) = encode_compact_rays(
            gpu,
            &pipelines.compact,
            &mut encoder,
            cu,
            &rays_buf,
            &output_buffer,
            ray_count,
            timing_set.as_ref(),
            &mut opened_ts,
            &mut temp_buffers,
        );
        (act, Some(ind))
    } else {
        (pipelines.empty_buf.clone(), None)
    };

    let trace_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("entelechy trace hw"),
        layout: &pipelines.trace_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: trace_u.expect("trace uniform").binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: tlas.as_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: rays_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: verts_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: tris_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: meshes_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: insts_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: active_rays_buf.as_entire_binding(),
            },
        ],
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("entelechy trace hw"),
            timestamp_writes: timing_set
                .as_ref()
                .map(|t| t.compute((!opened_ts).then_some(0), Some(1))),
        });
        pass.set_pipeline(&pipelines.trace);
        pass.set_bind_group(0, &trace_group, &[]);
        if let Some(ref ind) = indirect_args_opt {
            pass.dispatch_workgroups_indirect(ind, 0);
        } else {
            let (gx, gy) = wgsl::folded(ray_count.div_ceil(TRACE_WORKGROUP).max(1));
            pass.dispatch_workgroups(gx, gy, 1);
        }
    }
    finish_trace_encoder(
        gpu,
        encoder,
        batched,
        scopes,
        timing_set,
        est_ops,
        temp_buffers,
    )?;
    let dev = Box::new(DeviceImage::new(
        output_buffer,
        Some(gpu.spare().clone()),
        None,
    ));
    Ok(vec![scope.adopt_resident(hit_schema(), domain, dev)?])
}

fn flatten_bvh_nodes(bvh: &BuiltBvh, out: &mut Vec<u32>) {
    for node in &bvh.nodes {
        out.extend_from_slice(&node.to_words());
    }
}

enum SingleLbvhPlan {
    Reuse,
    Refit,
    FullBuild,
}

fn trace_bvh(
    gpu: &WgpuBackend,
    pipelines_lock: &OnceLock<Result<BvhTracePipelines, String>>,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
    op: NativeOp,
) -> Result<Vec<Handle>, ExecutionError> {
    let pipelines = match pipelines_lock.get_or_init(|| build_bvh_pipelines(&gpu.device, gpu.wait))
    {
        Ok(p) => p,
        Err(err) => return Err(ExecutionError::Device(err.clone())),
    };
    let _guard = gpu
        .lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let domain = invocation.domain;
    let ray_count = domain.count() as u32;
    let eval_now = gpu.trace_cache.current_eval();

    let (rays_h, verts_h, tris_h, meshes_h, insts_h, flags, mask, instanced) = match op {
        NativeOp::Trace => {
            let inp = check_trace(invocation, scope)?;
            (
                inp.rays,
                inp.vertices,
                inp.triangles,
                None,
                None,
                inp.flags,
                0xFFu8,
                false,
            )
        }
        NativeOp::TraceInstances => {
            let inp = check_trace_instances(invocation, scope)?;
            (
                inp.rays,
                inp.vertices,
                inp.triangles,
                Some(inp.meshes),
                Some(inp.instances),
                inp.flags,
                inp.mask,
                true,
            )
        }
        _ => unreachable!("only Trace and TraceInstances enter trace::trace"),
    };

    let v_count = scope.size(verts_h)?.count() as u32;
    let t_count = scope.size(tris_h)?.count() as u32;
    if v_count == 0 || t_count == 0 || mask == 0 {
        return dispatch_clear_misses(
            gpu,
            &pipelines.clear_layout,
            &pipelines.clear,
            ray_count,
            domain,
            scope,
        );
    }

    let rays_buf = gpu.bind_input(scope, rays_h)?;
    let verts_buf = gpu.bind_input(scope, verts_h)?;
    let tris_buf = gpu.bind_input(scope, tris_h)?;
    let verts_id = image_id(scope, verts_h).unwrap_or(0);
    let tris_id = image_id(scope, tris_h).unwrap_or(0);

    // Phase 1: Single-mesh uses 100% GPU-resident Zero-Readback LBVH + 1-pass dynamic refit!
    // Instanced multi-mesh builds/caches static mesh SAH BVHs + TLAS before opening batch encoder.
    let mut lbvh_plan = None;
    let mut inst_bufs = None;
    if !instanced {
        let key = (tris_id, v_count, t_count);
        let cache = gpu
            .trace_cache
            .inner
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        lbvh_plan = Some(match cache.lbvh_single.get(&key) {
            Some(mesh) if mesh.built_verts_id == verts_id && verts_id != 0 => SingleLbvhPlan::Reuse,
            Some(mesh) if mesh.refits_since_build < LBVH_REFIT_PERIOD => SingleLbvhPlan::Refit,
            _ => SingleLbvhPlan::FullBuild,
        });
    } else {
        let mh = meshes_h.expect("instanced trace has meshes");
        let ih = insts_h.expect("instanced trace has instances");
        if scope.size(mh)?.count() == 0 || scope.size(ih)?.count() == 0 {
            return dispatch_clear_misses(
                gpu,
                &pipelines.clear_layout,
                &pipelines.clear,
                ray_count,
                domain,
                scope,
            );
        }
        let _ = gpu.bind_input(scope, mh)?;
        let _ = gpu.bind_input(scope, ih)?;
        let meshes_id = image_id(scope, mh).unwrap_or(0);
        let insts_id = image_id(scope, ih).unwrap_or(0);
        let mesh_key = (verts_id, tris_id, meshes_id);
        let cached_meshes = {
            let mut cache = gpu
                .trace_cache
                .inner
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            cache.bvh_meshes.get_mut(&mesh_key).map(|entry| {
                entry.last_eval = eval_now;
                (
                    entry.nodes.clone(),
                    entry.ordered.clone(),
                    entry.mesh_descs.clone(),
                    entry.bounds.clone(),
                )
            })
        };
        let (nodes_buf, ord_buf, descs_buf, mesh_bounds) = if let Some(hit) = cached_meshes {
            hit
        } else {
            let v_words = read_words(gpu, scope, verts_h)?;
            let t_words = read_words(gpu, scope, tris_h)?;
            let m_words = read_words(gpu, scope, mh)?;
            let slices = trace::parse_meshes(
                &m_words,
                v_words.len() / POSITION_WORDS,
                t_words.len() / TRIANGLE_WORDS,
            );
            let mut all_nodes = Vec::new();
            let mut all_ord = Vec::new();
            let mut desc_words = Vec::with_capacity(slices.len() * 8);
            let mut bounds = Vec::with_capacity(slices.len());
            for &slice in &slices {
                let bvh = trace::build_mesh_bvh(&v_words, &t_words, slice);
                let node_base = (all_nodes.len() / 8) as u32;
                let ord_base = all_ord.len() as u32;
                let node_cnt = bvh.nodes.len() as u32;
                if bvh.nodes.is_empty() {
                    bounds.push(None);
                } else {
                    bounds.push(Some((bvh.bounds_min, bvh.bounds_max)));
                    flatten_bvh_nodes(&bvh, &mut all_nodes);
                    all_ord.extend_from_slice(&bvh.ordered);
                }
                desc_words.extend_from_slice(&[
                    slice.vertex_offset as u32,
                    slice.vertex_count as u32,
                    slice.triangle_offset as u32,
                    node_base,
                    ord_base,
                    node_cnt,
                    0,
                    0,
                ]);
            }
            let nodes_buf = upload_words_buf(
                &gpu.device,
                &gpu.queue,
                "entelechy bvh meshes nodes",
                &all_nodes,
            );
            let ord_buf = upload_words_buf(
                &gpu.device,
                &gpu.queue,
                "entelechy bvh meshes ord",
                &all_ord,
            );
            let descs_buf = upload_words_buf(
                &gpu.device,
                &gpu.queue,
                "entelechy bvh mesh descs",
                &desc_words,
            );
            let mut cache = gpu
                .trace_cache
                .inner
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            evict_oldest_before(&mut cache.bvh_meshes, eval_now, |v| v.last_eval);
            cache.bvh_meshes.insert(
                mesh_key,
                CachedMeshSetBvh {
                    nodes: nodes_buf.clone(),
                    ordered: ord_buf.clone(),
                    mesh_descs: descs_buf.clone(),
                    bounds: bounds.clone(),
                    last_eval: eval_now,
                },
            );
            (nodes_buf, ord_buf, descs_buf, bounds)
        };

        let tlas_key = (verts_id, tris_id, meshes_id, insts_id);
        let cached_tlas = {
            let mut cache = gpu
                .trace_cache
                .inner
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            cache.bvh_instanced.get_mut(&tlas_key).map(|entry| {
                entry.last_eval = eval_now;
                (
                    entry.tlas_nodes.clone(),
                    entry.tlas_ordered.clone(),
                    entry.inst_descs.clone(),
                    entry.tlas_node_count,
                )
            })
        };
        let (tlas_nodes_buf, tlas_ord_buf, inst_descs_buf, tlas_node_count) =
            if let Some(hit) = cached_tlas {
                hit
            } else {
                let inst_words = read_words(gpu, scope, ih)?;
                let prepared = trace::parse_instances(&inst_words, &mesh_bounds);
                let aabbs: Vec<(u32, [f32; 3], [f32; 3])> = prepared
                    .iter()
                    .enumerate()
                    .map(|(i, inst)| (i as u32, inst.world_min, inst.world_max))
                    .collect();
                let tlas_bvh = trace::build_aabb_bvh(&aabbs);
                let mut tlas_node_words = Vec::with_capacity(tlas_bvh.nodes.len() * 8);
                flatten_bvh_nodes(&tlas_bvh, &mut tlas_node_words);
                let mut inst_desc_words = Vec::with_capacity(prepared.len() * 24);
                for inst in &prepared {
                    for row in &inst.inv_transform {
                        for &v in row {
                            inst_desc_words.push(v.to_bits());
                        }
                    }
                    inst_desc_words.extend_from_slice(&[
                        inst.world_min[0].to_bits(),
                        inst.world_min[1].to_bits(),
                        inst.world_min[2].to_bits(),
                        inst.row,
                        inst.world_max[0].to_bits(),
                        inst.world_max[1].to_bits(),
                        inst.world_max[2].to_bits(),
                        inst.mesh,
                        inst.custom as u32,
                        u32::from(inst.mask),
                        u32::from(inst.negative_det),
                        0,
                    ]);
                }
                let tn_buf = upload_words_buf(
                    &gpu.device,
                    &gpu.queue,
                    "entelechy bvh tlas nodes",
                    &tlas_node_words,
                );
                let to_buf = upload_words_buf(
                    &gpu.device,
                    &gpu.queue,
                    "entelechy bvh tlas ord",
                    &tlas_bvh.ordered,
                );
                let id_buf = upload_words_buf(
                    &gpu.device,
                    &gpu.queue,
                    "entelechy bvh inst descs",
                    &inst_desc_words,
                );
                let tlas_cnt = tlas_bvh.nodes.len() as u32;
                let mut cache = gpu
                    .trace_cache
                    .inner
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                evict_oldest_before(&mut cache.bvh_instanced, eval_now, |v| v.last_eval);
                cache.bvh_instanced.insert(
                    tlas_key,
                    CachedInstancedBvh {
                        tlas_nodes: tn_buf.clone(),
                        tlas_ordered: to_buf.clone(),
                        inst_descs: id_buf.clone(),
                        tlas_node_count: tlas_cnt,
                        last_eval: eval_now,
                    },
                );
                (tn_buf, to_buf, id_buf, tlas_cnt)
            };
        if tlas_node_count == 0 {
            return dispatch_clear_misses(
                gpu,
                &pipelines.clear_layout,
                &pipelines.clear,
                ray_count,
                domain,
                scope,
            );
        }
        inst_bufs = Some((
            nodes_buf,
            ord_buf,
            descs_buf,
            tlas_nodes_buf,
            tlas_ord_buf,
            inst_descs_buf,
            tlas_node_count,
        ));
    }

    let use_compact = !instanced && ray_count >= COMPACT_RAY_THRESHOLD && (flags & 6) != 6;
    let est_ops = u64::from(ray_count)
        .saturating_mul(160)
        .saturating_add(u64::from(t_count).saturating_mul(32));
    if gpu.wait.parks_with_host()
        && !gpu.measured()
        && gpu.page_queue.full(est_ops, gpu.page_pace_ops_budget())
    {
        gpu.batch.flush()?;
        gpu.drain_page()?;
    }
    let batched = gpu.batch.can_batch(gpu.measured(), &gpu.frames);
    if !batched {
        gpu.batch.flush()?;
    }

    // Phase 2: Allocate all uniforms atomically before opening encoder.
    let full_payloads = trace_lbvh::full_build_uniform_payloads(v_count, t_count);
    let refit_payload = trace_lbvh::refit_uniform_payload(v_count, t_count);
    let mut compact_payload = [0u8; 16];
    compact_payload[..4].copy_from_slice(&ray_count.to_le_bytes());
    compact_payload[4..8].copy_from_slice(&(flags as u32).to_le_bytes());
    let tlas_node_count = inst_bufs.as_ref().map_or(0u32, |b| b.6);
    let trace_setup: [u32; 8] = [
        ray_count,
        v_count,
        t_count,
        flags as u32,
        u32::from(mask),
        u32::from(instanced),
        tlas_node_count,
        u32::from(use_compact),
    ];
    let mut trace_payload = [0u8; 32];
    for (slot, w) in trace_payload.chunks_exact_mut(4).zip(trace_setup) {
        slot.copy_from_slice(&w.to_le_bytes());
    }

    let mut all_payloads: Vec<&[u8]> = Vec::new();
    let lbvh_u_count = match lbvh_plan {
        Some(SingleLbvhPlan::FullBuild) => {
            for p in &full_payloads {
                all_payloads.push(p.as_slice());
            }
            8usize
        }
        Some(SingleLbvhPlan::Refit) => {
            all_payloads.push(refit_payload.as_slice());
            1usize
        }
        _ => 0usize,
    };
    if use_compact {
        all_payloads.push(&compact_payload);
    }
    all_payloads.push(&trace_payload);

    let mut temp_buffers = Vec::new();
    let uniform_slices = alloc_uniform_slices(gpu, batched, &all_payloads, &mut temp_buffers)?;
    let compact_u = use_compact.then(|| &uniform_slices[lbvh_u_count]);
    let trace_u = &uniform_slices[lbvh_u_count + usize::from(use_compact)];

    let scopes = (!batched).then(|| Scopes::push(&gpu.device));
    let timing_set = gpu.timing_set(2);
    let mut opened_ts = false;
    let mut encoder = if batched {
        gpu.batch.take_or_open()
    } else {
        gpu.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("entelechy trace bvh"),
            })
    };

    // Phase 3: Encode Full-GPU LBVH build or 1-pass dynamic refit (zero CPU readback!).
    let (bvh_nodes_buf, bvh_ord_buf, mesh_descs_buf, tlas_nodes_base, tlas_ord_buf, inst_descs_buf) =
        if let Some(plan) = lbvh_plan {
            let key = (tris_id, v_count, t_count);
            let mut cache = gpu
                .trace_cache
                .inner
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            if !cache.lbvh_single.contains_key(&key) {
                evict_oldest_before(&mut cache.lbvh_single, eval_now, |v| v.last_eval);
                cache
                    .lbvh_single
                    .insert(key, GpuLbvhMesh::allocate(&gpu.device, t_count, eval_now));
            }
            let mesh = cache
                .lbvh_single
                .get_mut(&key)
                .expect("lbvh entry allocated");
            match plan {
                SingleLbvhPlan::Reuse => {
                    mesh.last_eval = eval_now;
                }
                SingleLbvhPlan::Refit => {
                    mesh.encode_refit(
                        &gpu.device,
                        &pipelines.lbvh,
                        &mut encoder,
                        &verts_buf,
                        &tris_buf,
                        &uniform_slices[0],
                        verts_id,
                        eval_now,
                        timing_set.as_ref(),
                        &mut opened_ts,
                    );
                }
                SingleLbvhPlan::FullBuild => {
                    mesh.encode_full_build(
                        &gpu.device,
                        &gpu.spare(),
                        &pipelines.lbvh,
                        &mut encoder,
                        &verts_buf,
                        &tris_buf,
                        &uniform_slices[..8],
                        verts_id,
                        eval_now,
                        &mut temp_buffers,
                        timing_set.as_ref(),
                        &mut opened_ts,
                    );
                }
            }
            (
                mesh.nodes.clone(),
                mesh.ordered.clone(),
                mesh.mesh_desc.clone(),
                pipelines.empty_buf.clone(),
                pipelines.empty_buf.clone(),
                pipelines.empty_buf.clone(),
            )
        } else {
            let (n, o, d, tn, to, id, _) = inst_bufs.expect("instanced buffers");
            (n, o, d, tn, to, id)
        };

    // Phase 4: Optional active-ray stream compaction + indirect BVH trace!
    let out_bytes = (ray_count as u64 * HIT_WORDS as u64 * 4).max(16);
    let output_buffer = gpu.spare().take(
        &gpu.device,
        out_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let (binding8_buf, indirect_args_opt) = if let Some(cu) = compact_u {
        let (act, ind) = encode_compact_rays(
            gpu,
            &pipelines.compact,
            &mut encoder,
            cu,
            &rays_buf,
            &output_buffer,
            ray_count,
            timing_set.as_ref(),
            &mut opened_ts,
            &mut temp_buffers,
        );
        (act, Some(ind))
    } else {
        (tlas_nodes_base, None)
    };

    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("entelechy trace bvh"),
        layout: &pipelines.trace_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: trace_u.binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rays_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: verts_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: tris_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: bvh_nodes_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: bvh_ord_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: mesh_descs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: binding8_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: tlas_ord_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: inst_descs_buf.as_entire_binding(),
            },
        ],
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("entelechy trace bvh"),
            timestamp_writes: timing_set
                .as_ref()
                .map(|t| t.compute((!opened_ts).then_some(0), Some(1))),
        });
        pass.set_pipeline(&pipelines.trace);
        pass.set_bind_group(0, &group, &[]);
        if let Some(ref ind) = indirect_args_opt {
            pass.dispatch_workgroups_indirect(ind, 0);
        } else {
            let (gx, gy) = wgsl::folded(ray_count.div_ceil(TRACE_WORKGROUP).max(1));
            pass.dispatch_workgroups(gx, gy, 1);
        }
    }
    finish_trace_encoder(
        gpu,
        encoder,
        batched,
        scopes,
        timing_set,
        est_ops,
        temp_buffers,
    )?;
    let dev = Box::new(DeviceImage::new(
        output_buffer,
        Some(gpu.spare().clone()),
        None,
    ));
    Ok(vec![scope.adopt_resident(hit_schema(), domain, dev)?])
}
