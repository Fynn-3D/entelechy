//! Native ray tracing (`native.trace@1` and `native.trace@2`) contract,
//! deterministic SAH-binned BVH builder, and CPU/JIT reference traversal.
//!
//! It is the fast path of the `geometry` package's `geometry.trace@1` and
//! `geometry.trace_instances@1` Kernels: shading, ray generation, GI, shadows,
//! and denoising live in `.ent` Kernels; acceleration-structure traversal is
//! device emulation.
//!
//! Contracts:
//! - `native.trace@1` (single triangle mesh):
//!   - input 0, `rays`: `Rows<{ origin: Vec3; tmin: Float; dir: Vec3; tmax: Float; }>` (8 words)
//!   - input 1, `vertices`: `Rows<Vec3>` (3 words)
//!   - input 2, `triangles`: `Rows<{ a: Int; b: Int; c: Int; }>` (3 words)
//!   - param 0, `flags`: `Int = 0` (`1` terminate on first hit, `2` cull back, `4` cull front)
//!   - output 0, `hits`: `Rows<{ t: Float; bary: Vec2; triangle: Int; instance: Int; custom: Int; front: Bool; }>` (7 words)
//!
//! - `native.trace@2` (instanced multi-mesh TLAS):
//!   - input 0, `rays`: `Rows<Ray>`
//!   - input 1, `vertices`: `Rows<Vec3>`
//!   - input 2, `triangles`: `Rows<Triangle>` (indices local to each mesh's `vertex_offset`)
//!   - input 3, `meshes`: `Rows<{ vertex_offset: Int; vertex_count: Int; triangle_offset: Int; triangle_count: Int; }>` (4 words)
//!   - input 4, `instances`: `Rows<{ row0: Vec4; row1: Vec4; row2: Vec4; mesh: Int; custom: Int; mask: Int; }>` (15 words, 3x4 row-major object-to-world affine transform)
//!   - param 0, `flags`: `Int = 0`
//!   - param 1, `mask`: `Int = 255` (8-bit ray visibility mask)
//!   - output 0, `hits`: `Rows<Hit>`

use entelechy_abi::{Handle, Invocation, Scalar, Schema, Scope, Shape, Value};

use crate::ExecutionError;

/// Words per ray: `origin: Vec3, tmin: Float, dir: Vec3, tmax: Float`.
pub const RAY_WORDS: usize = 8;
/// Words per vertex position: `Vec3`.
pub const POSITION_WORDS: usize = 3;
/// Words per triangle: `a: Int, b: Int, c: Int`.
pub const TRIANGLE_WORDS: usize = 3;
/// Words per mesh slice: `vertex_offset, vertex_count, triangle_offset, triangle_count`.
pub const MESH_WORDS: usize = 4;
/// Words per instance: `row0: Vec4, row1: Vec4, row2: Vec4, mesh: Int, custom: Int, mask: Int`.
pub const INSTANCE_WORDS: usize = 15;
/// Words per hit: `t: Float, bary: Vec2, triangle: Int, instance: Int, custom: Int, front: Bool`.
pub const HIT_WORDS: usize = 7;

/// Flag bit: stop traversal at the first valid intersection (shadow / occlusion rays).
pub const FLAG_TERMINATE_ON_FIRST_HIT: i32 = 1;
/// Flag bit: ignore back-facing hits (`dot(obj_dir, cross(v1 - v0, v2 - v0)) >= 0`).
pub const FLAG_CULL_BACK: i32 = 2;
/// Flag bit: ignore front-facing hits (`dot(obj_dir, cross(v1 - v0, v2 - v0)) < 0`).
pub const FLAG_CULL_FRONT: i32 = 4;

/// Words of a miss hit: `t = -1.0, bary = (0, 0), triangle = -1, instance = -1, custom = 0, front = false`.
pub const MISS_WORDS: [u32; HIT_WORDS] = [
    (-1.0f32).to_bits(),
    0,
    0,
    (-1i32) as u32,
    (-1i32) as u32,
    0,
    0,
];

/// Schema of `rays`: `origin: Vec3, tmin: Float, dir: Vec3, tmax: Float`.
pub fn ray_schema() -> Schema {
    Schema::new(vec![Scalar::Float; RAY_WORDS])
}

/// Schema of `vertices`: `Vec3`.
pub fn position_schema() -> Schema {
    Schema::new(vec![Scalar::Float; POSITION_WORDS])
}

/// Schema of `triangles`: `a: Int, b: Int, c: Int`.
pub fn triangle_schema() -> Schema {
    Schema::new(vec![Scalar::Int; TRIANGLE_WORDS])
}

/// Schema of `meshes`: four `Int` slice bounds.
pub fn mesh_schema() -> Schema {
    Schema::new(vec![Scalar::Int; MESH_WORDS])
}

/// Schema of `instances`: twelve `Float` affine matrix words plus three `Int` fields.
pub fn instance_schema() -> Schema {
    let mut scalars = vec![Scalar::Float; 12];
    scalars.extend([Scalar::Int; 3]);
    Schema::new(scalars)
}

/// Schema of `hits`: `t: Float, bary: Vec2, triangle: Int, instance: Int, custom: Int, front: Bool`.
pub fn hit_schema() -> Schema {
    Schema::new(vec![
        Scalar::Float,
        Scalar::Float,
        Scalar::Float,
        Scalar::Int,
        Scalar::Int,
        Scalar::Int,
        Scalar::Bool,
    ])
}

/// Checked handles and parameters for `native.trace@1`.
#[derive(Clone, Copy, Debug)]
pub struct TraceInputs {
    /// Ray rows.
    pub rays: Handle,
    /// Vertex position rows.
    pub vertices: Handle,
    /// Triangle index rows.
    pub triangles: Handle,
    /// Ray tracing flags bitmask.
    pub flags: i32,
}

/// Checked handles and parameters for `native.trace@2`.
#[derive(Clone, Copy, Debug)]
pub struct TraceInstancesInputs {
    /// Ray rows.
    pub rays: Handle,
    /// Shared vertex position rows.
    pub vertices: Handle,
    /// Shared triangle index rows.
    pub triangles: Handle,
    /// Per-mesh slice rows.
    pub meshes: Handle,
    /// Per-instance transform and metadata rows.
    pub instances: Handle,
    /// Ray tracing flags bitmask.
    pub flags: i32,
    /// 8-bit visibility mask tested against each instance's `mask`.
    pub mask: u8,
}

/// Validates an invocation against `native.trace@1`.
pub fn check_trace(
    invocation: &Invocation,
    scope: &Scope<'_>,
) -> Result<TraceInputs, ExecutionError> {
    if invocation.inputs.len() != 3 {
        return Err(ExecutionError::InputCount {
            expected: 3,
            actual: invocation.inputs.len(),
        });
    }
    let (rays, vertices, triangles) = (
        invocation.inputs[0],
        invocation.inputs[1],
        invocation.inputs[2],
    );
    check_common_trace(invocation, scope, rays, vertices, triangles)?;
    let flags = match invocation.params.first() {
        Some(Value::Int(v)) => *v,
        _ => 0,
    };
    Ok(TraceInputs {
        rays,
        vertices,
        triangles,
        flags,
    })
}

/// Validates an invocation against `native.trace@2`.
pub fn check_trace_instances(
    invocation: &Invocation,
    scope: &Scope<'_>,
) -> Result<TraceInstancesInputs, ExecutionError> {
    if invocation.inputs.len() != 5 {
        return Err(ExecutionError::InputCount {
            expected: 5,
            actual: invocation.inputs.len(),
        });
    }
    let (rays, vertices, triangles, meshes, instances) = (
        invocation.inputs[0],
        invocation.inputs[1],
        invocation.inputs[2],
        invocation.inputs[3],
        invocation.inputs[4],
    );
    check_common_trace(invocation, scope, rays, vertices, triangles)?;
    let contract = |message: &str| ExecutionError::Device(format!("trace: {message}"));
    if scope.schema(meshes)? != mesh_schema() {
        return Err(contract(
            "`meshes` must be rows of `vertex_offset: Int, vertex_count: Int, triangle_offset: Int, triangle_count: Int`",
        ));
    }
    if scope.size(meshes)?.shape() != Shape::Rows {
        return Err(contract("`meshes` must be rows"));
    }
    if scope.schema(instances)? != instance_schema() {
        return Err(contract(
            "`instances` must be rows of `row0: Vec4, row1: Vec4, row2: Vec4, mesh: Int, custom: Int, mask: Int`",
        ));
    }
    if scope.size(instances)?.shape() != Shape::Rows {
        return Err(contract("`instances` must be rows"));
    }
    let flags = match invocation.params.first() {
        Some(Value::Int(v)) => *v,
        _ => 0,
    };
    let mask = match invocation.params.get(1) {
        Some(Value::Int(v)) => (*v & 0xFF) as u8,
        _ => 0xFF,
    };
    Ok(TraceInstancesInputs {
        rays,
        vertices,
        triangles,
        meshes,
        instances,
        flags,
        mask,
    })
}

fn check_common_trace(
    invocation: &Invocation,
    scope: &Scope<'_>,
    rays: Handle,
    vertices: Handle,
    triangles: Handle,
) -> Result<(), ExecutionError> {
    let contract = |message: String| ExecutionError::Device(format!("trace: {message}"));
    if scope.schema(rays)? != ray_schema() {
        return Err(contract(
            "`rays` must be rows of `origin: Vec3, tmin: Float, dir: Vec3, tmax: Float`".into(),
        ));
    }
    let ray_extent = scope.size(rays)?;
    if ray_extent.shape() != Shape::Rows {
        return Err(contract("`rays` must be rows".into()));
    }
    if scope.schema(vertices)? != position_schema() {
        return Err(contract("`vertices` must be rows of `Vec3`".into()));
    }
    if scope.size(vertices)?.shape() != Shape::Rows {
        return Err(contract("`vertices` must be rows".into()));
    }
    if scope.schema(triangles)? != triangle_schema() {
        return Err(contract(
            "`triangles` must be rows of three `Int` indices".into(),
        ));
    }
    if scope.size(triangles)?.shape() != Shape::Rows {
        return Err(contract("`triangles` must be rows".into()));
    }
    if invocation.domain.shape() != Shape::Rows {
        return Err(ExecutionError::DomainShape {
            expected: Shape::Rows,
            actual: invocation.domain.shape(),
        });
    }
    if invocation.domain.is_empty() {
        return Err(ExecutionError::EmptyDomain);
    }
    if invocation.domain != ray_extent {
        return Err(contract(format!(
            "`hits` has {} row(s), but `rays` has {}",
            invocation.domain.count(),
            ray_extent.count()
        )));
    }
    scope.can_hold(&[(&hit_schema(), invocation.domain)])?;
    Ok(())
}

/// Flat 32-byte BVH node shared between CPU/JIT reference traversal and GPU
/// Compute BVH fallback shaders (`8 * u32` words).
#[derive(Clone, Copy, Debug)]
pub struct BvhNode {
    /// Axis-aligned bounding box minimum corner.
    pub min: [f32; 3],
    /// Left child index when `count == 0`, else first primitive index in `ordered`.
    pub left_or_first: u32,
    /// Axis-aligned bounding box maximum corner.
    pub max: [f32; 3],
    /// `0` for interior node (children at `left_or_first` and `left_or_first + 1`),
    /// `> 0` for leaf containing `count` primitives.
    pub count: u32,
}

impl BvhNode {
    /// Encodes the node into 8 `u32` words for GPU storage upload.
    pub fn to_words(self) -> [u32; 8] {
        [
            self.min[0].to_bits(),
            self.min[1].to_bits(),
            self.min[2].to_bits(),
            self.left_or_first,
            self.max[0].to_bits(),
            self.max[1].to_bits(),
            self.max[2].to_bits(),
            self.count,
        ]
    }
}

/// A built flat BVH: nodes (`nodes[0]` is root when non-empty) and reordered
/// original primitive indices referenced by leaf nodes.
#[derive(Clone, Debug, Default)]
pub struct BuiltBvh {
    /// Flat array of BVH nodes (`nodes[0]` is root when non-empty).
    pub nodes: Vec<BvhNode>,
    /// Original primitive indices grouped per leaf.
    pub ordered: Vec<u32>,
    /// Whole-tree AABB minimum (valid when `!nodes.is_empty()`).
    pub bounds_min: [f32; 3],
    /// Whole-tree AABB maximum (valid when `!nodes.is_empty()`).
    pub bounds_max: [f32; 3],
}

/// Validated mesh slice bounds inside shared vertex/triangle arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshSlice {
    /// Starting vertex index.
    pub vertex_offset: usize,
    /// Number of vertices belonging to this mesh.
    pub vertex_count: usize,
    /// Starting triangle index.
    pub triangle_offset: usize,
    /// Number of triangles belonging to this mesh.
    pub triangle_count: usize,
}

/// Parsed and validated instance with precomputed inverse object-to-world matrix.
#[derive(Clone, Copy, Debug)]
pub struct PreparedInstance {
    /// Original row index in the `instances` Field.
    pub row: u32,
    /// Index into `meshes`.
    pub mesh: u32,
    /// User-supplied 32-bit custom identifier.
    pub custom: i32,
    /// 8-bit visibility mask.
    pub mask: u8,
    /// Forward 3x4 row-major object-to-world transform.
    pub transform: [f32; 12],
    /// Inverse 3x4 row-major world-to-object transform.
    pub inv_transform: [[f32; 4]; 3],
    /// Whether the linear 3x3 part has negative determinant (flips triangle winding in world space).
    pub negative_det: bool,
    /// World-space AABB minimum of the transformed mesh bounds.
    pub world_min: [f32; 3],
    /// World-space AABB maximum of the transformed mesh bounds.
    pub world_max: [f32; 3],
}

#[derive(Clone, Copy)]
struct PrimInfo {
    index: u32,
    min: [f32; 3],
    max: [f32; 3],
    centroid: [f32; 3],
}

#[inline]
fn f32_at(words: &[u32], idx: usize) -> f32 {
    f32::from_bits(words[idx])
}

#[inline]
fn vec3_at(words: &[u32], base: usize) -> [f32; 3] {
    [
        f32_at(words, base),
        f32_at(words, base + 1),
        f32_at(words, base + 2),
    ]
}

#[inline]
fn finite3(v: [f32; 3]) -> bool {
    v[0].is_finite() && v[1].is_finite() && v[2].is_finite()
}

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn min3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])]
}

#[inline]
fn max3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])]
}

#[inline]
fn surface_area(min: [f32; 3], max: [f32; 3]) -> f32 {
    let dx = (max[0] - min[0]).max(0.0);
    let dy = (max[1] - min[1]).max(0.0);
    let dz = (max[2] - min[2]).max(0.0);
    dx * dy + dy * dz + dz * dx
}

/// Clamps raw triangle vertex indices into `0..vertex_count` (local to the mesh).
#[inline]
pub fn clamped_triangle_indices(
    triangles: &[u32],
    tri_index: usize,
    vertex_count: usize,
) -> [usize; 3] {
    let max_v = vertex_count.saturating_sub(1) as i64;
    let base = tri_index * TRIANGLE_WORDS;
    let clamp = |off: usize| -> usize {
        let raw = triangles[base + off] as i32 as i64;
        raw.clamp(0, max_v) as usize
    };
    [clamp(0), clamp(1), clamp(2)]
}

/// Builds a flat SAH-binned BVH over one mesh slice (`vertices`, `triangles`).
///
/// Triangles with non-finite vertices or zero area are skipped so hardware and
/// software BVHs never store degenerate bounds.
pub fn build_mesh_bvh(vertices: &[u32], triangles: &[u32], slice: MeshSlice) -> BuiltBvh {
    if slice.vertex_count == 0 || slice.triangle_count == 0 {
        return BuiltBvh::default();
    }
    let total_verts = vertices.len() / POSITION_WORDS;
    let total_tris = triangles.len() / TRIANGLE_WORDS;
    let v_off = slice.vertex_offset.min(total_verts);
    let v_cnt = slice.vertex_count.min(total_verts.saturating_sub(v_off));
    let t_off = slice.triangle_offset.min(total_tris);
    let t_cnt = slice.triangle_count.min(total_tris.saturating_sub(t_off));
    if v_cnt == 0 || t_cnt == 0 {
        return BuiltBvh::default();
    }
    let mut prims = Vec::with_capacity(t_cnt);
    for local_t in 0..t_cnt {
        let [ia, ib, ic] = clamped_triangle_indices(triangles, t_off + local_t, v_cnt);
        let v0 = vec3_at(vertices, (v_off + ia) * POSITION_WORDS);
        let v1 = vec3_at(vertices, (v_off + ib) * POSITION_WORDS);
        let v2 = vec3_at(vertices, (v_off + ic) * POSITION_WORDS);
        if !(finite3(v0) && finite3(v1) && finite3(v2)) {
            continue;
        }
        let n = cross3(sub3(v1, v0), sub3(v2, v0));
        if !(finite3(n) && dot3(n, n) > 0.0) {
            continue;
        }
        let min = min3(min3(v0, v1), v2);
        let max = max3(max3(v0, v1), v2);
        let centroid = [
            0.5 * (min[0] + max[0]),
            0.5 * (min[1] + max[1]),
            0.5 * (min[2] + max[2]),
        ];
        prims.push(PrimInfo {
            index: local_t as u32,
            min,
            max,
            centroid,
        });
    }
    build_bvh_from_prims(prims)
}

/// Builds a flat SAH-binned BVH over world-space AABBs (used for TLAS instances).
pub fn build_aabb_bvh(aabbs: &[(u32, [f32; 3], [f32; 3])]) -> BuiltBvh {
    let prims: Vec<PrimInfo> = aabbs
        .iter()
        .filter(|(_, mn, mx)| finite3(*mn) && finite3(*mx))
        .map(|&(index, min, max)| PrimInfo {
            index,
            min,
            max,
            centroid: [
                0.5 * (min[0] + max[0]),
                0.5 * (min[1] + max[1]),
                0.5 * (min[2] + max[2]),
            ],
        })
        .collect();
    build_bvh_from_prims(prims)
}

fn build_bvh_from_prims(mut prims: Vec<PrimInfo>) -> BuiltBvh {
    if prims.is_empty() {
        return BuiltBvh::default();
    }
    let count = prims.len();
    let mut nodes = Vec::with_capacity(count * 2);
    nodes.push(BvhNode {
        min: [0.0; 3],
        left_or_first: 0,
        max: [0.0; 3],
        count: 0,
    });
    build_node(&mut prims, &mut nodes, 0, 0, count);
    let root = nodes[0];
    let ordered = prims.into_iter().map(|p| p.index).collect();
    BuiltBvh {
        nodes,
        ordered,
        bounds_min: root.min,
        bounds_max: root.max,
    }
}

fn build_node(
    prims: &mut [PrimInfo],
    nodes: &mut Vec<BvhNode>,
    node_idx: usize,
    start: usize,
    end: usize,
) {
    let mut bmin = [f32::INFINITY; 3];
    let mut bmax = [f32::NEG_INFINITY; 3];
    let mut cmin = [f32::INFINITY; 3];
    let mut cmax = [f32::NEG_INFINITY; 3];
    for p in &prims[start..end] {
        bmin = min3(bmin, p.min);
        bmax = max3(bmax, p.max);
        cmin = min3(cmin, p.centroid);
        cmax = max3(cmax, p.centroid);
    }
    let count = end - start;
    const LEAF_MAX: usize = 4;
    if count <= LEAF_MAX {
        nodes[node_idx] = BvhNode {
            min: bmin,
            left_or_first: start as u32,
            max: bmax,
            count: count as u32,
        };
        return;
    }

    // 8-bin Surface Area Heuristic split across all 3 axes.
    const BINS: usize = 8;
    let parent_area = surface_area(bmin, bmax);
    let mut best_cost = f32::INFINITY;
    let mut best_axis = 0usize;
    let mut best_split = 0usize;
    let mut best_scale = 0.0f32;
    let mut best_cmin = 0.0f32;

    for axis in 0..3 {
        let extent = cmax[axis] - cmin[axis];
        if !(extent > 1e-7 && extent.is_finite()) {
            continue;
        }
        let scale = BINS as f32 / extent;
        let mut bin_min = [[f32::INFINITY; 3]; BINS];
        let mut bin_max = [[f32::NEG_INFINITY; 3]; BINS];
        let mut bin_count = [0u32; BINS];
        for p in &prims[start..end] {
            let b = (((p.centroid[axis] - cmin[axis]) * scale) as usize).min(BINS - 1);
            bin_count[b] += 1;
            bin_min[b] = min3(bin_min[b], p.min);
            bin_max[b] = max3(bin_max[b], p.max);
        }
        let mut left_area = [0.0f32; BINS - 1];
        let mut left_cnt = [0u32; BINS - 1];
        let mut lmin = [f32::INFINITY; 3];
        let mut lmax = [f32::NEG_INFINITY; 3];
        let mut lsum = 0u32;
        for i in 0..(BINS - 1) {
            lsum += bin_count[i];
            left_cnt[i] = lsum;
            if bin_count[i] > 0 {
                lmin = min3(lmin, bin_min[i]);
                lmax = max3(lmax, bin_max[i]);
            }
            left_area[i] = if lsum > 0 {
                surface_area(lmin, lmax)
            } else {
                0.0
            };
        }
        let mut rmin = [f32::INFINITY; 3];
        let mut rmax = [f32::NEG_INFINITY; 3];
        let mut rsum = 0u32;
        for i in (0..(BINS - 1)).rev() {
            rsum += bin_count[i + 1];
            if bin_count[i + 1] > 0 {
                rmin = min3(rmin, bin_min[i + 1]);
                rmax = max3(rmax, bin_max[i + 1]);
            }
            if left_cnt[i] > 0 && rsum > 0 {
                let cost =
                    left_cnt[i] as f32 * left_area[i] + rsum as f32 * surface_area(rmin, rmax);
                if cost < best_cost {
                    best_cost = cost;
                    best_axis = axis;
                    best_split = i + 1;
                    best_scale = scale;
                    best_cmin = cmin[axis];
                }
            }
        }
    }

    let leaf_cost = count as f32 * parent_area;
    let mid = if best_cost.is_finite() && (best_cost < leaf_cost || count > 8) {
        let mut left = start;
        let mut right = end;
        while left < right {
            let b = (((prims[left].centroid[best_axis] - best_cmin) * best_scale) as usize)
                .min(BINS - 1);
            if b < best_split {
                left += 1;
            } else {
                right -= 1;
                prims.swap(left, right);
            }
        }
        if left > start && left < end {
            left
        } else {
            start + count / 2
        }
    } else {
        // Fallback: split at median along longest AABB axis if still large.
        if count <= 8 {
            nodes[node_idx] = BvhNode {
                min: bmin,
                left_or_first: start as u32,
                max: bmax,
                count: count as u32,
            };
            return;
        }
        start + count / 2
    };

    let left_child = nodes.len();
    nodes.push(BvhNode {
        min: [0.0; 3],
        left_or_first: 0,
        max: [0.0; 3],
        count: 0,
    });
    nodes.push(BvhNode {
        min: [0.0; 3],
        left_or_first: 0,
        max: [0.0; 3],
        count: 0,
    });
    nodes[node_idx] = BvhNode {
        min: bmin,
        left_or_first: left_child as u32,
        max: bmax,
        count: 0,
    };
    build_node(prims, nodes, left_child, start, mid);
    build_node(prims, nodes, left_child + 1, mid, end);
}

/// Inverts a 3x4 row-major affine matrix `[row0, row1, row2]` in double
/// precision, returning `None` if the linear 3x3 block is singular or any
/// entry is non-finite.
pub fn invert_affine_3x4(row0: [f32; 4], row1: [f32; 4], row2: [f32; 4]) -> Option<[[f32; 4]; 3]> {
    for row in [row0, row1, row2] {
        if !row.iter().all(|v| v.is_finite()) {
            return None;
        }
    }
    let m00 = f64::from(row0[0]);
    let m01 = f64::from(row0[1]);
    let m02 = f64::from(row0[2]);
    let m03 = f64::from(row0[3]);
    let m10 = f64::from(row1[0]);
    let m11 = f64::from(row1[1]);
    let m12 = f64::from(row1[2]);
    let m13 = f64::from(row1[3]);
    let m20 = f64::from(row2[0]);
    let m21 = f64::from(row2[1]);
    let m22 = f64::from(row2[2]);
    let m23 = f64::from(row2[3]);

    let c00 = m11 * m22 - m12 * m21;
    let c01 = m02 * m21 - m01 * m22;
    let c02 = m01 * m12 - m02 * m11;
    let det = m00 * c00 + m10 * c01 + m20 * c02;
    if !det.is_finite() || det.abs() < 1e-20 {
        return None;
    }
    let inv_det = 1.0 / det;
    let i00 = c00 * inv_det;
    let i01 = c01 * inv_det;
    let i02 = c02 * inv_det;
    let i10 = (m12 * m20 - m10 * m22) * inv_det;
    let i11 = (m00 * m22 - m02 * m20) * inv_det;
    let i12 = (m02 * m10 - m00 * m12) * inv_det;
    let i20 = (m10 * m21 - m11 * m20) * inv_det;
    let i21 = (m01 * m20 - m00 * m21) * inv_det;
    let i22 = (m00 * m11 - m01 * m10) * inv_det;
    let i03 = -(i00 * m03 + i01 * m13 + i02 * m23);
    let i13 = -(i10 * m03 + i11 * m13 + i12 * m23);
    let i23 = -(i20 * m03 + i21 * m13 + i22 * m23);
    let out = [
        [i00 as f32, i01 as f32, i02 as f32, i03 as f32],
        [i10 as f32, i11 as f32, i12 as f32, i13 as f32],
        [i20 as f32, i21 as f32, i22 as f32, i23 as f32],
    ];
    if out.iter().all(|r| r.iter().all(|v| v.is_finite())) {
        Some(out)
    } else {
        None
    }
}

/// Transforms an object-space AABB by a 3x4 row-major affine matrix into a
/// tight world-space AABB.
pub fn transform_aabb(
    transform: &[f32; 12],
    local_min: [f32; 3],
    local_max: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let mut wmin = [transform[3], transform[7], transform[11]];
    let mut wmax = wmin;
    for row in 0..3 {
        for col in 0..3 {
            let m = transform[row * 4 + col];
            let a = m * local_min[col];
            let b = m * local_max[col];
            wmin[row] += a.min(b);
            wmax[row] += a.max(b);
        }
    }
    (wmin, wmax)
}

/// Parses raw `meshes` words into validated [`MeshSlice`] entries.
pub fn parse_meshes(meshes: &[u32], total_verts: usize, total_tris: usize) -> Vec<MeshSlice> {
    let count = meshes.len() / MESH_WORDS;
    let mut out = Vec::with_capacity(count);
    for m in 0..count {
        let base = m * MESH_WORDS;
        let vo = meshes[base] as i32;
        let vc = meshes[base + 1] as i32;
        let to = meshes[base + 2] as i32;
        let tc = meshes[base + 3] as i32;
        if vo < 0 || vc <= 0 || to < 0 || tc <= 0 {
            out.push(MeshSlice {
                vertex_offset: 0,
                vertex_count: 0,
                triangle_offset: 0,
                triangle_count: 0,
            });
            continue;
        }
        let v_off = (vo as usize).min(total_verts);
        let v_cnt = (vc as usize).min(total_verts.saturating_sub(v_off));
        let t_off = (to as usize).min(total_tris);
        let t_cnt = (tc as usize).min(total_tris.saturating_sub(t_off));
        out.push(MeshSlice {
            vertex_offset: v_off,
            vertex_count: v_cnt,
            triangle_offset: t_off,
            triangle_count: t_cnt,
        });
    }
    out
}

/// Parses raw `instances` words against per-mesh AABBs, filtering out hidden,
/// out-of-range, empty-mesh, or singular instances.
pub fn parse_instances(
    instances: &[u32],
    mesh_bounds: &[Option<([f32; 3], [f32; 3])>],
) -> Vec<PreparedInstance> {
    let count = instances.len() / INSTANCE_WORDS;
    let mut out = Vec::with_capacity(count);
    for row in 0..count {
        let base = row * INSTANCE_WORDS;
        let row0 = [
            f32_at(instances, base),
            f32_at(instances, base + 1),
            f32_at(instances, base + 2),
            f32_at(instances, base + 3),
        ];
        let row1 = [
            f32_at(instances, base + 4),
            f32_at(instances, base + 5),
            f32_at(instances, base + 6),
            f32_at(instances, base + 7),
        ];
        let row2 = [
            f32_at(instances, base + 8),
            f32_at(instances, base + 9),
            f32_at(instances, base + 10),
            f32_at(instances, base + 11),
        ];
        let mesh = instances[base + 12] as i32;
        let custom = instances[base + 13] as i32;
        let mask = (instances[base + 14] as i32 & 0xFF) as u8;
        if mask == 0 || mesh < 0 || (mesh as usize) >= mesh_bounds.len() {
            continue;
        }
        let Some((local_min, local_max)) = mesh_bounds[mesh as usize] else {
            continue;
        };
        let Some(inv_transform) = invert_affine_3x4(row0, row1, row2) else {
            continue;
        };
        let det3 = f64::from(row0[0])
            * (f64::from(row1[1]) * f64::from(row2[2]) - f64::from(row1[2]) * f64::from(row2[1]))
            - f64::from(row0[1])
                * (f64::from(row1[0]) * f64::from(row2[2])
                    - f64::from(row1[2]) * f64::from(row2[0]))
            + f64::from(row0[2])
                * (f64::from(row1[0]) * f64::from(row2[1])
                    - f64::from(row1[1]) * f64::from(row2[0]));
        let negative_det = det3 < 0.0;
        let transform = [
            row0[0], row0[1], row0[2], row0[3], row1[0], row1[1], row1[2], row1[3], row2[0],
            row2[1], row2[2], row2[3],
        ];
        let (world_min, world_max) = transform_aabb(&transform, local_min, local_max);
        out.push(PreparedInstance {
            row: row as u32,
            mesh: mesh as u32,
            custom,
            mask,
            transform,
            inv_transform,
            negative_det,
            world_min,
            world_max,
        });
    }
    out
}

#[inline]
fn slab_hit(
    origin: [f32; 3],
    inv_dir: [f32; 3],
    bmin: [f32; 3],
    bmax: [f32; 3],
    tmin: f32,
    tmax: f32,
) -> Option<f32> {
    let mut near = tmin;
    let mut far = tmax;
    for a in 0..3 {
        let t0 = (bmin[a] - origin[a]) * inv_dir[a];
        let t1 = (bmax[a] - origin[a]) * inv_dir[a];
        let lo = t0.min(t1);
        let hi = t0.max(t1);
        near = near.max(lo);
        far = far.min(hi);
    }
    if far >= near { Some(near) } else { None }
}

#[inline]
fn inv_ray_dir(dir: [f32; 3]) -> [f32; 3] {
    [
        if dir[0] == 0.0 {
            f32::copysign(1e30, dir[0])
        } else {
            1.0 / dir[0]
        },
        if dir[1] == 0.0 {
            f32::copysign(1e30, dir[1])
        } else {
            1.0 / dir[1]
        },
        if dir[2] == 0.0 {
            f32::copysign(1e30, dir[2])
        } else {
            1.0 / dir[2]
        },
    ]
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn intersect_triangle(
    origin: [f32; 3],
    dir: [f32; 3],
    tmin: f32,
    tmax: f32,
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    flip_winding: bool,
    cull_back: bool,
    cull_front: bool,
) -> Option<(f32, f32, f32, bool)> {
    let e1 = sub3(v1, v0);
    let e2 = sub3(v2, v0);
    let h = cross3(dir, e2);
    let det = dot3(e1, h);
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    let front = (det > 0.0) ^ flip_winding;
    if cull_back && !front {
        return None;
    }
    if cull_front && front {
        return None;
    }
    let inv_det = 1.0 / det;
    let s = sub3(origin, v0);
    let u = dot3(s, h) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross3(s, e1);
    let v = dot3(dir, q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = dot3(e2, q) * inv_det;
    if t >= tmin && t <= tmax {
        Some((t, u, v, front))
    } else {
        None
    }
}

#[derive(Clone, Copy)]
struct BestHit {
    t: f32,
    u: f32,
    v: f32,
    triangle: i32,
    instance: i32,
    custom: i32,
    front: bool,
}

impl BestHit {
    fn miss(tmax: f32) -> Self {
        Self {
            t: tmax,
            u: 0.0,
            v: 0.0,
            triangle: -1,
            instance: -1,
            custom: 0,
            front: false,
        }
    }

    fn write_into(&self, dst: &mut [u32]) {
        if self.triangle < 0 {
            dst.copy_from_slice(&MISS_WORDS);
        } else {
            dst[0] = self.t.to_bits();
            dst[1] = self.u.to_bits();
            dst[2] = self.v.to_bits();
            dst[3] = self.triangle as u32;
            dst[4] = self.instance as u32;
            dst[5] = self.custom as u32;
            dst[6] = u32::from(self.front);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn traverse_mesh(
    bvh: &BuiltBvh,
    vertices: &[u32],
    triangles: &[u32],
    slice: MeshSlice,
    origin: [f32; 3],
    dir: [f32; 3],
    tmin: f32,
    instance: i32,
    custom: i32,
    flip_winding: bool,
    flags: i32,
    best: &mut BestHit,
) {
    if bvh.nodes.is_empty() {
        return;
    }
    let any_hit = (flags & FLAG_TERMINATE_ON_FIRST_HIT) != 0;
    let cull_back = (flags & FLAG_CULL_BACK) != 0;
    let cull_front = (flags & FLAG_CULL_FRONT) != 0;
    let inv_dir = inv_ray_dir(dir);
    let mut stack = [0u32; 64];
    if slab_hit(
        origin,
        inv_dir,
        bvh.nodes[0].min,
        bvh.nodes[0].max,
        tmin,
        best.t,
    )
    .is_none()
    {
        return;
    }
    stack[0] = 0;
    let mut sp = 1usize;
    while sp > 0 {
        sp -= 1;
        let node = bvh.nodes[stack[sp] as usize];
        if node.count > 0 {
            let first = node.left_or_first as usize;
            let end = first + node.count as usize;
            for &local_t in &bvh.ordered[first..end] {
                let tri = local_t as usize;
                let [ia, ib, ic] = clamped_triangle_indices(
                    triangles,
                    slice.triangle_offset + tri,
                    slice.vertex_count,
                );
                let v0 = vec3_at(vertices, (slice.vertex_offset + ia) * POSITION_WORDS);
                let v1 = vec3_at(vertices, (slice.vertex_offset + ib) * POSITION_WORDS);
                let v2 = vec3_at(vertices, (slice.vertex_offset + ic) * POSITION_WORDS);
                if let Some((t, u, v, front)) = intersect_triangle(
                    origin,
                    dir,
                    tmin,
                    best.t,
                    v0,
                    v1,
                    v2,
                    flip_winding,
                    cull_back,
                    cull_front,
                ) {
                    let tri_i = local_t as i32;
                    let closer = best.triangle < 0
                        || t < best.t
                        || (t == best.t
                            && (instance < best.instance
                                || (instance == best.instance && tri_i < best.triangle)));
                    if closer {
                        *best = BestHit {
                            t,
                            u,
                            v,
                            triangle: tri_i,
                            instance,
                            custom,
                            front,
                        };
                        if any_hit {
                            return;
                        }
                    }
                }
            }
        } else {
            let l_idx = node.left_or_first as usize;
            let r_idx = l_idx + 1;
            let l_hit = slab_hit(
                origin,
                inv_dir,
                bvh.nodes[l_idx].min,
                bvh.nodes[l_idx].max,
                tmin,
                best.t,
            );
            let r_hit = slab_hit(
                origin,
                inv_dir,
                bvh.nodes[r_idx].min,
                bvh.nodes[r_idx].max,
                tmin,
                best.t,
            );
            match (l_hit, r_hit) {
                (Some(lt), Some(rt)) => {
                    // Push farther child first so nearer child pops first.
                    let (first_child, second_child) = if lt <= rt {
                        (r_idx as u32, l_idx as u32)
                    } else {
                        (l_idx as u32, r_idx as u32)
                    };
                    if sp + 2 <= stack.len() {
                        stack[sp] = first_child;
                        stack[sp + 1] = second_child;
                        sp += 2;
                    }
                }
                (Some(_), None) => {
                    if sp < stack.len() {
                        stack[sp] = l_idx as u32;
                        sp += 1;
                    }
                }
                (None, Some(_)) => {
                    if sp < stack.len() {
                        stack[sp] = r_idx as u32;
                        sp += 1;
                    }
                }
                (None, None) => {}
            }
        }
    }
}

/// Traces `rays` against a single mesh `(vertices, triangles)` on CPU,
/// returning flat `HIT_WORDS`-per-ray storage words.
pub fn trace_words(rays: &[u32], vertices: &[u32], triangles: &[u32], flags: i32) -> Vec<u32> {
    let ray_count = rays.len() / RAY_WORDS;
    let mut out = vec![0u32; ray_count * HIT_WORDS];
    let slice = MeshSlice {
        vertex_offset: 0,
        vertex_count: vertices.len() / POSITION_WORDS,
        triangle_offset: 0,
        triangle_count: triangles.len() / TRIANGLE_WORDS,
    };
    let bvh = build_mesh_bvh(vertices, triangles, slice);
    if bvh.nodes.is_empty() {
        for r in 0..ray_count {
            out[r * HIT_WORDS..(r + 1) * HIT_WORDS].copy_from_slice(&MISS_WORDS);
        }
        return out;
    }
    for r in 0..ray_count {
        let base = r * RAY_WORDS;
        let origin = vec3_at(rays, base);
        let tmin = f32_at(rays, base + 3);
        let dir = vec3_at(rays, base + 4);
        let tmax = f32_at(rays, base + 7);
        let dst = &mut out[r * HIT_WORDS..(r + 1) * HIT_WORDS];
        if !finite3(origin)
            || !finite3(dir)
            || !tmin.is_finite()
            || tmin < 0.0
            || tmax.is_nan()
            || tmax <= tmin
        {
            dst.copy_from_slice(&MISS_WORDS);
            continue;
        }
        let mut best = BestHit::miss(tmax);
        traverse_mesh(
            &bvh, vertices, triangles, slice, origin, dir, tmin, 0, 0, false, flags, &mut best,
        );
        best.write_into(dst);
    }
    out
}

/// Traces `rays` against an instanced multi-mesh scene on CPU, returning flat
/// `HIT_WORDS`-per-ray storage words.
pub fn trace_instances_words(
    rays: &[u32],
    vertices: &[u32],
    triangles: &[u32],
    meshes: &[u32],
    instances: &[u32],
    flags: i32,
    mask: u8,
) -> Vec<u32> {
    let ray_count = rays.len() / RAY_WORDS;
    let mut out = vec![0u32; ray_count * HIT_WORDS];
    let total_verts = vertices.len() / POSITION_WORDS;
    let total_tris = triangles.len() / TRIANGLE_WORDS;
    let mesh_slices = parse_meshes(meshes, total_verts, total_tris);
    let mesh_bvhs: Vec<BuiltBvh> = mesh_slices
        .iter()
        .map(|&s| build_mesh_bvh(vertices, triangles, s))
        .collect();
    let mesh_bounds: Vec<Option<([f32; 3], [f32; 3])>> = mesh_bvhs
        .iter()
        .map(|b| (!b.nodes.is_empty()).then_some((b.bounds_min, b.bounds_max)))
        .collect();
    let prepared = parse_instances(instances, &mesh_bounds);
    let tlas_aabbs: Vec<(u32, [f32; 3], [f32; 3])> = prepared
        .iter()
        .enumerate()
        .map(|(i, inst)| (i as u32, inst.world_min, inst.world_max))
        .collect();
    let tlas = build_aabb_bvh(&tlas_aabbs);
    if tlas.nodes.is_empty() || mask == 0 {
        for r in 0..ray_count {
            out[r * HIT_WORDS..(r + 1) * HIT_WORDS].copy_from_slice(&MISS_WORDS);
        }
        return out;
    }
    let any_hit = (flags & FLAG_TERMINATE_ON_FIRST_HIT) != 0;
    for r in 0..ray_count {
        let base = r * RAY_WORDS;
        let origin = vec3_at(rays, base);
        let tmin = f32_at(rays, base + 3);
        let dir = vec3_at(rays, base + 4);
        let tmax = f32_at(rays, base + 7);
        let dst = &mut out[r * HIT_WORDS..(r + 1) * HIT_WORDS];
        if !finite3(origin)
            || !finite3(dir)
            || !tmin.is_finite()
            || tmin < 0.0
            || tmax.is_nan()
            || tmax <= tmin
        {
            dst.copy_from_slice(&MISS_WORDS);
            continue;
        }
        let inv_dir = inv_ray_dir(dir);
        let mut best = BestHit::miss(tmax);
        if slab_hit(
            origin,
            inv_dir,
            tlas.nodes[0].min,
            tlas.nodes[0].max,
            tmin,
            best.t,
        )
        .is_some()
        {
            let mut stack = [0u32; 64];
            stack[0] = 0;
            let mut sp = 1usize;
            while sp > 0 {
                sp -= 1;
                let node = tlas.nodes[stack[sp] as usize];
                if node.count > 0 {
                    let first = node.left_or_first as usize;
                    let end = first + node.count as usize;
                    for &prep_idx in &tlas.ordered[first..end] {
                        let inst = &prepared[prep_idx as usize];
                        if (inst.mask & mask) == 0 {
                            continue;
                        }
                        if slab_hit(
                            origin,
                            inv_dir,
                            inst.world_min,
                            inst.world_max,
                            tmin,
                            best.t,
                        )
                        .is_none()
                        {
                            continue;
                        }
                        let m = &inst.inv_transform;
                        let obj_origin = [
                            m[0][0] * origin[0]
                                + m[0][1] * origin[1]
                                + m[0][2] * origin[2]
                                + m[0][3],
                            m[1][0] * origin[0]
                                + m[1][1] * origin[1]
                                + m[1][2] * origin[2]
                                + m[1][3],
                            m[2][0] * origin[0]
                                + m[2][1] * origin[1]
                                + m[2][2] * origin[2]
                                + m[2][3],
                        ];
                        let obj_dir = [
                            m[0][0] * dir[0] + m[0][1] * dir[1] + m[0][2] * dir[2],
                            m[1][0] * dir[0] + m[1][1] * dir[1] + m[1][2] * dir[2],
                            m[2][0] * dir[0] + m[2][1] * dir[1] + m[2][2] * dir[2],
                        ];
                        let m_idx = inst.mesh as usize;
                        traverse_mesh(
                            &mesh_bvhs[m_idx],
                            vertices,
                            triangles,
                            mesh_slices[m_idx],
                            obj_origin,
                            obj_dir,
                            tmin,
                            inst.row as i32,
                            inst.custom,
                            inst.negative_det,
                            flags,
                            &mut best,
                        );
                        if any_hit && best.triangle >= 0 {
                            sp = 0;
                            break;
                        }
                    }
                } else {
                    let l_idx = node.left_or_first as usize;
                    let r_idx = l_idx + 1;
                    let l_hit = slab_hit(
                        origin,
                        inv_dir,
                        tlas.nodes[l_idx].min,
                        tlas.nodes[l_idx].max,
                        tmin,
                        best.t,
                    );
                    let r_hit = slab_hit(
                        origin,
                        inv_dir,
                        tlas.nodes[r_idx].min,
                        tlas.nodes[r_idx].max,
                        tmin,
                        best.t,
                    );
                    match (l_hit, r_hit) {
                        (Some(lt), Some(rt)) => {
                            let (first_child, second_child) = if lt <= rt {
                                (r_idx as u32, l_idx as u32)
                            } else {
                                (l_idx as u32, r_idx as u32)
                            };
                            if sp + 2 <= stack.len() {
                                stack[sp] = first_child;
                                stack[sp + 1] = second_child;
                                sp += 2;
                            }
                        }
                        (Some(_), None) => {
                            if sp < stack.len() {
                                stack[sp] = l_idx as u32;
                                sp += 1;
                            }
                        }
                        (None, Some(_)) => {
                            if sp < stack.len() {
                                stack[sp] = r_idx as u32;
                                sp += 1;
                            }
                        }
                        (None, None) => {}
                    }
                }
            }
        }
        best.write_into(dst);
    }
    out
}

/// CPU/JIT execution of `native.trace@1`.
pub fn trace(
    invocation: &Invocation,
    scope: &mut Scope<'_>,
) -> Result<Vec<Handle>, ExecutionError> {
    let inputs = check_trace(invocation, scope)?;
    let words = trace_words(
        &scope.values(inputs.rays)?,
        &scope.values(inputs.vertices)?,
        &scope.values(inputs.triangles)?,
        inputs.flags,
    );
    let output = scope.create(hit_schema(), invocation.domain)?;
    if let Err(error) = scope.write_values(output, &words) {
        let _ = scope.release(output);
        return Err(error.into());
    }
    Ok(vec![output])
}

/// CPU/JIT execution of `native.trace@2`.
pub fn trace_instances(
    invocation: &Invocation,
    scope: &mut Scope<'_>,
) -> Result<Vec<Handle>, ExecutionError> {
    let inputs = check_trace_instances(invocation, scope)?;
    let words = trace_instances_words(
        &scope.values(inputs.rays)?,
        &scope.values(inputs.vertices)?,
        &scope.values(inputs.triangles)?,
        &scope.values(inputs.meshes)?,
        &scope.values(inputs.instances)?,
        inputs.flags,
        inputs.mask,
    );
    let output = scope.create(hit_schema(), invocation.domain)?;
    if let Err(error) = scope.write_values(output, &words) {
        let _ = scope.release(output);
        return Err(error.into());
    }
    Ok(vec![output])
}
