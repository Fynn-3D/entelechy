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

use crate::{DeviceImage, WgpuBackend};

/// What every stage reads: the canvas, the counts, the cutout, and how
/// small a triangle the sorting pass draws itself.
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
    pad: u32,
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

/// The cutout's mask, at the binding `{mask}`.
const MASK_WGSL: &str = "
@group(0) @binding({mask}) var<storage, read> mask: array<u32>;

fn texel(x: i32, y: i32, w: i32, h: i32) -> f32 {
    let cx = clamp(x, 0, max(w - 1, 0));
    let cy = clamp(y, 0, max(h - 1, 0));
    return bitcast<f32>(mask[u32(cy * w + cx) * 4u + 3u]);
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

/// Step 1: which triangles are drawn, and how.
const KEEP_WGSL: &str = "
@group(0) @binding(3) var<storage, read_write> marks: array<u32>;
@group(0) @binding(4) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> indices: array<u32>;
@group(0) @binding(6) var<storage, read_write> rows: array<u32>;
@group(0) @binding(7) var<storage, read_write> args: array<u32>;

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

// How a triangle is drawn: 0 not at all (a corner not usable, no area,
// no pixel center inside its box on the canvas), 1 by the device's
// rasterizer, 2 here and now, its box being small.
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
    if !(abs(edge(a.xy, b.xy, c.xy)) > 0.0) {
        return 0u;
    }
    let x0 = max(ceil(min(min(a.x, b.x), c.x) - 0.5), 0.0);
    let y0 = max(ceil(min(min(a.y, b.y), c.y) - 0.5), 0.0);
    let x1 = min(floor(max(max(a.x, b.x), c.x) - 0.5), setup.size.x - 1.0);
    let y1 = min(floor(max(max(a.y, b.y), c.y) - 0.5), setup.size.y - 1.0);
    if !(x1 >= x0 && y1 >= y0) {
        return 0u;
    }
    let small = f32(setup.small);
    if x1 - x0 < small && y1 - y0 < small {
        draw_here(t, ia, ib, ic, a, b, c, vec4<u32>(u32(x0), u32(y0), u32(x1), u32(y1)));
        return 2u;
    }
    return 1u;
}

var<workgroup> marked: array<atomic<u32>, 8>;
var<workgroup> kept: atomic<u32>;

// Each block of triangles marks the ones the device draws, a bit each,
// and counts them.
@compute @workgroup_size(256)
fn mark(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>) {
    if local < 8u {
        atomicStore(&marked[local], 0u);
    }
    if local == 0u {
        atomicStore(&kept, 0u);
    }
    workgroupBarrier();
    let block = group.x + group.y * setup.blocks_x;
    let t = block * BLOCK + local;
    if block < setup.block_count && t < setup.triangle_count && sort(t) == 1u {
        atomicOr(&marked[local / 32u], 1u << (local % 32u));
        atomicAdd(&kept, 1u);
    }
    workgroupBarrier();
    if block < setup.block_count {
        if local < 8u {
            marks[block * 8u + local] = atomicLoad(&marked[local]);
        }
        if local == 0u {
            offsets[block] = atomicLoad(&kept);
        }
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
        // An indexed draw reads five words, a pulled one the first four.
        args[0] = sums[255] * 3u;
        args[1] = 1u;
        args[2] = 0u;
        args[3] = 0u;
        args[4] = 0u;
    }
}

// Each kept triangle writes its corners and its row at its place.
@compute @workgroup_size(256)
fn place(@builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>) {
    let block = group.x + group.y * setup.blocks_x;
    let t = block * BLOCK + local;
    if block >= setup.block_count || t >= setup.triangle_count {
        return;
    }
    let word = marks[block * 8u + local / 32u];
    let bit = local % 32u;
    if ((word >> bit) & 1u) == 0u {
        return;
    }
    var before = countOneBits(word & ((1u << bit) - 1u));
    for (var w = 0u; w < local / 32u; w++) {
        before += countOneBits(marks[block * 8u + w]);
    }
    let at = offsets[block] + before;
    indices[at * 3u] = corner(t, 0u);
    indices[at * 3u + 1u] = corner(t, 1u);
    indices[at * 3u + 2u] = corner(t, 2u);
    rows[at] = t;
}
";

/// Step 1's own drawing, where the device takes 64-bit atomics: every
/// pixel center a small triangle covers, by the rule's edges and depth,
/// keeps the greatest complement of the key `depth << 32 | row`, so the
/// nearest fragment and, of two at one depth, the earlier row, and a
/// cleared zero is none.
const HERE_WGSL: &str = "
@group(0) @binding(8) var<storage, read_write> keys: array<atomic<u64>>;

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
";

/// Where the device has no 64-bit atomics: nothing is drawn in step 1
/// (`small` is zero there).
const NOT_HERE_WGSL: &str = "
fn draw_here(t: u32, ia: u32, ib: u32, ic: u32, a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, box: vec4<u32>) {}
";

/// Step 2 where a fragment knows its primitive: the kept triangles drawn
/// indexed, a vertex shaded once for every triangle that shares it.
const INDEXED_WGSL: &str = "
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

// The kept triangle's place, one above it, so a cleared zero is none.
@fragment
fn fs(@builtin(primitive_index) p: u32) -> @location(0) u32 {
    return p + 1u;
}

@fragment
fn fs_cut(in: Cut, @builtin(primitive_index) p: u32) -> @location(0) u32 {
    if mask_alpha(in.uv) < setup.cutout {
        discard;
    }
    return p + 1u;
}
";

/// Step 2 where it does not: each triangle's corners pulled from the kept
/// index buffer, its place carried flat.
const PULLED_WGSL: &str = "
@group(0) @binding(2) var<storage, read> indices: array<u32>;

struct Plain {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) place: u32,
};

struct Cut {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) place: u32,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) at: u32) -> Plain {
    var out: Plain;
    out.clip = clip_of(indices[at]);
    out.place = at / 3u;
    return out;
}

@vertex
fn vs_cut(@builtin(vertex_index) at: u32) -> Cut {
    let v = indices[at];
    var out: Cut;
    out.clip = clip_of(v);
    out.place = at / 3u;
    out.uv = uv_of(v);
    return out;
}

@fragment
fn fs(in: Plain) -> @location(0) u32 {
    return in.place + 1u;
}

@fragment
fn fs_cut(in: Cut) -> @location(0) u32 {
    if mask_alpha(in.uv) < setup.cutout {
        discard;
    }
    return in.place + 1u;
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
        row = rows[place - 1u];
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
        row = rows[place - 1u];
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
    draw_layout: wgpu::BindGroupLayout,
    draw: wgpu::RenderPipeline,
    draw_cut: wgpu::RenderPipeline,
    resolve_layout: wgpu::BindGroupLayout,
    resolve: wgpu::ComputePipeline,
    /// The resolving pass of `native.cover`: keys, not a surface.
    cover: wgpu::ComputePipeline,
    /// A single unread texel bound when a call carries no cutout mask.
    empty_mask: wgpu::Buffer,
    /// Whether the draw is indexed: see [`GpuBackend::primitive_index`].
    indexed: bool,
    /// Whether the sorting pass draws small triangles itself: where the
    /// device lands 64-bit atomics.
    here: bool,
}

const PLACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

fn build(
    device: &wgpu::Device,
    wait: crate::Wait,
    indexed: bool,
    here: bool,
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

    let keep_mask = MASK_WGSL.replace("{mask}", "9");
    let keep_module = module(
        "entelechy rasterize keep",
        &[
            SETUP_WGSL,
            TRIANGLES_WGSL,
            &keep_mask,
            KEEP_WGSL,
            if here { HERE_WGSL } else { NOT_HERE_WGSL },
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
            storage(9, c, true),
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

    let feed = if indexed { INDEXED_WGSL } else { PULLED_WGSL };
    // An `enable` directive comes first in a module.
    let enable = if indexed {
        "enable primitive_index;\n"
    } else {
        ""
    };
    let draw_module = module(
        "entelechy rasterize draw",
        &[enable, SETUP_WGSL, &MASK_WGSL.replace("{mask}", "3"), feed],
    );
    let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
    let (draw_layout, draw_pipeline_layout) = layout(
        "entelechy rasterize draw",
        &[
            uniform(vf),
            storage(1, wgpu::ShaderStages::VERTEX_FRAGMENT, true),
            storage(2, wgpu::ShaderStages::VERTEX, true),
            storage(3, wgpu::ShaderStages::FRAGMENT, true),
        ],
    );
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
                targets: &[Some(wgpu::ColorTargetState {
                    format: PLACE_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    let draw = render("vs", "fs");
    let draw_cut = render("vs_cut", "fs_cut");

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
    let empty_mask = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rasterize mask"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
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
        draw_layout,
        draw,
        draw_cut,
        resolve_layout,
        resolve,
        cover,
        empty_mask,
        indexed,
        here,
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
        NativeOp::RasterizeCanvas => {
            let (vertices, triangles) =
                entelechy_rules::raster::check_rasterize_canvas(invocation, scope)?;
            (vertices, triangles, None, 0.0)
        }
        NativeOp::Rasterize => {
            let (vertices, triangles) = check_rasterize(invocation, scope)?;
            (vertices, triangles, None, 0.0)
        }
    };
    let pipelines = match pipelines.get_or_init(|| {
        build(
            &gpu.device,
            gpu.wait,
            gpu.primitive_index,
            gpu.int64_atomics,
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

    // The mask is always bound; with none it is a single texel the shaders
    // leave unread.
    let mask_size = match mask {
        Some(mask) => scope.size(mask)?,
        None => entelechy_abi::Domain::Grid(1, 1),
    };
    let mask_buffer = match mask {
        Some(mask) => gpu.bind_input(scope, mask)?,
        None => pipelines.empty_mask.clone(),
    };
    let block_count = triangle_count.div_ceil(BLOCK);
    let blocks_x = block_count.clamp(1, MOST_GROUPS);
    let blocks_y = block_count.div_ceil(blocks_x).max(1);
    let setup: [u32; 12] = [
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
        if pipelines.here { SMALL } else { 0 },
        0,
    ];
    let mut setup_bytes = [0u8; 48];
    for (slot, word) in setup_bytes.chunks_exact_mut(4).zip(setup) {
        slot.copy_from_slice(&word.to_le_bytes());
    }
    let setup_buffer = gpu.take_uniform(&setup_bytes);

    // The keeping pass's buffers, from the pile and back to it.
    let spare = gpu.spare();
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
    let args = scratch(8, wgpu::BufferUsages::INDIRECT);
    // The keys the sorting pass draws small triangles into, a pixel each,
    // cleared to none.
    let pixels = u64::from(width) * u64::from(height);
    let est_ops = u64::from(triangle_count)
        .saturating_mul(64)
        .saturating_add(pixels.saturating_mul(16));
    if gpu.wait.parks_with_host() && !gpu.measured() && gpu.page_queue.full(pixels, est_ops) {
        gpu.drain_page()?;
    }
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
    let keep_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("entelechy rasterize keep"),
        layout: &pipelines.keep_layout,
        entries: &[
            entry(0, &setup_buffer),
            entry(1, &vertex_buffer),
            entry(2, &triangle_buffer),
            entry(3, &marks),
            entry(4, &offsets),
            entry(5, &indices),
            entry(6, &rows),
            entry(7, &args),
            entry(8, &keys),
            entry(9, &mask_buffer),
        ],
    });
    let draw_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("entelechy rasterize draw"),
        layout: &pipelines.draw_layout,
        entries: &[
            entry(0, &setup_buffer),
            entry(1, &vertex_buffer),
            entry(2, &indices),
            entry(3, &mask_buffer),
        ],
    });
    let places = target(gpu, "surface places", width, height, PLACE_FORMAT);
    let depth = target(gpu, "surface depth", width, height, DEPTH_FORMAT);
    let places_view = places.create_view(&Default::default());
    let depth_view = depth.create_view(&Default::default());

    // A surface, or two words a key.
    let (schema, words) = match before {
        Some(_) => (entelechy_rules::raster::canvas_schema(), 2),
        None => (surface_schema(), SURFACE_WORDS as u64),
    };
    let output_size = width as u64 * height as u64 * words * 4;
    // The keys before the cover; the surface's pass binds the small
    // triangles' keys there, unread.
    let before_buffer = match before {
        Some(keys) => Some(gpu.bind_input(scope, keys)?),
        None => None,
    };
    let output_buffer = spare.take(
        &gpu.device,
        output_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let resolve_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("entelechy rasterize resolve"),
        layout: &pipelines.resolve_layout,
        entries: &[
            entry(0, &setup_buffer),
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
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("entelechy rasterize"),
        });
    let drawing = triangle_count > 0;
    if pipelines.here {
        encoder.clear_buffer(&keys, 0, None);
    }
    if drawing {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("entelechy rasterize keep"),
            timestamp_writes: timing_set
                .as_ref()
                .map(|timing| timing.compute(Some(0), None)),
        });
        pass.set_bind_group(0, &keep_group, &[]);
        pass.set_pipeline(&pipelines.mark);
        pass.dispatch_workgroups(blocks_x, blocks_y, 1);
        pass.set_pipeline(&pipelines.sum);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&pipelines.place);
        pass.dispatch_workgroups(blocks_x, blocks_y, 1);
    }
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("entelechy rasterize draw"),
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
            pass.set_pipeline(if mask.is_some() {
                &pipelines.draw_cut
            } else {
                &pipelines.draw
            });
            pass.set_bind_group(0, &draw_group, &[]);
            if pipelines.indexed {
                pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed_indirect(&args, 0);
            } else {
                pass.draw_indirect(&args, 0);
            }
        }
    }
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
    let guard = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    // Under a measured run a native operation waits for the device like any
    // node, and reports the device's clock where there is one.
    let timing = gpu.resolve_timing(&mut encoder, timing_set.as_ref(), 2);
    let submission = gpu.submit(encoder.finish());
    if gpu.wait.parks_with_host() && !gpu.measured() {
        gpu.page_queue.note(pixels, est_ops);
        if triangle_count >= 500_000 {
            gpu.drain_page()?;
        }
    }
    gpu.settle_measured(submission, timing);
    if let Some(error) = gpu.wait.block_on(guard.pop()) {
        return Err(ExecutionError::Device(format!(
            "rasterize: {}",
            crate::device_error(&error)
        )));
    }
    let output = scope.adopt_resident(
        schema,
        domain,
        Box::new(DeviceImage {
            buffer: output_buffer,
            spare: Some(gpu.spare()),
            holds: None,
        }),
    )?;
    // The call's own targets and buffers go back to the piles for the next
    // rasterize of the same shape; the queue orders their reuse after this
    // run.
    for texture in [places, depth] {
        gpu.targets().keep(texture);
    }
    for buffer in [setup_buffer, marks, offsets, indices, rows, args, keys] {
        spare.keep(buffer, None);
    }
    Ok(vec![output])
}
