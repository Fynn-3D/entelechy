//! The native `rasterize` operation's contract, and the CPU rasterizer
//! the interpreter and the JIT each run. It is the fast path of the
//! `raster` package's rule and gives its surface bit for bit. It is
//! device emulation; shading, projection and geometry live in Kernels.
//!
//! Contract:
//! - input 0, `vertices`: rows laid out as `clip: Vec4, normal: Vec3,
//!   uv: Vec2` (nine floats). `clip` is a clip-space position: after the
//!   divide by `w`, `x` and `y` span `[-1, 1]` (y up) and `z` spans
//!   `[0, 1]` (near to far).
//! - input 1, `triangles`: rows of three `Int` vertex indices, clamped
//!   into range.
//! - output, `surface`: a grid laid out as `depth: Float, normal: Vec3,
//!   uv: Vec2, id: Int`, the nearest fragment at each pixel centre with
//!   perspective-correct attributes and the triangle's row index as `id`.
//!   Pixels nothing covers hold depth `1.0`, zeros, and id `-1`.
//!
//! No culling and no clipping: a triangle with any vertex at `w <= 0`,
//! or any clip component that is not finite, is skipped whole. A
//! triangle of no area is skipped too, so a Kernel drops one by putting
//! its three corners on one vertex. Triangles are drawn in row order; a
//! nearer fragment replaces a farther one, and an equal depth keeps the
//! earlier triangle.
//!
//! A pixel is covered where its centre, at `x + 0.5, y + 0.5`, is inside
//! the triangle's box and on or inside all three of its edges, the edge
//! functions taken in floats. The box test is part of the rule.
//!
//! The cutout form takes one more input and one parameter:
//! - input 2, `mask`: an `Image`. A fragment whose alpha, read as
//!   `at_linear(mask, uv * size(mask))`, is below `cutout` writes no
//!   surface and settles no depth.
//! - param 0, `cutout`: the alpha a fragment must have, `0.5` by
//!   default.

use entelechy_abi::{Handle, Invocation, Scalar, Schema, Scope, Shape};

use crate::ExecutionError;

/// Words per vertex.
pub const VERTEX_WORDS: usize = 9;
/// Words per triangle.
pub const TRIANGLE_WORDS: usize = 3;
/// Words per surface element.
pub const SURFACE_WORDS: usize = 7;

/// The vertex layout: `clip: Vec4, normal: Vec3, uv: Vec2`.
pub fn vertex_schema() -> Schema {
    Schema::new(vec![Scalar::Float; VERTEX_WORDS])
}

/// The triangle layout: three `Int` indices.
pub fn triangle_schema() -> Schema {
    Schema::new(vec![Scalar::Int; TRIANGLE_WORDS])
}

/// The mask layout: an `Image`, whose alpha a cutout tests.
pub fn mask_schema() -> Schema {
    Schema::new(vec![Scalar::Float; 4])
}

/// The alpha of a mask at a position in its own texels, as `at_linear`
/// reads it: the four around it, blended, clamped at the edges. Every
/// backend samples it this way.
pub fn mask_alpha(words: &[u32], width: u32, height: u32, uv: [f32; 2]) -> f32 {
    if width == 0 || height == 0 || words.is_empty() {
        return 1.0;
    }
    let at = |x: i64, y: i64| -> f32 {
        let x = x.clamp(0, width as i64 - 1) as usize;
        let y = y.clamp(0, height as i64 - 1) as usize;
        f32::from_bits(words[(y * width as usize + x) * 4 + 3])
    };
    let place = |v: f32, size: u32| -> (i64, f32) {
        let scaled = v * size as f32 - 0.5;
        let floor = scaled.floor();
        let frac = if scaled.is_finite() {
            scaled - floor
        } else {
            0.0
        };
        (if floor.is_finite() { floor as i64 } else { 0 }, frac)
    };
    let (x0, fx) = place(uv[0], width);
    let (y0, fy) = place(uv[1], height);
    let top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * fx;
    let bottom = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * fx;
    top + (bottom - top) * fy
}

/// The layout of a canvas: one `Long` key a pixel, as the raster
/// package's empty canvas holds.
pub fn canvas_schema() -> Schema {
    // A `Long` is kept as two words.
    Schema::new(vec![Scalar::Long; 2])
}

/// The surface layout: `depth: Float, normal: Vec3, uv: Vec2, id: Int`.
pub fn surface_schema() -> Schema {
    let mut scalars = vec![Scalar::Float; SURFACE_WORDS - 1];
    scalars.push(Scalar::Int);
    Schema::new(scalars)
}

/// Checks the invocation against the contract, returning the vertex and
/// triangle handles and the output grid. Every backend applies it.
pub fn check_rasterize(
    invocation: &Invocation,
    scope: &Scope<'_>,
) -> Result<(Handle, Handle), ExecutionError> {
    let (vertices, triangles, _) = check_rasterize_with(invocation, scope, Extra::None)?;
    Ok((vertices, triangles))
}

/// The same for the canvas form: the surface is the canvas's extent.
pub fn check_rasterize_canvas(
    invocation: &Invocation,
    scope: &Scope<'_>,
) -> Result<(Handle, Handle), ExecutionError> {
    let (vertices, triangles, _) = check_rasterize_with(invocation, scope, Extra::Canvas)?;
    Ok((vertices, triangles))
}

/// The same for the cutout form, also returning the mask and the alpha
/// a fragment must reach.
pub fn check_rasterize_cutout(
    invocation: &Invocation,
    scope: &Scope<'_>,
) -> Result<(Handle, Handle, Handle, f32), ExecutionError> {
    let (vertices, triangles, mask) = check_rasterize_with(invocation, scope, Extra::Mask)?;
    let mask = mask.expect("a cutout asked for its mask");
    let cutout = match invocation.params.first() {
        Some(entelechy_abi::Value::Float(value)) => *value,
        _ => 0.5,
    };
    Ok((vertices, triangles, mask, cutout))
}

/// The third input a form of the rasterizer takes, if any.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Extra {
    None,
    Mask,
    Canvas,
}

fn check_rasterize_with(
    invocation: &Invocation,
    scope: &Scope<'_>,
    extra: Extra,
) -> Result<(Handle, Handle, Option<Handle>), ExecutionError> {
    let wanted = if extra == Extra::None { 2 } else { 3 };
    if invocation.inputs.len() != wanted {
        return Err(ExecutionError::InputCount {
            expected: wanted,
            actual: invocation.inputs.len(),
        });
    }
    let (vertices, triangles) = (invocation.inputs[0], invocation.inputs[1]);
    let mask = (extra == Extra::Mask).then(|| invocation.inputs[2]);
    let contract = |message: String| ExecutionError::Device(format!("rasterize: {message}"));
    if scope.schema(vertices)? != vertex_schema() {
        return Err(contract(
            "`vertices` must be rows of `clip: Vec4, normal: Vec3, uv: Vec2`".into(),
        ));
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
    if invocation.domain.shape() != Shape::Grid {
        return Err(ExecutionError::DomainShape {
            expected: Shape::Grid,
            actual: invocation.domain.shape(),
        });
    }
    if invocation.domain.is_empty() {
        return Err(ExecutionError::EmptyDomain);
    }
    if extra == Extra::Canvas {
        let canvas = invocation.inputs[2];
        if scope.schema(canvas)? != canvas_schema() {
            return Err(contract("`canvas` must be a grid of `Long`".into()));
        }
        let extent = scope.size(canvas)?;
        if extent.shape() != Shape::Grid {
            return Err(contract("`canvas` must be a grid".into()));
        }
        if extent != invocation.domain {
            return Err(contract(format!(
                "the surface is the canvas's extent, {} by {}, not {} by {}: its domain is `size(canvas)`",
                extent.width(),
                extent.height(),
                invocation.domain.width(),
                invocation.domain.height()
            )));
        }
    }
    if let Some(mask) = mask {
        if scope.schema(mask)? != mask_schema() {
            return Err(contract("`mask` must be an Image".into()));
        }
        if scope.size(mask)?.shape() != Shape::Grid {
            return Err(contract("`mask` must be a grid".into()));
        }
    }
    // The surface is charged before any backend draws it, so an
    // oversized domain is refused by its cost.
    scope.can_hold(&[(&surface_schema(), invocation.domain)])?;
    Ok((vertices, triangles, mask))
}

/// What a cutout tests a fragment against.
pub struct Cutout<'m> {
    /// The mask's words, four per pixel.
    pub words: &'m [u32],
    /// Its width in pixels.
    pub width: u32,
    /// Its height in pixels.
    pub height: u32,
    /// The alpha a fragment must reach.
    pub alpha: f32,
}

/// Rasterizes on the CPU into a new surface grid owned by the scope's Host.
pub fn rasterize(
    invocation: &Invocation,
    scope: &mut Scope<'_>,
) -> Result<Vec<Handle>, ExecutionError> {
    let (vertices, triangles) = check_rasterize(invocation, scope)?;
    let domain = invocation.domain;
    // The rows are read in place.
    let words = rasterize_words(
        &scope.values(vertices)?,
        &scope.values(triangles)?,
        domain.width(),
        domain.height(),
    );
    let output = scope.create(surface_schema(), domain)?;
    if let Err(error) = scope.write_values(output, &words) {
        let _ = scope.release(output);
        return Err(error.into());
    }
    Ok(vec![output])
}

/// Rasterizes onto a canvas: as [`rasterize`], over the canvas's extent.
pub fn rasterize_canvas(
    invocation: &Invocation,
    scope: &mut Scope<'_>,
) -> Result<Vec<Handle>, ExecutionError> {
    check_rasterize_canvas(invocation, scope)?;
    let mut drawn = invocation.clone();
    drawn.inputs.truncate(2);
    rasterize(&drawn, scope)
}

/// Rasterizes with a cutout: a fragment whose mask alpha is below the
/// parameter is dropped.
pub fn rasterize_cutout(
    invocation: &Invocation,
    scope: &mut Scope<'_>,
) -> Result<Vec<Handle>, ExecutionError> {
    let (vertices, triangles, mask, alpha) = check_rasterize_cutout(invocation, scope)?;
    let mask_size = scope.size(mask)?;
    let mask_words = scope.values(mask)?;
    let domain = invocation.domain;
    let words = rasterize_masked(
        &scope.values(vertices)?,
        &scope.values(triangles)?,
        domain.width(),
        domain.height(),
        Some(Cutout {
            words: &mask_words,
            width: mask_size.width(),
            height: mask_size.height(),
            alpha,
        }),
    );
    let output = scope.create(surface_schema(), domain)?;
    if let Err(error) = scope.write_values(output, &words) {
        let _ = scope.release(output);
        return Err(error.into());
    }
    Ok(vec![output])
}

/// The pure rasterization: the surface words for the given vertices and
/// triangles over a `width` by `height` grid.
pub fn rasterize_words(vertices: &[u32], triangles: &[u32], width: u32, height: u32) -> Vec<u32> {
    rasterize_masked(vertices, triangles, width, height, None)
}

/// The same, with a cutout when one is given.
pub fn rasterize_masked(
    vertices: &[u32],
    triangles: &[u32],
    width: u32,
    height: u32,
    cutout: Option<Cutout<'_>>,
) -> Vec<u32> {
    let pixel_count = width as usize * height as usize;
    let mut out = vec![0u32; pixel_count * SURFACE_WORDS];
    let mut depth = vec![1.0f32; pixel_count];
    let drawn = if vertices.len() < VERTEX_WORDS || pixel_count == 0 {
        Vec::new()
    } else {
        set_up(vertices, triangles, width, height)
    };
    let many: usize = drawn.iter().map(Vec::len).sum();
    // Bands of rows, one to a thread. Each pixel is drawn by one thread
    // taking the triangles in row order, so the surface is the same bit
    // for bit for any thread count.
    let threads = crate::cores()
        .min(height as usize)
        .min(many.div_ceil(BAND_TRIANGLES))
        .max(1);
    let band = (height as usize).div_ceil(threads).max(1);
    let row = width as usize;
    let cutout = cutout.as_ref();
    let draw = |first: usize, out: &mut [u32], depth: &mut [f32]| {
        for pixel in out.chunks_exact_mut(SURFACE_WORDS) {
            pixel[0] = 1.0f32.to_bits();
            pixel[SURFACE_WORDS - 1] = (-1i32) as u32;
        }
        let rows = first as u32..(first + depth.len() / row.max(1)) as u32;
        for chunk in &drawn {
            draw_band(chunk, vertices, width, rows.clone(), out, depth, cutout);
        }
    };
    if threads == 1 {
        draw(0, &mut out, &mut depth);
        return out;
    }
    std::thread::scope(|s| {
        for (index, (out, depth)) in out
            .chunks_mut(band * row * SURFACE_WORDS)
            .zip(depth.chunks_mut(band * row))
            .enumerate()
        {
            let draw = &draw;
            s.spawn(move || draw(index * band, out, depth));
        }
    });
    out
}

/// The fewest triangles per band thread; below this, one thread draws
/// them all.
const BAND_TRIANGLES: usize = 256;

/// A corner in pixels: `x` and `y` from the top left, `z` after the
/// divide, and `1 / w`.
#[derive(Clone, Copy)]
struct Point {
    x: f32,
    y: f32,
    z: f32,
    inv_w: f32,
}

/// A triangle ready to draw: corners in pixels, wound so the edge
/// functions are positive inside, with their vertex rows, its area, its
/// row, and the pixel centres its box holds. Normals and uvs are read
/// from the vertex rows when a fragment is kept.
struct Drawn {
    a: Point,
    b: Point,
    c: Point,
    rows: [u32; 3],
    area: f32,
    id: u32,
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
}

/// Where `(px, py)` is against the edge from `a` to `b`: positive to its
/// left.
fn edge(a: &Point, b: &Point, px: f32, py: f32) -> f32 {
    (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x)
}

/// Every triangle that may cover a pixel centre, in row order, as runs
/// of the setup threads. Triangles that draw nothing are left out.
fn set_up(vertices: &[u32], triangles: &[u32], width: u32, height: u32) -> Vec<Vec<Drawn>> {
    let vertex_count = vertices.len() / VERTEX_WORDS;
    let float = |w: u32| f32::from_bits(w);
    let point = |index: u32| -> Option<Point> {
        let base = index as usize * VERTEX_WORDS;
        let clip: [f32; 4] = std::array::from_fn(|k| float(vertices[base + k]));
        if clip.iter().any(|v| !v.is_finite()) || clip[3] <= 0.0 {
            return None;
        }
        let inv_w = 1.0 / clip[3];
        Some(Point {
            x: (clip[0] * inv_w * 0.5 + 0.5) * width as f32,
            y: (0.5 - clip[1] * inv_w * 0.5) * height as f32,
            z: clip[2] * inv_w,
            inv_w,
        })
    };
    let one = |id: usize, tri: &[u32]| -> Option<Drawn> {
        let index = |word: u32| (word as i32).clamp(0, vertex_count as i32 - 1) as u32;
        let (ia, ib, ic) = (index(tri[0]), index(tri[1]), index(tri[2]));
        let a = point(ia)?;
        let b = point(ib)?;
        let c = point(ic)?;
        // Wind consistently so the edge functions are positive inside.
        let (b, c, ib, ic) = if edge(&a, &b, c.x, c.y) < 0.0 {
            (c, b, ic, ib)
        } else {
            (b, c, ib, ic)
        };
        let area = edge(&a, &b, c.x, c.y);
        if area.is_nan() || area <= 0.0 {
            return None;
        }
        // The first and last pixel whose centre, at `x + 0.5`, is in the
        // box.
        let x0 = (a.x.min(b.x).min(c.x) - 0.5).ceil().max(0.0);
        let y0 = (a.y.min(b.y).min(c.y) - 0.5).ceil().max(0.0);
        let x1 = (a.x.max(b.x).max(c.x) - 0.5)
            .floor()
            .min(width as f32 - 1.0);
        let y1 = (a.y.max(b.y).max(c.y) - 0.5)
            .floor()
            .min(height as f32 - 1.0);
        if x1 < x0 || y1 < y0 {
            return None;
        }
        Some(Drawn {
            a,
            b,
            c,
            rows: [ia, ib, ic],
            area,
            id: id as u32,
            x0: x0 as u32,
            x1: x1 as u32,
            y0: y0 as u32,
            y1: y1 as u32,
        })
    };
    let count = triangles.len() / TRIANGLE_WORDS;
    let threads = crate::cores().min(count.div_ceil(SETUP_TRIANGLES)).max(1);
    let per_thread = count.div_ceil(threads).max(1);
    let set_up_run = |first: usize, words: &[u32]| -> Vec<Drawn> {
        words
            .chunks_exact(TRIANGLE_WORDS)
            .enumerate()
            .filter_map(|(k, tri)| one(first + k, tri))
            .collect()
    };
    let words = &triangles[..count * TRIANGLE_WORDS];
    if threads == 1 {
        return vec![set_up_run(0, words)];
    }
    // Runs set up in parallel, kept in row order.
    std::thread::scope(|s| {
        let handles: Vec<_> = words
            .chunks(per_thread * TRIANGLE_WORDS)
            .enumerate()
            .map(|(index, words)| {
                let set_up_run = &set_up_run;
                s.spawn(move || set_up_run(index * per_thread, words))
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("raster setup panicked"))
            .collect()
    })
}

/// The fewest triangles per setup thread.
const SETUP_TRIANGLES: usize = 16_384;

/// Draws `drawn` into the rows `rows` of the surface: `out` and `depth`
/// hold those rows and nothing else.
fn draw_band(
    drawn: &[Drawn],
    vertices: &[u32],
    width: u32,
    rows: std::ops::Range<u32>,
    out: &mut [u32],
    depth: &mut [f32],
    cutout: Option<&Cutout<'_>>,
) {
    if rows.is_empty() {
        return;
    }
    let float = |w: u32| f32::from_bits(w);
    for t in drawn {
        let y0 = t.y0.max(rows.start);
        let y1 = t.y1.min(rows.end - 1);
        if y0 > y1 {
            continue;
        }
        let (a, b, c, area) = (&t.a, &t.b, &t.c, t.area);
        let attribute =
            |corner: usize, k: usize| float(vertices[t.rows[corner] as usize * VERTEX_WORDS + k]);
        for y in y0..=y1 {
            for x in t.x0..=t.x1 {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                let w0 = edge(b, c, px, py);
                let w1 = edge(c, a, px, py);
                let w2 = edge(a, b, px, py);
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let (l0, l1, l2) = (w0 / area, w1 / area, w2 / area);
                let z = l0 * a.z + l1 * b.z + l2 * c.z;
                let pixel = (y - rows.start) as usize * width as usize + x as usize;
                if !(0.0..=1.0).contains(&z) || z >= depth[pixel] {
                    continue;
                }
                let inv_w = l0 * a.inv_w + l1 * b.inv_w + l2 * c.inv_w;
                let lerp = |k: usize| {
                    (l0 * attribute(0, k) * a.inv_w
                        + l1 * attribute(1, k) * b.inv_w
                        + l2 * attribute(2, k) * c.inv_w)
                        / inv_w
                };
                // A fragment the mask drops settles no depth either.
                if let Some(cutout) = cutout {
                    let uv = [lerp(7), lerp(8)];
                    if mask_alpha(cutout.words, cutout.width, cutout.height, uv) < cutout.alpha {
                        continue;
                    }
                }
                depth[pixel] = z;
                let base = pixel * SURFACE_WORDS;
                out[base] = z.to_bits();
                for k in 0..3 {
                    out[base + 1 + k] = lerp(4 + k).to_bits();
                }
                for k in 0..2 {
                    out[base + 4 + k] = lerp(7 + k).to_bits();
                }
                out[base + 6] = t.id;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A surface larger than the Host may hold is refused before any
    /// allocation or drawing.
    #[test]
    fn a_surface_past_the_budget_is_refused_before_it_is_drawn() {
        use entelechy_abi::{Capabilities, Capability, Context, Domain, Host, Invocation, Scope};
        let mut host = Host::new(1 << 20);
        let vertices = host.create(vertex_schema(), Domain::Rows(3)).unwrap();
        let triangles = host.create(triangle_schema(), Domain::Rows(1)).unwrap();
        let invocation = Invocation {
            inputs: vec![vertices, triangles],
            params: vec![],
            domain: Domain::Grid(40000, 40000),
            context: Context::default(),
        };
        let all: Capabilities = Capability::ALL.into_iter().collect();
        let mut scope = Scope::new(&mut host, all);
        let refused = rasterize(&invocation, &mut scope).unwrap_err();
        assert!(
            matches!(
                refused,
                ExecutionError::Abi(entelechy_abi::AbiError::Host(
                    entelechy_abi::HostError::BudgetExceeded { .. }
                ))
            ),
            "{refused:?}"
        );
    }

    /// One vertex after the perspective divide, in pixel space.
    struct Screen {
        x: f32,
        y: f32,
        z: f32,
        /// `1 / w`, for perspective-correct interpolation.
        inv_w: f32,
        normal: [f32; 3],
        uv: [f32; 2],
    }

    /// The rule written plainly, on one thread: every triangle in turn,
    /// each pixel of a generous box tested against the triangle's box and
    /// its three edges. The fast rasterizer matches it bit for bit.
    fn definition(
        vertices: &[u32],
        triangles: &[u32],
        width: u32,
        height: u32,
        cutout: Option<Cutout<'_>>,
    ) -> Vec<u32> {
        let vertex_count = vertices.len() / VERTEX_WORDS;
        let pixel_count = width as usize * height as usize;
        let mut out = vec![0u32; pixel_count * SURFACE_WORDS];
        let mut depth = vec![1.0f32; pixel_count];
        for pixel in 0..pixel_count {
            out[pixel * SURFACE_WORDS] = 1.0f32.to_bits();
            out[pixel * SURFACE_WORDS + SURFACE_WORDS - 1] = (-1i32) as u32;
        }
        if vertex_count == 0 {
            return out;
        }
        let float = |w: u32| f32::from_bits(w);
        let vertex = |index: u32| -> Option<Screen> {
            let base = index as usize * VERTEX_WORDS;
            let clip: [f32; 4] = std::array::from_fn(|k| float(vertices[base + k]));
            if clip.iter().any(|v| !v.is_finite()) || clip[3] <= 0.0 {
                return None;
            }
            let inv_w = 1.0 / clip[3];
            Some(Screen {
                x: (clip[0] * inv_w * 0.5 + 0.5) * width as f32,
                y: (0.5 - clip[1] * inv_w * 0.5) * height as f32,
                z: clip[2] * inv_w,
                inv_w,
                normal: std::array::from_fn(|k| float(vertices[base + 4 + k])),
                uv: std::array::from_fn(|k| float(vertices[base + 7 + k])),
            })
        };
        let edge = |a: &Screen, b: &Screen, px: f32, py: f32| -> f32 {
            (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x)
        };
        for (id, tri) in triangles.chunks_exact(TRIANGLE_WORDS).enumerate() {
            let index = |word: u32| (word as i32).clamp(0, vertex_count as i32 - 1) as u32;
            let (Some(a), Some(b), Some(c)) = (
                vertex(index(tri[0])),
                vertex(index(tri[1])),
                vertex(index(tri[2])),
            ) else {
                continue;
            };
            // Wind consistently so the edge functions are positive inside.
            let (b, c) = if edge(&a, &b, c.x, c.y) < 0.0 {
                (c, b)
            } else {
                (b, c)
            };
            let area = edge(&a, &b, c.x, c.y);
            if area.is_nan() || area <= 0.0 {
                continue;
            }
            let min_x = a.x.min(b.x).min(c.x).floor().max(0.0) as u32;
            let max_x = (a.x.max(b.x).max(c.x).ceil() as i64).min(width as i64 - 1);
            let min_y = a.y.min(b.y).min(c.y).floor().max(0.0) as u32;
            let max_y = (a.y.max(b.y).max(c.y).ceil() as i64).min(height as i64 - 1);
            if max_x < 0 || max_y < 0 {
                continue;
            }
            for y in min_y..=max_y as u32 {
                for x in min_x..=max_x as u32 {
                    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                    // A centre outside the triangle's box is outside the
                    // triangle.
                    if px < a.x.min(b.x).min(c.x)
                        || px > a.x.max(b.x).max(c.x)
                        || py < a.y.min(b.y).min(c.y)
                        || py > a.y.max(b.y).max(c.y)
                    {
                        continue;
                    }
                    let w0 = edge(&b, &c, px, py);
                    let w1 = edge(&c, &a, px, py);
                    let w2 = edge(&a, &b, px, py);
                    if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                        continue;
                    }
                    let (l0, l1, l2) = (w0 / area, w1 / area, w2 / area);
                    let z = l0 * a.z + l1 * b.z + l2 * c.z;
                    let pixel = y as usize * width as usize + x as usize;
                    if !(0.0..=1.0).contains(&z) || z >= depth[pixel] {
                        continue;
                    }
                    let inv_w = l0 * a.inv_w + l1 * b.inv_w + l2 * c.inv_w;
                    let lerp = |va: f32, vb: f32, vc: f32| {
                        (l0 * va * a.inv_w + l1 * vb * b.inv_w + l2 * vc * c.inv_w) / inv_w
                    };
                    // A fragment the mask drops settles no depth either.
                    if let Some(cutout) = &cutout {
                        let uv = [
                            lerp(a.uv[0], b.uv[0], c.uv[0]),
                            lerp(a.uv[1], b.uv[1], c.uv[1]),
                        ];
                        if mask_alpha(cutout.words, cutout.width, cutout.height, uv) < cutout.alpha
                        {
                            continue;
                        }
                    }
                    depth[pixel] = z;
                    let base = pixel * SURFACE_WORDS;
                    out[base] = z.to_bits();
                    for k in 0..3 {
                        out[base + 1 + k] = lerp(a.normal[k], b.normal[k], c.normal[k]).to_bits();
                    }
                    for k in 0..2 {
                        out[base + 4 + k] = lerp(a.uv[k], b.uv[k], c.uv[k]).to_bits();
                    }
                    out[base + 6] = id as u32;
                }
            }
        }
        out
    }

    /// A small deterministic generator.
    struct Soup(u64);

    impl Soup {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn float(&mut self, low: f32, high: f32) -> f32 {
            low + (high - low) * (self.next() % 1_000_000) as f32 / 1_000_000.0
        }
        /// A clip position, sometimes one a rasterizer must skip.
        fn clip(&mut self) -> [f32; 4] {
            let w = match self.next() % 40 {
                0 => -0.5,
                1 => 0.0,
                2 => f32::NAN,
                3 => 1e-6,
                _ => self.float(0.3, 3.0),
            };
            [
                self.float(-2.5, 2.5),
                self.float(-2.5, 2.5),
                self.float(-0.2, 1.2),
                w,
            ]
        }
    }

    #[test]
    fn the_rasterizer_draws_what_its_definition_draws() {
        let mut soup = Soup(0x9E37_79B9_7F4A_7C15);
        let mask: Vec<u32> = (0..16 * 16)
            .flat_map(|k| {
                let alpha = ((k * 37) % 100) as f32 / 100.0;
                [0.0f32, 0.0, 0.0, alpha].map(f32::to_bits)
            })
            .collect();
        for (width, height, count) in [
            (1, 1, 20),
            (37, 23, 300),
            (64, 64, 2000),
            (200, 3, 500),
            (3, 200, 500),
            (320, 240, 5000),
        ] {
            // Tiny triangles among large ones: most of a far-off mesh
            // covers no pixel centre at all.
            let vertex_count = 3 * count / 2 + 3;
            let mut vertices = Vec::new();
            for k in 0..vertex_count {
                let mut clip = soup.clip();
                if k % 3 == 1 && soup.next().is_multiple_of(2) {
                    let scale = soup.float(0.0005, 0.02);
                    let base = &vertices[(k - 1) * VERTEX_WORDS..(k - 1) * VERTEX_WORDS + 4];
                    let base: Vec<f32> = base.iter().map(|w| f32::from_bits(*w)).collect();
                    clip = [base[0] + scale, base[1] - scale, base[2], base[3]];
                }
                vertices.extend(vertex(
                    clip,
                    [
                        soup.float(-1.0, 1.0),
                        soup.float(-1.0, 1.0),
                        soup.float(-1.0, 1.0),
                    ],
                    [soup.float(0.0, 1.0), soup.float(0.0, 1.0)],
                ));
            }
            let triangles: Vec<u32> = (0..count * 3)
                .map(|_| match soup.next() % 50 {
                    // Out of range, both ways: clamped.
                    0 => (-3i32) as u32,
                    1 => vertex_count as u32 + 7,
                    _ => (soup.next() % vertex_count as u64) as u32,
                })
                .collect();
            assert_eq!(
                rasterize_masked(&vertices, &triangles, width, height, None),
                definition(&vertices, &triangles, width, height, None),
                "{width}x{height}, {count} triangles"
            );
            fn cut(mask: &[u32]) -> Cutout<'_> {
                Cutout {
                    words: mask,
                    width: 16,
                    height: 16,
                    alpha: 0.4,
                }
            }
            assert_eq!(
                rasterize_masked(&vertices, &triangles, width, height, Some(cut(&mask))),
                definition(&vertices, &triangles, width, height, Some(cut(&mask))),
                "{width}x{height}, {count} triangles, cut out"
            );
        }
    }

    fn vertex(clip: [f32; 4], normal: [f32; 3], uv: [f32; 2]) -> Vec<u32> {
        clip.iter()
            .chain(normal.iter())
            .chain(uv.iter())
            .map(|v| v.to_bits())
            .collect()
    }

    fn surface(words: &[u32], x: u32, y: u32, width: u32) -> ([f32; 6], i32) {
        let base = (y * width + x) as usize * SURFACE_WORDS;
        (
            std::array::from_fn(|k| f32::from_bits(words[base + k])),
            words[base + 6] as i32,
        )
    }

    #[test]
    fn nearer_triangles_win_and_attributes_interpolate() {
        // A full-screen quad at depth 0.5 (two triangles), then a nearer
        // triangle over the left half.
        let mut vertices = Vec::new();
        vertices.extend(vertex([-1.0, -1.0, 0.5, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0]));
        vertices.extend(vertex([1.0, -1.0, 0.5, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0]));
        vertices.extend(vertex([1.0, 1.0, 0.5, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0]));
        vertices.extend(vertex([-1.0, 1.0, 0.5, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0]));
        vertices.extend(vertex([-1.0, -1.0, 0.2, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0]));
        vertices.extend(vertex([0.0, -1.0, 0.2, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0]));
        vertices.extend(vertex([-1.0, 1.0, 0.2, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0]));
        let triangles: Vec<u32> = [0, 1, 2, 0, 2, 3, 4, 5, 6].into_iter().collect();
        let words = rasterize_words(&vertices, &triangles, 8, 8);
        // Top-right pixel: the quad, uv near (1, 0), the second triangle.
        let (s, id) = surface(&words, 7, 0, 8);
        assert_eq!(s[0], 0.5);
        assert_eq!(&s[1..4], &[0.0, 0.0, 1.0]);
        assert!(
            (s[4] - 0.9375).abs() < 1e-5 && (s[5] - 0.0625).abs() < 1e-5,
            "{s:?}"
        );
        assert_eq!(id, 0);
        // Bottom-left pixel: the nearer triangle.
        let (s, id) = surface(&words, 0, 7, 8);
        assert!((s[0] - 0.2).abs() < 1e-6, "{}", s[0]);
        assert_eq!(&s[1..4], &[1.0, 0.0, 0.0]);
        assert_eq!(id, 2);
        // Nothing drawn: the empty grid keeps depth 1 and id -1.
        let empty = rasterize_words(&[], &[], 2, 2);
        assert_eq!(
            surface(&empty, 1, 1, 2),
            ([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], -1)
        );
    }

    #[test]
    fn the_canvas_form_draws_what_the_plain_one_draws_over_its_canvas() {
        use entelechy_abi::{Capabilities, Capability, Context, Domain, Host, Invocation, Scope};
        let mut host = Host::new(1 << 20);
        let corners: Vec<u32> = [
            vertex([-1.0, -1.0, 0.5, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0]),
            vertex([1.0, -1.0, 0.5, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0]),
            vertex([0.0, 1.0, 0.3, 1.0], [1.0, 0.0, 0.0], [0.5, 0.0]),
        ]
        .concat();
        let vertices = host.create(vertex_schema(), Domain::Rows(3)).unwrap();
        host.write_values(vertices, &corners).unwrap();
        let triangles = host.create(triangle_schema(), Domain::Rows(1)).unwrap();
        host.write_values(triangles, &[0, 1, 2]).unwrap();
        let canvas = host.create(canvas_schema(), Domain::Grid(12, 9)).unwrap();
        let all: Capabilities = Capability::ALL.into_iter().collect();
        let mut scope = Scope::new(&mut host, all);
        let at = |inputs: Vec<Handle>, domain| Invocation {
            inputs,
            params: vec![],
            domain,
            context: Context::default(),
        };
        let plain = rasterize(
            &at(vec![vertices, triangles], Domain::Grid(12, 9)),
            &mut scope,
        )
        .unwrap()[0];
        let drawn = rasterize_canvas(
            &at(vec![vertices, triangles, canvas], Domain::Grid(12, 9)),
            &mut scope,
        )
        .unwrap()[0];
        assert_eq!(scope.size(drawn).unwrap(), Domain::Grid(12, 9));
        assert_eq!(scope.values(drawn).unwrap(), scope.values(plain).unwrap());
        // A domain other than the canvas's is refused before anything is drawn.
        let refused = rasterize_canvas(
            &at(vec![vertices, triangles, canvas], Domain::Grid(8, 8)),
            &mut scope,
        )
        .unwrap_err();
        assert!(
            format!("{refused:?}").contains("size(canvas)"),
            "{refused:?}"
        );
    }

    #[test]
    fn perspective_correct_interpolation_uses_w() {
        // A triangle whose far vertex has w = 4: with perspective-correct
        // interpolation the uv at the midpoint of the edge from the near
        // vertex (w = 1) leans toward the near one.
        let mut vertices = Vec::new();
        vertices.extend(vertex([-1.0, -1.0, 0.1, 1.0], [0.0; 3], [0.0, 0.0]));
        vertices.extend(vertex([4.0, -4.0, 0.4, 4.0], [0.0; 3], [1.0, 0.0]));
        vertices.extend(vertex([-1.0, 1.0, 0.1, 1.0], [0.0; 3], [0.0, 1.0]));
        let triangles: Vec<u32> = [0, 1, 2].into_iter().collect();
        let words = rasterize_words(&vertices, &triangles, 16, 16);
        // Bottom row, middle: screen midpoint between vertex 0 and 1.
        let (s, id) = surface(&words, 7, 15, 16);
        assert_eq!(id, 0);
        assert!(s[4] > 0.15 && s[4] < 0.3, "u is {}, not linear 0.5", s[4]);
    }
}
