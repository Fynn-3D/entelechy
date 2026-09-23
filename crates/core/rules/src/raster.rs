//! The native `rasterize` operation's contract, which every backend
//! keeps, and the reference rasterizer a backend with no rasterizer of
//! its own runs: the interpreter and the JIT alike, neither through the
//! other. It is device emulation, not a product algorithm; shading,
//! projection, and geometry live in Kernels on either side of it.
//!
//! Contract:
//! - input 0, `vertices`: rows laid out as `clip: Vec4, normal: Vec3,
//!   uv: Vec2` (nine floats). `clip` is a clip-space position as a vertex
//!   shader would emit it: after the divide by `w`, `x` and `y` span
//!   `[-1, 1]` (y up) and `z` spans `[0, 1]` (near to far).
//! - input 1, `triangles`: rows of three `Int` vertex indices, clamped
//!   into range.
//! - output, `surface`: a grid laid out as `depth: Float, normal: Vec3,
//!   uv: Vec2, id: Int`, the nearest fragment at each pixel centre with
//!   perspective-correct attributes and the triangle's row index as `id`.
//!   Pixels nothing covers hold depth `1.0`, zeros, and id `-1`.
//!
//! No culling and no clipping: a triangle with any vertex at `w <= 0`,
//! or any clip component that is not finite, is skipped whole — not
//! clipped at the near plane and drawn in part. A triangle of no area is
//! skipped too, so a Kernel says "not this one" by putting its three
//! corners on one vertex, which is how a fixed-length row of triangles
//! draws a varying number of them. Triangles are drawn in row order; a
//! nearer fragment replaces a farther one, and an equal depth keeps the
//! earlier triangle.
//!
//! The cutout form takes one more input and one parameter:
//! - input 2, `mask`: an `Image` whose alpha says what the surface
//!   covers. A fragment whose alpha, read at its own `uv` as
//!   `at_linear(mask, uv * size(mask))` reads it, is below `cutout` is
//!   not drawn at all: it neither writes the surface nor settles depth,
//!   which is what a leaf's edge needs.
//! - param 0, `cutout`: the alpha a fragment must have, `0.5` by
//!   default.

use orla_abi::{Handle, Invocation, Scalar, Schema, Scope, Shape};

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
/// reads it: the four around it, blended, and clamped at the edges.
/// Every backend samples it this way, so a cutout's edge is the same
/// pixel on all of them.
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

/// The surface layout: `depth: Float, normal: Vec3, uv: Vec2, id: Int`.
pub fn surface_schema() -> Schema {
    let mut scalars = vec![Scalar::Float; SURFACE_WORDS - 1];
    scalars.push(Scalar::Int);
    Schema::new(scalars)
}

/// Checks the invocation against the contract, returning the vertex and
/// triangle handles and the output grid. Every backend applies this.
pub fn check_rasterize(
    invocation: &Invocation,
    scope: &Scope<'_>,
) -> Result<(Handle, Handle), ExecutionError> {
    let (vertices, triangles, _) = check_rasterize_with(invocation, scope, false)?;
    Ok((vertices, triangles))
}

/// The same, for the form that takes a mask and a cutout: the third
/// input and the parameter are checked too, and what comes back is the
/// mask with the alpha a fragment must reach.
pub fn check_rasterize_cutout(
    invocation: &Invocation,
    scope: &Scope<'_>,
) -> Result<(Handle, Handle, Handle, f32), ExecutionError> {
    let (vertices, triangles, mask) = check_rasterize_with(invocation, scope, true)?;
    let mask = mask.expect("a cutout asked for its mask");
    let cutout = match invocation.params.first() {
        Some(orla_abi::Value::Float(value)) => *value,
        _ => 0.5,
    };
    Ok((vertices, triangles, mask, cutout))
}

fn check_rasterize_with(
    invocation: &Invocation,
    scope: &Scope<'_>,
    cutout: bool,
) -> Result<(Handle, Handle, Option<Handle>), ExecutionError> {
    let wanted = if cutout { 3 } else { 2 };
    if invocation.inputs.len() != wanted {
        return Err(ExecutionError::InputCount {
            expected: wanted,
            actual: invocation.inputs.len(),
        });
    }
    let (vertices, triangles) = (invocation.inputs[0], invocation.inputs[1]);
    let mask = cutout.then(|| invocation.inputs[2]);
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
    if let Some(mask) = mask {
        if scope.schema(mask)? != mask_schema() {
            return Err(contract("`mask` must be an Image".into()));
        }
        if scope.size(mask)?.shape() != Shape::Grid {
            return Err(contract("`mask` must be a grid".into()));
        }
    }
    Ok((vertices, triangles, mask))
}

/// What a cutout tests a fragment against: the mask's words, its extent,
/// and the alpha a fragment must reach to be drawn.
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
    let vertex_words = scope.values(vertices)?.into_owned();
    let triangle_words = scope.values(triangles)?.into_owned();
    let domain = invocation.domain;
    let words = rasterize_words(
        &vertex_words,
        &triangle_words,
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

/// Rasterizes with a cutout: a fragment whose alpha in the mask is
/// below the parameter is not drawn at all.
pub fn rasterize_cutout(
    invocation: &Invocation,
    scope: &mut Scope<'_>,
) -> Result<Vec<Handle>, ExecutionError> {
    let (vertices, triangles, mask, alpha) = check_rasterize_cutout(invocation, scope)?;
    let vertex_words = scope.values(vertices)?.into_owned();
    let triangle_words = scope.values(triangles)?.into_owned();
    let mask_words = scope.values(mask)?.into_owned();
    let mask_size = scope.size(mask)?;
    let domain = invocation.domain;
    let words = rasterize_masked(
        &vertex_words,
        &triangle_words,
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
                // A fragment the mask does not cover is not drawn: it
                // settles no depth either, so what is behind it is seen.
                if let Some(cutout) = &cutout {
                    let uv = [
                        lerp(a.uv[0], b.uv[0], c.uv[0]),
                        lerp(a.uv[1], b.uv[1], c.uv[1]),
                    ];
                    if mask_alpha(cutout.words, cutout.width, cutout.height, uv) < cutout.alpha {
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

#[cfg(test)]
mod tests {
    use super::*;

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
