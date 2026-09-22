//! The native `rasterize` operation's contract, which every backend
//! keeps; the reference rasterizer itself is the CPU crate's
//! `raster`. It is device emulation, not a product algorithm; shading,
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
