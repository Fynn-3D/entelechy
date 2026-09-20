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
//! No culling and no clipping: a triangle with any vertex at `w <= 0` is
//! skipped. Triangles are drawn in row order; a nearer fragment replaces
//! a farther one, and an equal depth keeps the earlier triangle.

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
    let [vertices, triangles] = invocation.inputs[..] else {
        return Err(ExecutionError::InputCount {
            expected: 2,
            actual: invocation.inputs.len(),
        });
    };
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
    Ok((vertices, triangles))
}
