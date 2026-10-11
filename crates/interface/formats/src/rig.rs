//! The row layouts of a mesh and its rig
//! ([RIGS](../../../../docs/RIGS.md)), shared by import and export, and
//! their matrix product.

use entelechy_host::{Scalar, Schema};

/// One part of a mesh or of its rig, as a file of rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// A vertex at rest: position, normal, uv (`f×8`).
    Vertices,
    /// A vertex at rest with the tangent a normal map is read along:
    /// position, normal, uv, tangent (`f×12`; the tangent's `w` the
    /// handedness, the bitangent `cross(normal, tangent) * w`).
    TangentVertices,
    /// Three vertex indices (`i×3`).
    Triangles,
    /// A vertex's four joints, then their weights (`i×4, f×4`).
    Weights,
    /// A joint's parent, then its inverse bind matrix (`i, f×16`).
    Skeleton,
    /// A joint's local translation, rotation and scale in one frame
    /// (`f×10`).
    Clip,
}

impl Part {
    /// Every part.
    pub const ALL: [Part; 6] = [
        Part::Vertices,
        Part::TangentVertices,
        Part::Triangles,
        Part::Weights,
        Part::Skeleton,
        Part::Clip,
    ];

    /// The layout of one row.
    pub fn schema(self) -> Schema {
        let (ints, floats) = match self {
            Part::Vertices => (0, 8),
            Part::TangentVertices => (0, 12),
            Part::Triangles => (3, 0),
            Part::Weights => (4, 4),
            Part::Skeleton => (1, 16),
            Part::Clip => (0, 10),
        };
        Schema::new([vec![Scalar::Int; ints], vec![Scalar::Float; floats]].concat())
    }

    /// The part `schema`'s rows are, matched by scalars and ignoring
    /// storage encodings (`u16` triangles, `u8` joints).
    pub fn of(schema: &Schema) -> Option<Part> {
        Part::ALL
            .into_iter()
            .find(|part| part.schema().scalars() == schema.scalars())
    }
}

/// A 4×4 matrix, column by column, as glTF and the ABI keep one.
pub type Matrix = [f32; 16];

/// The product `a · b`.
pub fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = [0.0; 16];
    for column in 0..4 {
        for row in 0..4 {
            out[column * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[column * 4 + k]).sum();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Narrow triangles and joints are still their parts.
    #[test]
    fn a_part_kept_narrow_is_the_part_it_is() {
        let narrow = |part: Part, encoding: &str| {
            part.schema()
                .with_encoding(entelechy_host::encoding::Encoding::parse(encoding).unwrap())
                .unwrap()
        };
        assert_eq!(
            Part::of(&narrow(Part::Triangles, "u16")),
            Some(Part::Triangles)
        );
        assert_eq!(
            Part::of(&narrow(Part::Triangles, "u8")),
            Some(Part::Triangles)
        );
        assert_eq!(Part::of(&Part::Triangles.schema()), Some(Part::Triangles));
        assert_eq!(Part::of(&Part::Vertices.schema()), Some(Part::Vertices));
        assert_eq!(
            Part::of(&Part::TangentVertices.schema()),
            Some(Part::TangentVertices)
        );
        assert_eq!(Part::of(&Schema::pixel()), None);
    }
}
