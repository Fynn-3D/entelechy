//! The reference rasterizer: the CPU side of the native `rasterize`
//! operation, under the contract in [`orla_rules::raster`]. It is
//! device emulation, not a product algorithm; shading, projection, and
//! geometry live in Kernels on either side of it. No culling and no
//! clipping: a triangle with any vertex at `w <= 0` is skipped.
//! Triangles are drawn in row order; a nearer fragment replaces a
//! farther one, and an equal depth keeps the earlier triangle.

use orla_abi::{Handle, Invocation, Scope};

use crate::ExecutionError;

pub use orla_rules::raster::{
    SURFACE_WORDS, TRIANGLE_WORDS, VERTEX_WORDS, check_rasterize, surface_schema, triangle_schema,
    vertex_schema,
};

/// Rasterizes on the CPU into a new surface grid owned by the scope's Host.
pub fn rasterize(
    invocation: &Invocation,
    scope: &mut Scope<'_>,
) -> Result<Vec<Handle>, ExecutionError> {
    let (vertices, triangles) = check_rasterize(invocation, scope)?;
    let vertex_words = scope.words(vertices)?.to_vec();
    let triangle_words = scope.words(triangles)?.to_vec();
    let domain = invocation.domain;
    let words = rasterize_words(
        &vertex_words,
        &triangle_words,
        domain.width(),
        domain.height(),
    );
    let output = scope.create(surface_schema(), domain)?;
    if let Err(error) = scope.write_all(output, &words) {
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

/// The pure rasterization: the surface words for the given vertices and
/// triangles over a `width` by `height` grid.
pub fn rasterize_words(vertices: &[u32], triangles: &[u32], width: u32, height: u32) -> Vec<u32> {
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
                depth[pixel] = z;
                let inv_w = l0 * a.inv_w + l1 * b.inv_w + l2 * c.inv_w;
                let lerp = |va: f32, vb: f32, vc: f32| {
                    (l0 * va * a.inv_w + l1 * vb * b.inv_w + l2 * vc * c.inv_w) / inv_w
                };
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
