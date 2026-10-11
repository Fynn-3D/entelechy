# Mesh

A subdivided torus drawn by the raster package.

- `mesh.torus` and `mesh.torus_faces` build the vertex and triangle rows.
- `mesh.split_vertices` and `mesh.split_faces` split each triangle into
  four, one level per placement of the sub-graph `level.esn`, placed
  `level` times with `carry` (`--param level=2` is two levels).
- `mesh.project` projects the vertices into clip space.
- The [raster package](../../packages/raster.entpkg/kernels/raster.ent)
  turns the rows into a `Grid<Surface>` of depth, normal, uv and triangle
  id, the same on all three backends (`tests/edges.replay`).
- `mesh.shade` lights the surface.

`tests/native.esn` puts `native.rasterize@1` in the package's place: bit
for bit on the interpreter and the JIT; on the GPU it is the device's
rasterizer, and edge pixels may differ. `tests/canvas.replay` holds the
package's canvas preset to its plain one and to `native.rasterize@3`. `tests/near.replay` pins the
rasterizer's edge contract: a triangle with a corner at `w <= 0` is
skipped whole, and so is a triangle whose three corners are one vertex.
`tests/degenerate.replay` holds that a triangle with two corners at one
point draws nothing, and that the tiled forms draw what the one-pass forms
draw, on every backend: each triangle's box is found once and every step
reads it, so no step can round it differently from another.

```sh
entelechy player examples/mesh/scene.esn --backend gpu
entelechy replay examples/mesh/tests/torus.replay
entelechy replay examples/mesh/tests/near.replay
```
