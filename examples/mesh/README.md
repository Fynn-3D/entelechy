# Mesh

A subdivided torus drawn by the raster package.

- `mesh.torus` and `mesh.torus_faces` build the vertex and triangle rows.
- `mesh.split_vertices` and `mesh.split_faces` split each triangle into
  four, one level per placement of the sub-graph `level.osn`, placed
  `level` times with `carry` (`--param level=2` is two levels).
- `mesh.project` projects the vertices into clip space.
- The [raster package](../../packages/raster.orlapkg/kernels/raster.orla)
  turns the rows into a `Grid<Surface>` of depth, normal, uv and triangle
  id, the same on all three backends (`tests/edges.replay`).
- `mesh.shade` lights the surface.

`tests/native.osn` puts `native.rasterize@1` in the package's place: bit
for bit on the interpreter and the JIT; on the GPU it is the device's
rasterizer, and edge pixels may differ. `tests/near.replay` pins the
rasterizer's edge contract: a triangle with a corner at `w <= 0` is
skipped whole, and so is a triangle whose three corners are one vertex.

```sh
orla player examples/mesh/scene.osn --backend gpu
orla replay examples/mesh/tests/torus.replay
orla replay examples/mesh/tests/near.replay
```
