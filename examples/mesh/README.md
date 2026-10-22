# Mesh

A subdivided torus drawn by the raster package.

- `mesh.torus` and `mesh.torus_faces` build the vertex and triangle rows.
- `mesh.split_vertices` and `mesh.split_faces` split each triangle into
  four, one level per placement of the sub-graph `level.esn`, placed
  `level` times with `carry` (`--param level=2` is two levels).
- `mesh.project` projects the vertices into clip space.
- The [raster package](../../packages/raster.entpkg/kernels/triangle.ent)
  turns the rows into a `Grid<Surface>` of depth, normal, uv and triangle
  id, the same on all three backends (`tests/edges.replay`).
- `mesh.shade` lights the surface.

`tests/native.esn` puts `native.rasterize@1` in the package's place: bit
for bit on the interpreter and the JIT; on the GPU the device's
rasterizer draws the larger triangles, and their edge pixels may differ. `tests/canvas.replay` holds the
package's canvas preset to its plain one and to `native.rasterize@3`. The
rasterizer's own contract at its edges, near planes and degenerate
triangles is held by the raster package's tests
([`packages/raster.entpkg/tests`](../../packages/raster.entpkg/tests/)).

```sh
entelechy player examples/mesh/scene.esn --backend gpu
entelechy replay examples/mesh/tests/torus.replay
```

## Parts that run alone

- [model](parts/model/README.md): a GLB crate imported by `entelechy import`, rasterized and textured with `at_linear`; `entelechy player examples/mesh/parts/model/scene.esn`, its replays `tests/model*.replay`.
- [detail](parts/detail/README.md): geometry sized inside the frame by `domain = value(detail) * value(detail)`; `entelechy player examples/mesh/parts/detail/scene.esn`, its replays `tests/detail*.replay`.
