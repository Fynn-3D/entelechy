# Mesh

A torus that never becomes pixels until the last node. `mesh.torus` builds
one vertex per element and `mesh.torus_faces` two triangles per lattice
cell, both as rows; `mesh.split_vertices` and `mesh.split_faces` subdivide
every triangle into four, sizing their rows from the inputs with
`domain = count(mesh) + 3 * count(faces);` and `domain = 4 * count(faces);`;
`mesh.project` rotates and projects the vertices into clip space; `native.rasterize@1`, the backend's rasterizer, turns the two
rows into a `Grid<Surface>` of depth, normal, uv, and triangle id; and
`mesh.shade` lights that surface. The rasterizer is device access, not an
algorithm: on the GPU it is a render pipeline, on the CPU a reference
implementation, and they agree on every interior pixel.

The subdivision is `level.osn`, one level as a sub-graph, placed `level`
times with `times = level;` and `carry`, so `param level 2` over the
protocol (or `--param level=2`) is two levels, the graph resized while
the scene runs; the replay checks the rounder silhouette.

`tests/near.replay` is the rasterizer's contract at its edges, where the
three backends have to agree and a device would rather not: a triangle
with a corner at `w <= 0` is skipped whole, not clipped at the near
plane and drawn in part, and a triangle whose three corners are one
vertex is skipped too. The second is how a Kernel culls — a map over the
triangles writes a collapsed one for what it does not want, and the row
keeps the length its domain fixed. It is worth measuring before relying
on it: culling that way removes the rasterizer's per-triangle work and
nothing else, because the Kernel that projects the vertices runs over
every one of them either way.

```sh
cargo run --release -p orla-player -- examples/mesh/scene.osn --backend gpu
cargo run -- replay examples/mesh/tests/torus.replay
cargo run -- replay examples/mesh/tests/near.replay
```
