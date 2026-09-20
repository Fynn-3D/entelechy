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

```sh
cargo run --release -p orla-player -- examples/mesh/scene.osn --backend gpu
cargo run -- replay examples/mesh/tests/torus.replay
```
