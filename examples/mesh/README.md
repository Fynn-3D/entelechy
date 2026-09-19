# Mesh

A torus that never becomes pixels until the last node. `mesh.torus` builds
one vertex per element and `mesh.torus_faces` two triangles per lattice
cell, both as rows; `mesh.project` rotates and projects the vertices into
clip space; `native.rasterize@1`, the backend's rasterizer, turns the two
rows into a `Grid<Surface>` of depth, normal, uv, and triangle id; and
`mesh.shade` lights that surface. The rasterizer is device access, not an
algorithm: on the GPU it is a render pipeline, on the CPU a reference
implementation, and they agree on every interior pixel.

```sh
cargo run --release -p orla-player -- examples/mesh/scene.osn --backend gpu
cargo run -- replay examples/mesh/tests/torus.replay
```
