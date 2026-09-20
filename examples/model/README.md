# Model

A model from a file. `orla-import` reads a glTF or GLB and writes what a
scene takes as inputs: the vertices as rows of `{position, normal, uv}`
with every node's transform baked in, the triangles as rows of three
indices, and the base colour texture as a PNG. The scene declares the
three as inputs the Host reads and watches:

```text
input mesh: Model = "assets/crate.vertices.field";
input faces: Triangles = "assets/crate.triangles.field";
input albedo: Image = "assets/crate.albedo.png";
```

`model.project` turns the vertices into clip space under an orbit, the
backend's native rasterizer lands the triangles, and `model.shade` reads
the texture at each pixel's uv with `at_linear`, lit by a key light and
the sky. Nothing in the runtime knows glTF; the importer is a tool that
speaks the Host's formats, and a model in any other format needs only
another tool.

The crate in `source/crate.glb` was written by hand for the example: a
cube with per-face normals and uvs, a plank texture embedded as PNG, and
a node that turns and halves it, which the import applies. Skins,
animations, sparse accessors and JPEG textures are reported and left
out; convert a JPEG to PNG and it comes in.

```sh
cargo run -p orla-import -- examples/model/source/crate.glb --out examples/model/assets
cargo run --release -p orla-player -- examples/model/scene.osn --backend gpu
cargo run -- replay examples/model/tests/model.replay
```
