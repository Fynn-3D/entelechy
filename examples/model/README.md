# Model

A textured model from a glTF file. `orla-import` writes the vertices
(`{position, normal, uv}`, node transforms baked in), the triangles and
the base colour texture as files the scene reads and watches:

```text
input mesh: Model = "assets/crate.vertices.field";
input faces: Triangles = "assets/crate.triangles.field";
input albedo: Image = "assets/crate.albedo.png";
```

`model.project` orbits and projects the vertices, the raster package
lands the triangles, and `model.shade` samples the texture at each
pixel's uv with `at_linear`. The runtime does not read glTF; the importer
is a tool that speaks the Host's formats. It reports and skips skins,
animations, sparse accessors and JPEG textures. `orla-export` writes the
three files back as a GLB.

`source/crate.glb` is a hand-written cube with per-face normals and uvs,
an embedded PNG texture, and a node transform.

```sh
orla import examples/model/source/crate.glb --out examples/model/assets
orla export examples/model/assets/crate.vertices.field examples/model/assets/crate.triangles.field examples/model/assets/crate.albedo.png --out crate.glb
orla player examples/model/scene.osn --backend gpu
orla replay examples/model/tests/model.replay
```
