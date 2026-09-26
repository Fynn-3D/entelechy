# Cube

Matrices in one Kernel. The cube's placement is a `Mat4` built from a
`Mat3` rotation and a translation; the solid is ray-marched in the cube's
own space through `inverse(model)`. The edges are a corner table
(`[Vec3; 8]`) and an edge table (`[IVec2; 12]`) projected through
`perspective(f) * model`, drawn as an x-ray wireframe over the solid.

Parameters: `lens` is a `Mat4`, `edges` an `[IVec2; 12]`, and `style` a
`Style` struct declared in the Kernel's document. The replay sets a
`mat4(..)` lens and a positional `Style`.

```sh
cargo run --release -p orla-player -- examples/cube/scene.osn
cargo run -- replay examples/cube/tests/cube.replay
```
