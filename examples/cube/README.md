# Cube

Matrices at work in one Kernel. The cube's placement is a `Mat4` built
from a `Mat3` rotation (two rotations composed by `*`) and a translation,
its columns joined as `vec4(r[0], 0.0)`;
the solid is ray-marched in the cube's own space, where the box distance
is simple, by taking the eye and the ray through `inverse(model)`; the
normal comes back to the world through the rotation. The edges are a
corner table (`[Vec3; 8]`) and an edge table (`[IVec2; 12]`) projected
through `perspective(f) * model`, a `Mat4 * Vec4` per corner, and every
pixel lights up near the closest of the twelve segments, an x-ray
wireframe over the shaded solid.

```sh
cargo run --release -p orla-player -- examples/cube/scene.osn
cargo run -- replay examples/cube/tests/cube.replay
```
