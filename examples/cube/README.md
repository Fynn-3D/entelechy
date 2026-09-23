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

The camera, the edge table and the style are parameters: `lens` is the
`Mat4` the scene hands over whole (its focal length read back as
`lens[0].x`), `edges` an `[IVec2; 12]`, and `style` a `Style`, a struct
the Kernel's document declares at the top so the scene can write one by
name. The replay turns to a longer lens with a `mat4(..)` literal and
restyles the picture with the struct given by position, `[vec3(0.9, 0.9,
0.9), vec3(0.8, 0.1, 0.1), 2.0]`, which the parameter takes in its shape.

```sh
cargo run --release -p orla-player -- examples/cube/scene.osn
cargo run -- replay examples/cube/tests/cube.replay
```
