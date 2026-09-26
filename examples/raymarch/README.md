# Ray march

A signed-distance scene (sphere, torus, rounded box, checker floor) ray
marched with soft shadows. The march, the shadow and the material choice
are `loop` and `if` expressions in `kernels/march.orla`, carrying
`struct` values. The Kernel declares `time.read`, so the camera orbits.
Edit the `scene` function or `steps` in `scene.osn` while it runs.

```sh
cargo run --release -p orla-player -- examples/raymarch/scene.osn
```
