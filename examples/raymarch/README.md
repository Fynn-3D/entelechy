# Ray march

A signed-distance scene (sphere, torus, rounded box, checker floor) rendered
by ray marching with soft shadows. The march, the shadow, and the material
selection are ordinary `loop` and `if` expressions in `kernels/march.orla`;
the camera orbits because the Kernel declares `time.read`.

```sh
cargo run --release -p orla-player -- examples/raymarch/scene.osn
```

Edit the `scene` function to add or move shapes, change `steps` in
`scene.osn`, or lower the loop bounds to trade quality for speed. A Kernel
that fails to compile leaves the previous version rendering.
