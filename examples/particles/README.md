# Particles

The first scene whose data cannot be an image: a `Rows<Particle>` of
structs advanced every frame and drawn only at the end.

```text
seed (Rows) -> step (Rows, fed back) -> draw (Image)
```

`particles.step@1` runs once per element, reads its own previous element
with `sample`, a neighbour with `at`, and the count with `count`; the Field
stays resident on the GPU between frames. `particles.draw@1` gathers every
particle per pixel inside a bounded `loop`. Hold the mouse button to pull
the particles toward the pointer.

```sh
cargo run --release -p orla-player -- examples/particles/scene.osn --scale 3
```

The replay in `tests/` seeds four particles without spread and checks that
they fall to the floor.
