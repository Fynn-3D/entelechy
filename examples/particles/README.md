# Particles

The first scene whose data cannot be an image: a `Rows<Particle>` of
structs advanced every frame, splatted onto a canvas, and drawn from it.

```text
seed (Rows) -> step (Rows, fed back) -> splat (Grid, onto fade) -> draw (Image)
clear (Grid) -> fade (Grid, fed back from splat) -^
```

`particles.step@1` runs once per element, reads its own previous element
with `sample` and the count with `count`; the Field stays resident on the
GPU between frames. `particles.splat@1` is a scatter: it maps over the
particles and yields, per particle, the canvas cell under it and its
tint, and the output is the canvas with every tint summed into its cell,
in element order, the same on both backends. No pixel walks the
particles: the cost is one write per particle, however many pixels there
are. `particles.fade@1` dims last frame's canvas through Graph state, so
the particles leave trails, and `particles.draw@1` gathers a fixed window
of cells around each pixel into a glow. Hold the mouse button to pull the
particles toward the pointer.

```sh
cargo run --release -p orla-player -- examples/particles/scene.osn --scale 3
cargo run -- replay examples/particles/tests/settle.replay
```

The replay in `tests/` seeds one particle without spread and no fade,
pins the canvas cell it lands in and the pixel under it, and follows it
as it falls.
