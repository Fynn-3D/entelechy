# Particles

A `Rows<Particle>` of structs advanced every frame, splatted onto a
canvas, and drawn from it.

```text
seed (Rows) -> step (Rows, fed back) -> splat (Grid, onto fade) -> draw (Image)
clear (Grid) -> fade (Grid, fed back from splat) -^
```

- `particles.step@1` advances each particle from its previous value.
- `particles.splat@1` is a scatter: each particle adds its tint to the
  canvas cell under it, one write per particle.
- `particles.fade@1` dims last frame's canvas, leaving trails.
- `particles.draw@1` gathers a window of cells around each pixel into a
  glow.

Hold the mouse button to pull the particles toward the pointer.
`tests/settle.replay` follows one particle and pins its cell and pixel.

```sh
cargo run --release -p orla-player -- examples/particles/scene.osn --scale 3
cargo run -- replay examples/particles/tests/settle.replay
```
