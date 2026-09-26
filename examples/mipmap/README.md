# Mipmap

A checkered ground running to the horizon, drawn from one texture twice:
read as it is, and read at the level each pixel asks for.

- [`picture.orlapkg`](../../packages/picture.orlapkg/graphs/mip.osn)
  builds the level strip: a sub-graph placed `levels` times with `carry`,
  each averaging four texels into one, with the level number carried
  beside the strip as a row of one `Int`.
- [`mip_wanted`](../../packages/picture.orlapkg/kernels/mip.orla) takes
  the level from the uv difference to the neighbouring pixel, using the
  triangle `id` to stay on one surface; `mip_at` blends the two levels
  either side.

The levels are isotropic: one level for both axes. `tests/ground.replay`
checks, on all three backends, that two distant pixels are the same grey
with levels and far apart without them.

```sh
cargo run --release -p orla-player -- examples/mipmap/scene.osn
cargo run --release -- run examples/mipmap/scene.osn --window 800x450 --param mip=false --out flat.ppm
cargo run --release -- replay examples/mipmap/tests/ground.replay
```
