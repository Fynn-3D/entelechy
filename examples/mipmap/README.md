# Mipmap

A checkered ground running to the horizon, drawn from one texture twice:
read as it is, and read at the level each pixel asks for.

- [`image.entpkg`](../../packages/image.entpkg/graphs/mipmap.esn)
  builds the level strip: a sub-graph placed `levels` times with `carry`,
  each averaging four texels into one, with the level number carried
  beside the strip as a row of one `Int`.
- [`mip_wanted`](../../packages/image.entpkg/kernels/mipmap.ent) takes
  the level from the uv difference to the neighboring pixel, using the
  triangle `id` to stay on one surface; `mip_at` blends the two levels
  either side.

The levels are isotropic: one level for both axes. `tests/ground.replay`
checks, on all three backends, that two distant pixels are the same gray
with levels and far apart without them.

```sh
entelechy player examples/mipmap/scene.esn
entelechy run examples/mipmap/scene.esn --window 800x450 --param mip=false --out flat.ppm
entelechy replay examples/mipmap/tests/ground.replay
```
