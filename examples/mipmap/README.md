# mipmap

A checkered ground running to the horizon, drawn twice from the same
texture: once reading it as it is, once reading the level each pixel
asks for. The first crawls with moiré from the middle of the picture
back; the second resolves into the texture's own average, with the red
grid still readable as far as it goes.

```sh
cargo run --release -p orla-player -- examples/mipmap/scene.osn
cargo run --release -- run examples/mipmap/scene.osn --window 800x450 --param mip=false --out flat.ppm
cargo run --release -- replay examples/mipmap/tests/ground.replay
```

## What it is for

`at_linear` blends the four texels around a place, which is right for a
surface about the size of the screen and wrong for one running away from
it: once a pixel covers more than a texel, the blend samples a moving
subset of them and the picture crawls. The old answer is to keep the
texture at several scales and read the scale the pixel asks for, and it
needs nothing the language does not have:

- **The strip** is built by [`shared/mip.osn`](../shared/mip.osn), a
  sub-graph placed `levels` times with `carry`: each placement averages
  four texels of the level above into one of the level below, into one
  Field `side` wide and `2 * side` tall. A placement does not know which
  repetition it is, so the level being written rides beside the strip as
  a row of one `Int` and is carried on with it.
- **The level** comes from the surface. A rasterized surface carries its
  `uv` at every pixel, so the pixel next door says how fast the texture
  is moving across the screen — the difference between them is what a
  derivative would tell a fragment shader, and the triangle's `id` says
  whether the neighbour is on the same surface at all. That is
  [`mip_wanted`](../shared/mip.orla); [`mip_at`](../shared/mip.orla)
  reads the two levels either side of it and blends.

The same replay checks both on all three backends: two pixels a little
apart in the distance are the same grey with the levels and are nearly
black and nearly white without them, which is the crawl written down.

## What it does not give

Anisotropy. A surface seen at a grazing angle is minified along one
axis and not the other, and one level for both blurs it along the axis
that did not need it — the distance here goes grey a little early for
that reason. Several taps along the wider axis would fix it, in the
same library and at the same kind of cost; nothing in the language
stands in the way.
