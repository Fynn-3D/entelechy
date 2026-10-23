# image

What Entelechy does to pictures: color spaces, blending, filters, tone
and its curves, histograms, distance fields, mipmaps, pixel art, sprite
sheets and a camera's finish.

**Layer:** Graphics. **Uses:** [`random`](../random.entpkg/README.md).

```text
use "image.entpkg";
```

## Kernels

- `kernels/blend.ent` — Blending. Kernels: `image.blend@1`.
- `kernels/color.ent` — Color as numbers. Kernels: `image.encode@1`,
  `image.decode@1`.
- `kernels/distance.ent` — Distance. Kernels: `image.seeds@1`,
  `image.first_step@1`, `image.next_step@1`, `image.flood@1`,
  `image.distance@1`.
- `kernels/film.ent` — A camera's finish, as Kernels over a picture of
  light (rgb, which may be far brighter than white) that keeps in w how
  far away what each pixel shows is. Kernels: `image.bright@1`,
  `image.grade@2`.
- `kernels/filter.ent` — Filters. Kernels: `image.blur@1`,
  `image.shrink@1`, `image.convolve@1`, `image.sharpen@1`,
  `image.edges@1`, `image.morph@1`, `image.resize@1`.
- `kernels/histogram.ent` — Histograms. Kernels: `image.zeros@1`,
  `image.histogram@1`, `image.cumulate@1`, `image.equalize@1`.
- `kernels/mipmap.ent` — A mip strip. Kernels: `image.mip_first@1`,
  `image.mip_one@1`, `image.mip_next@1`, `image.mip_halve@1`.
- `kernels/pixel_art.ent` — Pixel effects. Kernels: `image.pixelate@1`,
  `image.palette@2`, `image.ramp@1`, `image.dither@1`,
  `image.outline@1`, `image.shadow@1`.
- `kernels/sprite_sheet.ent` — Sheets.
- `kernels/tone.ent` — Tone and color grading. Kernels:
  `image.adjust@1`, `image.levels@1`, `image.curve@1`,
  `image.neutral@1`, `image.lut@1`.
- `kernels/tone_map.ent` — Tone curves, ACES's fit and AgX, light of any brightness to
  0..1. Kernels: `image.tonemap@2`.
- `kernels/wrap.ent` — Wrap addressing for a linear read.

## Graph presets

- `graphs/bloom.esn` — The glow round what is past white, as a sub-
  graph.
- `graphs/blur.esn` — A Gaussian blur, both ways, as a sub-graph.
- `graphs/display.esn` — Tone mapping as a sub-graph.
- `graphs/distance.esn` — The signed distance from every pixel to a
  shape, in pixels, negative inside.
- `graphs/equalize.esn` — Histogram equalization as a sub-graph.
- `graphs/finish.esn` — The whole finish, as a sub-graph, usually a
  scene's last node.
- `graphs/flood.esn` — Jump flooding as a sub-graph.
- `graphs/flood_pass.esn` — One jump-flooding pass and the step after
  it.
- `graphs/mipmap.esn` — The mip strip of a square texture, as a sub-
  graph.
- `graphs/mipmap_level.esn` — One halving of a mip strip.
- `graphs/nes.esn` — Coarse pixels held to a few colors.
- `graphs/sharpen.esn` — Unsharp masking as a sub-graph.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`looks.replay`, `operations.replay`. Every Kernel of the package is
reached by one of them.

Used by the [darkroom](../../examples/darkroom/README.md), [image](../../examples/image/README.md), [mipmap](../../examples/mipmap/README.md), [nebula](../../examples/nebula/README.md) examples.
