# image

What Entelechy does to a picture, knowing nothing of the world it shows:
color spaces, blending and masks, two pictures compared, filters, a
grade, tone and its curves,
glow, metered and local exposure, a film's steps, scaling and
sharpening, antialiasing within a frame, histograms,
distance fields, mipmaps, pixel art and sprite sheets. Each effect is a
word a kernel calls on a pixel, its look data a scene makes (`Grade`),
its kernel a thin pass; what needs the world (a
G-buffer, a lens, light) is [`light`](../light.entpkg/README.md)'s.

**Layer:** Graphics. **Uses:** [`random`](../random.entpkg/README.md).

```text
use "image.entpkg";
```

## Kernels

- `kernels/antialias.ent` — Antialiasing within one frame (FXAA, `fxaa`).
  Kernels: `image.fxaa@1`.
- `kernels/blend.ent` — Blending, and any picture kept to part of another by a mask. Kernels: `image.blend@1`, `image.masked@1`.
- `kernels/bloom.ent` — Glow round bright light, every size at once: the
  bright parts shrunk by halves and grown back (`bloom_at`). Kernels:
  `image.bloom_down@1`, `image.bloom_up@1`.
- `kernels/color.ent` — Color as numbers. Kernels: `image.encode@1`,
  `image.decode@1`.
- `kernels/distance.ent` — Distance. Kernels: `image.seeds@1`,
  `image.first_step@1`, `image.next_step@1`, `image.flood@1`,
  `image.distance@1`.
- `kernels/compare.ent` — Two pictures compared: each pixel's difference
  (`color_apart`), and how far apart they are as a whole, the most a pixel
  differs and how many differ by more than a step. Kernels:
  `image.difference@1`, `image.apart_rows@1`, `image.apart@1`.
- `kernels/display.ent` — The picture seen: glow laid in, toned
  (`tone_curve`), vivid and warm (`tone_warmth`), the corners darkened,
  sRGB. Kernels: `image.blank@1`, `image.bright@2`, `image.spread@1`,
  `image.tonemap@3`.
- `kernels/exposure.ent` — Metered exposure as an eye adapts, and local
  exposure evening bright and dark neighborhoods. Kernels:
  `image.exposure_tiles@1`, `image.exposure_histogram@1`,
  `image.exposure_meter@1`, `image.exposure_start@1`,
  `image.exposure_apply@1`, `image.exposure_local_base@1`,
  `image.exposure_local_blur@1`, `image.exposure_local@1`.
- `kernels/film.ent` — A camera's and a film's steps over a picture: the
  bright part taken smaller, the lens's fringe, defocus by distance,
  darkened corners, grain, bars, halation, a split and an S curve
  (`fringed`, `defocus`, `vignetted`, `film_grain`, `film_bars`,
  `film_halation`, `film_split`, `film_curve`). Kernels: `image.bright@1`.
- `kernels/filter.ent` — Filters. Kernels: `image.blur@1`,
  `image.shrink@1`, `image.convolve@1`, `image.sharpen@1`,
  `image.edges@1`, `image.morph@1`, `image.resize@1`.
- `kernels/grade.ent` — A grade as data (`Grade`: exposure, white
  balance, saturation, contrast, gain, a drain toward one color), blended
  from one look to another (`grade_color`, `grade_mix`,
  `grade_balance`). Kernels: `image.grade_start@1`, `image.grade@3`.
- `kernels/histogram.ent` — Histograms. Kernels: `image.histogram@1`,
  `image.cumulate@1`, `image.equalize@1`.
- `kernels/mipmap.ent` — A mip strip. Kernels: `image.mip_first@1`,
  `image.mip_next@1`, `image.mip_halve@1`.
- `kernels/pixel_art.ent` — Pixel effects. Kernels: `image.pixelate@1`,
  `image.palette@2`, `image.ramp@1`, `image.dither@1`,
  `image.outline@1`, `image.shadow@1`.
- `kernels/sprite_sheet.ent` — Sheets.
- `kernels/tone.ent` — Tone and color grading. Kernels:
  `image.adjust@1`, `image.levels@1`, `image.curve@1`,
  `image.neutral@1`, `image.lut@1`.
- `kernels/tone_map.ent` — Tone curves, ACES's fit and AgX, light of any
  brightness to 0..1 (`tone_curve`), sharpened against its neighbors
  (`tone_sharpened`), made vivid and warm (`tone_warmth`).
- `kernels/upscale.ent` — A picture scaled to another size, edge-aware
  (FSR 1's EASU, `upscaled`) or nearest-neighbor (`upscaled_near`), and a
  sharpening that cannot overshoot (RCAS). Kernels: `image.upscale@1`,
  `image.contrast_sharpen@1`.
- `kernels/wrap.ent` — Wrap addressing for a linear read.

## Graph presets

- `graphs/bloom.esn` — Glow round bright light, six levels down and back
  up; what `image.tonemap` takes as its `bloom`.
- `graphs/blur.esn` — A Gaussian blur, both ways, as a sub-graph.
- `graphs/compare.esn` — Two pictures compared: the difference to look at
  and how far apart, as a test holds a scene to another's picture or a
  person sees where two renders part (two pictures read as inputs, then
  `entelechy replay` with `show apart.apart[0]`).
- `graphs/distance.esn` — The signed distance from every pixel to a
  shape, in pixels, negative inside.
- `graphs/equalize.esn` — Histogram equalization as a sub-graph.
- `graphs/exposure.esn` — Metered exposure and local exposure.
- `graphs/flood.esn` — Jump flooding as a sub-graph.
- `graphs/glow.esn` — A glow: a near and a far
  copy of what is past white.
- `graphs/flood_pass.esn` — One jump-flooding pass and the step after
  it.
- `graphs/mipmap.esn` — The mip strip of a square texture, as a sub-
  graph.
- `graphs/mipmap_level.esn` — One halving of a mip strip.
- `graphs/sharpen.esn` — Unsharp masking as a sub-graph.
- `graphs/tonemap.esn` — The picture seen at its own size: glow, tone
  and sRGB, sharpened.
- `graphs/upscale.esn` — Scaled to the window and sharpened.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`antialias.replay`, `bloom.replay`, `compare.replay`, `display.replay`,
`exposure.replay`, `finish.replay`, `grade.replay`, `looks.replay`,
`operations.replay`, `upscale.replay`. Every Kernel of the package is
reached by one of them.

Used by the [darkroom](../../examples/darkroom/README.md), [image](../../examples/image/README.md), [mipmap](../../examples/mipmap/README.md), [nebula](../../examples/nebula/README.md) examples.
