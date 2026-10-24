# effects

Effects as light gathered along each pixel's ray, over a picture with
its depth. Its words:

- **Functions.** Each form's light along a ray (`body` for an orb, a
  beam, a pillar, a cone, a funnel and an aura; `shell`, `wave`,
  `sigil`, `motes`, `sparks`, `shards`, `bolts`, `slash`, `glyph`,
  `darts`), the light primitives under them (`sphere_span`, `stroke`,
  `jag`, `shaft`, `on_floor`, `rising`), a matter's color, flicker and
  grain, a layer placed and keyed at a moment (`layer_now`), and every
  live layer's light along one ray (`lives_light`). A program's own
  Kernel calls any of them.
- **Kernels.** The layers of each cast now (`effects.now`), their light
  over a picture seen through `geometry`'s `Lens` (`effects.light`), that
  light laid over the picture (`effects.draw`), and a cel look
  (`effects.cel`).
- **Data.** An effect is rows: its layers of forms in time, each at an
  anchor between where a cast goes from and to, moved its own way and
  keyed over its life, made of an element's matter. Which effects a
  program has and their matters are its own tables (the study's
  `spell_book` is one book of them).

**Layer:** Graphics. **Uses:** [`geometry`](../geometry.entpkg/README.md), [`random`](../random.entpkg/README.md).

```text
use "effects.entpkg";
```

## Kernels

- `kernels/draw.ent` — The live layers drawn over a picture whose alpha
  holds how far each pixel is (1 + the distance; 0 for the sky), as
  `glow.ent` draws. Kernels: `effects.light@2`, `effects.draw@1`,
  `effects.cel@1`.
- `kernels/effect.ent` — The rows an effect is made of.
- `kernels/form.ent` — The forms as light along a ray (a crystal's
  distance among them, for shards).
- `kernels/glow.ent` — Light gathered along a ray (`Glow`): a sphere's
  span, a segment's nearest pass, a stroke, a jagged line, a shaft, a
  box's span, marks on the floor and motes rising.
- `kernels/matter.ent` — What a layer is made of, at a point.
- `kernels/moment.ent` — Each cast's layers as they are now. Kernels:
  `effects.now@1`.

## Graph presets

- `graphs/draw.esn` — Casts drawn over a picture.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`cast.replay`, over two effects and two matters of their own
(`tests/data/`). Every Kernel of the package is reached by one of them.
