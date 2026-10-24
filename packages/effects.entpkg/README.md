# effects

Effects as light gathered along each pixel's ray, over a picture with
its depth: an effect is layers of forms in time (an orb, a beam, a
pillar, a cone, a shell, a funnel, an aura, a wave, a sigil, motes,
sparks, shards, bolts, a slash, a flash, a glyph, darts), each placed at
an anchor, moved its own way and keyed over its life, made of a matter
(its colors from edge to core, its heat, its noise moving its own way).
Which effects a game has, their layers and their matters are the game's
own rows, handed in as tables (the study's `spell_book` is one book of
them).

**Layer:** Graphics. **Uses:** [`geometry`](../geometry.entpkg/README.md), [`random`](../random.entpkg/README.md).

```text
use "effects.entpkg";
```

## Kernels

- `kernels/draw.ent` — The live layers drawn over a picture whose alpha
  holds how far each pixel is (1 + the distance; 0 for the sky), as
  `glow.ent` draws. Kernels: `effects.light@1`, `effects.draw@1`,
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
