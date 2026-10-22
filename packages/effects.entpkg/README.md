# effects

Magic and skills as data: an effect is layers of forms in time, made of
its element's matter, drawn as light gathered along each pixel's ray
over a picture with its depth.

**Layer:** Graphics. **Uses:** [`geometry`](../geometry.entpkg/README.md), [`random`](../random.entpkg/README.md).

```text
use "effects.entpkg";
```

## Kernels

- `kernels/blast.ent` — The great spells' bursts as light.
- `kernels/draw.ent` — The live layers drawn over a picture whose alpha
  holds how far each pixel is (1 + the distance; 0 for the sky), as
  `glow.ent` draws. Kernels: `effects.light@1`, `effects.draw@1`,
  `effects.cel@1`.
- `kernels/effect.ent` — The rows an effect is made of.
- `kernels/element.ent` — The elements' magick as light.
- `kernels/form.ent` — The forms as light along a ray.
- `kernels/glow.ent` — Magick's light, gathered along a ray (`Glow`).
- `kernels/matter.ent` — What a layer is made of, at a point.
- `kernels/moment.ent` — Each cast's layers as they are now. Kernels:
  `effects.now@1`.
- `kernels/spell.ent` — The spells by name.

## Graph presets

- `graphs/draw.esn` — Casts drawn over a picture.

## Tables

`data/effects.csv`, `data/elements.csv`, `data/layers.csv`, `data/names.bin`, `data/spells.csv`

## Tests

`tests/` holds scenes and their replays, run on every backend:
`cast.replay`. Every Kernel of the package is reached by one of them.

Used by the [spells](../../examples/spells/README.md) example.
