# vector

Drawings filled pixel by pixel: the shapes `entelechy import` reads from
SVG, their fill rules, paints and gradients, drawn whole or tiled.

**Layer:** Graphics. **Uses:** [`parallel`](../parallel.entpkg/README.md).

```text
use "vector.entpkg";
```

## Kernels

- `kernels/fill.ent` — Drawing as Kernels. Kernels: `vector.frame@1`,
  `vector.whole@1`, `vector.draw@1`.
- `kernels/shape.ent` — A drawing's shapes as `entelechy import` writes
  them from an SVG.
- `kernels/tile.ent` — The tiled form. Kernels: `vector.reach@1`,
  `vector.shape_chunks@1`, `vector.tile_counts@1`, `vector.tile_fill@1`,
  `vector.placed@1`, `vector.segment_chunks@1`, `vector.units@1`,
  `vector.kept@1`, `vector.keep@1`, `vector.zeros@1`,
  `vector.backdrop@1`, `vector.draw_tiled@1`.

## Graph presets

- `graphs/draw.esn` — A drawing through `look` at the window's size,
  tiled.
- `graphs/drawing.esn` — A drawing, whole and fitted to the window, over
  `background`.

## Tested by

[vector](../../examples/vector/README.md), whose replays hold it on every backend.
