# raster

Triangles turned into the surface a camera sees, drawn alike on every
backend: plain, tiled and through a cutout mask; and translucent
triangles over it, in any order. `native.rasterize`, and `native.cover`
for the keys alone, are its fast paths.

**Layer:** Graphics. **Uses:** [`parallel`](../parallel.entpkg/README.md).

```text
use "raster.entpkg";
```

## Kernels

- `kernels/clip.ent` — Triangles cut at the near plane, before the
  rasterizer, which skips whole a triangle reaching behind it. Kernels:
  `raster.clip_near@1`, `raster.clip_plane@1`, `raster.clip_faces@1`.
- `kernels/cutout.ent` — Triangles drawn through a mask. Kernels:
  `raster.cover_cutout@3`, `raster.cover_cutout@4`,
  `raster.cover_cutout_tiles@2`.
- `kernels/tile.ent` — Triangles larger than a tile cut across many.
  Kernels: `raster.boxes@1`, `raster.tiles@2`, `raster.offsets@1`,
  `raster.total@1`, `raster.cover_tiles@2`.
- `kernels/triangle.ent` — Rasterization as Kernels. Kernels:
  `raster.cover@3`, `raster.cover@4`, `raster.resolve@2`.
- `kernels/veil.ent` — Translucent triangles over an opaque surface: the
  light they add and the share of the surface left showing, summed whole
  or tiled so any order of triangles gives the same pixel. Kernels:
  `raster.unveiled@1`, `raster.veil@1`, `raster.veil_boxes@1`,
  `raster.veil_small@1`, `raster.veil_tiles@1`, `raster.veiled@1`.

## Graph presets

- `graphs/canvas.esn` — The raster preset over a canvas.
- `graphs/cutout.esn` — The raster preset with a cutout.
- `graphs/cutout_tiled.esn` — The tiled form of `cutout.esn`.
- `graphs/cutout_tiled_window.esn` — The tiled form of
  `cutout_window.esn`.
- `graphs/cutout_window.esn` — The cutout preset at the window's size,
  as `window.esn` is the plain one's.
- `graphs/surface.esn` — Vertex and triangle rows in, the surface out.
- `graphs/tiled.esn` — The tiled form of `surface.esn`.
- `graphs/tiled_window.esn` — The tiled form of `window.esn`.
- `graphs/veil.esn` — Translucent triangles, their tints and the opaque
  surface in; the light and what shows through out.
- `graphs/veil_tiled.esn` — The tiled form of `veil.esn`.
- `graphs/window.esn` — The raster preset at the window's size.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`clip.replay`, `degenerate.replay`, `near.replay`,
`tiles.replay`, `veil.replay`. Every Kernel of the package is
reached by one of them.

Used by the [animation](../../examples/animation/README.md), [culled](../../examples/culled/README.md), [mesh](../../examples/mesh/README.md), [mipmap](../../examples/mipmap/README.md), [surface](../../examples/surface/README.md), [voxel](../../examples/voxel/README.md) examples.
