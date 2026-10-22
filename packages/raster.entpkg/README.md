# raster

Triangles turned into the surface a camera sees, drawn alike on every
backend: plain, tiled and through a cutout mask. `native.rasterize` is
its fast path.

**Layer:** Graphics. **Uses:** no other package.

```text
use "raster.entpkg";
```

## Kernels

- `kernels/cutout.ent` — Triangles drawn through a mask. Kernels:
  `raster.cover_cutout@3`, `raster.cover_cutout@4`,
  `raster.cover_cutout_tiles@2`.
- `kernels/tile.ent` — Triangles larger than a tile cut across many.
  Kernels: `raster.boxes@1`, `raster.tiles@2`, `raster.offsets@1`,
  `raster.total@1`, `raster.cover_tiles@2`.
- `kernels/triangle.ent` — Rasterization as Kernels. Kernels:
  `raster.cover@3`, `raster.cover@4`, `raster.resolve@2`.

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
- `graphs/window.esn` — The raster preset at the window's size.

## Tested by

[crowd](../../examples/crowd/README.md), [culled](../../examples/culled/README.md), [decided](../../examples/decided/README.md), [mesh](../../examples/mesh/README.md), [mipmap](../../examples/mipmap/README.md), [model](../../examples/model/README.md), [reach](../../examples/reach/README.md), [skin](../../examples/skin/README.md), [stride](../../examples/stride/README.md), [surface](../../examples/surface/README.md), [voxel](../../examples/voxel/README.md), whose replays hold it on every backend.
