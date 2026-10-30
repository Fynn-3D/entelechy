# texture

What a surface is drawn with, point by point as a shader asks: patterns,
the ways a pattern is laid on a surface, layers and normals. Each is a function a scene's own material calls where it shades a
point (a G-buffer kernel, a distance field's march, a mesh's pixels), so
a material is these words composed, not a feature of the renderer.

**Layer:** Graphics. **Uses:** [`random`](../random.entpkg/README.md).

```text
use "texture.entpkg";
```

Detail is drawn so it holds still on screen: a pattern's lines, stripes
and shapes take the width of the pixel they fall in (`pix`, in the
pattern's units: a point's distance times the pixel's angle) and are
averaged over it, and detail that would crowd closer than a few pixels
is let go (`texture_fade`).

## Kernels

- `kernels/layer.ent` — Layers blended by their heights; a normal bent
  by a height's slope, made from a height's differences, taken from a
  face's frame to the world, and two normals joined.
- `kernels/pattern.ent` — Patterns on a surface's plane: a grid's cells,
  courses of bricks or boards, panels split by a hash, an opening inset
  in a cell; lines, stripes and any signed distance covered over a pixel.
- `kernels/projection.ent` — A pattern laid on a surface: three planes
  blended by how it faces, the one face it is on, parallax, and the room
  seen through a window.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`patterns.replay`, `surfaces.replay`.
