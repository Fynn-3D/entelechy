# terrain

Ground as heights on a grid of cells: read back as a smooth surface, its
normal, where a ray first meets it and how open the sky is over it; worn
down by erosion, and laid out as a mesh.

**Layer:** Graphics. **Uses:** [`mesh`](../mesh.entpkg/README.md).

```text
use "terrain.entpkg";
```

## Kernels

- `kernels/height.ent` — Ground as a `Grid<Float>` of heights, one a
  cell, the cells `cell` meters apart.
- `kernels/erosion.ent` — Erosion. Kernels: `terrain.erode@1`.
- `kernels/surface.ent` — The ground as a mesh. Kernels:
  `terrain.points@1`.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`ground.replay`. Every Kernel of the package is reached by one of them.
