# mesh

Triangle meshes as two rows, points and faces: lattices that make the
torus, the sphere and the plane, and midpoint subdivision. The faces are
the raster package's, so a mesh projected point by point is drawn as it
is.

**Layer:** Graphics. **Uses:** [`raster`](../raster.entpkg/README.md).

```text
use "mesh.entpkg";
```

## Kernels

- `kernels/lattice.ent` — A lattice of points, `columns` a row and
  `rows` of them, and two faces to each of its cells. Kernels:
  `mesh.lattice_faces@1`, `mesh.torus_points@1`, `mesh.sphere_points@1`,
  `mesh.plane_points@1`.
- `kernels/subdivision.ent` — Midpoint subdivision, the step that
  changes a mesh's topology. Kernels: `mesh.split_points@1`,
  `mesh.split_faces@1`.

## Graph presets

- `graphs/subdivision.esn` — One level of midpoint subdivision.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`lattice.replay`. Every Kernel of the package is reached by one of them.

Used by the [animation](../../examples/animation/README.md), [culled](../../examples/culled/README.md), [mesh](../../examples/mesh/README.md), [surface](../../examples/surface/README.md) examples.
