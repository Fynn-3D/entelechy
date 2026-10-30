# mesh

Triangle meshes as two rows, points and faces: lattices that make the
torus, the sphere and the plane, midpoint subdivision, and the surface
where a field sampled on a lattice crosses a level. The faces are the
raster package's, so a mesh projected point by point is drawn as it is.

**Layer:** Graphics. **Uses:** [`parallel`](../parallel.entpkg/README.md), [`raster`](../raster.entpkg/README.md).

```text
use "mesh.entpkg";
```

## Kernels

- `kernels/iso.ent` — The surface where a field sampled on a lattice
  (`IsoLattice`) crosses a level (dual contouring): a point in each cell
  it crosses, where the planes it crosses the cell's edges on meet, so
  edges and corners stay sharp as far as the samples show them, and a
  quad across each lattice edge it crosses. Kernels:
  `mesh.iso_crossed_cells@1`, `mesh.iso_slots@1`, `mesh.iso_points@1`,
  `mesh.iso_crossed_edges@1`, `mesh.iso_faces@1`.
- `kernels/lattice.ent` — A lattice of points, `columns` a row and
  `rows` of them, and two faces to each of its cells. Kernels:
  `mesh.lattice_faces@1`, `mesh.torus_points@1`, `mesh.sphere_points@1`,
  `mesh.plane_points@1`.
- `kernels/subdivision.ent` — Midpoint subdivision, the step that
  changes a mesh's topology. Kernels: `mesh.split_points@1`,
  `mesh.split_faces@1`.

## Graph presets

- `graphs/iso.esn` — A field's samples on a lattice in, the surface
  where they cross a level out, as points and faces.
- `graphs/subdivision.esn` — One level of midpoint subdivision.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`iso.replay`, `lattice.replay`. Every Kernel of the package is reached by one of them.

Used by the [animation](../../examples/animation/README.md), [culled](../../examples/culled/README.md), [mesh](../../examples/mesh/README.md), [surface](../../examples/surface/README.md) examples.
