# mesh

Triangle meshes as two rows, points and faces: lattices that make the
torus, the sphere and the plane, midpoint subdivision, and a mesh built
as levels, each coarser than the one before, cut into clusters a renderer
draws each at the level its size on the screen asks for. The faces are
the raster package's, so a mesh projected point by point is drawn as it
is.

**Layer:** Graphics. **Uses:** [`parallel`](../parallel.entpkg/README.md), [`raster`](../raster.entpkg/README.md).

```text
use "mesh.entpkg";
```

## Kernels

- `kernels/cluster.ent` — A mesh as levels cut into clusters: each level
  its level below's points merged cell by cell on a lattice twice as wide
  (vertex clustering, each cell's point where its error quadric is least),
  each level's faces in the clusters of the bricks their corners lie in,
  how far each brick of each level lies from the mesh (the planes its
  points stray from and the area it lost or gained), and libraries of
  such meshes joined. Kernels: `mesh.cluster_box@1`,
  `mesh.cluster_edges@1`, `mesh.cluster_frame@1`,
  `mesh.cluster_no_quadrics@1`, `mesh.cluster_face_quadrics@1`,
  `mesh.cluster_keys@1`, `mesh.cluster_heads@1`, `mesh.cluster_starts@1`,
  `mesh.cluster_merge@1`, `mesh.cluster_up@1`, `mesh.cluster_deviation@1`,
  `mesh.cluster_face_flags@1`, `mesh.cluster_faces_up@1`,
  `mesh.cluster_own_faces@1`, `mesh.cluster_sources8@1`,
  `mesh.cluster_join_sources@1`,
  `mesh.cluster_corner_bricks@1`, `mesh.cluster_entry_keys@1`,
  `mesh.cluster_run_starts@1`, `mesh.cluster_chunk_heads@1`,
  `mesh.cluster_sorted_faces@1`, `mesh.cluster_rows@1`,
  `mesh.cluster_grid_size@1`, `mesh.cluster_no_errors@1`,
  `mesh.cluster_exact@1`, `mesh.cluster_point_errors@1`,
  `mesh.cluster_no_areas@1`, `mesh.cluster_brick_areas@1`,
  `mesh.cluster_area_errors@1`,
  `mesh.cluster_grid_up@1`, `mesh.cluster_points8@1`,
  `mesh.cluster_up8@1`, `mesh.cluster_faces8@1`, `mesh.cluster_rows8@1`,
  `mesh.cluster_errors8@1`, `mesh.cluster_mesh@1`,
  `mesh.cluster_join_points@1`, `mesh.cluster_join_up@1`,
  `mesh.cluster_join_faces@1`, `mesh.cluster_join_rows@1`,
  `mesh.cluster_join_errors@1`, `mesh.cluster_join_meshes@1`.
- `kernels/lattice.ent` — A lattice of points, `columns` a row and
  `rows` of them, and two faces to each of its cells. Kernels:
  `mesh.lattice_faces@1`, `mesh.torus_points@1`, `mesh.sphere_points@1`,
  `mesh.plane_points@1`.
- `kernels/subdivision.ent` — Midpoint subdivision, the step that
  changes a mesh's topology. Kernels: `mesh.split_points@1`,
  `mesh.split_faces@1`.

## Graph presets

- `graphs/cluster_brick.esn` — A level's clusters.
- `graphs/cluster.esn` — A mesh built as levels and cut into clusters:
  the library a renderer draws from (`render.entpkg`'s
  `graphs/cluster.esn`).
- `graphs/library.esn` — Two built meshes or libraries as one.
- `graphs/cluster_level.esn` — One level up.
- `graphs/subdivision.esn` — One level of midpoint subdivision.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`cluster.replay`, `lattice.replay`. Every Kernel of the package is reached by one of them.

Used by the [animation](../../examples/animation/README.md), [culled](../../examples/culled/README.md), [mesh](../../examples/mesh/README.md), [surface](../../examples/surface/README.md) examples.
