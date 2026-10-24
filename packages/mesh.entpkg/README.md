# mesh

Triangle meshes as two rows, points and faces: lattices that make the
torus, the sphere and the plane, midpoint subdivision, the surface
where a field sampled on a lattice crosses a level, and a mesh built
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
  a point only with those of its own group (a part, a material), open
  edges and seams between parts held by planes square to their faces,
  each level's faces in the clusters of the bricks their corners lie in,
  how far each brick of each level lies from the mesh (the planes its
  points stray from and the area each of its parts lost or gained), and libraries of
  such meshes joined. Kernels: `mesh.cluster_box@1`,
  `mesh.cluster_edges@1`, `mesh.cluster_frame@1`,
  `mesh.cluster_no_quadrics@1`, `mesh.cluster_face_quadrics@1`,
  `mesh.cluster_edge_keys@1`, `mesh.cluster_edge_quadrics@1`,
  `mesh.cluster_quadrics_sum@1`, `mesh.cluster_keys@1`, `mesh.cluster_no_groups@1`,
  `mesh.cluster_group_keys@1`, `mesh.cluster_compose@1`,
  `mesh.cluster_heads@1`, `mesh.cluster_starts@1`,
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
- `kernels/iso.ent` — The surface where a field sampled on a lattice
  (`IsoLattice`) crosses a level (surface nets): a point in each cell it
  crosses, a quad across each lattice edge it crosses. Kernels:
  `mesh.iso_crossed_cells@1`, `mesh.iso_slots@1`, `mesh.iso_points@1`,
  `mesh.iso_crossed_edges@1`, `mesh.iso_faces@1`.
- `kernels/lattice.ent` — A lattice of points, `columns` a row and
  `rows` of them, and two faces to each of its cells. Kernels:
  `mesh.lattice_faces@1`, `mesh.torus_points@1`, `mesh.sphere_points@1`,
  `mesh.plane_points@1`.
- `kernels/stream.ent` — A world streamed in chunks: one chunk out of a
  mesh (the faces in a cell of the world's lattice, their points
  renumbered), and the chunks a scene's `fields` nodes read joined into
  one library, each chunk's indices moved past the chunks before it.
  Kernels: `mesh.stream_up@1`, `mesh.stream_faces@1`,
  `mesh.stream_clusters@1`, `mesh.stream_sources@1`,
  `mesh.stream_meshes@1`, `mesh.stream_ready@1`, `mesh.chunk_flags@1`,
  `mesh.chunk_clip@1`, `mesh.chunk_points@1`, `mesh.chunk_faces@1`,
  `mesh.far_flags@1`, `mesh.far_cluster_faces@1`, `mesh.far_point_flags@1`,
  `mesh.far_points@1`, `mesh.far_faces@1`, `mesh.far_clusters@1`,
  `mesh.far_errors@1`, `mesh.far_meshes@1`, `mesh.finest_counts@1`,
  `mesh.finest_faces@1`.
- `kernels/subdivision.ent` — Midpoint subdivision, the step that
  changes a mesh's topology. Kernels: `mesh.split_points@1`,
  `mesh.split_faces@1`.

## Graph presets

- `graphs/chunk.esn` — One chunk of a world out of a mesh, its faces
  clipped to the chunk's cell, built as a clustered library with its
  parts, near as it is and far with only the clusters it can draw from
  where it stands in, for a scene to save and stream.
- `graphs/cluster_brick.esn` — A level's clusters.
- `graphs/cluster.esn` — A mesh built as levels and cut into clusters:
  the library a renderer draws from (`light.entpkg`'s
  `graphs/cluster.esn`), every point in one group.
- `graphs/cluster_grouped.esn` — The same with each point in a group of
  its own (`groups`): parts kept apart on every level, so letters on a
  board or a frame on a wall do not merge into what they lie on.
- `graphs/finest.esn` — A library's faces at each mesh's finest level, for
  what is baked from a mesh as it is (a streamed world's proxy).
- `graphs/iso.esn` — A field's samples on a lattice in, the surface
  where they cross a level out, as points and faces.
- `graphs/library.esn` — Two built meshes or libraries as one.
- `graphs/cluster_level.esn` — One level up.
- `graphs/streamed.esn` — A world's streamed chunks as one clustered
  library, and which chunks are in whole.
- `graphs/subdivision.esn` — One level of midpoint subdivision.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`cluster.replay`, `iso.replay`, `lattice.replay`, `stream.replay`, `stream_build.replay` (`stream/` holds the
chunks `stream_build.esn` built, which `stream.esn` and `light.entpkg`'s `tests/stream.esn` stream). Every Kernel of the package is reached by one of them.

Used by the [animation](../../examples/animation/README.md), [culled](../../examples/culled/README.md), [mesh](../../examples/mesh/README.md), [surface](../../examples/surface/README.md) examples.
