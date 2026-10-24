# surface

What is lit, as a renderer takes it: the G-buffer (normals and
distances, paint, the rest of the contract `light` reads), settled from a
rough one while the lens moves, the layer seen through glass, two
G-buffers made one, decals pressed into it and outlines drawn from it; the
coarse scene light is traced through (proxies baked from things' own
distances or from a mesh, walked roughly along a ray); and meshes drawn
into it as clusters, each place at the level the screen asks for,
streamed round the lens. How the G-buffer is lit is
[`light`](../light.entpkg/README.md)'s.

**Layer:** Graphics. **Uses:** [`geometry`](../geometry.entpkg/README.md), [`mesh`](../mesh.entpkg/README.md), [`parallel`](../parallel.entpkg/README.md), [`raster`](../raster.entpkg/README.md).

```text
use "surface.entpkg";
```

## Kernels

- `kernels/buffer.ent` — The G-buffer as a renderer takes it: the sky's
  depth (`no_hit`), the small depth map the lens picks with, the G-buffer
  settled from a rough one while the lens moves and a fine one once it
  rests (the layer seen through glass too), a ray walked across it
  (`screen_march`), and a world plane through the lens's clip space.
  Kernels: `surface.depths@1`, `surface.settle@1`,
  `surface.settle_behind@1`, `surface.lens_clip_plane@1`.
- `kernels/cluster.ent` — Clustered meshes drawn (as Nanite draws): each
  place of each thing at the level whose error spans at most `error`
  pixels, clusters out of the picture or behind the last frame's nearer
  things left out, where two levels meet the finer one's corners taken up
  to the coarser one's points. Kernels: `surface.cluster_pick@1`,
  `surface.cluster_offsets@1`, `surface.cluster_cull@1`,
  `surface.cluster_kept@1`, `surface.cluster_shown@1`, `surface.cluster_corners@1`,
  `surface.cluster_faces@1`, `surface.cluster_hzb@1`.
- `kernels/composite.ent` — Two G-buffers of one view made one. Kernels:
  `surface.nearer@1`.
- `kernels/decal.ent` — Decals pressed onto a G-buffer. Kernels:
  `surface.decals_none@1`, `surface.decal_looks_none@1`, `surface.decal_paint@1`,
  `surface.decal_extra@1`, `surface.decal_normals@1`.
- `kernels/field.ent` — Instanced distance fields: the scene's proxy
  baked from its things' own distances. Kernels: `surface.field_atlas8@1`,
  `surface.field_shapes8@1`, `surface.field_stack@1`,
  `surface.field_frames_join@1`, `surface.field_frame@1`, `surface.field_near@1`,
  `surface.field_bin_counts@1`, `surface.field_bin_fill@1`,
  `surface.field_bake@1`.
- `kernels/outline.ent` — Outlines from the G-buffer: an edge's near side
  and a crease. Kernels: `surface.outline@1`.
- `kernels/proxy.ent` — The proxies: the coarse scene a renderer traces
  shadows and bounces through (`Proxy`, `Proxies`), a volume of distance
  and color in levels and a ground height map, read (`proxy_at`,
  `land_at`) and walked roughly along a ray (`rough_trace`,
  `rough_shadow`, `rough_normal`, `rough_albedo`).
- `kernels/proxy_bake.ent` — A mesh baked into a proxy: each cell (a
  point of the frame's lattice, as `proxy_at` reads it) the nearest face's
  distance and its color, found exactly within a band (the faces' boxes
  cut into tiles of cells) and spread past it by jump flooding; a pass a
  frame for geometry that streams; a fine level of it round the lens.
  Kernels: `surface.proxy_bake_tiles@1`, `surface.proxy_bake_faces@1`,
  `surface.proxy_seeds@1`, `surface.proxy_flood@1`, `surface.proxy_baked@1`,
  `surface.proxy_stage@1`, `surface.proxy_flood_staged@1`,
  `surface.proxy_published@1`, `surface.proxy_published_frame@1`,
  `surface.proxy_made@1`, `surface.proxy_detail_frame@1`,
  `surface.proxy_within@1`, `surface.proxy_stack@1`.
- `kernels/stream.ent` — A world streamed round the camera: the chunks of
  its lattice within reach of the eye, near ones as built and far ones
  coarser, their files spelled for a scene's `fields` nodes, each chunk
  whose arrays are in placed to draw. Kernels: `surface.stream_wanted@1`,
  `surface.stream_paths@1`, `surface.stream_insts@1`, `surface.stream_none@1`.

## Graph presets

- `graphs/cluster.esn` — Clustered meshes drawn (`kernels/cluster.ent`):
  a library `mesh.entpkg`'s `graphs/cluster.esn` built, the things
  placed (`FieldInst`s, the field's own), the corners and faces for the
  rasterizer.
- `graphs/decals.esn` — Decals laid onto a scene's G-buffer before a
  renderer (`kernels/decal.ent`).
- `graphs/field.esn` — The scene's proxy baked round the lens from its
  things' shapes and places (`kernels/field.ent`), in three levels, each
  coarser and wider than the one before.
- `graphs/field_level.esn` — One level of `field.esn`.
- `graphs/picture.esn` — The picture's raster.
- `graphs/proxy_bake.esn` — A mesh baked into a proxy
  (`kernels/proxy_bake.ent`), for a scene drawn as one mesh rather than
  placed things.
- `graphs/proxy_detail.esn` — A fine level of the proxy round the lens,
  baked from the mesh a pass a frame as the eye moves a step, joined with
  what lies past it and stacked before the scene's proxy (finest first), so
  rays near the eye meet thin things the scene's cells are too coarse for.
- `graphs/proxy_stream.esn` — A mesh baked into a proxy a pass a frame, the
  last one made in use until the new one is (with the frame it was made
  over): for geometry that streams.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`buffer.replay`, `clip_plane.replay`, `cluster.replay`, `decals.replay`,
`field.replay`, `outline.replay`, `proxy_bake.replay`,
`proxy_detail.replay`, `proxy_stream.replay`, `stream.replay`. Every
Kernel of the package is reached by one of them.
