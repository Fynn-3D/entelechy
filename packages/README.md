# Packages

What several scenes share is a package, not an example: a directory
with a `package.osn` naming it and the least language and IR it needs,
a `kernels/` directory it `use`s, and `graphs/` presets a scene places
as nodes. `orla pack` carries one as a single `.orlapkg` file and
`orla unpack` opens it again; `orla package <dir>` prints what one
holds.

A scene reaches a package by path:

```text
use "../../packages/picture.orlapkg";
node display: graph "../../packages/picture.orlapkg/graphs/display.osn" { ... }
```

or by name, when `ORLA_PACKAGES` lists the directory it is in:

```sh
ORLA_PACKAGES=/path/to/orla/packages orla run scene.osn
```

```text
use "picture.orlapkg";
```

- **picture** — what a scene does to pictures that are not its subject.
  `wrap.orla` blends four texels found through `mod`, which `at_linear`
  cannot do at a Field's edge; `tonemap.orla` and `graphs/display.osn`
  are usually a scene's last node; `mip.orla` and `graphs/mip.osn` are
  the strip of halvings and the reads that pick a level, for a surface
  running away from the eye.
- **text** — a 5 by 7 font baked into `font.png`, the layout that turns
  a string into glyph rows, and the Kernels that draw them. A scene
  binds the font beside using the package:
  `input font: Grid<Float> = "../../packages/text.orlapkg/font.png";`
- **compact** — a row of flags in, the kept indices and their count
  out. A `domain` reading `value(count)` says how many things a frame
  works on; this says which ones. `graphs/compact.osn` is the three
  steps — a prefix sum for the rank, a reduce for the count, a scatter
  that puts each kept index where its rank says — and what comes back
  sizes and steers whatever does the work. Frustum culling, live
  particles, the lights that reach a tile: the shape is the same.
- **sort** — the permutation that orders a row of integer keys, stable,
  one pass a bit: a `prefix with sum` for the rank and a scatter for the
  placement. What moves is the permutation, never the elements, so a
  scene reads its own rows through it. `examples/neighbours` sorts
  bodies by the grid cell they are in, which is what turns every pair
  into the nine cells around one.
- **skin** — a skeleton, the clips that pose it, and the mesh that
  follows, its rotations the `transform` package's. `orla import`
  writes the rows out of a rigged glTF and the
  Blender bridge writes the same ones ([the contract](../docs/RIGS.md)); sampling a clip, blending two,
  composing a joint's parents and moving a vertex by the four joints it
  hangs off are Kernels, so a pose is data all the way through and a
  scene can drive it. The same Kernels do a crowd when the node is sized
  for one — `instances * count(skeleton)` poses that many skeletons —
  and `skin.blend@1` takes a weight a joint, which is a mask, while
  `skin.add@1` lays a clip over a pose rather than into it.
  `graphs/pose.osn` plays one clip as one node; `examples/skin` blends
  two by hand and `examples/crowd` poses sixty-four.
- **transform** — rotations as quaternions and rigid transforms as
  matrices, as functions and no Kernels: a turn about an axis, one
  rotation after another and back, the short way between two, the
  shortest turn from one direction to another, a vector turned, a move,
  turn and scale as one `Mat4`, and a turn about x, y or z as a `Mat3`
  for a camera's orbit. `skin` poses joints with them; any scene that
  turns things uses the same.
- **raster** — triangles turned into the surface a camera sees, as
  Kernels: for every pixel the nearest triangle's depth, normal and uv,
  perspective-correct, and its row as `id`. `graphs/raster.osn` draws at
  a `width` and `height`, `graphs/window.osn` at the window's size;
  `graphs/cutout.osn` and `graphs/cutout_window.osn` take a `mask` and
  leave out a fragment whose alpha there is below `cutout`, which is how
  leaves and fences are drawn. Each covers a triangle with one
  invocation; a `tiled` form of each (`tiled.osn`, `tiled_window.osn`,
  `cutout_tiled.osn`, `cutout_tiled_window.osn`) cuts a triangle larger
  than a `tile` across many, which a mesh of few large triangles wants on
  a device and a mesh of small ones pays for. It
  is the rasterizer every example places; `native.rasterize@1` in its
  place is the fast path, bit for bit the same on the interpreter and
  the JIT and the device's approximate rasterizer on a GPU
  ([why](../docs/DECISIONS.md)).
- **pixelfx** — a coarse pixel grid and a four-colour palette, and the
  NES look made of both as a preset. The example of what a package is.
