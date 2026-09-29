# Packages

A package is a capability with a stable interface, meant to be reused:
a directory with a `package.osn` naming it and the least language and IR
it needs, a `kernels/` directory it `use`s, and `graphs/` presets a
scene places as nodes. A scene, an example or another package uses it
whole, `use "…/name.orlapkg";`, never a file inside it, so the
manifest's versions hold; a preset in `graphs/` and a file it ships
(`font.png`) are reached by their path inside it. Every package is used
by an example, whose replay tests it. `orla pack` carries one as a single `.orlapkg` file and
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
  `graphs/runs.osn` is the same rank for lists: each of a row of owners
  counts its entries, and the preset gives where each owner's run ends,
  how many entries there are and the row of -1 a scatter fills, walking
  what it counted; `owner` finds an entry's owner, `rounds` and
  `round_of` walk more than one loop counts. The vector, physics and font
  packages build their lists with it.
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
- **pixelfx** — a coarse pixel grid and a four-color palette, and the
  NES look made of both as a preset. The example of what a package is.
- **font** — type from a TrueType or OpenType font `orla import` read:
  a row of character codes laid out in lines, each character its font's
  glyph at its own advance, turned into the shapes and segments the
  `vector` package draws, so text is as sharp at any size as any
  drawing. `graphs/text.osn` draws a row of codes over a picture;
  `font.join@1` joins rows into lines. Used by the `type` example.
- **physics** — rigid circles and boxes in the plane that fall, collide,
  rest, slide and bounce. `graphs/step.osn` is one step of a world of
  `Body` rows: pairs found through a grid of cells sorted by the `sort`
  package, manifolds of up to two points, every contact solved at once
  over substeps against its share of each body, and bounces; a world is
  given its last step's contacts, which the next starts from. Static,
  kinematic and sensor bodies, collision layers, and `physics.ray@1`
  for what a ray meets. Used by the `physics` example.
- **vector** — drawings filled pixel by pixel: the segments and shapes
  `orla import` reads from SVG, drawn through a look (a middle point and a
  zoom) with nonzero or even-odd fills, colours and linear or radial
  gradients, each edge smoothed over a pixel. `graphs/draw.osn` draws it
  tiled: each pixel asks only the shapes whose boxes reach its tile and
  their segments that reach the tile, over the winding of those past it,
  word for word what the plain `vector.draw@1` draws, so a drawing of
  many thousand shapes draws in time. `graphs/drawing.osn` draws one
  whole, fitted to the window. Used by the `vector` example.
