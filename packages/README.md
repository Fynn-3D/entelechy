# Packages

A package is a capability with a stable interface, meant to be reused:
a directory with a `package.osn` naming it and the least language and IR
it needs, a `kernels/` directory it `use`s, and `graphs/` presets a
scene places as nodes. A scene, an example or another package uses it
whole, `use "…/name.orlapkg";`, never a file inside it, so the
manifest's versions hold; a preset in `graphs/` and a file it ships
(`font.png`) are reached by their path inside it, or through its name
(`input font: Grid<Float> = "text.orlapkg/font.png";`). Every package is used
by an example, whose replay tests it. `orla pack` carries one as a single `.orlapkg` file and
`orla unpack` opens it again; `orla package <dir>` prints what one
holds.

A scene reaches a package by path:

```text
use "../../packages/image.orlapkg";
node display: graph "../../packages/image.orlapkg/graphs/display.osn" { ... }
```

or by name, when `ORLA_PACKAGES` lists the directory it is in:

```sh
ORLA_PACKAGES=/path/to/orla/packages orla run scene.osn
```

```text
use "image.orlapkg";
```

A package names the packages it uses by name, `use "compact.orlapkg";`,
and finds them beside itself wherever the two are kept together. A
directory is named for the package its manifest declares, `name@1` in
`name.orlapkg`, and a scene reaches one package of a name: two editions
of one are refused, naming both. A Kernel's own id carries the version
its interface is held to.

- **image** — what Orla does to pictures, over `Image`s in linear light:
  - colour: luminance, sRGB's transfer, HSV and Oklab, the colours of
    heat (`blackbody`) and of the wheel (`hue`), as functions, and
    `image.encode` / `image.decode` for a window or a file;
  - blending: the common modes (multiply, screen, overlay, soft light
    and others), and compositing one picture over another as the W3C
    spec does, on sRGB's stored values as painting programs blend or on
    light itself;
  - filters: Gaussian blur (`graphs/blur.osn`), a box shrink, any
    convolution, unsharp sharpening (`graphs/sharpen.osn`), Sobel
    edges, dilation and erosion, bicubic resizing;
  - tone: exposure, contrast, saturation and hue in Oklab; levels;
    curves; 3D LUTs, and the neutral LUT a grade is baked into; ACES;
  - histograms by Oklab lightness, so equal bins are equal steps to the
    eye: counting by scatter, the cumulative count by prefix,
    equalization (`graphs/equalize.osn`);
  - distance: the signed distance from every pixel to a shape, by jump
    flooding (`graphs/distance.osn`);
  - sampling: four texels blended through `mod` for a tile, which
    `at_linear` cannot do at a Field's edge (`wrap.orla`), and the strip
    of mip levels with the reads that pick one, for a surface running
    away from the eye (`graphs/mip.osn`);
  - looks a scene places last: `graphs/display.osn` (tone and gamma);
    `graphs/finish.osn`, a camera's, over a picture that keeps each
    pixel's depth in w: a glow round what is past white
    (`graphs/bloom.osn`), depth of field at a focus the scene gives as
    a row each frame, ACES, colour fringing, darkened corners and grain
    from the noise package's hash, each step a function another look
    composes (`fringed`, `defocus`, `vignetted`, `film_grain`);
    `graphs/nes.osn`, coarse pixels held
    to the colours a scene gives;
  - palettes a scene gives as rows: each pixel the nearest colour in
    Oklab (`image.palette@2`), or the colour its lightness falls to,
    rows from dark to light (`image.ramp@1`), or dithered between the
    two nearest by a Bayer matrix (`image.dither@1`);
  - pixel art: sheets of frames in cells, the cell of a frame and the
    frame and pixel of a texel (`sheet_cell`, `sheet_texel`,
    `sheet_place`, `sheet_pixel`), the frame an animation shows at a
    moment (`frame_at`); a line round what is drawn
    (`image.outline@1`) and the shadow it casts (`image.shadow@1`).

  The `darkroom` example develops a photograph with it, and its replay
  holds each operation to an answer worked by hand on three backends;
  `nebula`, `mipmap` and `package` use its looks and sampling.
- **text** — a 5 by 7 font baked into `font.png`, the layout that turns
  a string into glyph rows, and the Kernels that draw them. `write.orla`
  draws from a Kernel's own code, pixel by pixel: a character, a run of
  codes (a word of a `text` input), a number, and the lengths a message
  is put together with. A scene
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
- **voxel** — worlds of voxels drawn by walking rays through them, as
  functions: a ray walked voxel by voxel through a volume to the first
  voxel that is not empty, passing through numbers it sees through
  (`voxel_walk`, `voxel_walk_past`); a box of voxels placed in the world,
  turned about its up axis, met by the same walk carried into its frame,
  with a near plane so nothing beside the camera fills the picture
  (`voxel_meet`); a sky's and a sun's light with shadows, the colours
  of each the caller's (`voxel_daylight`,
  `voxel_shaded`), lights fading with distance and flickering
  (`voxel_light`, `voxel_flicker`), light kept to a few steps
  (`voxel_stepped`), haze and the dark (`voxel_haze`, `voxel_dark`); a
  camera going round a point, its rays and its projection
  (`voxel_orbit`, `voxel_ray`, `voxel_project`); shapes as data, a
  sphere, a rounded cuboid or a capsule each painting a number or
  carving (`VoxelShape`), baked into a volume (`voxel.shapes@1`) or
  painted over one (`voxel.paint@1`), which fed back is a world edited a
  frame at a time; one look of a volume by day at the window's size,
  its sky and lights the scene's (`graphs/view.osn`), with the camera
  and box a Kernel picks through
  (`voxel_view_camera`, `voxel_view_box`, `voxel_cell`); bodies, a
  volume at rest whose voxels carry a look and the joint they move with
  (`voxel_body`), posed into a box by carrying each voxel back to rest
  through each joint's matrix (`voxel_posed`, `voxel.posed@1`), the
  matrices the `skin` package's palette turned back
  (`voxel.unposing@1`). Used by `examples/crypt`, `examples/sculpt` and
  `examples/golem`.
- **noise** — randomness a Kernel can repeat, as functions and no
  Kernels:
  - hashes of whole numbers by integer arithmetic, the same bits on
    every backend, read as a number from 0 to 1 (`unit_of`) or a whole
    number below `n` (`below_of`, never negative as `h % n` can be);
  - value and gradient noise; fbm and ridged sums of them;
  - Worley's cells;
  - curl noise, a swirl with no sources and no sinks, for pushing smoke
    about.
  
  Used by the `nebula` example, whose replay holds the hashes to their
  bits on three backends.
- **physics** — rigid circles and boxes in the plane that fall, collide,
  rest, slide and bounce. `graphs/step.osn` is one step of a world of
  `Body` rows: pairs found through a grid of cells sorted by the `sort`
  package, manifolds of up to two points, every contact solved at once
  over substeps against its share of each body, and bounces; a world is
  given its last step's contacts, which the next starts from. Static,
  kinematic and sensor bodies, collision layers, and `physics.ray@1`
  for what a ray meets. Used by the `physics` example.
- **ui** — a screen's widgets, drawn per pixel, and the text on them in
  any language:
  - layout: rectangles cut into rows, columns, cells and lists, a
    table's row into its columns by where each starts, a thing placed
    in one by alignment, the row a pixel falls in;
  - paint: coats laid over one another, rounded boxes, rims, shadows,
    discs, rings, lines, arrows and ticks, each edge smooth over a pixel;
  - letters: a text, UTF-8 a row a line, decoded once into rows of
    letters of a fixed width (`graphs/text.osn`), a printable ASCII
    letter a cell from the text package's bitmap font and a wide one,
    Chinese say, two cells from an atlas the scene draws, square glyphs
    in order of the codes it gives (`graphs/letters.osn`); the package
    draws no font, and a character neither holds is a `?`; one text's
    letters or another's chosen by a number (`ui.either@1`), read anew
    only when it changes;
  - writing: a row of letters at a scale, aligned in a rectangle and cut
    at its sides, with its shadow; numbers; a name and a number on one
    row;
  - widgets over a theme: panels, titled panels, the lit row, a list's
    items and the row a pixel near the lit one is drawn by, buttons,
    switches, check boxes, sliders, meters with a trail, options with
    arrows or a switch, an option's name beside a control of the
    screen's own, scroll bars, rules, keys' caps, a pad's face buttons
    and a prompt beside one;
  - navigation (`ui.navigate@1`): the row lit and what was asked of it
    this frame, from the arrows or W A S D, Enter, Esc, a pad's stick and
    face buttons, the pointer and the wheel, a direction held repeating.

  Every widget is nothing outside its rectangle but its shadow, so a
  screen is its widgets laid over one another. Used by the `ui`
  example.
- **vector** — drawings filled pixel by pixel: the segments and shapes
  `orla import` reads from SVG, drawn through a look (a middle point and a
  zoom) with nonzero or even-odd fills, colours and linear or radial
  gradients, each edge smoothed over a pixel. `graphs/draw.osn` draws it
  tiled: each pixel asks only the shapes whose boxes reach its tile and
  their segments that reach the tile, over the winding of those past it,
  word for word what the plain `vector.draw@1` draws where every Kernel
  rounds alike, so a drawing of many thousand shapes draws in time.
  `graphs/drawing.osn` draws one whole, fitted to the window. Used by the
  `vector` example.
