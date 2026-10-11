# Examples

Each example is a directory with a `scene.esn` entry, usually a
`kernels/` directory of the `.ent` libraries it uses and a `tests/`
directory of `.replay` scripts and their fixture scenes, and a
`README.md` saying what it shows and how to run it.

An example is a whole program; a reusable capability it draws on is a
package in [`packages/`](../packages/README.md), used whole by path
(`use "../../packages/skin.entpkg";`) or, with `ENTELECHY_PACKAGES` set, by
name. An example uses packages and nothing uses an example. Small
helpers stay in the example, so one directory reads whole.

```sh
entelechy run examples/<example>/scene.esn --out output.ppm --watch
```

## Language and execution modes

- [grayscale](grayscale/README.md): a gradient through grayscale and pixelate Kernels; the first proof.
- [life](life/README.md): Conway's Life on a `Grid<Cell>` through a feedback binding, painted with the pointer.
- [ant](ant/README.md): Langton's ant as a `scan` over a grid memory.
- [modes](modes/README.md): how a Kernel runs, in five parts: a `merge` reduction meters a frame, two maps sort, one Kernel writes three outputs by name, a scatter splats particles, and a machine steps off the frame with `async`.
- [image](image/README.md): pictures from arrays, matrices, volumes and samplers, in five parts: a Gaussian from a weight table and a `Mat3` emboss, a cube through its model matrix, marching squares from a case table, a `Volume<Float>` cloud, and a tile scrolled through `mod`.
- [data](data/README.md): how Fields hold what they hold, in four parts: every Field as narrow as it holds, a part half a million metres out in `Double` and `Long`, a `Grid<Tile>` flooded a step a frame, and a tally written to a file and read back.

## Rendering

- [path](path/README.md): a progressive path tracer; samples and bounces are graph parameters, the running sum is state.
- [mesh](mesh/README.md): a torus subdivided, projected, rasterized by the raster package and shaded; `native.rasterize@1` is the fast path.
- [model](model/README.md): a GLB crate imported by `entelechy-import`, rasterized and textured with `at_linear`.
- [mipmap](mipmap/README.md): a ground read at the mip level each pixel needs, levels from a sub-graph placed with `carry`.
- [decided](decided/README.md): geometry sized inside the frame by `domain = value(detail) * value(detail)`.
- [culled](culled/README.md): ten million triangles, of which a frame places the chunks it sees through the compact package.
- [crypt](crypt/README.md): a voxel crypt by firelight or by day, a statue turning in a box of its own, drawn with the voxel package.
- [sculpt](sculpt/README.md): an island made of voxel shapes as data, dug and raised by hand and drawn by the voxel package's view.
- [surface](surface/README.md): a blob meshed from a `Volume<Int>` every frame, the cells compacted and counted.
- [text](text/README.md): rows of glyphs drawn through a font atlas baked by `run --save`.
- [ui](ui/README.md): a settings screen and a hero's panel drawn with the ui package, moved through by keys, a pad or the pointer.
- [vector](vector/README.md): an SVG imported by `entelechy-import` and drawn sharp at any size by the vector package; drag and zoom.
- [showcase](showcase/README.md): a signed-distance gamepad under physically based shading, with a toolbar in logical pixels.
- [darkroom](darkroom/README.md): a raw photograph developed with the image package: equalized, graded, sharpened, and a neon outline drawn from a signed distance field.
- [nebula](nebula/README.md): a drift through a nebula made by the noise package, finished by the image package's camera look: glow, depth of field, tone and grain.
- [logo](logo/README.md): Entelechy's logo, a constructed mark and a name drawn in strokes, every version from one Kernel, and the mark's drawing sheet drawn from the same functions.

## Simulation and learning

- [million](million/README.md): a million bodies stepped by a map and drawn by a scatter.
- [neighbours](neighbours/README.md): sixty-five thousand bodies finding neighbours through a grid built by scatter `sum` and `prefix with sum`.
- [physics](physics/README.md): boxes and balls that fall, stack, slide and bounce, stepped by the physics package; a sensor, a ray, a click drops a ball.
- [fluid](fluid/README.md): smoke by Navier-Stokes, the pressure solve one Jacobi Kernel placed with `times`.
- [collapse](collapse/README.md): wave function collapse on a `Grid<Int>` of tile bitmasks.
- [learn](learn/README.md): a 2-16-1 network trained by gradient descent, the gradient a scatter.
- [cube-puzzle](cube-puzzle/README.md): a playable cube puzzle and a 54-32-1 network trained on its scrambles; `play.esn` reads kept weights.

## Sound

- [strings](strings/README.md): six plucked strings, each a delay line in a `memory`.
- [sound](sound/README.md): sound written as Kernels, heard and seen: a synthesizer's rows filtered by a `scan` and drawn as a scope, beside a voice and its spectrum, the two mixed into one output.
- [sampler](sampler/README.md): a drum WAV, baked by a Kernel, played and resampled with `at_linear`.
- [score](score/README.md): a MIDI file imported, played and drawn as a piano roll, exported back as MIDI.

## Rigs

- [skin](skin/README.md): a Blender arm imported as rig rows and posed by the `skin` package; the pointer blends two clips.
- [crowd](crowd/README.md): sixty-four arms from one rig, each at its own moment of the clip.
- [reach](reach/README.md): a shoulder aimed by `skin.aim@1` while the elbow plays the clip.
- [stride](stride/README.md): root motion by `skin.stride@1` and `skin.travel@1`, carried through `<~`.
- [golem](golem/README.md): a body of voxel shapes baked at rest, walked by the `skin` package's rows and posed into voxels by the `voxel` package.

## Games, tools and machines

- [snake](snake/README.md): a game in five Kernels; the replay plays it through.
- [blocks](blocks/README.md): falling blocks, pieces as a bit-mask table; the replay clears a line.
- [slg](slg/README.md): a tactics game with a turn machine and movement ranges as a sub-graph placed three times.
- [voxel](voxel/README.md): a voxel editor over a `Volume<Int>` with undo and a toolbar.
- [chip8](chip8/README.md): a CHIP-8 interpreter as a `scan` over RAM, registers and screen.
- [thumb](thumb/README.md): an ARM7TDMI Thumb core as a `scan`, running a program that paints a framebuffer.

## Agents, packages and files

- [package](package/README.md): a scene using `image.entpkg` and its preset; what `entelechy pack` carries.
- [level](level/README.md): a binary level format read by a Kernel from `bytes`.
- [clip](clip/README.md): an input read frame by frame from a directory with `frames`.
