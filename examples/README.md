# Examples

Each example is a directory with a `scene.esn` entry, usually a
`kernels/` directory of the `.ent` libraries it uses and a `tests/`
directory of `.replay` scripts and their fixture scenes, and a
`README.md` saying what it shows and how to run it.

An example is a whole program; a reusable capability it draws on is a
package in [`packages/`](../packages/README.md), used whole by path
(`use "../../packages/animation.entpkg";`) or, with `ENTELECHY_PACKAGES` set, by
name. An example uses packages and nothing uses an example. A package is tested by its own `tests/`, not by an example: an example's replays test the program it is. Small
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
- [data](data/README.md): how Fields hold what they hold, in five parts: every Field as narrow as it holds, a part half a million meters out in `Double` and `Long`, a `Grid<Tile>` flooded a step a frame, and a tally written to a file and read back, and an input read frame by frame from a directory with `frames`.

## Rendering

- [path](path/README.md): a progressive path tracer; samples and bounces are graph parameters, the running sum is state.
- [mesh](mesh/README.md): a torus subdivided, projected, rasterized by the raster package and shaded, `native.rasterize@1` its fast path; parts: a GLB crate imported and textured with `at_linear`, and geometry sized inside the frame by `domain = value(detail) * value(detail)`.
- [mipmap](mipmap/README.md): a ground read at the mip level each pixel needs, levels from a sub-graph placed with `carry`.
- [culled](culled/README.md): ten million triangles, of which a frame places the chunks it sees through the parallel package.
- [surface](surface/README.md): a blob meshed from a `Volume<Int>` every frame, the cells compacted and counted.
- [text](text/README.md): rows of glyphs drawn through a font atlas baked by `run --save`.
- [ui](ui/README.md): a settings screen and a hero's panel drawn with the ui package, moved through by keys, a pad or the pointer.
- [controls](controls/README.md): a dot walked by the keyboard or a pad alike, actions read from a table by the controls package.
- [vector](vector/README.md): an SVG imported by `entelechy-import` and drawn sharp at any size by the vector package; drag and zoom.
- [material_balls](material_balls/README.md): rows of material balls handed over as a G-buffer and lit by renderers of its own (probe GI, a path tracer, clay) over the light package's atoms; fireflies and scattered pebbles over them.
- [showcase](showcase/README.md): a signed-distance gamepad under physically based shading, with a toolbar in logical pixels.
- [darkroom](darkroom/README.md): a raw photograph developed with the image package: equalized, graded, sharpened, and a neon outline drawn from a signed distance field.
- [nebula](nebula/README.md): a drift through a nebula made by the random package, finished by a camera look of its own over the image package's steps: glow, depth of field, tone and grain.
- [logo](logo/README.md): Entelechy's logo, a constructed mark and a name drawn in strokes, every version from one Kernel, and the mark's drawing sheet drawn from the same functions.

## Simulation and learning

- [genesis](genesis/README.md): a world that rewrites its own law of growth until its life is in balance, writing each law through an output `to files`.
- [swarm](swarm/README.md): many bodies at once: a million stepped by a map and drawn by a scatter, and sixty-five thousand finding their neighbors through a grid built by scatter `sum` and `prefix with sum`.
- [physics](physics/README.md): boxes and balls that fall, stack, slide and bounce, stepped by the physics package; a sensor, a ray, a click drops a ball.
- [fluid](fluid/README.md): smoke by Navier-Stokes, the pressure solve one Jacobi Kernel placed with `times`.
- [collapse](collapse/README.md): wave function collapse on a `Grid<Int>` of tile bitmasks.
- [learn](learn/README.md): a 2-16-1 network trained by gradient descent, the gradient a scatter.
- [cube-puzzle](cube-puzzle/README.md): a playable cube puzzle and a 54-32-1 network trained on its scrambles; `play.esn` reads kept weights.

## Sound

- [sound](sound/README.md): sound written as Kernels, heard and seen: a synthesizer's rows filtered by a `scan` and drawn as a scope, beside a voice and its spectrum, the two mixed into one output; parts: six plucked strings each a delay line in a `memory`, a drum WAV played and resampled, a MIDI file played as a piano roll and exported back, a game's music from a mood, a band and a seed, and every event sound in each material and style.

## Rigs

- [animation](animation/README.md): a skeleton, its clips and the mesh that follows, in four parts: a Blender arm imported as rig rows with two clips blended by the pointer, a crowd of sixty-four, a shoulder aimed at a point while the elbow plays its clip, and root motion carried through `<~`.

## Games, tools and machines

- [slg](slg/README.md): a tactics game with a turn machine and movement ranges as a sub-graph placed three times.
- [arcade](arcade/README.md): small games of a few Kernels: falling blocks with pieces as a bit-mask table, and snake; the replays play them through.
- [machines](machines/README.md): processors as a `scan` over their memory: a CHIP-8 interpreter and an ARM7TDMI Thumb core painting a framebuffer.
- [voxel](voxel/README.md): a voxel editor over a `Volume<Int>` with undo and a toolbar, drawn by the voxel package; parts: a crypt by firelight or by day with a statue turning in a box of its own, and a body of voxel shapes walked by the animation package's rows.

## Agents, packages and files

- [level](level/README.md): a binary level format read by a Kernel from `bytes`.
