# Examples

Each example is a directory with a `scene.osn` entry, usually a `kernels/`
directory of `.orla` libraries it uses, and, once it has behavior worth
pinning down, a `tests/` directory of `.replay` scripts with their fixture
scenes. Documentation for an example belongs in its own `README.md`.

What several scenes share is not an example: it is a package, and those
live in [`packages/`](../packages/README.md) beside this directory —
`picture` (wrapped reads, tone mapping, mip levels), `text` (a font and
the Kernels that draw with it), `pixelfx` (a look). An example uses one
the way anyone's scene would, by path or, with `ORLA_PACKAGES` set, by
name.

```sh
cargo run -- run examples/<example>/scene.osn --out output.ppm --watch
```

## Language and execution modes

- [grayscale](grayscale/README.md): the first proof. A generated gradient flows
  through grayscale and pixelate Kernels; edit any Kernel while it runs.
- [material-ball](material-ball/README.md): a shaded sphere under an orbiting
  light, tone-mapped by a second Kernel; one self-contained file with both
  Kernels inline.
- [life](life/README.md): Conway's Life on a `Grid<Cell>` of real booleans
  and exact integers through a feedback binding, painted with the pointer,
  with a declarative replay test.
- [exposure](exposure/README.md): the first reduction. A `merge` Kernel
  meters a frame's luma into one element and a second Kernel tone-maps by
  it; both backends fold in the same tree.
- [sort](sort/README.md): sorting as two map Kernels, rank then gather,
  each a bounded loop over the rows; no primitive, quadratic, exact on
  both backends.
- [blur](blur/README.md): a two-pass Gaussian from a table of weights and
  an emboss whose kernel is a `Mat3`, every edge read folded back by two
  lines of integer arithmetic: mirror addressing composed from `at`.
- [marching](marching/README.md): marching squares over orbiting blobs,
  the sixteen cases a `[[Int; 4]; 16]` and the crossing points a
  `[Vec2; 4]`, no `if` chain anywhere.
- [cube](cube/README.md): a cube ray-marched in its own space through
  the inverse of its model matrix, its edges drawn from a corner table
  and an edge table projected by `perspective * model`; matrices end to
  end in one Kernel.
- [ant](ant/README.md): Langton's ant as a `scan` over a grid memory,
  cells and headings as integer vectors, a direction table as an array;
  the trail feeds back frame to frame.
- [scroll](scroll/README.md): a tile scrolled without end three ways,
  nearest through `mod`, `at_linear` through `mod` (which seams) and the
  shared library's wrapped blend (which does not); the replay pins the
  seam.
- [precision](precision/README.md): a ring and a part half a million
  metres out, drawn and measured in doubles, in floats (a single point),
  and in whole nanometres as longs; `Double` and `Long` in the language.
- [packed](packed/README.md): every Field kept as narrow as what it holds:
  a PNG as bytes, a tile map as a byte a tile, heights in 16 bits, normals
  in three signed bytes, light carried in half floats, with `as` in the
  port type and the Kernels computing in `Float` and `Int` as ever.

## Rendering

- [deferred](deferred/README.md): deferred shading: one Kernel finds the
  surface under each pixel and writes its colour, normal and depth as three
  Fields, given by name; lighting and an outline from the depth are passes
  over them, fused into one.
- [raymarch](raymarch/README.md): a signed-distance scene marched with `loop`
  and shaded with `if`, displayed through the shared tone-mapping sub-graph
  with `exposure` as a graph parameter; interactive on the GPU backend,
  a few frames per second on the CPU.
- [volume](volume/README.md): a `Volume<Float>` of cloud density ray-marched
  into an image; the volume never leaves the GPU.
- [path](path/README.md): a path tracer that gets better while you
  watch: the samples a frame traces and the bounces each one takes are
  graph parameters the loops take as their counts, and the sum of every
  sample so far is an Image carried frame to frame. Drag to move the
  camera and it starts over.
- [mesh](mesh/README.md): a torus as vertex and triangle rows, subdivided
  by Kernels whose output rows are sized from their inputs, projected by a
  Kernel, rasterized by the raster package, and shaded by a Kernel; the
  package draws the same surface on every backend, edges included, and
  `native.rasterize@1` in its place is the one-line fast path.
- [model](model/README.md): a crate from a GLB, brought in by `orla-import`
  as vertex and triangle rows and a PNG the scene reads as inputs, spun
  and rasterized by the raster package, textured with `at_linear`.
- [mipmap](mipmap/README.md): a ground running to the horizon, read at
  the level each pixel asks for. The strip of halvings is a shared
  sub-graph placed `levels` times with `carry`, and the level comes from
  the neighbouring pixel's `uv` in the rasterized surface, which is what
  a derivative would have said.
- [decided](decided/README.md): how much geometry a frame places, decided
  inside the frame that places it. A Kernel turns where the camera is
  into one number in a Field, and `domain = value(detail) *
  value(detail)` makes the ground exactly that big: 24 by 24 vertices
  high up, 263 by 263 on the deck, no parameter carrying the number. The
  replay pins the counts with `expect count`.
- [culled](culled/README.md): a world of ten million triangles, of which
  a frame places what it can see. A Kernel writes a flag per chunk, the
  compact package turns the flags into the kept indices and their count,
  and every node that builds geometry is `domain = value(count) * <per
  chunk>`. The replay runs the same frame with the culling on and off
  and gets the same picture from a third of the geometry; sixteen times
  the world costs a millisecond and a half more.
- [surface](surface/README.md): a blob meshed out of a `Volume<Int>`
  every frame by the cells its surface passes through: a flag a cell, the
  compact package ranks them, and the geometry nodes are sized by the
  count.
- [text](text/README.md): text as rows of glyphs drawn by a Kernel through
  a font atlas the Host watches; a title, the whole font, and the pointer's
  x as digits; the font was baked from rows of bits by `run --save`.
- [showcase](showcase/README.md): a gamepad as a signed-distance model
  under physically based shading, filling the window, orbited by a drag,
  recoloured and refinished from a toolbar laid out in logical pixels;
  one state row, one picture, the replay asserts on the row.

## Simulation and learning

- [particles](particles/README.md): a `Rows<Particle>` advanced per element
  and splatted onto a canvas by a scatter, each particle writing its own
  cell in element order; the first scene whose data is not an image.
- [million](million/README.md): a million bodies stepped as a map and drawn
  by a scatter of a million writes, at 2.4 ms a frame on the GPU and 20 on the
  JIT; the example where making the count an `Int` instead of a `Float` took
  24 ms out of the frame.
- [neighbours](neighbours/README.md): sixty-five thousand bodies that
  push apart close up and draw together further out, each finding the
  others near it through a grid rebuilt every frame from a scatter with
  `sum` and a `prefix with sum`: no walk over every pair.
- [fluid](fluid/README.md): smoke by the Navier-Stokes equations: a
  staggered velocity grid carried by itself, a pressure solve as one
  Jacobi Kernel placed forty times, the projection, the smoke it carries;
  the pointer stirs, a replay reads the divergence before and after.
- [collapse](collapse/README.md): wave function collapse on a grid of pipe
  tiles, a `Grid<Int>` of bitmasks: a reduce picks the cell with the fewest
  tiles left, a map settles it, and six placements of a map spread what that
  rules out; the pointer melts what is under it and the picture grows back.
- [learn](learn/README.md): a 2-16-1 network trained by gradient descent
  to tell the inside of a circle from the outside, all of it Kernels: the
  gradient as a scatter, the step placed eight times a frame, the
  weights carried on; the plane painted as the network answers.
- [rubik](rubik/README.md): a Rubik's cube in three dimensions, turned
  by the keys and orbited by the pointer, and a 54-32-1 network trained
  while you play on scrambles it makes itself, guessing how many turns
  from solved the cube on screen is: training and inference, all of it
  Kernels; `play.osn` reads the weights `render --save` kept.

## Sound

- [audio](audio/README.md): a synthesizer as rows, its phase carried across
  blocks by Graph state, low-passed by a `scan` whose memory crosses blocks
  the same way, mixed to stereo and drawn as an oscilloscope; the Player
  plays the stereo rows and shows the scope.
- [strings](strings/README.md): six plucked strings, `1` to `6`, each a delay
  line in a `memory` carried from frame to frame; a scan walks the block of
  samples, a key fills a line with noise, and what comes out is the scene's
  sound.
- [sampler](sampler/README.md): a drum from a WAV file the Host reads
  and watches, played by a state row that keeps the playhead in the
  file's frames and resamples to the device's rate with `at_linear`;
  the file was baked by a Kernel and `run --save`.
- [spectrum](spectrum/README.md): a sound and what it is made of: a
  voice writes a block of samples a frame, a map reads it back as
  frequencies (its loop counts the block's own length), and the bars
  name the loudest. The Player plays what the bars describe.
- [score](score/README.md): a MIDI file played and drawn: `orla-import`
  reads it into rows of notes, a Kernel sums what sounds at each sample
  into the block the sound card takes, and another paints the piano
  roll. What a Kernel writes goes back out as MIDI through
  `orla-export`.

## Rigs

- [skin](skin/README.md): an arm rigged and animated in Blender, brought
  in by `orla import` as joint, weight and clip rows and posed by the
  `skin` package; the pointer blends its two clips.
- [crowd](crowd/README.md): sixty-four arms out of one rig, each at its
  own moment in the clip: the `skin` package's own Kernels on nodes sized
  `instances * count(skeleton)`.
- [reach](reach/README.md): an arm whose shoulder turns every frame to
  face a point while its elbow plays the clip: `skin.aim@1` between the
  sample and the palette, a pose partly authored and partly computed.
- [stride](stride/README.md): root motion. A clip whose root travels,
  played in place by `skin.stride@1`, its travel (`skin.travel@1`)
  carried by the scene through `<~`, so the arm crawls on instead of
  snapping back each loop.

## Games, tools and machines

- [snake](snake/README.md): a whole game in five Kernels: a row of state
  and a grid of cells fed back through the Graph, arrow keys, ticks from
  the clock, food by hashing and probing; the replay plays it through.
- [tetris](tetris/README.md): the game with its pieces as a table of bit
  masks, `[[Int; 4]; 7]`, and every cell question as integer vector
  arithmetic; the replay clears a line.
- [slg](slg/README.md): a tactics game in the NES manner: a turn machine
  that issues orders, a unit row that applies them, movement ranges as a
  relaxation sub-graph placed three times, an enemy that walks to the
  fords; the replay asserts on the state rows.
- [tiles](tiles/README.md): a tile map as a `Grid<Tile>` behind a `type`
  alias, flooded a step per frame, drawn through an atlas image; two grids
  of different element types in one Kernel.
  The atlas is a file the Host reads and watches, baked by `run --save`.
- [voxel](voxel/README.md): a voxel editor, the first tool: a
  `Volume<Int>` edited a voxel per click under a minimal bar with a label
  from the text library, the pointer's voxel found by a grid walk in a
  state row, a backup volume for undo, the same walk per pixel for the
  picture.
- [chip8](chip8/README.md): a CHIP-8 interpreter, the first machine: RAM,
  registers and screen as memories a `scan` reads and writes step by
  step and feeds back frame to frame; every opcode; the replay plays it
  with the keys.
- [thumb](thumb/README.md): a Thumb core, the ARM7TDMI's 16-bit
  instruction set as a `scan` over RAM and registers, every format,
  running a hand-assembled program that paints a framebuffer; the JIT
  steps it faster than a Game Boy Advance's own clock.
- [async](async/README.md): the Thumb machine off the frame, `async =
  true;` on its node: each step runs on a thread while the picture keeps
  its pace, and the replay's `settle` waits for a step to land.

## Agents, packages and files

- [scanline](scanline/README.md): the third phase's demonstration on
  record: an agent, over the creation protocol, proposes a Kernel, is
  refused at a line and column, repairs it, publishes, composes it into
  the running scene and sees the picture change; a test replays the
  record.
- [migrate](migrate/README.md): state migration on record: an agent
  changes what a running tally carries, publishes the new Kernel and a
  migration, and the count goes on; without a migration the state
  restarts and the report says so.
- [package](package/README.md): a scene using the `pixelfx.orlapkg`
  package beside it for its Kernels and placing its NES preset by path;
  the package is what `orla pack` carries as one file.
- [level](level/README.md): a binary format nothing in Orla knows, read
  by a Kernel: the scene takes the file with `bytes`, and the Kernel
  walks the header, the tiles and the entity list. No importer, no Rust.
- [clip](clip/README.md): an input read frame by frame from a directory,
  `frames "clip/" at 2`, the frame of the moment at each evaluation; with
  `render`, the video pipeline's input half.
