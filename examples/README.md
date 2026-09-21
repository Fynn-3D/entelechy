# Examples

Each example is a directory with a `scene.osn` entry, usually a `kernels/`
directory of `.orla` libraries it uses, and, once it has behavior worth
pinning down, a `tests/` directory of `.replay` scripts with their fixture
scenes. `shared/` holds libraries several examples use. Documentation for an
example belongs in its own `README.md`.

```sh
cargo run -- run examples/<example>/scene.osn --out output.ppm --watch
```

- [grayscale](grayscale/README.md): the first proof. A generated gradient flows
  through grayscale and pixelate Kernels; edit any Kernel while it runs.
- [material-ball](material-ball/README.md): a shaded sphere under an orbiting
  light, tone-mapped by a second Kernel; one self-contained file with both
  Kernels inline.
- [raymarch](raymarch/README.md): a signed-distance scene marched with `loop`
  and shaded with `if`, displayed through the shared tone-mapping sub-graph
  with `exposure` as a graph parameter; interactive on the GPU backend,
  a few frames per second on the CPU.
- [life](life/README.md): Conway's Life on a `Grid<Cell>` of real booleans
  and exact integers through a feedback binding, painted with the pointer,
  with a declarative replay test.
- [particles](particles/README.md): a `Rows<Particle>` advanced per element
  and splatted onto a canvas by a scatter, each particle writing its own
  cell in element order; the first scene whose data is not an image.
- [tiles](tiles/README.md): a tile map as a `Grid<Tile>` behind a `type`
  alias, flooded a step per frame, drawn through an atlas image; two grids
  of different element types in one Kernel.
  The atlas is a file the Host reads and watches, baked by `run --save`.
- [volume](volume/README.md): a `Volume<Float>` of cloud density ray-marched
  into an image; the volume never leaves the GPU.
- [mesh](mesh/README.md): a torus as vertex and triangle rows, subdivided
  by Kernels whose output rows are sized from their inputs, projected by a
  Kernel, rasterized by the backend's native `native.rasterize@1`, and
  shaded by a Kernel; the CPU and GPU rasterizers agree.
- [audio](audio/README.md): a synthesizer as rows, its phase carried across
  blocks by Graph state, low-passed by a `scan` whose memory crosses blocks
  the same way, mixed to stereo and drawn as an oscilloscope; the Player
  plays the stereo rows and shows the scope.
- [exposure](exposure/README.md): the first reduction. A `merge` Kernel
  meters a frame's luma into one element and a second Kernel tone-maps by
  it; both backends fold in the same tree.
- [sort](sort/README.md): sorting as two map Kernels, rank then gather,
  each a bounded loop over the rows; no primitive, quadratic, exact on
  both backends.
- [snake](snake/README.md): a whole game in five Kernels: a row of state
  and a grid of cells fed back through the Graph, arrow keys, ticks from
  the clock, food by hashing and probing; the replay plays it through.
- [slg](slg/README.md): a tactics game in the NES manner: a turn machine
  that issues orders, a unit row that applies them, movement ranges as a
  relaxation sub-graph placed three times, an enemy that walks to the
  fords; the replay asserts on the state rows.
- [showcase](showcase/README.md): a gamepad as a signed-distance model
  under physically based shading, filling the window, orbited by a drag,
  recoloured and refinished from a toolbar laid out in logical pixels;
  one state row, one picture, the replay asserts on the row.
- [fluid](fluid/README.md): smoke by the Navier-Stokes equations: a
  staggered velocity grid carried by itself, a pressure solve as one
  Jacobi Kernel placed forty times, the projection, the smoke it carries;
  the pointer stirs, a replay reads the divergence before and after.
- [voxel](voxel/README.md): a voxel editor, the first tool: a
  `Volume<Int>` edited a voxel per click under a minimal bar with a label
  from the text library, the pointer's voxel found by a grid walk in a
  state row, a backup volume for undo, the same walk per pixel for the
  picture.
- [sampler](sampler/README.md): a drum from a WAV file the Host reads
  and watches, played by a state row that keeps the playhead in the
  file's frames and resamples to the device's rate with `at_linear`;
  the file was baked by a Kernel and `run --save`.
- [model](model/README.md): a crate from a GLB, brought in by `orla-import`
  as vertex and triangle rows and a PNG the scene reads as inputs, spun
  and rasterized by the native rasterizer, textured with `at_linear`.
- [chip8](chip8/README.md): a CHIP-8 interpreter, the first machine: RAM,
  registers and screen as memories a `scan` reads and writes step by
  step and feeds back frame to frame; every opcode; the replay plays it
  with the keys.
- [thumb](thumb/README.md): a Thumb core, the ARM7TDMI's 16-bit
  instruction set as a `scan` over RAM and registers, every format,
  running a hand-assembled program that paints a framebuffer; the JIT
  steps it faster than a Game Boy Advance's own clock.
- [text](text/README.md): text as rows of glyphs drawn by a Kernel through
  a font atlas the Host watches; a title, the whole font, and the pointer's
  x as digits; the font was baked from rows of bits by `run --save`.
- [ant](ant/README.md): Langton's ant as a `scan` over a grid memory,
  cells and headings as integer vectors, a direction table as an array;
  the trail feeds back frame to frame.
- [cube](cube/README.md): a cube ray-marched in its own space through
  the inverse of its model matrix, its edges drawn from a corner table
  and an edge table projected by `perspective * model`; matrices end to
  end in one Kernel.
- [tetris](tetris/README.md): the game with its pieces as a table of bit
  masks, `[[Int; 4]; 7]`, and every cell question as integer vector
  arithmetic; the replay clears a line.
- [scroll](scroll/README.md): a tile scrolled without end three ways,
  nearest through `mod`, `at_linear` through `mod` (which seams) and the
  shared library's wrapped blend (which does not); the replay pins the
  seam.
- [blur](blur/README.md): a two-pass Gaussian from a table of weights and
  an emboss whose kernel is a `Mat3`, every edge read folded back by two
  lines of integer arithmetic: mirror addressing composed from `at`.
- [marching](marching/README.md): marching squares over orbiting blobs,
  the sixteen cases a `[[Int; 4]; 16]` and the crossing points a
  `[Vec2; 4]`, no `if` chain anywhere.
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
- [async](async/README.md): the Thumb machine off the frame, `async =
  true;` on its node: each step runs on a thread while the picture keeps
  its pace, and the replay's `settle` waits for a step to land.
