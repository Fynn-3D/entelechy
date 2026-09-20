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
  seconds per frame on the CPU.
- [life](life/README.md): Conway's Life on a `Grid<Cell>` of real booleans
  and exact integers through a feedback binding, painted with the pointer,
  with a declarative replay test.
- [particles](particles/README.md): a `Rows<Particle>` advanced per element
  and gathered into an image; the first scene whose data is not an image.
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
  under physically based shading at 720p, orbited by a drag, recoloured
  and refinished from a centred toolbar; one state row, one picture, the
  replay asserts on the row.
- [fluid](fluid/README.md): smoke by the Navier-Stokes equations: a
  staggered velocity grid carried by itself, a pressure solve as one
  Jacobi Kernel placed forty times, the projection, the smoke it carries;
  the pointer stirs, a replay reads the divergence before and after.
- [voxel](voxel/README.md): a voxel editor, the first tool: a
  `Volume<Int>` edited a voxel per click under a minimal bar, the
  pointer's voxel found by a grid walk in a state row, a backup volume
  for undo, the same walk per pixel for the picture.
- [sampler](sampler/README.md): a drum from a WAV file the Host reads
  and watches, played by a state row that keeps the playhead in the
  file's frames and resamples to the device's rate with `at_linear`;
  the file was baked by a Kernel and `run --save`.
- [model](model/README.md): a crate from a GLB, brought in by `orla-tools import`
  as vertex and triangle rows and a PNG the scene reads as inputs, spun
  and rasterized by the native rasterizer, textured with `at_linear`.
