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
- [volume](volume/README.md): a `Volume<Float>` of cloud density ray-marched
  into an image; the volume never leaves the GPU.
