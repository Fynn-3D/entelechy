# Data and precision

How Fields hold what they hold: as narrow as it needs, as wide as it
needs, as structs of exact integers and booleans, and kept in a file
across runs. Four parts, each a scene of its own in `parts/<part>/`, and
`scene.osn` placing all four as sub-graphs and drawing their pictures in
one window: packed top left, precision top right, tiles bottom left,
save bottom right.

```sh
orla player examples/data/scene.osn
orla run examples/data/scene.osn --window 640x400 --out data.ppm
orla replay examples/data/tests/data.replay
```

A placed graph reads no files, so a part that reads files is two
documents: `<part>.osn`, the graph with plain inputs, and `scene.osn`,
which reads the files and places it. The composite places the same
graphs and reads the same files itself. Precision reads none; its
`scene.osn` is placed as it is.

`data.panels` in `kernels/data.orla` fits each picture into its panel
without changing its shape, reading it pixel by pixel. In the composite
the save part counts presses of space from nothing and writes nothing: a
placed graph's outputs are wired, never written, so the tally is kept
only when the save part runs as its own scene. `tests/data.replay` checks
a pixel or two of each panel and a press of space.

## Packed

Each Field stored as narrow as what it holds; Kernels still compute in
`Float` and `Int`, and the port type sets the storage:

| Field | Port type | Bytes an element |
| --- | --- | --- |
| the picture, from `albedo.png` | `Grid<Vec4 as unorm8>` | 4 |
| the tile map, from `tiles.bin` | `Rows<Int as u8>` | 1 |
| the height | `Grid<Float as unorm16>` | 2 |
| the normals | `Grid<Vec3 as snorm8>` | 3 |
| the light, carried through `<~` | `Grid<Vec4 as f16>` | 8 |

`light` combines picture, normals and tiles into half-float radiance
carried frame to frame, and `tonemap` makes the picture. A narrow Field
between two nodes is materialized, not fused, because its rounding is
part of the scene. `albedo.png` is baked by `bake.osn`; `tiles.bin` is
64 bytes of tile numbers. `tests/packed.replay` pins height, a normal
and the light with tolerances of one encoding step.

```sh
orla player examples/data/parts/packed/scene.osn
orla run examples/data/parts/packed/bake.osn --save picture.result=examples/data/parts/packed/albedo.png
```

## Precision

Survey coordinates, half a million metres east and four million north,
in `Double` and `Long`. `ring` draws a ring half a millimetre across at
`origin`: the right half tests inside-ness in doubles, the left in
floats, where every pixel collapses to one point and the half is solid.
`corners` places a 2.5 x 1.2 mm part in whole nanometres (`LVec2`);
`edges` measures each edge in doubles, floats and nanometres.

![the ring: floats on the left, doubles on the right](parts/precision/ring.png)

These nodes run on the JIT when the session's backend is the GPU.
`tests/precision.replay` pins the edges, a corner and the picture, then
moves the part near the origin, where both halves draw the ring.

```sh
orla player examples/data/parts/precision/scene.osn
```

## Tiles

A tile map as `Grid<Tile>`, each element `{ kind: Int, height: Float,
passable: Bool }`. `generate` lays out the map; `flood` spreads water to
passable neighbours each frame through `<~`; `render` draws each pixel
from the map and a sprite atlas, at `domain = size(map) * tile`. The
atlas is `assets/atlas.png`, an input the Host reads and watches, so
edits to it show while the Player runs; `bake.osn` draws it.
`tests/tiles.replay` floods an 8 x 6 map from `tests/tiles.osn`.

```sh
orla player examples/data/parts/tiles/scene.osn
orla run examples/data/parts/tiles/bake.osn --save atlas=examples/data/parts/tiles/assets/atlas.png
```

## Save

A tally kept across runs; the session, not a Kernel, writes the file.
`save.tally` counts presses of space, once per press. The scene writes
its result `to "saves/tally.field"` after each committed change, when
started with `--allow file.write`; without it, the session reports the
skipped write. On start, `saved` reads the file, or `fresh` when there
is none. `save.show` lights one cell per press. `tests/save.replay` uses
the fixture `tests/save.osn`, which writes nothing.

```sh
orla player examples/data/parts/save/scene.osn --allow file.write=examples/data/parts/save/saves
```
