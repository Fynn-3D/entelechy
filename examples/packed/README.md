# Packed

Every Field kept as narrow as what it holds. A Kernel still computes in
`Float` and `Int`; the port type says how the Field keeps its scalars:

| Field | Port type | Bytes an element | As four-byte words |
| --- | --- | --- | --- |
| the picture, from `albedo.png` | `Grid<Vec4 as unorm8>` | 4 | 16 |
| the tile map, from `tiles.bin` | `Rows<Int as u8>` | 1 | 4 |
| the height | `Grid<Float as unorm16>` | 2 | 4 |
| the normals | `Grid<Vec3 as snorm8>` | 3 | 12 |
| the light, carried frame to frame | `Grid<Vec4 as f16>` | 8 | 16 |

```sh
cargo run --release -p orla-player -- examples/packed/scene.osn
```

The files land narrow: the session reads the PNG and the bytes as it
always did and keeps them as the inputs' types say, so the picture takes
a word a pixel from the moment it is read. `height` writes 16-bit
heights, `normal` reads them at its neighbours and writes three signed
bytes a normal, `light` blends the picture, the normals and the tiles into
radiance above one in half floats and carries it from frame to frame
through `<~`, and `tonemap` brings it into a picture. The type aliases in
`kernels/packed.orla` (`type Normals = Grid<Vec3 as snorm8>;`) name what
the Kernels share.

What narrowing does is one rule wherever it happens, the Host's: a
normalized float is clamped and rounded to nearest, ties to even, and a
half float rounds the same way; an integer keeps its low bits. A normal
comes back as a whole number of 127ths on every backend. A narrow Field
between two nodes is written and read rather than fused away, since the
rounding it does is part of the scene.

The picture was baked by `bake.osn` with `run --save
picture.result=examples/packed/albedo.png`; `tiles.bin` is sixty-four
bytes of tile numbers. The replay pins the kept values — the height, a
normal, the light — at the first frame, after the light has carried two
frames on, and after the ground turns, on all three backends; a
device's sine may land one step of the encoding away, which is all the
tolerances allow.
