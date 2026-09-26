# Packed

Each Field stored as narrow as what it holds; Kernels still compute in
`Float` and `Int`, and the port type sets the storage:

| Field | Port type | Bytes an element |
| --- | --- | --- |
| the picture, from `albedo.png` | `Grid<Vec4 as unorm8>` | 4 |
| the tile map, from `tiles.bin` | `Rows<Int as u8>` | 1 |
| the height | `Grid<Float as unorm16>` | 2 |
| the normals | `Grid<Vec3 as snorm8>` | 3 |
| the light, carried through `<~` | `Grid<Vec4 as f16>` | 8 |

`height` writes 16-bit heights, `normal` writes signed-byte normals,
`light` combines picture, normals and tiles into half-float radiance
carried frame to frame, and `tonemap` makes the picture. Type aliases in
`kernels/packed.orla` name the shared types. Inputs are stored narrow as
they are read. A narrow Field between two nodes is materialized, not
fused, because its rounding is part of the scene.

`albedo.png` is baked by `bake.osn`; `tiles.bin` is 64 bytes of tile
numbers. `tests/packed.replay` pins height, a normal and the light on all
three backends, with tolerances of one encoding step.

```sh
cargo run --release -p orla-player -- examples/packed/scene.osn
```
