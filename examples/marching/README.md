# Marching

Marching squares over a field of three orbiting blobs. The field is a
coarse grid; every pixel finds its cell with `icoord() / cell`, reads
the four corners into a `[Float; 4]`, classifies the cell into one of
sixteen cases, and looks the case up in `segments`, a `[[Int; 4]; 16]`
of the edges its contour crosses (a saddle crosses four). The crossing
points sit in a `[Vec2; 4]` picked by edge number, the pixel measures
its distance to the one or two segments, and the inside is the level
blended across the cell. Tables and cells as data, with no `if` chain
anywhere.

```sh
cargo run --release -p orla-player -- examples/marching/scene.osn
cargo run -- replay examples/marching/tests/marching.replay
```
