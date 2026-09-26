# Marching

Marching squares over a field of three orbiting blobs. Each pixel finds
its cell with `icoord() / cell`, reads the four corners into a
`[Float; 4]`, classifies the cell into one of sixteen cases, and looks
the crossed edges up in `segments`, a `[[Int; 4]; 16]`. The crossing
points sit in a `[Vec2; 4]` picked by edge number; the pixel measures its
distance to the one or two segments. Cases and cells are data, not an
`if` chain.

```sh
orla player examples/marching/scene.osn
orla replay examples/marching/tests/marching.replay
```
