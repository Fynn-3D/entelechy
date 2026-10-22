# Culled

A world of ten million triangles; a frame places only the chunks it can
see. The world is `cols` by `cols` chunks, each a `side` by `side` patch.
`world.visible` writes one flag per chunk, the math package turns the
flags into kept indices and a count, and every geometry node is sized by
that count:

```
node kept: graph "../../packages/parallel.entpkg/graphs/compact.esn" {
    flags <- visible.result;
}
node ground: world.ground@1 {
    kept <- kept.indices;
    count <- kept.count;
    domain = max(value(count), 1) * side * side;
}
```

`tests/culled.replay` checks that the frame with culling on and off
(`--param culling=false`) is the same picture to the last pixel. The left
stick or arrow keys steer; without input the camera sweeps on its own.

```sh
entelechy player examples/culled/scene.esn --backend gpu
entelechy replay examples/culled/tests/culled.replay
```
