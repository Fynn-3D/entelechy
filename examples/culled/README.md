# culled

A world of ten million triangles, of which a frame places the ones it
can see.

The world is `cols` by `cols` chunks, each a `side` by `side` patch of
ground. `world.visible` writes one flag per chunk — is any of it in
front of the eye and inside the two side planes — and the compact
package turns those flags into the kept chunk indices and how many there
are. Every node that builds geometry is sized by that number:

```
node kept: graph "../../packages/compact.orlapkg/graphs/compact.osn" {
    flags <- visible.result;
}
node ground: world.ground@1 {
    kept <- kept.indices;
    count <- kept.count;
    domain = max(value(count), 1) * side * side;
}
```

A chunk nobody can see is not skipped: it is never placed. There is no
Field holding it, no vertex projected for it, no triangle handed to the
rasterizer.

`tests/culled.replay` is the claim: the same frame with the culling on
and off is the same picture to the last pixel, having built a third of
the geometry. `--param culling=false` is that switch.

Measured at 1920x1080, least of runs of twenty frames:

| world | placed | whole world placed |
|---|---|---|
| 64x64 chunks, 10.0M triangles | 12.0 ms | 26.1 ms |
| 128x128 chunks, 40.1M triangles | 13.1 ms | 73.0 ms |
| 256x256 chunks, 160.6M triangles | 13.6 ms | (does not fit) |

Sixteen times the world for a millisecond and a half, which is the cull
test and its prefix sum over sixteen times the chunks. What a frame
costs is what it can see, and the world above that is free — which is
the answer to how a scene carries geometry it could never draw.

Nothing here is a renderer. The shading is one lambert term and a fog;
the mechanism is `value(count)` and the compact package, and what an
agent builds on top of them is its own.

```sh
cargo run --release -p orla-player -- examples/culled/scene.osn --backend gpu
cargo run -- replay examples/culled/tests/culled.replay
```
