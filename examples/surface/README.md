# surface

A blob meshed out of a volume every frame, by the cells its surface
passes through.

The volume is `size` cells on a side — 128 by default, so 2,097,152 of
them — and holds an integer field: three balls on integer paths, each a
quadratic falloff, with the surface's threshold already taken off, so
the surface is where the field changes sign. `surface.flags` writes one
flag a cell, the compact package ranks them, and the geometry nodes are
sized by the count:

```
node kept: graph "../../packages/compact.orlapkg/graphs/compact.osn" {
    flags <- flags.result;
}
node corners: surface.corners@1 {
    kept <- kept.indices;
    count <- kept.count;
    domain = max(value(count), 1) * 8;
}
```

At the first frame the surface touches 19,132 of the two million cells —
nine in a thousand — and the graph places 229,584 triangles. A cell the
surface misses is not skipped: it is never placed.

The field is integer arithmetic throughout, so every backend flags
exactly the same cells: `tests/surface.replay` pins the counts with no
tolerance at all, on the interpreter, the JIT and the GPU.
`--param compaction=false` flags every cell instead — the cells the
surface misses put their eight corners on one point, and a triangle of
no area is not drawn — so the same picture comes out of thirty times the
geometry, which is the other half of the replay.

## What it costs, and what it asks for

At 1280x720 on the GPU, the least of five runs of thirty frames: 28.3 ms
a frame, of which the rank over the two million cells is 12.2 — the most
expensive node in the scene, more than the rasterizer's 4.3. A prefix
sum is ordered work and walks its row on one thread, and a GPU session
hands its scans to the JIT, so the same node costs 11.2 ms on the JIT,
where the whole frame is 122.9.

That is this scene's point. It is the first one here that wants a rank
over a million things rather than over the few thousand chunks `culled`
decides about, and what it pays for it is nearly half its frame. A
blocked prefix sum on the device would be about a quarter of a
millisecond of that 12.2 (HANDOFF's Open section has the measurement),
which would take this frame to about 16 ms and leave the scatter that
gathers the kept indices, at 7.1 ms, as the next thing to look at.

Placing every cell instead costs 115.9 ms a frame for the same picture,
so the compaction is worth 4.1 times even while the rank is this
expensive.
