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

## What it costs

At 1280x720 on the GPU, the least of five runs of thirty frames: 5.9 ms
a frame, of which the rank over the two million cells is 1.2 — about
what the rasterizer costs.

This scene is why the language has `prefix with sum`. It was written
first against `scan`, the ordered form, and the same frame was 18.7 ms:
the rank alone was 10.0 of it, and the scatter that reads the ranks was
another 4.0 because a scan on a GPU session runs on the JIT, so the row
went to Host memory and had to be uploaded again. Saying the rule
instead of carrying a running total lets the device fold the row in
blocks and keep it where it lies:

| node | `scan` | `prefix with sum` |
| --- | --- | --- |
| kept.offsets | 10.007 ms | 1.194 ms |
| kept.gathered | 4.016 ms | 1.188 ms |
| the frame | 18.68 ms | 5.89 ms |

Placing every cell instead of the ones the surface touches is 63.5 ms a
frame for the same picture, so the compaction is worth eleven times over
— and the rank is what makes the compaction affordable at this size.
