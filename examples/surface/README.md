# surface

A blob meshed from a volume every frame, placing geometry only for the
cells its surface passes through.

The volume is `size` cells on a side (128 by default) holding an
integer field: three balls on integer paths, offset so the surface is
where the field changes sign. `surface.flags` writes one flag per cell,
the compact package ranks them with `prefix with sum`, and the geometry
nodes are sized by the count:

```
node kept: graph "../../packages/compact.entpkg/graphs/compact.esn" {
    flags <- flags.result;
}
node corners: surface.corners@1 {
    kept <- kept.indices;
    count <- kept.count;
    domain = max(value(count), 1) * 8;
}
```

The field is integer arithmetic, so every backend flags the same cells:
`tests/surface.replay` pins the counts exactly on the interpreter, the
JIT and the GPU. `--param compaction=false` flags every cell instead;
the missed cells collapse to degenerate triangles and the picture is
the same, which the replay also checks.
