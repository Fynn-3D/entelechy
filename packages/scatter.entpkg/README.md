# scatter

Many things scattered over the ground by layers, round a center, one
place always growing the same things.

**Layer:** Graphics. **Uses:** [`parallel`](../parallel.entpkg/README.md), [`random`](../random.entpkg/README.md).

```text
use "scatter.entpkg";
```

## Kernels

- `kernels/placement.ent` — Things are scattered over cells round a
  center (the camera, often) (`scatter.around`). Kernels:
  `scatter.around@1`.

## Tested by

[material_balls](../../examples/material_balls/README.md), whose replays hold it on every backend.
