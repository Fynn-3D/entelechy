# navigation

Ways over a grid: how far each cell is from a seed by what each costs to
enter, relaxed a pass at a time, and the step a walk takes toward it.

**Layer:** Foundation. **Uses:** no other package.

```text
use "navigation.entpkg";
```

## Kernels

- `kernels/grid_path.ent` — Ways over a grid of cells. Kernels:
  `navigation.relax@1`.

## Tested by

[slg](../../examples/slg/README.md), whose replays hold it on every backend.
