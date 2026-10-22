# genesis

Laws a world writes itself: a law's template filled with the numbers a
world computed, written where the person running it allowed.

**Layer:** Simulation. **Uses:** [`parallel`](../parallel.entpkg/README.md).

```text
use "genesis.entpkg";
```

## Kernels

- `kernels/law.ent` — Filling a law's template with numbers, and the
  operation that writes the law. Kernels: `genesis.widths@1`,
  `genesis.spliced@1`, `genesis.writing@1`.

## Graph presets

- `graphs/law.esn` — A law written from its template.

## Tested by

[genesis](../../examples/genesis/README.md), whose replays hold it on every backend.
