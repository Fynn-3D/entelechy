# parallel

The data-parallel algorithms other packages build on: a stable sort of
integer keys, the compaction of flagged rows, and runs of entries laid
out by owner.

**Layer:** Foundation. **Uses:** no other package.

```text
use "parallel.entpkg";
```

## Kernels

- `kernels/compact.ent` — Stream compaction, the three steps, over a row
  of flags where a non-zero flag keeps that element's index. Kernels:
  `parallel.compact_blank@1`, `parallel.compact_offsets@1`,
  `parallel.compact_total@1`, `parallel.compact_gather@1`.
- `kernels/runs.ent` — Runs. Kernels: `parallel.runs_ends@1`,
  `parallel.runs_entries@1`, `parallel.runs_slots@1`.
- `kernels/sort.ent` — A permutation that puts a row of integer keys in
  order. Kernels: `parallel.sort_identity@1`, `parallel.sort_blank@1`,
  `parallel.sort_first_bit@1`, `parallel.sort_next_bit@1`,
  `parallel.sort_zeros@1`, `parallel.sort_place@1`.

## Graph presets

- `graphs/compact.esn` — Filters a row, stream compaction.
- `graphs/runs.esn` — Runs.
- `graphs/sort.esn` — The permutation that orders `keys`, least
  significant bit first.
- `graphs/sort_pass.esn` — One bit of the sort.

## Tested by

[culled](../../examples/culled/README.md), [neighbors](../../examples/neighbors/README.md), [physics](../../examples/physics/README.md), [surface](../../examples/surface/README.md), whose replays hold it on every backend.
