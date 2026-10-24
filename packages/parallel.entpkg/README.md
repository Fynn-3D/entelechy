# parallel

The data-parallel algorithms other packages build on: a stable sort of
integer keys, the compaction of flagged rows, runs of entries laid out
by owner, and the connected components of nodes joined by edges.

**Layer:** Foundation. **Uses:** no other package.

```text
use "parallel.entpkg";
```

## Kernels

- `kernels/compact.ent` — Stream compaction, the three steps, over a row
  of flags where a non-zero flag keeps that element's index. Kernels:
  `parallel.compact_blank@1`, `parallel.compact_offsets@1`,
  `parallel.compact_total@1`, `parallel.compact_gather@1`.
- `kernels/components.ent` — Connected components: the nodes an edge
  joins share a label, the least index of their component, found by
  hooking roots under roots and flattening the trees, a few passes.
  Kernels: `parallel.components_start@1`, `parallel.components_hook@1`,
  `parallel.components_flatten@1`, `parallel.components_apart@1`.
- `kernels/runs.ent` — Runs. Kernels: `parallel.runs_ends@1`,
  `parallel.runs_entries@1`, `parallel.runs_slots@1`.
- `kernels/sort.ent` — A permutation that puts a row of integer keys in
  order. Kernels: `parallel.sort_identity@1`, `parallel.sort_blank@1`,
  `parallel.sort_next_bit@1`, `parallel.sort_zeros@1`,
  `parallel.sort_place@1`.

## Graph presets

- `graphs/compact.esn` — Filters a row, stream compaction.
- `graphs/components.esn` — Connected components: every node's label,
  the least index of the nodes the edges join it to.
- `graphs/components_pass.esn` — One pass of `components.esn`.
- `graphs/runs.esn` — Runs.
- `graphs/sort.esn` — The permutation that orders `keys`, least
  significant bit first.
- `graphs/sort_pass.esn` — One bit of the sort.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`components.replay`, `rows.replay`. Every Kernel of the package is reached by one of them.

Used by the [culled](../../examples/culled/README.md), [physics](../../examples/physics/README.md), [surface](../../examples/surface/README.md), [swarm](../../examples/swarm/README.md) examples.
