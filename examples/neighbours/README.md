# neighbours

Sixty-five thousand bodies that push apart close up and draw together
further out, on a grid rebuilt every frame.

Every body needs the bodies within `radius` of it. Asking that of every
pair is the whole row per body. Asking it of a grid is the nine cells
around its own, and the grid is three things the language already has:

- `flock.counts@1` — a scatter with `sum`, one per body into its cell:
  how many are in each cell.
- `flock.ends@1` — `prefix with sum` over those counts: where each
  cell's run ends in the sorted order, so a run begins where the one
  before it ended.
- `packages/sort.orlapkg` — the permutation that puts the bodies of a
  cell side by side, one pass a bit, each pass a `prefix with sum` for
  the rank and a scatter for the placement.

Nothing in the engine knows what a grid is. The radius is one cell, so
the nine cells hold every neighbour there is, and `cap` is how many of a
cell a body will look at.

## What it claims

`tests/neighbours.replay` checks three things on the interpreter, the
JIT and the GPU:

- no pair in the sorted order is out of order, and no equal pair is
  swapped — the sort is a sort, and stable;
- no cell holds more bodies than `cap`, which is the one thing that
  would make the grid answer differently from every pair;
- `--param grid=false`, which asks every pair instead, draws the same
  picture. The same forces summed in another order, so to rounding
  rather than to the last bit.

## What it costs

At 640x360 on the GPU, the least of five runs of thirty frames:

| bodies | the grid | every pair |
| --- | --- | --- |
| 16,384 | 4.49 ms | 24.80 ms |
| 65,536 | 10.42 ms | 330.36 ms |

Four times the bodies is twice the frame for the grid and thirteen times
for every pair, which is the difference between a neighbourhood and a
row. Of the 65,536-body frame, the sort's ten passes are 6.2 ms and the
step itself 10.1 — the sort costs about what the work it makes possible
costs, and without it that work is 330.

## What to watch

The bodies start scattered. Filaments appear in the first fifty frames,
draw into drops, and the drops settle into a lattice by about three
hundred — which is what a crowd with a short-range push and a long-range
pull does. `--param count`, `--param cols` (with `--param radius` set to
one cell, `1/cols`) and `--param bits` (enough for `cols * cols`) are
the dials.
