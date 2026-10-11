# Neighbours

65,536 bodies that push apart close up and draw together further out.
Each body reads the nine grid cells around its own, and the grid is
rebuilt every frame from Kernels:

- `flock.counts@1`: a scatter with `sum`, the bodies in each cell.
- `flock.ends@1`: `prefix with sum` over the counts, where each cell's
  run ends in sorted order.
- `packages/sort.entpkg`: a stable radix sort that puts each cell's
  bodies side by side.

`radius` is one cell; `cap` is how many bodies of a cell a body reads.
`--param grid=false` compares every pair instead. `--param count`,
`cols`, `radius` (`1/cols`) and `bits` (enough for `cols * cols`) set the
size.

`tests/neighbours.replay` checks on all three backends that the sort is
ordered and stable, that no cell exceeds `cap`, and that every-pair draws
the same picture to rounding.

```sh
entelechy player examples/neighbours/scene.esn
entelechy replay examples/neighbours/tests/neighbours.replay
```
