# Collapse

Wave function collapse as Fields. Each cell of a `Grid<Int>` holds the
set of tiles it could still be, one bit a tile. The tiles are the fifteen
non-blank pipes; two cells fit when their shared sides agree, and the
grid's edge is a closed side. A cell left with no tile forgets.

```text
start ─ steps (×8) ─┬─ pick (the fewest left: a reduce)
                    ├─ settled (one tile, or melted under the pointer)
                    └─ spread (×6: what the neighbours rule out)
                                   └─ tally ─ label ─ glyphs ─ draw
```

A step (`step.osn`) picks the cell with the fewest options, settles it,
and spreads the constraint six times; the scene places eight steps a
frame and carries the grid through `<~`. Hold the pointer down to make
cells forget; idle spots also forget on their own.

Every step is integer work; the replay pins its counts exactly on the
interpreter, the JIT and the GPU.

```sh
orla player examples/collapse/scene.osn
orla replay examples/collapse/tests/collapse.replay
```
