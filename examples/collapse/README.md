# Collapse

Wave function collapse as Fields. Every cell of a grid holds the set of
tiles it could still be, one bit a tile, in a `Grid<Int>`; a step picks
the cell with the fewest left, settles it on one of them, and spreads
what that rules out to its neighbours. A frame takes eight steps, and
the picture is wherever the grid has got to.

The tiles are the sixteen pipes: a tile is four bits, one a side, each
saying the pipe leaves that way, and two cells fit when the sides they
share agree. The blank tile is left out, so every cell leads somewhere,
the grid's edge counts as a closed side, so nothing runs off it, and a
cell left with no tile at all forgets instead of sticking, which is
this collapse's whole backtracking.

```text
start ─ steps (×8) ─┬─ pick (the fewest left: a reduce)
                    ├─ settled (one tile, or melted under the pointer)
                    └─ spread (×6: what the neighbours rule out)
                                   └─ tally ─ label ─ glyphs ─ draw
```

Each of the three kernels is one of the modes: `pick` is a reduce whose
merge keeps the smaller rank, `settled` and `spread` are maps, and the
grid goes back into the next frame through `<~`. Spreading is placed
six times a step, so what a settled cell rules out travels six cells
before the next pick; the count in the corner is a second reduce.

Two of the language's newer forms are here because a collapse wants
them: a `break` ends the walk over the bits as soon as it reaches the
tile it drew, and a `let { x, y } = icoord();` names a cell's
coordinates. The drawing picks its arm with a `match` on the side.

Hold the pointer down and the cells under it forget; the collapse grows
back through the hole. With nothing left to decide a spot of the grid
forgets on its own a few times a second, so the picture is never
finished and never the same twice.

```sh
cargo run --release -p orla-player -- examples/collapse/scene.osn
cargo run --release -- replay examples/collapse/tests/collapse.replay
```

Every step is integer work, so the three backends agree exactly: the
replay's counts are pinned with no tolerance at all, on the interpreter,
the JIT and the GPU.
