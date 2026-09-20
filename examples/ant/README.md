# Ant

Langton's ant on a grid memory: a `scan` takes one step per tick,
reading the cell under the ant, flipping it, turning by the rule and
stepping on; the grid and the ant's cell are memories that feed back
frame to frame, so a frame is `speed` steps and the picture is the
trail. Cells are integer vectors end to end: `icoord() / zoom` finds a
pixel's cell, `isize(cells)` wraps the walk, the direction table is an
array of `IVec2` picked by the ant's heading, and `put` writes the cell
it leaves by its integer address.

```sh
cargo run --release -p orla-player -- examples/ant/scene.osn
cargo run -- replay examples/ant/tests/ant.replay
```
