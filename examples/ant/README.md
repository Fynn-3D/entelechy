# Ant

Langton's ant on a grid memory. A `scan` takes one step per tick: it
reads the cell under the ant, flips it, turns by the rule and steps on.
The grid and the ant's cell are memories fed back frame to frame, so a
frame is `speed` steps and the picture is the trail. Cells are integer
vectors throughout: `icoord() / zoom` finds a pixel's cell, `isize(cells)`
wraps the walk, and `put` writes a cell by its integer address.

```sh
cargo run --release -p orla-player -- examples/ant/scene.osn
cargo run -- replay examples/ant/tests/ant.replay
```
