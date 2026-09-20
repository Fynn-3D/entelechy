# Tetris

The game in five Kernels, the shape of the snake example: a row of
state and a grid of settled cells fed back through the Graph, the rules
once per frame, the picture from both. What is new is the data: the
seven pieces in their four rotations are one table, `[[Int; 4]; 7]` of
sixteen-bit masks over a 4 by 4 box, and every question about a cell is
integer vector arithmetic against it. `covers(mask, c)` tests a bit,
`fits(board, mask, at)` walks the sixteen cells of the box with
`ivec2(index() % 4, index() / 4)`, a landing merges the piece by colour,
and the rows above a full row fall by counting, from the floor up,
which unfull row lands on each cell's own. Left and right step, up
turns, down hurries, space drops; the next piece waits in a panel to
the right; the board reddens when a piece cannot spawn.

The replay deals a known first piece from `param seed`, prefills the
bottom row but for one cell with `param prefill`, moves, turns and
drops the piece into the gap, and checks that the row clears and the
cells above fall.

```sh
cargo run --release -p orla-player -- examples/tetris/scene.osn --scale 2
cargo run -- replay examples/tetris/tests/tetris.replay
```
