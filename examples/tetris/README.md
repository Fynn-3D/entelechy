# Tetris

Tetris in five Kernels: a state row and a grid of settled cells fed
back through the Graph, the rules once per frame, the picture from
both. The seven pieces in four rotations are one `[[Int; 4]; 7]` table
of 16-bit masks over a 4 by 4 box; `covers` and `fits` test cells by
integer arithmetic, and full rows clear by counting. Left and right
step, up turns, down hurries, space drops; the next piece shows to the
right; the board reddens when a piece cannot spawn.

The replay sets `param seed` and `param prefill`, drops a piece into
the gap, and checks the row clears and the cells above fall.

```sh
cargo run --release -p orla-player -- examples/tetris/scene.osn --scale 2
cargo run -- replay examples/tetris/tests/tetris.replay
```
