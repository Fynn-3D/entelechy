# Snake

A whole game with nothing added to the language: five Kernels, two of
them fed back through the Graph.

```text
start ─┐                    ┌─ view (draw)
       ├─ step (rules) ─────┤
empty ─┴─ board (cells) ────┘
```

`snake.step` is the rules over one row of `Game`: it reads the previous
state and the previous board, the arrow keys, and the clock; it steers,
advances one cell per tick, dies at a wall or its own body, eats, and
finds new food by hashing a cell and probing forward for an empty one in a
bounded `loop`. `snake.board` is a grid of `Cell` where the body is a
time-to-live counting down from the head, so the tail disappears by
itself and eating simply skips one countdown. `snake.draw` shows it at
`domain = size(board) * cell` and reddens the board once the snake is
dead. F5 in the Player, or `reset` in a replay, starts over.

Both backends play the same game pixel for pixel; the replay drives it
through moving, turning, an ignored reversal, eating, dying at the wall,
and a reset.

```sh
cargo run --release -p orla-player -- examples/snake/scene.osn --scale 2
cargo run -- replay examples/snake/tests/snake.replay
```
