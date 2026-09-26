# Snake

Snake in five Kernels, two fed back through the Graph.

```text
start ─┐                    ┌─ view (draw)
       ├─ step (rules) ─────┤
empty ─┴─ board (cells) ────┘
```

- `snake.step`: the rules over one `Game` row: steering, one cell per
  tick, death at a wall or the body, eating, and new food found by a
  hash and a bounded `loop`.
- `snake.board`: a grid of `Cell`, the body as a time-to-live counting
  down from the head.
- `snake.draw`: the board at `size(board) * cell`, red once dead.

F5 in the Player, or `reset` in a replay, restarts. The replay covers
moving, turning, an ignored reversal, eating, dying, and reset.

```sh
orla player examples/snake/scene.osn --scale 2
orla replay examples/snake/tests/snake.replay
```
