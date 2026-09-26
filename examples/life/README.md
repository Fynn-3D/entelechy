# Life

Conway's Game of Life as three Kernels and one feedback edge. The board
is `type Cells = Grid<Cell>` with `struct Cell { alive: Bool; age: Int; }`;
`show` turns it into pixels.

```text
seed -> life -> show
         ^  |
         '--'   life.previous <~ life.result from seed.result
```

`life` is pure: it reads `previous`, the node's own output from the last
committed evaluation. Hold the mouse button to paint, hold C to clear,
press F5 to restart from the seed.

```sh
cargo run --release -p orla-player -- examples/life/scene.osn --scale 4
cargo run -- replay examples/life/tests/blinker.replay
```
