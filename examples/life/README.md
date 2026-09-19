# Life

Conway's Game of Life as three Kernels and one feedback edge. The board is
a `Grid<Cell>` with `struct Cell { alive: Bool; age: Int; }`, named once as
`type Cells`; only `show` turns it into pixels.

```text
seed -> life -> show
         ^  |
         '--'   life.previous <~ life.result from seed.result
```

The `life` Kernel is pure: it reads `previous`, which the Graph binds to the
node's own output as committed by the previous evaluation. State lives in the
Graph, commits atomically at the end of each evaluation, and never advances
past a failed one. Hold the mouse button to paint cells, hold C to clear,
press F5 to restart from the seed.

```sh
cargo run --release -p orla-player -- examples/life/scene.osn --scale 4
```

The declarative test in `tests/` steps a lone blinker, paints, clears, and
resets, checking pixels after each frame:

```sh
cargo run -- replay examples/life/tests/blinker.replay
```
