# Sort

Sorting as two map Kernels, no primitive needed. `sort.rank` counts, for
each value, how many values are smaller (ties broken by index), and
`sort.gather` picks, for each index, the value with that rank. Each is a
bounded `loop` over `count(values)`, so the work is quadratic: exact on
both backends and fine for rows in the thousands; larger sorts wait for a
scatter primitive, which the map mode does not have by design. `sort.bars`
draws the values before and after, and the values reshuffle every half
second.

```sh
cargo run --release -p orla-player -- examples/sort/scene.osn --backend gpu
cargo run -- replay examples/sort/tests/sort.replay
```
