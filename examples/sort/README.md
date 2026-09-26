# Sort

Sorting as two map Kernels: `sort.rank` counts, for each value, the
values below it (ties by index), and `sort.gather` picks the value of
each rank. Each is a bounded `loop` over `count(values)`, so the work is
quadratic. `sort.bars` draws the values before and after; they reshuffle
every half second. The [`sort` package](../../packages/sort.orlapkg) is
a radix sort, built from `prefix` and `scatter`.

```sh
cargo run --release -p orla-player -- examples/sort/scene.osn --backend gpu
cargo run -- replay examples/sort/tests/sort.replay
```
