# Exposure

A reduction. `exposure.measure` runs `compute` once per pixel and folds
the results with `merge` into one `Rows<Light>` element holding the
frame's luma sum, maximum and count. `exposure.apply` reads it with
`at(light, 0)` and scales every pixel so the mean lands on `target`.

```sh
cargo run --release -p orla-player -- examples/exposure/scene.osn --backend gpu
cargo run -- replay examples/exposure/tests/exposure.replay
```
