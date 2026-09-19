# Exposure

The first reduction. `exposure.measure` runs its `compute` once per pixel
of the scene and folds the results with `merge` into one `Rows<Light>`
element holding the frame's luma sum, maximum, and count; both backends
fold in the same pairwise tree, so integers are exact and floats agree
to rounding. `exposure.apply` reads that element with `at(light, 0)` and
scales every pixel so the mean lands on `target`, whatever the sun does.

A reduce node has no `domain`: it produces one element and maps over its
first input. It is never fused, because it neither maps over its output
nor yields one element per consumer element.

```sh
cargo run --release -p orla-player -- examples/exposure/scene.osn --backend gpu
cargo run -- replay examples/exposure/tests/exposure.replay
```
