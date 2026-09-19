# Volume

A `Volume<Float>` of cloud density, 48 by 48 by 32 elements, filled by one
Kernel with three-dimensional `coord()` and `domain()`, and ray-marched
into an image by another with `at(density, p * size(density))`. On the
GPU the volume stays resident between the two nodes and across frames;
only the final image is presented.

```sh
cargo run --release -p orla-player -- examples/volume/scene.osn --backend gpu
cargo run -- replay examples/volume/tests/cloud.replay
```
