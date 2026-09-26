# Volume

A 48 by 48 by 32 `Volume<Float>` of cloud density, filled by one Kernel
with three-dimensional `coord()` and `domain()`, and ray-marched into an
image by another with trilinear `at_linear`. On the GPU the volume
stays resident between nodes and across frames.

```sh
orla player examples/volume/scene.osn --backend gpu
orla replay examples/volume/tests/cloud.replay
```
