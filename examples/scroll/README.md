# Scroll

A 64 by 64 tile scrolled without end at three texels a pixel, in three
bands. The top band reads the nearest texel through `mod`, which wraps
exactly. The middle band reads `at_linear` through `mod`: inside the
tile every square's edge is a three-pixel ramp, but at the tile's edge
the blend cannot reach the texel on the other side, so the ramp is
missing there and the seam shows every 192 pixels. The bottom band
blends four `at` reads found through `mod` itself, `blend_wrapped`, and
has no seam: `at_linear_wrapped` in the shared library, ten lines of
Kernel, which is what `at_linear` would otherwise need a mode for.

```sh
cargo run --release -p orla-player -- examples/scroll/scene.osn
cargo run -- replay examples/scroll/tests/scroll.replay
```
