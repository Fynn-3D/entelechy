# Scroll

A 64 by 64 tile scrolled endlessly at three texels a pixel, in three
bands: nearest reads through `mod` (exact wrap); `at_linear` through
`mod` (a seam every 192 pixels, since the blend cannot reach across the
tile edge); and `at_linear_wrapped` from the shared library, which
blends four wrapped `at` reads and has no seam.

```sh
cargo run --release -p orla-player -- examples/scroll/scene.osn
cargo run -- replay examples/scroll/tests/scroll.replay
```
