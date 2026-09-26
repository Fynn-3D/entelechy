# Blur

A five-tap Gaussian as two passes of one Kernel, its weights an array
and its axis a parameter, then an emboss whose 3 by 3 kernel is a
`Mat3`. Reads past an edge are folded back by `mirror`, integer
arithmetic on the cell composed from `at`. The source puts shapes
against the edges, where the folding shows.

```sh
cargo run --release -p orla-player -- examples/blur/scene.osn
cargo run -- replay examples/blur/tests/blur.replay
```
