# Blur

A five-tap Gaussian as two passes of one Kernel, its weights an array
and its axis a parameter, then an emboss whose 3 by 3 kernel is a
`Mat3` read column by column with the offsets that fetch the
neighbours. Every read past an edge is folded back by `mirror`, two
lines of integer arithmetic on the cell: mirror addressing composed
from `at`, which is why the language has no mode for it. The source
puts a disc against the top edge, a bar along the left and stripes into
a corner, so the edges are where the folding shows.

```sh
cargo run --release -p orla-player -- examples/blur/scene.osn
cargo run -- replay examples/blur/tests/blur.replay
```
