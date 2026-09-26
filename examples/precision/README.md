# Precision

Survey coordinates, half a million metres east and four million north,
in `Double` and `Long`.

- `ring` draws a ring half a millimetre across at `origin`: the right
  half tests inside-ness in doubles, the left in floats, where every
  pixel collapses to one point and the half is solid.
- `corners` places a 2.5 x 1.2 mm part in whole nanometres (`LVec2`) and
  derives its corners in metres.
- `edges` measures each edge in doubles, floats and nanometres.

![the ring: floats on the left, doubles on the right](ring.png)

These nodes run on the JIT when the session's backend is the GPU.
`tests/precision.replay` pins the edges, a corner and the picture on all
three backends, then moves the part near the origin, where both halves
draw the ring.

```sh
cargo run --release -p orla-player -- examples/precision/scene.osn
cargo run -- replay examples/precision/tests/precision.replay
```
