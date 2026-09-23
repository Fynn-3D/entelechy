# Precision

Drawing where a survey puts things. A site plan in a national grid
lives half a million metres east and four million north of its origin;
a float keeps about seven significant digits, so out there it cannot
tell two points a few centimetres apart, and a part a few millimetres
across is a single point. A double keeps about sixteen.

```sh
cargo run --release -p orla-player -- examples/precision/scene.osn
```

`ring` draws a ring half a millimetre across around `origin`, ten
micrometres a pixel. Every pixel's place in the world is computed in
doubles, as a drawing keeps it; the right half then asks, in doubles,
whether the pixel is inside the ring, and the left half asks the same
question in floats of the same coordinates. The left half is solid:
every pixel is the nearest float to the centre, which is a few
centimetres off it, and so every pixel is inside.

![the ring: floats on the left, doubles on the right](ring.png)

`corners` places a part two and a half millimetres by one point two at
the same spot, in whole nanometres (`LVec2`), which is what robust
geometry computes with, and derives each corner in metres from them.
`edges` measures each edge three ways: in doubles, where 2.5 mm comes
out within a few picometres; in floats of the same coordinates, where it
comes out as nothing at all; and in nanometres, exactly.

The numbers in `kernels/precision.orla` are written as they are meant:
`param origin: DVec2 = dvec2(500000.123456789, 4000000.987654321)` keeps
every digit, and `* 0.000000001` beside a double is the double nearest a
nanometre. Nothing here runs on the GPU: a device has no doubles worth
the name, so a session on it runs these nodes on the JIT.

The replay pins the edges, a corner, and the picture on the three
backends, then moves the part near the origin, where a float sees as
well as a double and the left half draws the ring too.
