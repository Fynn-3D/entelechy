# decided

How much geometry a frame places, decided inside the frame that places it.

A camera flies over a sheet of ground. `decided.detail` turns where the
camera is into one number — how fine the mesh should be — and writes it
into a Field of one element. Every node that builds the ground is sized
by that number:

```
node ground: decided.ground@1 {
    detail <- detail.result;
    domain = value(detail) * value(detail);
}
node faces: decided.faces@1 {
    detail <- detail.result;
    domain = 2 * (value(detail) - 1) * (value(detail) - 1);
}
```

`value(port)` is the one thing a `domain` may read that is data rather
than shape. The number never leaves the graph: no parameter carries it,
no host decides it. High up the sheet is 24 by 24 vertices; down on the
deck it is 263 by 263 — a hundred times the geometry — and nothing in
the scene is rewritten between the two. `tests/decided.replay` pins the
count at three heights on all three backends, which is what `expect
count` is for.

This is the only way a scene places less work. A Kernel that reads a
count and returns early saves its own arithmetic and still writes an
element for every one of them; what costs is the Field, and only a
shorter `domain` makes the Field shorter. Measured on a sheet of two
million triangles at 1920x1080: the early return took the frame from
5.5 to 4.8 ms, and the same cull through `domain` took it to 2.7. The
price is that the Field holding the number is brought to the Host to
read it, which on a device is a wait — about a fifth of a millisecond,
once a frame, however many nodes read the same Field.

```sh
cargo run --release -p orla-player -- examples/decided/scene.osn --backend gpu
cargo run -- replay examples/decided/tests/decided.replay
```
