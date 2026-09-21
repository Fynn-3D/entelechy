# Path

A path tracer that gets better while you watch. Every frame traces
`samples` paths per pixel through `bounces` bounces and adds what they
found to the sum the frames before it left, so the picture is the mean
of every sample so far: noisy at first, clean after a few seconds. Drag
the pointer to move the camera, and the sum starts over.

Both counts are graph parameters, and the loops take them: `loop
samples with ...` around `loop bounces with ...`, a count a scene sets
rather than a number the Kernel was written with. The sum and the count
of samples live in one Image carried frame to frame through `<~`, the
count in the alpha channel, which is also what seeds the frame's
randomness. The room is three spheres and a floor; the lamp is the one
sphere that emits, and the sky is the other light.

```text
begin ─ eye (the pointer drags it) ─┬─ sample (paths, added to last frame's sum) ─ view (the mean)
black (a filled Field) ─────────────┘   sum carried through `<~`
```

```sh
cargo run --release -p orla-player -- examples/path/scene.osn
cargo run --release -- replay examples/path/tests/path.replay
cargo run --release -- render examples/path/scene.osn --frames 32 --backend gpu --window 320x240 --param samples=8 --out /tmp/path
```

The replay pins the sum after eight frames of four samples, four pixels
of the mean, and what a drag does to the count. The CPU and the JIT
trace the same paths word for word; the GPU's differ in the last place.
