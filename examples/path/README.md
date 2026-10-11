# Path

A progressive path tracer: three spheres (one emitting) and a floor
under a sky. Each frame traces `samples` paths per pixel through
`bounces` bounces (`loop samples` around `loop bounces`, both graph
parameters) and adds them to a sum carried through `<~`; the sample
count sits in the alpha channel and seeds the frame's randomness. The
picture is the mean. Dragging moves the camera and restarts the sum.

```text
begin ─ eye (the pointer drags it) ─┬─ sample (paths, added to last frame's sum) ─ view (the mean)
black (a filled Field) ─────────────┘   sum carried through `<~`
```

`tests/path.replay` pins the sum after eight frames, four pixels of the
mean, and the reset on drag; the interpreter and the JIT match word for
word, the GPU to the last place.

```sh
entelechy player examples/path/scene.esn
entelechy replay examples/path/tests/path.replay
entelechy render examples/path/scene.esn --frames 32 --backend gpu --window 320x240 --param samples=8 --out /tmp/path
```
