# Learn

A 2-16-1 perceptron trained by gradient descent to tell the inside of a
circle from the outside, all in Kernels. The weights are a Field of 65
floats carried through `<~`. A step (`train.esn`) is one scatter that
sums gradients (`domain = 16640;`, 256 points by 65 weights) and one map
that updates; the scene places eight steps a frame. `loss` is a reduce;
`view` paints the network's answer; `dots` scatters the points.

```text
points ─┬─ train (x8: grads, a scatter summing gradients; step) ─┬─ loss
init ───┤            weights carried frame to frame               ├─ view ─ dots
zeros ──┘  (a filled Field, written in the scene)                 ┘
```

The replay pins the loss at the start and after 160 steps, and three
pixels of the answer, on all three backends.

```sh
entelechy player examples/learn/scene.esn
entelechy replay examples/learn/tests/learn.replay --backend jit
entelechy render examples/learn/scene.esn --frames 120 --backend gpu --out /tmp/learn
```
