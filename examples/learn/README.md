# Learn

A network that learns, all of it Kernels: a 2-16-1 perceptron trained by
gradient descent to tell the inside of a circle from the outside, which
no line can. The weights are a Field of 65 floats carried frame to frame
through `<~`; a step is one scatter (element `e` is weight `e % 65` of
point `e / 65`: one forward pass, one term of the chain rule, added into
the weight's sum, so the gradient needs no array) and one map (the
update). The scene places the step eight times a frame with `times`,
carrying the weights on. `loss` is a reduce over the points; `view`
paints the network's answer over the plane, warm for one and cool for
zero, with the circle drawn faintly; `dots` scatters the points onto it.

```text
points ─┬─ train (x8: grads, a scatter summing gradients; step) ─┬─ loss
init ───┤            weights carried frame to frame               ├─ view ─ dots
zeros ──┘  (a filled Field, written in the scene)                 ┘
```

Watch it learn in the Player; `render` writes the frames. The replay
pins the loss at the start and after 160 steps and three pixels of the
answer, on all three backends; the CPU and the JIT agree bit for bit.

```sh
cargo run --release -p orla-player -- examples/learn/scene.osn
cargo run --release -- replay examples/learn/tests/learn.replay --backend jit
cargo run --release -- render examples/learn/scene.osn --frames 120 --backend gpu --out /tmp/learn
```

What the round found, and what came of it: a reduction's output port
wanted a declared `type`, and now takes its element (`output result:
Loss;`); a scatter mapped over its first input, so a step over `points
x weights` elements needed a Field of that size, and now sets `domain =
16640;` instead; and the learning rate is the whole story, 0.5 barely
moves in 480 steps, 5 diverges, 2 learns.
