# Rubik

A Rubik's cube in three dimensions, and a network learning to read it,
all of it Kernels. Drag to orbit; U, R, F, D, L and B turn a face (with
shift, the other way); space scrambles ten turns; enter resets. The
state is 54 stickers, a `Rows<Int>` of face colours, and a turn is a
permutation of them, `perm(m, i)` from a table generated once from a
cubelet model; the picture is two boxes cast per pixel, the cube less
the layer that is turning and the layer in its own turned frame, the
nearer hit painting its sticker with a border.

The network is a 54-32-1 perceptron trained while you play, on cubes it
scrambles itself: every frame, 64 solved cubes turned 0 to 8 random
times, the count their label, the colours the input. A frame's step is
four Kernels: the hidden units of every cube (a map), the output less
the label (a map), the gradient summed weight by weight (a scatter,
element `e` being weight `e % 1793` of cube `e / 1793`, one term of the
chain rule from the units and deltas already computed), and the update
(a map); the weights are carried frame to frame through `<~`. Inference
is one more map over the 54 stickers on screen: `net` is how many turns
from solved the network thinks the cube is, `you` how many you made,
`err` the batch's mean squared error, halved. Two seconds in, the guess
is within a turn or two; scramble and watch it answer.

```text
begin ─ game (input, rules) ─┬─ stickers (a landed turn permutes) ─┬─ guess ─ readout
solved ─────────────────────┘                                      └─ view ─ shown
init ─ step <~ ─ act ─ delta ─ grads (scatter) ─ step; loss           (64 scrambles a frame)
```

`play.osn` is the cube with the network already trained: `weights.field`
is what `render --save step.result=weights.field` kept after 300 frames
of `scene.osn`, read back as `input weights: Rows<Float> =
"weights.field";`; `guess` runs over it, and two `text.label` rows say
what it thinks and what you did. Training is one scene, inference
another, the weights a file between them.

```sh
cargo run --release -p orla-player -- examples/rubik/scene.osn
cargo run --release -p orla-player -- examples/rubik/play.osn
cargo run --release -- replay examples/rubik/tests/rubik.replay --backend jit
cargo run --release -- render examples/rubik/scene.osn --frames 300 --backend jit --window 64x64 --out /tmp/rubik --save step.result=examples/rubik/weights.field
```

What the round found: two libraries that need each other's
declarations take a third they both `use`; a turn that lands must be
over the next frame, or it lands twice; and a learning rate of 0.05 with
a batch of 64 takes the loss from ten to under one in 150 frames.
