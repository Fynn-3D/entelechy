# Rubik

A Rubik's cube and a network learning to read it, all Kernels. Drag to
orbit; U, R, F, D, L and B turn a face (shift reverses); space scrambles
ten turns; enter resets.

- The state is 54 stickers in a `Rows<Int>`; a turn is a permutation,
  `perm(m, i)`, from a generated table.
- The picture casts two boxes per pixel: the cube without the turning
  layer, and the turning layer in its own frame.
- A 54-32-1 perceptron trains every frame on 64 cubes scrambled 0 to 8
  turns, labelled by the count: hidden units (map), output error (map),
  gradient (scatter), update (map), weights carried through `<~`.
- `guess` shows `net` (the network's estimate), `you` (turns made) and
  `err` (half the batch's mean squared error).

```text
begin ─ game (input, rules) ─┬─ stickers (a landed turn permutes) ─┬─ guess ─ readout
solved ─────────────────────┘                                      └─ view ─ shown
init ─ step <~ ─ act ─ delta ─ grads (scatter) ─ step; loss           (64 scrambles a frame)
```

`play.osn` is inference only: it reads `weights.field`, saved from 300
frames of `scene.osn`.

```sh
cargo run --release -p orla-player -- examples/rubik/scene.osn
cargo run --release -p orla-player -- examples/rubik/play.osn
cargo run --release -- replay examples/rubik/tests/rubik.replay --backend jit
cargo run --release -- render examples/rubik/scene.osn --frames 300 --backend jit --window 64x64 --out /tmp/rubik --save step.result=examples/rubik/weights.field
```
