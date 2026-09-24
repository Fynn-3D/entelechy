# Save

A tally kept across runs, with nothing in any Kernel that touches a
file.

```text
fresh ─┐
       ├─ saved (input, else fresh) ─ tally (<~ itself) ─┬─ show
file ──┘                                                 └─ kept (to the file)
```

`save.tally` counts presses of space, holding whether space was down the
frame before so a press counts once. The scene writes its result `to
"saves/tally.field"`: the session writes the file after an evaluation that
changed the tally commits, whole, and only because the Player was started
with `--allow file.write=examples/save/saves`; without it, it says what it
did not write and why. The next time the scene starts, the input `saved`
reads the file; while there is no file it reads `fresh`, a tally of
nothing. `save.show` lights one cell per press.

The replay runs the same Kernels through a fixture that saves nowhere (a
replay allows no writes), so it always starts from nothing, and holds that
a press counts once however long it is held.

```sh
cargo run --release -p orla-player -- examples/save/scene.osn --allow file.write=examples/save/saves
cargo run -- replay examples/save/tests/tally.replay
```
