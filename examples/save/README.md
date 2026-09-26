# Save

A tally kept across runs; the session, not a Kernel, writes the file.

```text
fresh ─┐
       ├─ saved (input, else fresh) ─ tally (<~ itself) ─┬─ show
file ──┘                                                 └─ kept (to the file)
```

`save.tally` counts presses of space, once per press. The scene writes
its result `to "saves/tally.field"` after each committed change, when
started with `--allow file.write=examples/save/saves`; without it, the
session reports the skipped write. On start, `saved` reads the file, or
`fresh` when there is none. `save.show` lights one cell per press.
`tests/tally.replay` uses a fixture that writes nothing.

```sh
orla player examples/save/scene.osn --allow file.write=examples/save/saves
orla replay examples/save/tests/tally.replay
```
