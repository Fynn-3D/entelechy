# Async

The Thumb machine off the frame. `machine` is the `thumb.step` scan of
the [thumb](../thumb/README.md) example with `async = true;` and half a
million instructions per step: an evaluation starts the step on a thread
of its own and goes on, `view` shows the RAM the last landed step left,
and `clock` sweeps a hand over it by the frame's time, so the picture
keeps moving while the machine runs. The first step is on the frame, so
`view` has RAM to show; while a step runs, the state it feeds stays as
it was, and the step that lands is read by the next evaluation, which
commits it and starts the next.

```text
rom ─ boot (RAM) ──┬─ machine (async: on its own thread) ─ view ─ clock
start (registers) ─┘
```

The replay pins the sequence: the count the program keeps in RAM after
the first step, the same count while the second runs (and the hand
elsewhere), and the landed count after `settle`, which waits for the
run before the next frame reads it.

```sh
cargo run --release -p orla-player -- examples/async/scene.osn
cargo run --release -- replay examples/async/tests/async.replay --backend jit
cargo run --release -- run --backend jit --window 64x64 examples/async/scene.osn
```
