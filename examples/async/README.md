# Async

The Thumb machine off the frame. `machine` is the `thumb.step` scan of
the [thumb](../thumb/README.md) example with `async = true;`: an
evaluation starts the step on its own thread and continues, `view` shows
the RAM the last landed step left, and `clock` sweeps a hand over it by
the frame's time. The first step runs on the frame; a landed step is
committed by the next evaluation, which starts the next step.

```text
rom ─ boot (RAM) ──┬─ machine (async: on its own thread) ─ view ─ clock
start (registers) ─┘
```

The replay pins the count in RAM after the first step, the same count
while the second runs, and the landed count after `settle`.

```sh
orla player examples/async/scene.osn
orla replay examples/async/tests/async.replay --backend jit
orla run --backend jit --window 64x64 examples/async/scene.osn
```
