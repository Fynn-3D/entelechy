# Audio

A synthesizer written as Kernels:

- `audio.voice` produces a block of 800 frames as `Rows<Voice>`,
  continuing the phase through the Graph's `<~` state.
- `audio.filter` is a one-pole low-pass written as a `scan`.
- `audio.mix` pans the result into `Rows<Vec2>`.
- `audio.scope` draws the block.

The graph has two outputs, `view` and `sound`. The Player shows the first
output that is a grid of pixels and plays the first that is rows of one
or two floats. The device's rate reaches the Kernel as `rate()`, and the
device paces evaluation: the next block is computed when the Player's
ring can take it. Hold the pointer to bend the pitch with x and pan with y.

```sh
orla player examples/audio/scene.osn
orla player examples/audio/scene.osn --no-audio
orla replay examples/audio/tests/scope.replay
```
