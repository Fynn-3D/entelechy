# Audio

A synthesizer that never mentions a sound card. `audio.voice` produces a
block of 800 frames per evaluation as `Rows<Voice>`, continuing the phase
from the previous block through the Graph's `<~` state; `audio.filter`
is a one-pole low-pass written as a `scan`, ordered work whose carried
value starts from the previous block's last output; `audio.mix` pans the
result into `Rows<Vec2>`; `audio.scope` draws the block. The graph has two
outputs, `view` and `sound`. The Player shows the first output that is a
grid of pixels and plays the first that is rows of one or two floats, so
both are claimed by structure, not by name.

The device adapter in the Player takes each block into a ring and feeds
the sound card from it in the real-time callback; the callback never waits
on the Graph or the GPU and never allocates. The device's clock reaches
the Kernel as `rate()`, context like `time()`, and the device paces the
evaluation: the next block is computed when the ring can take it, so at
48 kHz a block of 800 frames means sixty evaluations a second whatever
the display does, and nothing is dropped. Hold the pointer to bend the
pitch with x and pan with y.

```sh
cargo run --release -p orla-player -- examples/audio/scene.osn
cargo run --release -p orla-player -- examples/audio/scene.osn --no-audio
cargo run -- replay examples/audio/tests/scope.replay
```
