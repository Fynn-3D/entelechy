# Sound

Sound written as Kernels, heard and seen. Two parts, each a graph that
runs on its own, placed together in `scene.esn`: the synth's voice and
the spectrum's voice a fifth above it (330 Hz over 220 Hz) are mixed by
`sound.mix` into one stereo block, the scene's sound output, and
`sound.side` shows the synth's scope beside the spectrum in one window.

The Player shows the first output that is a grid of pixels and plays the
first that is rows of one or two floats. The device's rate reaches a
Kernel as `rate()`, and the device paces evaluation: the next block is
computed when the Player's ring can take it.

```text
synth.sound ─────┐
spectrum.sound ──┴─ mix ─── (the sound output)
synth.view ──────┐
spectrum.result ─┴─ side ── (the picture)
```

## The synth (`parts/synth/`)

A synthesizer as rows:

- `audio.voice` produces a block of 800 frames as `Rows<Voice>`,
  continuing the phase through the Graph's `<~` state.
- `audio.filter` is a one-pole low-pass written as a `scan`, its memory
  carried across blocks through `<~` as the phase is.
- `audio.mix` pans the result into `Rows<Vec2>`.
- `audio.scope` draws the block.

Hold the pointer to bend the pitch with x and pan with y.

## The spectrum (`parts/spectrum/`)

A sound and its spectrum. `voice` writes a block of samples each frame
(a pitch with two partials and vibrato); `bins` transforms it into
frequency bins, one map over `count(block)`; `peak` finds the loudest
bin; `draw` paints the bars and `text.label` names the frequency.

`spectrum.esn` is the graph, its font and caption wired by whoever
places it; `scene.esn` reads them from the text package and places it,
and the whole places it beside the synth with `across = 2`, so its
picture takes half the window.

## Running

```sh
entelechy player examples/sound/scene.esn
entelechy player examples/sound/parts/synth/scene.esn
entelechy player examples/sound/parts/spectrum/scene.esn
entelechy run examples/sound/scene.esn --window 640x400 --out sound.ppm
entelechy run examples/sound/parts/spectrum/scene.esn --window 320x200 --save spectrum.voice.result=tone.wav
entelechy replay examples/sound/tests/sound.replay
entelechy replay examples/sound/tests/scope.replay
entelechy replay examples/sound/tests/spectrum.replay
```
