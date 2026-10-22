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

- `synth.voice` produces a block of 800 frames as `Rows<Voice>`,
  continuing the phase through the Graph's `<~` state.
- `synth.filter` is the audio package's one-pole low-pass in a `scan`, its memory
  carried across blocks through `<~` as the phase is.
- `synth.mix` pans the result into `Rows<Vec2>`.
- `synth.scope` draws the block.

Hold the pointer to bend the pitch with x and pan with y.

## The spectrum (`parts/spectrum/`)

A sound and its spectrum. `voice` writes a block of samples each frame
(a pitch with two partials and vibrato); `bins` transforms it into
frequency bins, one map over `count(block)`; `peak` finds the loudest
bin; `draw` paints the bars and `text.label` names the frequency.

`spectrum.esn` is the graph, its font and caption wired by whoever
places it; `scene.esn` reads them from the widgets package and places it,
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

## Parts that run alone

- [strings](parts/strings/README.md): six plucked strings, each a delay line in a `memory`; `entelechy player examples/sound/parts/strings/scene.esn`, its replays `tests/strings*.replay`.
- [sampler](parts/sampler/README.md): a drum WAV baked by a Kernel, played and resampled with `at_linear`; `entelechy player examples/sound/parts/sampler/scene.esn`, its replays `tests/sampler*.replay`.
- [score](parts/score/README.md): a MIDI file imported, played and drawn as a piano roll, exported back as MIDI; `entelechy player examples/sound/parts/score/scene.esn`, its replays `tests/score*.replay`.
- [music](parts/music/README.md): a game's music made from a mood, a band and a seed by the music package; `entelechy player examples/sound/parts/music/scene.esn`, its replays `tests/music*.replay`.
- [effects](parts/effects/README.md): every event sound of the sound package, in each material and style; `entelechy player examples/sound/parts/effects/scene.esn`, its replays `tests/effects*.replay`.
