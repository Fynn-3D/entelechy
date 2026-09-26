# Spectrum

A sound and its spectrum. `voice` writes a block of samples each frame
(a pitch with two partials and vibrato); `bins` transforms it into
frequency bins, one map over `count(block)`; `peak` finds the loudest
bin; `draw` paints the bars and `text.label` names the frequency. The
block is the scene's sound output. The scene sets the pitch and the
bins' spacing.

```text
voice ─┬─ bins ─┬─ peak ─┬─ hertz ─ label ─ glyphs ─┐
       │        └─ view ─────────────────────────── shown
       └─ (the scene's sound output)
```

```sh
cargo run --release -p orla-player -- examples/spectrum/scene.osn
cargo run --release -- replay examples/spectrum/tests/spectrum.replay
cargo run --release -- run examples/spectrum/scene.osn --window 320x200 --save voice.result=tone.wav
```
