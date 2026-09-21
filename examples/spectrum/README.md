# Spectrum

A sound and what it is made of. `voice` writes a block of samples every
frame, at a pitch the scene sets, with two partials over it and a
vibrato; `bins` reads the block back as frequencies; `peak` is the
loudest of them; `draw` paints the bars and `text.label` names the
frequency. The block is the scene's sound output, so the Player plays
what the bars describe.

The transform is one map: element `i` is how much of the block turns at
bin `i`'s frequency, which is a loop over the block's samples. Its count
is `count(block)`, the block's own length, so the Kernel does not know
it when it is written and a scene may hand it a block of any size. The
bins are 55 hertz apart from 110 by default, which the scene sets too.

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

The replay pins the loudest bin and two bin powers at 220 hertz, then
sets the pitch to 440 and pins the bin that takes over.
