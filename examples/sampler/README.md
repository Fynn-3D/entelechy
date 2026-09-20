# Sampler

A sound from a file: `input kick: Rows<Float> = "assets/kick.wav";` reads
a WAV into the Host as rows of floats, and the Host watches the file. The
sample rate is a fact about the file, not the Field, so the loader reports
it and the scene sets `clip_rate`; `sampler.state` keeps the playhead in
the clip's own frames and advances it by a block of the device's frames
scaled by the two rates, and `sampler.block` reads the clip with
`at_linear` at the same ratio, so the pitch is right whatever the sound
card runs at. The clock retriggers the kick every second; Space too.

The kick was made here: `sampler.kick` is a sine whose pitch falls and
whose loudness decays, and `bake.osn` with `run --save` wrote it as the
WAV the scene reads. Replace the file with any drum and the scene plays
it.

```sh
cargo run -- run examples/sampler/bake.osn --save kick=examples/sampler/assets/kick.wav
cargo run --release -p orla-player -- examples/sampler/scene.osn --scale 3
cargo run -- replay examples/sampler/tests/sampler.replay
```
