# Sampler

A sound from a file. `input kick: Rows<Float> = "assets/kick.wav";`
reads a WAV as rows of floats, and the Host watches the file. The scene
sets `clip_rate` to the file's sample rate; `sampler.state` advances the
playhead in clip frames, and `sampler.block` reads the clip with
`at_linear` at the ratio of the two rates, so pitch holds at any device
rate. The clock retriggers every second; so does Space.

`sampler.kick`, a falling-pitch decaying sine, is baked to the WAV by
`bake.esn`. Any drum in its place plays.

```sh
entelechy run examples/sound/parts/sampler/bake.esn --save kick=examples/sound/parts/sampler/assets/kick.wav
entelechy player examples/sound/parts/sampler/scene.esn --scale 3
entelechy replay examples/sound/tests/sampler.replay
```
