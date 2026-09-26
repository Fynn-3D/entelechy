# Sampler

A sound from a file. `input kick: Rows<Float> = "assets/kick.wav";`
reads a WAV as rows of floats, and the Host watches the file. The scene
sets `clip_rate` to the file's sample rate; `sampler.state` advances the
playhead in clip frames, and `sampler.block` reads the clip with
`at_linear` at the ratio of the two rates, so pitch holds at any device
rate. The clock retriggers every second; so does Space.

`sampler.kick`, a falling-pitch decaying sine, is baked to the WAV by
`bake.osn`. Any drum in its place plays.

```sh
orla run examples/sampler/bake.osn --save kick=examples/sampler/assets/kick.wav
orla player examples/sampler/scene.osn --scale 3
orla replay examples/sampler/tests/sampler.replay
```
