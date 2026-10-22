# sound

Sound effects: a game's events in a material or a style, and the casts
of the effects package heard as they are drawn.

**Layer:** Media. **Uses:** [`audio`](../audio.entpkg/README.md), [`effects`](../effects.entpkg/README.md), [`random`](../random.entpkg/README.md).

```text
use "sound.entpkg";
```

## Kernels

- `kernels/cast.ent` — The sounds a voice makes, `s` seconds into it,
  `len` long, at sample `n`, in its element's voice.
- `kernels/event.ent` — The sounds of a game's events, each a cue (the
  audio package's) a project sets going. Kernels: `sound.events@1`.
- `kernels/mix.ent` — The voices summed into a block of stereo samples.
  Kernels: `sound.sound@1`.
- `kernels/voice.ent` — What each cast's layers are as sound. Kernels:
  `sound.voices@1`.

## Graph presets

- `graphs/cast.esn` — The vfx package's casts heard.

## Tables

`data/materials.csv`, `data/styles.csv`, `data/timbres.csv`

## Tested by

[soundboard](../../examples/soundboard/README.md), [spells](../../examples/spells/README.md), whose replays hold it on every backend.
