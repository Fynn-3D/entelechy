# sound

The sound of effects: each layer of an effects cast heard as its form
sounds (a burst, a hum, a crackle, a whoosh), in its matter's voice (a
timbre row: pitch, brightness, crackle, ring, body), when it begins and
ends, how big it is and where it is from the eye. Which voices a game's
matters have is its own table (the study's `spell_book` keeps one).

**Layer:** Media. **Uses:** [`audio`](../audio.entpkg/README.md), [`effects`](../effects.entpkg/README.md), [`geometry`](../geometry.entpkg/README.md), [`random`](../random.entpkg/README.md).

```text
use "sound.entpkg";
```

## Kernels

- `kernels/cast.ent` — The sounds a voice makes, `s` seconds into it,
  `len` long, at sample `n`, in its element's voice.
- `kernels/mix.ent` — The voices summed into a block of stereo samples.
  Kernels: `sound.sound@2`.
- `kernels/voice.ent` — What each cast's layers are as sound. Kernels:
  `sound.voices@1`.

## Graph presets

- `graphs/cast.esn` — An effects cast heard, a block of stereo samples.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`sounds.replay`, over two effects and their matters' voices of its own
(`tests/data/`). Every Kernel of the package is reached by one of them.
