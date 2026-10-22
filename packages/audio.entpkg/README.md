# audio

Sound made from tones and noise, a block of stereo samples a frame: the
clock, the cues a program's events set going, and the instruments they
sound through.

**Layer:** Media. **Uses:** [`random`](../random.entpkg/README.md).

```text
use "audio.entpkg";
```

## Kernels

- `kernels/clock.ent` — Sound made from tones and noise, a block of
  stereo samples a frame. Kernels: `audio.clock_start@1`,
  `audio.clock@1`, `audio.paced_start@1`, `audio.paced@1`,
  `audio.paced_clock@1`, `audio.paced_size@1`.
- `kernels/cue.ent` — Cues. Kernels: `audio.cues_start@1`.
- `kernels/instrument.ent` — The sounds a cue is made of, as functions
  of time.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`clock.replay`. Every Kernel of the package is reached by one of them.

Used by the [slg](../../examples/slg/README.md), [sound](../../examples/sound/README.md), [spells](../../examples/spells/README.md) examples.
