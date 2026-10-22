# music

A game's music made, not recorded: a mood played by a band from a seed,
the seed making the tune's motif, so one seed through every mood is one
score.

**Layer:** Media. **Uses:** [`audio`](../audio.entpkg/README.md), [`random`](../random.entpkg/README.md).

```text
use "music.entpkg";
```

## Kernels

- `kernels/instrument.ent` — An instrument's note.
- `kernels/part.ent` — A song's parts at a moment.
- `kernels/song.ent` — A track's music, a block of stereo samples.
  Kernels: `music.play@1`.
- `kernels/theory.ent` — What a song is made of.

## Tables

`data/bands.csv`, `data/instruments.csv`, `data/moods.csv`, `data/names.txt`

## Tested by

[jukebox](../../examples/jukebox/README.md), whose replays hold it on every backend.
