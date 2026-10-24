# music

Music made, not recorded: a mood (a mode, a key, a tempo, its chords,
how busy) played by a band (which instruments take the tune, the
counter-tune, the chords, the arpeggio, the bass and the drums) from a
seed, the seed making the tune's motif, so one seed through every mood
is one score. A palette of instruments ships with it
(`data/instruments.csv`: strings, brass, flutes, bells, plucked strings,
an old console's voices); which moods and bands a game has are its own
tables (the study's `game_music` keeps one).

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

`data/instruments.csv`: the instruments' palette.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`song.replay`, over two moods and two bands of its own (`tests/data/`).
Every Kernel of the package is reached by one of them.
