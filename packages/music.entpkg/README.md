# music

Music's words, not a program's music: scales and chords (the modes and
the semitones of their degrees, a note's pitch, the chords stacked on a
degree), an instrument's note (an `Instrument` row: how it sounds, its
envelope, its vibrato, its chorus). A program
makes its songs, tunes and palette of instruments of them (the study's
`game_music` is one).

**Layer:** Media. **Uses:** [`audio`](../audio.entpkg/README.md), [`random`](../random.entpkg/README.md).

```text
use "music.entpkg";
```

## Kernels

- `kernels/instrument.ent` — An instrument and its note.
- `kernels/scale.ent` — Scales and chords.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`notes.replay`.
