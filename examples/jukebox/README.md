# Jukebox

A game's music made, not recorded, by the
[music package](../../packages/music.entpkg/package.esn): a
mood (title, town, field, battle and the rest) played by a band
(orchestral, chiptune, folk and others) from a seed. The seed makes the
tune's motif, so one seed heard through every mood is one score whose
songs share a theme. The moods, bands and instruments are rows in the
package's `data/` tables.

Up and Down change the mood, Left and Right the band, R a new theme,
S plays the song again, and A tours the whole score.

```sh
entelechy player examples/jukebox/scene.esn
entelechy replay examples/jukebox/tests/*.replay --backend gpu
```

`tests/jukebox.replay` checks the first block's length and words, a mood
and a band changed by the keys and the track begun again, and two
samples of the town's song two seconds in.
