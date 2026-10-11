# Score

A MIDI file played and drawn. `entelechy-import` reads `song.mid` into rows
of notes (start, duration, key, velocity, channel), which the scene
takes as an input Field:

```sh
entelechy import examples/score/song.mid --out examples/score/assets
```

`voice` sums the sounding notes into each sound block; `roll` paints
them as a piano roll with a playhead. Both loop over `count(song)`, so
any song length works. Channel 0 (melody) and channel 1 (bass) take
different tones.

```text
song.mid ─ entelechy-import ─ song.notes.field ─┬─ voice (the sound)
                                           └─ roll (the picture)
```

```sh
entelechy player examples/score/scene.esn
entelechy replay examples/score/tests/score.replay
entelechy run examples/score/scene.esn --window 480x240 --save voice.result=bar.wav
```

Notes written by a Kernel export the same way: `--save
notes.result=song.field`, then `entelechy-export song.field --out song.mid
--beats 100`.
