# Score

A MIDI file played and drawn. `orla-import` reads `song.mid` into rows
of notes (start, duration, key, velocity, channel), which the scene
takes as an input Field:

```sh
orla import examples/score/song.mid --out examples/score/assets
```

`voice` sums the sounding notes into each sound block; `roll` paints
them as a piano roll with a playhead. Both loop over `count(song)`, so
any song length works. Channel 0 (melody) and channel 1 (bass) take
different tones.

```text
song.mid ─ orla-import ─ song.notes.field ─┬─ voice (the sound)
                                           └─ roll (the picture)
```

```sh
orla player examples/score/scene.osn
orla replay examples/score/tests/score.replay
orla run examples/score/scene.osn --window 480x240 --save voice.result=bar.wav
```

Notes written by a Kernel export the same way: `--save
notes.result=song.field`, then `orla-export song.field --out song.mid
--beats 100`.
