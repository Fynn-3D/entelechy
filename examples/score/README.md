# Score

A MIDI file played and drawn. `orla-import` reads `song.mid` into rows
of notes, five numbers each (the second it starts, the seconds it
lasts, the key, how hard it was struck, and the channel), which the
scene takes as an input like any other Field:

```sh
cargo run -p orla-import -- examples/score/song.mid --out examples/score/assets
```

`voice` sums every note that sounds at each sample into the block the
sound card takes, and `roll` paints the same notes as a piano roll with
a playhead over them. Both loop over `count(song)`, the score's own
length, so a scene may hand them a song of any size. The melody is
channel 0 and the bass channel 1; they take different tones.

```text
song.mid ─ orla-import ─ song.notes.field ─┬─ voice (the sound)
                                           └─ roll (the picture)
```

```sh
cargo run --release -p orla-player -- examples/score/scene.osn
cargo run --release -- replay examples/score/tests/score.replay
cargo run --release -- run examples/score/scene.osn --window 480x240 --save voice.result=bar.wav
```

A Kernel that writes notes rather than reading them goes out the same
way: `--save notes.result=song.field` and then `orla-export
song.field --out song.mid --beats 100`, which a music editor opens.
