# Strings

Six plucked strings, played with the keys `1` to `6`. A string is a
delay line: a burst of noise goes in, and every time the head goes
round, each sample is averaged with the next and written back a little
quieter. What comes out is a note whose pitch is the line's length and
whose bright partials fade first, which is what a plucked string does.
Karplus and Strong published it in 1983; it is four lines of Kernel.

```text
block ─┐
quiet ─┼─ play (a scan over the block; the lines and heads are memories)
ready ─┘        ├─ line ─ draw (the strings as they stand)
                └─ result ─ the scene's sound
```

The delay lines and the heads are `memory` ports carried from frame to
frame with `<~`, so a note rings on across the blocks of samples the
scene plays: the block is 800 samples, a frame is a sixtieth of a
second, and a note lasts seconds. The scan walks the block sample by
sample, because a delay line is ordered work: what the head reads now
is what an earlier sample wrote.

A key that has just gone down fills its string with noise, at the first
sample of the block. Nothing reads what that loop gives, and the
optimizer keeps it anyway, because it holds writes.

The picture is the lines themselves, each drawn across the window at
its own length, so a short string and a long one both fill the width.

```sh
cargo run --release -p orla-player -- examples/strings/scene.osn
cargo run --release -- replay examples/strings/tests/pluck.replay
```

Every step is the same on the three backends: the replay pins the
samples and the lines to a ten-thousandth, and the interpreter, the JIT
and the GPU all give the same words.
