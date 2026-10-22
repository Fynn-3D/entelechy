# Strings

Six plucked strings (Karplus-Strong), played with keys `1` to `6`. Each
string is a delay line: a key fills it with noise, and each pass of the
head averages neighboring samples and writes them back slightly
quieter, so the pitch is the line's length and the high partials fade
first.

```text
block ─┐
quiet ─┼─ play (a scan over the block; the lines and heads are memories)
ready ─┘        ├─ line ─ draw (the strings as they stand)
                └─ result ─ the scene's sound
```

The lines and heads are `memory` ports carried across frames with `<~`,
so a note rings across sound blocks. `play` is a `scan` because a delay
line is ordered: the head reads what an earlier sample wrote. `draw`
stretches each line across the window.

```sh
entelechy player examples/sound/parts/strings/scene.esn
entelechy replay examples/sound/tests/strings.replay
```

The replay pins samples and lines, and the interpreter, the JIT and the
GPU give the same words.
