# UI

A settings screen and a hero's panel drawn with the
[ui package](../../packages/ui.entpkg/package.esn). The options are a
difficulty, a volume, subtitles, hints and a button to apply them; the
hero's panel holds two meters, the health falling every few seconds with
its trail after it, a purse and a journal scrolled a line a second.

The arrows or W A S D move through the options and change them, Enter or
Space confirms, Esc goes back; a pad's left stick, south and east do the
same; the pointer lights the row it is over, and a click confirms it or
sets the volume where it falls on the slider. The words are
`assets/ui.txt`, a row a line.

```sh
entelechy player examples/ui/scene.esn
entelechy replay examples/ui/tests/*.replay --backend gpu
```

`tests/ui.replay` checks, on all three backends, each option changed by
the keys, a row held repeating after its delay, the pointer lighting a
row and a click setting the volume, and the slider's fill and a key's
cap by their pixels. `tests/wide.replay` draws a line with a wide letter
from an atlas its scene draws itself, a box, and a character the atlas
does not hold as a `?`.
