# Text

Text drawn by Kernels: a label is data, a font is a picture, drawing is
a Kernel. The scene shows a title at twice the size, the whole font as
a specimen, and the pointer's x as digits.

```text
title (text) ─ layout ─┐
codes ─ layout ─┼─ draw ─ draw ─ draw ─ view
probe ─ digits ─ layout ─┘        (font.png feeds every draw)
```

- **The layout is a row.** `Text = Rows<Glyph>`, a code, origin and
  scale per glyph, in
  [the text package](../../packages/text.entpkg/kernels/text.ent).
- **The font is an atlas.** [`font.ent`](../../packages/text.entpkg/kernels/font.ent)
  holds a 5x7 ASCII face and draws it into 6x8 cells; `bake.esn` saves
  it as [`font.png`](../../packages/text.entpkg/font.png), which the
  scene reads as an input and reloads when it changes.
- **Drawing is a Kernel.** `text.draw@1` paints glyphs from the atlas
  through the shared function `ink`; `text.layout@1` spaces codes,
  optionally wrapping; `text.digits@1` and `text.label@1` turn numbers
  and captions into codes. The title is a `text` input of the scene.

The replay pins pixels of the title, the specimen, and the readout
before and after the pointer moves.
`tests/write.replay` pins what a Kernel draws with `write.ent`'s
functions: two codes of a `text` input, a negative number, a character
at twice the size, and the lengths a message is put together with.

```sh
entelechy run examples/text/bake.esn --save font=packages/text.entpkg/font.png
entelechy player examples/text/scene.esn --scale 4
entelechy replay examples/text/tests/text.replay examples/text/tests/write.replay
```
