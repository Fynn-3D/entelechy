# Text

Text drawn by Kernels, with nothing added to the language or the Host: a
label is data, a font is a picture, and drawing is a Kernel. The scene
shows a title at twice the size, the whole font as a specimen, and the
pointer's x as digits that follow it.

```text
title (text) ─ layout ─┐
codes ─ layout ─┼─ draw ─ draw ─ draw ─ view
probe ─ digits ─ layout ─┘        (font.png feeds every draw)
```

- **The layout is a row.** `struct Glyph { code: Int; origin: Vec2;
  scale: Float; }` and `type Text = Rows<Glyph>;` in
  [the shared library](../shared/text.orla): a character code, the pixel
  where the glyph's cell begins, and its scale. Any Kernel, file, or
  protocol `write` can produce one, so a tool's labels are state like
  anything else.
- **The font is an atlas.** [`font.orla`](../shared/font.orla) holds a
  5x7 face for the printable ASCII codes as rows of bits and a Kernel that
  draws them into a `Grid<Float>` of 6x8 cells, sixteen per line; `bake.osn`
  and `run --save` wrote it as [`font.png`](../shared/font.png). A scene
  names it with `input font: Grid<Float> = "../shared/font.png";`, so the
  Host reads it and re-reads it when it changes: replace the file with
  another face in the same cells and every label changes.
- **Drawing is a Kernel.** `text.draw@1` takes a picture, a `Text`, and
  the atlas, and paints each pixel by the atlas texel under it, nearest,
  through the shared function `ink`, which any Kernel may call with its
  own rows and font. `text.layout@1` spaces a row of codes from an origin,
  optionally wrapping every `wrap` glyphs; `text.digits@1` writes a
  number as digit codes. A string is an input of the document, `input
  title: Rows<Int> = text "ORLA";`, the characters' codes as rows the
  session writes and rewrites as the document changes; a Kernel may also
  compute codes, as `codes` does for the specimen.

The replay pins pixels of the title, the specimen, and the readout before
and after the pointer moves, on both backends.

```sh
cargo run -- run examples/text/bake.osn --save font=examples/shared/font.png
cargo run --release -p orla-player -- examples/text/scene.osn --scale 4
cargo run -- replay examples/text/tests/text.replay
```
