# text

Text: a 5 by 7 bitmap font, glyph rows laid out and drawn, text decoded
into rows of letters, and the wide glyphs a project's words use drawn
into an atlas.

**Layer:** Interaction. **Uses:** [`ui`](../ui.entpkg/README.md).

```text
use "text.entpkg";
```

## Kernels

- `kernels/atlas.ent` — The wide glyphs a project's words use (Chinese,
  Japanese), as `entelechy import` reads them from a font of ten pixels
  (`--chars` the words; the study's `tools/fonts.py` imports and checks
  them), drawn into the atlas the ui package's letters read
  (`ui.letters`). Kernels: `text.glyph_atlas@1`, `text.glyph_codes@1`,
  `text.unlettered@1`.
- `kernels/font.ent` —  Kernels: `text.font@1`.
- `kernels/glyph.ent` — Text as rows, drawn by a Kernel, with no
  primitive. Kernels: `text.layout@1`, `text.digits@1`, `text.draw@1`,
  `text.label@1`.
- `kernels/ink.ent` — Text drawn by a Kernel's own code, pixel by pixel.
- `kernels/letters.ent` — Letters. Kernels: `text.no_codes@1`,
  `text.letters@1`, `text.rows@1`, `text.starts@1`, `text.lettered@1`,
  `text.either@1`.

## Graph presets

- `graphs/decode.esn` — A text's rows as letters, `width` cells a row.
- `graphs/letters.esn` — The letters' grid (`ui.letters`).

## Assets

`assets/font.png`

## Tests

`tests/` holds scenes and their replays, run on every backend:
`glyphs.replay`, `rows.replay`, `wide.replay`, `write.replay`. Every
Kernel of the package is reached by one of them.

Used by the [collapse](../../examples/collapse/README.md), [cube-puzzle](../../examples/cube-puzzle/README.md), [level](../../examples/level/README.md), [sound](../../examples/sound/README.md), [spells](../../examples/spells/README.md), [text](../../examples/text/README.md), [ui](../../examples/ui/README.md), [voxel](../../examples/voxel/README.md) examples.
