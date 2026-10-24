# text

Text: glyph rows laid out and drawn from a program's font atlas, text
decoded into rows of letters, the wide glyphs a project's words use drawn into an
atlas, numbers written with their decimals, a template's holes filled
with numbers, and text written to a file through an output `to files`.

**Layer:** Interaction. **Uses:** [`parallel`](../parallel.entpkg/README.md).

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
- `kernels/glyph.ent` — Text as rows, drawn by a Kernel, with no
  primitive. Kernels: `text.layout@1`, `text.digits@1`, `text.draw@1`,
  `text.label@1`.
- `kernels/ink.ent` — Text drawn by a Kernel's own code, pixel by pixel.
- `kernels/number.ent` — A number written with decimals: how many
  characters it takes and each one's code (`number_width`,
  `number_code`).
- `kernels/letters.ent` — Letters. Kernels: `text.no_codes@1`,
  `text.letters@1`, `text.rows@1`, `text.starts@1`, `text.lettered@1`,
  `text.either@1`.
- `kernels/template.ent` — A template's holes `$0` to `$9` filled with the
  values they name. Kernels: `text.template_widths@1`,
  `text.template_filled@1`.
- `kernels/writing.ent` — The operation an output `to files` takes to
  write a text at a path. Kernels: `text.writing@1`.

## Graph presets

- `graphs/decode.esn` — A text's rows as letters, `width` cells a row.
- `graphs/letters.esn` — The letters' grid (`ui.letters`).
- `graphs/template.esn` — A template filled and written: a world writing
  its own law, a report, a save.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`glyphs.replay`, `rows.replay`, `template.replay`, `wide.replay`, `write.replay`. Every
Kernel of the package is reached by one of them.

Used by the [collapse](../../examples/collapse/README.md), [cube-puzzle](../../examples/cube-puzzle/README.md), [level](../../examples/level/README.md), [sound](../../examples/sound/README.md), [spells](../../examples/spells/README.md), [text](../../examples/text/README.md), [ui](../../examples/ui/README.md), [voxel](../../examples/voxel/README.md) examples.
