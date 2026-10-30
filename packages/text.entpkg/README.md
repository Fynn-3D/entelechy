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
  Japanese) drawn into the atlas the ui package's letters read
  (`ui.letters`): a font node's glyphs at any side and density
  (`text.face_atlas`, its ascent from the font's metrics), or a ten-pixel font's as
  `entelechy import` reads them (`text.glyph_atlas`). Kernels:
  `text.glyph_atlas@1`, `text.face_atlas@1`, `text.glyph_codes@1`,
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
- `kernels/wanted.ent` — The wide characters a text uses, each once, in
  order of code: the codes a font node reads. Kernels:
  `text.char_codes@1`, `text.first_codes@1`, `text.code_list@1`.
- `kernels/template.ent` — A template's holes `$0` to `$9` filled with the
  values they name. Kernels: `text.template_widths@1`,
  `text.template_filled@1`.
- `kernels/writing.ent` — The operation an output `to files` takes to
  write a text at a path. Kernels: `text.writing@1`.

## Graph presets

- `graphs/decode.esn` — A text's rows as letters, `width` cells a row.
- `graphs/face.esn` — A font's glyphs as the letters' grid, `side` texels
  square at `density` pixels a texel (a denser glyph stands as large,
  drawn finer and read blended), and the codes the atlas holds: a text's
  font is a font node's path, a side and a density.
- `graphs/letters.esn` — The letters' grid (`ui.letters`).
- `graphs/wanted.esn` — The wide characters a text uses, for a font node.
- `graphs/template.esn` — A template filled and written: a world writing
  its own law, a report, a save.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`face.replay`, `glyphs.replay`, `rows.replay`, `template.replay`, `wide.replay`, `write.replay`. Every
Kernel of the package is reached by one of them.

Used by the [collapse](../../examples/collapse/README.md), [cube-puzzle](../../examples/cube-puzzle/README.md), [level](../../examples/level/README.md), [sound](../../examples/sound/README.md), [spells](../../examples/spells/README.md), [text](../../examples/text/README.md), [ui](../../examples/ui/README.md), [voxel](../../examples/voxel/README.md) examples.
