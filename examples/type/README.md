# Type

Text in two fonts, drawn by the
[font package](../../packages/font.orlapkg/kernels/font.orla) and the
[vector package](../../packages/vector.orlapkg/kernels/vector.orla): a
title, four lines in a Latin font joined by new lines, and a line in a
Chinese font. Each character is its font's glyph, placed at the pen and
moving it by its advance, and drawn from its outline, so it is sharp at
any size.

`orla import` reads the fonts in `source/`:

- `NotoSans-Regular.ttf`, whose printable ASCII characters are in
  `assets/`;
- a Chinese font, NotoSansSC-Regular, of which only the characters of
  the line drawn are in `assets/`, the whole font being too large to
  keep here.

Both are under the SIL Open Font License, in `source/`.

```sh
orla import examples/type/source/NotoSans-Regular.ttf --out examples/type/assets --chars " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_\`abcdefghijklmnopqrstuvwxyz{|}~"
orla import NotoSansSC-Regular.otf --out examples/type/assets --chars "宿主是世界，内核是法则，图是结构，智能是创造者。"
orla player examples/type/scene.osn
orla replay examples/type/tests/*.replay --backend gpu
```

`tests/type.replay` sets large type and checks, on all three backends,
where the pen puts each character and pixels worked out from the glyphs'
boxes: ink in the middle of a stem, paper in the hole of an `o` and of a
`。`, and a character the font lacks drawn as its missing glyph.
`tests/lines.replay` checks the lines joined and each started a line's
height below the last.
