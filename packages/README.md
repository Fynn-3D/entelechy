# Packages

What several scenes share is a package, not an example: a directory
with a `package.osn` naming it and the least language and IR it needs,
a `kernels/` directory it `use`s, and `graphs/` presets a scene places
as nodes. `orla pack` carries one as a single `.orlapkg` file and
`orla unpack` opens it again; `orla package <dir>` prints what one
holds.

A scene reaches a package by path:

```text
use "../../packages/picture.orlapkg";
node display: graph "../../packages/picture.orlapkg/graphs/display.osn" { ... }
```

or by name, when `ORLA_PACKAGES` lists the directory it is in:

```sh
ORLA_PACKAGES=/path/to/orla/packages cargo run -- run scene.osn
```

```text
use "picture.orlapkg";
```

- **picture** — what a scene does to pictures that are not its subject.
  `wrap.orla` blends four texels found through `mod`, which `at_linear`
  cannot do at a Field's edge; `tonemap.orla` and `graphs/display.osn`
  are usually a scene's last node; `mip.orla` and `graphs/mip.osn` are
  the strip of halvings and the reads that pick a level, for a surface
  running away from the eye.
- **text** — a 5 by 7 font baked into `font.png`, the layout that turns
  a string into glyph rows, and the Kernels that draw them. A scene
  binds the font beside using the package:
  `input font: Grid<Float> = "../../packages/text.orlapkg/font.png";`
- **pixelfx** — a coarse pixel grid and a four-colour palette, and the
  NES look made of both as a preset. The example of what a package is.
