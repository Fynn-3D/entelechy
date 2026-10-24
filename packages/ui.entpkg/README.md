# ui

A screen drawn per pixel: rectangles laid out, shapes painted, text
written, and navigation through a list by the hands `controls` reads.
What a program's widgets look like is its own.

**Layer:** Interaction. **Uses:** [`text`](../text.entpkg/README.md), [`controls`](../controls.entpkg/README.md).

```text
use "ui.entpkg";
```

## Kernels

- `kernels/layout.ent` — Rectangles a screen is laid out in, in its
  pixels, y down.
- `kernels/navigation.ent` — Moving through a list of rows by a
  person's hands as `controls` reads them. Kernels:
  `ui.navigation_start@1`, `ui.navigation_keys@1`, `ui.navigate@2`.
- `kernels/paint.ent` — What a pixel of a screen is painted.
- `kernels/writing.ent` — Text drawn.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`navigation.replay`. Every Kernel of
the package is reached by one of them.

Used by the [collapse](../../examples/collapse/README.md), [cube-puzzle](../../examples/cube-puzzle/README.md), [level](../../examples/level/README.md), [sound](../../examples/sound/README.md), [spells](../../examples/spells/README.md), [ui](../../examples/ui/README.md), [voxel](../../examples/voxel/README.md) examples.
