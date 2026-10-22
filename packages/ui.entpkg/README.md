# ui

A screen's widgets drawn per pixel: layout, paint, widgets over a theme,
writing, navigation by keys, pad or pointer, a command search, and marks
over a 3D picture.

**Layer:** Interaction. **Uses:** [`text`](../text.entpkg/README.md).

```text
use "ui.entpkg";
```

## Kernels

- `kernels/command.ent` — Command search. Kernels: `ui.command_start@1`,
  `ui.command_newlines@1`, `ui.command_line_count@1`,
  `ui.command_no_lines@1`, `ui.command_line_ends@1`,
  `ui.command_query@1`, `ui.command_matched@1`, `ui.command_ranks@1`,
  `ui.command_ordered@1`, `ui.command_search@2`, `ui.command_overlay@1`.
- `kernels/layout.ent` — Rectangles a screen is laid out in, in its
  pixels, y down.
- `kernels/mark.ent` — Marks over a 3D picture.
- `kernels/navigation.ent` — Moving through a list of rows by keys, a
  pad or the pointer. Kernels: `ui.navigation_start@1`, `ui.navigate@1`.
- `kernels/paint.ent` — What a pixel of a screen is painted.
- `kernels/widget.ent` — The widgets.
- `kernels/writing.ent` — Text drawn.

## Graph presets

- `graphs/command_search.esn` — The command search (`ui.command_search`)
  over `names`, a command a line.

## Tested by

[collapse](../../examples/collapse/README.md), [cube-puzzle](../../examples/cube-puzzle/README.md), [level](../../examples/level/README.md), [sound](../../examples/sound/README.md), [spells](../../examples/spells/README.md), [ui](../../examples/ui/README.md), [voxel](../../examples/voxel/README.md), whose replays hold it on every backend.
