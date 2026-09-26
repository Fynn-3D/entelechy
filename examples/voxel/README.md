# Voxel

A voxel editor: a `Volume<Int>` of colours edited one voxel at a time
under a toolbar, with one level of undo.

```text
seed ─┬─ state (feeds back; reads last frame's world) ─┬─ world (feeds back) ─ view
begin ┘                                                └─ backup (feeds back)
```

- `voxel.state`: one row of camera, pointer, tool, colour, hovered
  item, the voxel under the pointer (an Amanatides-Woo walk through last
  frame's volume) and the stroke in progress. Middle button orbits,
  Shift+middle pans, wheel and arrows zoom; left applies the tool, right
  erases, both repeating while held; Z undoes.
- `voxel.edit`: maps over the volume, changing the target voxel, or
  returning the backup on undo and the seed on reset.
- `voxel.keep`: the backup, the volume as it stood before each stroke.
- `voxel.label`: the hovered item or tool name and the voxel's
  coordinates, as glyph rows from the text package.
- `voxel.draw`: the volume per pixel with the same camera and walk,
  shaded faces, the hovered voxel outlined, a floor grid, an axis
  compass, the toolbar (place, erase, paint, colour, undo, reset), and
  the label.

The picture fills `window()`; the toolbar and compass use logical pixels
(`scale()`). The replay, in a 240 by 135 window, asserts on the state
row and the volume through placing, undo and redo, orbit, pan, zoom,
erase, and reset.

Saving and loading use the protocol: `read` the node for the volume's
words, `write` words into `world.result`. The Player speaks the
protocol with `--serve`.

```sh
cargo run --release -p orla-player -- examples/voxel/scene.osn --scale 1
cargo run -- replay examples/voxel/tests/voxel.replay --backend gpu
```
