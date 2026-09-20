# Voxel

A small voxel editor, and the first example that is a tool rather than a
picture or a game: a `Volume<Int>` of colours edited one voxel at a time
under a minimal bar, with one level of undo. Nothing was added to the
language or the Host for it.

```text
seed ─┬─ state (feeds back; reads last frame's world) ─┬─ world (feeds back) ─ view
begin ┘                                                └─ backup (feeds back)
```

- `voxel.state` is one row: the camera's yaw, pitch, zoom and pan, the
  pointer's last position and buttons, the tool, the colour, the bar item
  under the pointer, the voxel under the pointer with the face seen,
  found by an Amanatides-Woo walk through last frame's volume, and the
  stroke in progress. The middle button orbits, with Shift held it pans
  by what the pointer covers at the model's distance, the wheel and the
  arrows zoom: Blender's hands. The left button applies the tool to the
  voxel under the pointer at once and again every fourth frame while
  held, so a stroke paints; the right button erases the same way
  whatever the tool. Z undoes.
- `voxel.edit` maps over the volume: the one voxel that is the target
  takes the colour, or empties, and every other keeps itself; an undo
  returns the backup; a reset returns the seed.
- `voxel.keep` is the backup: it takes the volume as it stood before
  each stroke, so undo restores it, and undo again restores the stroke.
- The camera, the walk, the palette and the solidity test are functions
  declared once at the top of the file; `cast` and `solid` take the
  volume as a parameter, so the state and the picture call the same
  code on whichever volume they read.
- `voxel.label` is the words: the item under the pointer, else the tool
  in hand, by name from a `text` input of the scene, and the voxel under
  the pointer as three numbers, as glyph rows the text library lays out
  and `voxel.draw` paints through `ink` at the top left, in logical
  pixels like the bar.
- `voxel.draw` walks the same volume per pixel with the same camera,
  lights faces by direction, darkens them where neighbours crowd the
  face, outlines the voxel under the pointer, draws a floor whose grid
  fades with distance with the two axes through the origin, a compass of
  the axes at the top right as the camera sees them, and the bar last:
  place, erase, paint, the colour (a click cycles it), undo, reset; and
  the label last.

The picture fills the window at the window's resolution (`domain =
window();`); the bar and the compass are laid out in logical pixels, with
the window's scale factor (`scale()`) as the unit, so they keep their
size on screen however the window is sized, and the replay declares a
240 by 135 window at a quarter of the scale: a sixteenth of the pixels
with the bar at a quarter size. It asserts on
the state row and the volume: the voxel under the pointer, a held left
button that places a voxel and then another, undo and redo of the
stroke, the middle button orbiting, Shift and the middle button panning,
the wheel, a held right button erasing, and the reset.

Saving and loading go through the protocol: `read` with the node returns
the volume's words, and `write` puts words back into `world.result`,
which is state, so the next frame carries on from them. The Player
speaks the protocol with `--serve`.

One frame at 960×540 over a 48³ volume is about 15 ms on the GPU
including the read back.

```sh
cargo run --release -p orla-player -- examples/voxel/scene.osn --scale 1
cargo run -- replay examples/voxel/tests/voxel.replay --backend gpu
```
