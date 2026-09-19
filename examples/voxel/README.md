# Voxel

A small voxel editor, and the first example that is a tool rather than a
picture or a game: a `Volume<Int>` of colours edited one voxel at a time
under a minimal bar, with one level of undo. Nothing was added to the
language or the Host for it.

```text
seed ─┬─ state (feeds back; reads last frame's world) ─┬─ world (feeds back) ─ view
begin ┘                                                └─ backup (feeds back)
```

- `voxel.state` is one row: the camera's yaw, pitch and zoom, the
  pointer's last position and button, the tool, the colour, the bar item
  under the pointer, and the voxel under the pointer with the face seen,
  found by an Amanatides-Woo walk through last frame's volume. A press
  on the picture starts a drag that orbits; a release that moved less
  than three pixels is a click, which applies the tool; the right button
  erases whatever the tool. The wheel and the arrows zoom; Z undoes.
- `voxel.edit` maps over the volume: the one voxel that is the target
  takes the colour, or empties, and every other keeps itself; an undo
  returns the backup; a reset returns the seed.
- `voxel.keep` is the backup: it takes the volume as it stood before
  each change, so undo restores it, and undo again restores the change.
- `voxel.draw` walks the same volume per pixel with the same camera,
  lights faces by direction, darkens them where neighbours crowd the
  face, outlines the voxel under the pointer, draws a floor, and the bar
  last: place, erase, paint, the colour (a click cycles it), undo, reset.

The bar's geometry is the frame size, which the Kernels take as
parameters, so the replay runs at a quarter of the pixels and asserts on
the state row and the volume: the voxel under the pointer, a placed
voxel at its index, undo and redo, an erase, a painted voxel, a right
click that erases it, the wheel, and a drag that orbits without editing.

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
