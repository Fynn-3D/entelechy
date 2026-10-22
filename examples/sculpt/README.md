# Sculpt

An island of 64 by 32 by 64 voxels made of shapes as data and sculpted
by hand, with the render package (`packages/render.voxel_entpkg`). Ten rows of
`VoxelShape` (rounded cuboids of sand, grass and earth, a sphere of
stone, a pond dug out and filled, two trees of a capsule and a ball) are
baked into a volume by `voxel.shapes@1`.

The left button digs a ball where the pointer points; the right raises
earth on the face pointed at. The brush picks against the world as last
drawn, through the same camera the view draws with
(`voxel_view_camera`, `voxel_view_box`, `voxel_cell`), and gives one
shape a frame; `voxel.paint@1`, fed back, paints it over the world. With
no button held the brush gives a shape holding nothing, the world is
unchanged, and what reads it does not run again. The package's
`graphs/view.esn` draws it by day: the sun's shadows, haze before the
sky.

```sh
entelechy-player examples/sculpt/scene.esn
entelechy-player examples/sculpt/scene.esn --param yaw=2.2
```

## Tests

- `look.replay`: the sky, the pond, grass in the sun and in a tree's
  shadow, the stone hill.
- `dig.replay`: the left button held over the hill digs its voxel out
  and the pixel looks into a shaded pit; the right button raises earth
  on the grass.
