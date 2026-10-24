# voxel

Worlds of voxels drawn by walking rays through them: boxes of voxels
placed in the world, a place shaded from the sun, shapes as data, a view
by day and bodies posed by their joints. A view is seen through
`geometry`'s `Lens` and lit by `light`'s words for a game without a
renderer (`plain_daylight`, `plain_light`, `flame_flicker`,
`light_stepped`, `plain_haze`, `plain_dark`).

**Layer:** Graphics. **Uses:** [`geometry`](../geometry.entpkg/README.md), [`light`](../light.entpkg/README.md).

```text
use "voxel.entpkg";
```

## Kernels

- `kernels/body.ent` — Bodies. Kernels: `voxel.unposing@1`,
  `voxel.posed@1`.
- `kernels/ray.ent` — Worlds of voxels.
- `kernels/shade.ent` — A place shaded from the sun: a walk toward it
  that meets a voxel (`voxel_shaded`).
- `kernels/shape.ent` — Shapes as data, a sphere, a rounded cuboid or a
  capsule, each painting a number or carving, baked into a volume or
  painted over one. Kernels: `voxel.shapes@1`, `voxel.paint@1`.
- `kernels/view.ent` — One look of a volume by day: the lens going round
  its middle (`voxel_view_camera`), the box it stands in. Kernels:
  `voxel.view@1`.

## Graph presets

- `graphs/view.esn` — A volume seen by day at the window's size, one
  look a scene may take or write its own of from the package's
  functions.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`world.replay`. Every Kernel of the package is reached by one of them.

Used by the [voxel](../../examples/voxel/README.md) example.
