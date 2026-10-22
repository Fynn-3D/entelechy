# voxel

Worlds of voxels drawn by walking rays through them: boxes of voxels
placed in the world, light, shapes as data, a view by day and bodies
posed by their joints.

**Layer:** Graphics. **Uses:** no other package.

```text
use "voxel.entpkg";
```

## Kernels

- `kernels/body.ent` — Bodies. Kernels: `voxel.unposing@1`,
  `voxel.posed@1`.
- `kernels/light.ent` — What a ray meets, lit.
- `kernels/ray.ent` — Worlds of voxels.
- `kernels/shape.ent` — Shapes as data, a sphere, a rounded cuboid or a
  capsule, each painting a number or carving, baked into a volume or
  painted over one. Kernels: `voxel.shapes@1`, `voxel.paint@1`.
- `kernels/view.ent` — The camera rays leave from, and one look of a
  volume by day. Kernels: `voxel.view@1`.

## Graph presets

- `graphs/view.esn` — A volume seen by day at the window's size, one
  look a scene may take or write its own of from the package's
  functions.

## Tested by

[crypt](../../examples/crypt/README.md), [golem](../../examples/golem/README.md), [sculpt](../../examples/sculpt/README.md), [voxel](../../examples/voxel/README.md), whose replays hold it on every backend.
