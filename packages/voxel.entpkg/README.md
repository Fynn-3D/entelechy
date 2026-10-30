# voxel

Worlds of voxels drawn by walking rays through them: boxes of voxels
placed in the world, a place shaded from the sun, shapes as data, and
bodies posed by their joints.

**Layer:** Graphics. **Uses:** no other package.

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

## Tests

`tests/` holds scenes and their replays, run on every backend:
`world.replay`. Every Kernel of the package is reached by one of them.

Used by the [voxel](../../examples/voxel/README.md) example.
