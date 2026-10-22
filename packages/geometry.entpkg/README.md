# geometry

Rotations as quaternions and rigid transforms as matrices: turning,
composing, interpolating and moving points and directions.

**Layer:** Foundation. **Uses:** no other package.

```text
use "geometry.entpkg";
```

## Kernels

- `kernels/quaternion.ent` — Rotations as unit quaternions, held in a
  `Vec4` as `x y z w`, the order glTF and Blender write.
- `kernels/transform.ent` — Rigid transforms.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`turns.replay`. Every Kernel of the package is reached by one of them.

Used by the [animation](../../examples/animation/README.md), [mesh](../../examples/mesh/README.md), [voxel](../../examples/voxel/README.md) examples.
