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

## Tested by

[crowd](../../examples/crowd/README.md), [golem](../../examples/golem/README.md), [model](../../examples/model/README.md), [skin](../../examples/skin/README.md), [stride](../../examples/stride/README.md), whose replays hold it on every backend.
