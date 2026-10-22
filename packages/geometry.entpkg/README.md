# geometry

Rotations as quaternions and rigid transforms as matrices; a camera's
matrices and the ways between the world, clip space and the picture; and
signed distances to shapes, with the ways they are joined.

**Layer:** Foundation. **Uses:** no other package.

```text
use "geometry.entpkg";
```

## Kernels

- `kernels/camera.ent` — A camera's matrices and the ways between the
  world, the clip space a rasterizer takes and the picture.
- `kernels/quaternion.ent` — Rotations as unit quaternions, held in a
  `Vec4` as `x y z w`, the order glTF and Blender write.
- `kernels/sdf.ent` — Signed distance functions.
- `kernels/transform.ent` — Rigid transforms.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`camera.replay`, `distances.replay`, `turns.replay`. Every Kernel of the
package is reached by one of them.

Used by the [animation](../../examples/animation/README.md), [culled](../../examples/culled/README.md), [image](../../examples/image/README.md), [mesh](../../examples/mesh/README.md), [showcase](../../examples/showcase/README.md), [surface](../../examples/surface/README.md), [voxel](../../examples/voxel/README.md) examples.
