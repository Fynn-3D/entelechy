# animation

A skeleton, the clips that pose it and the mesh that follows: sampling
and blending clips, walking, aiming, and skinning by the joint palette.

**Layer:** Simulation. **Uses:** [`geometry`](../geometry.entpkg/README.md).

```text
use "animation.entpkg";
```

## Kernels

- `kernels/aim.ent` — Aiming. Kernels: `animation.aim@1`.
- `kernels/clip.ent` — Clips. Kernels: `animation.clock@1`,
  `animation.sample@1`, `animation.blend@1`, `animation.add@1`.
- `kernels/gait.ent` — Walking. Kernels: `animation.stride@1`,
  `animation.travel@1`.
- `kernels/skeleton.ent` — A skeleton, the clips that pose it, and the
  mesh that follows.
- `kernels/skin.ent` — Skinning. Kernels: `animation.palette@1`,
  `animation.apply@1`, `animation.place@1`.

## Graph presets

- `graphs/pose.esn` — One clip playing on one skeleton, as one node.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`aim.replay`, `clips.replay`. Every Kernel of the package is reached by
one of them.

Used by the [animation](../../examples/animation/README.md), [voxel](../../examples/voxel/README.md) examples.
