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

## Tested by

[crowd](../../examples/crowd/README.md), [golem](../../examples/golem/README.md), [reach](../../examples/reach/README.md), [skin](../../examples/skin/README.md), [stride](../../examples/stride/README.md), whose replays hold it on every backend.
