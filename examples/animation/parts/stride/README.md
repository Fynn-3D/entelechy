# Stride

Root motion: a clip whose root travels, played in place, with the
travel carried by the scene. The arm from [`skin`](../blend/README.md)
crawls like an inchworm on a clip made here by `stride.clip@1`, whose
last frame is its first moved on by the stride (see
[the rig contract](../../../../docs/RIGS.md)).

```text
clip ─ skin.stride ─ skin.palette ─ skin.apply ─ skin.place ─ raster
   └── skin.travel ─ stand ◀─┐ (<~ fed back)
          ▲                  └───────────────
   clock, and the clock as it was last evaluation (<~)
```

- [`animation.stride@1`](../../../../packages/animation.entpkg/kernels/skin.ent) loops
  the clip with the root's ground travel removed.
- `animation.travel@1` gives the root's travel between the previous clock
  and now, loops counted.
- `stand` accumulates it through `<~`; `animation.place@1` puts the posed arm
  there. The position is independent of how often the scene is
  evaluated.

The camera follows the body over a striped floor. `tests/stride.replay`
pins the body's position with sparse and dense frames, the in-place
root, and both joints' matrices, on the interpreter, the JIT and the
GPU.
