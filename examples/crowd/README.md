# Crowd

Sixty-four arms from one rig, each at its own moment in the clip. The
[`animation` package](../../packages/animation.entpkg) Kernels pose a crowd by
their `domain`: `instances * count(skeleton)` poses the skeletons and
`instances * count(mesh)` moves the meshes.

```text
node swinging: animation.sample@1 { ... domain = instances * count(skeleton); }
node posed:    animation.apply@1  { ... domain = instances * count(mesh); }
node standing: animation.place@1  { mesh <- posed.result; places <- places.result; one <- mesh; }
```

`times` is a clock per instance, `places` where each stands and faces,
and `mask` a weight per joint.

- `animation.blend@1` blends the two clips; `--param upper=0` keeps the joint
  above the root on the first clip.
- `animation.add@1` lays a clip's difference from its first frame over the
  pose; `--param lean=1` turns it on.

`tests/crowd.replay` pins row counts, two instances' joint matrices, and
the effect of the mask and the additive layer, on the interpreter, the
JIT and the GPU.
