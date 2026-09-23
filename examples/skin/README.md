# skin

An arm rigged in Blender, animated in Blender, posed in Orla.

Two bones, a box weighted to them, and two actions. `orla import` wrote
five rows out of the glTF and nothing here is baked: the joints, what
each vertex is bound to, and one clip per action, sampled at thirty
frames a second.

```sh
blender -b --python bridges/blender/make_rig.py -- --out examples/skin/source/arm.glb
orla import examples/skin/source/arm.glb --out examples/skin/assets
```

The posing is the [`skin` package](../../packages/skin.orlapkg): a clip
sampled at the moment the clock says, the two clips blended, each
joint's transform composed up its parent chain and taken through its
inverse bind, and the mesh moved by the four joints it hangs off. Every
one of those is a Kernel over a Field, which is the point — a pose is
data the whole way, so this scene blends two clips by where the pointer
is, and a scene that wanted to drive a joint from a physics step would
write it between the sample and the palette.

Hold the pointer down and move it up and down to blend the swing into
the curl; let go and it sweeps on its own.

`tests/skin.replay` pins the second joint's matrix at rest (the
identity) and a third of a second in (a turn of about seventy-seven
degrees, and the move that goes with it), with three pixels either side,
on the interpreter, the JIT and the GPU.
