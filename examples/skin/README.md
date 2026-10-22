# skin

An arm rigged and animated in Blender, posed in Entelechy. `entelechy import`
reads the glTF into rows: joints, vertex bindings, and one clip per
action sampled at 30 frames a second.

```sh
blender -b --python bridges/blender/make_rig.py -- --out examples/skin/source/arm.glb
entelechy import examples/skin/source/arm.glb --out examples/skin/assets
```

The [`animation` package](../../packages/animation.entpkg) does the posing, each
step a Kernel over a Field: sample each clip at the clock, blend the
two, compose joint transforms up the parent chain through the inverse
bind, and move each vertex by the four joints it is bound to. Held, the
pointer's height sets the blend; released, the arm sweeps on its own.

`tests/skin.replay` pins the second joint's matrix at rest and at a
third of a second, on the interpreter, the JIT and the GPU.
