# Animation

A skeleton, its clips and the mesh that follows, through the animation package, in four parts. Each part is a whole program that runs alone; `scene.esn` places
`parts/blend/`.

- [blend](parts/blend/README.md): an arm rigged and animated in Blender, imported as rows and posed by its own skeleton; the pointer blends two clips.
- [crowd](parts/crowd/README.md): sixty-four arms from one rig, each at its own moment of the clip.
- [aim](parts/aim/README.md): a shoulder aimed at a point while the elbow plays the clip.
- [stride](parts/stride/README.md): root motion: a clip whose root travels, played in place, carried through `<~`.

```sh
entelechy player examples/animation/scene.esn
entelechy player examples/animation/parts/blend/scene.esn
entelechy player examples/animation/parts/crowd/scene.esn
entelechy player examples/animation/parts/aim/scene.esn
entelechy player examples/animation/parts/stride/scene.esn
entelechy replay examples/animation/tests/*.replay --backend gpu
```
