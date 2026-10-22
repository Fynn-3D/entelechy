# Aim

The two-bone arm from [`skin`](../blend/README.md): the elbow plays the
clip's curl while the shoulder turns every frame toward a point (a slow
loop, or the pointer while held), drawn as a ring.

```text
clip ─ skin.sample ─ skin.aim ─ skin.palette ─ skin.apply ─ raster
                        │
            the point, the weight
```

[`animation.aim@1`](../../../../packages/animation.entpkg/kernels/skin.ent) turns the
joint's own axis (y, as in Blender) the shortest way onto the point, in
skin space, and lays the turn over the clip's pose; joints below follow.
Any joint may reach, each instance of a crowd toward its own point, and
`weight` blends clip and reach.

`../../tests/aim.replay` pins both joints' matrices at two moments on all
three backends; the animation package's own tests
([`packages/animation.entpkg/tests`](../../../../packages/animation.entpkg/tests/))
aim a non-root joint and check that weight zero gives the clip's pose.

```sh
entelechy player examples/animation/parts/aim/scene.esn
entelechy replay examples/animation/tests/aim.replay
```
