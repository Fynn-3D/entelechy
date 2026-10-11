# Reach

The two-bone arm from [`skin`](../skin/README.md): the elbow plays the
clip's curl while the shoulder turns every frame toward a point (a slow
loop, or the pointer while held), drawn as a ring.

```text
clip ─ skin.sample ─ skin.aim ─ skin.palette ─ skin.apply ─ raster
                        │
            the point, the weight
```

[`skin.aim@1`](../../packages/skin.entpkg/kernels/skin.ent) turns the
joint's own axis (y, as in Blender) the shortest way onto the point, in
skin space, and lays the turn over the clip's pose; joints below follow.
Any joint may reach, each instance of a crowd toward its own point, and
`weight` blends clip and reach.

`tests/reach.replay` pins both joints' matrices at two moments on all
three backends; `tests/elbow.replay` aims a non-root joint and checks
that weight zero gives the clip's pose.

```sh
entelechy player examples/reach/scene.esn
entelechy replay examples/reach/tests/reach.replay
entelechy replay examples/reach/tests/elbow.replay
```
