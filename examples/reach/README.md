# reach

An arm whose shoulder is computed and whose elbow is authored.

The rig is the one [`skin`](../skin/README.md) imported from Blender: two
bones, a box weighted to them, and a curl. Here the curl plays in the
elbow while the shoulder turns every frame to face a point — a slow loop
in front of the arm, or the pointer while it is held down — drawn as a
ring so what the joint is doing shows.

The turn is [`skin.aim@1`](../../packages/skin.orlapkg/kernels/skin.orla),
a Kernel between the sample and the palette:

```text
clip ─ skin.sample ─ skin.aim ─ skin.palette ─ skin.apply ─ raster
                        │
            the point, the weight
```

It turns the joint's own axis (a Blender bone runs along its y) the
shortest way onto the point and lays that turn over the pose the clip
gave it, so the clip's twist stays and the joints below follow. The
point is in the skin's space; the Kernel works out the joint's parent
space up the chain, so any joint may reach, and a crowd sized
`instances * joints` has each instance reach for its own point.
`weight` blends between the clip and the reach.

Nothing in the runtime knows what a joint is: the reach is a Kernel over
the same rows the rest of the package reads.

`tests/reach.replay` pins both joints' matrices a third of a second in
and two seconds in, with the ring and the arm in the picture, on the
interpreter, the JIT and the GPU. `tests/elbow.replay` holds the reach
on a joint that is not the root: the elbow faces a point exactly, worked
out from the palette alone, at two moments of the clip, and with the
weight at zero the pose is the clip's own.
