# crowd

Sixty-four arms out of one rig, each at its own moment in the clip.

One mesh, one skeleton, two clips, and a row per instance saying when it
is and where it stands. The [`skin` package](../../packages/skin.orlapkg)
does a crowd with the same Kernels that do one: a node sized `instances *
count(skeleton)` poses that many skeletons, and one sized `instances *
count(mesh)` moves that many meshes.

```text
node swinging: skin.sample@1 { ... domain = instances * count(skeleton); }
node posed:    skin.apply@1  { ... domain = instances * count(mesh); }
node standing: skin.place@1  { mesh <- posed.result; places <- places.result; one <- mesh; }
```

The rows this scene writes for itself are what a game writes: `times` is
a clock an instance, `places` is where each one stands and which way it
faces, and `mask` is a weight a joint.

## Three ways to put a pose together

- **Blended.** `skin.blend@1` takes the two clips; `mix` of one row
  blends the whole skeleton, and a row a joint is a mask. `--param
  upper=0` leaves the joint above the root on the first clip alone,
  which is the shape of an upper body doing one thing while the legs do
  another.
- **Laid over.** `skin.add@1` takes the difference between a clip and
  its own first frame and lays that over whatever the body is already
  doing. `--param lean=1` turns it on; it is how a flinch or a breath
  goes over a walk without replacing it.
- **Written.** A Kernel between the sample and the palette can write a
  joint outright, which is what the next thing to build here is.

## What it costs

At 1280x720 on the GPU, the least of five runs of thirty frames: 12.5 ms
a frame for 64 arms and 15.2 for 256. Four times the crowd for a fifth
more frame, because what grows is the vertices — 444 thousand at 64 and
1.8 million at 256 — and the poses are a rounding error beside them: two
joints an instance.

`tests/crowd.replay` pins sixteen of them on the interpreter, the JIT
and the GPU: the row counts, two instances' joint matrices (out of step
with each other), and what the mask and the additive layer each do to
one of them.
