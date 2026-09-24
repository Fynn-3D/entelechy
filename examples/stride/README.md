# stride

Root motion: a clip whose root travels, played in place, and the travel
carried by the scene.

The arm from [`skin`](../skin/README.md) crawls like an inchworm: a clip
made by a Kernel here (`stride.clip@1`) curls the forearm, rocks and
bobs the shoulder, and moves the shoulder on 1.2 along -z a loop. Its
last frame is its first moved on by that stride, which is what makes a
clip travel (see [the rig contract](../../docs/RIGS.md)).

Played with `skin.sample@1`, the arm would slide forward and snap back
each loop. Instead:

```text
clip ─ skin.stride ─ skin.palette ─ skin.apply ─ skin.place ─ raster
   └── skin.travel ─ stand ◀─┐ (<~ fed back)
          ▲                  └───────────────
   clock, and the clock as it was last evaluation (<~)
```

- [`skin.stride@1`](../../packages/skin.orlapkg/kernels/skin.orla) plays
  the clip looping every `frames - 1` frames, with the root's travel
  along the ground taken out; the bob and the rock stay in the pose.
- `skin.travel@1` says how far the root travelled between the clock
  now and the clock at the last evaluation, loops counted.
- `stand` adds that to where the body stood, which the scene keeps
  through `<~`, and `skin.place@1` puts the posed arm there.

The travel is a difference of where the root had got to at two moments,
so what `stand` adds up is where the clip says the body is, however
often the scene is evaluated. The camera follows the body over a floor
striped across the way it goes, so the crawl shows.

`tests/stride.replay` pins where the body stands (three units on after
two and a half loops) with frames far apart and again with fifty of
them, the root in place with only its bob, and both joints' matrices at
two moments, on the interpreter, the JIT and the GPU.
