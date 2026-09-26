# Million

A million bodies, stepped and drawn every frame. Each has a place, a
speed and a tint in a `Rows<Body>` carried through `<~`, and is pulled
towards the middle by an inverse square; a body that leaves the picture
respawns on the disc.

```text
spawn ─ bodies (a map)
          └─ splat (a scatter: one Int count per body)
               └─ draw (the counts, as their logarithm)
```

The step reads no other body, so it is a map. The scatter's counts are
`Int`, which add the same in any order, so the GPU sums them with one
atomic each and every backend gives the same words.

```sh
orla bench examples/million/scene.osn --window 960x540 --backend all --nodes
orla player examples/million/scene.osn
orla run examples/million/scene.osn --param count=4000000 --window 960x540 --backend gpu --out four.ppm
```
