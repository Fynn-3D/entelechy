# Modes

How a Kernel runs: as a map, a reduction, a Kernel with several outputs,
a scatter, and a node off the frame. Each is a part in `parts/`, a whole
scene of its own; `scene.osn` places the five as sub-graphs
(`node reduction: graph "parts/exposure/scene.osn" {}`) and
`modes.panels@1` draws their pictures in one window, three panels over
two.

```text
reduction (exposure)   maps (sort)          outputs (deferred)
scatter (particles)    off_frame (async)
```

```sh
orla player examples/modes/scene.osn
orla run examples/modes/scene.osn --window 640x400 --out modes.ppm
orla replay examples/modes/tests/*.replay
orla player examples/modes/parts/<part>/scene.osn
```

`tests/modes.replay` checks each panel of the whole and a value inside
each placed part, read under the node that places it
(`reduction.measure.result[0]`). The other replays test one part each.

## Reduction: exposure

`exposure.measure` runs `compute` once per pixel and folds the results
with `merge` into one `Rows<Light>` element holding the frame's luma
sum, maximum and count. `exposure.apply` reads it with `at(light, 0)`
and scales every pixel so the mean lands on `target`.

## Maps: sort

Sorting as two map Kernels: `sort.rank` counts, for each value, the
values below it (ties by index), and `sort.gather` picks the value of
each rank. Each is a bounded `loop` over `count(values)`, so the work is
quadratic. `sort.bars` draws the values before and after; they reshuffle
every half second. The [`sort` package](../../packages/sort.orlapkg) is
a radix sort, built from `prefix` and `scatter`.

## Several outputs: deferred

`deferred.surface@1` finds the surface under each pixel once and writes
a G-buffer by name, `albedo: Image`, `normal: Grid<Vec3>` and
`depth: Grid<Float>`:

```text
{ albedo: vec4(hit.albedo, 1.0), normal: hit.normal, depth: hit.t }
```

`light` samples the color and normal; `outline` reads the depth at each
pixel's neighbours by address and marks silhouette edges. The `turn`
parameter orbits the camera.

## Scatter: particles

A `Rows<Particle>` of structs advanced every frame by a map, splatted
onto a canvas, and drawn from it.

```text
seed (Rows) -> step (Rows, fed back) -> splat (Grid, onto fade) -> draw (Image)
clear (Grid) -> fade (Grid, fed back from splat) -^
```

`particles.splat@1` is a scatter: each particle adds its tint to the
canvas cell under it, one write per particle. `fade` dims last frame's
canvas, leaving trails; `draw` gathers a window of cells around each
pixel into a glow. Hold the mouse button to pull the particles toward
the pointer. `tests/particles.replay` follows one particle and pins its
cell and pixel.

## Off the frame: async

A Thumb machine (`parts/async/kernels/thumb.orla`) whose `step` node
says `async = true;`: an evaluation starts the step on its own thread
and continues, `view` shows the RAM the last landed step left, and
`clock` sweeps a hand over it by the frame's time. The first step runs
on the frame; a landed step is committed by the next evaluation, which
starts the next step.

```text
rom ─ boot (RAM) ──┬─ machine (async: on its own thread) ─ view ─ clock
start (registers) ─┘
```

`tests/async.replay` pins the count in RAM after the first step, the
same count while the second runs, and the landed count after `settle`.
The JIT runs its half a million instructions a step fastest:

```sh
orla run --backend jit --window 64x64 examples/modes/parts/async/scene.osn
```
