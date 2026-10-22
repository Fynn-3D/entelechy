# Physics

Boxes and balls stepped by the
[physics package](../../packages/physics.entpkg/kernels/solve.ent):
a pyramid of thirty-six boxes, balls rolling down a ramp, a sensor on the
floor that lights while something lies on it, and a ray from the upper
left corner toward the pointer that stops at the first body it meets. A
click drops a heavy ball where the pointer is.

The world is rows of `Body`, carried from frame to frame through `<~`
with the contacts of the last step, which the next one starts from. The
scene's own Kernels in `kernels/world.ent` make the bodies, drop the
pointer's balls into slots kept apart until used, and draw: the picture
is cut into tiles, each listing the bodies that reach it with the
`parallel` package's runs, so a pixel walks its tile's few bodies and a
larger window costs its pixels, not its pixels times every body.
`--param steps` steps more than one sixtieth of a second a frame.

The replays, on all three backends:

- `rest.replay`: a ball rests on the ground, a column of five boxes
  stands still for seconds, and rays meet the tops of both.
- `motion.replay`: a bounce, a box held by friction on a slope and one
  sliding down it, and a box carried by a kinematic platform, each to
  its closed form.
- `many.replay`: the pairs the grid finds among four hundred circles
  are every pair asked one by one, and none gets out of the box they are
  poured into.
- `pile.replay`: three hundred circles poured into a narrow box settle
  some twenty deep in four substeps, none pushed out, none overlapping
  another by a tenth of a unit.
- `physics.replay`: the pyramid stands, the sensor is touched, the ray
  stops on the pyramid, and a click drops a ball that falls.

```sh
entelechy player examples/physics/scene.esn
entelechy replay examples/physics/tests/*.replay --backend gpu
```
