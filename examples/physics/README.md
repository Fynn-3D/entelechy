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

`tests/physics.replay` holds, on every backend, the pyramid standing,
the sensor touched, the ray stopped on the pyramid, and a click dropping
a ball that falls. The package's own tests
([`packages/physics.entpkg/tests`](../../packages/physics.entpkg/tests/))
hold resting, bouncing, sliding, rolling and carried bodies to their
closed forms, the pairs the grid finds, and piles that settle.

```sh
entelechy player examples/physics/scene.esn
entelechy replay examples/physics/tests/physics.replay --backend gpu
```
