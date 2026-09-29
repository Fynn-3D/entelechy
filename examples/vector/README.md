# Vector

A drawing read from SVG and drawn by the
[vector package](../../packages/vector.orlapkg/kernels/vector.orla), sharp
at any size. `orla import` turns `source/garden.svg` into the four Fields
in `assets/`:

- the segments its curves are cut into;
- its shapes in paint order;
- its gradients' stops;
- its size.

`vector.draw@1` fills each pixel from them. The scene's `eye` Kernel
keeps where the eye is from frame to frame: a drag moves the drawing, and
the wheel zooms about the pointer.

The drawing uses the SVG this reads:

- a linear sky;
- a radial sun, with a ring of light whose hole is the even-odd rule;
- hills of Bézier curves;
- a dashed, round-capped path;
- petals turned about the flower's heart, with a gradient in each petal's
  own box and a stroke;
- leaves in a group drawn half clear.

The replay pins, on all three backends, the colour the SVG's rules give
at eight points (worked out by hand, not read back), the wheel's zoom
about the pointer and a drag.

```sh
orla import examples/vector/source/garden.svg --out examples/vector/assets
orla player examples/vector/scene.osn
orla replay examples/vector/tests/vector.replay --backend gpu
```
