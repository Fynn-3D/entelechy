# Material balls

Rows of balls on a checkered floor, for looking at materials under a
studio's sun and sky. The first row is a gray ball of 18%, a car's paint
under its clearcoat, velvet's sheen, wax lit through by subsurface
scattering, brushed steel's anisotropic highlight, a glowing ball and a
chrome mirror; then a red plastic and gold, each from mirror-smooth to
fully rough; then clear things (glass, frosted glass, bottle glass,
water, ice, diamond), refracted in and out on each sphere's exact chord.

The scene owns geometry and materials alone (`graphs/balls.esn`): it
hands over the lens, the sky, a G-buffer and proxies as the
[render package](../../packages/render.entpkg/README.md)'s
contract asks, and the renderer placed after it owns the light. A
different renderer is a different `light` node:

- `scene.esn`: the [render package](../../packages/render.entpkg/package.esn)'s light probes;
- `tests/traced.esn`: the [render package](../../packages/render.entpkg/package.esn)'s path tracer;
- `tests/bench.esn`: the [render package](../../packages/render.entpkg/package.esn)'s view.

The left button drags to turn, the right or middle to pan, the wheel
dollies, a double-click or F flies to the ball under the pointer, W A S
D Q E walk (Shift fast), R goes home.

```sh
entelechy player examples/material_balls/scene.esn
entelechy replay examples/material_balls/tests*.replay --backend gpu
```

`tests/balls.replay`, `tests/traced.replay` and `tests/bench.replay`
hold each renderer's picture of the same balls to its pixels on every
backend.
