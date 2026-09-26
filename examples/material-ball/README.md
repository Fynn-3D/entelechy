# Material ball

A shaded sphere on a checker floor under an orbiting light. Both Kernels
are written inline in `scene.osn`:

```text
material.ball@1 -> image.tonemap@1 -> window
```

The ball declares `time.read`, so the light orbits; the tone map is pure
and is reused until its input changes. Edit `albedo`, `roughness`,
`metallic`, the `compute` block or `exposure` and save while the window
is open; a Kernel that does not compile reports a diagnostic and the last
good version keeps rendering.

```sh
orla player examples/material-ball/scene.osn
orla run examples/material-ball/scene.osn --out ball.ppm
```
