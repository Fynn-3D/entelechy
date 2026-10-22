# Darkroom

A darkroom: a raw photograph developed with the image package.

- **The photograph** (`kernels/darkroom.ent`) is made of the math package's noise: a dusk over hills and a lake that mirrors them. It is in linear light, flat and with its blacks lifted, as a raw picture is.
- **Developing it** takes the image package's Kernels and graphs, one node each:
  1. its histogram spread (`graphs/equalize.esn`, mixed in by a third);
  2. its color graded (`image.adjust`: exposure, contrast, saturation and hue in Oklab);
  3. its detail sharpened (`graphs/sharpen.esn`);
  4. a neon outline round a turning mark, drawn from the mark's signed distance (`graphs/distance.esn`), screened over it (`image.blend`);
  5. its corners darkened (`image.blend`, multiply);
  6. encoded for the window (`image.encode`).

`view` shows a stage: 0 the raw photograph, 1 to 4 each step, 5 the finished picture. Change it while the Player runs.

```sh
entelechy player examples/darkroom/scene.esn --window 1280x720
entelechy replay examples/darkroom/tests/darkroom.replay --backend gpu
```

Every part of the image package is held on small pictures whose answers
are worked by hand by the package's own tests
([`packages/image.entpkg/tests`](../../packages/image.entpkg/tests/)).

`tests/darkroom.replay` holds the scene to the window, the mark's distance and the histogram's bins.
