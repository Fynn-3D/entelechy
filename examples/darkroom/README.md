# Darkroom

A darkroom: a raw photograph developed with the image package.

- **The photograph** (`kernels/darkroom.orla`) is made of the noise package's noise: a dusk over hills and a lake that mirrors them. It is in linear light, flat and with its blacks lifted, as a raw picture is.
- **Developing it** takes the image package's Kernels and graphs, one node each:
  1. its histogram spread (`graphs/equalize.osn`, mixed in by a third);
  2. its colour graded (`image.adjust`: exposure, contrast, saturation and hue in Oklab);
  3. its detail sharpened (`graphs/sharpen.osn`);
  4. a neon outline round a turning mark, drawn from the mark's signed distance (`graphs/distance.osn`), screened over it (`image.blend`);
  5. its corners darkened (`image.blend`, multiply);
  6. encoded for the window (`image.encode`).

`view` shows a stage: 0 the raw photograph, 1 to 4 each step, 5 the finished picture. Change it while the Player runs.

```sh
orla player examples/darkroom/scene.osn --window 1280x720
orla replay examples/darkroom/tests/image.replay examples/darkroom/tests/darkroom.replay --backend gpu
```

`tests/image.replay` runs every part of the package on small pictures whose answers are worked by hand. It covers:
- colour round trips;
- the blend modes on known values;
- a blur's Gaussian weights;
- a shrink's mean;
- convolution;
- bicubic enlarging of a ramp;
- dilation and erosion of a dot;
- each tone change;
- the neutral LUT;
- equalization of two greys;
- the distance from a dot.

`tests/darkroom.replay` holds the scene to the window, the mark's distance and the histogram's bins.
