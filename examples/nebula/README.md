# Nebula

A drift through a nebula made of noise, finished as film.

- **The random package** gives the cloud and the stars. The cloud's thickness is a fractal sum of gradient noise carved into filaments by ridged noise, and another sum turns its color round the wheel. The stars are the cells of a hash.
- **The image package** gives the colors and the finish. `nebula.look` keeps in w how far away each pixel's content is, and `graphs/finish.esn` gives the glow, the depth of field (focused where `nebula.lens` says), ACES tone, color fringing, darkened corners and grain.

```sh
entelechy player examples/nebula/scene.esn --window 1280x720
entelechy replay examples/nebula/tests/nebula.replay --backend gpu
```

`tests/nebula.replay` holds the finish to the window, the depth to the cloud or the night, and the lens to its focus.
