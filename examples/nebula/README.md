# Nebula

A drift through a nebula made of noise, finished as film.

- **The noise package** gives the cloud and the stars. The cloud's thickness is a fractal sum of gradient noise carved into filaments by ridged noise, and another sum turns its colour round the wheel. The stars are the cells of a hash.
- **The film package** gives the colours and the finish. `nebula.look` keeps in w how far away each pixel's content is, and `graphs/finish.osn` gives the glow, the depth of field (focused where `nebula.lens` says), ACES tone, colour fringing, darkened corners and grain.

```sh
orla player examples/nebula/scene.osn --window 1280x720
orla replay examples/nebula/tests/nebula.replay examples/nebula/tests/noise.replay --backend gpu
```

`tests/noise.replay` holds the noise package to its promises on every backend:
- a hash of whole numbers is the same bits;
- value noise stays within -1 and 1;
- a Worley distance is never more than a cell's diagonal;
- curl noise neither gathers nor spreads.

`tests/nebula.replay` holds the finish to the window, the depth to the cloud or the night, and the lens to its focus.
