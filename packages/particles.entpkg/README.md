# particles

Particles stepped on the device, binned into the screen's tiles and
drawn over a picture with its depth.

**Layer:** Graphics. **Uses:** [`geometry`](../geometry.entpkg/README.md), [`random`](../random.entpkg/README.md), [`render`](../render.entpkg/README.md).

```text
use "particles.entpkg";
```

## Kernels

- `kernels/system.ent` — Particles are rows of `Particle`, each frame
  made from the last (wired in a scene as `<~`). Kernels:
  `particles.none@1`, `particles.step@1`, `particles.bin@1`,
  `particles.draw@1`.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`fireflies.replay`. Every Kernel of the package is reached by one of
them.
