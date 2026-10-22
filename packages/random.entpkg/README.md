# random

Randomness a Kernel can repeat: hashes of whole numbers, the same bits
on every backend, and the value, gradient, fractal, cellular and curl
noise built on them.

**Layer:** Foundation. **Uses:** no other package.

```text
use "random.entpkg";
```

## Kernels

- `kernels/cellular.ent` — Worley's cells.
- `kernels/curl.ent` — Curl noise.
- `kernels/hash.ent` — Hashes of whole numbers, by integer arithmetic,
  so a point's hash is the same bits on every backend, read as a number
  from 0 to 1 or a whole number below a bound.
- `kernels/noise.ent` — Value and gradient noise in two and three
  dimensions, interpolated between hashed corners, and their fractal
  sums.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`noise.replay`. Every Kernel of the package is reached by one of them.

Used by the [animation](../../examples/animation/README.md), [darkroom](../../examples/darkroom/README.md), [nebula](../../examples/nebula/README.md) examples.
