# signal

Fast Fourier transforms of a cube of complex numbers, `1 << bits` along
each side (at most 256), laid as rows, x fastest: one axis a pass, each
a group of lanes running every round of butterflies in place. The
inverse divides by the cube's count, so one undoes the other.

**Layer:** Foundation. **Uses:** no other package.

```text
use "signal.entpkg";
```

## Kernels

- `kernels/fourier.ent` — Fast Fourier transforms of a cube of complex
  numbers (`Vec2`, real then imaginary), `n = 1 << bits` a side, element
  `x + n * (y + n * z)`. Kernels: `signal.fourier_turns@1`,
  `signal.fourier_complex@1`, `signal.fourier_real@1`,
  `signal.fourier_rows@1`.

## Graph presets

- `graphs/fourier_forward.esn` — A cube's transform, along z, then y,
  then x, `1 << bits` a side (at most 256), laid as it came.
- `graphs/fourier_inverse.esn` — A spectrum's cube again, along z, then
  y, then x, each axis divided by `n`, so it undoes
  `fourier_forward.esn`.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`fourier.replay`. Every Kernel of the package is reached by one of them.
