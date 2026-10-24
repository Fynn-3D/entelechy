# signal

Fast Fourier transforms of complex numbers in rows `1 << bits` long (at
most 256), each row a group of lanes running every round of butterflies
in place: rows laid end to end, each transformed alone (blocks of a
sound, lines of a picture), or a cube `1 << bits` a side, x fastest,
one axis a pass. The inverse divides by the row's length, so one undoes
the other.

**Layer:** Foundation. **Uses:** no other package.

```text
use "signal.entpkg";
```

## Kernels

- `kernels/fourier.ent` — Fast Fourier transforms of complex numbers
  (`Vec2`, real then imaginary) in rows `n = 1 << bits` long: rows laid
  end to end, or the rows of a cube `n` a side, element
  `x + n * (y + n * z)`, along an axis. Kernels:
  `signal.fourier_turns@1`, `signal.fourier_complex@1`,
  `signal.fourier_real@1`, `signal.fourier_rows@2`.

## Graph presets

- `graphs/fourier_forward.esn` — A cube's transform, along z, then y,
  then x, `1 << bits` a side (at most 256), laid as it came.
- `graphs/fourier_inverse.esn` — A spectrum's cube again, along z, then
  y, then x, each axis divided by `n`, so it undoes
  `fourier_forward.esn`.
- `graphs/fourier_rows_forward.esn` — Rows of `1 << bits` complex
  numbers laid end to end (at most 256 long), each transformed alone,
  laid as they came.
- `graphs/fourier_rows_inverse.esn` — Rows of spectra laid end to end
  again, each divided by its length, so it undoes
  `fourier_rows_forward.esn`.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`fourier.replay`. Every Kernel of the package is reached by one of them.
