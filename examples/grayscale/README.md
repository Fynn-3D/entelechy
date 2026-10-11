# Grayscale

The first proof from the architecture plan:

```text
gradient -> grayscale -> pixelate -> output.ppm
```

```sh
entelechy run examples/grayscale/scene.esn --out output.ppm --watch
```

While it runs, edit `amount` in `scene.esn`, the luma weights in
`kernels/grayscale.ent`, or `size` in `kernels/pixelate.ent`, and save.
The session reports the republished Kernel and the re-executed nodes; a
version that does not compile reports a diagnostic and the running
version stays.
