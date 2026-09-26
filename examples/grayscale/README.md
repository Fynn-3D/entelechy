# Grayscale

The first proof from the architecture plan:

```text
gradient -> grayscale -> pixelate -> output.ppm
```

```sh
orla run examples/grayscale/scene.osn --out output.ppm --watch
```

While it runs, edit `amount` in `scene.osn`, the luma weights in
`kernels/grayscale.orla`, or `size` in `kernels/pixelate.orla`, and save.
The session reports the republished Kernel and the re-executed nodes; a
version that does not compile reports a diagnostic and the running
version stays.
