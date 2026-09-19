# Grayscale

The first-proof demonstration from the architecture plan:

```text
gradient -> grayscale -> pixelate -> output.ppm
```

```sh
cargo run -- run examples/grayscale/scene.osn --out output.ppm --watch
```

While it runs, change `amount` in `scene.osn`, the luma weights in
`kernels/grayscale.orla`, or `size` in `kernels/pixelate.orla`, and save. The
session reports which Kernel was republished and which nodes re-executed. Save
a version that does not compile to see the diagnostic and the running version
survive.
