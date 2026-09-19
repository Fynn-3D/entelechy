# Examples

Each example is a directory with a `scene.osn` entry and a `kernels/` directory
of `.orla` sources. Documentation for an example belongs in its own `README.md`.

```sh
cargo run -- run examples/<example>/scene.osn --out output.ppm --watch
```

- [grayscale](grayscale/README.md): the first proof. A generated gradient flows
  through grayscale and pixelate Kernels; edit any Kernel while it runs.
- [material-ball](material-ball/README.md): a shaded sphere under an orbiting
  light, tone-mapped by a second Kernel, shown live in the Player.
