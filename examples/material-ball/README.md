# Material ball

A shaded sphere on a checker floor under an orbiting light, rendered entirely
by two `.orla` Kernels and composed by `scene.osn`:

```text
material.ball@1 -> image.tonemap@1 -> window
```

Open it in the Player (release builds keep the frame rate up):

```sh
cargo run --release -p orla-player -- examples/material-ball/scene.osn
```

While the window is open, edit and save:

- `albedo`, `roughness`, or `metallic` in `scene.osn` to change the material;
- `kernels/ball.orla` to change the lighting or the floor;
- `exposure` in `scene.osn` or the curve in `kernels/tonemap.orla`.

The light orbits because the ball Kernel declares `time.read`; the tone-mapping
node is pure and is reused until its input changes. Save a Kernel that does
not compile and the diagnostic appears on the terminal while the last good
version keeps rendering. Press Escape to quit.

A single frame without a window:

```sh
cargo run --release -- run examples/material-ball/scene.osn --out ball.ppm
```
