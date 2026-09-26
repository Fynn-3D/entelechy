# Fluid

Smoke on a grid: incompressible Navier-Stokes, solved Stam-style. Drag
the pointer to stir; a plume at the bottom keeps the fluid moving.

```text
still ─┐
clear ─┼─ advect ─ divergence ─ solve ×5 (8 Jacobi steps each) ─ project ─ smoke ─ view
rest ──┴─ hand
```

- The velocity is a staggered `Grid<Vec2>` fed back through `<~`.
  `fluid.advect` carries it semi-Lagrangian and adds the hand, the
  plume, buoyancy and vorticity confinement.
- `fluid.divergence`, `fluid.relax` and `fluid.project` are face
  differences on the staggered grid.
- The pressure solve is `solve.osn` (eight Jacobi steps) placed five
  times, starting from last frame's pressure.
- `fluid.smoke` carries the smoke; `fluid.draw` paints it, tinted by speed.
- `fluid.hand` is one row: the pointer, its button, and its motion.

The replay runs the plume on a 64 by 64 grid, reads `fluid.total` (the
summed |divergence| before and after projection), then stirs and reads
the velocities and the smoke.

```sh
cargo run --release -p orla-player -- examples/fluid/scene.osn --scale 3
cargo run -- replay examples/fluid/tests/fluid.replay --backend gpu
```
