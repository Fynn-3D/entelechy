# Fluid

Smoke on a grid: the Navier-Stokes equations for an incompressible fluid,
solved the way real-time smoke has been since Stam, and nothing was added
to the language for it. Drag the pointer to stir; a plume at the bottom
keeps the fluid moving when nobody does.

```text
still ─┐
clear ─┼─ advect ─ divergence ─ solve ×5 (8 Jacobi steps each) ─ project ─ smoke ─ view
rest ──┴─ hand
```

- The velocity is a `Grid<Vec2>` fed back through the Graph from the end
  of one frame to the start of the next; `fluid.advect` carries it by
  itself (semi-Lagrangian, a midpoint step, `at_linear` for the reads
  between cells) and adds the forces: the hand's motion over a soft
  disc, the plume's push, the lift of warm smoke, and vorticity
  confinement, which keeps the small eddies the grid would smooth away.
- The grid is staggered: a cell's x-velocity stands on its left face and
  its y-velocity on its top face, so `fluid.divergence` and the pressure
  gradient in `fluid.project` are face differences that agree with the
  Jacobi stencil in `fluid.relax`, and the projection can drive the
  divergence toward zero rather than toward a checkerboard.
- The pressure solve is one Kernel placed forty times: `solve.osn` is
  eight steps, the scene places it five times, and it starts from last
  frame's pressure, fed back the same way as the velocity.
- `fluid.smoke` carries the smoke by the projected velocity and feeds it
  from the plume and the hand; `fluid.draw` paints it, tinted by speed.
- `fluid.hand` is one row: the pointer, its button, and how far it moved
  since last frame while down, which is what stirs.

The replay runs the plume on a 64 by 64 grid and reads `fluid.total`, a
`merge` Kernel summing |divergence| before and after the projection:
forty Jacobi steps remove about four fifths of it in one frame, and
the rest the next frames. It then stirs and reads the hand's velocity,
the fluid's, and the smoke it carried.

One frame at 384 by 216 with forty Jacobi steps is 52 nodes and about
5 ms on the GPU over the protocol; on the CPU the fixture runs at a
few frames a second.

```sh
cargo run --release -p orla-player -- examples/fluid/scene.osn --scale 3
cargo run -- replay examples/fluid/tests/fluid.replay --backend gpu
```
