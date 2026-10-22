# physics

Rigid circles and boxes in the plane that fall, collide, rest, slide and
bounce: pairs found through a grid, manifolds solved all at once over
substeps.

**Layer:** Simulation. **Uses:** [`parallel`](../parallel.entpkg/README.md).

```text
use "physics.entpkg";
```

## Kernels

- `kernels/body.ent` — Rigid bodies in the plane.
- `kernels/broad_phase.ent` — Finding pairs. Kernels:
  `physics.bounds@1`, `physics.spans@1`, `physics.owners@1`,
  `physics.cells@1`, `physics.no_cells@1`, `physics.cell_counts@1`,
  `physics.places@1`, `physics.no_entries@1`, `physics.pair_counts@1`,
  `physics.no_pairs@1`, `physics.pairs@1`.
- `kernels/contact.ent` — Colliding. Kernels: `physics.collide@1`,
  `physics.shared@1`, `physics.no_counts@1`, `physics.no_pushes@1`.
- `kernels/query.ent` — Asking the world. Kernels: `physics.ray@1`,
  `physics.no_contacts@1`.
- `kernels/solve.ent` — Solving. Kernels: `physics.warm@1`,
  `physics.solve@1`, `physics.bounce@1`, `physics.pushes@1`,
  `physics.push@1`, `physics.fall@1`, `physics.move@1`.

## Graph presets

- `graphs/step.esn` — One step of a world of bodies, `dt` long.
- `graphs/substep.esn` — One substep of `step.esn`.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`many.replay`, `motion.replay`, `pile.replay`, `rest.replay`,
`roll.replay`, `wide.replay`. Every Kernel of the package is reached by
one of them.

Used by the [physics](../../examples/physics/README.md) example.
