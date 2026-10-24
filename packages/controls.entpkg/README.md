# controls

What a person's hands do, read for a program, and the one package that
reads them: a game's actions from the keyboard, the mouse and a pad
alike, from a table of actions (held, pressed and released, the ways to
move and look, the stick's flicks); the pointer (where, its buttons, the
wheel); what was typed; and a camera they turn (an orbit worked by the
pointer, the wheel and keys).

**Layer:** Interaction. **Uses:** [`geometry`](../geometry.entpkg/README.md), [`surface`](../surface.entpkg/README.md).

```text
use "controls.entpkg";
```

## Kernels

- `kernels/action.ent` — A game's actions from the keyboard and a pad
  alike (`../package.esn`). Kernels: `controls.start@1`,
  `controls.read@1`.
- `kernels/pointer.ent` — The pointer: where it is, its buttons held,
  pressed and released, the wheel, whether it moved. Kernels:
  `controls.pointer_start@1`, `controls.pointer@1`.
- `kernels/typing.ent` — What was typed this frame. Kernels:
  `controls.typing@1`.
- `kernels/orbit.ent` — A camera a person turns (`Orbit`): the left
  button drags to turn, the right or middle to pan, the wheel dollies, a
  double-click or F flies to the point under the pointer, W A S D walk,
  Q E fall and rise, Shift quickens, R returns home, never under the
  ground; the lens it sees through, and whether it rests. Kernels:
  `controls.orbit_start@1`, `controls.orbit@1`, `controls.orbit_rests@1`,
  `controls.orbit_lens@1`.

## Tests

`tests/` holds scenes and their replays, run on every backend:
`actions.replay`, `orbit.replay`, `pointer.replay`. Every Kernel of the package is reached by one of them.

Used by the [controls](../../examples/controls/README.md) example.
