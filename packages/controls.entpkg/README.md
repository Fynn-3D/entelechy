# controls

A game's actions read from the keyboard, the mouse and a pad alike, from
a table of actions: held, pressed and released, the ways to move and
look, and the stick's flicks.

**Layer:** Interaction. **Uses:** no other package.

```text
use "controls.entpkg";
```

## Kernels

- `kernels/action.ent` — A game's actions from the keyboard and a pad
  alike (`../package.esn`). Kernels: `controls.start@1`,
  `controls.read@1`.

## Tested by

[controls](../../examples/controls/README.md), whose replays hold it on every backend.
