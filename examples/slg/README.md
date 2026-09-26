# SLG

A small NES-style tactics game. Arrows move the cursor, Z confirms, X
cancels, Enter ends the turn; a gamepad's D-pad, A, B and Start do the
same. Each side has four soldiers; the river has two fords.

```text
begin ─┐
       ├─ game (the turn machine) ──┬─ units (applies orders) ──┐
spawn ─┘                            │                            ├─ view
map ─────────────────────────────────┼─ seed ─ range (6 passes) ─┤
blank ──────────────────────────────┴─ lure ─ hunt (12 passes) ─┘
```

- `slg.game`: one row of state (phase, cursor, selection, this frame's
  order, last frame's keys); it issues orders.
- `slg.units`: applies the order to the soldiers: move, strike, wait,
  new turn.
- `slg.relax`, placed repeatedly through `passes.osn`: movement range,
  and the distance field the enemy follows to the fords.
- `slg.draw`: tiles, sprites, cursor, health bars, status bar.
- `bgm.play`: a sixteen-step chiptune sequencer with a noise hit on each
  strike; `bgm.stereo` feeds the `sound` output.

The skirmish replay asserts on the state rows through a selection, a
strike, and an enemy turn.

```sh
orla player examples/slg/scene.osn --scale 3
orla replay examples/slg/tests/skirmish.replay
```
