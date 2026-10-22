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
- `grid.relax` from the math package, placed repeatedly through
  `passes.esn` over `slg.costs` (a tile's cost, or none where a foe
  stands): movement range, and the distance field the enemy follows to
  the fords.
- `slg.draw`: tiles, sprites, cursor, health bars, status bar.
- `bgm.play`: a sixteen-step chiptune sequencer on the audio package's
  clock, with a noise hit on each strike; `bgm.stereo` feeds the `sound`
  output.

The skirmish replay asserts on the state rows through a selection, a
strike, and an enemy turn.

```sh
entelechy player examples/slg/scene.esn --scale 3
entelechy replay examples/slg/tests/skirmish.replay
```
