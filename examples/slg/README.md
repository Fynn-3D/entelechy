# SLG

A small tactics game in the NES manner, and nothing was added to the
language for it. Arrows move the cursor, Z confirms, X cancels, Enter
ends the turn; each side has four soldiers, the river has two fords.

```text
begin ─┐
       ├─ game (the turn machine) ──┬─ units (applies orders) ──┐
spawn ─┘                            │                            ├─ view
map ─────────────────────────────────┼─ seed ─ range (6 passes) ─┤
blank ──────────────────────────────┴─ lure ─ hunt (12 passes) ─┘
```

- `slg.game` is one row of state: phase, cursor, selection, an order for
  this frame, and last frame's key bits so a key counts once per press.
  It never touches a unit; it issues orders.
- `slg.units` applies the order to the row of soldiers: move, strike
  (attack less defence, at least one), wait, or a new turn.
- Movement range is `slg.relax` placed six times through `passes.osn`, a
  sub-graph taking Fields as inputs: each pass takes the cheapest way in
  from a neighbour, water and the other side's soldiers blocking. The same
  passes seeded at the enemy's prey measure how far every cell is from
  it, so the enemy walks to the fords instead of at the river.
- The enemy acts one soldier at a time on a timer, which is also the
  animation: choose, step, strike, next.
- `slg.draw` is 16-pixel tiles, sprites from arithmetic, a blinking
  cursor, health bars, a status bar, and a tint for victory or defeat.

The music is two more Kernels: `bgm.play` is a sixteen-step chiptune
sequencer, square lead over triangle bass, whose sample counter and
phases cross blocks through Graph state, and which starts a decaying
noise hit when the rules issue a strike, so the sound follows the game;
`bgm.stereo` hands the frames to the sound card as the `sound` output.
A gamepad plays it too: the D-pad or stick moves, A confirms, B cancels,
Start ends the turn.

Both backends play the same game; the skirmish replay moves the camps
into reach and asserts on the state rows: a selection, a strike that
leaves the enemy at 7 and starts the hit sound, a sample counter that
never skips, an enemy turn that leaves the soldier at 5.

```sh
cargo run --release -p orla-player -- examples/slg/scene.osn --scale 3
cargo run -- replay examples/slg/tests/skirmish.replay
```
