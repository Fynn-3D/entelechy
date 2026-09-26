# CHIP-8

A CHIP-8 interpreter written as ordered work with memories.

```text
rom ─ boot (RAM) ──┐
start (registers) ─┼─ machine (three memories feeding back) ─ view
blank (screen) ────┘
```

- `chip8.step` runs one instruction per element of `ticks`: a `scan`
  whose three `memory` ports are RAM, the registers and the screen. It
  implements every opcode. The keypad is the keyboard's 1 2 3 4 / Q W E R
  / A S D F / Z X C V.
- The memories persist across frames through `<~`, from `boot`, `start`
  and `blank`; F5 in the Player resets.
- `speed` is the steps per frame, a graph param.
- `chip8.rom` is the program: a box the keypad's 4, 6, 2 and 8 move.
- `chip8.view` draws the screen memory.

`--backend jit` runs the step as native code.

```sh
cargo run --release -p orla-player -- examples/chip8/scene.osn --window 640x320
cargo run -- replay examples/chip8/tests/chip8.replay
```
