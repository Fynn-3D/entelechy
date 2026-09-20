# CHIP-8

A CHIP-8 interpreter: the machine of 1977, written as ordered work with
memories, and the first machine Orla runs. Nothing was added to the Host
for it; the language gained memories, `put`, and the bit operators.

```text
rom ─ boot (RAM) ──┐
start (registers) ─┼─ machine (three memories feeding back) ─ view
blank (screen) ────┘
```

- `chip8.step` is one instruction per element of `ticks`: a `scan` whose
  three `memory` ports are RAM (4096 bytes), the registers (V0..VF, I, PC,
  SP, the timers, the random state, the stack) and the screen (64 by 32
  cells). Each step reads them as the earlier steps left them, `put`s
  what it changes, and records the opcode it ran. The timers count down
  at the first step of a frame. The keypad is the keyboard's 1 2 3 4 /
  Q W E R / A S D F / Z X C V. Every opcode of the original is here.
- The memories persist across frames through `<~`, from `boot` (the font
  at 0x50, the program at 0x200), `start` and `blank`; F5 in the Player
  is the reset button.
- `speed` is the steps per frame: a graph param, so it changes while the
  machine runs.
- `chip8.rom` is the program, bytes by arithmetic: a box the keypad's
  4, 6, 2 and 8 move one cell a frame, paced by the delay timer so the
  picture never flickers; the replay drives it with the keys and reads
  the screen memory, the registers, and the pixels on both backends.
- `chip8.view` draws the screen memory into the window.

A frame of 40 steps takes about 0.4 ms on the CPU, and `--backend jit`
runs the step as native code, four million steps in 0.18 s against 1.67
s on the interpreter (`--param speed=4000000`); the whole machine is
under 200 lines of `.orla`. Another program is another `rom` Kernel, or
a `.field` of bytes read as an input.

```sh
cargo run --release -p orla-player -- examples/chip8/scene.osn --window 640x320
cargo run -- replay examples/chip8/tests/chip8.replay
```
