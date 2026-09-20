# Thumb

A Thumb core: the ARM7TDMI's 16-bit instruction set, the mode a Game
Boy Advance runs most of its code in, written as ordered work with two
memories, and the measurement the GBA plan asked for: a real
fetch-decode-execute, stepped by the JIT.

```text
rom ─ boot (RAM) ──┬─ machine (two memories feeding back) ─ view
start (registers) ─┘
```

- `thumb.step` is one instruction per element of `ticks`: a `scan`
  whose `memory` ports are RAM (8 KiB as 2048 words, addressed by the
  byte through `rd8`/`rd16`/`rd32` and `wr8`/`wr16`/`wr32`, functions
  that `put` into their Field parameter) and the registers (r0..r15 and
  the N Z C V flags). All nineteen instruction formats are here: the
  shifts with their carry out, the ALU including ADC, SBC, ROR and MUL,
  the high-register moves and BX, the loads and stores of words,
  halfwords and bytes, sign-extended ones, push, pop, LDMIA, STMIA, the
  conditional and unconditional branches, and BL in its two halves.
  Each step records the halfword it ran.
- The memories persist across frames through `<~`, from `boot` (the
  program's halfwords packed little-endian from address 0) and `start`
  (the stack pointer at 0x0FE0, the pc at 0); F5 in the Player is the
  reset button.
- `thumb.rom` is the program, halfword by halfword with its listing:
  a loop that paints a 32 by 32 framebuffer at 0x1000 with `(x ^ y) +
  t`, then a `tick` routine that counts the frame into RAM at 0x0FF0,
  going through the stack, a high register and a halfword store on the
  way, so most formats run every frame. `speed` is the instructions per
  frame; its default, 9363, is exactly one frame of the program, so the
  picture shifts one shade a frame.
- `thumb.view` draws the framebuffer's bytes as colours.

The replay's values come from a reference simulator of the same program
(the assembler that produced the halfwords ran it), and hold on the CPU,
the JIT and the GPU alike. Speed, one frame of 128 by 128 and
`--param speed=936300`, about a million instructions:

| backend | time | rate |
|---|---|---|
| CPU interpreter | 0.57 s | 1.6 M instructions/s |
| JIT | 0.05 s | 19 M instructions/s |
| GPU session (the scan on the JIT, the picture on the GPU) | about the JIT's | |

A GPU walks a scan on one thread (0.45 M instructions a second here),
so a GPU session hands the scan to the JIT and keeps the picture on the
device. A GBA's ARM7TDMI runs at 16.78 MHz and a Thumb instruction takes one to
three of those cycles plus wait states, so the JIT already steps this
core faster than the real machine; what a GBA adds is the memory map,
the PPU, DMA, timers and interrupts. The core is under 300 lines of
`.orla`; another program is another `rom` Kernel.

```sh
cargo run --release -p orla-player -- examples/thumb/scene.osn --backend jit --window 512x512
cargo run -- replay examples/thumb/tests/thumb.replay --backend jit
```
