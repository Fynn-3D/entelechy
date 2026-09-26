# Thumb

A Thumb core (the ARM7TDMI's 16-bit instruction set) as ordered work
over two memories, stepped by the JIT.

```text
rom ─ boot (RAM) ──┬─ machine (two memories feeding back) ─ view
start (registers) ─┘
```

- `thumb.step`: one instruction per element of `ticks`, a `scan` whose
  `memory` ports are 8 KiB of RAM and the registers (r0..r15, N Z C V).
  It implements all nineteen Thumb instruction formats and records each
  halfword it runs.
- The memories persist across frames through `<~`, seeded by `boot`
  (the program at address 0) and `start` (sp at 0x0FE0, pc at 0); F5
  resets.
- `thumb.rom`: the program, with its listing. It paints a 32 by 32
  framebuffer at 0x1000 with `(x ^ y) + t` and counts frames at 0x0FF0.
  `speed` is instructions per frame; the default, 9363, is one frame of
  the program.
- `thumb.view`: draws the framebuffer.

The replay's values come from a reference simulator of the same
program, and hold on the CPU, the JIT and the GPU. A GPU session runs
the scan on the JIT and the picture on the device.

```sh
cargo run --release -p orla-player -- examples/thumb/scene.osn --backend jit --window 512x512
cargo run -- replay examples/thumb/tests/thumb.replay --backend jit
```
