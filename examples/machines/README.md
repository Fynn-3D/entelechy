# Machines

Processors written as a `scan` over their memory, running real programs, in two parts. Each part is a whole program that runs alone; `scene.esn` places
`parts/thumb/`.

- [chip8](parts/chip8/README.md): a CHIP-8 interpreter over RAM, registers and screen.
- [thumb](parts/thumb/README.md): an ARM7TDMI Thumb core running a program that paints a framebuffer.

```sh
entelechy player examples/machines/scene.esn
entelechy player examples/machines/parts/chip8/scene.esn
entelechy player examples/machines/parts/thumb/scene.esn
entelechy replay examples/machines/tests/*.replay --backend gpu
```
