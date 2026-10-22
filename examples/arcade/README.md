# Arcade

Small games written as a few Kernels, in two parts. Each part is a whole program that runs alone; `scene.esn` places
`parts/blocks/`.

- [blocks](parts/blocks/README.md): falling blocks, pieces as a bit-mask table.
- [snake](parts/snake/README.md): a game in five Kernels.

```sh
entelechy player examples/arcade/scene.esn
entelechy player examples/arcade/parts/blocks/scene.esn
entelechy player examples/arcade/parts/snake/scene.esn
entelechy replay examples/arcade/tests/*.replay --backend gpu
```
