# Swarm

Many bodies at once, in two parts. Each part is a whole program that runs alone; `scene.esn` places
`parts/million/`.

- [million](parts/million/README.md): a million bodies stepped by a map and drawn by a scatter.
- [neighbors](parts/neighbors/README.md): sixty-five thousand bodies finding their neighbors through a grid built by a scatter `sum` and a `prefix with sum`.

```sh
entelechy player examples/swarm/scene.esn
entelechy player examples/swarm/parts/million/scene.esn
entelechy player examples/swarm/parts/neighbors/scene.esn
entelechy replay examples/swarm/tests/*.replay --backend gpu
```
