# Tiles

A tile map as a `Grid<Tile>`: each element is `{ kind: Int, height: Float,
passable: Bool }`, exact integers and real booleans rather than values
smuggled through colour channels. `type Map = Grid<Tile>;` names it once
for every Kernel in the file.

`generate` lays out the map, `flood` spreads water to passable neighbours
every frame through the Graph's `<~` state, and `render` draws each pixel
by finding its tile with `at(map, cell)` and its sprite in an atlas image
with `at(atlas, ...)`: two grids of different element types in one Kernel.
The image is sized by the map: `domain = size(map) * tile;` reads the
render node's own `tile` parameter, so changing the map's size changes
the picture's.

```sh
cargo run --release -p orla-player -- examples/tiles/scene.osn
cargo run -- replay examples/tiles/tests/flood.replay
```
