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

The atlas is a file: `input atlas: Image = "assets/atlas.ppm";` declares
an input the Host reads from disk and watches, so editing the image in
any paint program changes the tiles while the Player runs. The file was
baked from `tiles.atlas`, a Kernel that draws the sprites by arithmetic,
through `bake.osn` and `run --save`; a Field of any shape saves and loads
the same way as a `.field` file.

```sh
cargo run -- run examples/tiles/bake.osn --save atlas=examples/tiles/assets/atlas.ppm
cargo run --release -p orla-player -- examples/tiles/scene.osn
cargo run -- replay examples/tiles/tests/flood.replay
```
