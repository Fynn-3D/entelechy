# Tiles

A tile map as `Grid<Tile>`, each element `{ kind: Int, height: Float,
passable: Bool }`. `generate` lays out the map; `flood` spreads water
to passable neighbours each frame through `<~`; `render` draws each
pixel from the map and a sprite atlas, at `domain = size(map) * tile`.

The atlas is `assets/atlas.png`, an input the Host reads and watches,
so edits to it show while the Player runs. `bake.osn` draws it from
`tiles.atlas` and saves it with `run --save`.

```sh
cargo run -- run examples/tiles/bake.osn --save atlas=examples/tiles/assets/atlas.png
cargo run --release -p orla-player -- examples/tiles/scene.osn
cargo run -- replay examples/tiles/tests/flood.replay
```
