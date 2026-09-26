# Level

A custom binary format read by a Kernel. `level.bin` holds a magic, a
version, the map's width and height, the tiles, a two-byte entity count
and the entities. The scene takes the file with `bytes` (rows of one
integer a byte), and `level.draw` walks it by offset and loops over the
count; the compiler bounds that loop. The format lives in the Kernel, so
changing it is a saved edit.

```text
level.bin ─ bytes ─┬─ draw (the map and its entities)
                   └─ about (what the header says) ─ label
```

```sh
cargo run --release -p orla-player -- examples/level/scene.osn
cargo run --release -- replay examples/level/tests/level.replay
```
