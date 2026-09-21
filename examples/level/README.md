# Level

A binary format nothing in Orla knows, read by a Kernel. `level.bin` is
four bytes of magic, a version, the map's width and height, that many
tiles, a count two bytes wide and that many entities; the scene takes
the file with `bytes`, which is rows of one integer a byte, and
`level.draw` walks it: the header by its offsets, a tile by
`7 + y * width + x`, the count as `low + high * 256`, and then a loop
over that count to draw every entity the file lists.

No importer and no Rust: the format lives in the Kernel, so changing it
is a saved edit while the scene runs, and it runs on whichever backend
the session has. A count read from a file is still a count the compiler
bounds, so a corrupt one cannot run away.

```text
level.bin ─ bytes ─┬─ draw (the map and its entities)
                   └─ about (what the header says) ─ label
```

```sh
cargo run --release -p orla-player -- examples/level/scene.osn
cargo run --release -- replay examples/level/tests/level.replay
```

This is the way in for a format of your own. What other software writes
(glTF, MIDI, a compressed sound) goes through `orla-import`, which has
a reader for each; what is yours needs no reader at all.
