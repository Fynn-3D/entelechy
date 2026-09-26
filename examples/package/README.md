# Package

A scene through a package. [`pixelfx.orlapkg`](../../packages/pixelfx.orlapkg)
names itself `pixelfx@1` in `package.osn`, with the language and IR it
needs; `graphs/nes.osn` is a preset that pixelates a picture to four
colours. The scene `use`s the package for its Kernels and places the
preset by path as a node; both reach one Kernel file.

```sh
cargo run --release -p orla-player -- examples/package/scene.osn
cargo run -- replay examples/package/tests/package.replay
cargo run -- package packages/pixelfx.orlapkg
cargo run -- pack packages/pixelfx.orlapkg /tmp/pixelfx.orlapkg
cargo run -- unpack /tmp/pixelfx.orlapkg /tmp/pixelfx
```
