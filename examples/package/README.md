# Package

A scene through a package. `../pixelfx.orlapkg` is a package directory:
its `package.osn` names it, `pixelfx@1`, and the least language and IR
it needs, and uses its `kernels/`; `graphs/nes.osn` is a preset, a graph
that pixelates a picture and holds it to four colours. The scene says
`use "../pixelfx.orlapkg";` for the Kernels and places the preset by
path as a node; the two reach the same Kernel file by different paths,
which is one file and one owner.

```sh
cargo run --release -p orla-player -- examples/package/scene.osn
cargo run -- replay examples/package/tests/package.replay
cargo run -- package examples/pixelfx.orlapkg
cargo run -- pack examples/pixelfx.orlapkg /tmp/pixelfx.orlapkg
cargo run -- unpack /tmp/pixelfx.orlapkg /tmp/pixelfx
```
