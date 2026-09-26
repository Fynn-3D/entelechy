# Package

A scene through a package. [`pixelfx.orlapkg`](../../packages/pixelfx.orlapkg)
names itself `pixelfx@1` in `package.osn`, with the language and IR it
needs; `graphs/nes.osn` is a preset that pixelates a picture to four
colors. The scene `use`s the package for its Kernels and places the
preset by path as a node; both reach one Kernel file.

```sh
orla player examples/package/scene.osn
orla replay examples/package/tests/package.replay
orla package packages/pixelfx.orlapkg
orla pack packages/pixelfx.orlapkg /tmp/pixelfx.orlapkg
orla unpack /tmp/pixelfx.orlapkg /tmp/pixelfx
```
