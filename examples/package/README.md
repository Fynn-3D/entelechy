# Package

A scene through a package. [`image.orlapkg`](../../packages/image.orlapkg)
names itself `image@1` in `package.osn`, with the language and IR it
needs; `graphs/nes.osn` is a preset that pixelates a picture and holds
it to the colours the scene gives, here four from dark to light. The
scene `use`s the package for its Kernels and places the preset by path
as a node; both reach one Kernel file.

```sh
orla player examples/package/scene.osn
orla replay examples/package/tests/package.replay
orla package packages/image.orlapkg
orla pack packages/image.orlapkg /tmp/image.orlapkg
orla unpack /tmp/image.orlapkg /tmp/image
```
