# Package

A scene through a package. [`image.entpkg`](../../packages/image.entpkg)
names itself `image@1` in `package.esn`, with the language and IR it
needs; `graphs/nes.esn` is a preset that pixelates a picture and holds
it to the colours the scene gives, here four from dark to light. The
scene `use`s the package for its Kernels and places the preset by path
as a node; both reach one Kernel file.

```sh
entelechy player examples/package/scene.esn
entelechy replay examples/package/tests/package.replay
entelechy package packages/image.entpkg
entelechy pack packages/image.entpkg /tmp/image.entpkg
entelechy unpack /tmp/image.entpkg /tmp/image
```
