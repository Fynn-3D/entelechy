# Image

Pictures from arrays, matrices, volumes and samplers. Five parts, each
a scene of its own that shows one thing a Kernel does with data: a
weight table and a `Mat3`, a `Mat4` and its inverse, a case table, a
`Volume<Float>`, and sampling through `mod`. The whole places every
part's scene as a sub-graph (`node blur: graph "parts/blur/scene.esn"
{}`) and lays their pictures out in the window with one small Kernel,
`panels.grid@1`: three across and two down, each scaled whole into its
panel.

```sh
entelechy player examples/image/scene.esn
entelechy run examples/image/scene.esn --window 640x400 --out image.ppm
entelechy replay examples/image/tests/image.replay
```

Each part runs and is tested on its own:

```sh
entelechy player examples/image/parts/<part>/scene.esn
entelechy replay examples/image/tests/<part>.replay
```

## Blur

A five-tap Gaussian as two passes of one Kernel, its weights an array
and its axis a parameter, then an emboss whose 3 by 3 kernel is a
`Mat3`. Reads past an edge are folded back by `mirror`, integer
arithmetic on the cell composed from `at`. The source puts shapes
against the edges, where the folding shows. Tested by `blur.replay`.

## Cube

The cube's placement is a `Mat4` built from a `Mat3` rotation and a
translation; the solid is ray-marched in the cube's own space through
`inverse(model)`. The edges are a corner table (`[Vec3; 8]`) and an edge
table (`[IVec2; 12]`) projected through `lens * model`, drawn as an
x-ray wireframe over the solid. `lens` is a `Mat4` parameter and `style`
a `Style` struct declared in the Kernel's document; `cube.replay` sets a
`mat4(..)` lens and a positional `Style`.

## Marching

Marching squares over a field of three orbiting blobs. Each pixel finds
its cell with `icoord() / cell`, reads the four corners into a
`[Float; 4]`, classifies the cell into one of sixteen cases, and looks
the crossed edges up in `segments`, a `[[Int; 4]; 16]`. Cases and cells
are data, not an `if` chain. Tested by `marching.replay`.

## Volume

A 48 by 48 by 32 `Volume<Float>` of cloud density, filled by one Kernel
with three-dimensional `coord()` and `domain()`, and ray-marched into an
image by another with trilinear `at_linear`. On the GPU the volume stays
resident between nodes and across frames. Tested by `cloud.replay`.

## Scroll

A 64 by 64 tile scrolled endlessly at three texels a pixel, in three
bands: nearest reads through `mod` (exact wrap); `at_linear` through
`mod` (a seam every 192 pixels, since the blend cannot reach across the
tile edge); and `at_linear_wrapped` from the shared `picture` package,
which blends four wrapped `at` reads and has no seam. `scroll.replay`
pins the seam at exact pixels of the part's own 384 by 240 picture.
