# Packages

A package is a domain of reusable capability with a stable interface: a
directory with a `package.esn` naming it and the least language and IR it
needs, a `kernels/` directory it `use`s, `graphs/` presets a scene places
as nodes, and, as it needs them, `data/` tables and `assets/` files. A
scene, an example or another package uses it whole, `use "…/name.entpkg";`,
never a file inside it, so the manifest's versions hold; a preset, a table
or an asset is reached by its path inside it
(`input font: Grid<Float> = "text.entpkg/assets/font.png";`). Every package
has a README and its own `tests/`: scenes and their replays, run on every
backend, that reach every Kernel it declares, so a package carried alone
is proved alone. Examples show what a whole program does with packages;
they are not what tests them. `entelechy pack` carries one as a single `.entpkg` file and
`entelechy unpack` opens it again; `entelechy package <dir>` prints what
one holds.

## How a package is made

- **Name.** One English noun in lower case naming its domain: `image`,
  `raster`, `physics`. No verbs, no abbreviations, no `utils`.
- **Layer.** A package sits in one layer and uses packages of its own
  layer or those below, never one above, and no two use each other.
- **Files.** `kernels/<topic>.ent` and `graphs/<product>.esn`, singular
  nouns in snake_case, never the package's own name: `color.ent`,
  `tone_map.ent`, `broad_phase.ent`.
- **Tests.** `tests/<topic>.esn` and its `<topic>.replay`, with whatever
  data they read under `tests/data/`; a test uses the package by
  `use "../kernels/";` and its presets by `graph "../graphs/…"`, and other
  packages by name.
- **Kernels.** An id begins with the package's name and says what it
  makes or does: `raster.cover@4`, `parallel.sort_place@1`,
  `light.probes_gather@1`.
- **Names.** A function is a verb or a noun phrase in snake_case, a struct
  or type a noun in PascalCase. No name is declared twice among all the
  packages, and a general word is qualified by its topic (`text_ink`,
  `TilePlace`), so a scene using every package calls each name plainly.
- **Spelling.** American English, in names and in prose: `color`,
  `center`, `gray`.

A project's own packages are named by two words or more (`game_music`,
`pals_island`), so they never meet these; a scene using one that declares
one of these names writes it with the package's name before it,
`random::fbm2`. `cargo xtask packages` holds the packages to all of this,
and `cargo xtask verify` runs it.

## Reaching a package

A scene reaches a package by path:

```text
use "../../packages/image.entpkg";
node view: graph "../../packages/image.entpkg/graphs/tonemap.esn" { ... }
```

or by name, when `ENTELECHY_PACKAGES` lists the directory it is in:

```sh
ENTELECHY_PACKAGES=/path/to/entelechy/packages entelechy run scene.esn
```

A package names the packages it uses by name, `use "random.entpkg";`, and
finds them beside itself. A directory is named for the package its
manifest declares, `name@1` in `name.entpkg`, and a scene reaches one
package of a name: two editions of one are refused, naming both. A
Kernel's own id carries the version its interface is held to.

## The packages

### Foundation

Arithmetic and the algorithms everything else builds on.

| Package | Holds |
| --- | --- |
| [`random`](random.entpkg/README.md) | Hashes and noise a Kernel can repeat, the same bits on every backend |
| [`geometry`](geometry.entpkg/README.md) | Quaternions, transforms, camera projection, signed distances and the warps of space |
| [`parallel`](parallel.entpkg/README.md) | Sorting, compaction and runs over rows |
| [`signal`](signal.entpkg/README.md) | Fast Fourier transforms, forward and inverse |
| [`navigation`](navigation.entpkg/README.md) | Ways over a grid of costs |

### Graphics

Pictures and what a camera sees.

| Package | Holds |
| --- | --- |
| [`image`](image.entpkg/README.md) | A picture alone: color, blending and masks, filters, a grade, tone, glow, exposure, film, scaling, antialiasing, histograms, distance fields, mipmaps, pixel art |
| [`raster`](raster.entpkg/README.md) | Triangles rasterized alike on every backend, plain, tiled or cut out |
| [`voxel`](voxel.entpkg/README.md) | Rays walked through voxels, shapes as data, bodies posed by joints |
| [`vector`](vector.entpkg/README.md) | SVG drawings filled pixel by pixel, whole or tiled |
| [`mesh`](mesh.entpkg/README.md) | Triangle meshes as points and faces: lattices, primitives and subdivision |
| [`terrain`](terrain.entpkg/README.md) | Ground as heights: sampled, ray-marched, sculpted, eroded and meshed |
| [`texture`](texture.entpkg/README.md) | Patterns, projections, layers, normals and the marks of age a material is drawn with |
| [`light`](light.entpkg/README.md) | Light in a world a scene hands over: the G-buffer contract, shading, the sun, sky and lamps, shadows, bounced light, fog and air, and the renderers |
| [`particles`](particles.entpkg/README.md) | Particles stepped, binned and drawn over a picture's depth |
| [`scatter`](scatter.entpkg/README.md) | Things scattered over the ground by layers |
| [`effects`](effects.entpkg/README.md) | Effects as layers of forms in time, made of a matter, drawn as light along each ray |

### Media

Sound and music.

| Package | Holds |
| --- | --- |
| [`audio`](audio.entpkg/README.md) | Clocks, cues, instruments, filters and delay lines for blocks of samples |
| [`music`](music.entpkg/README.md) | Music made from a mood, a band and a seed |
| [`sound`](sound.entpkg/README.md) | The sound of effects: each layer heard as its form sounds, in its matter's voice |

### Simulation

Bodies, skeletons and worlds that change.

| Package | Holds |
| --- | --- |
| [`physics`](physics.entpkg/README.md) | Rigid bodies in the plane |
| [`animation`](animation.entpkg/README.md) | Skeletons, clips, walking, aiming and skinning |
| [`genesis`](genesis.entpkg/README.md) | Laws a world writes itself |

### Interaction

What a person reads and does.

| Package | Holds |
| --- | --- |
| [`text`](text.entpkg/README.md) | A bitmap font, glyph rows, letters and wide-glyph atlases |
| [`ui`](ui.entpkg/README.md) | Layout, paint, widgets, navigation, command search and marks |
| [`controls`](controls.entpkg/README.md) | Actions from the keyboard and a pad alike |
