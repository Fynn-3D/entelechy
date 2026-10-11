# Logo

Entelechy's logo, drawn by Entelechy: a gold seed on the ground grows a
graph, one, two, three, then the ten thousand things, whole and thick near
the seed, dotted and hollow as it reaches out, still becoming. Beside it is
the name, each letter a few strokes measured in one Kernel, no font; the C
holds the seed.

- **A construction.** The mark stands on a lattice of step `a`, the
  diamond's half-diagonal, and every radius and stroke is a whole number of
  modules `m = a / 40`. The one, the two and the middle of the three are the
  diamond's corners; the outer two of the three lie on a 3-4-5 slope from
  the two; the nine rings of the crown stand fifteen degrees apart on a
  circle of three steps. `kernels/logo.ent` writes each as a function.
- **A lockup in the same steps.** A margin of `a`, the mark, a gap of `a`,
  the name two steps high on the diamond's middle line, as heavy as the
  trunk.
- **Its own drawing sheet.** `construction.esn` draws the sheet the mark
  is built on: the grid and the lattice, the guides each point stands
  on, the crown's circle, the diamond, the 3-4-5 triangles of the outer
  branches, every circle's outline, the mark faint beneath. The sheet is
  drawn by the same functions that draw the mark (`kernels/construction.ent`
  uses `kernels/logo.ent`), so it cannot disagree with it: change the
  lattice or a measure and the mark and its sheet move together.
- **One Kernel, every version.** `light`, `ink`, `alone` and `clear` choose
  dark or light, colour or one ink (`ink = 2` pure black or white), the
  lockup or the mark alone, on a ground or on nothing.

```sh
entelechy player examples/logo/scene.esn
entelechy player examples/logo/construction.esn --window 800x800
entelechy run examples/logo/scene.esn --param light=1 --out logo.png
entelechy run examples/logo/scene.esn --param alone=1 --param width=1024 --param height=1024 --out mark.png
entelechy replay examples/logo/tests/logo.replay examples/logo/tests/construction.replay --backend gpu
```

The front page's two logos are this scene at 1200 by 360 with `clear=1`,
`light=0` and `light=1`.

`tests/logo.replay` holds the seed, the first node, a hollow ring, a letter
and the C's seed to their colours, on dark, on light and in one ink;
`tests/construction.replay` holds the sheet's guide at the seed's middle
and its outline at the seed's radius over the faint seed, and the
diamond's edge.
