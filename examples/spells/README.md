# Spells

Every spell of the [effects package](../../packages/effects.entpkg/package.esn)
at once, each on its own square of ground, cast again each time it ends,
its name over it, thirty to a page; and each heard as it is drawn
through the [sound package](../../packages/sound.entpkg/package.esn).

A spell is data: an effect (a few layers of forms in time,
`data/effects.csv` and `data/layers.csv`) in an element (colors, light,
smoke, flicker, `data/elements.csv`) at a tier (`data/spells.csv`). The
package draws a project's casts over its picture, whose alpha holds how
far each pixel is, as light gathered along each pixel's ray with the
effects package's glow.

The right button turns the eye, the wheel brings it near, the arrows
move it over the grid, Q and E turn the page. The params `element` and `tier` draw them all in one
element or at one tier.

```sh
entelechy player examples/spells/scene.esn
entelechy replay examples/spells/tests*.replay --backend gpu
```

`tests/spells.replay` checks, on every backend, a fireball's core and
flame, a lightning wheel and the ground on the first frame, the thirty
casts heard together, and where they have gone a second on.
