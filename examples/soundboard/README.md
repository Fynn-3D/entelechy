# Sound board

Every event sound of the
[sound package](../../packages/sound.entpkg/package.esn),
one at a time: steps, the menus, a fight's blows, the world's doors and
tools, the short tunes of what happens. Each sounds in a material
(`data/materials.csv`: grass, stone, wood, metal and the rest) or a style
(`data/styles.csv`: wood, metal, crystal, arcane, retro, soft), made by
the audio package's voices.

Up and Down choose the event, Left and Right its material or style, T
how (a step's gait and weight, a blow's strength), Space plays it, and A
plays them all one after another.

```sh
entelechy player examples/soundboard/scene.esn
entelechy replay examples/soundboard/tests/*.replay --backend gpu
```

`tests/soundboard.replay` checks the silence before anything plays, the
first event played by Space and the next played as Down chooses it, by
their samples.
