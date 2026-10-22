# Genesis

A world that writes its own law. Life grows on a grid by the law in
`laws/grow.ent`, `world.grow@1`: a cell grows where the life around it
is near the law's center, within its width, and fades elsewhere. The
world counts what lives; while its life is too crowded or too sparse,
it fills the law's template, `templates/grow.tmpl`, with another center
and width (`composer.entpkg`) and writes the law again through its
output `laws`. The session reads the law it wrote, compiles it and puts
it in the old one's place a frame later; a law that fails stays out,
the last one runs on, and the input `problems` (`= checks "laws/"`)
tells the world why, which paints a red frame. Each law the world
writes tints its life another hue.

The world writes only where the person running it allows:

```sh
entelechy-player examples/genesis/scene.esn --allow file.write=examples/genesis/laws
```

rewrites `laws/grow.ent` on disk, as the world finds it; `git checkout
examples/genesis/laws` brings the seed back. Played from its replay,
what the world writes is kept for the play alone:

```sh
entelechy-player --replay examples/genesis/tests/genesis.replay
entelechy replay examples/genesis/tests/genesis.replay
```

A served world's `history` names each law it wrote as its own, its
origin `{"output": "laws"}` ([PROTOCOL](../../docs/PROTOCOL.md)).
