# landscape

A valley with a lake in it: forested hills, rock and snow above them,
clouds over it all, and the far shore standing upside down in the water.
Nothing in the scene is a model. The ground is a function of where you
are, and what the lake shows is that same function marched again along
the reflected ray, so the mountains in the water are the mountains.

```sh
cargo run --release -p orla-player -- examples/landscape/scene.osn
cargo run --release -- run examples/landscape/scene.osn --window 1920x1080 --out frame.ppm
cargo run --release -- replay examples/landscape/tests/landscape.replay
```

## What makes it hold a frame

The noise is the whole cost of a landscape like this: eight octaves of
it, hashed per octave, asked again at every step of every ray. Marching
it directly costs 43 ns a pixel — 24 ms at 960x540, and four times that
at a real window.

So the ground is **baked once into a Field** (`land.bake@1`, a
`Grid<Float>` of 2048 by 2048 over 2600 units of world) and the frame
marches that Field: four words and a blend per step instead of forty
integer multiplies. The bake does not change with the clock, so the
graph runs it at load and reuses it every frame after — `ran 1 reused 1`
on the Player's line. That one change is 4x, and it is the same trade a
renderer makes when it builds an acceleration structure: pay once for
the world, and let the frame cost the pixels.

A frame on this machine's GPU, least of three runs of a dozen frames:

| window | frame | fps |
| --- | --- | --- |
| 960x540 | 2.8 ms | 360 |
| 1280x720 | 4.5 ms | 220 |
| 1920x1080 | 5.5 ms | 180 |
| 2560x1440 | 10.1 ms | 99 |
| 3024x1964 | 16.5 ms | 61 |

The bake itself is 8.3 ms, once. On a Retina display `domain = window()`
asks for every physical pixel; `window() / scale()` renders at the
logical size and lets the Player scale it, which is a quarter of the
work and what a game would do.

The interpreter and the JIT run it too — so the replay checks the same pixels on all three backends. Its
`detail` parameter bakes a small Field, which is what makes that
affordable.

## How it is put together

- `land.bake@1` — the ground: a basin for the lake, hills around it,
  ridged octaves for the mountains beyond, and a canopy raised where the
  trees stand. Hashed with integers, which every backend computes the
  same way; a hash of a float's sine would drift between them.
- `land.view@1` — one ray per pixel. It walks the Field carrying the
  sample it took and the one before it, so a hit is a span to halve
  rather than a step to shade. The lake is a plane; where the ray meets
  it first, the water is what is seen and a second march along the
  reflected ray is what the water shows. The sun is a third march, short
  and coarse, for the shadow.

## What to change first

- `sun_turn` moves the sun around the valley; low and to the side is
  where the hills model best.
- `water` raises the lake. Drown the forest and the shoreline follows,
  since the trees are placed by height above the water.
- `span` and `detail` trade the Field's reach against its grain.
- The camera walks the near shore on the clock. A state node with the
  arrow keys would make it a place to walk in rather than a place to
  watch, which is the next thing this example wants.
