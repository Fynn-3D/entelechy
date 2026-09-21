# Million

A million bodies, stepped and drawn every frame. Each has a place, a
speed and a tint in a `Rows<Body>` carried frame to frame with `<~`;
each is pulled towards the middle by an inverse square, which the
inside answers faster than the outside, so what starts as a disc winds
into rings. One that leaves the picture is made again on the disc.

```text
spawn ─ bodies (a map: a million independent sums)
          └─ splat (a scatter: a million writers, one count each)
               └─ draw (the counts, as their logarithm)
```

Nothing here reads another body, so the step is a map and the three
backends run it the way they run any map. The one place a million
writers meet is the scatter, and what they write is a count.

## What it costs

A frame at 960x540, measured with `orla bench --frames 15 --repeat 3`,
the least of the runs, on an M-series laptop:

| bodies | interpreter | JIT | GPU |
| --- | --- | --- | --- |
| 100,000 | 19.8 ms | 3.9 ms | 0.88 ms |
| 1,000,000 | 52.9 ms | 12.7 ms | 2.5 ms |
| 4,000,000 | 157 ms | 42.0 ms | 9.7 ms |

A million bodies is 400 frames a second on the GPU and 79 on ten CPU
cores through the JIT; four million is 103 and 24. Of the GPU's 2.5 ms,
the step is 1.5, the drawing 0.6 and the scatter the rest.

```sh
cargo run --release -- bench examples/million/scene.osn --window 960x540 --backend all --nodes
cargo run --release -p orla-player -- examples/million/scene.osn
cargo run --release -- run examples/million/scene.osn --param count=4000000 --window 960x540 --backend gpu --out four.ppm
```

## A count is an Int

The scatter sums, and what it sums used to be a `Float`. Floats do not
add the same way in every order, so to give the same word every time a
GPU has to sort a million pairs by the cell they land on and add them
up in element order: 24 ms of a 28 ms frame, which made the GPU slower
than the JIT. Integers do add the same way in every order, so a count
that is an `Int` is one atomic each and the same 24 ms became less than
half of one. The picture is identical, and so are the words.

That is the whole lesson of this example: the rule you write decides
whether a million writers need an order between them.

## Beside other engines

`bevymark` is the usual thing to hold a number like this against, and
it is not the same measurement: it draws textured sprites through a
renderer, which is more work per entity than adding one to a cell, and
its entities are rows of an ECS rather than elements of a Field. What
this example says is what Orla's own parts cost: a map over a million
structs, a scatter of a million writes, and a picture, each measured on
its own by `--nodes`.
