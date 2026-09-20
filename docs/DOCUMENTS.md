# Orla documents

Two file roles share one grammar, one lexer, and one literal syntax.

- **`.orla` is a Kernel library.** It holds `kernel` items and nothing else.
  A library answers "which capabilities exist".
- **`.osn` is a node graph.** The file is the graph: it uses libraries,
  declares its inputs and outputs, and places nodes. A node either references
  a published Kernel by id or carries its own Kernel inline, so one `.osn`
  can be complete on its own.

Both use `//` comments, `;`-terminated statements, and `{ }` blocks.
Diagnostics are `file:line:column: message` for both.

## Kernel

```text
struct Particle { position: Vec2; velocity: Vec2; }   // a document struct: every
                                                       // Kernel below sees it
type Swarm = Rows<Particle>;                           // a port type alias, shared
                                                       // the same way
kernel image.grayscale@1 {
    requires { field.read; field.write; }
    input source: Image;
    param amount: Float = 1.0;
    output result: Image;
    struct Hit { t: Float; id: Int; }
    fn luma(c: Vec4) -> Float { return c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722; }
    compute {
        let c = sample(source);
        let g = luma(c);
        return mix(c, vec4(g, g, g, c.a), amount);
    }
}
```

The id `name@version` is the Kernel's identity; graphs bind to it and never
to a file.

Every port carries a Field: elements of one type in one shape. The shapes
are the dense lattices a Kernel can be mapped over, and there are three:

| Shape | Address | `size(f)` | `domain =` in a graph |
|---|---|---|---|
| `Rows<T>` | `Int` | `Int` | `256` |
| `Grid<T>` | `Vec2` | `Vec2` | `vec2(w, h)` |
| `Volume<T>` | `Vec3` | `Vec3` | `vec3(w, h, d)` |

`Image` is the built-in alias for `Grid<Vec4>` and nothing more: a
`Grid<Vec4>` output shows in the window and reads as pixels wherever an
`Image` would. `type Name = Shape<Element>;` declares another alias; aliases
are transparent, so a `Rows<Vec2>` wire connects to a `type Stereo =
Rows<Vec2>;` port and any mismatch is reported by structure, never by name.

The element may be a scalar, a vector, or a struct; its layout is the
struct's scalars in order, integers and booleans exact. Inside `compute`,
Expressions are `let` bindings and a result; `+ - * /` and `%` (the
remainder, truncated like `/`, on `Int` and `Float` alike, and
componentwise on vectors) with `Int` and `Float` kept apart except that
`vec2`, `vec3`, and `vec4` take `Int` components and widen them; `==` and
`!=` compare scalars and whole vectors; `<` and the rest compare scalars;
`if c { a } else { b }` and `loop n with x = init until done { next }` are
expressions; `Name { a: .., b: .. }` builds a struct and
`Name { ..base, a: .. }` copies one with some fields replaced.

The math builtins are `floor`, `ceil`, `round`, `sign`, `fract`, `abs`,
`sqrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `exp`, `log`,
`pow`, `min`, `max`, `clamp`, `mix`, `step`, `smoothstep`, `dot`, `cross`,
`length`, `normalize`, `reflect`, and the conversions `float` and `int`.
The componentwise ones take a `Float` or any vector; where two operands
meet, a `Float` beside a vector broadcasts, so `clamp(v, 0.0, 1.0)` and
`mix(a, b, 0.5)` read as they would anywhere.

A `.orla` file may `use` other `.orla` files or directories for their
declarations: the structs and `type` aliases they declare at the top,
and what those files use in turn, come before the file's own, so a
layout declared once is the same layout wherever it is read. A scene's
`use` brings the same declarations to its inline Kernels. The same
declaration reached twice is one; two different declarations of one name
are refused with both files named.
`sample(f)` reads an input at this element's own address (its shape must
match the output's), `at(f, address)` reads anywhere (clamped into range;
`fetch` is the same word for grids), `at_linear(f, position)` reads
between elements, blending the neighbours (bilinear over a grid,
trilinear over a volume, a `Float` position over rows; element centres
sit at half-integers, edges clamp) and is allowed only where the layout
is all floats, whatever the spelling, `count(f)` is the element count,
`element()` the current linear index, and `coord()` and `domain()` the
current address and extent over a grid or volume. `time()` and `rate()`,
the session clock in seconds and the sound device's clock in samples per
second (zero without a device), need `time.read`; `pointer()`,
`pointer_down()`, `button(n)` (0 left, 1 right, 2 middle), `wheel()` (what
scrolled since the last evaluation, as a `Vec2`), `key(code)`, and the
window they arrive through, `window()` (its size in physical pixels as a
`Vec2`, zero by zero without one) and `scale()` (physical pixels per
logical pixel, one without a window), need `input.read`; a bar laid out
in units of `scale()` keeps its size on screen whatever the window's
size or density. Reading a port needs
`field.read`, writing `field.write`. A document's `input name: Type;` is a
Field the caller supplies; `input atlas: Image = "atlas.ppm";` names a file
the Host reads for it, relative to the document, and reads again whenever
the file changes: a picture as `.png` or `.ppm` (as pixels, or as a
`Grid<Float>` of luminance when the input says `Grid<Float>`), a
`Grid<Float>` as `.pgm`, a sound as `.wav` (rows of one or two floats; the
rate is reported, not stored, and a Kernel resamples with `at_linear`), and
any Field as `.field`, the Host's own format. `run --save` and the
protocol's `read` write all of them. `input title: Rows<Int> = text
"ORLA";` holds text: the session supplies the characters' codes as rows
and writes them again when the document's text changes, so a label is a
string in the scene rather than a Kernel of cases. A struct, `type`, or `fn` declared at
the top of a document is shared by every Kernel in it, and by every Kernel
of a document that `use`s it; a function calls only what is declared above
it. A function may take a Field: `fn cast(voxels: Volume<Int>, eye: Vec3)
-> Hit` reads its parameter with `at`, `at_linear`, `size`, or `count`, and
a call names an input port for it, `cast(world, eye)`; the function is
instantiated once per port it is called with, checked by shape and
layout. `sample` stays in `compute`, where the element's own address is.

A Kernel with a `merge` is a reduction, the one execution mode besides
the map:

```text
kernel stats.luma@1 {
    requires { field.read; field.write; }
    input source: Image;
    output result: Rows<Light>;          // exactly one element comes out
    struct Light { sum: Float; max: Float; count: Int; }
    compute {                            // once per element of `source`
        let l = luma(sample(source));
        return Light { sum: l, max: l, count: 1 };
    }
    merge(a, b) {                        // folds two results into one
        return Light { sum: a.sum + b.sum, max: max(a.max, b.max), count: a.count + b.count };
    }
}
```

`compute` maps over the first input's domain (so `coord()`, `sample`, and
`domain()` refer to it) and `merge` folds the results pairwise into the
single element of a `Rows` output. `merge` must be associative and
commutative: the fold's tree belongs to the backend, integers and
booleans come out exact, floats agree to rounding. A reduce node takes no
`domain` and is never fused.

A Kernel with `scan` in place of `compute` is ordered work, the third
execution mode:

```text
kernel audio.lowpass@1 {
    requires { field.read; field.write; }
    input samples: Rows<Float>;
    param k: Float = 0.1;
    output result: Rows<Float>;        // one element per input element
    scan with y = 0.0 {                // the carried value, and its start
        return y + (sample(samples) - y) * k;
    }
}
```

The body runs once per element of the first input, in order, exactly as a
`loop` body runs once per iteration: `y` is the previous step's result
(`init` before the first), the result becomes this element of the output
and the next step's `y`, and `index()` is the element index. A scan writes
as many elements as its input has, so a node's `domain`, if written, must
say so. Ordered work runs on one thread on either backend; that is its
nature, not a limit of the backend, and a scan is never fused.

A Kernel with `scatter` in place of `compute` writes where it computes,
the fourth and last execution mode:

```text
kernel particles.splat@1 {
    requires { field.read; field.write; }
    input particles: Rows<Particle>;
    input canvas: Grid<Vec3>;
    output result: Grid<Vec3>;         // starts as `canvas`
    struct Splat { cell: Vec2; heat: Vec3; }
    scatter into canvas with sum {     // once per element of `particles`
        let p = sample(particles);
        return Splat { cell: floor(p.position * size(canvas)), heat: tint(p) };
    }
}
```

The body runs once per element of the first input and yields a struct of
two fields: an address into the target named after `into` (an `Int` for
rows, a `Vec2` for a grid, a `Vec3` for a volume, floored like `at`) and a
value of the target's element type. The output is the target with every
value combined into the element at its address by the rule after `with`:
`sum` adds fieldwise, `min` and `max` take the fieldwise extreme, `last`
replaces. The rule of order is total and the same on both backends, floats
included: the result is as if the sources were applied one after another
in element order, so `last` is the greatest element index and a float
`sum` rounds in that order. `min` and `max` order floats by one total
order (every negative below every positive, negative zero below positive
zero, NaN beyond the infinities), so they, an integer `sum`, and `last`
do not depend on the order at all, and the GPU applies them in parallel
with atomics; a float `sum` the GPU sorts by target and source and folds
per target in source order, the same sequence the CPU adds in. An
address outside the target is dropped, never clamped; NaN is outside. `sum`, `min`, and `max` need an element of
numbers (a Bool takes `last`). The output has the target's shape, layout,
and extent, so a scatter node takes no `domain`, and a scatter is never
fused. The exact language is defined by [the frontend](../crates/language/src/lib.rs)
and its tests.

## Graph

```text
use "kernels/";                       // every .orla directly in the directory
use "../shared/tonemap.orla";         // one library file

input photo: Image;                   // supplied by the caller
param amount: Float = 1.0;            // a graph parameter, overridable by the caller

node gray: image.grayscale@1 {        // reference a published Kernel
    source <- photo;                  // wire: this evaluation's value
    amount = amount;                  // value: a literal, or a graph param by name
}

node trail: kernel {                  // an anonymous Kernel, placed here
    requires { field.read; field.write; }
    input current: Image <- gray.result;                     // wired where declared
    input previous: Image <~ trail.result from gray.result;  // previous evaluation
    param decay: Float = 0.9;         // the default is this node's value
    output result: Image;
    compute { return sample(current) + sample(previous) * decay; }
}

node glow: kernel fx.glow@1 { ... }   // a named inline Kernel: placed here and
                                      // also publishable to other nodes by id

node display: graph "../shared/display.osn" {   // another graph placed as a node
    source <- glow.result;            // wires its `input`s
    exposure = 1.2;                   // sets its `param`s
}

output result <- display.result;      // named graph output; sub-graph outputs
                                      // are read as node.output
```

### Rules

1. Wiring is written only at the consumer. `port <- source` reads the
   source in this evaluation and is a dependency. `port <~ source from
   initial` reads the source as committed by the previous successful
   evaluation, is not a dependency, and reads `initial` until something has
   been committed.
2. A source is a graph input by name or `node.port`. Inputs and nodes share
   one namespace. A graph may declare several outputs; which device takes
   which is decided by structure (a grid of pixels for the window, rows of
   one or two floats for the sound card), never by an output's name.
3. Every declared input port must be wired; a port the Kernel does not
   declare is an error; both are reported before anything runs.
4. Values use `=`: numbers, `true`/`false`, negation, `vec2`/`vec3`/`vec4`.
   An integer literal is accepted where a Float is expected.
5. `domain = count;`, `domain = vec2(width, height);`, or
   `domain = vec3(width, height, depth);` sets a node's output extent in
   its output's shape; nodes with an input default to the extent of their
   first declared input. The extent may also be an expression over what
   comes in: `size(port)` (the extent a port is bound to, as a count, a
   pair, or a triple), `count(port)`, the node's parameters and the
   graph's by name, numbers, `vec2`/`vec3`, `+ - * /`, `max`, `min`,
   `floor`, `ceil`, and the window as context: `window()` (its size in
   pixels) and `scale()` (its scale factor); it resolves when the node
   binds, and its sides must come out positive and whole. `domain =
   size(map) * tile;` draws a map at its tile size; `domain = 4 *
   count(faces);` sizes a subdivision; `domain = window();` fills the
   window at the window's resolution, and the node runs again at the new
   extent when the window is resized. The Player provides the window;
   `run` takes `--window <w>x<h>[@<scale>]` and a replay says `window w h
   [scale]`, so the same scene runs and replays without one.
   A binding must match the port's shape and element layout, whatever
   either side calls it.
6. A Kernel id is unique across everything a graph loads; the same id from
   two documents is an error, never a silent override. An anonymous inline
   Kernel is named `<document>.<node>@1`.
7. A `use` path is relative to the document; a trailing `/` means the
   directory's `.orla` files. Used documents must hold Kernels only.
   A graph's `input` may be declared with any port type; a parent graph
   wires it like a node port, and what arrives is checked by shape and
   layout where it binds. Only images can be supplied from files.
8. A node may bind a native operation of the backend by id, exactly as it
   binds a Kernel: `node raster: native.rasterize@1 { ... }`. Native
   operations are published by the runtime itself, cannot be republished
   from a document, and are never fused. Their contract is their manifest:
   ports by shape and layout, so any Kernel whose element lays out the
   same way connects. `native.rasterize@1` takes `vertices` (rows of
   `clip: Vec4; normal: Vec3; uv: Vec2`) and `triangles` (rows of three
   `Int`s) and writes `surface`, a grid of `depth: Float; normal: Vec3;
   uv: Vec2; id: Int` over the node's `domain`.
9. The graph exposes `input`, `param`, and `output` like a Kernel does, so
   a graph is placed as a node with `node name: graph "path.osn" { ... }`.
   Placement flattens it: its nodes join the parent under the prefix
   `name.`, its inputs take the parent's wires, its params take the parent's
   values (or its own defaults), and `name.output` reads its outputs. An
   inline Kernel's `param x: Float = amount;` takes the graph param `amount`
   as its value. The caller overrides a graph's params at evaluation; nested
   graphs cannot place themselves.
10. A node placed more than once is written once: `times = N;` on a
   Kernel node or a placed graph stands for `name/1` to `name/N`, each
   wired as written, and `carry port <- output;` feeds the port from the
   previous time's output from the second time on, so a relaxation, a
   Jacobi solve, or a subdivision is one node with a count. `name.port`
   anywhere means the last time's. `N` is a literal or a graph param; a
   param that sizes the graph lowers the document again when it changes,
   so `param level 3` over the protocol resizes a running scene.

### Loading

The session parses the graph, publishes its inline Kernels, follows its
`use` items, and rebuilds the Graph model. Changed documents republish only
the Kernels whose text changed; a rejected replacement keeps the running
version; a Kernel that leaves its document retires.

## Replay

Application tests are `.replay` files: `scene <path>`, optional
`library <path>`, then `frame`, `pointer`, `key`, `reset`, and `expect`
lines. See [the replay module](../crates/cli/src/replay.rs).
