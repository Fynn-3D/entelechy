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
`sample(f)` reads an input at this element's own address (its shape must
match the output's), `at(f, address)` reads anywhere (clamped into range;
`fetch` is the same word for grids), `count(f)` is the element count,
`element()` the current linear index, and `coord()` and `domain()` the
current address and extent over a grid or volume. Reading needs
`field.read`, writing `field.write`. A struct or `type` declared at the top
of a document is shared by every Kernel in it. The exact language is
defined by [the frontend](../crates/language/src/lib.rs) and its tests.

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
   one namespace.
3. Every declared input port must be wired; a port the Kernel does not
   declare is an error; both are reported before anything runs.
4. Values use `=`: numbers, `true`/`false`, negation, `vec2`/`vec3`/`vec4`.
   An integer literal is accepted where a Float is expected.
5. `domain = count;`, `domain = vec2(width, height);`, or
   `domain = vec3(width, height, depth);` sets a node's output extent in
   its output's shape; nodes with an input default to the extent of their
   first declared input. A binding must match the port's shape and element
   layout, whatever either side calls it.
6. A Kernel id is unique across everything a graph loads; the same id from
   two documents is an error, never a silent override. An anonymous inline
   Kernel is named `<document>.<node>@1`.
7. A `use` path is relative to the document; a trailing `/` means the
   directory's `.orla` files. Used documents must hold Kernels only.
8. The graph exposes `input`, `param`, and `output` like a Kernel does, so
   a graph is placed as a node with `node name: graph "path.osn" { ... }`.
   Placement flattens it: its nodes join the parent under the prefix
   `name.`, its inputs take the parent's wires, its params take the parent's
   values (or its own defaults), and `name.output` reads its outputs. An
   inline Kernel's `param x: Float = amount;` takes the graph param `amount`
   as its value. The caller overrides a graph's params at evaluation; nested
   graphs cannot place themselves.

### Loading

The session parses the graph, publishes its inline Kernels, follows its
`use` items, and rebuilds the Graph model. Changed documents republish only
the Kernels whose text changed; a rejected replacement keeps the running
version; a Kernel that leaves its document retires.

## Replay

Application tests are `.replay` files: `scene <path>`, optional
`library <path>`, then `frame`, `pointer`, `key`, `reset`, and `expect`
lines. See [the replay module](../crates/cli/src/replay.rs).
