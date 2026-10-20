# Entelechy documents

**Answers:** how a Kernel and a scene are written.

Every type, builtin and form, at once: [REFERENCE](REFERENCE.md).

Two file roles share one grammar, one lexer, and one literal syntax.

- **`.ent` is a Kernel library.** It holds `kernel` items and the
  declarations they share.
- **`.esn` is a node graph.** The file is the graph: it uses libraries,
  declares its inputs and outputs, and places nodes. A node references a
  published Kernel by id or carries its own Kernel inline, so one `.esn`
  can be complete on its own.

Both use `//` comments, `;`-terminated statements, and `{ }` blocks.
Diagnostics are `file:line:column: message`. The exact language is
defined by [the frontend](../crates/frontend/language/src/lib.rs) and its
tests; `entelechy help <name>` shows any builtin's or form's signatures.

Whole scenes are in [`examples/`](../examples/README.md); the fragments
below explain.

### Docs

A capability says what it is for in its source, and `kernel` and
`kernels` answer a published Kernel's:

```text
//! A preset graph's or a package manifest's doc: the lines a document
//! opens with.
use "kernels/";

/// A Kernel's doc: the lines directly above it.
kernel tally.twice@1 {
    requires { field.read; field.write; }
    /// A port's or a param's doc, directly above it.
    input counts: Rows<Int>;
    output result: Rows<Int>;
    compute { return sample(counts) * 2; }
}
```

- `///` lines directly above a `kernel`, `input`, `param`, `output` or
  `memory` are its doc; a blank line or a `//` line ends them. `//!`
  lines a document opens with say what the document is for: a preset
  graph, a package.
- Lines join into a paragraph, an empty `///` parts paragraphs, and an
  indented line is code and keeps its own.
- Both are comments to the grammar: a document compiles the same with or
  without them. A Kernel's doc lines travel with its source, through
  `publish`, `kernel` and `snapshot`; changing them publishes it again.

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

The id `name@version` is the Kernel's identity; graphs bind to it, never
to a file.

### Ports and shapes

Every port carries a Field: elements of one type in one shape. A Kernel
declares any number of ports; a GPU backend refuses a Kernel whose
inputs and outputs exceed what its adapter binds at once, and the
refusal says the numbers. There are three shapes:

| Shape | Address | `size(f)` | `domain =` in a graph |
|---|---|---|---|
| `Rows<T>` | `Int` | `Int` | `256` |
| `Grid<T>` | `Vec2` | `Vec2` | `vec2(w, h)` |
| `Volume<T>` | `Vec3` | `Vec3` | `vec3(w, h, d)` |

`Image` is the built-in alias for `Grid<Vec4>`: a `Grid<Vec4>` output
shows in the window and reads as pixels wherever an `Image` would.
`type Name = Shape<Element>;` declares an alias. Aliases are
transparent: a `Rows<Vec2>` wire connects to a `type Stereo =
Rows<Vec2>;` port, and a mismatch is reported by structure, never by
name.

The element is a scalar, a vector, a struct or an array; its layout is
its scalars in order, integers and booleans exact.

### Narrow encodings

After `as`, a port keeps its element's scalars narrower than a word:
`Grid<Vec4 as unorm8>` is four bytes a pixel, `Rows<Int as u8>` a byte a
row. The Kernel still computes in `Float` and `Int`; the Field widens a
scalar when read and narrows it when written.

| Encoding | Keeps | Narrowed by |
|---|---|---|
| `f16` | a float in half precision | to nearest, ties to even; too large is infinity |
| `unorm8`, `unorm16` | a float in `0..=1` in 8 or 16 bits | clamped, NaN as 0, scaled and rounded to nearest, ties to even |
| `snorm8` | a float in `-1..=1` in a signed byte | the same, and -128 reads as -1 |
| `u8`, `i8`, `u16`, `i16` | an integer or a boolean | its low bits; read back zero- or sign-extended, a boolean as whether any bit is set |

The rule is the Host's; every backend keeps it bit for bit. A picture
file or `bytes` input declared narrow lands narrow. A narrow and a wide
Field are different layouts; a wire between them is refused. Only a map
or a scan's steps write a narrow output (a scan's memory may be narrow);
a reduction, a prefix and a scatter fold into their output in whole
words. A narrow Field between two nodes is never fused away. A 64-bit
scalar is never narrow.

### Expressions

Inside `compute`, a body is `let` bindings and a result.

- **Bindings.** `let n = ..;`; `let { x, y } = window();` binds fields by
  name; `let { x: width, y: height } = window();` renames them. A binding
  named `_` binds nothing. `expr;` is a statement for its effect, the same
  as `let _ = expr;`.
- **Arithmetic.** `+ - * /` and `%` (remainder truncated like `/`, on
  `Int` and `Float`, componentwise on vectors). `Int` and `Float` are kept
  apart, except that `vec2`, `vec3`, `vec4` take `Int` components and
  widen them. Constructors take shorter vectors among their components
  (`vec4(v3, 1.0)`, `vec3(1.0, v2)`), as `ivec2` and `ivec3` take integer
  vectors. A `Float` beside a vector broadcasts, so `clamp(v, 0.0, 1.0)`
  and `mix(a, b, 0.5)` work.
- **Swizzles.** `v.xy`, `v.zyx`, `c.rgb`: components in that order from
  one alphabet; `v.x` one component.
- **Bits.** `& | ^` and `<< >> >>>` on `Int`s (count modulo 32, `>>`
  arithmetic, `>>>` logical), binding between comparison and arithmetic
  as in Rust. `0xFF` is hexadecimal, its 32 bits as written; more than
  eight digits, or an `l`, make a `Long`. A float may carry an exponent,
  `1e30` or `2.5e-3`, and a `Double` its `d` after it (`3e2d`).
- **Logic.** `&&` and `||` short-circuit: `n != 0 && x / n > 1` does not
  divide when `n` is zero. `==` and `!=` compare scalars and whole
  vectors; `<` and the rest compare scalars.
- **`if`.** `if c { a } else { b }` is an expression. Among a block's
  statements, `if c { return a; }` with no `else` guards what follows.
  An `if` whose arm is statements alone, `if hit { emit(p); }`, runs them
  and the block goes on; its `else` or `else if` is statements too, and
  `{ }` does nothing. It equals `let _ = if hit { emit(p); 0 } else { 0 };`,
  so an arm giving a value beside one that does not is refused, and so
  is a guard inside such an arm (`if a { if b { return x; } }`), which
  would return on some paths and go on past the arm on others: join the
  conditions, `if a && b { return x; }`, or give the inner `if` an `else`.
- **`match`.** `match i { 0 => a, 1 | 2 => { let x = ..; x }, 8..16 => c,
  16..=31 => d, _ => b }` on an `Int`. The `_` arm comes last;
  a range `..` stops before its end, `..=` takes it, at most 256 values;
  no value in two arms; every arm the same type. Arms that name every
  variant of an enum need no `_`; a value that is no variant takes the
  last arm.
- **Enums.** `type Mode = enum { Idle, Walk, Attack = 10, Guard };` names
  `Int`s: a variant is the number written after it, else one past the
  variant before it, the first 0 (`Mode.Guard` is 11), and no two are one
  number. `Mode.Walk` is its `Int`; `Mode` is a type a value, a field, a
  port's element and a parameter may have, and it is an `Int` there; a
  scene writes a variant where it writes a number, a node's param or its
  own. A
  `match` naming variants names them all or ends with `_`. A variant whose
  number is kept, in a memory or a file, is written with it, so a variant
  written before it does not move it.
- **`loop`.** `loop n with x = init until done { next }` is an
  expression. A nested loop reads its enclosing loops' carried values,
  and a scan's step its own, by name. The count is a literal (its bound)
  or any `Int` expression, read once and clamped to `0..=65536`, so
  `loop depth with ...` takes a parameter. A count of zero yields the
  carried value. `if done { break x; }` among a loop body's statements
  stops the innermost loop after that iteration with that value, of the
  carried type.
- **Structs.** `Name { a: .., b: .. }` builds one; `Name { ..base, a: .. }`
  copies one with fields replaced.
- **Nesting.** An expression or block nests at most 256 deep, counting
  what encloses it and operators chained before it (`a + b + c` is three
  deep at `c`); past that the document is refused, saying where; split a
  chain with `let` ([`MAX_NESTING`](../crates/frontend/language/src/lib.rs)).

### Arrays

`[T; N]` holds `N` values of one value type: `[Int; 16]`, `[Vec3; 4]`,
`[Hit; 8]`, `[[Int; 4]; 4]`. An array may be a struct field, a function
argument or result, a loop's carried value or a Field's element. `[a, b,
c]` builds one, `[v; 8]` eight copies, `a[i]` takes an element: a literal
index is checked at compile time, a computed `Int` is clamped into the
array. `[each N { body }]` builds one element by element, the body run
once per element with `index()` its position (a loop inside has its own `index()`):
`[each 32 { unit(w, x, index()) }]`, or `[each 4 { if index() == k { v }
else { a[index()] } }]` for `a` with element `k` replaced. An array has
1 to 4096 elements, laid out in order. A value of any type holds at most
65536 words and structs, and structs and arrays nest at most 64 deep.

### Parameters

A parameter is any value a Kernel holds, crossing as its words:
`param lens: Mat4 = mat4(1.0);`, `param cell: IVec2 = ivec2(3, 4);`,
`param edges: [IVec2; 12] = [..];`, `param style: Style = Style {
ground: vec3(0.1, 0.1, 0.1), width: 1.0 };`. A default is a literal:
numbers and booleans, `vec2`..`vec4`, `ivec2`, `ivec3`, `mat2`..`mat4`
(from columns, or one value down the diagonal), `[a, b]`, `[v; N]`, and
a struct literal naming every field.

A value matches by layout, never spelling: an integer stands in for a
float, and wherever a value is given from outside (a scene, a replay's
`param`, the protocol) an array of values in order takes the
parameter's shape: `[vec3(0.9, 0.9, 0.9), vec3(0.8, 0.1, 0.1), 2.0]` is a
`Style`, `[[1, 0], [0, 1]]` a `Mat2`. A scene may name a struct its
documents declare at the top; a struct declared inside a Kernel is that
Kernel's alone and is given by position.

### Matrices

`Mat2`, `Mat3`, `Mat4` are square, of floats, stored by columns.
`mat3(c0, c1, c2)` builds from columns, `mat3(1.0)` puts a value down the
diagonal. `m[i]` is a column and `v[i]` a component (literal index
checked, computed index clamped). `m * v` applies to a column vector,
`v * m` to a row vector, `m * m` composes, `m * f` and `f * m` scale,
`+` and `-` take two of one order, `-m` negates, `==` compares whole.
`transpose`, `determinant`, `inverse` (a singular matrix's inverse holds
no finite numbers). Products and these builtins, like `dot`, `cross`,
`length`, `normalize` and `reflect`, compile to one formula in one
order: the interpreter and the JIT agree bit for bit.

### Integer vectors

`IVec2` and `IVec3` are cells and offsets: `ivec2(x, y)` from `Int`s,
`icoord()` this element's cell, `isize(f)` a Field's extent, `idomain()`
the output's. They take `+ - * / %`, `& | ^` and shifts componentwise,
with each other or an `Int` on either side (failing where an `Int`
fails), `-v`, `==`, `.x .y .z`, `v[i]`, `abs`, `sign`, `min`, `max`,
`clamp` (with `Int` bounds too). `float(v)` widens to `Vec2`/`Vec3`,
`int(v)` truncates to one. `at`, `fetch` and `put` take an integer vector
where they take a `Vec2` or `Vec3`, and a scatter's address may be one,
with the same Field rules (clamped reads, dropped writes).

### 64-bit numbers

`Double` and `Long`, with `DVec2`..`DVec4`, `LVec2`, `LVec3`,
`DMat2`..`DMat4`. `dvec3(..)`, `lvec2(..)`, `dmat4(..)` build as their
32-bit kin do, widening numbers of any kind and shorter vectors.
`double(x)` and `long(x)` convert; `float(x)` and `int(x)` convert back
(a long to an int keeps its low 32 bits). Each gives a value already of
its kind as it is, and fixes a written number at that width
(`float(0.1)` is a `Float` beside a `Double`). A `Double` never meets a
`Float`, nor a `Long` an `Int`, without a conversion; `x + f` of the two
is refused, saying which to write.

A written number is exact until it meets a width: arithmetic between
written numbers is exact (`0.1 + 0.2`, `1.0 / 3.0`, `2 * 1000000007`)
and stays a written number. Beside a `Double` a written `0.1` is the
double nearest a tenth; beside a `Long` a written integer is a long;
beside a `Float` a written integer is a float; an integer too big for an
`Int` is a `Long`. Division of written whole numbers is whole (`7 / 2` is
`3`); one leaving a remainder beside a float is refused (`x * (1 / 2)`
says to write `1.0 / 2`). A vector, matrix or array of written numbers
alone takes the width it meets as a whole (given to a `DVec3` or `[Double;
2]`, beside one in an operation, among `if` branches or array elements).
With no width named, a written float is a `Float` and a written integer
an `Int`; `0.1d` and `8l` name it. A `let` keeps the width it was named
with. A written number past the largest `Float` that comes to be one
(`return 1e39;` into a `Float`, `1e20 * 1e20`) is refused, saying to
write it as a `Double`; one past the largest `Double` is refused as it
is read.

Each 64-bit scalar is two words, low first, in a Field, a parameter and
the protocol (an element of `Rows<DVec3>` is six words). Doubles take
every math builtin, `dot`, `cross`, `length`, `normalize`, the matrix
products, `inverse` and the rest; longs take arithmetic (overflow and
zero divisor fail as for `Int`), bit operators, shifts (count modulo
64), `abs`, `sign`, `min`, `max`, `clamp`, `add_wrap`, `sub_wrap`,
`mul_wrap`. The interpreter and the JIT compute as Rust's `f64` and
`i64`. A GPU computes a `Long` in a map, a reduction, and a scatter
whose target is all `Long`s under `min` or `max`: as the device's own
64-bit integer where it has one, and else (a page's WebGPU) as a pair
of 32-bit words, scalars alone; it never computes a `Double`. A GPU
session runs any other node holding them on the JIT
([`examples/data`](../examples/data/README.md)).

### Builtins

Math: `floor`, `ceil`, `round`, `trunc`, `sign`, `fract`, `abs`, `sqrt`,
`inversesqrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`,
`sinh`, `cosh`, `tanh`, `exp`, `exp2`, `log`, `log2`, `pow`, `fma`,
`min`, `max`, `clamp`, `saturate`, `mix`, `step`, `smoothstep`, `mod`
(remainder with the divisor's sign; `%` truncates), `radians`,
`degrees`, `isnan`, `isinf`, `dot`, `cross`, `length`, `distance`,
`normalize`, `reflect`, `refract`, and the conversions `float`, `int`,
`double`, `long`. The componentwise ones take a `Float` or any vector.

On `Int`s: `add_wrap`, `sub_wrap`, `mul_wrap` (modulo 2^32, where `+ -
*` fail on overflow); `popcount`, `clz`, `ctz`, `rotl`, `rotr`, `ult`,
`udiv`, `urem`, `mulhi`, `umulhi` (bit counts, rotations, unsigned
compare and division, the high half of a 64-bit product). `bits` and
`from_bits` carry a Float's 32 bits to an Int and back unchanged; the
Ints of non-negative Floats order as the Floats do.

Float `min`/`max` follow one rule on every backend
([decision](DECISIONS.md)).

Fields and context:

- `sample(f)` reads an input at this element's own address (its shape
  must match the output's). `sample` is only in `compute`.
- `at(f, address)` reads anywhere, clamped (`fetch` is the same for
  grids). `at_linear(f, position)` blends neighbours (bilinear over a
  grid, trilinear over a volume, a `Float` position over rows; centres at
  half-integers, edges clamp), only where the layout is all floats.
- `count(f)` the element count, `size(f)` the extent, `element()` the
  linear index, `coord()` and `domain()` the address and extent over a
  grid or volume.
- `clock()` (the session's time in seconds, a `Double`, whole), `time()`
  (the same clock rounded to a `Float`), `rate()` (sound device clock,
  samples per second, zero without a device) and `queued()` (the samples
  per channel the sound device holds and has not yet played, an `Int`,
  zero without one: a sound made to the device keeps it that far ahead)
  need `time.read`. A Kernel
  meant to run for days times with `clock()`
  ([why](DECISIONS.md#the-clock-crosses-whole)).
- `pointer()`, `pointer_down()`, `button(n)` (0 left, 1 right, 2
  middle), `wheel()` (scroll since the last evaluation, `Vec2`),
  `key(code)`, `typed(k)` (the `k`th character typed since the last
  evaluation as a code point, 0 after the last: what keys or an input
  method wrote, while `key` says which keys are held), `window()`
  (physical pixels, `Vec2`, zero by zero without one) and `scale()`
  (physical per logical pixel, one without a window) need `input.read`. A bar laid out in units of `scale()` keeps its
  on-screen size.
- Reading a port needs `field.read`, writing `field.write`.

A function may take a math builtin's name (`step`, `mix`, `abs`) and
then stands for it in its document; the Field, memory, context and
construction builtins (`at`, `put`, `coord`, `vec2` and the rest) keep
their names.

### Failures

Integer arithmetic without a value (overflow of `+ - *`, division or
remainder by zero, the least integer negated, `udiv` or `urem` by zero)
fails on every backend, naming the first element in row-major order that
has it and the first failure within it; nothing the evaluation computed
commits. A device records the failure and the evaluation waits for it
once, before commit, only when a node that ran can fail.

Every backend agrees, and the feature matrix holds it: a positive
divided by zero is infinity, zero by zero is NaN, and nothing equals or
orders against NaN. A NaN read from a Field is seen alike everywhere; a
NaN made by arithmetic on the GPU (`inf * 0`, `sqrt` of a negative) may
come out a number under a driver's fast math.

An integer `sum` a scatter or prefix folds is each element's whole
total, independent of order ([decision](DECISIONS.md)).

Work is bounded in loop iterations ([decision](DECISIONS.md)):

- **Budget.** The iterations one element may run, all loops together (a
  scan step and a reduction merge are each an element): 16777216 unless
  the Kernel declares `budget N;` among its items, 1 to 2^30.
- **Host bound.** One run, all elements together: four thousand million
  unless the session is given `--work <M>` (millions), beside `--memory`.
- A loop counts the iterations whose body ran (not the one its `until`
  stops) as it finishes. An element checks its count when it finishes,
  and at each iteration of a loop that can run more than 2^20 by itself:
  past its budget it fails the evaluation, named. A run fails, without
  naming an element, when its elements' exact sum passes the Host's
  bound; a run past it with a failing element may be named for either.
  The running system keeps its last state.
- A Kernel whose elements cannot run 1024 iterations or pass their
  budget (loop counts all literals multiplying to less) is never counted.

### Declarations, `use` and functions

A struct, `type` or `fn` at the top of a document is shared by every
Kernel in it and in every document that `use`s it. A `.ent` file may
`use` other `.ent` files or directories: their top-level declarations
come before the file's own. A scene's `use` brings them to its inline
Kernels. A document sees what it uses itself: the declarations of the
documents its `use` items name (each file of a directory, each file a
package's manifest uses), not what those use in turn, whose functions
their own code still calls. A function, a struct and a `type` are
scoped by their document: a name reaches the Kernel's own declaration
of it, then the document's own, or else the one declaration of it in
the documents it uses, so two libraries may each keep a helper or a
struct of one name, and an application or a Kernel may name its own as
a package it uses does, the package's code still reaching the
package's. A name a document reaches only through another's `use` is
said with the `use` that would bring it. The same struct or `type` declared alike in two documents is one.
A function may take a math builtin's name and stands for it in its own
document only: a document that uses it keeps the builtin.
A name two used documents answer differently is refused where it is
named, naming both. Two documents that `use` each other are refused by
name. A name a Kernel
cannot find is looked for beside it and beside the scene, and the
diagnostic names the declaring document and the `use` to add.

- A function's name is any word but `let`, `return`, `if`, `match`,
  `loop`; `fn scatter(...)` is a function.
- Functions call one another in any order; no chain of calls comes back
  to a function, so none recurses, and a cycle is refused naming it
  ([decision](DECISIONS.md)). Repetition is a `loop` carrying the work.
- A struct may name a struct declared after it; structs that contain one
  another are refused, naming them.
- A short function is inlined at compile time; a trace still shows the
  call.
- A function may take a Field: `fn cast(voxels: Volume<Int>, eye: Vec3)
  -> Hit` reads it with `at`, `at_linear`, `size`, `count`; a call names
  an input port, `cast(world, eye)`, and the function is instantiated
  once per port, checked by shape and layout.
- Type parameters: `fn twice<T>(v: T) -> T`, each bound by its argument's
  type and the same everywhere it appears; one instance is compiled per
  set of types, and an error names the instance.

### Inputs from files

A document's `input name: Type;` is a Field the caller supplies. With a
source, the Host reads it relative to the document and again whenever it
changes:

- `input atlas: Image = "atlas.ppm";`: `.png` or `.ppm` as pixels, or as
  luminance for `Grid<Float>`; `.pgm` as `Grid<Float>`; `.wav` as rows of
  one or two floats (the rate is reported, not stored; resample with
  `at_linear`); `.mid` as rows of `{start, duration, pitch, velocity,
  channel}`, ticks and tempo read as seconds; `.field`, any Field, the
  Host's format. A path through a package's name
  (`"text.entpkg/font.png"`) reads the file inside that package, as a
  `use` of it does. `run --save` and the protocol's `read` write all of them.
- `input skills: Rows<Skill> = "data/skills.csv";`: a data table, `.csv`
  or `.json`, read as rows of the element the Kernel reading the input
  lays out, and read again when that layout changes. Each number is named
  as `show` names it: a field by name, a nested one with a dot
  (`look.race`), a vector's lane `x` to `w` or `0` to `3`, an array's
  element by its index (`items.0`). A CSV's first line names its columns
  and each line after it is a row; a JSON file is an array of objects,
  nested objects and arrays naming their numbers as the dots do. Every
  number must be given and every column or key must name one, but for a
  note, a column or key whose name begins with `#` (`#name`). A row's
  `#name` names it to other tables: a whole-number cell written
  `table:name` (`beasts:goblin`) is the place, from 0, of the row so
  named in the table `table.csv` or `table.json` beside this one, read
  again when that table changes; a name no row has, or two share, is
  refused; a table
  that cannot be read says the line or row and leaves the input as it
  was. One number alone is unnamed: a CSV's one column, or a JSON array
  of numbers.
- `input title: Rows<Int> = text "ENTELECHY";`: the characters' codes, updated
  when the document's text changes. A string writes a new line, a tab, a
  quote and a backslash as `\n`, `\t`, `\"` and `\\`, so one string may
  hold several lines of text.
- `input level: Rows<Int> = bytes "level.bin";`: a file's bytes, one per
  element, 0 to 255 (`examples/level`).
- `input clip: Image = frames "clip/" at 30;`: a directory's files in
  name order (same formats), frame `floor(time * rate)` wrapping at the
  end, brought in each evaluation; listed again on sync.
- `input problems: Rows<Int> = checks "src/";`: what the language finds
  wrong in the documents of a file or a directory (every `.ent` and `.esn`
  under it), each checked as `entelechy check` checks it, as text, a line
  a problem: `file:line:column: message`, the file relative to the
  directory, a problem with no place at line and column 0, each once, in
  order; a newline alone when there is none. Checked again on sync when
  one of them changes, or which there are: while the session prepares in
  the background (the Player, a served session) a check after the first is
  made off the frames and taken in by the first sync after it is done,
  the last text held until then; a replay or a run takes each in at once.
- `input defined: Rows<Int> = outline "src/";`: what the documents of a
  file or a directory define (every `.ent` and `.esn` under it), as
  text, a line a definition: `file:line:column: kind name: signature`,
  the file relative to the directory, the column where the name is
  written; kinds `fn`, `kernel`, `struct`, `type`, `input`, `param`,
  `output` and `node`, the signature a function's arguments and result,
  a Kernel's ports and params, a struct's fields, an alias's or an
  input's or a param's type, a node's Kernel, graph or type (none for an
  output); in order of file and place. A document that does not parse
  defines nothing; a newline alone when nothing is defined. Read again on
  sync when one of them changes, or which there are, off the frames as a
  `checks` input is checked.
- `input disk: Rows<Int> = files "project/";`: every file under a
  directory (hidden ones and those under hidden directories left out),
  in path order, a word each count, path byte and file byte: `[n,
  (path_len, size, path.., bytes..)*]`, the path relative to the
  directory, parted by `/`. Read again on sync when one of them changes,
  or which there are.
- `input pasted: Rows<Int> = clipboard;`: the clipboard's text, a
  character (a code point) a row and a 0 after the last; a scene's own
  inputs only. In the Player the clipboard is the system's, read only
  with `--allow clipboard.read` (empty without), looked at again at most
  four times a second; elsewhere it is the session's own, which a
  replay's `clipboard "text"` fills.
- `input saved: Rows<Tally> = "saves/tally.field" else fresh.result;`
  reads `fresh.result` until the file exists (`examples/data/parts/save`).

### Execution modes

A Kernel runs in one of six modes. `emit`, `put` and several outputs
belong to the modes named below.

**Map** (`compute`): once per element of the output domain. A map may
write several outputs over one domain, sharing a shape:

```text
kernel shade.surface@1 {
    requires { field.write; }
    output albedo: Image;
    output normal: Grid<Vec3>;
    output depth: Grid<Float>;
    compute {
        let hit = cast(eye, ray());
        if hit.t < 0.0 {
            return { albedo: sky, normal: vec3(0.0, 0.0, 0.0), depth: 100.0 };
        }
        { albedo: vec4(hit.albedo, 1.0), normal: hit.normal, depth: hit.t }
    }
}
```

`{ albedo: .., normal: .., depth: .. }` gives every output once, in any
order; the record is a value (an `if` or `match` arm may give one, a
`let` may hold one, `r.depth` reads a field). A graph reads
`node.albedo`. Only a map, and a group, writes several outputs; other
modes give one struct. When one consumer is the only reader of every output of a map
and reads them all by `sample`, the two fuse and no output is written;
otherwise the map writes them all.

**Reduction** (`compute` and `merge`):

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

`compute` maps over the first input's domain (`coord()`, `sample`,
`domain()` refer to it); `merge` folds results pairwise into the single
element of a `Rows` output. `merge` must be associative and commutative:
the fold's tree is the backend's; integers and booleans are exact,
floats agree to rounding. `merge` may read inputs, params and context.
A reduce node takes no `domain` and is never fused.

**Scan** (`scan`): ordered work.

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

The body runs once per element of the first input, in order: `y` is the
previous result (`init` first), the result is this output element and
the next `y`, `index()` the element index. A scan writes as many
elements as its input; a written `domain` must agree. It runs on one
thread and is never fused.

A scan may own memory, a Field its steps read and write:

```text
kernel machine.step@1 {
    requires { field.read; field.write; }
    input ticks: Rows<Int>;            // one step per element
    memory ram: Rows<Int>;             // bound like an input, read like an output
    output result: Rows<Int>;
    scan with pc = 0 {
        let op = at(ram, pc);          // what an earlier step wrote is there
        put(ram, pc + 100, op * 2);    // a write, in step order
        return pc + 1;
    }
}
```

`memory name: Type;` is bound like an input (`ram <~ machine.ram from
zeros.result` persists across evaluations; `ram <- fresh.result` starts
each over) and read like an output (`machine.ram`, listed after the
Kernel's own output). Steps start from what was bound; `at` reads the
memory as left so far; `put(name, address, value)` writes an element
(address floored as `at` takes it, outside dropped, never clamped) and
yields the value. Only ordered work writes, so every backend applies
writes in one order; the bound Field is never touched, and a failed
evaluation leaves it as it was. A function may `put` into a Field
parameter (`fn store(m: Rows<Int>, a: Int, v: Int) -> Int { return
put(m, a, v); }`), as a step's write. The initial carried value may
read a memory with `at` (what was bound) but not write.

**Scatter** (`scatter`): writes where it computes.

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

- The body runs once per element of the first input, or of the node's
  `domain` when set (in the first input's shape; `domain = points *
  weights;` steps over every pair without a Field of that size).
- It yields a struct of two fields: an address into the target after
  `into` (`Int` for rows, `Vec2` grid, `Vec3` volume, floored like `at`)
  and a value of the target's element type.
- The output is the target with each value combined at its address by
  the rule after `with`: `sum` adds fieldwise, `min` and `max` take the
  fieldwise extreme, `last` replaces. The result equals applying sources
  in element order: `last` is the greatest element index, a float `sum`
  rounds in that order. `min`/`max` order floats totally (negatives below
  positives, -0 below +0, NaN beyond the infinities). A float `sum` on a
  GPU is sorted by target and source and folded in source order.
- An address outside the target, NaN included, is dropped, never
  clamped. `sum`, `min`, `max` need numbers (a Bool takes `last`).
- The output has the target's shape, layout and extent: no `domain`,
  never fused.
- `emit(pair);`, in the body and in no function it calls, possibly in a
  loop, gives one more pair. An element's pairs are those it emits, in
  order, then the one it returns; a body with nothing more returns an
  address outside the target. `emit` takes only `min`, `max`, or a `sum`
  of no floats. The interpreter and the JIT hold emitted pairs against
  the Host's budget; past it the run fails and commits nothing
  ([decision](DECISIONS.md)).

**Prefix** (`prefix`): each output element is its value combined with
every value before it.

```text
kernel compact.offsets@1 {
    requires { field.read; field.write; }
    input flags: Rows<Int>;
    output result: Rows<Int>;          // one element per input element
    prefix with sum {                  // the rank of each kept element
        return if sample(flags) != 0 { 1 } else { 0 };
    }
}
```

The body runs once per element of the first input, a `Rows`, reading as
a map does; the output `Rows` of the same count holds at `i` the values
at `0..=i` combined by `sum` (fieldwise), `min` or `max` (fieldwise,
floats in the scatter's total order). A prefix may fold in any grouping,
so `sum` over an element holding a float is refused (use `scan`), and so
is `last`. A device folds in passes over blocks; the interpreter and the
JIT walk in order, the reference grouping. An integer `sum` that
overflows fails where the reference walks it and is not reported on a
device; the words that come out are the same.

**Group** (`group N`): a map whose elements run `N` at a time (1 to 256)
and meet at each `exchange`.

```text
kernel fft.row@1 {
    requires { field.read; field.write; }
    input line: Rows<Vec2>;
    output result: Rows<Vec2>;
    group 64 {                          // one row of 64 per group
        let x = sample(line);
        let pair = exchange(x, lane() ^ 1);
        return if lane() % 2 == 0 { x + pair } else { pair - x };
    }
}
```

Element `e` is lane `e % N` (`lane()`) of group `e / N`; it reads and
writes its outputs as a map does. `exchange(v, k)` is
the `v` lane `k` holds at that same exchange, `k` wrapped into the group
(`lane() + 1` and `-1` go round). Every lane of a group reaches each
exchange alike, so one stands under no branch, `match`, loop count or
`until` that turns on what a lane holds alone: its lane, its element,
what it reads, or what an exchange handed it, and anything made from
those. What the whole group holds (params, literals, the domain) may
steer one: `loop rounds with s = x { s + exchange(s, lane() ^ (1 <<
index())) }` sums a group of eight in three rounds. `exchange` is
written in the body, never in a function. The last group runs whole
when the domain is not a multiple of `N`: its lanes past the end give at
exchanges, fail as any element fails (placed as the domain lays elements
out), and write nothing.

On a device a group is a workgroup and an exchange a pass through its
shared memory between barriers. The interpreter and the JIT run each
lane to its next exchange and run the group again from the start with
what every lane gave there, until all finish; only the finished run's
work counts ([decision](DECISIONS.md)). The GPU runs groups of one
output whose work is not counted; others run on the JIT.

## Graph

```text
use "kernels/";                       // every .ent directly in the directory
use "../shared/tonemap.ent";         // one library file

input photo: Image;                   // supplied by the caller
param amount: Float = 1.0;            // a graph parameter, overridable by the caller
require gentle: amount <= 4;          // what the parameters must hold to run

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

node display: graph "../shared/display.esn" {   // another graph placed as a node
    source <- glow.result;            // wires its `input`s
    exposure = 1.2;                   // sets its `param`s
}

output result <- display.result;      // named graph output; sub-graph outputs
                                      // are read as node.output
output kept <- state.result to "saves/state.field";   // also written to a file
```

A node's name may be any word, a keyword too (`node scan: …`, `… <-
scan.result`); a Kernel's own names (ports, parameters, values) cannot
be keywords.

### Rules

1. Wiring is written only at the consumer. `port <- source` reads the
   source in this evaluation and is a dependency. `port <~ source from
   initial` reads the source as committed by the previous successful
   evaluation, is not a dependency, and reads `initial` until something
   is committed. When a republished Kernel changes the port's layout,
   `port <~ source from initial migrate kernel.id@v` carries the state
   over: the named Kernel is a map over one input in the old layout,
   giving the new, run once before the nodes; the evaluation reports it
   as migrated. The migration is the state's: one of its `<~` readers
   naming it carries it for all of them, and two naming different ones
   are refused. Without one, the state restarts from `initial` and the
   evaluation reports the reset. A migration that is not published,
   not such a map, of the wrong layout, or failing, fails the evaluation
   and leaves the state as it was. A node runs again when its Kernel,
   its values, its extent, the clock or input it reads, or a source it
   reads changed; one that gives what it gave last, word for word, is
   unchanged to what reads it, so a cheap node that settles spares a
   costly one after it, and state at a fixed point does not advance.
   Words are compared where they are: in Host memory always; held on a
   device only when the node ran for nothing but the state it reads, as
   comparing there waits for the device. A node run for a changed input
   whose outputs stay on a device is changed to what reads it, whatever
   it gave; to spare what reads it, keep what it reads unchanged instead,
   a small row (Host memory) that holds still while nothing is done.

   State spread over several nodes advances in one order. A node a port
   reads `<~` holds state; another that reads it `<-` sees its new
   state, the one this evaluation made before it, and one that reads it
   `<~` its state as the last evaluation left it. So a moment the nodes
   must agree on, a battle beginning, is decided by one node, and the
   others take it from that node's new state `<-` and do not decide it
   again from what they read `<~`; a node that steps first reads the
   others `<~`, and those after it read it `<-`. `entelechy graph` says under
   each node of its plan which nodes' new state and which committed
   state it reads.
2. A source is a graph input by name or `node.port`. Inputs and nodes
   share one namespace. A graph may declare several outputs; which
   device takes which is decided by structure (a grid of pixels for the
   window, rows of one or two floats for the sound card), never by name.
3. Every declared input port must be wired, and a port the Kernel does
   not declare is an error; both are reported before anything runs.
4. Values use `=`: numbers, `true`/`false`, negation,
   `vec2`/`vec3`/`vec4`. An integer is accepted where a Float is
   expected. A number stays exact until a Kernel's param gives it a
   width: a graph `param` reads its default in its declared type, struct
   or array included (`param origin: Double = 500000.123456789;`), and a
   node's value keeps every digit until it reaches the Kernel. A
   value is written, never computed: `gain = 0.5 * 2.0;`, `gain = n +
   1;` and `times = n + 1;` are refused
   ([why](DECISIONS.md#the-graph-computes-structure-never-values)).
5. `domain = count;`, `domain = vec2(width, height);` or `domain =
   vec3(width, height, depth);` sets a node's output extent in its
   output's shape; a node with an input defaults to its first declared
   input's extent. A node with no `domain` that carries its own output
   back through another input (`previous <~ node.result` not first)
   holds state of its first input's size; a check notes it, and one
   whose Kernel compares `element()` with its first input's count, whose
   elements past it are never run. The extent may be an expression of: `size(port)` (the
   bound extent, as a count, pair or triple), `count(port)`,
   `value(port)` and `value(port, i)`, the node's and the graph's
   parameters by name, numbers, `vec2`/`vec3`, `+ - * / <<`, `max`, `min`,
   `floor`, `ceil`, `window()` (pixels) and `scale()`. It resolves when
   the node binds; its sides must come out positive and whole. Counts
   are exact whole numbers, and one too large is an error, never
   wrapped; the window, `scale()`, a `Float` parameter and a division
   with a remainder are measures, whole only after `floor` or `ceil`, so
   `ceil(size(p) / 2)` and `ceil(window() / 2)` agree. A boolean, color
   or struct sizes nothing. Examples: `domain = size(map) * tile;`,
   `domain = 4 * count(faces);`, `domain = window();` (runs again at the
   new extent on resize), `domain = value(detail) * value(detail);`. The
   Player provides the window; `run` takes `--window <w>x<h>[@<scale>]`
   and a replay says `window w h [scale]`. A binding must match the
   port's shape and element layout, whatever either side calls it.
6. A Kernel id is unique across everything a graph loads; the same id
   from two documents is an error. An anonymous inline Kernel is named
   `<document>.<node>@1`.
7. A `use` path is relative to the document; a trailing `/` means the
   directory's `.ent` files. Used documents hold Kernels only. A graph's
   `input` may have any port type; a parent wires it like a node port,
   checked by shape and layout where it binds.
8. A node may bind a backend's native operation by id: `node raster:
   native.rasterize@1 { ... }`. Native operations are published by the
   runtime, cannot be republished from a document, and are never fused;
   their contract is their manifest, ports by shape and layout.
   `native.rasterize@1` takes `vertices` (rows of `clip: Vec4; normal:
   Vec3; uv: Vec2`) and `triangles` (rows of three `Int`s) and writes
   `surface`, a grid of `depth: Float; normal: Vec3; uv: Vec2; id: Int`
   over the node's `domain`: bit for bit the raster package's surface on
   the interpreter and the JIT, the device's rasterizer, approximate at
   edges, on a GPU ([decision](DECISIONS.md)). `native.rasterize@3`
   takes a `canvas` (a `Grid<Long>`, the empty canvas the package's
   Kernels take their extent from) as well and draws over its extent,
   `domain = size(canvas);`, so a surface sized by another node's output
   is drawn as the package draws it.
9. A graph exposes `input`, `param` and `output` like a Kernel, so it is
   placed with `node name: graph "path.esn" { ... }`. Placement
   flattens it: its nodes join the parent under `name.`, its inputs take
   the parent's wires, its params the parent's values (or its defaults),
   and `name.output` reads its outputs. An inline Kernel's `param x:
   Float = amount;` takes the graph param `amount`. A graph is the file
   it is in: placed many times, from any level, its path written any way
   or reached through a package's name, it is one document, its Kernels
   defined once. The caller overrides
   a graph's params at evaluation; a graph cannot place itself. A scene's
   own output may also be written to a file, `output kept <- state.result
   to "saves/state.field";` in the extension's format, or `to bytes
   "f.bin"` for a `Rows<Int>` of bytes: the session writes it whole after
   an evaluation that changed it commits, only into a directory allowed
   with `--allow file.write=<dir>`; a failed evaluation writes nothing.
   `to files "project/"` takes a `Rows<Int>` of operations on the files
   under the directory, `[n, (kind, path_len, size, path.., data..)*]`
   (words after the `n`th unread): kind 0 writes `data` as the file,
   making it and its directories when absent; 1 removes it (`size` 0),
   and the directories it leaves empty; 2 renames it to the path `data`
   spells, never onto a file that is there. Paths are as a `files` input
   spells them; a malformed list is refused whole. Each list is done once,
   in order, when it differs from the last one done; a write needs
   `--allow file.write`, a removal or a rename `--allow file.delete` too.
   `to clipboard` takes a `Rows<Int>` of characters and puts them, to
   the first 0, on the clipboard an input `= clipboard` reads, when it
   changes and is not empty; one output writes it. The system's
   clipboard, in the Player, needs `--allow clipboard.write`.
   A placed graph's outputs are wired, never written
   ([decision](DECISIONS.md)).
   A font is placed as a node too: `node g: font "x.ttf" { codes <-
   wanted.result; }`. The Host reads the font file (TrueType or
   OpenType, read as `entelechy import` reads one) and gives the
   characters `codes` held when last evaluated, so a frame later, as
   three Fields read like outputs: `g.glyphs` a row a code in order
   (`code: Int; advance: Int; first: Int; count: Int; lo: Vec2; hi:
   Vec2`; a code the font lacks -1), `g.outlines` their line segments in
   ems (`Rows<Vec4>`), `g.metrics` one row (`units: Int; ascent: Float;
   descent: Float; gap: Float`). They are read again only when `codes`
   or the file changes; a font node takes no domain or params. Before
   any codes, `g.glyphs` is the missing glyph's one row (-1); a font
   that cannot be read gives that row, one zero segment and zero
   metrics, and is reported. The font file is read like any input
   file, from wherever its path names.
   A directory is placed as a node too: `node d: files "dir/" { paths <-
   wanted.result; }`. `d.listing` is every file under the directory but
   hidden ones, those under hidden directories and those a `.gitignore` on
   the way leaves out (git's rules, each `.gitignore` held to its own
   directory; a directory left out is not walked), each its path and size
   and none of its bytes: `[n, (path_len, size, path..)*]`, in path order,
   listed again on sync when a file comes, goes or changes. `d.opened` is
   the files `paths` named when last evaluated, so a frame later: `paths`
   is a `Rows<Int>` of each path's UTF-8 bytes ended by a 0 (an empty one
   no path), and `d.opened` those files in the order named, as `files`
   holds them (`[n, (path_len, size, path.., bytes..)*]`), a path naming no
   file left out, an ignored file read all the same; read again when
   `paths` or one of the files changes. A path out of the directory or
   one not readable refuses the paths whole and is reported, what was read
   staying. A files node takes no domain or params. It lists a folder as
   an editor does, reading only what is opened, where `files` reads every
   byte.
   Field files of one element are placed as a node too: `node near:
   Rows<Spot> fields "chunks/" { paths <- want.result; }`, `paths` as a
   files node's. `near.result` is the elements of the files `paths` named
   when last evaluated that are in, one file after another in the order
   named (each a `.field` file of any extent, its elements in order),
   and `near.ranges` a `Rows<IVec2>`, a row a path: where its elements
   start in `near.result` and how many there are, `-1` and `0` for a path
   whose file is not in: on its way, absent, or refused (another element,
   past the memory budget), which is reported. Before any are in,
   `near.result` is one zero element, and with no path `near.ranges` one
   row of `-1` and `0`. A file is asked for when first named and let go
   when no longer named; one that changes is read again. Where the
   session prepares in the background (the Player, a served session, a
   page) a file is read off the frames, taken in by the first sync after
   it is in, the frames going on with what was in meanwhile; a replay or
   a run takes each in a frame later, as a files node does. Its type is a
   `Rows` of the element; a fields node takes no domain or params. It is
   how a world larger than memory keeps in it only what its frames are
   near ([why](DECISIONS.md#a-world-larger-than-memory-is-read-by-what-it-asks-for)).
10. `times = N;` on a Kernel node or placed graph stands for `name/1` to
    `name/N`, each wired as written; `carry port <- output;` feeds the
    port from the previous time's output from the second time on.
    `name.port` means the last time's, in the scene and wherever a port is
    named: a replay, a read, a save, a pin. `N = 0` places nothing: a carried
    output is then what its port is first bound to, while an output no
    carry names, or a carry started by `<~`, has no value and is refused
    where read. `N` is a literal or a graph param; a param that sizes the
    graph lowers the document again when it changes (`param level 3`
    over the protocol resizes a running scene).
11. A node may be a Field: `node zeros: Rows<Float> { domain = 65; }` is
    filled with its element's zero, and `value = 0.5;` (a literal or a
    graph param) fills it otherwise. The type may be an alias a used
    document declares (`node blank: Cells { domain = vec2(96, 64); }`);
    its element is `Float`, `Int`, `Bool`, `Vec2`, `Vec3`, `Vec4` or
    `Image`'s pixel. It has a `domain` and no inputs.
12. `async = true;` runs a node off the frame on its own thread. Until
    the run lands, readers take its last output and the state it feeds
    through `<~` stays; the evaluation that finds it landed reads the new
    output, re-runs the readers, and commits. Its first run is on the
    frame; one run goes at a time; inputs are copied when it starts. It
    is never fused. A replay's `settle`, and `evaluate` with `settle:
    true` over the protocol, wait for every run to land.
13. `require name: a <= b;` is what a graph's parameters must hold for
    it to run: two counts, as a `domain` writes them from the graph's
    parameters, numbers, `window()` and `scale()`, compared with `<`,
    `<=`, `>`, `>=`, `==` or `!=`. It is checked as the document loads
    (so `check` and a `compose` refuse one broken by the parameters as
    given; one that reads the window waits for an evaluation), and
    before every evaluation with the parameters as the caller set them;
    one that does not hold stops the evaluation before anything runs.
    Each names the requirement, where it is written, and what each side
    came to. A
    placed graph's requirements come with it, named under the node that
    places it, reading what the parent passed. `require cells: columns *
    rows <= 1 << bits;` is how the physics package says its sort holds
    every cell ([why](DECISIONS.md#a-graph-says-what-its-parameters-must-hold)).

### Packages

A package is a directory, `name.entpkg/`, with a manifest `package.esn`
at its root:

```text
package image@1 { language = 1; ir = 1; }     // its id, and the least language and IR it needs
use "kernels/";                                // its Kernels, as any document uses them
```

It holds its Kernel files, its graph presets (`.esn` placed by path,
`node nes: graph "image.entpkg/graphs/nes.esn" { ... }`), and assets
a scene reads by path. `use "image.entpkg";` brings its Kernels and
declarations in, as if the scene used what the manifest uses, when the
manifest's versions are no newer than this Entelechy's (`entelechy version` prints
`entelechy.api`, `entelechy.language`, `entelechy.ir`, `entelechy.package`). The directory
is named for the package it holds, and a name is one package: a scene
reaching two packages of one name, two editions of it, is refused. A
`use` naming nothing beside the document is looked for by name beside
the package the document is in, so a package names the packages it uses
(`use "compact.entpkg";`) wherever they are kept together, then in the
directories `ENTELECHY_PACKAGES` lists. `entelechy pack` writes a package as one `.entpkg`
file, `entelechy unpack` reads it back, `entelechy package` says what one holds. A
Kernel reached by two paths is one file and one owner.

### Loading

The session parses the graph, publishes its inline Kernels, follows its
`use` items, packages included, and rebuilds the Graph model. It
publishes the Kernels of the documents the scene's `use` items name; of
a document those use in turn it reaches the declarations only, so a node
bound to a Kernel written there is said at load, and by `entelechy check`,
with the `use` that publishes it. A changed
document republishes only the Kernels whose text changed; a rejected
replacement keeps the running version; a Kernel that leaves its document
retires.

`size(port)` or `count(port)` in a `domain` is the extent of what the
port is bound to; bound to a scan's memory port, the memory's extent
(its initial value's before the first frame).

`value(port)` reads element `i` (0 unless given) of the Field the port
is bound to, as its `Int` or `Float`, this frame's. It brings that Field
to the Host (a wait on a device), so the port must be a row of scalars,
at most 1024. Several nodes reading one Field wait once. A count that
can reach zero wants `max(value(port), 1)`.

## Replay

Application tests are `.replay` files: `scene <path>`, optional
`library <path>`, then `frame`, `pointer`, `key`, `reset` (all state, or
`reset node.port` for one port's), and `expect` lines: a pixel, a field
of an element (`expect step.result[0].score = 1`), or an output's
element count (`expect count ground.result = 576`). Every form is in
[USAGE](USAGE.md#replay-tests). A replay is a test,
not a way to keep a piece made with a scene; see `docs/DECISIONS.md`,
*A replay is a test*.
