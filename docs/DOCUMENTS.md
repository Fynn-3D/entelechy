# Orla documents

**Answers:** how a Kernel and a scene are written.

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

Every port carries a Field: elements of one type in one shape. A Kernel
may declare as many ports as it likes; a device binds them all at once,
so a GPU backend refuses a Kernel whose inputs and outputs come to more
than its adapter allows (twenty-nine on an Apple M4, and the refusal
says the numbers), where the interpreter and the JIT have no limit. The
shapes are the dense lattices a Kernel can be mapped over, and there
are three:

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

A port may keep its element's scalars narrower than a word, after `as`:
`Grid<Vec4 as unorm8>` is a picture of four bytes a pixel, `Rows<Int as
u8>` a byte a row. The Kernel sees a `Vec4` and an `Int` as ever and
computes in `Float` and `Int`; the Field widens a scalar when it is read
and narrows it when it is written.

| Encoding | Keeps | Narrowed by |
|---|---|---|
| `f16` | a float in half precision | to nearest, ties to even; too large is infinity |
| `unorm8`, `unorm16` | a float in `0..=1` in 8 or 16 bits | clamped, NaN as 0, scaled and rounded to nearest, ties to even |
| `snorm8` | a float in `-1..=1` in a signed byte | the same, and -128 reads as -1 |
| `u8`, `i8`, `u16`, `i16` | an integer or a boolean | its low bits; read back zero- or sign-extended, a boolean as whether any bit is set |

The rule is the Host's and every backend keeps it bit for bit. A picture
file or `bytes` input declared narrow lands narrow; a narrow Field and a
wide one are different layouts, and a wire between them is refused.
Only a map or the steps of a scan write a narrow output (a scan's memory
may be narrow too): a reduction, a prefix and a scatter fold into their
output in place, in whole words. A narrow Field between two nodes is
written and read, never fused away, since the rounding it does is part
of the scene.

The element may be a scalar, a vector, or a struct; its layout is the
struct's scalars in order, integers and booleans exact. Inside `compute`,
Expressions are `let` bindings and a result; a binding takes a name,
`let n = ..;`, or the fields of one value, `let { x, y } = window();`,
each under its own name or the one written after it, `let { x: width,
y: height } = window();`, which is the same as reading them one at a
time; `+ - * /` and `%` (the
remainder, truncated like `/`, on `Int` and `Float` alike, and
componentwise on vectors) with `Int` and `Float` kept apart except that
`vec2`, `vec3`, and `vec4` take `Int` components and widen them, and
take shorter vectors among their components (`vec4(v3, 1.0)`,
`vec3(1.0, v2)`), as `ivec2` and `ivec3` take integer vectors; `v.xy`,
`v.zyx`, `c.rgb` and the like are swizzles, a vector of those
components in that order from one alphabet, and `v.x` one component;
`& | ^`
and `<< >> >>>` work on `Int`s (the shift count modulo 32, `>>`
arithmetic, `>>>` logical),
binding between comparison and arithmetic as in Rust; `&&` and `||`
answer without asking the other side when the first settles it, so
`n != 0 && x / n > 1` does not divide when `n` is zero; `==` and
`!=` compare scalars and whole vectors; `<` and the rest compare scalars;
`if c { a } else { b }` (and, among a block's statements, `if c {
return a; }` with no `else`, which guards what follows: the rest of the
block is its other arm), `match i { 0 => a, 1 | 2 => { let x = ..; x },
8..16 => c, 16..=31 => d, _ => b }` (on an `Int`; the arm whose values
hold it runs, else the `_`
arm, which comes last and is required; an arm's values may be written
as a range, `..` stopping before its end and `..=` taking it, which is
the values it holds and so at most 256 of them; no value in two arms;
every arm the same type; the GPU runs it as a `switch`, the JIT as a jump table
when the values are close) and `loop n with x = init until done { next }`
are expressions; a nested loop reads the values the loops around it
carry, and a scan's step its own, by name. A loop's count is a literal,
which is its bound, or any `Int` expression, read once before the loop
and clamped to `0..=65536`: `loop depth with ...` takes a parameter's
value, so the same Kernel runs a solve to a depth the scene sets. A
loop with a count of zero yields what it carries. A `break` among a
loop body's statements, `if done { break x; }`, stops the loop after
that iteration and gives that value, of the type the loop carries: the
iterations after it do not run, and it belongs to the innermost loop
around it.

Work is bounded twice, both counts of loop iterations and not of time,
so what fails fails alike on every backend and every machine. A Kernel
has a budget, the iterations one element may run, every loop's together
(a step of a scan and a merge of a reduction are each an element):
16777216 unless the Kernel declares `budget N;` among its items, 1 to
2^30. And one run of a Kernel, all its elements together, may take what
the Host allows: four thousand million iterations unless the session is
given another number of millions with `--work <M>`, beside `--memory`.
A loop adds the iterations whose body ran as it finishes (the one its
`until` stops is not one of them), and an element looks at its count
when it finishes, and at the start of each iteration of a loop that can
run more than 2^20 by itself: past its budget, the element fails the
evaluation there, named, as an integer overflow does below. The run
fails when its elements' iterations, summed exactly, pass the Host's
bound, without naming an element, and a run that has passed it is
stopped while it runs (so one that also has a failing element may be
named for either). Either way the running system keeps its last state. A
Kernel whose elements cannot run 1024 iterations or pass their budget,
their loop counts all literals that multiply to less, is never counted
and pays nothing. `Name { a: .., b: .. }` builds
a struct and `Name { ..base, a: .. }` copies one with some fields
replaced.

A fixed-size array `[T; N]` holds `N` values of one type `T`: any value
type, so `[Int; 16]`, `[Vec3; 4]`, `[Hit; 8]` and `[[Int; 4]; 4]` are
all arrays, and one may be a struct's field, a function's argument or
result, a loop's carried value or a Field's element. `[a, b, c]` builds
one from its elements, `[v; 8]` from eight copies of a value, and
`a[i]` takes an element: with a literal index it is a field, checked
when the Kernel compiles; with an `Int` computed at run time it is
clamped into the array, as `at` clamps into a Field. `[each N { body }]`
builds an array element by element: the body runs once per element,
unrolled, with `index()` its position (a loop inside the body has its
own `index()`), so `[each 32 { unit(w, x, index()) }]` is a layer's
activations and `[each 4 { if index() == k { v } else { a[index()] } }]`
is `a` with element `k` replaced. An array has one
to 4096 elements and is laid out as its elements in order, so a lookup
table, a ROM or a register file is written as data (the CHIP-8's font
and the Thumb core's program are).

A parameter is any value a Kernel can hold, and crosses as its words:
`param lens: Mat4 = mat4(1.0);`, `param cell: IVec2 = ivec2(3, 4);`,
`param edges: [IVec2; 12] = [..];`, `param style: Style = Style {
ground: vec3(0.1, 0.1, 0.1), width: 1.0 };`. A default is a literal:
numbers and booleans, `vec2`..`vec4`, `ivec2` and `ivec3`, `mat2`..`mat4`
from their columns or one value down the diagonal (`mat4(1.0)` is the
identity), `[a, b]` and `[v; N]`, and a struct literal naming every
field. What a value must match is its type's layout, never a spelling:
an integer stands in for a float, and anywhere a value is given from
outside (a scene, a replay's `param`, the protocol) an array of values
in order is taken in the shape of the parameter it is given to, so
`[vec3(0.9, 0.9, 0.9), vec3(0.8, 0.1, 0.1), 2.0]` is a `Style` and
`[[1, 0], [0, 1]]` a `Mat2`. A scene may name a struct its documents
declare at the top; a struct declared inside a Kernel is that Kernel's
alone, and is given by position.

Matrices `Mat2`, `Mat3` and `Mat4` are square, of floats, stored by
columns. `mat3(c0, c1, c2)` builds one from its column vectors (where
each axis lands), `mat3(1.0)` puts a value down the diagonal, `m[i]` is
a column (a literal index checked when the Kernel compiles, a computed
one clamped) and `v[i]` a vector's component the same way. `m * v`
applies a matrix to a column vector, `v * m` to a row vector, `m * m`
composes, `m * f` and `f * m` scale, `+` and `-` take two matrices of
one order, `-m` negates and `==` compares whole matrices; `transpose`,
`determinant` and `inverse` are the builtins (a singular matrix's
inverse holds no finite numbers). The products and these builtins, like
`dot`, `cross`, `length`, `normalize` and `reflect`, are written into
the compiled Kernel as the arithmetic they are, one formula in one
order, so every backend computes the same one and the interpreter and
the JIT agree on its words bit for bit.

Integer vectors `IVec2` and `IVec3` are cells and offsets: `ivec2(x,
y)` from `Int`s, `icoord()` this element's cell, `isize(f)` a Field's
extent and `idomain()` the output's, all as integers, so a rule on a
grid never converts a coordinate to write it down. They take `+ - * /
%`, `& | ^` and the shifts componentwise, with each other or with an
`Int` on either side (a failure where an `Int` would fail: overflow, a
zero divisor), `-v`, `==`, `.x .y .z` and `v[i]`, and `abs`, `sign`,
`min`, `max`, `clamp` (with `Int` bounds too); `float(v)` widens one
to a `Vec2` or `Vec3` and `int(v)` truncates a vector to one. `at`,
`fetch` and `put` take an integer vector where they take a `Vec2` or
`Vec3`, and a scatter's address may be one; each is the float address
converted, which is exact for every coordinate a Field can have, so
the Field rules are one set (clamped reads, dropped writes).

`Double` and `Long` are the 64-bit float and integer, with `DVec2` to
`DVec4`, `LVec2` and `LVec3`, and `DMat2` to `DMat4`: `dvec3(..)`,
`lvec2(..)` and `dmat4(..)` build them as their 32-bit kin do, taking
numbers of any kind and shorter vectors, widened, and `double(x)` and
`long(x)` convert, as `float(x)` and `int(x)` convert back (a long to an
int keeps its low 32 bits). A `Double` never meets a `Float`, nor a
`Long` an `Int`, without one of them; `x + f` of the two is refused and
says which to write. A number written without a width is exact until it
meets one, as a constant is in Go: arithmetic between written numbers
is done on their exact values (`0.1 + 0.2`, `1.0 / 3.0`, `2 * 1000000007`),
and the result is still a written number; beside a `Double` a written
`0.1` is the double nearest a tenth, not the float's, beside a `Long` a
written integer is a long, beside a `Float` a written integer is a
float, and an integer too big for an `Int` is a `Long` wherever it is
written. A division of written whole numbers is whole, as an `Int`'s
is: `7 / 2` is `3`, and one that leaves a remainder is refused beside a
float rather than taken as the float it truncated to (`x * (1 / 2)`
says to write `1.0 / 2`). A vector, a matrix or an array built of written numbers alone
(`vec3(0.1, 0.2, 0.3)`, `[0.5, 0.25]`) takes the width it meets as a
whole, given to a `DVec3` or a `[Double; 2]`, beside one in an
operation, or among the branches of an `if` or the elements of an
array. Where nothing names a width a written float is a `Float` and a
written integer an `Int`; `0.1d` and `8l` say it outright, and a value
named by a `let` keeps the width it was named with. Each is two words, the low one first, in a Field, a
parameter and the protocol, so an element of `Rows<DVec3>` is six words;
a 64-bit scalar is never kept narrow. They take what their 32-bit kin
take: on doubles every math builtin, `dot`, `cross`, `length`,
`normalize`, the matrix products, `inverse` and the rest; on longs the
arithmetic (an overflow and a zero divisor fail as an `Int`'s do), the
bit operators and the shifts (the count modulo 64), `abs`, `sign`,
`min`, `max`, `clamp`, `add_wrap`, `sub_wrap` and `mul_wrap`. The
interpreter and the JIT compute them as Rust's own `f64` and `i64` do.
A GPU computes a `Long` in a map or a reduction where the device has
64-bit integers, and in a scatter whose target is all `Long`s under
`min` or `max` where it also takes them atomically, and never a `Double` (few devices have one, and those
at a fraction of a float's rate); a session on the GPU runs such a node
on the JIT, and the scene does not see the difference.
[`examples/precision`](../examples/precision/README.md) is what they are
for.

A function may take a math builtin's name (`step`, `mix`, `abs`) and
then stands for it in its document; the Field, memory, context and
construction builtins (`at`, `put`, `coord`, `vec2` and the rest) keep
their names.

The math builtins are `floor`, `ceil`, `round`, `trunc`, `sign`,
`fract`, `abs`, `sqrt`, `inversesqrt`, `sin`, `cos`, `tan`, `asin`,
`acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `exp`, `exp2`, `log`,
`log2`, `pow`, `fma`, `min`, `max`, `clamp`, `saturate`, `mix`, `step`,
`smoothstep`, `mod` (the remainder with the divisor's sign, for wrapping
coordinates; `%` truncates), `radians`, `degrees`, `isnan`, `isinf`,
`dot`, `cross`, `length`, `distance`, `normalize`, `reflect`, `refract`,
and the conversions `float`, `int`, `double` and `long`; on `Int`s, `add_wrap`, `sub_wrap`
and `mul_wrap` are the arithmetic modulo 2^32 (a machine's ALU), where
`+ - *` report an overflow as a failure, and `popcount`, `clz`, `ctz`,
`rotl`, `rotr`, `ult`, `udiv`, `urem`, `mulhi` and `umulhi` are the bit
counts, rotations, unsigned compare and division, and the high half of a
64-bit product. `bits` and `from_bits` carry a Float's 32 bits to an Int
and back unchanged; the Ints of Floats of no sign order as the Floats
do, which is how a depth and a name share one key. `orla help <name>` shows any builtin's signatures.

Integer arithmetic that cannot produce a value — an overflow of `+ - *`,
a division or remainder by zero, the least integer negated, `udiv` or
`urem` by zero — is a failure on every backend, naming the first element
in row-major order that has it and the first failure within it: the
evaluation fails there and nothing it computed commits. A device records
the failure as it runs and the evaluation waits for the device once,
before it commits, when a node that ran can fail; one whose arithmetic
cannot pays nothing. Two caveats remain, both about a device. A NaN made
by arithmetic on the GPU (`inf * 0`, or `sqrt` of a negative) may come
out a number under a driver's fast math, which wgpu does not let a
program turn off, where the interpreter and the JIT have NaN; a NaN read
from a Field is seen alike on every backend. An integer `sum` a
scatter or a prefix folds is each element's whole total, which no order
of adding changes: it fails only when that total does not fit, at the
first element whose total does not, and a sum that passes the end of
the range on the way and comes back is the total it comes back to. What
every backend does agree on, and what
the feature matrix holds them to: a positive divided by zero is an
infinity, zero divided by zero is not a number, and nothing equals or
orders against a not-a-number. An `Int` literal may be hexadecimal, `0xFF`, its 32 bits as
written; more than eight digits, or an `l`, make it a `Long`'s 64.
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
rate is reported, not stored, and a Kernel resamples with `at_linear`), a
score as `.mid` (rows of `{start, duration, pitch, velocity, channel}`,
its ticks and tempo map read as seconds), and any Field as `.field`, the
Host's own format. `run --save` and the
protocol's `read` write all of them. `input title: Rows<Int> = text
"ORLA";` holds text: the session supplies the characters' codes as rows
and writes them again when the document's text changes, so a label is a
string in the scene rather than a Kernel of cases. `input level: Rows<Int> = bytes "level.bin";` reads a file as its
bytes, one to an element, 0 to 255: what they mean is the Kernel's
business, which is how a scene reads a format nothing here knows
(`examples/level`). `input clip: Image =
frames "clip/" at 30;` reads a directory frame by frame: its files in
name order (the same formats), one per frame at the rate, the frame
`floor(time * rate)` of the session's time wrapping at the end, brought
in at each evaluation; the directory is listed again on sync. A function's name is any word but the five that begin an expression
(`let`, `return`, `if`, `match`, `loop`), keywords included: `fn
scatter(...)` is a function, since a call is a word and a `(` wherever
it stands. A short function is written out where it is called when the
Kernel is compiled, so calling one costs nothing a run can measure; a
trace still shows the call. A struct, `type`, or `fn` declared at
the top of a document is shared by every Kernel in it, and by every Kernel
of a document that `use`s it; a function calls only what is declared above
it. A name a Kernel cannot find is looked for in the documents beside it
and beside the scene, and the diagnostic says which one declares it and
the `use` that would bring it in. Two documents that `use` each other are
refused by name: what they share belongs in a third both use. A function may take a Field: `fn cast(voxels: Volume<Int>, eye: Vec3)
-> Hit` reads its parameter with `at`, `at_linear`, `size`, or `count`, and
a call names an input port for it, `cast(world, eye)`; the function is
instantiated once per port it is called with, checked by shape and
layout. A function may also name type parameters, `fn twice<T>(v: T)
-> T`, each bound by the type of the argument written for it and the
same in every place it appears; one instance is compiled per set of
types a call gives it, and its body is checked there, so an error says
which instance was being compiled. A function calls only what is
declared above it, so it never calls itself: a Kernel runs where there
is no stack, and a `loop` with the work as its carried value is the
form for repetition. `sample` stays in `compute`, where the element's own address is.

A map may write several outputs, each a Field of its own over the one
domain, so they share a shape:

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

`compute` gives them by name, `{ albedo: .., normal: .., depth: .. }`,
every output once, in any order; that record is a value like any other,
so an `if` or a `match` may give one from each arm and a `let` may hold
one (`r.depth` reads a field of it). A graph reads each output as
`node.albedo`, `node.depth`. Only a map writes several: a reduction, a
scan and a prefix fold or carry one element and a scatter writes one
target, and there several results are one struct. When one consumer is
the only reader of every output of a map and reads them all by
`sample`, the two fuse whole, and none of the outputs is written; while
anything else reads one, the map runs on its own and writes them all.

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
booleans come out exact, floats agree to rounding. Like any function,
`merge` may read the inputs (`at`, `at_linear`, `count`, `size`), the
params and the context, on every backend. A reduce node takes no
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

Ordered work may also own memory: a Field its steps read and write.

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

`memory name: Type;` declares a port that is both: a graph binds it like
an input (`ram <~ machine.ram from zeros.result` makes it persist across
evaluations; `ram <- fresh.result` starts each evaluation over) and reads
it like an output (`machine.ram`). The steps start from what was bound,
`at` reads the memory as the steps have left it so far, and `put(name,
address, value)` writes an element (the address as `at` takes it, floored;
an address outside the memory is dropped, never clamped) and yields the
value written. What the steps left there is the output of that name,
listed after the Kernel's own output. Only ordered work may write, so
every backend applies the writes in one order, and the bound Field is
never touched: a failed evaluation leaves it as it was. A function may
`put` into a Field parameter, so a store is written once and called from
the steps (`fn store(m: Rows<Int>, a: Int, v: Int) -> Int { return
put(m, a, v); }`); it is a step's write, in the step's order, and the
initial carried value may not write, though it may read a memory with
`at` (what was bound, before any step: the ant example starts from the
cell its memory holds). An expression
followed by `;` is a statement for its effect, `put(ram, a, v);`, the
same as `let _ = put(ram, a, v);`; a binding named `_` binds nothing.
An `if` whose arm is statements alone, `if hit { emit(p); }`, does them
when it holds and the block goes on; an `else` or an `else if` after it
is statements too, and an empty arm `{ }` does nothing. It is the same
as `let _ = if hit { emit(p); 0 } else { 0 };`, so an arm that gives a
value beside one that does not is refused, and a guard that returns is
still a guard.

An expression or a block nests at most 256 deep, counting what encloses
it and the operators chained before it (`a + b + c` is three deep at
`c`); past that the document is refused, saying where, and a long chain
is split with `let`. The bound is what lets a live session take any
Kernel an agent sends: a failed update is an error, never a process
that ran out of stack
([`MAX_NESTING`](../crates/frontend/language/src/lib.rs)).

A Kernel with `scatter` in place of `compute` writes where it computes,
the fourth execution mode:

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

The body runs once per element of the first input, or of the node's
`domain` when it sets one (of the first input's shape, since `element()`,
`coord()` and `domain()` are read in it) (a step over every point and every weight sets
`domain = points * weights;` and needs no Field of that size), and
yields a struct of two fields: an address into the target named after `into` (an `Int` for
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
fused. The exact language is defined by [the frontend](../crates/frontend/language/src/lib.rs)
and its tests.

An element may give more pairs than the one it returns. `emit(pair);`,
in the body and in no function it calls, gives the target one more pair
of the same form, and may run inside a loop, so an element gives as many
as it computes (a triangle one pair a pixel it covers). An element's
pairs are the ones it emits, in order, then the one it returns; a body
that has nothing more to give returns an address outside the target.
`emit` takes only a rule whose answer is the same in any order — `min`,
`max`, or a `sum` of no floats — so a backend is free to land a pair the
moment it is emitted: the GPU combines it into the target with an atomic
from inside the body and keeps no list of pairs. The interpreter and the
JIT do keep one, and hold it against the Host's budget: a run whose
pairs need more room than the Host has left fails as a run, and commits
nothing ([decision](DECISIONS.md)).

A Kernel with `prefix` in place of `compute` is a prefix fold, the fifth
mode: every element of the output is its own value combined with every
value before it.

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

The body runs once per element of the first input, a `Rows`, exactly as a
map's does — `sample`, `element()`, `at`, the params and the context all
read what they read in a map — and the output is a `Rows` of the same
count whose element `i` is the body's values at `0..=i` combined by the
rule after `with`: `sum` adds fieldwise, `min` and `max` take the
fieldwise extreme, ordering floats by the same total order a scatter
uses.

The rule is declared rather than written as a carried value because a
prefix is free to fold in any grouping: it must give one answer whatever
grouping that is. So `sum` over an element holding a float is refused —
a float sum depends on the order it is added in, and `scan` is the form
that walks a row in element order — and so is `last`, which as a prefix
is every element's own value, a map. What the freedom buys is that a
device folds a row in three passes over blocks rather than walking every
element in one invocation: a rank over two million elements is
sub-millisecond work on a GPU and about six milliseconds as a scan. The
interpreter and the JIT walk it, which is one grouping of many and the
one the reference is held to. An integer `sum` that overflows is a
failure where the reference walks it, as any integer overflow is, and
(like every arithmetic failure) is not reported on a device; the words
that come out are the same either way.

## Graph

A node may take any word for its name, a keyword too (`node scan: …`,
`… <- scan.result`), since the name follows `node` and a source follows
`<-`, `<~` or `from`; a Kernel's own names (ports, parameters, values)
cannot be keywords.

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
output kept <- state.result to "saves/state.field";   // also written to a file
```

### Rules

1. Wiring is written only at the consumer. `port <- source` reads the
   source in this evaluation and is a dependency. `port <~ source from
   initial` reads the source as committed by the previous successful
   evaluation, is not a dependency, and reads `initial` until something has
   been committed. When a republished Kernel changes what the port
   carries, so the committed state no longer has the port's layout,
   `port <~ source from initial migrate kernel.id@v` carries it over:
   the named Kernel is a map over one input laid out as the old state,
   giving the new layout, run once over the state before the nodes run,
   and the evaluation reports it as migrated. Without `migrate`, such a
   port restarts from `initial` and the evaluation reports the reset. A
   migration that is not published, is not such a map, takes or gives
   the wrong layout, or fails, fails the evaluation and leaves the state
   as it was.
2. A source is a graph input by name or `node.port`. Inputs and nodes share
   one namespace. A graph may declare several outputs; which device takes
   which is decided by structure (a grid of pixels for the window, rows of
   one or two floats for the sound card), never by an output's name.
3. Every declared input port must be wired; a port the Kernel does not
   declare is an error; both are reported before anything runs.
4. Values use `=`: numbers, `true`/`false`, negation, `vec2`/`vec3`/`vec4`.
   An integer literal is accepted where a Float is expected. A number
   is kept exact until a Kernel's param gives it a width: a graph
   `param` reads its default in the type it declares, a struct or an
   array included (`param origin: Double = 500000.123456789;` holds the
   double nearest what is written), and a node's value keeps every digit
   until it reaches the Kernel. A value is written, never worked out:
   `gain = 0.5 * 2.0;` and `gain = n + 1;` are refused, and so is `times
   = n + 1;`. Arithmetic on values belongs in a Kernel
   ([why](DECISIONS.md#the-graph-computes-structure-never-values)).
5. `domain = count;`, `domain = vec2(width, height);`, or
   `domain = vec3(width, height, depth);` sets a node's output extent in
   its output's shape; nodes with an input default to the extent of their
   first declared input. The extent may also be an expression over what
   comes in: `size(port)` (the extent a port is bound to, as a count, a
   pair, or a triple), `count(port)`, `value(port)` and `value(port, i)`
   (a number a Kernel put in a Field), the node's parameters and the
   graph's by name, numbers, `vec2`/`vec3`, `+ - * /`, `max`, `min`,
   `floor`, `ceil`, and the window as context: `window()` (its size in
   pixels) and `scale()` (its scale factor); it resolves when the node
   binds, and its sides must come out positive and whole. Counts and
   sides are whole numbers, exact, and a count too large to hold is an
   error rather than a wrapped one; the window, `scale()`, a `Float`
   parameter and a division that leaves a remainder are measures, which
   may carry a fraction until `floor` or `ceil` makes them whole, so
   `ceil(size(p) / 2)` and `ceil(window() / 2)` agree. A `domain` computes an extent and
   nothing else: a boolean, a colour or a struct sizes nothing. `domain =
   size(map) * tile;` draws a map at its tile size; `domain = 4 *
   count(faces);` sizes a subdivision; `domain = window();` fills the
   window at the window's resolution, and the node runs again at the new
   extent when the window is resized. `domain = value(detail) *
   value(detail);` places as much as a Kernel decided there is this
   frame — the one term that reads data rather than shape, and so the
   only way a scene does less work when there is less to do. The Player provides the window;
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
   uv: Vec2; id: Int` over the node's `domain`. A scene places the
   raster package for that and binds `native.rasterize@1` where it wants
   the fast path: the same surface bit for bit on the interpreter and the
   JIT, the device's own rasterizer, approximate at edges, on a GPU.
9. The graph exposes `input`, `param`, and `output` like a Kernel does, so
   a graph is placed as a node with `node name: graph "path.osn" { ... }`.
   Placement flattens it: its nodes join the parent under the prefix
   `name.`, its inputs take the parent's wires, its params take the parent's
   values (or its own defaults), and `name.output` reads its outputs. An
   inline Kernel's `param x: Float = amount;` takes the graph param `amount`
   as its value. The caller overrides a graph's params at evaluation; nested
   graphs cannot place themselves.
   An output of the scene itself may be written to a file as well:
   `output kept <- state.result to "saves/state.field";` in the format
   the extension names (as `run --save` writes them), or `to bytes
   "f.bin"` for a `Rows<Int>` of bytes. A Kernel never writes a file: the
   session writes the output once an evaluation that changed it has
   committed, whole or not at all, and only into a directory the person
   running it allowed with `--allow file.write=<dir>`; a failed evaluation
   writes nothing. A placed graph's outputs are wired, never written. An
   input read from a file may fall back on a source while the file is not
   there: `input saved: Rows<Tally> = "saves/tally.field" else
   fresh.result;` reads `fresh.result` until the file exists, which is how
   a scene starts from nothing and later from what it saved
   (`examples/save`).
10. A node placed more than once is written once: `times = N;` on a
   Kernel node or a placed graph stands for `name/1` to `name/N`, each
   wired as written, and `carry port <- output;` feeds the port from the
   previous time's output from the second time on, so a relaxation, a
   Jacobi solve, or a subdivision is one node with a count. `name.port`
   anywhere means the last time's. `N = 0` places no node, and a carried
   output is then what its port is first bound to — no steps leave the
   start, level 0 of a subdivision is the mesh — while an output no
   carry names, or a carry started by `<~`, has no value and is refused
   where it is read. `N` is a literal or a graph param; a
   param that sizes the graph lowers the document again when it changes,
   so `param level 3` over the protocol resizes a running scene.
11. A node may be a Field rather than a Kernel: `node zeros: Rows<Float>
   { domain = 65; }` is a Field of that type and extent, filled with its
   element's zero, and `value = 0.5;` fills it with something else (a
   literal or a graph param, so a scene's starting value is one line and
   no Kernel). The type may be an alias a used document declares
   (`node blank: Cells { domain = vec2(96, 64); }`); its element must be
   what a parameter can hold, `Float`, `Int`, `Bool`, `Vec2`, `Vec3`,
   `Vec4`, or `Image`'s pixel, and anything else is a Kernel's work.
   The node has a `domain` and no inputs.
12. A node with `async = true;` runs off the frame: an evaluation starts
   its execution on a thread of its own and goes on. Until the run
   lands, the node's readers take its last output and the state it
   feeds through `<~` stays as it was; the evaluation that finds the run
   landed reads the new output, re-runs the readers, and commits. Its
   first run is on the frame, so its readers have something to read; one
   run goes at a time. The inputs are copied when the run starts, so it
   is for work that takes longer than its inputs take to copy: a machine
   stepping a million instructions, a solve, a long reduction. Such a
   node is never fused. A replay's `settle`, and `evaluate` with
   `settle: true` over the protocol, wait for every run to land.

### Packages

A package is a directory, `name.orlapkg/`, with a manifest `package.osn`
at its root:

```text
package pixelfx@1 { language = 1; ir = 1; }   // its id, and the least language and IR it needs
use "kernels/";                                // its Kernels, as any document uses them
```

beside which it holds its Kernel files, its graph presets (`.osn`
documents placed as nodes by path, `node nes: graph
"pixelfx.orlapkg/graphs/nes.osn" { ... }`), and any assets a scene reads
by path. A scene's `use "pixelfx.orlapkg";` brings the package's Kernels
and declarations in, as if it had used what the manifest uses, once the
manifest's versions are no newer than this Orla's (`orla version` prints
them: `orla.api`, `orla.language`, `orla.ir` and `orla.package`, each
its own); a `use` that names nothing beside the document is looked for
by name in the directories `ORLA_PACKAGES` lists. `orla pack` writes a
package as one `.orlapkg` file, `orla unpack` reads it back, `orla
package` says what one holds. A Kernel reached by two paths, a scene's
`use` of the package and a preset's `use "../kernels/"`, is one file
and one owner.

### Loading

The session parses the graph, publishes its inline Kernels, follows its
`use` items, packages included, and rebuilds the Graph model. Changed documents republish only
the Kernels whose text changed; a rejected replacement keeps the running
version; a Kernel that leaves its document retires.

A `domain` expression's `size(port)` or `count(port)` is the extent
of what the port is bound to; bound to a memory port of a `scan`, it
is the memory's extent (its initial value's, before the first frame),
not the scan's domain.

`value(port)` is the one term that reads what is *in* a Field rather
than how big it is: element `i` (0 unless given) of the Field the port
is bound to, as the `Int` or `Float` it holds. The producer has run by
the time the node binds, so the number is this frame's; reading it
brings that Field to the Host, which on a device is a wait, so the port
must be a row of scalars and hold at most 1024 of them — it carries
counts, not data. Several nodes reading the same Field wait once. A
count that can reach zero wants `max(value(port), 1)`, since a domain's
sides must be positive.

## Replay

Application tests are `.replay` files: `scene <path>`, optional
`library <path>`, then `frame`, `pointer`, `key`, `reset` (all state, or
`reset node.port` for one port's), and `expect` lines — a pixel, a field of an element (`expect step.result[0].score =
1`), or how many elements an output holds (`expect count ground.result
= 576`, which is what a `domain` came out as). See [the replay
module](../crates/interface/session/src/replay.rs). A replay is a test,
not a way to keep a piece made with a scene; see `docs/DECISIONS.md`,
*A replay is a test*.
