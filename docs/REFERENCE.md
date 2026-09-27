<!-- rendered: reference -->
# Orla

Kernels (`.orla`) compute; scenes (`.osn`) compose them into a running world.

## The loop

- understand (`help`, `status`, `kernels`, `kernel`, `graph`, `document`): what Orla is, and what this session holds and runs
- search (`search`): what the world already knows, before writing anew
- generate (`prepare`): a Kernel written, compiled and checked, publishing nothing
- compose (`publish`, `retire`, `compose`, `sync`, `param`, `write`): the world changed: a law published or retired, the scene rewritten on disk, a param or a state set; a refused change leaves it as it was
- run (`evaluate`, `advance`, `window`, `reset`): frames evaluated, one or many, under a clock and inputs
- observe (`read`, `trace`, `bench`, `history`, `pin`, `unpin`): what was computed and why: a picture seen, a Field summed up, an element traced, the history of changes
- improve (`snapshot`, `restore`): a world kept and returned to, to try a change and undo it

## language: what a Kernel may write

### types: the types a port, a param or a value has

- **Rows**: `Rows<T>` A row of elements, element() the index of one; count(f) says how many, at(f, i) reads one.
- **Grid**: `Grid<T>` A grid of elements, coord() the column and row of one; size(f) and at(f, vec2) read it.
- **Volume**: `Volume<T>` A volume of elements, coord() the column, row and layer of one.
- **Image**: `Image` Grid<Vec4>: a picture, each channel a Float from 0 to 1, the first output that is one is what a scene shows.
- **Float**: `Float` A 32-bit float; written 1.0, never 1.
- **Int**: `Int` A 32-bit integer; an overflow fails the element, add_wrap and its kin wrap.
- **Bool**: `Bool` true or false.
- **Double**: `Double` A 64-bit float; a literal is taken exact.
- **Long**: `Long` A 64-bit integer.
- **Vec2**: `Vec2   Vec3   Vec4` Floats in two, three or four components: v.x, v.y, v.z, v.w, and v.rgb, v.xy and the like as swizzles.
- **Vec3**: `Vec3` Three Floats; see Vec2.
- **Vec4**: `Vec4` Four Floats, a pixel's r g b a; see Vec2.
- **IVec2**: `IVec2   IVec3` Integers in two or three components: a cell, an offset, an address for at and put.
- **IVec3**: `IVec3` Three integers; see IVec2.
- **Mat2**: `Mat2   Mat3   Mat4` Float matrices by columns: m[i] a column, m * v applies it.
- **Mat3**: `Mat3` A 3 by 3 Float matrix; see Mat2.
- **Mat4**: `Mat4` A 4 by 4 Float matrix; see Mat2.
- **DVec2**: `DVec2   DVec3   DVec4` Doubles in two, three or four components.
- **DVec3**: `DVec3` Three Doubles.
- **DVec4**: `DVec4` Four Doubles.
- **LVec2**: `LVec2   LVec3` Longs in two or three components.
- **LVec3**: `LVec3` Three Longs.
- **DMat2**: `DMat2   DMat3   DMat4` Double matrices by columns.
- **DMat3**: `DMat3` A 3 by 3 Double matrix.
- **DMat4**: `DMat4` A 4 by 4 Double matrix.
- **as**: `input albedo: Grid<Vec4 as unorm8>;` How a port's Field is stored: f16, unorm8, snorm8, unorm16, u8, i8, u16, i16, or word; a Kernel reads and writes the type, the storage is narrower.

### builtins: what a Kernel may call

- **floor**: `floor(x: Float) -> Float`; `floor(x: Vec2) -> Vec2`; `floor(x: Vec3) -> Vec3`; `floor(x: Vec4) -> Vec4`; `floor(x: Double) -> Double`; `floor(x: DVec2) -> DVec2`; `floor(x: DVec3) -> DVec3`; `floor(x: DVec4) -> DVec4` The largest integer not greater than x, componentwise.
- **ceil**: `ceil(x: Float) -> Float`; `ceil(x: Vec2) -> Vec2`; `ceil(x: Vec3) -> Vec3`; `ceil(x: Vec4) -> Vec4`; `ceil(x: Double) -> Double`; `ceil(x: DVec2) -> DVec2`; `ceil(x: DVec3) -> DVec3`; `ceil(x: DVec4) -> DVec4` The smallest integer not less than x, componentwise.
- **round**: `round(x: Float) -> Float`; `round(x: Vec2) -> Vec2`; `round(x: Vec3) -> Vec3`; `round(x: Vec4) -> Vec4`; `round(x: Double) -> Double`; `round(x: DVec2) -> DVec2`; `round(x: DVec3) -> DVec3`; `round(x: DVec4) -> DVec4` The nearest integer, ties to even, componentwise.
- **sign**: `sign(x: Float) -> Float`; `sign(x: Int) -> Int`; `sign(x: Vec2) -> Vec2`; `sign(x: Vec3) -> Vec3`; `sign(x: Vec4) -> Vec4`; `sign(x: IVec2) -> IVec2`; `sign(x: IVec3) -> IVec3`; `sign(x: Double) -> Double`; `sign(x: Long) -> Long`; `sign(x: DVec2) -> DVec2`; `sign(x: DVec3) -> DVec3`; `sign(x: DVec4) -> DVec4`; `sign(x: LVec2) -> LVec2`; `sign(x: LVec3) -> LVec3` -1, 0 or 1 by the sign of x, componentwise; an Int stays an Int.
- **log**: `log(x: Float) -> Float`; `log(x: Vec2) -> Vec2`; `log(x: Vec3) -> Vec3`; `log(x: Vec4) -> Vec4`; `log(x: Double) -> Double`; `log(x: DVec2) -> DVec2`; `log(x: DVec3) -> DVec3`; `log(x: DVec4) -> DVec4` The natural logarithm, componentwise.
- **tan**: `tan(x: Float) -> Float`; `tan(x: Vec2) -> Vec2`; `tan(x: Vec3) -> Vec3`; `tan(x: Vec4) -> Vec4`; `tan(x: Double) -> Double`; `tan(x: DVec2) -> DVec2`; `tan(x: DVec3) -> DVec3`; `tan(x: DVec4) -> DVec4` The tangent, componentwise.
- **asin**: `asin(x: Float) -> Float`; `asin(x: Vec2) -> Vec2`; `asin(x: Vec3) -> Vec3`; `asin(x: Vec4) -> Vec4`; `asin(x: Double) -> Double`; `asin(x: DVec2) -> DVec2`; `asin(x: DVec3) -> DVec3`; `asin(x: DVec4) -> DVec4` The arc sine, componentwise.
- **acos**: `acos(x: Float) -> Float`; `acos(x: Vec2) -> Vec2`; `acos(x: Vec3) -> Vec3`; `acos(x: Vec4) -> Vec4`; `acos(x: Double) -> Double`; `acos(x: DVec2) -> DVec2`; `acos(x: DVec3) -> DVec3`; `acos(x: DVec4) -> DVec4` The arc cosine, componentwise.
- **atan**: `atan(x: Float) -> Float`; `atan(x: Vec2) -> Vec2`; `atan(x: Vec3) -> Vec3`; `atan(x: Vec4) -> Vec4`; `atan(x: Double) -> Double`; `atan(x: DVec2) -> DVec2`; `atan(x: DVec3) -> DVec3`; `atan(x: DVec4) -> DVec4` The arc tangent of one value, componentwise; atan2 takes two.
- **fract**: `fract(x: Float) -> Float`; `fract(x: Vec2) -> Vec2`; `fract(x: Vec3) -> Vec3`; `fract(x: Vec4) -> Vec4`; `fract(x: Double) -> Double`; `fract(x: DVec2) -> DVec2`; `fract(x: DVec3) -> DVec3`; `fract(x: DVec4) -> DVec4` The fractional part, x - floor(x), componentwise.
- **abs**: `abs(x: Float) -> Float`; `abs(x: Int) -> Int`; `abs(x: Vec2) -> Vec2`; `abs(x: Vec3) -> Vec3`; `abs(x: Vec4) -> Vec4`; `abs(x: IVec2) -> IVec2`; `abs(x: IVec3) -> IVec3`; `abs(x: Double) -> Double`; `abs(x: Long) -> Long`; `abs(x: DVec2) -> DVec2`; `abs(x: DVec3) -> DVec3`; `abs(x: DVec4) -> DVec4`; `abs(x: LVec2) -> LVec2`; `abs(x: LVec3) -> LVec3` The absolute value, componentwise; an Int or IVec stays one.
- **sqrt**: `sqrt(x: Float) -> Float`; `sqrt(x: Vec2) -> Vec2`; `sqrt(x: Vec3) -> Vec3`; `sqrt(x: Vec4) -> Vec4`; `sqrt(x: Double) -> Double`; `sqrt(x: DVec2) -> DVec2`; `sqrt(x: DVec3) -> DVec3`; `sqrt(x: DVec4) -> DVec4` The square root, componentwise.
- **sin**: `sin(x: Float) -> Float`; `sin(x: Vec2) -> Vec2`; `sin(x: Vec3) -> Vec3`; `sin(x: Vec4) -> Vec4`; `sin(x: Double) -> Double`; `sin(x: DVec2) -> DVec2`; `sin(x: DVec3) -> DVec3`; `sin(x: DVec4) -> DVec4` The sine, componentwise.
- **cos**: `cos(x: Float) -> Float`; `cos(x: Vec2) -> Vec2`; `cos(x: Vec3) -> Vec3`; `cos(x: Vec4) -> Vec4`; `cos(x: Double) -> Double`; `cos(x: DVec2) -> DVec2`; `cos(x: DVec3) -> DVec3`; `cos(x: DVec4) -> DVec4` The cosine, componentwise.
- **min**: `min(a: Float, b: Float) -> Float`; `min(a: Int, b: Int) -> Int`; `min(a: Vec2, b: Float) -> Vec2`; `min(a: Vec2, b: Vec2) -> Vec2`; `min(a: Vec3, b: Float) -> Vec3`; `min(a: Vec3, b: Vec3) -> Vec3`; `min(a: Vec4, b: Float) -> Vec4`; `min(a: Vec4, b: Vec4) -> Vec4`; `min(a: IVec2, b: Int) -> IVec2`; `min(a: IVec2, b: IVec2) -> IVec2`; `min(a: IVec3, b: Int) -> IVec3`; `min(a: IVec3, b: IVec3) -> IVec3`; `min(a: Double, b: Double) -> Double`; `min(a: Long, b: Long) -> Long`; `min(a: DVec2, b: Double) -> DVec2`; `min(a: DVec2, b: DVec2) -> DVec2`; `min(a: DVec3, b: Double) -> DVec3`; `min(a: DVec3, b: DVec3) -> DVec3`; `min(a: DVec4, b: Double) -> DVec4`; `min(a: DVec4, b: DVec4) -> DVec4`; `min(a: LVec2, b: Long) -> LVec2`; `min(a: LVec2, b: LVec2) -> LVec2`; `min(a: LVec3, b: Long) -> LVec3`; `min(a: LVec3, b: LVec3) -> LVec3` The smaller of a and b, componentwise; a Float beside a vector broadcasts.
- **max**: `max(a: Float, b: Float) -> Float`; `max(a: Int, b: Int) -> Int`; `max(a: Vec2, b: Float) -> Vec2`; `max(a: Vec2, b: Vec2) -> Vec2`; `max(a: Vec3, b: Float) -> Vec3`; `max(a: Vec3, b: Vec3) -> Vec3`; `max(a: Vec4, b: Float) -> Vec4`; `max(a: Vec4, b: Vec4) -> Vec4`; `max(a: IVec2, b: Int) -> IVec2`; `max(a: IVec2, b: IVec2) -> IVec2`; `max(a: IVec3, b: Int) -> IVec3`; `max(a: IVec3, b: IVec3) -> IVec3`; `max(a: Double, b: Double) -> Double`; `max(a: Long, b: Long) -> Long`; `max(a: DVec2, b: Double) -> DVec2`; `max(a: DVec2, b: DVec2) -> DVec2`; `max(a: DVec3, b: Double) -> DVec3`; `max(a: DVec3, b: DVec3) -> DVec3`; `max(a: DVec4, b: Double) -> DVec4`; `max(a: DVec4, b: DVec4) -> DVec4`; `max(a: LVec2, b: Long) -> LVec2`; `max(a: LVec2, b: LVec2) -> LVec2`; `max(a: LVec3, b: Long) -> LVec3`; `max(a: LVec3, b: LVec3) -> LVec3` The larger of a and b, componentwise; a Float beside a vector broadcasts.
- **clamp**: `clamp(x: Float, low: Float, high: Float) -> Float`; `clamp(x: Int, low: Int, high: Int) -> Int`; `clamp(x: Vec2, low: Float, high: Float) -> Vec2`; `clamp(x: Vec2, low: Vec2, high: Vec2) -> Vec2`; `clamp(x: Vec3, low: Float, high: Float) -> Vec3`; `clamp(x: Vec3, low: Vec3, high: Vec3) -> Vec3`; `clamp(x: Vec4, low: Float, high: Float) -> Vec4`; `clamp(x: Vec4, low: Vec4, high: Vec4) -> Vec4`; `clamp(x: IVec2, low: Int, high: Int) -> IVec2`; `clamp(x: IVec2, low: IVec2, high: IVec2) -> IVec2`; `clamp(x: IVec3, low: Int, high: Int) -> IVec3`; `clamp(x: IVec3, low: IVec3, high: IVec3) -> IVec3`; `clamp(x: Double, low: Double, high: Double) -> Double`; `clamp(x: Long, low: Long, high: Long) -> Long`; `clamp(x: DVec2, low: Double, high: Double) -> DVec2`; `clamp(x: DVec2, low: DVec2, high: DVec2) -> DVec2`; `clamp(x: DVec3, low: Double, high: Double) -> DVec3`; `clamp(x: DVec3, low: DVec3, high: DVec3) -> DVec3`; `clamp(x: DVec4, low: Double, high: Double) -> DVec4`; `clamp(x: DVec4, low: DVec4, high: DVec4) -> DVec4`; `clamp(x: LVec2, low: Long, high: Long) -> LVec2`; `clamp(x: LVec2, low: LVec2, high: LVec2) -> LVec2`; `clamp(x: LVec3, low: Long, high: Long) -> LVec3`; `clamp(x: LVec3, low: LVec3, high: LVec3) -> LVec3` x held between low and high, componentwise; Float bounds broadcast.
- **mix**: `mix(a: Float, b: Float, t: Float) -> Float`; `mix(a: Vec2, b: Vec2, t: Float) -> Vec2`; `mix(a: Vec2, b: Vec2, t: Vec2) -> Vec2`; `mix(a: Vec3, b: Vec3, t: Float) -> Vec3`; `mix(a: Vec3, b: Vec3, t: Vec3) -> Vec3`; `mix(a: Vec4, b: Vec4, t: Float) -> Vec4`; `mix(a: Vec4, b: Vec4, t: Vec4) -> Vec4`; `mix(a: Double, b: Double, t: Double) -> Double`; `mix(a: DVec2, b: DVec2, t: Double) -> DVec2`; `mix(a: DVec2, b: DVec2, t: DVec2) -> DVec2`; `mix(a: DVec3, b: DVec3, t: Double) -> DVec3`; `mix(a: DVec3, b: DVec3, t: DVec3) -> DVec3`; `mix(a: DVec4, b: DVec4, t: Double) -> DVec4`; `mix(a: DVec4, b: DVec4, t: DVec4) -> DVec4` a + (b - a) * t, componentwise; t a Float or componentwise.
- **add_wrap**: `add_wrap(a: Int, b: Int) -> Int`; `add_wrap(a: Long, b: Long) -> Long` a + b modulo 2^32 on Ints: never a failure, unlike +.
- **sub_wrap**: `sub_wrap(a: Int, b: Int) -> Int`; `sub_wrap(a: Long, b: Long) -> Long` a - b modulo 2^32 on Ints: never a failure, unlike -.
- **mul_wrap**: `mul_wrap(a: Int, b: Int) -> Int`; `mul_wrap(a: Long, b: Long) -> Long` The low 32 bits of a * b on Ints: never a failure, unlike *.
- **float**: `float(x: Int) -> Float`; `float(x: Bool) -> Float`; `float(x: IVec2) -> Vec2`; `float(x: IVec3) -> Vec3`; `float(x: Double) -> Float`; `float(x: Long) -> Float`; `float(x: DVec2) -> Vec2`; `float(x: DVec3) -> Vec3`; `float(x: DVec4) -> Vec4`; `float(x: LVec2) -> Vec2`; `float(x: LVec3) -> Vec3`; `float(x: DMat2) -> Mat2`; `float(x: DMat3) -> Mat3`; `float(x: DMat4) -> Mat4` An Int, a Double, a Long, or a Bool as 0 or 1, as a Float; a vector or matrix componentwise.
- **int**: `int(x: Float) -> Int`; `int(x: Bool) -> Int`; `int(x: Vec2) -> IVec2`; `int(x: Vec3) -> IVec3`; `int(x: Double) -> Int`; `int(x: Long) -> Int`; `int(x: DVec2) -> IVec2`; `int(x: DVec3) -> IVec3`; `int(x: LVec2) -> IVec2`; `int(x: LVec3) -> IVec3` A float truncated toward zero, a Long wrapped to 32 bits, or a Bool as 0 or 1, as an Int; a vector componentwise.
- **double**: `double(x: Float) -> Double`; `double(x: Int) -> Double`; `double(x: Bool) -> Double`; `double(x: Vec2) -> DVec2`; `double(x: Vec3) -> DVec3`; `double(x: Vec4) -> DVec4`; `double(x: IVec2) -> DVec2`; `double(x: IVec3) -> DVec3`; `double(x: Mat2) -> DMat2`; `double(x: Mat3) -> DMat3`; `double(x: Mat4) -> DMat4`; `double(x: Long) -> Double`; `double(x: LVec2) -> DVec2`; `double(x: LVec3) -> DVec3` A Float, an Int, a Long, or a Bool as 0 or 1, as a Double; a vector or matrix componentwise.
- **long**: `long(x: Float) -> Long`; `long(x: Int) -> Long`; `long(x: Bool) -> Long`; `long(x: Vec2) -> LVec2`; `long(x: Vec3) -> LVec3`; `long(x: IVec2) -> LVec2`; `long(x: IVec3) -> LVec3`; `long(x: Double) -> Long`; `long(x: DVec2) -> LVec2`; `long(x: DVec3) -> LVec3` A float truncated toward zero, an Int, or a Bool as 0 or 1, as a Long; a vector componentwise.
- **pow**: `pow(base: Float, exponent: Float) -> Float`; `pow(base: Vec2, exponent: Float) -> Vec2`; `pow(base: Vec2, exponent: Vec2) -> Vec2`; `pow(base: Vec3, exponent: Float) -> Vec3`; `pow(base: Vec3, exponent: Vec3) -> Vec3`; `pow(base: Vec4, exponent: Float) -> Vec4`; `pow(base: Vec4, exponent: Vec4) -> Vec4`; `pow(base: Double, exponent: Double) -> Double`; `pow(base: DVec2, exponent: Double) -> DVec2`; `pow(base: DVec2, exponent: DVec2) -> DVec2`; `pow(base: DVec3, exponent: Double) -> DVec3`; `pow(base: DVec3, exponent: DVec3) -> DVec3`; `pow(base: DVec4, exponent: Double) -> DVec4`; `pow(base: DVec4, exponent: DVec4) -> DVec4` base raised to exponent, componentwise; a Float exponent broadcasts.
- **exp**: `exp(x: Float) -> Float`; `exp(x: Vec2) -> Vec2`; `exp(x: Vec3) -> Vec3`; `exp(x: Vec4) -> Vec4`; `exp(x: Double) -> Double`; `exp(x: DVec2) -> DVec2`; `exp(x: DVec3) -> DVec3`; `exp(x: DVec4) -> DVec4` e raised to x, componentwise.
- **step**: `step(edge: Float, x: Float) -> Float`; `step(edge: Float, x: Vec2) -> Vec2`; `step(edge: Float, x: Vec3) -> Vec3`; `step(edge: Float, x: Vec4) -> Vec4`; `step(edge: Vec2, x: Vec2) -> Vec2`; `step(edge: Vec3, x: Vec3) -> Vec3`; `step(edge: Vec4, x: Vec4) -> Vec4`; `step(edge: Double, x: Double) -> Double`; `step(edge: Double, x: DVec2) -> DVec2`; `step(edge: Double, x: DVec3) -> DVec3`; `step(edge: Double, x: DVec4) -> DVec4`; `step(edge: DVec2, x: DVec2) -> DVec2`; `step(edge: DVec3, x: DVec3) -> DVec3`; `step(edge: DVec4, x: DVec4) -> DVec4` 1 where x >= edge, else 0, componentwise; a Float edge broadcasts.
- **smoothstep**: `smoothstep(low: Float, high: Float, x: Float) -> Float`; `smoothstep(low: Float, high: Float, x: Vec2) -> Vec2`; `smoothstep(low: Float, high: Float, x: Vec3) -> Vec3`; `smoothstep(low: Float, high: Float, x: Vec4) -> Vec4`; `smoothstep(low: Vec2, high: Vec2, x: Vec2) -> Vec2`; `smoothstep(low: Vec3, high: Vec3, x: Vec3) -> Vec3`; `smoothstep(low: Vec4, high: Vec4, x: Vec4) -> Vec4`; `smoothstep(low: Double, high: Double, x: Double) -> Double`; `smoothstep(low: Double, high: Double, x: DVec2) -> DVec2`; `smoothstep(low: Double, high: Double, x: DVec3) -> DVec3`; `smoothstep(low: Double, high: Double, x: DVec4) -> DVec4`; `smoothstep(low: DVec2, high: DVec2, x: DVec2) -> DVec2`; `smoothstep(low: DVec3, high: DVec3, x: DVec3) -> DVec3`; `smoothstep(low: DVec4, high: DVec4, x: DVec4) -> DVec4` 0 below low, 1 above high, a smooth curve between, componentwise; Float edges broadcast.
- **atan2**: `atan2(y: Float, x: Float) -> Float`; `atan2(y: Double, x: Double) -> Double` The angle of the point (x, y) from the x axis, in -pi..pi.
- **trunc**: `trunc(x: Float) -> Float`; `trunc(x: Vec2) -> Vec2`; `trunc(x: Vec3) -> Vec3`; `trunc(x: Vec4) -> Vec4`; `trunc(x: Double) -> Double`; `trunc(x: DVec2) -> DVec2`; `trunc(x: DVec3) -> DVec3`; `trunc(x: DVec4) -> DVec4` The integer part of x, toward zero, componentwise.
- **exp2**: `exp2(x: Float) -> Float`; `exp2(x: Vec2) -> Vec2`; `exp2(x: Vec3) -> Vec3`; `exp2(x: Vec4) -> Vec4`; `exp2(x: Double) -> Double`; `exp2(x: DVec2) -> DVec2`; `exp2(x: DVec3) -> DVec3`; `exp2(x: DVec4) -> DVec4` 2 raised to x, componentwise.
- **log2**: `log2(x: Float) -> Float`; `log2(x: Vec2) -> Vec2`; `log2(x: Vec3) -> Vec3`; `log2(x: Vec4) -> Vec4`; `log2(x: Double) -> Double`; `log2(x: DVec2) -> DVec2`; `log2(x: DVec3) -> DVec3`; `log2(x: DVec4) -> DVec4` The base-2 logarithm, componentwise.
- **sinh**: `sinh(x: Float) -> Float`; `sinh(x: Vec2) -> Vec2`; `sinh(x: Vec3) -> Vec3`; `sinh(x: Vec4) -> Vec4`; `sinh(x: Double) -> Double`; `sinh(x: DVec2) -> DVec2`; `sinh(x: DVec3) -> DVec3`; `sinh(x: DVec4) -> DVec4` The hyperbolic sine, componentwise.
- **cosh**: `cosh(x: Float) -> Float`; `cosh(x: Vec2) -> Vec2`; `cosh(x: Vec3) -> Vec3`; `cosh(x: Vec4) -> Vec4`; `cosh(x: Double) -> Double`; `cosh(x: DVec2) -> DVec2`; `cosh(x: DVec3) -> DVec3`; `cosh(x: DVec4) -> DVec4` The hyperbolic cosine, componentwise.
- **tanh**: `tanh(x: Float) -> Float`; `tanh(x: Vec2) -> Vec2`; `tanh(x: Vec3) -> Vec3`; `tanh(x: Vec4) -> Vec4`; `tanh(x: Double) -> Double`; `tanh(x: DVec2) -> DVec2`; `tanh(x: DVec3) -> DVec3`; `tanh(x: DVec4) -> DVec4` The hyperbolic tangent, componentwise.
- **fma**: `fma(a: Float, b: Float, c: Float) -> Float`; `fma(a: Vec2, b: Vec2, c: Vec2) -> Vec2`; `fma(a: Vec3, b: Vec3, c: Vec3) -> Vec3`; `fma(a: Vec4, b: Vec4, c: Vec4) -> Vec4`; `fma(a: Double, b: Double, c: Double) -> Double`; `fma(a: DVec2, b: DVec2, c: DVec2) -> DVec2`; `fma(a: DVec3, b: DVec3, c: DVec3) -> DVec3`; `fma(a: DVec4, b: DVec4, c: DVec4) -> DVec4` a * b + c rounded once, componentwise.
- **popcount**: `popcount(x: Int) -> Int` How many bits of x are one.
- **clz**: `clz(x: Int) -> Int` How many zero bits lead x; 32 for zero.
- **ctz**: `ctz(x: Int) -> Int` How many zero bits trail x; 32 for zero.
- **rotl**: `rotl(x: Int, n: Int) -> Int` x's bits rotated left by n modulo 32.
- **rotr**: `rotr(x: Int, n: Int) -> Int` x's bits rotated right by n modulo 32.
- **ult**: `ult(a: Int, b: Int) -> Bool` Whether a is below b as unsigned 32-bit values.
- **udiv**: `udiv(a: Int, b: Int) -> Int` The quotient of a and b as unsigned 32-bit values; a zero b is a failure, like /.
- **urem**: `urem(a: Int, b: Int) -> Int` The remainder of a and b as unsigned 32-bit values; a zero b is a failure, like %.
- **mulhi**: `mulhi(a: Int, b: Int) -> Int` The high 32 bits of the signed 64-bit product of a and b.
- **umulhi**: `umulhi(a: Int, b: Int) -> Int` The high 32 bits of the unsigned 64-bit product of a and b.
- **bits**: `bits(x: Float) -> Int` x's 32 bits as an Int, unchanged: for a Float of no sign, the Ints order as the Floats do.
- **from_bits**: `from_bits(x: Int) -> Float` x's 32 bits as a Float, unchanged; bits(from_bits(x)) is x.
- **saturate**: `saturate(x: Float) -> Float`; `saturate(x: Vec2) -> Vec2`; `saturate(x: Vec3) -> Vec3`; `saturate(x: Vec4) -> Vec4` x held between 0 and 1, componentwise: clamp(x, 0.0, 1.0).
- **dot**: `dot(a: Vec2, b: Vec2) -> Float`; `dot(a: Vec3, b: Vec3) -> Float`; `dot(a: Vec4, b: Vec4) -> Float`; `dot(a: DVec3, b: DVec3) -> Double` The dot product of two vectors of one size: the componentwise product, summed first to last.
- **cross**: `cross(a: Vec3, b: Vec3) -> Vec3`; `cross(a: DVec3, b: DVec3) -> DVec3` The cross product of two three-component vectors.
- **length**: `length(v: Vec2) -> Float`; `length(v: Vec3) -> Float`; `length(v: Vec4) -> Float`; `length(v: DVec3) -> Double` The Euclidean length of a vector: sqrt(dot(v, v)).
- **normalize**: `normalize(v: Vec2) -> Vec2`; `normalize(v: Vec3) -> Vec3`; `normalize(v: Vec4) -> Vec4`; `normalize(v: DVec3) -> DVec3` The unit vector in v's direction; the zero vector stays zero.
- **reflect**: `reflect(incident: Vec3, normal: Vec3) -> Vec3`; `reflect(incident: Vec2, normal: Vec2) -> Vec2`; `reflect(incident: DVec3, normal: DVec3) -> DVec3` incident reflected off the surface with unit normal normal: incident - 2 * dot(normal, incident) * normal.
- **transpose**: `transpose(m: Mat2) -> Mat2`; `transpose(m: Mat3) -> Mat3`; `transpose(m: Mat4) -> Mat4`; `transpose(m: DMat4) -> DMat4` The matrix with its rows and columns exchanged.
- **determinant**: `determinant(m: Mat2) -> Float`; `determinant(m: Mat3) -> Float`; `determinant(m: Mat4) -> Float`; `determinant(m: DMat4) -> Double` The determinant of a square matrix, expanded along the first row.
- **inverse**: `inverse(m: Mat2) -> Mat2`; `inverse(m: Mat3) -> Mat3`; `inverse(m: Mat4) -> Mat4`; `inverse(m: DMat4) -> DMat4` The matrix that multiplies m to the identity: its adjugate over its determinant; a singular m gives no finite numbers.
- **distance**: `distance(a: Vec2, b: Vec2) -> Float`; `distance(a: Vec3, b: Vec3) -> Float`; `distance(a: Vec4, b: Vec4) -> Float` The length of a - b.
- **radians**: `radians(x: Float) -> Float`; `radians(x: Vec2) -> Vec2`; `radians(x: Vec3) -> Vec3`; `radians(x: Vec4) -> Vec4` Degrees to radians, componentwise.
- **degrees**: `degrees(x: Float) -> Float`; `degrees(x: Vec2) -> Vec2`; `degrees(x: Vec3) -> Vec3`; `degrees(x: Vec4) -> Vec4` Radians to degrees, componentwise.
- **inversesqrt**: `inversesqrt(x: Float) -> Float`; `inversesqrt(x: Vec2) -> Vec2`; `inversesqrt(x: Vec3) -> Vec3`; `inversesqrt(x: Vec4) -> Vec4` 1 / sqrt(x), componentwise.
- **mod**: `mod(x: Float, y: Float) -> Float`; `mod(x: Vec2, y: Vec2) -> Vec2`; `mod(x: Vec2, y: Float) -> Vec2`; `mod(x: Vec3, y: Vec3) -> Vec3`; `mod(x: Vec3, y: Float) -> Vec3`; `mod(x: Vec4, y: Vec4) -> Vec4`; `mod(x: Vec4, y: Float) -> Vec4` x - y * floor(x / y): the remainder with y's sign, for wrapping coordinates, where % truncates toward zero.
- **isnan**: `isnan(x: Float) -> Bool` Whether x is not a number.
- **isinf**: `isinf(x: Float) -> Bool` Whether x is infinite.
- **refract**: `refract(incident: Vec2, normal: Vec2, eta: Float) -> Vec2`; `refract(incident: Vec3, normal: Vec3, eta: Float) -> Vec3`; `refract(incident: Vec4, normal: Vec4, eta: Float) -> Vec4` incident bent through a surface with unit normal normal by the ratio of indices eta; the zero vector past total internal reflection.
- **at**: `at(field: Rows<T>, index: Int) -> T`; `at(field: Grid<T>, at: Vec2) -> T`; `at(field: Grid<T>, at: IVec2) -> T`; `at(field: Volume<T>, at: Vec3) -> T`; `at(field: Volume<T>, at: IVec3) -> T` The element at an address: the index, or the coordinates floored, clamped into the Field (NaN as zero). A memory reads as the steps have left it, and as it was bound in a scan's initial value. Needs `field.read`.
- **fetch**: `fetch(field: Rows<T>, index: Int) -> T`; `fetch(field: Grid<T>, at: Vec2) -> T`; `fetch(field: Grid<T>, at: IVec2) -> T`; `fetch(field: Volume<T>, at: Vec3) -> T`; `fetch(field: Volume<T>, at: IVec3) -> T` Another name for at. Needs `field.read`.
- **at_linear**: `at_linear(field: Rows<T>, position: Float) -> T`; `at_linear(field: Grid<T>, position: Vec2) -> T`; `at_linear(field: Volume<T>, position: Vec3) -> T` The neighbouring elements blended by the fractional position (element centres at .5), T of floats only; clamped at the edges. Needs `field.read`.
- **sample**: `sample(field: Rows<T>) -> T`; `sample(field: Grid<T>) -> T`; `sample(field: Volume<T>) -> T` The input's element at this element's own coordinates, clamped into the input. Needs `field.read`.
- **count**: `count(field: Rows<T>) -> Int`; `count(field: Grid<T>) -> Int`; `count(field: Volume<T>) -> Int` How many elements the Field has. Needs `field.read`.
- **size**: `size(field: Grid<T>) -> Vec2`; `size(field: Volume<T>) -> Vec3` The Field's extent along each axis. Needs `field.read`.
- **isize**: `isize(field: Rows<T>) -> Int`; `isize(field: Grid<T>) -> IVec2`; `isize(field: Volume<T>) -> IVec3` The Field's extent along each axis, as integers. Needs `field.read`.
- **put**: `put(memory: Rows<T>, index: Int, value: T) -> T`; `put(memory: Grid<T>, at: Vec2, value: T) -> T`; `put(memory: Grid<T>, at: IVec2, value: T) -> T`; `put(memory: Volume<T>, at: Vec3, value: T) -> T`; `put(memory: Volume<T>, at: IVec3, value: T) -> T` Writes an element of a memory port in step order (scan only; from the entry or a function with the memory as a parameter); an address outside is dropped, never clamped. Yields the value written. Needs `field.write`.
- **emit**: `emit(pair: P) -> P` Gives a scatter one more pair, a struct of an address into its target and a value of its element, as it returns one: an element's pairs are the ones it emits, in order, then the one it returns, and an address outside the target is dropped. Scatter only, from its body, under `min`, `max`, or a `sum` of integers, whose answer is the same in any order. Needs `field.write`.
- **element**: `element() -> Int` This element's index over rows; a scan's step.
- **coord**: `coord() -> Vec2`; `coord() -> Vec3` This element's column and row (and layer) over a grid or a volume, whole numbers from 0 with row 0 at the top; the element's centre is coord() + 0.5. Rows have element().
- **icoord**: `icoord() -> IVec2`; `icoord() -> IVec3` This element's coordinates as integers, for arithmetic on cells.
- **domain**: `domain() -> Int`; `domain() -> Vec2`; `domain() -> Vec3` The output's extent in its shape.
- **idomain**: `idomain() -> Int`; `idomain() -> IVec2`; `idomain() -> IVec3` The output's extent as integers.
- **index**: `index() -> Int` The iteration of the innermost loop, or a scan's step outside every loop.
- **time**: `time() -> Float` The session's time in seconds. Needs `time.read`.
- **rate**: `rate() -> Float` The sound device's samples per second, zero without one. Needs `time.read`.
- **window**: `window() -> Vec2` The window's size in physical pixels, zero by zero without one; a node's domain may take it. Needs `input.read`.
- **scale**: `scale() -> Float` The window's physical pixels per logical pixel, one without a window. Needs `input.read`.
- **pointer**: `pointer() -> Vec2` The pointer in the output's pixels from its top left; a player maps its window's into them. Needs `input.read`.
- **pointer_down**: `pointer_down() -> Bool` Whether the left button is held. Needs `input.read`.
- **button**: `button(n: Int) -> Bool` Whether button n is held: 0 left, 1 right, 2 middle of a pointer; a pad's from 8 (south, east, west, north, shoulders, thumbs, start, select). Needs `input.read`.
- **stick**: `stick(n: Int) -> Vec2` A pad's axis pair: 0 the left stick, 1 the right, 2 the two triggers (x left, y right). Sticks run -1 to 1 with y away from the person, triggers 0 to 1; all zero without a pad. Needs `input.read`.
- **wheel**: `wheel() -> Vec2` What the wheel scrolled since the last evaluation, in lines: x sideways, y positive away. Needs `input.read`.
- **key**: `key(code: Int) -> Bool` Whether the key with this code is held: letters and digits by uppercase ASCII, arrows 37 to 40, space 32, enter 13, shift 16, escape 27. Needs `input.read`.
- **vec2**: `vec2(x: Float, y: Float) -> Vec2`; `vec2(x: Int, y: Int) -> Vec2` A vector from components, Float or Int (widened), in order; v.xy, v.yx and the like are swizzles.
- **vec3**: `vec3(x: Float, y: Float, z: Float) -> Vec3`; `vec3(x: Int, y: Int, z: Int) -> Vec3`; `vec3(xy: Vec2, z: Float) -> Vec3`; `vec3(x: Float, yz: Vec2) -> Vec3` A vector from components, Float, Int (widened) or shorter vectors, in order; v.xyz, v.zyx and the like are swizzles.
- **vec4**: `vec4(x: Float, y: Float, z: Float, w: Float) -> Vec4`; `vec4(x: Int, y: Int, z: Int, w: Int) -> Vec4`; `vec4(xyz: Vec3, w: Float) -> Vec4`; `vec4(xy: Vec2, zw: Vec2) -> Vec4`; `vec4(xy: Vec2, z: Float, w: Float) -> Vec4`; `vec4(x: Float, yzw: Vec3) -> Vec4` A vector from components, Float, Int (widened) or shorter vectors, in order; v.xyz, v.wzyx, c.rgb and the like are swizzles.
- **ivec2**: `ivec2(x: Int, y: Int) -> IVec2` An integer vector from components: a cell, an offset; at and put take one as an address; v.yx and the like are swizzles.
- **ivec3**: `ivec3(x: Int, y: Int, z: Int) -> IVec3`; `ivec3(xy: IVec2, z: Int) -> IVec3`; `ivec3(x: Int, yz: IVec2) -> IVec3` An integer vector from components or a shorter integer vector: a cell, an offset; at and put take one as an address.
- **mat2**: `mat2(c0: Vec2, c1: Vec2) -> Mat2`; `mat2(diagonal: Float) -> Mat2` A matrix from its columns, or with one value down the diagonal and zeros elsewhere; m[i] is a column, m * v applies it.
- **mat3**: `mat3(c0: Vec3, c1: Vec3, c2: Vec3) -> Mat3`; `mat3(diagonal: Float) -> Mat3` A matrix from its columns, or with one value down the diagonal and zeros elsewhere; m[i] is a column, m * v applies it.
- **mat4**: `mat4(c0: Vec4, c1: Vec4, c2: Vec4, c3: Vec4) -> Mat4`; `mat4(diagonal: Float) -> Mat4` A matrix from its columns, or with one value down the diagonal and zeros elsewhere; m[i] is a column, m * v applies it.
- **dvec2**: `dvec2(x: Double, y: Double) -> DVec2` A vector of doubles from components or shorter vectors, any number widened; a written number is taken exact, so dvec2(0.1, 0.2) holds the doubles nearest a tenth and a fifth.
- **dvec3**: `dvec3(x: Double, y: Double, z: Double) -> DVec3`; `dvec3(xy: DVec2, z: Double) -> DVec3`; `dvec3(v: Vec3) -> DVec3` A vector of doubles from components or shorter vectors, any number widened; a written number is taken exact.
- **dvec4**: `dvec4(x: Double, y: Double, z: Double, w: Double) -> DVec4`; `dvec4(xyz: DVec3, w: Double) -> DVec4`; `dvec4(v: Vec4) -> DVec4` A vector of doubles from components or shorter vectors, any number widened; a written number is taken exact.
- **lvec2**: `lvec2(x: Long, y: Long) -> LVec2`; `lvec2(v: IVec2) -> LVec2` A vector of longs from components or shorter integer vectors, an Int widened.
- **lvec3**: `lvec3(x: Long, y: Long, z: Long) -> LVec3`; `lvec3(xy: LVec2, z: Long) -> LVec3`; `lvec3(v: IVec3) -> LVec3` A vector of longs from components or shorter integer vectors, an Int widened.
- **dmat2**: `dmat2(c0: DVec2, c1: DVec2) -> DMat2`; `dmat2(diagonal: Double) -> DMat2` A matrix of doubles from its columns, or with one value down the diagonal; m[i] is a column, m * v applies it.
- **dmat3**: `dmat3(c0: DVec3, c1: DVec3, c2: DVec3) -> DMat3`; `dmat3(diagonal: Double) -> DMat3` A matrix of doubles from its columns, or with one value down the diagonal; m[i] is a column, m * v applies it.
- **dmat4**: `dmat4(c0: DVec4, c1: DVec4, c2: DVec4, c3: DVec4) -> DMat4`; `dmat4(diagonal: Double) -> DMat4` A matrix of doubles from its columns, or with one value down the diagonal; m[i] is a column, m * v applies it.

### kernel: how a Kernel is written: its declarations and expressions

- **requires**: `requires { field.read; field.write; time.read; input.read; }` The capabilities a Kernel uses, and no others: reading inputs, writing outputs, the clock, the pointer, keys and window.
- **input**: `input x: Image;   = "atlas.png";   = text "ORLA";   = frames "clip/" at 30;` In a Kernel, a Field it reads, bound by the node that runs it (`source <- other.result;`); in a scene, a Field the caller supplies, or one the Host reads from a file, from text, or frame by frame from a directory.
- **param**: `param gain: Float = 1.0;` In a Kernel, a value the node that runs it may set (`gain = 0.5;`), with its default; in a scene, a value the caller sets (`--param`, the protocol's `param`) and its nodes read by name.
- **output**: `output result: Grid<Vec4>;   output shown <- node.port;   output kept <- node.port to "saves/n.field";` In a Kernel, a Field it writes, an element per element of its domain (a map may write several); in a scene, a named result: the first picture is what it shows, and one written `to` a file is saved after each evaluation that changed it, where allowed.
- **type**: `type Cells = Grid<Int>;   struct Cell { alive: Int; age: Float; }` A name for a port type, or a struct of named fields, shared by the Kernels of a document.
- **let**: `let name = expr;   let { x, y: top } = value;` A binding, immutable and scoped to its block; `let _ = put(...)` is a statement for its effect. A `{ ... }` names fields of one value instead, each under its own name or the one written after `:`.
- **if**: `if condition { a } else { b }   if condition { return a; }` Both arms give one type and only the arm taken runs, which a trace shows; with no `else`, among a block's statements, it guards, and the rest of the block is its other arm.
- **and**: `a && b   a || b   !a` Truth: `&&` and `||` ask the other side only where the first settles nothing, so a guard guards and what it guards does not run.
- **match**: `match i { 0 => a, 1 | 2 => b, 8..16 => c, 16..=31 => d, _ => e }` On an Int: the arm whose values hold it, else the `_` arm, which comes last and is required; the GPU runs it as a switch. A range is the values it holds, `..` stopping before its end and `..=` taking it, at most 256 to an arm.
- **loop**: `loop n with x = init until done { next }` A bounded fold: `n` is a literal, which is its bound, or any Int expression read once and clamped to 0..=65536; `index()` is the iteration, and `until` stops it early, as `break` does from inside.
- **break**: `loop n with x = init { if done { break x; } next }` Among a loop body's statements: the loop stops after this iteration and gives that value, which is of the type it carries; the iterations after it do not run, and a `break` belongs to the innermost loop around it.
- **each**: `[each n { body }]` An array built element by element, unrolled, with `index()` its position; a loop inside the body has its own `index()`.
- **array**: `[a, b, c]   [value; n]   a[i]` A fixed-size array of one element type, or `n` copies of a value; `a[i]` takes an element, clamped, and a literal index is checked when compiled.
- **struct**: `Name { a: 1.0, b: 2 }   Name { ..base, a: 3.0 }` A value of a declared struct, or a copy of one with some fields replaced.
- **fn**: `fn name(a: Float) -> Float { ... }   fn name<T>(a: T) -> T { ... }` A function of the document or the Kernel; it calls only what is declared above it, and so never itself, and may take a Field parameter. A `<T>` names a type parameter, bound by the type of the argument written for it, and one instance is compiled per set of types a call gives it.
- **kernel**: `kernel id@version { requires { ... } input ... output ... compute { ... } }` A Kernel: what it needs, its ports and parameters, and one of `compute`, `scan`, `scatter` or `prefix`.
- **budget**: `budget 100000;` The loop iterations one element may run (a step of a scan, a merge of a reduction), 16777216 unless declared, a loop's added as it finishes; an element past it fails the evaluation on every backend, as an overflow does, and the running system keeps its last state. A run of all the elements together is bounded by the Host (`--work`).
- **compute**: `compute { return value; }` The map mode: the body runs once per element of the node's domain and gives that element.
- **merge**: `compute { ... } merge(a, b) { ... }` The reduce mode: `compute` runs per element of the first input and `merge` folds the results pairwise into one element; the output may be written as its element type.
- **scan**: `scan with y = init { return next; }` The ordered mode: one element per element of the first input, in order, carrying `y` from the step before; `memory` ports are read and written step by step.
- **prefix**: `prefix with sum { return value; }` The prefix mode: the body runs once per element of the first input, a `Rows`, and that element of the output is its value combined with every value before it by `sum`, `min` or `max`. The rule must give one answer whatever order it is folded in, so a float `sum` is refused (a `scan` walks that in element order) and a device folds the row in blocks rather than walking it.
- **scatter**: `scatter into target with sum { return Pair { at: address, value: v }; }` The scatter mode: the body runs once per element of the node's domain and gives a struct the document declares, of two fields, an address into the target and a value of its element (`struct Pair { at: Int; value: Float; }`); the value is combined into the target at the address with `sum`, `min`, `max` or `last`, in element order.
- **memory**: `memory ram: Rows<Int>;   put(ram, address, value)` A port an ordered Kernel both reads and writes, step by step; it comes out as an output.
- **reserved**: `let return if else match loop with until break kernel struct type node use from input output memory graph merge scan scatter prefix migrate package` The words the language keeps for itself: a Kernel, node, port, param, function, struct or binding may not be named one.

## graph: how a scene composes Kernels into a running world

### scene: how a scene is written: nodes, bindings, inputs and packages

- **node**: `node name: fx.glow@1 { port <- source.result; param = 2; domain = vec2(w, h); }` A node runs a Kernel: its inputs wired at the consumer, its parameters set, its extent explicit or the first input's.
- **inline**: `node name: kernel { ... input x: Image <- src; ... }` A Kernel written in place; its inputs carry their wiring, a param whose default names a scene param follows that param (`param gain: Float = amount;`), and the session publishes it under the document's name.
- **fill**: `node zeros: Rows<Float> { domain = 65; value = 0.5; }` A node that is a Field rather than a Kernel: its element's zero, or the `value` given, over the domain; the type may be an alias a used document declares.
- **graph**: `node name: graph "other.osn" { source <- src; param = 2; }` Another document placed as a node: its nodes join under `name.`, its inputs take the wiring, its outputs are read as `name.output`.
- **previous**: `port <~ node.output from initial [migrate id@v];` State: the port reads what the last evaluation committed, the initial source before the first, and a migration carries it when a republished Kernel changes its layout.
- **times**: `times = 5;   carry port <- output;` A node written once and placed N times, each wired as written, `carry` feeding a port from the previous time's output; at N = 0, a carried output is what its port is first bound to.
- **async**: `async = true;` The node runs off the frame: the evaluation starts it on a thread and goes on, its readers taking the last output until the run lands.
- **use**: `use "kernels/";   use "name.orlapkg";` The documents whose Kernels and declarations this one brings in: a file, a directory, or a package.
- **package**: `package name@1 { language = 1; ir = 1; }` The manifest of a package directory: what it is called and the least language and IR it needs.
- **vec2**: `vec2(w, h)` A grid's extent.
- **vec3**: `vec3(w, h, d)` A volume's extent.
- **count**: `count(port)` How many elements an input port of this node holds.
- **size**: `size(port)` An input port's extent: a count, a pair or a triple.
- **value**: `value(port)   value(port, i)` A number a Kernel wrote: element i (0 when not given) of an input port of this node, read as the node binds.
- **window**: `window()` The window's size in pixels; the node runs again at the new extent when it changes.
- **scale**: `scale()` The window's pixels per logical pixel.
- **max**: `max(a, b)` The larger, componentwise.
- **min**: `min(a, b)` The smaller, componentwise.
- **floor**: `floor(x)` A measure made whole, rounding down.
- **ceil**: `ceil(x)` A measure made whole, rounding up.

## capabilities: what the world grants a Kernel, and what it already knows (`search`)

### authority: the capabilities a Kernel declares in `requires`, each what it grants

- **field.read**: `field.read` Read the elements and extents of the Fields bound to its inputs, images included.
- **field.write**: `field.write` Create and write its output Fields, images included.
- **time.read**: `time.read` Read the session clock and the sound rate.
- **input.read**: `input.read` Read the pointer, buttons, keys, pad and wheel, and the window's pixel size and scale.

### packages: what a package holds and how a scene reaches one; `search` finds what they hold

- **manifest**: `name.orlapkg/package.osn: package name@1 { language = 1; ir = 1; }` What the directory is called, and the least language and IR it needs: one this Orla does not have refuses the package, naming both.
- **kernels**: `use "kernels/";   (in the manifest)` The documents the manifest uses, inside the directory, are the package's Kernels; a scene that uses the package publishes every one of them.
- **graphs**: `name.orlapkg/graphs/*.osn` Preset graphs, placed with `node n: graph "name.orlapkg/graphs/g.osn" { ... }`; `search` lists each with its inputs, params and outputs.
- **docs**: `//! at the top of a manifest or a preset;   /// above a Kernel, a port or a param` What each part is for, in its source: `search` matches the words and answers them.
- **assets**: `name.orlapkg/font.png` Any other file in the directory, which a scene's `input` reads; a package's files are readable wherever reads are held.
- **by name**: `use "name.orlapkg";` A package not beside the scene is found by its name in each directory of `ORLA_PACKAGES`, then in `packages/` beside the program; `search` writes such a package's `use` by its name.

## doors: how an agent reaches the surfaces

### protocol: the operations a session answers

- **status**: `status` {} The session at a glance. Answers backend, scene, watched, kernels (published Kernels), natives (the backend's native operations), nodes, loaded (whether a scene is), refused (why not, when not), revision (moved by every change but running), frame (the frames committed since that revision), fusion, grants, versions (`orla`, `api`, `language`, `ir`, `package`).
- **kernels**: `kernels` {} Every published Kernel, with its generation. Answers a list of manifests, each with its generation, which `publish`'s `expect` names.
- **kernel**: `kernel` {`kernel`?: string, `generation`?: integer} One Kernel's manifest and source. Answers manifest, generation, source, as `publish` takes it.
- **graph**: `graph` {} The running scene's structure. Answers inputs, params, nodes, outputs.
- **document**: `document` {`version`?: string} The scene document's text. Answers path, text, and version, which `compose`'s `expect` names.
- **history**: `history` {`since`?: integer} What changed the world, and what was refused. Answers revision, frame (the frames committed since that revision), the changes after `since` and the attempts refused since, each with its origin.
- **compose**: `compose` {`text`?: string, `edits`?: [{node?: string, output?: string, param?: string, input?: string, kernel?: string, struct?: string, type?: string, fn?: string, use?: string, member?: string, text: string | null}], `expect`?: string} Changes the scene: rewrites its document on disk, the file at `path` (`document` and `status` name it), once the text is checked; answered once the scene stands. A refused text writes nothing and is an error: why, at `line` and `column`, the document's `version`, and for `edits` the `text` they made, which `line` counts in. Answers events, version, path (the document), written.
- **sync**: `sync` {} Reads every changed document again. Answers events.
- **prepare**: `prepare` {`source`: string} Compiles a Kernel and says what is wrong with it; publishes nothing. Answers id, manifest; or an error at a line and column.
- **publish**: `publish` {`source`: string, `expect`?: "absent" | "any" | integer} Compiles, validates and prepares a Kernel, then swaps it in whole; answered once it stands. The running world changes, not a file: the document it came from is left as it is. Answers id, generation; and when nodes running it start state, `keeps` (those `node.port`s, which keep what they committed) and `reset` (the requests that start them from the new law).
- **retire**: `retire` {`kernel`?: string} Removes a published Kernel. Answers retired.
- **param**: `param` {`name`: string, `value`: value} Sets a parameter the scene declares. Answers name, value, events; a node's own value is refused, and changes by `compose`.
- **evaluate**: `evaluate` {`nodes`?: [string], `settle`?: bool, `time`?: number, `rate`?: number, `pointer`?: [x, y], `down`?: bool, `buttons`?: [integer], `wheel`?: [x, y], `sticks`?: [[x, y]], `keys`?: [integer], `window`?: [w, h], `scale`?: number} Runs one frame. Answers executed, reused, committed, outputs.
- **advance**: `advance` {`frames`: integer, `fps`?: number, `until`?: {target: string, below?: number, above?: number, equals?: number}, `save`?: {node.port: string}, `time`?: number, `rate`?: number, `pointer`?: [x, y], `down`?: bool, `buttons`?: [integer], `wheel`?: [x, y], `sticks`?: [[x, y]], `keys`?: [integer], `window`?: [w, h], `scale`?: number} Runs many frames, until a count or a value says to stop. Answers frames, stopped, target, value, saved, millis, report.
- **bench**: `bench` {`frames`?: integer, `fps`?: number, `repeat`?: integer, `nodes`?: bool, `time`?: number, `rate`?: number, `pointer`?: [x, y], `down`?: bool, `buttons`?: [integer], `wheel`?: [x, y], `sticks`?: [[x, y]], `keys`?: [integer], `window`?: [w, h], `scale`?: number} Times the scene's frames. Answers backend, frames, repeat, best_ms, median_ms, nodes, ran.
- **window**: `window` {`size`?: [w, h] | null, `scale`?: number} Sets the window a headless session reports to evaluations that name none. Answers the window.
- **read**: `read` {`output`?: string, `node`?: string, `port`?: string, `format`?: "png" | "rgba32f" | "ppm" | "summary" | "words" | "field", `from`?: integer, `count`?: integer} Reads an output or a node's port. Answers a picture: width, height, format, data (base64; over MCP the picture itself). Any other Field: shape, extent, count, schema, and its summary (each number's min, max, sum, mean, how many were not zero, and how many were not finite), or `values` from `from` (a number, or a list for an element of several), or format and data (base64).
- **write**: `write` {`node`: string, `port`?: string, `data`: string, `format`?: "words" | "field"} Replaces a node's committed state. Answers the state replaced.
- **snapshot**: `snapshot` {`to`?: string} The whole world: its Kernels, its scene and every committed state. Answers the world's revision and frame (the frames committed since it), every Kernel with its generation and source, every port's committed state; or the file written.
- **restore**: `restore` {`snapshot`?: object, `from`?: string, `laws`: "current" | "snapshot", `structure`?: "current" | "snapshot", `time`?: number, `rate`?: number, `pointer`?: [x, y], `down`?: bool, `buttons`?: [integer], `wheel`?: [x, y], `sticks`?: [[x, y]], `keys`?: [integer], `window`?: [w, h], `scale`?: number} Puts a snapshot's world back, after a trial frame that commits nothing: the state is the snapshot's, which the next frame reads; until then `read` shows the trial frame's outputs. Answers revision, frame, report.
- **reset**: `reset` {`node`?: string, `port`?: string} Forgets committed state; `<~` readers start from their initial sources. Answers what was reset.
- **help**: `help` {`section`?: string, `full`?: bool, `name`?: string} What Orla answers: this index, a section, one item by name, or with `full` everything at once. Answers the index; a section's items; one item in full; or with `full`, `reference`, the whole of it as one Markdown text. An unknown name is an error with `similar`.
- **search**: `search` {`text`?: string, `takes`?: string, `gives`?: string, `requires`?: string, `limit`?: integer} The capabilities that exist: published Kernels, and every Kernel and preset graph of the packages this session reaches. Answers capabilities, nearest first, each with kind, name, doc, package, reach (the line a scene writes to use it: an installed package by its name, any other path relative to `base`), inputs, params, outputs (name, type, doc), requires, granted, published; base, the scene document's directory; count, how many matched.
- **trace**: `trace` {`node`: string, `at`?: [x, y] | [x, y, z], `element`?: integer, `time`?: number, `rate`?: number, `pointer`?: [x, y], `down`?: bool, `buttons`?: [integer], `wheel`?: [x, y], `sticks`?: [[x, y]], `keys`?: [integer], `window`?: [w, h], `scale`?: number} Runs one element of a node's Kernel and shows every step of it. Answers kernel, at, bindings, results, calls.
- **pin**: `pin` {`node`: string} Keeps a node materialized so `read` can see it. Answers pinned.
- **unpin**: `unpin` {`node`: string} Lets fusion have a pinned node back. Answers unpinned.
- **quit**: `quit` {} Ends the connection, or on standard input the process. Answers bye.

### wire: a request and an answer as they cross a door, and the doors

- **op**: `request: `op`: string` The operation: one of the protocol section's, each with its own fields.
- **id**: `request: `id`: value` Anything; the answer carries it back as it was sent, to pair them.
- **id**: `answer: `id`: value` The request's `id`, or null when it sent none or was not JSON.
- **ok**: `answer: `ok`: object` What the operation answered, as its entry in the protocol section says.
- **error**: `answer: `error`: object` Why the request was refused, and what acts on it as data; the world is as it was.
- **message**: `error: `message`: string` Why, one or more sentences for a person or an agent to read.
- **line**: `error: `line`: integer` Where in the text the request carried the fault is, from 1.
- **column**: `error: `column`: integer` Where on that line, from 1.
- **try**: `error: `try`: [object]` The requests to send next, in order, as they are: the request corrected, what it needs first, or what the caller needs to know; never one that widens authority.
- **similar**: `error: `similar`: [string]` The names nearest an unknown one.
- **fields**: `error: `fields`: [string]` The fields the operation takes.
- **field**: `error: `field`: string` The field at fault.
- **expected**: `error: `expected`: string` The type that field is.
- **operations**: `error: `operations`: [string]` Every operation, for an unknown one.
- **sections**: `error: `sections`: [string]` Every help section, for an unknown one.
- **version**: `error: `version`: string` The scene document's version as it stands, for a compose refused.
- **path**: `error: `path`: string` The scene document a refused compose would have written.
- **written**: `error: `written`: bool` Whether a compose wrote its document: false when refused.
- **text**: `error: `text`: string` The document an `edits` compose made, which `line` counts in.
- **node**: `error: `node`: string` The node an evaluation failed at.
- **port**: `error: `port`: string` The state port whose migration failed.
- **migration**: `error: `migration`: string` The Kernel that failed to migrate that state.
- **frame**: `error: `frame`: integer` The frame a run failed at, counted from its first.
- **stdin**: `orla serve <scene.osn>` Request lines on standard input, answer lines on standard output; `quit` ends the process.
- **socket**: `orla serve <scene.osn> --listen <host:port>` The same lines over each TCP connection to the address; `quit` ends the connection.
- **websocket**: `orla serve <scene.osn> --listen <host:port>` The same lines as WebSocket text messages to the same address, one request a message.
- **mcp**: `orla serve <scene.osn> --mcp` MCP over standard input and output: each operation a tool whose input schema is its request's, a tool call's arguments its fields, the answer's `ok` or `error` the tool's result.

### replay: the lines of a replay test

- **scene**: `scene <path>` the scene to test, relative to the replay file; required
- **library**: `library <path>` Kernels beyond the scene's `use`
- **frame**: `frame <seconds>` evaluates one frame at this session time
- **settle**: `settle` waits for every `async` run to land
- **pointer**: `pointer [<x> <y>] [down | up]` moves the pointer, in the output's pixels from its top left (what `pointer()` reads), and presses or releases it
- **button**: `button <name> down | up` left, right, middle, or a pad's south, east, west, north, l1, r1, l3, r3, start, select
- **stick**: `stick left | right | triggers <x> <y>` a pad's axis pair
- **wheel**: `wheel <x> <y>` the scroll for the next frame only
- **window**: `window <w> <h> [<scale>]` the window in pixels, for `window()`, and its scale factor; one when not given
- **key**: `key <code> down | up` a key by the browser's code: letters and digits by uppercase ASCII, space 32, enter 13, arrows 37 to 40
- **reset**: `reset [<node.port>]` forgets every node's committed state, or one port's
- **param**: `param <name> <literal>` sets a graph parameter
- **expect**: `expect <x> <y> = <r> <g> <b> [<tolerance>]` the picture's color at x y from its top left, each channel 0 to 1 as the Kernel wrote it; tolerance 0.01 when not given
- **expect**: `expect <node.port[i].field> = <value>... [<tolerance>]` a field of an element of a node's output: ints and bools exactly, floats and vectors within the tolerance
- **expect**: `expect <node.port[i].field> < | <= | > | >= <number>...` every component of it past a bound
- **expect**: `expect count <node.port> = <n>` how many elements a node's output holds
- **expect**: `expect placed <node> main | ordered` where a node runs: the session's main backend, or ordered work off it; a session with one backend reads both as main

## Examples

Whole scenes the tests load and run; each says what it shows.

### tally.orlapkg/package.osn

```osn
//! A package of one Kernel, as the graph example reaches it by its path.
//! shows: package
package tally@1 { language = 1; ir = 1; }
use "kernels/";
```

### scene.osn

```osn
//! The smallest whole scene: one Kernel, written in place, draws a
//! picture over a domain, and the scene shows it.
//! shows: inline, requires, output, compute, Image, coord, vec4, vec2
node picture: kernel {
    requires { field.write; }
    domain = vec2(64, 48);
    output result: Image;
    compute {
        let uv = (coord() + 0.5) / vec2(64.0, 48.0);
        return vec4(uv.x, uv.y, 0.5, 1.0);
    }
}

output result <- picture.result;
```

### loops.osn

```osn
//! Repetition with a bound: a `loop` carries a value until a condition
//! holds or it breaks, an `[each n { ... }]` builds an array, `index()`
//! counts the turn, and `budget` caps the work an element may do.
//! shows: loop, break, each, index, budget
node loops: kernel {
    requires { field.write; }
    budget 100000;
    domain = 4;
    output result: Rows<Float>;
    compute {
        let steps = loop 16 with total = 0.0 until total > 3.0 { total + float(index()) };
        let found = loop 16 with x = 0.0 { if x > 2.0 { break x; } x + 1.0 };
        let squares = [each 4 { float(index() * index()) }];
        return steps + found + squares[element()];
    }
}

output result <- loops.result;
```

### values.osn

```osn
//! Values a Kernel computes with: names bound by `let`, choices by `if`
//! and `match`, truth by `&&`, `||` and `!`, arrays, a struct of its own,
//! and a function over any type.
//! shows: let, if, and, match, array, struct, type, fn, element, float, Rows, Float, Int, reserved
struct Step { at: Float; size: Int; }
type Values = Rows<Float>;

fn twice<T>(v: T) -> T { return v + v; }

node values: kernel {
    requires { field.write; }
    domain = 8;
    output result: Values;
    compute {
        let i = element();
        let step = Step { at: float(i), size: 1 };
        let wider = Step { ..step, size: 3 };
        let { at, size: width } = wider;
        let even = i % 2 == 0;
        let small = i < 4 && !(i == 0 || i == 1);
        let tier = match i { 0 => 0.0, 1 | 2 => 1.0, 3..6 => 2.0, 6..=7 => 3.0, _ => 4.0 };
        let table = [1.0, 2.0, 4.0];
        let zeros = [0.0; 4];
        let pick = if even { table[1] } else { zeros[0] };
        if small { return twice(at) + tier; }
        return pick + float(width);
    }
}

output result <- values.result;
```

### context.osn

```osn
//! What a Kernel reads of the world around it, each with the capability
//! that grants it: the clock and the sound rate (`time.read`), and the
//! window, the pointer, buttons, keys, pad and wheel (`input.read`).
//! shows: time, rate, window, scale, pointer, pointer_down, button, stick, wheel, key, domain, idomain, field.read, field.write, time.read, input.read
node world: kernel {
    requires { field.write; time.read; input.read; }
    domain = vec2(8, 8);
    output result: Grid<Float>;
    compute {
        let clock = time() + rate() * 0.0;
        let screen = window().x * scale() + domain().x + float(idomain().y);
        let hand = pointer().x + if pointer_down() || button(1) { 1.0 } else { 0.0 };
        let pad = stick(0).x + wheel().y + if key(32) { 1.0 } else { 0.0 };
        return clock + screen * 0.0 + hand + pad;
    }
}

node read: kernel {
    requires { field.read; field.write; }
    input world: Grid<Float> <- world.result;
    domain = 4;
    output result: Rows<Float>;
    compute { return at(world, vec2(0.0, 0.0)); }
}

output result <- read.result;
```

### fields.osn

```osn
//! Fields flowing through a scene: a Kernel declared once by its id and
//! run by a node, a param the caller sets, a Field filled with a value,
//! an input read from text, state carried from the last frame with `<~`,
//! and the builtins that read a Field.
//! shows: kernel, node, param, fill, input, previous, at, fetch, at_linear, sample, count, size, isize, element, coord
kernel fields.decay@1 {
    requires { field.read; field.write; }
    input last: Rows<Float>;
    param rate: Float = 0.5;
    output result: Rows<Float>;
    compute { return sample(last) * rate + 1.0; }
}

param rate: Float = 0.9;
input word: Rows<Int> = text "ORLA";

node start: Rows<Float> { domain = 4; value = 0.0; }
node decay: fields.decay@1 {
    last <~ decay.result from start.result;
    rate = rate;
}

node plane: kernel {
    requires { field.write; }
    domain = vec2(4, 2);
    output result: Grid<Float>;
    compute { return coord().x + coord().y * 10.0; }
}

node read: kernel {
    requires { field.read; field.write; }
    input level: Rows<Float> <- decay.result;
    input letters: Rows<Int> <- word;
    input grid: Grid<Float> <- plane.result;
    domain = 4;
    output result: Rows<Float>;
    compute {
        let i = element();
        let cell = at(grid, vec2(1.0, 1.0)) + fetch(grid, ivec2(2, 0)) + at_linear(level, 1.5);
        let extent = size(grid).x + float(isize(grid).y + count(letters));
        return sample(level) + at(level, i) + cell + extent + float(at(letters, 0));
    }
}

output result <- read.result;
```

### graph.osn

```osn
//! A scene composed of what other documents hold: Kernels a file and a
//! package define (`use`), a graph placed as a node, one placed several
//! times with its output carried from each to the next, a node that runs
//! off the frame, and domains sized by what came before.
//! shows: use, graph, times, async, vec2, vec3, count, size, value, window, scale, max, min, floor, ceil
use "step.orla";
use "tally.orlapkg";

param levels: Int = 3;

node seed: Rows<Float> { domain = 4; value = 1.0; }
node doubled: tally.twice@1 { values <- seed.result; }
node placed: graph "placed.osn" { value <- doubled.result; by = 2.0; }
node counted: graph "placed.osn" {
    times = levels;
    carry value <- result;
    value <- seed.result;
}
node slow: steps.inc@1 { value <- counted.result; async = true; }

node screen: Grid<Float> { domain = max(floor(window() * scale() / 8.0), vec2(1, 1)); }
node cube: Volume<Float> { domain = vec3(2, 2, 2); }

node sized: kernel {
    requires { field.read; field.write; }
    input values: Rows<Float> <- seed.result;
    domain = max(count(values), min(4, floor(value(values) * 2.0)));
    output result: Rows<Float>;
    compute { return sample(values); }
}

node plane: kernel {
    requires { field.read; field.write; }
    input g: Grid<Float> <- screen.result;
    domain = ceil(size(g) / 2.0);
    output result: Grid<Float>;
    compute { return 1.0; }
}

node shown: kernel {
    requires { field.read; field.write; }
    input a: Rows<Float> <- placed.result;
    input b: Rows<Float> <- slow.result;
    input c: Grid<Float> <- plane.result;
    input s: Rows<Float> <- sized.result;
    input d: Volume<Float> <- cube.result;
    input e: Grid<Float> <- screen.result;
    domain = 4;
    output result: Rows<Float>;
    compute { return sample(a) + at(b, element()) + float(count(c) + count(d) + count(e) + count(s)); }
}

output result <- shown.result;
```

### math.osn

```osn
//! The math a Kernel calls, on Floats, Ints and vectors: each call here
//! runs on every element.
//! shows: floor, ceil, round, sign, log, tan, asin, acos, atan, fract, abs, sqrt, sin, cos, min, max, clamp, mix, add_wrap, sub_wrap, mul_wrap, float, int, pow, exp, step, smoothstep, atan2, trunc, exp2, log2, sinh, cosh, tanh, fma, popcount, clz, ctz, rotl, rotr, ult, udiv, urem, mulhi, umulhi, bits, from_bits, saturate, dot, cross, length, normalize, reflect, transpose, determinant, inverse, distance, radians, degrees, inversesqrt, mod, isnan, isinf, refract
node math: kernel {
    requires { field.write; }
    domain = 4;
    output result: Rows<Float>;
    compute {
        let x = float(element()) * 0.25 + 0.1;
        let n = element() + 1;
        let rounding = floor(x) + ceil(x) + round(x) + sign(x) + fract(x) + trunc(x) + abs(-x);
        let curves = log(x) + tan(x) + asin(x) + acos(x) + atan(x) + atan2(x, 1.0) + sqrt(x) + sin(x) + cos(x)
            + pow(x, 2.0) + exp(x) + exp2(x) + log2(x) + sinh(x) + cosh(x) + tanh(x) + inversesqrt(x)
            + radians(x) + degrees(x) + mod(x, 0.3) + fma(x, 2.0, 1.0) + saturate(x * 2.0);
        let shaping = min(x, 0.5) + max(x, 0.5) + clamp(x, 0.2, 0.4) + mix(0.0, 1.0, x) + step(0.5, x) + smoothstep(0.0, 1.0, x);
        let wrapped = add_wrap(n, 2) + sub_wrap(n, 1) + mul_wrap(n, 3) + popcount(n) + clz(n) + ctz(n)
            + rotl(n, 1) + rotr(n, 1) + udiv(n, 2) + urem(n, 2) + mulhi(n, n) + umulhi(n, n) + int(ult(n, 3));
        let raw = from_bits(bits(x));
        let v = vec3(x, 1.0, 0.5);
        let geometry = dot(v, v) + length(cross(v, vec3(0.0, 0.0, 1.0))) + normalize(v).x + reflect(v, vec3(0.0, 1.0, 0.0)).y
            + refract(v, vec3(0.0, 1.0, 0.0), 0.5).x + distance(v, vec3(0.0, 0.0, 0.0));
        let m = mat2(vec2(2.0, 0.0), vec2(0.0, 2.0));
        let matrices = transpose(m)[0].x + determinant(m) + inverse(m)[1].y;
        let odd = if isnan(x) || isinf(x) { 1.0 } else { 0.0 };
        return rounding + curves + shaping + float(wrapped) + raw + geometry + matrices + odd;
    }
}

output result <- math.result;
```

### types.osn

```osn
//! Every type a value, a port or a param has: scalars, vectors and
//! matrices of 32 and 64 bits, and the Fields that hold them, a picture's
//! element kept narrow with `as`.
//! shows: Float, Int, Bool, Double, Long, Vec2, Vec3, Vec4, IVec2, IVec3, LVec2, LVec3, DVec2, DVec3, DVec4, Mat2, Mat3, Mat4, DMat2, DMat3, DMat4, Rows, Grid, Volume, as, ivec2, ivec3, lvec2, lvec3, dvec2, dvec3, dvec4, mat2, mat3, mat4, dmat2, dmat3, dmat4, double, long, vec3, icoord
struct Every {
    f: Float; n: Int; yes: Bool; d: Double; l: Long;
    v2: Vec2; v3: Vec3; v4: Vec4; cell: IVec2; cube: IVec3; wide: LVec2; wider: LVec3;
    dv2: DVec2; dv3: DVec3; dv4: DVec4;
    m2: Mat2; m3: Mat3; m4: Mat4; dm2: DMat2; dm3: DMat3; dm4: DMat4;
}

node narrow: kernel {
    requires { field.write; }
    domain = vec2(4, 4);
    output result: Grid<Vec4 as unorm8>;
    compute {
        let d = double(0.5);
        let l = long(3);
        let cell = icoord() + ivec2(0, 0);
        let e = Every {
            f: 0.5, n: 3, yes: true, d: d, l: l,
            v2: vec2(0.5, 0.5), v3: vec3(0.5, 0.5, 0.5), v4: vec4(0.5, 0.5, 0.5, 1.0),
            cell: cell, cube: ivec3(cell.x, cell.y, 3), wide: lvec2(l, l), wider: lvec3(l, l, l),
            dv2: dvec2(d, d), dv3: dvec3(d, d, d), dv4: dvec4(d, d, d, d),
            m2: mat2(1.0), m3: mat3(1.0), m4: mat4(1.0), dm2: dmat2(d), dm3: dmat3(d), dm4: dmat4(d),
        };
        let v4 = e.m4 * vec4(e.m2 * e.v2, (e.m3 * e.v3).z, 1.0);
        let exact = float((e.dm2 * e.dv2).x + (e.dm3 * e.dv3).y + (e.dm4 * e.dv4).z);
        let count = float(e.wide.x + e.wider.z) + float(e.cube.z + e.n);
        return if e.yes { vec4(v4.x, exact * e.f, count / 16.0, 1.0) } else { e.v4 };
    }
}

node deep: kernel {
    requires { field.write; }
    domain = vec3(2, 2, 2);
    output result: Volume<Float>;
    compute { return coord().z; }
}

node row: kernel {
    requires { field.read; field.write; }
    input layers: Volume<Float> <- deep.result;
    domain = 4;
    output result: Rows<Float>;
    compute { return at(layers, vec3(0.0, 0.0, 1.0)); }
}

output shown <- narrow.result;
output layers <- row.result;
```

### modes.osn

```osn
//! How a Kernel runs over its domain: a map (`compute`) gives each element
//! alone; `merge` folds a map's elements into one; a `scan` walks them in
//! order, carrying a value and owning `memory` it may `put` into; a
//! `prefix` folds each element with every one before it; a `scatter`
//! combines values into a target at addresses, one per element or several
//! by `emit`; a fold that must give one answer in any order sums
//! integers.
//! shows: compute, merge, scan, memory, put, prefix, scatter, emit
node ones: Rows<Float> { domain = 8; value = 1.0; }
node zeros: Rows<Float> { domain = 4; value = 0.0; }
node counts: Rows<Int> { domain = 8; value = 1; }
node empty: Rows<Int> { domain = 4; value = 0; }

node total: kernel {
    requires { field.read; field.write; }
    input values: Rows<Float> <- ones.result;
    output result: Rows<Float>;
    compute { return sample(values); }
    merge(a, b) { return a + b; }
}

kernel modes.walk@1 {
    requires { field.read; field.write; }
    input steps: Rows<Float>;
    memory trail: Rows<Float>;
    output result: Rows<Float>;
    scan with y = 0.0 {
        let next = y + sample(steps);
        put(trail, element() % 4, next);
        return next;
    }
}

node walked: modes.walk@1 {
    steps <- ones.result;
    trail <~ walked.trail from zeros.result;
}

node running: kernel {
    requires { field.read; field.write; }
    input values: Rows<Int> <- counts.result;
    output result: Rows<Int>;
    prefix with sum { return sample(values); }
}

node binned: kernel {
    requires { field.read; field.write; }
    input bins: Rows<Int> <- empty.result;
    domain = 8;
    output result: Rows<Int>;
    struct Add { at: Int; value: Int; }
    scatter into bins with sum {
        emit(Add { at: 0, value: 2 });
        return Add { at: element() % 4, value: 1 };
    }
}

node summed: kernel {
    requires { field.read; field.write; }
    input a: Rows<Float> <- total.result;
    input b: Rows<Float> <- walked.result;
    input c: Rows<Int> <- running.result;
    input d: Rows<Int> <- binned.result;
    domain = 4;
    output result: Rows<Float>;
    compute { return at(a, 0) + at(b, element()) + float(at(c, element()) + at(d, element())); }
}

output result <- summed.result;
```
<!-- end -->
