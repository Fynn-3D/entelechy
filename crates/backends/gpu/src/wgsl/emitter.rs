//! The translation of a function's instructions into WGSL statements,
//! one value a line, and of the functions a Kernel declares.

use super::*;

pub(crate) fn emit_function(
    out: &mut String,
    module: &Module,
    function: &Function,
    name: &str,
    entry: bool,
    packed: &[usize],
) {
    let args = if entry && module.manifest.mode == Mode::Scan {
        format!(
            "coord: vec3<f32>, element: u32, carried: {}",
            wgsl_type(module.manifest.outputs[0].element())
        )
    } else if entry {
        "coord: vec3<f32>, element: u32".to_owned()
    } else {
        function
            .args
            .iter()
            .enumerate()
            .map(|(index, ty)| format!("a{index}: {}", wgsl_type(*ty)))
            .collect::<Vec<_>>()
            .join(", ")
    };
    // A map's entry with several outputs stores each where it goes: WGSL
    // returns one value, and there is nothing to gain by packing them.
    let stores = entry && written_outputs(module) > 1;
    if stores {
        let _ = writeln!(out, "fn {name}({args}) {{");
    } else {
        let ret = function.result_type();
        let _ = writeln!(out, "fn {name}({args}) -> {} {{", wgsl_type(ret));
    }
    let mut emitter = Emitter {
        out,
        module,
        function,
        packed,
        loops: Vec::new(),
    };
    emitter.instructions(&function.body, 1);
    if stores {
        for (k, result) in function.results.iter().enumerate() {
            let _ = writeln!(out, "    store_out{k}(element, v{});", result.0);
        }
    } else {
        let _ = writeln!(out, "    return v{};", function.results[0].0);
    }
    let _ = writeln!(out, "}}");
}

/// Whether a block is short plain arithmetic, which costs less computed
/// than branched around: no loop, branch, call or write.
pub(crate) fn speculable(block: &Block) -> bool {
    // An arm that can fail is not computed where it is not taken: the
    // failure would be recorded for work the reference never does.
    block.body.len() <= 16
        && block.body.iter().all(|instruction| {
            !matches!(
                instruction.op,
                Op::Loop { .. }
                    | Op::If { .. }
                    | Op::Match { .. }
                    | Op::Put(..)
                    | Op::Emit(_)
                    | Op::Call(..)
                    | Op::Spend(_)
                    | Op::Look
            ) && !fails(&instruction.op, instruction.ty)
        })
}

/// The suffix of the checked helper for an integer type: `i32`,
/// `i32x2`, `i64x3` and the rest.
pub(crate) fn checked_name(ty: Type) -> String {
    let base = if ty.is_64bit() { "i64" } else { "i32" };
    match ty.lanes() {
        Some((n, _)) => format!("{base}x{n}"),
        None => base.to_owned(),
    }
}

/// Whether WGSL's `select` chooses between two values of a type: a
/// scalar or a vector, and never a 64-bit one.
pub(crate) fn selectable(ty: Type) -> bool {
    matches!(ty, Type::Scalar(_) | Type::Vector(..)) && !ty.is_64bit()
}

pub(crate) struct Emitter<'a> {
    out: &'a mut String,
    module: &'a Module,
    function: &'a Function,
    /// The inputs read from the pack.
    packed: &'a [usize],
    /// Ids of enclosing loop instructions, innermost last.
    loops: Vec<ValueId>,
}

impl Emitter<'_> {
    fn line(&mut self, depth: usize, text: &str) {
        let _ = writeln!(self.out, "{}{text}", "    ".repeat(depth));
    }

    fn instructions(&mut self, instructions: &[Instruction], depth: usize) {
        for instruction in instructions {
            self.instruction(instruction, depth);
        }
    }

    fn block(&mut self, block: &Block, depth: usize) -> String {
        self.instructions(&block.body, depth);
        format!("v{}", block.result.0)
    }

    fn instruction(&mut self, Instruction { id, ty, op }: &Instruction, depth: usize) {
        let v = |id: &ValueId| format!("v{}", id.0);
        let ty_name = wgsl_type(*ty);
        let expression = match op {
            Op::Const(value) => literal(value),
            Op::Param(index) => {
                // The parameter's words from its slots, as the type builds
                // them: the uniform holds words, as the interpreter does.
                let (starts, _) = param_slots(self.module);
                let start = starts[*index as usize];
                let ty = self.module.manifest.params[*index as usize].ty;
                let word = |kind: Scalar, k: usize| {
                    let word = format!("params[{}u][{}u]", start + k / 4, k % 4);
                    match kind {
                        Scalar::Float => format!("bitcast<f32>({word})"),
                        Scalar::Int => format!("bitcast<i32>({word})"),
                        Scalar::Bool => format!("({word} != 0u)"),
                        Scalar::Long => {
                            let high = k + 1;
                            let high = format!("params[{}u][{}u]", start + high / 4, high % 4);
                            format!("long_of({word}, {high})")
                        }
                        other => unreachable!("the GPU refuses a module holding a {other:?}"),
                    }
                };
                compose(self.module, ty, 0, &word).0
            }
            Op::Arg(index) => format!("a{index}"),
            // naga has no negation of a matrix; a scale by -1 flips the
            // same bits.
            Op::Unary(UnaryOp::Neg, a) if self.type_of(*a).order().is_some() => {
                format!("({} * -1.0)", v(a))
            }
            // An integer negated records the least one's overflow.
            Op::Unary(UnaryOp::Neg, a) if ty.is_int_kind() => {
                format!("neg_{}({})", checked_name(*ty), v(a))
            }
            Op::Unary(UnaryOp::Neg, a) => format!("(-{})", v(a)),
            Op::Unary(UnaryOp::Not, a) => format!("(!{})", v(a)),
            Op::Binary(op, a, b) => {
                let symbol = match op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "/",
                    BinaryOp::Rem => "%",
                    BinaryOp::Less => "<",
                    BinaryOp::LessEqual => "<=",
                    BinaryOp::Greater => ">",
                    BinaryOp::GreaterEqual => ">=",
                    BinaryOp::Equal => "==",
                    BinaryOp::NotEqual => "!=",
                    BinaryOp::And => "&&",
                    BinaryOp::Or => "||",
                    BinaryOp::BitAnd => "&",
                    BinaryOp::BitOr => "|",
                    BinaryOp::BitXor => "^",
                    BinaryOp::Shl => "<<",
                    BinaryOp::Shr => ">>",
                    BinaryOp::Ushr => ">>",
                };
                // A whole-vector comparison folds the componentwise one; a
                // shift takes its count unsigned, masked to the width.
                let vector = self.type_of(*a).lanes().is_some();
                let matrix = self.type_of(*a).order().map(usize::from);
                // An integer vector beside an Int: WGSL's bit operators
                // and shifts take two operands of one type, so the Int
                // widens; the shift count is unsigned, masked to the width.
                let int_vector = ty.int_components().is_some();
                let widen = |id: &ValueId| {
                    if self.type_of(*id) == *ty {
                        v(id)
                    } else {
                        format!("{}({})", wgsl_type(*ty), v(id))
                    }
                };
                let count = |id: &ValueId| {
                    let n = ty.int_components().unwrap_or(1);
                    // A whole vector converts; a scalar converts first,
                    // then splats, as WGSL does not do both in one.
                    let unsigned = if self.type_of(*id) == Type::Int {
                        format!("u32({})", v(id))
                    } else {
                        v(id)
                    };
                    format!("(vec{n}<u32>({unsigned}) & vec{n}<u32>(31u))")
                };
                // A long: the bit operators with a scalar widened beside a
                // vector, a shift's count modulo 64, and the division and
                // remainder the preamble writes.
                let long = self.type_of(*a).scalar() == Some(Type::Long);
                let lanes = ty.lanes().map(|(n, _)| n);
                let count64 = |id: &ValueId| match lanes {
                    // A scalar count converts first, then splats.
                    Some(n) if self.type_of(*id) == Type::Long => {
                        format!("(vec{n}<u32>(u32({})) & vec{n}<u32>(63u))", v(id))
                    }
                    Some(n) => format!("(vec{n}<u32>({}) & vec{n}<u32>(63u))", v(id)),
                    None => format!("(u32({}) & 63u)", v(id)),
                };
                let suffix = match lanes {
                    Some(n) => format!("x{n}"),
                    None => String::new(),
                };
                let unsigned = match lanes {
                    Some(n) => format!("vec{n}<u64>"),
                    None => "u64".to_owned(),
                };
                // Integer arithmetic records what fails, both operands at
                // the result's width.
                let checked = match op {
                    BinaryOp::Add => Some("add"),
                    BinaryOp::Sub => Some("sub"),
                    BinaryOp::Mul => Some("mul"),
                    BinaryOp::Div => Some("div"),
                    BinaryOp::Rem => Some("rem"),
                    _ => None,
                };
                if let Some(name) = checked
                    && ty.is_int_kind()
                {
                    let helper = checked_name(*ty);
                    let text = format!("{name}_{helper}({}, {})", widen(a), widen(b));
                    self.line(depth, &format!("let v{}: {ty_name} = {text};", id.0));
                    return;
                }
                match op {
                    BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor if long => {
                        format!("({} {symbol} {})", widen(a), widen(b))
                    }
                    BinaryOp::Shl | BinaryOp::Shr if long => {
                        format!("({} {symbol} {})", widen(a), count64(b))
                    }
                    BinaryOp::Ushr if long => format!(
                        "bitcast<{ty_name}>(bitcast<{unsigned}>({}) >> {})",
                        widen(a),
                        count64(b)
                    ),
                    BinaryOp::Div if long => {
                        format!("div_i64{suffix}({}, {})", widen(a), widen(b))
                    }
                    BinaryOp::Rem if long => {
                        format!("rem_i64{suffix}({}, {})", widen(a), widen(b))
                    }
                    BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor if int_vector => {
                        format!("({} {symbol} {})", widen(a), widen(b))
                    }
                    BinaryOp::Shl | BinaryOp::Shr if int_vector => {
                        format!("({} {symbol} {})", widen(a), count(b))
                    }
                    BinaryOp::Ushr if int_vector => {
                        let n = ty.int_components().unwrap_or(1);
                        format!("vec{n}<i32>(vec{n}<u32>({}) >> {})", widen(a), count(b))
                    }
                    BinaryOp::Equal | BinaryOp::NotEqual if matrix.is_some() => {
                        let n = matrix.unwrap_or(0);
                        let same = (0..n)
                            .map(|c| format!("all({}[{c}] == {}[{c}])", v(a), v(b)))
                            .collect::<Vec<_>>()
                            .join(" && ");
                        if *op == BinaryOp::Equal {
                            format!("({same})")
                        } else {
                            format!("(!({same}))")
                        }
                    }
                    BinaryOp::Rem if *ty != Type::Int && !int_vector => {
                        // Both operands at the result's width: a Float
                        // beside a vector broadcasts.
                        let widen = |id: &ValueId| {
                            if self.type_of(*id) == *ty {
                                v(id)
                            } else {
                                format!("{}({})", wgsl_type(*ty), v(id))
                            }
                        };
                        format!(
                            "fmod_{}({}, {})",
                            ty.components().unwrap_or(1),
                            widen(a),
                            widen(b)
                        )
                    }
                    // `isnan(x)` is `x != x`, which a compiler told
                    // there are no NaNs folds to false — and this one is
                    // told that. On the words it is the ones exponent
                    // with something in the mantissa, which nothing can
                    // fold away.
                    BinaryOp::NotEqual if a == b && self.type_of(*a) == Type::Float => {
                        format!("is_nan({})", v(a))
                    }
                    BinaryOp::Equal if a == b && self.type_of(*a) == Type::Float => {
                        format!("!is_nan({})", v(a))
                    }
                    BinaryOp::Equal if vector => format!("all({} == {})", v(a), v(b)),
                    BinaryOp::NotEqual if vector => format!("any({} != {})", v(a), v(b)),
                    BinaryOp::Shl | BinaryOp::Shr => {
                        format!("({} {symbol} (u32({}) & 31u))", v(a), v(b))
                    }
                    BinaryOp::Ushr => {
                        format!("i32(u32({}) >> (u32({}) & 31u))", v(a), v(b))
                    }
                    // The dividend through `whole_*`: a value the compiler
                    // narrowed (the top half of a word) divided wrongly.
                    BinaryOp::Div | BinaryOp::Rem if *ty == Type::Int || int_vector => {
                        let whole = match ty.int_components() {
                            Some(2) => "whole_v2",
                            Some(_) => "whole_v3",
                            None => "whole_i32",
                        };
                        format!("({whole}({}) {symbol} {})", widen(a), v(b))
                    }
                    _ => format!("({} {symbol} {})", v(a), v(b)),
                }
            }
            Op::Construct(parts) => format!(
                "{ty_name}({})",
                parts.iter().map(v).collect::<Vec<_>>().join(", ")
            ),
            Op::Extract(source, component) => {
                if let Type::Struct(index) = self.type_of(*source) {
                    format!(
                        "{}{}",
                        v(source),
                        field_path(self.module, index, *component as usize)
                    )
                } else if self.type_of(*source).order().is_some() {
                    format!("{}[{component}]", v(source))
                } else {
                    format!(
                        "{}.{}",
                        v(source),
                        ["x", "y", "z", "w"][*component as usize]
                    )
                }
            }
            Op::Subscript(array, index) => {
                let count = match self.type_of(*array) {
                    Type::Struct(struct_index) => {
                        self.module.structs[struct_index as usize].fields.len()
                    }
                    other => other
                        .order()
                        .or(other.lanes().map(|(n, _)| n))
                        .map_or(1, usize::from),
                };
                format!("{}[clamp({}, 0, {})]", v(array), v(index), count - 1)
            }
            Op::Call(index, args) => format!(
                "{}({})",
                function_name(*index as usize),
                args.iter().map(v).collect::<Vec<_>>().join(", ")
            ),
            Op::Intrinsic(intrinsic, args) => self.intrinsic(*intrinsic, args, *ty),
            // The iterations whose body ran are the loop's own counter,
            // kept after it: one addition.
            Op::Spend(counted) => {
                self.line(depth, &format!("work = work + u32(it{});", counted.0));
                self.line(depth, &format!("let v{}: i32 = it{};", id.0, counted.0));
                return;
            }
            Op::Look => {
                self.line(depth, &format!("let v{}: bool = look();", id.0));
                return;
            }
            Op::Emit(pair) => {
                // Combined into the target now; the value is the pair.
                self.line(depth, &format!("{}({});", emit_name(*ty), v(pair)));
                self.line(depth, &format!("let v{}: {ty_name} = {};", id.0, v(pair)));
                return;
            }
            Op::Put(input, index, value) => {
                // The store, when the address is inside; the value is the
                // instruction's own.
                self.line(
                    depth,
                    &format!(
                        "if (inside_in{input}({})) {{ store_in{input}(index_in{input}({}), {}); }}",
                        v(index),
                        v(index),
                        v(value)
                    ),
                );
                self.line(depth, &format!("let v{}: {ty_name} = {};", id.0, v(value)));
                return;
            }
            // Two short arms of plain arithmetic are both computed and
            // one is chosen: a branch costs a device more than the few
            // operations it would skip, and nothing a device computes
            // fails. A long is never chosen by `select`, which Metal
            // cannot pick for one.
            Op::If {
                condition,
                then,
                otherwise,
            } if speculable(then) && speculable(otherwise) && selectable(*ty) => {
                let chosen = self.block(then, depth);
                let other = self.block(otherwise, depth);
                format!("select({other}, {chosen}, {})", v(condition))
            }
            Op::If {
                condition,
                then,
                otherwise,
            } => {
                self.line(depth, &format!("var v{}: {ty_name};", id.0));
                self.line(depth, &format!("if ({}) {{", v(condition)));
                let result = self.block(then, depth + 1);
                self.line(depth + 1, &format!("v{} = {result};", id.0));
                self.line(depth, "} else {");
                let result = self.block(otherwise, depth + 1);
                self.line(depth + 1, &format!("v{} = {result};", id.0));
                self.line(depth, "}");
                return;
            }
            Op::Match {
                scrutinee,
                arms,
                otherwise,
            } => {
                // WGSL's switch: every selector a literal, no fallthrough.
                self.line(depth, &format!("var v{}: {ty_name};", id.0));
                self.line(depth, &format!("switch ({}) {{", v(scrutinee)));
                for (values, block) in arms {
                    let selectors = values
                        .iter()
                        .map(|value| format!("{value}i"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    self.line(depth + 1, &format!("case {selectors}: {{"));
                    let result = self.block(block, depth + 2);
                    self.line(depth + 2, &format!("v{} = {result};", id.0));
                    self.line(depth + 1, "}");
                }
                self.line(depth + 1, "default: {");
                let result = self.block(otherwise, depth + 2);
                self.line(depth + 2, &format!("v{} = {result};", id.0));
                self.line(depth + 1, "}");
                self.line(depth, "}");
                return;
            }
            Op::Loop {
                count,
                bound,
                init,
                until,
                body,
            } => {
                self.line(depth, &format!("var v{}: {ty_name} = {};", id.0, v(init)));
                let limit = match bound {
                    Some(bound) => format!("clamp({}, 0i, {count}i)", v(bound)),
                    None => format!("{count}i"),
                };
                // `it` and the value's number: a plain `i` would spell
                // `i32` at value 32, which WGSL reads as the type.
                let counts = self.module.accounted();
                // A loop of a module that counts keeps `it` after it: the
                // iterations whose body ran, all of them unless `until`
                // stopped one, which the spend after it adds.
                if counts {
                    self.line(depth, &format!("var it{}: i32 = 0;", id.0));
                    self.line(
                        depth,
                        &format!("for (; it{0} < {limit}; it{0} = it{0} + 1) {{", id.0),
                    );
                } else {
                    self.line(
                        depth,
                        &format!(
                            "for (var it{0}: i32 = 0; it{0} < {limit}; it{0} = it{0} + 1) {{",
                            id.0
                        ),
                    );
                }
                self.loops.push(*id);
                if let Some(until) = until {
                    let done = self.block(until, depth + 1);
                    self.line(depth + 1, &format!("if ({done}) {{ break; }}"));
                }
                let next = self.block(body, depth + 1);
                self.line(depth + 1, &format!("v{} = {next};", id.0));
                self.loops.pop();
                self.line(depth, "}");
                return;
            }
            Op::Carried => match self.loops.last() {
                Some(id) => format!("v{}", id.0),
                None => "carried".to_owned(),
            },
            Op::Index => match self.loops.last() {
                Some(id) => format!("it{}", id.0),
                None => "i32(element)".to_owned(),
            },
            Op::Coord => match self.output_shape() {
                Shape::Volume => "coord".to_owned(),
                _ => "coord.xy".to_owned(),
            },
            Op::Domain => match self.output_shape() {
                Shape::Rows => "i32(ctx[0].x)".to_owned(),
                Shape::Grid => "ctx[0].xy".to_owned(),
                Shape::Volume => "vec3<f32>(ctx[0].xy, ctx[1].z)".to_owned(),
            },
            Op::Time => "ctx[0].z".to_owned(),
            Op::Rate => "ctx[1].w".to_owned(),
            Op::Pointer => "ctx[1].xy".to_owned(),
            Op::PointerDown => "button_down(0)".to_owned(),
            Op::Button(code) => format!("button_down({})", v(code)),
            Op::Wheel => "ctx[4].xy".to_owned(),
            Op::Stick(which) => format!("stick_at({})", v(which)),
            Op::Window => "ctx[4].zw".to_owned(),
            Op::Scale => "ctx[5].x".to_owned(),
            Op::Key(code) => format!("key_down({})", v(code)),
            Op::Element => "i32(element)".to_owned(),
            Op::Count(input) => {
                let slot = CONTEXT_HEAD as u32 + input;
                format!("i32(ctx[{slot}].x * ctx[{slot}].y * ctx[{slot}].z)")
            }
            Op::Size(input) => {
                let slot = CONTEXT_HEAD as u32 + input;
                match self.input_shape(*input) {
                    Shape::Rows => format!("i32(ctx[{slot}].x)"),
                    Shape::Grid => format!("ctx[{slot}].xy"),
                    Shape::Volume => format!("ctx[{slot}].xyz"),
                }
            }
            Op::Sample(input) => {
                // Validation made the input's shape the output's, so this
                // element's own address is the right one.
                let own = match self.output_shape() {
                    Shape::Rows => "i32(element)",
                    Shape::Grid => "coord.xy",
                    Shape::Volume => "coord",
                };
                format!("load_in{input}(index_in{input}({own}))")
            }
            Op::At(input, index) => format!("load_in{input}(index_in{input}({}))", v(index)),
            Op::Part { input, at, path } => {
                // The words this part is made of and nothing else:
                // where the path leads, held inside the element.
                let manifest = &self.module.manifest;
                let stride = manifest.inputs[*input as usize].ty.schema.value_words();
                let element = manifest.inputs[*input as usize].element();
                let layout = orla_ir::part_layout(&self.module.structs, element, path)
                    .expect("a path validation walked");
                let mut offset = format!("{}i", layout.base);
                for (index, count, part) in layout.steps {
                    let _ = write!(
                        offset,
                        " + clamp({}, 0i, {}i) * {part}i",
                        v(&index),
                        count as i64 - 1
                    );
                }
                let last = stride.saturating_sub(type_words(self.module, *ty));
                // The element the address names, or this element's
                // own where there is none, as a sample reads it.
                let element = match at {
                    Some(at) => v(at),
                    None => match self.output_shape() {
                        Shape::Rows => "i32(element)".to_owned(),
                        Shape::Grid => "coord.xy".to_owned(),
                        Shape::Volume => "coord".to_owned(),
                    },
                };
                let base = format!("b{}", id.0);
                // In words, or in bytes where the input keeps its scalars
                // narrow: the element's bytes, then the scalars'.
                let encoding = input_encoding(self.module, *input as usize);
                let (element_size, scalar_size) = if encoding == Encoding::Word {
                    (stride, 1)
                } else {
                    (
                        manifest.inputs[*input as usize].ty.schema.element_bytes(),
                        encoding.bytes(),
                    )
                };
                self.line(
                    depth,
                    &format!(
                        "let {base} = index_in{input}({element}) * {element_size}u + u32(clamp({offset}, 0i, {last}i)) * {scalar_size}u;"
                    ),
                );
                let (buffer, offset) =
                    input_source(self.module, self.packed, *input as usize, encoding);
                let (expression, _) = load_expr(
                    self.module,
                    *ty,
                    &buffer,
                    &format!("({offset}{base})"),
                    0,
                    encoding,
                );
                expression
            }
            Op::AtLinear(input, index) => format!("linear_in{input}({})", v(index)),
        };
        let _ = self.function;
        self.line(depth, &format!("let v{}: {ty_name} = {expression};", id.0));
    }

    fn intrinsic(&self, intrinsic: Intrinsic, args: &[ValueId], ty: Type) -> String {
        let v = |id: &ValueId| format!("v{}", id.0);
        let list = args.iter().map(v).collect::<Vec<_>>().join(", ");
        let splat = |id: &ValueId, target: Type| {
            if target.lanes().is_some()
                && matches!(self.type_of(*id), Type::Float | Type::Int | Type::Long)
            {
                format!("{}({})", wgsl_type(target), v(id))
            } else {
                v(id)
            }
        };
        let from = self.type_of(args[0]).scalar();
        let lanes = ty.lanes().map(|(n, _)| n);
        let suffix = |base: &str| match lanes {
            Some(n) => format!("{base}{n}"),
            None => base.to_owned(),
        };
        match intrinsic {
            Intrinsic::ToFloat => format!("{}({list})", wgsl_type(ty)),
            // A long keeps its low 32 bits, as the reference's does.
            Intrinsic::ToInt if from == Some(Type::Long) => format!("{}({list})", wgsl_type(ty)),
            // Saturating from a float, as `to_int` does; from an integer
            // or a Bool, WGSL's own.
            Intrinsic::ToLong if from == Some(Type::Float) => {
                format!("{}({list})", suffix("to_long"))
            }
            Intrinsic::ToLong => format!("{}({list})", wgsl_type(ty)),
            Intrinsic::Sign if from == Some(Type::Long) => match lanes {
                Some(n) => format!("sign_i64x{n}({list})"),
                None => format!("sign_i64({list})"),
            },
            // Saturating, as the reference converts; see `to_int` in
            // the preamble. A Bool is a 0 or a 1 and needs none of it.
            Intrinsic::ToInt => match (self.type_of(args[0]), ty) {
                (Type::Bool, _) => format!("{}({list})", wgsl_type(ty)),
                (_, Type::IVec2) => format!("to_int2({list})"),
                (_, Type::IVec3) => format!("to_int3({list})"),
                _ => format!("to_int({list})"),
            },
            Intrinsic::Pow => format!("pow({}, {})", v(&args[0]), splat(&args[1], ty)),
            // The angle at the origin, which a device's own does not
            // give; see `atan2_ref` in the preamble.
            Intrinsic::Atan2 => format!("atan2_ref({}, {})", v(&args[0]), v(&args[1])),
            // Float `min`, `max` and `clamp` keep the rule every backend
            // keeps (a NaN gives way, of two equal zeros `max` the positive
            // and `min` the negative); see `max_ref` in the preamble.
            Intrinsic::Min | Intrinsic::Max | Intrinsic::Clamp if float_lanes(ty) => {
                let suffix = match ty {
                    Type::Vec2 => "2",
                    Type::Vec3 => "3",
                    Type::Vec4 => "4",
                    _ => "",
                };
                let a: Vec<String> = args.iter().map(|arg| splat(arg, ty)).collect();
                match intrinsic {
                    Intrinsic::Min => format!("min_ref{suffix}({}, {})", a[0], a[1]),
                    Intrinsic::Max => format!("max_ref{suffix}({}, {})", a[0], a[1]),
                    _ => format!(
                        "min_ref{suffix}(max_ref{suffix}({}, {}), {})",
                        a[0], a[1], a[2]
                    ),
                }
            }
            // WGSL wants every operand the result's type; a Float broadcasts.
            Intrinsic::Min | Intrinsic::Max | Intrinsic::Clamp | Intrinsic::Mix => format!(
                "{}({})",
                intrinsic.name(),
                args.iter()
                    .map(|arg| splat(arg, ty))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            // WGSL's integer arithmetic wraps.
            Intrinsic::AddWrap => format!("({} + {})", v(&args[0]), v(&args[1])),
            Intrinsic::SubWrap => format!("({} - {})", v(&args[0]), v(&args[1])),
            Intrinsic::MulWrap => format!("({} * {})", v(&args[0]), v(&args[1])),
            Intrinsic::Popcount => format!("countOneBits({list})"),
            Intrinsic::Clz => format!("countLeadingZeros({list})"),
            Intrinsic::Bits => format!("bitcast<i32>({list})"),
            Intrinsic::FromBits => format!("bitcast<f32>({list})"),
            Intrinsic::Ctz => format!("countTrailingZeros({list})"),
            Intrinsic::Rotl => format!("rotl_i32({list})"),
            Intrinsic::Rotr => format!("rotr_i32({list})"),
            Intrinsic::Ult => format!("(u32({}) < u32({}))", v(&args[0]), v(&args[1])),
            Intrinsic::Udiv => format!(
                "i32(udiv_checked(u32({}), u32({})))",
                v(&args[0]),
                v(&args[1])
            ),
            Intrinsic::Urem => format!(
                "i32(urem_checked(u32({}), u32({})))",
                v(&args[0]),
                v(&args[1])
            ),
            Intrinsic::Mulhi => format!("mulhi_i32({list})"),
            Intrinsic::Umulhi => format!(
                "i32(umulhi_u32(u32({}), u32({})))",
                v(&args[0]),
                v(&args[1])
            ),
            Intrinsic::Step => format!("step({}, {})", splat(&args[0], ty), v(&args[1])),
            // The interpreter's rule, not the device's `smoothstep`, whose
            // result where `low >= high` is the driver's to choose; see
            // `smoothstep_ref` in the preamble.
            Intrinsic::Smoothstep => format!(
                "smoothstep_ref{}({}, {}, {})",
                match ty {
                    Type::Vec2 => "2",
                    Type::Vec3 => "3",
                    Type::Vec4 => "4",
                    _ => "",
                },
                splat(&args[0], ty),
                splat(&args[1], ty),
                v(&args[2])
            ),
            other => format!("{}({list})", other.name()),
        }
    }

    /// The shape the entry maps over: the output's, or the first input's
    /// for a reduction.
    fn output_shape(&self) -> Shape {
        let manifest = &self.module.manifest;
        match manifest.mode {
            Mode::Reduce | Mode::Scan | Mode::Scatter { .. } | Mode::Prefix { .. } => {
                manifest.inputs.first().map_or(Shape::Rows, |p| p.shape())
            }
            Mode::Map => manifest.outputs.first().map_or(Shape::Grid, |p| p.shape()),
        }
    }

    fn input_shape(&self, input: u32) -> Shape {
        self.module.manifest.inputs[input as usize].shape()
    }

    fn type_of(&self, id: ValueId) -> Type {
        self.function.type_of(id).unwrap_or(Type::Float)
    }
}

/// Whether a value of `ty` is floats, one or a vector of them: what the
/// float rule of `min`, `max` and `clamp` applies to.
pub(crate) fn float_lanes(ty: Type) -> bool {
    matches!(ty, Type::Float | Type::Vec2 | Type::Vec3 | Type::Vec4)
}
