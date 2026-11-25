//! Entelechy IR instructions lowered to GLSL ES 3.00 statements.

use super::*;
use prelude::{compose_glsl, input_source_glsl, load_expr_glsl};

const WRITTEN_PLACES: u32 = 64;

pub(crate) fn emit_function(
    out: &mut String,
    module: &Module,
    function: &Function,
    name: &str,
    entry_outputs: Option<&[usize]>,
    packed: &[usize],
) {
    match entry_outputs {
        Some(outputs) if outputs.len() > 1 => {
            let mut params = vec!["vec3 coord".to_owned(), "uint element".to_owned()];
            for (slot, &out_idx) in outputs.iter().enumerate() {
                let ty = glsl_type(module.manifest.outputs[out_idx].element());
                params.push(format!("out {ty} out_r{slot}"));
            }
            let _ = writeln!(out, "void {name}({}) {{", params.join(", "));
        }
        Some(outputs) => {
            let out_idx = outputs[0];
            let ret = glsl_type(module.manifest.outputs[out_idx].element());
            let _ = writeln!(out, "{ret} {name}(vec3 coord, uint element) {{");
        }
        None => {
            let args = function
                .args
                .iter()
                .enumerate()
                .map(|(index, ty)| format!("{} a{index}", glsl_type(*ty)))
                .collect::<Vec<_>>()
                .join(", ");
            let ret = glsl_type(function.result_type());
            let _ = writeln!(out, "{ret} {name}({args}) {{");
        }
    }

    let mut types = std::collections::HashMap::new();
    types_in(&function.body, &mut types);
    let mut emitter = Emitter {
        out,
        module,
        packed,
        loops: Vec::new(),
        types,
    };
    emitter.instructions(&function.body, 1);

    match entry_outputs {
        Some(outputs) if outputs.len() > 1 => {
            for (slot, &out_idx) in outputs.iter().enumerate() {
                let _ = writeln!(out, "    out_r{slot} = v{};", function.results[out_idx].0);
            }
        }
        Some(outputs) => {
            let out_idx = outputs[0];
            let _ = writeln!(out, "    return v{};", function.results[out_idx].0);
        }
        None => {
            let _ = writeln!(out, "    return v{};", function.results[0].0);
        }
    }
    let _ = writeln!(out, "}}");
}

fn types_in(body: &[Instruction], out: &mut std::collections::HashMap<u32, Type>) {
    for instruction in body {
        out.insert(instruction.id.0, instruction.ty);
        for block in instruction.op.blocks() {
            types_in(&block.body, out);
        }
    }
}

fn speculable(block: &Block) -> bool {
    block.body.len() <= 16
        && block.body.iter().all(|instruction| {
            !matches!(
                instruction.op,
                Op::Loop { .. }
                    | Op::Each { .. }
                    | Op::If { .. }
                    | Op::Match { .. }
                    | Op::Put(..)
                    | Op::Emit(_)
                    | Op::Call(..)
                    | Op::Spend(_)
                    | Op::Look
            ) && !fails_op(&instruction.op, instruction.ty)
        })
}

fn fails_op(op: &Op, ty: Type) -> bool {
    match op {
        Op::Binary(
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem,
            ..,
        )
        | Op::Unary(UnaryOp::Neg, _) => ty.is_int_kind(),
        Op::Intrinsic(Intrinsic::Udiv | Intrinsic::Urem, _) => true,
        _ => false,
    }
}

fn checked_name(ty: Type) -> String {
    let base = if ty.is_64bit() { "i64" } else { "i32" };
    match ty.lanes() {
        Some((n, _)) => format!("{base}x{n}"),
        None => base.to_owned(),
    }
}

fn selectable(ty: Type) -> bool {
    matches!(ty, Type::Scalar(_) | Type::Vector(..)) && !ty.is_64bit()
}

fn float_lanes(ty: Type) -> bool {
    matches!(ty, Type::Float | Type::Vec2 | Type::Vec3 | Type::Vec4)
}

struct Emitter<'a> {
    out: &'a mut String,
    module: &'a Module,
    packed: &'a [usize],
    loops: Vec<(ValueId, bool)>,
    types: std::collections::HashMap<u32, Type>,
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
        let ty_name = glsl_type(*ty);
        let expression = match op {
            Op::Const(value) => literal(value),
            Op::Param(index) => {
                let (starts, _) = param_slots(self.module);
                let start = starts[*index as usize];
                let param_ty = self.module.manifest.params[*index as usize].ty;
                let word = |kind: Scalar, k: usize| {
                    let lane = ["x", "y", "z", "w"][k % 4];
                    let word = format!("u_params[{}u].{lane}", start + k / 4);
                    match kind {
                        Scalar::Float => format!("uintBitsToFloat({word})"),
                        Scalar::Int => format!("int({word})"),
                        Scalar::Bool => format!("({word} != 0u)"),
                        Scalar::Long => {
                            let high_k = k + 1;
                            let high_lane = ["x", "y", "z", "w"][high_k % 4];
                            let high = format!("u_params[{}u].{high_lane}", start + high_k / 4);
                            format!("long_of({word}, {high})")
                        }
                        other => unreachable!("the GPU refuses a module holding a {other:?}"),
                    }
                };
                compose_glsl(self.module, param_ty, 0, &word).0
            }
            Op::Arg(index) => format!("a{index}"),
            Op::Unary(UnaryOp::Neg, a) if self.type_of(*a).order().is_some() => {
                format!("({} * -1.0)", v(a))
            }
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
                let vector = self.type_of(*a).lanes().is_some();
                let matrix = self.type_of(*a).order().map(usize::from);
                let int_vector = ty.int_components().is_some();
                let widen = |id: &ValueId| {
                    if self.type_of(*id) == *ty {
                        v(id)
                    } else {
                        format!("{}({})", glsl_type(*ty), v(id))
                    }
                };
                let count_vec = |id: &ValueId| {
                    let n = ty.int_components().unwrap_or(1);
                    let unsigned = if self.type_of(*id) == Type::Int {
                        format!("uint({})", v(id))
                    } else {
                        v(id)
                    };
                    format!("(uvec{n}({unsigned}) & uvec{n}(31u))")
                };
                let long = self.type_of(*a).scalar() == Some(Type::Long);
                let checked = match op {
                    BinaryOp::Add => Some("add"),
                    BinaryOp::Sub => Some("sub"),
                    BinaryOp::Mul => Some("mul"),
                    BinaryOp::Div => Some("div"),
                    BinaryOp::Rem => Some("rem"),
                    _ => None,
                };
                if long {
                    let count = |id: &ValueId| {
                        if self.type_of(*id) == Type::Long {
                            format!("({}).x", v(id))
                        } else {
                            format!("uint({})", v(id))
                        }
                    };
                    let pair = match op {
                        BinaryOp::Less => Some(format!("long_lt({}, {})", v(a), v(b))),
                        BinaryOp::LessEqual => Some(format!("long_le({}, {})", v(a), v(b))),
                        BinaryOp::Greater => Some(format!("long_gt({}, {})", v(a), v(b))),
                        BinaryOp::GreaterEqual => Some(format!("long_ge({}, {})", v(a), v(b))),
                        BinaryOp::Equal => Some(format!("long_eq({}, {})", v(a), v(b))),
                        BinaryOp::NotEqual => Some(format!("(!long_eq({}, {}))", v(a), v(b))),
                        BinaryOp::Shl => Some(format!("shl_i64({}, {})", v(a), count(b))),
                        BinaryOp::Shr => Some(format!("sar_i64({}, {})", v(a), count(b))),
                        BinaryOp::Ushr => Some(format!("shr_ulong({}, {})", v(a), count(b))),
                        BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor => {
                            Some(format!("({} {symbol} {})", widen(a), widen(b)))
                        }
                        _ => None,
                    };
                    if let Some(text) = pair {
                        self.line(depth, &format!("{ty_name} v{} = {text};", id.0));
                        return;
                    }
                }
                if let Some(name) = checked
                    && ty.is_int_kind()
                {
                    let helper = checked_name(*ty);
                    let text = format!("{name}_{helper}({}, {})", widen(a), widen(b));
                    self.line(depth, &format!("{ty_name} v{} = {text};", id.0));
                    return;
                }
                match op {
                    BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor if int_vector => {
                        format!("({} {symbol} {})", widen(a), widen(b))
                    }
                    BinaryOp::Shl if int_vector => {
                        let n = ty.int_components().unwrap_or(1);
                        format!("ivec{n}(uvec{n}({}) << {})", widen(a), count_vec(b))
                    }
                    BinaryOp::Shr if int_vector => {
                        let n = ty.int_components().unwrap_or(1);
                        format!("({} >> ivec{n}({}))", widen(a), count_vec(b))
                    }
                    BinaryOp::Ushr if int_vector => {
                        let n = ty.int_components().unwrap_or(1);
                        format!("ivec{n}(uvec{n}({}) >> {})", widen(a), count_vec(b))
                    }
                    BinaryOp::Equal | BinaryOp::NotEqual if matrix.is_some() => {
                        let n = matrix.unwrap_or(0);
                        let same = (0..n)
                            .map(|c| format!("all(equal({}[{c}], {}[{c}]))", v(a), v(b)))
                            .collect::<Vec<_>>()
                            .join(" && ");
                        if *op == BinaryOp::Equal {
                            format!("({same})")
                        } else {
                            format!("(!({same}))")
                        }
                    }
                    BinaryOp::Rem if *ty != Type::Int && !int_vector => format!(
                        "fmod_{}({}, {})",
                        ty.components().unwrap_or(1),
                        widen(a),
                        widen(b)
                    ),
                    BinaryOp::NotEqual if a == b && float_lanes(self.type_of(*a)) => {
                        match self.type_of(*a).lanes() {
                            Some((n, _)) => format!("any(is_nan{n}({}))", v(a)),
                            None => format!("is_nan({})", v(a)),
                        }
                    }
                    BinaryOp::Equal if a == b && float_lanes(self.type_of(*a)) => {
                        match self.type_of(*a).lanes() {
                            Some((n, _)) => format!("(!any(is_nan{n}({})))", v(a)),
                            None => format!("(!is_nan({}))", v(a)),
                        }
                    }
                    BinaryOp::Equal if vector => format!("all(equal({}, {}))", v(a), v(b)),
                    BinaryOp::NotEqual if vector => format!("any(notEqual({}, {}))", v(a), v(b)),
                    BinaryOp::Shl => format!("int(uint({}) << (uint({}) & 31u))", v(a), v(b)),
                    BinaryOp::Shr => format!("({} >> int(uint({}) & 31u))", v(a), v(b)),
                    BinaryOp::Ushr => format!("int(uint({}) >> (uint({}) & 31u))", v(a), v(b)),
                    BinaryOp::Less
                    | BinaryOp::LessEqual
                    | BinaryOp::Greater
                    | BinaryOp::GreaterEqual
                        if self.type_of(*a) == Type::Float =>
                    {
                        format!(
                            "(!is_nan({0}) && !is_nan({1}) && ({0} {symbol} {1}))",
                            v(a),
                            v(b)
                        )
                    }
                    _ => format!("({} {symbol} {})", v(a), v(b)),
                }
            }
            Op::Construct(parts) => {
                let list: Vec<String> = parts.iter().map(v).collect();
                construct_type(self.module, *ty, &list)
            }
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
            Op::Subscript(array, index) => match self.type_of(*array) {
                Type::Struct(struct_index) => {
                    let count = self.module.structs[struct_index as usize].fields.len();
                    format!("{}.m[clamp({}, 0, {})]", v(array), v(index), count - 1)
                }
                other => {
                    let count = other
                        .order()
                        .or(other.lanes().map(|(n, _)| n))
                        .map_or(1, usize::from);
                    format!("{}[clamp({}, 0, {})]", v(array), v(index), count - 1)
                }
            },
            Op::Call(index, args) => format!(
                "{}({})",
                function_name(*index as usize),
                args.iter().map(v).collect::<Vec<_>>().join(", ")
            ),
            Op::Intrinsic(intrinsic, args) => self.intrinsic(*intrinsic, args, *ty),
            Op::Spend(counted) => {
                self.line(depth, &format!("work = work + uint(it{});", counted.0));
                self.line(depth, &format!("int v{} = it{};", id.0, counted.0));
                return;
            }
            Op::Look => {
                self.line(depth, &format!("bool v{} = look();", id.0));
                return;
            }
            Op::If {
                condition,
                then,
                otherwise,
            } if speculable(then) && speculable(otherwise) && selectable(*ty) => {
                let chosen = self.block(then, depth);
                let other = self.block(otherwise, depth);
                format!("({} ? {chosen} : {other})", v(condition))
            }
            Op::If {
                condition,
                then,
                otherwise,
            } => {
                self.line(depth, &format!("{ty_name} v{};", id.0));
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
                self.line(depth, &format!("{ty_name} v{};", id.0));
                self.line(depth, &format!("switch ({}) {{", v(scrutinee)));
                for (values, block) in arms {
                    for &val in &values[..values.len().saturating_sub(1)] {
                        self.line(depth + 1, &format!("case {}:", case_int(val)));
                    }
                    if let Some(&last) = values.last() {
                        self.line(depth + 1, &format!("case {}: {{", case_int(last)));
                        let result = self.block(block, depth + 2);
                        self.line(depth + 2, &format!("v{} = {result};", id.0));
                        self.line(depth + 2, "break;");
                        self.line(depth + 1, "}");
                    }
                }
                self.line(depth + 1, "default: {");
                let result = self.block(otherwise, depth + 2);
                self.line(depth + 2, &format!("v{} = {result};", id.0));
                self.line(depth + 2, "break;");
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
                self.line(depth, &format!("{ty_name} v{} = {};", id.0, v(init)));
                let limit = match bound {
                    Some(bound) => format!("clamp({}, 0, {count})", v(bound)),
                    None => format!("{count}"),
                };
                if self.module.accounted() {
                    self.line(depth, &format!("int it{} = 0;", id.0));
                    self.line(
                        depth,
                        &format!("for (; it{0} < {limit}; it{0} = it{0} + 1) {{", id.0),
                    );
                } else {
                    self.line(
                        depth,
                        &format!(
                            "for (int it{0} = 0; it{0} < {limit}; it{0} = it{0} + 1) {{",
                            id.0
                        ),
                    );
                }
                self.loops.push((*id, true));
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
            Op::Each { count, body } => {
                self.line(depth, &format!("{ty_name} v{};", id.0));
                self.line(
                    depth,
                    &format!(
                        "for (int it{0} = 0; it{0} < {count}; it{0} = it{0} + 1) {{",
                        id.0
                    ),
                );
                self.loops.push((*id, false));
                let element = self.block(body, depth + 1);
                if *count <= WRITTEN_PLACES {
                    self.line(depth + 1, &format!("switch (it{}) {{", id.0));
                    for place in 0..*count {
                        let case = if place + 1 == *count {
                            "default".to_owned()
                        } else {
                            format!("case {place}")
                        };
                        self.line(
                            depth + 2,
                            &format!("{case}: {{ v{}.m[{place}] = {element}; break; }}", id.0),
                        );
                    }
                    self.line(depth + 1, "}");
                } else {
                    self.line(depth + 1, &format!("v{0}.m[it{0}] = {element};", id.0));
                }
                self.loops.pop();
                self.line(depth, "}");
                return;
            }
            Op::Carried => match self.loops.iter().rev().find(|(_, carries)| *carries) {
                Some((id, _)) => format!("v{}", id.0),
                None => unreachable!("WebGL2 runs map kernels only"),
            },
            Op::Index => match self.loops.last() {
                Some((id, _)) => format!("it{}", id.0),
                None => "int(element)".to_owned(),
            },
            Op::Coord => match self.output_shape() {
                Shape::Volume => "coord".to_owned(),
                _ => "coord.xy".to_owned(),
            },
            Op::Domain => match self.output_shape() {
                Shape::Rows => "int(u_ctx[0].x)".to_owned(),
                Shape::Grid => "vec2(u_ctx[0].xy)".to_owned(),
                Shape::Volume => "vec3(vec2(u_ctx[0].xy), float(u_ctx[1].z))".to_owned(),
            },
            Op::Time => "uintBitsToFloat(u_ctx[0].z)".to_owned(),
            Op::Rate => "uintBitsToFloat(u_ctx[1].w)".to_owned(),
            Op::Queued => "int(u_ctx[7].w)".to_owned(),
            Op::Pointer => "uintBitsToFloat(u_ctx[1].xy)".to_owned(),
            Op::PointerDown => "button_down(0)".to_owned(),
            Op::Button(code) => format!("button_down({})", v(code)),
            Op::Wheel => "uintBitsToFloat(u_ctx[4].xy)".to_owned(),
            Op::Pads => "int(uintBitsToFloat(u_ctx[7].z))".to_owned(),
            Op::Stick(which) => format!("stick_at({})", v(which)),
            Op::Window => "uintBitsToFloat(u_ctx[4].zw)".to_owned(),
            Op::Scale => "uintBitsToFloat(u_ctx[5].x)".to_owned(),
            Op::Key(code) => format!("key_down({})", v(code)),
            Op::Typed(k) => format!("typed_at({})", v(k)),
            Op::Element => "int(element)".to_owned(),
            Op::Count(input) => {
                let slot = CONTEXT_HEAD as u32 + input;
                format!("int(u_ctx[{slot}].x * u_ctx[{slot}].y * u_ctx[{slot}].z)")
            }
            Op::Size(input) => {
                let slot = CONTEXT_HEAD as u32 + input;
                match self.input_shape(*input) {
                    Shape::Rows => format!("int(u_ctx[{slot}].x)"),
                    Shape::Grid => format!("vec2(u_ctx[{slot}].xy)"),
                    Shape::Volume => format!("vec3(u_ctx[{slot}].xyz)"),
                }
            }
            Op::Sample(input) => {
                let own = match self.output_shape() {
                    Shape::Rows => "int(element)",
                    Shape::Grid => "coord.xy",
                    Shape::Volume => "coord",
                };
                format!("load_in{input}(index_in{input}({own}))")
            }
            Op::At(input, index) => format!("load_in{input}(index_in{input}({}))", v(index)),
            Op::Part { input, at, path } => {
                let manifest = &self.module.manifest;
                let stride = manifest.inputs[*input as usize].ty.schema.value_words();
                let element_ty = manifest.inputs[*input as usize].element();
                let layout = entelechy_ir::part_layout(&self.module.structs, element_ty, path)
                    .expect("a path validation walked");
                let mut offset = format!("{}", layout.base);
                for (index, count, part) in layout.steps {
                    let _ = write!(
                        offset,
                        " + clamp({}, 0, {}) * {part}",
                        v(&index),
                        count as i64 - 1
                    );
                }
                let last = stride.saturating_sub(type_words(self.module, *ty));
                let addr = match at {
                    Some(at) => v(at),
                    None => match self.output_shape() {
                        Shape::Rows => "int(element)".to_owned(),
                        Shape::Grid => "coord.xy".to_owned(),
                        Shape::Volume => "coord".to_owned(),
                    },
                };
                let base = format!("b{}", id.0);
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
                        "uint {base} = index_in{input}({addr}) * {element_size}u + uint(clamp({offset}, 0, {last})) * {scalar_size}u;"
                    ),
                );
                let (fetcher, pack_offset) =
                    input_source_glsl(self.module, self.packed, *input as usize, encoding);
                load_expr_glsl(
                    self.module,
                    *ty,
                    &fetcher,
                    &format!("({pack_offset}{base})"),
                    0,
                    encoding,
                )
                .0
            }
            Op::AtLinear(input, index) => format!("linear_in{input}({})", v(index)),
            Op::AtFiltered { .. }
            | Op::Clock
            | Op::Lane
            | Op::Exchange(..)
            | Op::Lanes { .. }
            | Op::Meet(_)
            | Op::Emit(_)
            | Op::Put(..) => unreachable!("WebGL2 declines non-map, Double, and filtered modules"),
        };
        self.line(depth, &format!("{ty_name} v{} = {expression};", id.0));
    }

    fn intrinsic(&self, intrinsic: Intrinsic, args: &[ValueId], ty: Type) -> String {
        let v = |id: &ValueId| format!("v{}", id.0);
        let list = args.iter().map(v).collect::<Vec<_>>().join(", ");
        let splat = |id: &ValueId, target: Type| {
            if target.lanes().is_some()
                && matches!(self.type_of(*id), Type::Float | Type::Int | Type::Long)
            {
                format!("{}({})", glsl_type(target), v(id))
            } else {
                v(id)
            }
        };
        let lanes = ty.lanes().map(|(n, _)| n);
        let suffix = |base: &str| match lanes {
            Some(n) => format!("{base}{n}"),
            None => base.to_owned(),
        };
        let long_args = args.iter().any(|arg| self.type_of(*arg) == Type::Long);
        if ty == Type::Long || long_args {
            let a = |k: usize| v(&args[k]);
            return match intrinsic {
                Intrinsic::ToFloat => format!("long_to_f32({})", a(0)),
                Intrinsic::ToInt => format!("int(({}).x)", a(0)),
                Intrinsic::ToLong => match self.type_of(args[0]) {
                    Type::Long => a(0),
                    Type::Float => format!("to_long({})", a(0)),
                    Type::Bool => format!("long_from_bool({})", a(0)),
                    _ => format!("long_from_int({})", a(0)),
                },
                Intrinsic::Sign => format!("sign_i64({})", a(0)),
                Intrinsic::Abs => format!("abs_i64({})", a(0)),
                Intrinsic::Min => format!("min_i64({}, {})", a(0), a(1)),
                Intrinsic::Max => format!("max_i64({}, {})", a(0), a(1)),
                Intrinsic::Clamp => format!("clamp_i64({}, {}, {})", a(0), a(1), a(2)),
                Intrinsic::AddWrap => format!("add_wrap_i64({}, {})", a(0), a(1)),
                Intrinsic::SubWrap => format!("sub_wrap_i64({}, {})", a(0), a(1)),
                Intrinsic::MulWrap => format!("mul_wrap_i64({}, {})", a(0), a(1)),
                other => unreachable!("pairs_take refuses `{}` on a Long", other.name()),
            };
        }
        match intrinsic {
            Intrinsic::Round => format!("roundEven({list})"),
            Intrinsic::Abs if ty.is_int_kind() => {
                let helper = checked_name(ty);
                format!("abs_{helper}({list})")
            }
            Intrinsic::ToFloat => format!("{}({list})", glsl_type(ty)),
            Intrinsic::ToInt => match (self.type_of(args[0]), ty) {
                (Type::Bool, _) => format!("{}({list})", glsl_type(ty)),
                (_, Type::IVec2) => format!("to_int2({list})"),
                (_, Type::IVec3) => format!("to_int3({list})"),
                (_, Type::IVec4) => format!("to_int4({list})"),
                _ => format!("to_int({list})"),
            },
            Intrinsic::Pow => format!(
                "{}({}, {})",
                suffix("pow_ref"),
                splat(&args[0], ty),
                splat(&args[1], ty)
            ),
            Intrinsic::Tanh => {
                let t = glsl_type(ty);
                format!("tanh(clamp({list}, {t}(-15.0), {t}(15.0)))")
            }
            Intrinsic::Atan2 => format!("atan2_ref({}, {})", v(&args[0]), v(&args[1])),
            Intrinsic::Min | Intrinsic::Max | Intrinsic::Clamp if float_lanes(ty) => {
                let s = match ty {
                    Type::Vec2 => "2",
                    Type::Vec3 => "3",
                    Type::Vec4 => "4",
                    _ => "",
                };
                let a: Vec<String> = args.iter().map(|arg| splat(arg, ty)).collect();
                match intrinsic {
                    Intrinsic::Min => format!("min_ref{s}({}, {})", a[0], a[1]),
                    Intrinsic::Max => format!("max_ref{s}({}, {})", a[0], a[1]),
                    _ => format!("min_ref{s}(max_ref{s}({}, {}), {})", a[0], a[1], a[2]),
                }
            }
            Intrinsic::Min | Intrinsic::Max | Intrinsic::Clamp | Intrinsic::Mix => format!(
                "{}({})",
                intrinsic.name(),
                args.iter()
                    .map(|arg| splat(arg, ty))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Intrinsic::AddWrap => format!("int(uint({}) + uint({}))", v(&args[0]), v(&args[1])),
            Intrinsic::SubWrap => format!("int(uint({}) - uint({}))", v(&args[0]), v(&args[1])),
            Intrinsic::MulWrap => format!("int(uint({}) * uint({}))", v(&args[0]), v(&args[1])),
            Intrinsic::Fma => format!(
                "(({}) * ({}) + ({}))",
                v(&args[0]),
                v(&args[1]),
                v(&args[2])
            ),
            Intrinsic::Popcount => format!("int(popcount_u32(uint({list})))"),
            Intrinsic::Clz => format!("int(clz_u32(uint({list})))"),
            Intrinsic::Ctz => format!("int(ctz_u32(uint({list})))"),
            Intrinsic::Bits => format!("floatBitsToInt({list})"),
            Intrinsic::FromBits => format!("intBitsToFloat({list})"),
            Intrinsic::Rotl => format!("rotl_i32({list})"),
            Intrinsic::Rotr => format!("rotr_i32({list})"),
            Intrinsic::Ult => format!("(uint({}) < uint({}))", v(&args[0]), v(&args[1])),
            Intrinsic::Udiv => format!(
                "int(udiv_checked(uint({}), uint({})))",
                v(&args[0]),
                v(&args[1])
            ),
            Intrinsic::Urem => format!(
                "int(urem_checked(uint({}), uint({})))",
                v(&args[0]),
                v(&args[1])
            ),
            Intrinsic::Mulhi => format!("mulhi_i32({list})"),
            Intrinsic::Umulhi => format!(
                "int(umulhi_u32(uint({}), uint({})))",
                v(&args[0]),
                v(&args[1])
            ),
            Intrinsic::Step => format!(
                "{}({}, {})",
                suffix("step_ref"),
                splat(&args[0], ty),
                splat(&args[1], ty)
            ),
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

    fn output_shape(&self) -> Shape {
        self.module
            .manifest
            .outputs
            .first()
            .map_or(Shape::Grid, |p| p.shape())
    }

    fn input_shape(&self, input: u32) -> Shape {
        self.module.manifest.inputs[input as usize].shape()
    }

    fn type_of(&self, id: ValueId) -> Type {
        self.types.get(&id.0).copied().unwrap_or(Type::Float)
    }
}

fn case_int(v: i32) -> String {
    if v == i32::MIN {
        "(-2147483647 - 1)".to_owned()
    } else {
        format!("{v}")
    }
}
