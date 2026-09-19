//! WGSL emission from validated Orla IR.
//!
//! Every value slot becomes a WGSL `let` (straight-line values) or `var`
//! (results of `if` and `loop`). Blocks map to WGSL blocks, so IR scoping
//! and WGSL scoping coincide. Resources are storage buffers of `vec4<f32>`
//! plus two small uniform arrays for context and parameters.

use std::fmt::Write;

use orla_ir::{
    BinaryOp, Block, Function, Instruction, Intrinsic, Module, Op, Type, UnaryOp, Value, ValueId,
};

/// Binding index of the context uniform.
pub const CONTEXT_BINDING: u32 = 0;
/// Binding index of the parameter uniform.
pub const PARAMS_BINDING: u32 = 1;
/// Binding index of the output storage buffer.
pub const OUTPUT_BINDING: u32 = 2;
/// Binding index of the first input storage buffer; input `i` is at
/// `INPUT_BINDING + i`.
pub const INPUT_BINDING: u32 = 3;
/// Workgroup edge length in both dimensions.
pub const WORKGROUP: u32 = 8;

fn wgsl_type(ty: Type) -> String {
    match ty {
        Type::Float => "f32".into(),
        Type::Int => "i32".into(),
        Type::Bool => "bool".into(),
        Type::Vec2 => "vec2<f32>".into(),
        Type::Vec3 => "vec3<f32>".into(),
        Type::Vec4 => "vec4<f32>".into(),
        Type::Image => "vec4<f32>".into(),
        Type::Struct(index) => format!("S{index}"),
    }
}

fn literal(value: &Value) -> String {
    let f = |v: f32| {
        if v.is_nan() {
            "(0.0 / 0.0)".to_owned()
        } else if v.is_infinite() {
            if v > 0.0 {
                "(1.0 / 0.0)".to_owned()
            } else {
                "(-1.0 / 0.0)".to_owned()
            }
        } else {
            format!("{v:?}")
        }
    };
    match value {
        Value::Float(v) => f(*v),
        Value::Int(v) => format!("{v}i"),
        Value::Bool(v) => v.to_string(),
        Value::Vec2([x, y]) => format!("vec2<f32>({}, {})", f(*x), f(*y)),
        Value::Vec3([x, y, z]) => format!("vec3<f32>({}, {}, {})", f(*x), f(*y), f(*z)),
        Value::Vec4([x, y, z, w]) => {
            format!("vec4<f32>({}, {}, {}, {})", f(*x), f(*y), f(*z), f(*w))
        }
    }
}

/// Emits a complete compute shader for the module.
pub fn emit(module: &Module) -> String {
    let manifest = &module.manifest;
    let mut out = String::new();
    let context_len = 1 + manifest.inputs.len();
    let params_len = manifest.params.len().max(1);
    let _ = writeln!(
        out,
        "@group(0) @binding({CONTEXT_BINDING}) var<uniform> ctx: array<vec4<f32>, {context_len}>;"
    );
    let _ = writeln!(
        out,
        "@group(0) @binding({PARAMS_BINDING}) var<uniform> params: array<vec4<f32>, {params_len}>;"
    );
    let _ = writeln!(
        out,
        "@group(0) @binding({OUTPUT_BINDING}) var<storage, read_write> out0: array<vec4<f32>>;"
    );
    for index in 0..manifest.inputs.len() {
        let _ = writeln!(
            out,
            "@group(0) @binding({}) var<storage, read> in{index}: array<vec4<f32>>;",
            INPUT_BINDING + index as u32
        );
    }
    out.push('\n');
    for (index, def) in module.structs.iter().enumerate() {
        let fields = def
            .fields
            .iter()
            .enumerate()
            .map(|(k, field)| format!("f{k}: {}", wgsl_type(field.ty)))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(out, "struct S{index} {{ {fields} }}");
    }
    for index in 0..manifest.inputs.len() {
        let slot = 1 + index;
        let _ = writeln!(
            out,
            "fn fetch{index}(c: vec2<f32>) -> vec4<f32> {{
    let w = ctx[{slot}].x;
    let h = ctx[{slot}].y;
    let x = clamp(select(floor(c.x), 0.0, c.x != c.x), 0.0, w - 1.0);
    let y = clamp(select(floor(c.y), 0.0, c.y != c.y), 0.0, h - 1.0);
    return in{index}[u32(y) * u32(w) + u32(x)];
}}"
        );
    }
    for ty in [Type::Vec2, Type::Vec3, Type::Vec4] {
        let name = wgsl_type(ty);
        let _ = writeln!(
            out,
            "fn safe_normalize_{}(v: {name}) -> {name} {{
    let l = length(v);
    if (l > 0.0) {{ return v / l; }}
    return {name}(0.0);
}}",
            ty.components().unwrap_or(0)
        );
    }
    out.push('\n');
    for (index, function) in module.functions.iter().enumerate() {
        emit_function(&mut out, module, function, &format!("f{index}"), false);
        out.push('\n');
    }
    emit_function(&mut out, module, &module.entry, "entry", true);
    let _ = writeln!(
        out,
        "
@compute @workgroup_size({WORKGROUP}, {WORKGROUP}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let w = u32(ctx[0].x);
    let h = u32(ctx[0].y);
    if (gid.x >= w || gid.y >= h) {{
        return;
    }}
    let coord = vec2<f32>(f32(gid.x), f32(gid.y));
    out0[gid.y * w + gid.x] = entry(coord);
}}"
    );
    out
}

fn emit_function(out: &mut String, module: &Module, function: &Function, name: &str, entry: bool) {
    let args = if entry {
        "coord: vec2<f32>".to_owned()
    } else {
        function
            .args
            .iter()
            .enumerate()
            .map(|(index, ty)| format!("a{index}: {}", wgsl_type(*ty)))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let ret = result_type(function);
    let _ = writeln!(out, "fn {name}({args}) -> {} {{", wgsl_type(ret));
    let mut emitter = Emitter {
        out,
        module,
        function,
        loops: Vec::new(),
    };
    emitter.instructions(&function.body, 1);
    let _ = writeln!(out, "    return v{};", function.results[0].0);
    let _ = writeln!(out, "}}");
}

fn result_type(function: &Function) -> Type {
    fn find(instructions: &[Instruction], id: ValueId) -> Option<Type> {
        for instruction in instructions {
            if instruction.id == id {
                return Some(instruction.ty);
            }
            let nested = match &instruction.op {
                Op::If {
                    then, otherwise, ..
                } => find(&then.body, id).or_else(|| find(&otherwise.body, id)),
                Op::Loop { until, body, .. } => until
                    .as_ref()
                    .and_then(|block| find(&block.body, id))
                    .or_else(|| find(&body.body, id)),
                _ => None,
            };
            if nested.is_some() {
                return nested;
            }
        }
        None
    }
    find(&function.body, function.results[0]).unwrap_or(Type::Float)
}

struct Emitter<'a> {
    out: &'a mut String,
    module: &'a Module,
    function: &'a Function,
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
                let slot = format!("params[{index}]");
                match self.module.manifest.params[*index as usize].default.ty() {
                    Type::Float => format!("{slot}.x"),
                    Type::Int => format!("bitcast<i32>({slot}.x)"),
                    Type::Bool => format!("({slot}.x != 0.0)"),
                    Type::Vec2 => format!("{slot}.xy"),
                    Type::Vec3 => format!("{slot}.xyz"),
                    Type::Vec4 | Type::Image | Type::Struct(_) => slot,
                }
            }
            Op::Arg(index) => format!("a{index}"),
            Op::Unary(UnaryOp::Neg, a) => format!("(-{})", v(a)),
            Op::Unary(UnaryOp::Not, a) => format!("(!{})", v(a)),
            Op::Binary(op, a, b) => {
                let symbol = match op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "/",
                    BinaryOp::Less => "<",
                    BinaryOp::LessEqual => "<=",
                    BinaryOp::Greater => ">",
                    BinaryOp::GreaterEqual => ">=",
                    BinaryOp::Equal => "==",
                    BinaryOp::NotEqual => "!=",
                    BinaryOp::And => "&&",
                    BinaryOp::Or => "||",
                };
                format!("({} {symbol} {})", v(a), v(b))
            }
            Op::Construct(parts) => format!(
                "{ty_name}({})",
                parts.iter().map(v).collect::<Vec<_>>().join(", ")
            ),
            Op::Extract(source, component) => {
                if let Type::Struct(_) = self.type_of(*source) {
                    format!("{}.f{component}", v(source))
                } else {
                    format!(
                        "{}.{}",
                        v(source),
                        ["x", "y", "z", "w"][*component as usize]
                    )
                }
            }
            Op::Call(index, args) => format!(
                "f{index}({})",
                args.iter().map(v).collect::<Vec<_>>().join(", ")
            ),
            Op::Intrinsic(intrinsic, args) => self.intrinsic(*intrinsic, args, *ty),
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
            Op::Loop {
                count,
                init,
                until,
                body,
            } => {
                self.line(depth, &format!("var v{}: {ty_name} = {};", id.0, v(init)));
                self.line(
                    depth,
                    &format!(
                        "for (var i{0}: i32 = 0; i{0} < {count}i; i{0} = i{0} + 1) {{",
                        id.0
                    ),
                );
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
            Op::Carried => format!("v{}", self.loops.last().map_or(0, |id| id.0)),
            Op::Index => format!("i{}", self.loops.last().map_or(0, |id| id.0)),
            Op::Coord => "coord".to_owned(),
            Op::Domain => "ctx[0].xy".to_owned(),
            Op::Time => "ctx[0].z".to_owned(),
            Op::ImageSize(input) => format!("ctx[{}].xy", 1 + input),
            Op::ImageSample(input) => format!("fetch{input}(coord)"),
            Op::ImageFetch(input, coord) => format!("fetch{input}({})", v(coord)),
        };
        let _ = self.function;
        self.line(depth, &format!("let v{}: {ty_name} = {expression};", id.0));
    }

    fn intrinsic(&self, intrinsic: Intrinsic, args: &[ValueId], ty: Type) -> String {
        let v = |id: &ValueId| format!("v{}", id.0);
        let list = args.iter().map(v).collect::<Vec<_>>().join(", ");
        let splat = |id: &ValueId, target: Type| {
            if target.components().is_some() && self.type_of(*id) == Type::Float {
                format!("{}({})", wgsl_type(target), v(id))
            } else {
                v(id)
            }
        };
        match intrinsic {
            Intrinsic::ToFloat => format!("f32({list})"),
            Intrinsic::ToInt => format!("i32({list})"),
            Intrinsic::Normalize => {
                format!("safe_normalize_{}({list})", ty.components().unwrap_or(0))
            }
            Intrinsic::Pow => format!("pow({}, {})", v(&args[0]), splat(&args[1], ty)),
            Intrinsic::Step => format!("step({}, {})", splat(&args[0], ty), v(&args[1])),
            Intrinsic::Smoothstep => format!(
                "smoothstep({}, {}, {})",
                splat(&args[0], ty),
                splat(&args[1], ty),
                v(&args[2])
            ),
            other => format!("{}({list})", other.name()),
        }
    }

    fn type_of(&self, id: ValueId) -> Type {
        fn find(instructions: &[Instruction], id: ValueId) -> Option<Type> {
            for instruction in instructions {
                if instruction.id == id {
                    return Some(instruction.ty);
                }
                let nested = match &instruction.op {
                    Op::If {
                        then, otherwise, ..
                    } => find(&then.body, id).or_else(|| find(&otherwise.body, id)),
                    Op::Loop { until, body, .. } => until
                        .as_ref()
                        .and_then(|block| find(&block.body, id))
                        .or_else(|| find(&body.body, id)),
                    _ => None,
                };
                if nested.is_some() {
                    return nested;
                }
            }
            None
        }
        find(&self.function.body, id).unwrap_or(Type::Float)
    }
}
