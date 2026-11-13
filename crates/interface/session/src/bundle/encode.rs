//! Binary encoder for [`super::Bundle`].

use std::collections::HashMap;

use entelechy_abi::{ABI_VERSION, Capability, Domain, Scalar, Shape, Type, Value};
use entelechy_graph::{
    Binding, Compare, DomainExpr, DomainOp, DomainSpec, Graph, ParamValue, Source,
};
use entelechy_ir::{
    Block, Combine, Function, IR_VERSION, Instruction, Intrinsic, KernelId, Mode, Module, Op, Port,
    Step, UnaryOp,
};
use entelechy_kernel::FieldData;

use super::{BINARIES, Bundle, MAGIC, VERSION};

impl Bundle {
    /// Encodes the bundle into its self-contained binary representation.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::default();
        w.list(&self.modules, Writer::module);
        w.graph(&self.graph);
        w.u32(self.params.len() as u32);
        for (name, value) in &self.params {
            w.str(name);
            w.value(value);
        }
        w.list(&self.inputs, |w, (name, data)| {
            w.str(name);
            w.field_data(data);
        });
        w.list(&self.presented, |w, (out, dev)| {
            w.str(out);
            w.str(dev);
        });
        w.list(&self.feeds, |w, (inp, dev)| {
            w.str(inp);
            w.str(dev);
        });
        w.list(&self.fonts, |w, font| {
            w.str(&font.node);
            w.source(&font.codes);
            for out in &font.outputs {
                w.str(out);
            }
            w.field_data(&font.glyphs);
            w.field_data(&font.segments);
            w.field_data(&font.metrics);
        });
        w.finish()
    }
}

#[derive(Default)]
struct Writer {
    strings: Vec<String>,
    intern: HashMap<String, u32>,
    body: Vec<u8>,
}

fn push_leb(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let mut b = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            b |= 0x80;
        }
        out.push(b);
        if v == 0 {
            break;
        }
    }
}

impl Writer {
    fn finish(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(32 + self.body.len());
        out.extend_from_slice(MAGIC);
        push_leb(&mut out, u64::from(VERSION));
        push_leb(&mut out, u64::from(IR_VERSION));
        push_leb(&mut out, u64::from(ABI_VERSION));
        push_leb(&mut out, self.strings.len() as u64);
        for s in &self.strings {
            push_leb(&mut out, s.len() as u64);
            out.extend_from_slice(s.as_bytes());
        }
        out.extend_from_slice(&self.body);
        out
    }

    fn byte(&mut self, b: u8) {
        self.body.push(b);
    }
    fn bool(&mut self, b: bool) {
        self.byte(u8::from(b));
    }
    fn u32(&mut self, v: u32) {
        push_leb(&mut self.body, u64::from(v));
    }
    fn i32(&mut self, v: i32) {
        self.u32(((v << 1) ^ (v >> 31)) as u32);
    }
    fn i64(&mut self, v: i64) {
        push_leb(&mut self.body, ((v << 1) ^ (v >> 63)) as u64);
    }
    fn f32(&mut self, v: f32) {
        self.body.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.body.extend_from_slice(&v.to_le_bytes());
    }

    fn str(&mut self, s: &str) {
        let idx = if let Some(&idx) = self.intern.get(s) {
            idx
        } else {
            let idx = self.strings.len() as u32;
            self.strings.push(s.to_owned());
            self.intern.insert(s.to_owned(), idx);
            idx
        };
        self.u32(idx);
    }

    fn list<T>(&mut self, items: &[T], mut f: impl FnMut(&mut Self, &T)) {
        self.u32(items.len() as u32);
        for item in items {
            f(self, item);
        }
    }

    fn opt<T>(&mut self, item: Option<&T>, f: impl FnOnce(&mut Self, &T)) {
        match item {
            None => self.byte(0),
            Some(v) => {
                self.byte(1);
                f(self, v);
            }
        }
    }

    fn tag_u32(&mut self, tag: u8, v: u32) {
        self.byte(tag);
        self.u32(v);
    }

    fn kernel_id(&mut self, id: &KernelId) {
        self.str(id.name());
        self.u32(id.version());
    }

    fn scalar(&mut self, s: Scalar) {
        self.byte(match s {
            Scalar::Float => 0,
            Scalar::Int => 1,
            Scalar::Bool => 2,
            Scalar::Double => 3,
            Scalar::Long => 4,
        });
    }

    fn ty(&mut self, ty: Type) {
        match ty {
            Type::Scalar(s) => {
                self.byte(0);
                self.scalar(s);
            }
            Type::Vector(s, n) => {
                self.byte(1);
                self.scalar(s);
                self.byte(n);
            }
            Type::Matrix(s, n) => {
                self.byte(2);
                self.scalar(s);
                self.byte(n);
            }
            Type::Struct(idx) => self.tag_u32(3, idx),
        }
    }

    fn shape(&mut self, shape: Shape) {
        self.byte(match shape {
            Shape::Rows => 0,
            Shape::Grid => 1,
            Shape::Volume => 2,
        });
    }

    fn domain(&mut self, domain: Domain) {
        match domain {
            Domain::Rows(n) => self.tag_u32(0, n),
            Domain::Grid(w, h) => {
                self.tag_u32(1, w);
                self.u32(h);
            }
            Domain::Volume(w, h, d) => {
                self.tag_u32(2, w);
                self.u32(h);
                self.u32(d);
            }
        }
    }

    fn port(&mut self, port: &Port) {
        self.str(&port.name);
        self.ty(port.ty.element);
        self.shape(port.ty.shape);
        self.str(&port.ty.schema.to_string());
    }

    fn f32s(&mut self, tag: u8, vs: &[f32]) {
        self.byte(tag);
        vs.iter().for_each(|x| self.f32(*x));
    }

    fn f64s(&mut self, tag: u8, vs: &[f64]) {
        self.byte(tag);
        vs.iter().for_each(|x| self.f64(*x));
    }

    fn i32s(&mut self, tag: u8, vs: &[i32]) {
        self.byte(tag);
        vs.iter().for_each(|x| self.i32(*x));
    }

    fn i64s(&mut self, tag: u8, vs: &[i64]) {
        self.byte(tag);
        vs.iter().for_each(|x| self.i64(*x));
    }

    fn value(&mut self, value: &Value) {
        match value {
            Value::Float(v) => self.f32s(0, &[*v]),
            Value::Int(v) => self.i32s(1, &[*v]),
            Value::Bool(v) => {
                self.byte(2);
                self.bool(*v);
            }
            Value::Vec2(v) => self.f32s(3, v),
            Value::Vec3(v) => self.f32s(4, v),
            Value::Vec4(v) => self.f32s(5, v),
            Value::IVec2(v) => self.i32s(6, v),
            Value::IVec3(v) => self.i32s(7, v),
            Value::IVec4(v) => self.i32s(8, v),
            Value::Mat2(v) => self.f32s(9, v),
            Value::Mat3(v) => self.f32s(10, v.as_slice()),
            Value::Mat4(v) => self.f32s(11, v.as_slice()),
            Value::Double(v) => self.f64s(12, &[*v]),
            Value::Long(v) => self.i64s(13, &[*v]),
            Value::DVec2(v) => self.f64s(14, v),
            Value::DVec3(v) => self.f64s(15, v.as_slice()),
            Value::DVec4(v) => self.f64s(16, v.as_slice()),
            Value::LVec2(v) => self.i64s(17, v),
            Value::LVec3(v) => self.i64s(18, v.as_slice()),
            Value::DMat2(v) => self.f64s(19, v.as_slice()),
            Value::DMat3(v) => self.f64s(20, v.as_slice()),
            Value::DMat4(v) => self.f64s(21, v.as_slice()),
            Value::Record(items) => {
                self.byte(22);
                self.list(items, Self::value);
            }
        }
    }

    fn combine(&mut self, c: Combine) {
        self.byte(match c {
            Combine::Sum => 0,
            Combine::Min => 1,
            Combine::Max => 2,
            Combine::Last => 3,
        });
    }

    fn module(&mut self, m: &Module) {
        self.kernel_id(&m.manifest.id);
        match m.manifest.mode {
            Mode::Map => self.byte(0),
            Mode::Reduce => self.byte(1),
            Mode::Scan => self.byte(2),
            Mode::Scatter { base, combine } => {
                self.tag_u32(3, base);
                self.combine(combine);
            }
            Mode::Prefix { combine } => {
                self.byte(4);
                self.combine(combine);
            }
            Mode::Group { size } => self.tag_u32(5, size),
        }
        let mut caps = 0u8;
        for (bit, cap) in Capability::ALL.into_iter().enumerate() {
            if m.manifest.requires.contains(cap) {
                caps |= 1 << bit;
            }
        }
        self.byte(caps);
        self.list(&m.manifest.inputs, Self::port);
        self.list(&m.manifest.params, |w, p| {
            w.str(&p.name);
            w.ty(p.ty);
            w.value(&p.default);
        });
        self.list(&m.manifest.outputs, Self::port);
        self.list(&m.manifest.memories, |w, name| w.str(name));
        self.u32(m.manifest.budget);
        self.list(&m.structs, |w, s| match s.element {
            Some(elem) => {
                w.byte(1);
                w.str(&s.name);
                w.ty(elem);
                w.u32(s.fields.len() as u32);
            }
            None => {
                w.byte(0);
                w.str(&s.name);
                w.list(&s.fields, |w, f| {
                    w.str(&f.name);
                    w.ty(f.ty);
                });
            }
        });
        self.list(&m.functions, Self::function);
        self.function(&m.entry);
        self.opt(m.merge.as_ref(), Self::function);
        self.opt(m.init.as_ref(), Self::function);
    }

    fn function(&mut self, f: &Function) {
        self.str(&f.name);
        self.list(&f.args, |w, &ty| w.ty(ty));
        self.list(&f.body, Self::instruction);
        self.list(&f.results, |w, id| w.u32(id.0));
    }

    fn block(&mut self, b: &Block) {
        self.list(&b.body, Self::instruction);
        self.u32(b.result.0);
    }

    fn instruction(&mut self, inst: &Instruction) {
        self.ty(inst.ty);
        match &inst.op {
            Op::Const(v) => {
                self.byte(0);
                self.value(v);
            }
            Op::Param(i) => self.tag_u32(1, *i),
            Op::Arg(i) => self.tag_u32(2, *i),
            Op::Unary(op, a) => {
                self.byte(3);
                self.byte(match op {
                    UnaryOp::Neg => 0,
                    UnaryOp::Not => 1,
                });
                self.u32(a.0);
            }
            Op::Binary(op, a, b) => {
                self.byte(4);
                self.byte(BINARIES.iter().position(|x| x == op).unwrap_or(0) as u8);
                self.u32(a.0);
                self.u32(b.0);
            }
            Op::Construct(vs) => {
                self.byte(5);
                self.list(vs, |w, v| w.u32(v.0));
            }
            Op::Extract(v, k) => {
                self.tag_u32(6, v.0);
                self.u32(*k);
            }
            Op::Subscript(v, idx) => {
                self.tag_u32(7, v.0);
                self.u32(idx.0);
            }
            Op::Call(fn_idx, args) => {
                self.tag_u32(8, *fn_idx);
                self.list(args, |w, v| w.u32(v.0));
            }
            Op::Intrinsic(intr, args) => {
                self.byte(9);
                let idx = Intrinsic::ALL
                    .iter()
                    .position(|(k, _)| k == intr)
                    .unwrap_or(0);
                self.byte(idx as u8);
                self.list(args, |w, v| w.u32(v.0));
            }
            Op::If {
                condition,
                then,
                otherwise,
            } => {
                self.tag_u32(10, condition.0);
                self.block(then);
                self.block(otherwise);
            }
            Op::Match {
                scrutinee,
                arms,
                otherwise,
            } => {
                self.tag_u32(11, scrutinee.0);
                self.list(arms, |w, (vals, blk)| {
                    w.list(vals, |w, &v| w.i32(v));
                    w.block(blk);
                });
                self.block(otherwise);
            }
            Op::Loop {
                count,
                bound,
                init,
                until,
                body,
            } => {
                self.tag_u32(12, *count);
                self.opt(bound.as_ref(), |w, id| w.u32(id.0));
                self.u32(init.0);
                self.opt(until.as_ref(), Self::block);
                self.block(body);
            }
            Op::Each { count, body } => {
                self.tag_u32(13, *count);
                self.block(body);
            }
            Op::Carried => self.byte(14),
            Op::Index => self.byte(15),
            Op::Coord => self.byte(16),
            Op::Domain => self.byte(17),
            Op::Time => self.byte(18),
            Op::Clock => self.byte(19),
            Op::Rate => self.byte(20),
            Op::Queued => self.byte(21),
            Op::Pointer => self.byte(22),
            Op::PointerDown => self.byte(23),
            Op::Button(v) => self.tag_u32(24, v.0),
            Op::Wheel => self.byte(25),
            Op::Pads => self.byte(26),
            Op::Stick(v) => self.tag_u32(27, v.0),
            Op::Window => self.byte(28),
            Op::Scale => self.byte(29),
            Op::Key(v) => self.tag_u32(30, v.0),
            Op::Typed(v) => self.tag_u32(31, v.0),
            Op::Element => self.byte(32),
            Op::Lane => self.byte(33),
            Op::Exchange(a, b) => {
                self.tag_u32(34, a.0);
                self.u32(b.0);
            }
            Op::Sample(i) => self.tag_u32(35, *i),
            Op::At(i, v) => {
                self.tag_u32(36, *i);
                self.u32(v.0);
            }
            Op::Part { input, at, path } => {
                self.tag_u32(37, *input);
                self.opt(at.as_ref(), |w, v| w.u32(v.0));
                self.list(path, |w, step| match step {
                    Step::Field(k) => w.tag_u32(0, *k),
                    Step::Index(v) => w.tag_u32(1, v.0),
                });
            }
            Op::AtLinear(i, v) => {
                self.tag_u32(38, *i);
                self.u32(v.0);
            }
            Op::Count(i) => self.tag_u32(39, *i),
            Op::Size(i) => self.tag_u32(40, *i),
            Op::Put(i, at, v) => {
                self.tag_u32(41, *i);
                self.u32(at.0);
                self.u32(v.0);
            }
            Op::Emit(v) => self.tag_u32(42, v.0),
            Op::Spend(v) => self.tag_u32(43, v.0),
            Op::Look => self.byte(44),
            Op::Lanes { body } => {
                self.byte(45);
                self.block(body);
            }
            Op::Meet(v) => {
                self.byte(46);
                self.opt(v.as_ref(), |w, id| w.u32(id.0));
            }
        }
    }

    fn source(&mut self, s: &Source) {
        match s {
            Source::External(name) => {
                self.byte(0);
                self.str(name);
            }
            Source::Node { node, port } => {
                self.byte(1);
                self.str(node.0.as_str());
                self.str(port);
            }
        }
    }

    fn domain_expr(&mut self, expr: &DomainExpr) {
        match expr {
            DomainExpr::Int(v) => {
                self.byte(0);
                self.i64(*v);
            }
            DomainExpr::Float(v) => {
                self.byte(1);
                self.f64(*v);
            }
            DomainExpr::Name(s) => {
                self.byte(2);
                self.str(s);
            }
            DomainExpr::Count(s) => {
                self.byte(3);
                self.str(s);
            }
            DomainExpr::Size(s) => {
                self.byte(4);
                self.str(s);
            }
            DomainExpr::Value(s, row) => {
                self.byte(5);
                self.str(s);
                self.u32(*row);
            }
            DomainExpr::Context(s) => {
                self.byte(6);
                self.str(s);
            }
            DomainExpr::Vector(parts) => {
                self.byte(7);
                self.list(parts, Self::domain_expr);
            }
            DomainExpr::Call(name, args) => {
                self.byte(8);
                self.str(name);
                self.list(args, Self::domain_expr);
            }
            DomainExpr::Neg(inner) => {
                self.byte(9);
                self.domain_expr(inner);
            }
            DomainExpr::Binary(op, a, b) => {
                self.byte(10);
                self.byte(match op {
                    DomainOp::Add => 0,
                    DomainOp::Sub => 1,
                    DomainOp::Mul => 2,
                    DomainOp::Div => 3,
                    DomainOp::Shl => 4,
                });
                self.domain_expr(a);
                self.domain_expr(b);
            }
        }
    }

    fn graph(&mut self, g: &Graph) {
        self.str(&g.name);
        self.list(&g.externals, |w, s| w.str(s));
        self.list(&g.texts, |w, (k, v)| {
            w.str(k);
            w.str(v);
        });
        self.list(&g.params, |w, p| {
            w.str(&p.name);
            w.str(&p.ty);
            w.value(&p.default);
        });
        self.list(&g.requirements, |w, req| {
            w.str(&req.name);
            w.domain_expr(&req.left);
            w.byte(match req.compare {
                Compare::Less => 0,
                Compare::AtMost => 1,
                Compare::Greater => 2,
                Compare::AtLeast => 3,
                Compare::Equal => 4,
                Compare::Unequal => 5,
            });
            w.domain_expr(&req.right);
        });
        self.list(&g.times_params, |w, s| w.str(s));
        self.list(&g.nodes, |w, node| {
            w.str(node.id.0.as_str());
            w.kernel_id(&node.kernel);
            w.u32(node.inputs.len() as u32);
            for (port, binding) in &node.inputs {
                w.str(port);
                match binding {
                    Binding::Now(src) => {
                        w.byte(0);
                        w.source(src);
                    }
                    Binding::Previous {
                        source,
                        initial,
                        migration,
                    } => {
                        w.byte(1);
                        w.source(source);
                        w.source(initial);
                        w.opt(migration.as_ref(), Self::kernel_id);
                    }
                }
            }
            w.u32(node.params.len() as u32);
            for (name, pv) in &node.params {
                w.str(name);
                match pv {
                    ParamValue::Literal(v) => {
                        w.byte(0);
                        w.value(v);
                    }
                    ParamValue::Graph(gname) => {
                        w.byte(1);
                        w.str(gname);
                    }
                }
            }
            w.opt(node.domain.as_ref(), |w, spec| match spec {
                DomainSpec::Literal(d) => {
                    w.byte(0);
                    w.domain(*d);
                }
                DomainSpec::Expr(e) => {
                    w.byte(1);
                    w.domain_expr(e);
                }
            });
            w.bool(node.is_async);
            w.list(&node.awake, |w, a| {
                w.source(&a.source);
                w.u32(a.row);
            });
        });
        let pairs = |w: &mut Self, items: &[(String, Source)]| {
            w.list(items, |w, (name, src)| {
                w.str(name);
                w.source(src);
            });
        };
        pairs(self, &g.outputs);
        pairs(self, &g.fallbacks);
        self.list(&g.placed, |w, (id, outs)| {
            w.str(id);
            pairs(w, outs);
        });
        self.list(&g.repeated, |w, (name, last)| {
            w.str(name);
            w.str(last.0.as_str());
        });
    }

    fn field_data(&mut self, data: &FieldData) {
        self.str(&data.schema().to_string());
        self.domain(data.domain());
        let byte_len = data
            .domain()
            .count()
            .saturating_mul(data.schema().element_bytes());
        self.u32(byte_len as u32);
        let mut left = byte_len;
        for &word in data.storage() {
            if left == 0 {
                break;
            }
            let bytes = word.to_le_bytes();
            let take = left.min(4);
            self.body.extend_from_slice(&bytes[..take]);
            left -= take;
        }
        if left > 0 {
            self.body.extend(std::iter::repeat_n(0, left));
        }
    }
}
