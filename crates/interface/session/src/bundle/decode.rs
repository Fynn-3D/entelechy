//! Binary decoder for [`super::Bundle`].

use std::collections::BTreeMap;
use std::sync::Arc;

use entelechy_abi::{
    ABI_VERSION, Capabilities, Capability, Domain, PortType, Scalar, Schema, Shape, Type, Value,
};
use entelechy_graph::{
    Awake, Binding, Compare, DomainExpr, DomainOp, DomainSpec, Graph, GraphParam, Node, NodeId,
    ParamValue, Requirement, Source,
};
use entelechy_ir::{
    Block, Combine, Field, Function, IR_VERSION, Instruction, Intrinsic, KernelId, MAX_ARRAY_COUNT,
    Manifest, Mode, Module, Op, Param, Port, Step, StructDef, UnaryOp, ValueId,
};
use entelechy_kernel::FieldData;

use super::{BINARIES, Bundle, BundledAtlas, BundledFont, MAGIC, MAX_ITEMS, MAX_NESTING, VERSION};
use crate::memory::COPY_LEAST;

impl Bundle {
    /// Decodes a bundle produced by [`Bundle::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Reader::open(bytes)?;
        let modules = r.list(MAX_ITEMS, Reader::module)?;
        let graph = r.graph()?;
        let param_count = r.len(MAX_ITEMS)?;
        let mut params = BTreeMap::new();
        for _ in 0..param_count {
            params.insert(r.str()?, r.value(0)?);
        }
        // Scan static input slices in stream order, deferring FieldData word
        // allocations until after permanent metadata and font atlases land,
        // and placing large device-bound inputs at the top of the heap last.
        let input_slices = r.list(MAX_ITEMS, |r| {
            let name = r.str()?;
            let (schema, domain, raw) = r.field_slice()?;
            Ok((name, schema, domain, raw))
        })?;
        let presented = r.list(MAX_ITEMS, |r| Ok((r.str()?, r.str()?)))?;
        let feeds = r.list(MAX_ITEMS, |r| Ok((r.str()?, r.str()?)))?;
        let font_count = r.len(MAX_ITEMS)?;
        let mut fonts: Vec<BundledFont> = Vec::with_capacity(font_count.min(64));
        for _ in 0..font_count {
            let node = r.str()?;
            let codes = r.source()?;
            let outputs = [r.str()?, r.str()?, r.str()?];
            let atlas = match r.byte()? {
                0 => {
                    let idx = r.len(fonts.len().saturating_sub(1))?;
                    Arc::clone(
                        &fonts
                            .get(idx)
                            .ok_or_else(|| "bundle font atlas index out of range".to_owned())?
                            .atlas,
                    )
                }
                1 => Arc::new(BundledAtlas::Fields {
                    glyphs: r.field_data()?,
                    segments: r.field_data()?,
                    metrics: r.field_data()?,
                }),
                2 => {
                    let compact = entelechy_formats::font::CompactFont::read_from(&mut r.rest)?;
                    Arc::new(BundledAtlas::Compact(compact))
                }
                other => return Err(format!("unknown bundled font tag {other}")),
            };
            fonts.push(BundledFont {
                node,
                codes,
                outputs,
                atlas,
            });
        }
        if !r.rest.is_empty() {
            return Err(format!(
                "bundle has {} trailing byte(s) after its end",
                r.rest.len()
            ));
        }
        drop(r);

        let mut staged: Vec<Option<(String, FieldData)>> = Vec::with_capacity(input_slices.len());
        staged.resize_with(input_slices.len(), || None);
        let mut inputs = Vec::with_capacity(input_slices.len());
        let mut pending = input_slices
            .into_iter()
            .enumerate()
            .map(Some)
            .collect::<Vec<_>>();
        for pass_large in [false, true] {
            for entry in &mut pending {
                let is_large = entry
                    .as_ref()
                    .is_some_and(|(_, (_, _, _, raw))| (raw.len() as u64) >= COPY_LEAST);
                if is_large == pass_large
                    && let Some((idx, (name, schema, domain, raw))) = entry.take()
                {
                    let data = decode_field_data(schema, domain, raw)?;
                    staged[idx] = Some((name, data));
                }
            }
        }
        drop(pending);
        for slot in staged {
            inputs.push(slot.expect("every input slice decoded"));
        }

        Ok(Self {
            modules,
            graph,
            params,
            inputs,
            presented,
            feeds,
            fonts,
        })
    }
}

struct Reader<'a> {
    strings: Vec<String>,
    rest: &'a [u8],
}

fn read_leb(rest: &mut &[u8]) -> Result<u64, String> {
    let mut val = 0u64;
    let mut shift = 0u32;
    loop {
        let (&b, tail) = rest
            .split_first()
            .ok_or_else(|| "bundle ended unexpectedly".to_owned())?;
        *rest = tail;
        if shift >= 64 || (shift == 63 && (b & 0x7f) > 1) {
            return Err("bundle integer overflow".to_owned());
        }
        val |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return Ok(val);
        }
        shift += 7;
    }
}

impl<'a> Reader<'a> {
    fn open(bytes: &'a [u8]) -> Result<Self, String> {
        let mut rest = bytes
            .strip_prefix(MAGIC)
            .ok_or_else(|| "not an Entelechy scene bundle".to_owned())?;
        let version = u32::try_from(read_leb(&mut rest)?).map_err(|_| "bad bundle version")?;
        if version != VERSION {
            return Err(format!(
                "bundle format v{version} is not v{VERSION} expected by this runtime"
            ));
        }
        let ir = u32::try_from(read_leb(&mut rest)?).map_err(|_| "bad IR version")?;
        if ir != IR_VERSION {
            return Err(format!("bundle IR v{ir} is not v{IR_VERSION}"));
        }
        let abi = u32::try_from(read_leb(&mut rest)?).map_err(|_| "bad ABI version")?;
        if abi != ABI_VERSION {
            return Err(format!("bundle ABI v{abi} is not v{ABI_VERSION}"));
        }
        let count = usize::try_from(read_leb(&mut rest)?).map_err(|_| "bad string count")?;
        if count > MAX_ITEMS {
            return Err("bundle string table too large".to_owned());
        }
        let mut strings = Vec::with_capacity(count.min(4096));
        for _ in 0..count {
            let len = usize::try_from(read_leb(&mut rest)?).map_err(|_| "bad string length")?;
            if len > rest.len() {
                return Err("bundle string truncated".to_owned());
            }
            let (raw, tail) = rest.split_at(len);
            rest = tail;
            let text = std::str::from_utf8(raw).map_err(|_| "bundle string is not valid UTF-8")?;
            strings.push(text.to_owned());
        }
        Ok(Self { strings, rest })
    }

    fn byte(&mut self) -> Result<u8, String> {
        let (&b, tail) = self
            .rest
            .split_first()
            .ok_or_else(|| "bundle ended unexpectedly".to_owned())?;
        self.rest = tail;
        Ok(b)
    }

    fn bytes(&mut self, len: usize) -> Result<&'a [u8], String> {
        if len > self.rest.len() {
            return Err("bundle ended unexpectedly".to_owned());
        }
        let (head, tail) = self.rest.split_at(len);
        self.rest = tail;
        Ok(head)
    }

    fn bool(&mut self) -> Result<bool, String> {
        match self.byte()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(format!("invalid bundle bool tag {other}")),
        }
    }

    fn u32(&mut self) -> Result<u32, String> {
        u32::try_from(read_leb(&mut self.rest)?).map_err(|_| "bundle u32 overflow".to_owned())
    }

    fn i32(&mut self) -> Result<i32, String> {
        let u = self.u32()?;
        Ok(((u >> 1) as i32) ^ -((u & 1) as i32))
    }

    fn i64(&mut self) -> Result<i64, String> {
        let u = read_leb(&mut self.rest)?;
        Ok(((u >> 1) as i64) ^ -((u & 1) as i64))
    }

    fn f32(&mut self) -> Result<f32, String> {
        let b: [u8; 4] = self.bytes(4)?.try_into().expect("4 bytes");
        Ok(f32::from_le_bytes(b))
    }

    fn f64(&mut self) -> Result<f64, String> {
        let b: [u8; 8] = self.bytes(8)?.try_into().expect("8 bytes");
        Ok(f64::from_le_bytes(b))
    }

    fn len(&mut self, most: usize) -> Result<usize, String> {
        let n = self.u32()? as usize;
        if n > most {
            return Err(format!("bundle list length {n} exceeds limit {most}"));
        }
        Ok(n)
    }

    fn str(&mut self) -> Result<String, String> {
        let idx = self.u32()? as usize;
        self.strings
            .get(idx)
            .cloned()
            .ok_or_else(|| format!("bundle string index {idx} out of bounds"))
    }

    fn list<T>(
        &mut self,
        most: usize,
        mut f: impl FnMut(&mut Self) -> Result<T, String>,
    ) -> Result<Vec<T>, String> {
        let n = self.len(most)?;
        let mut out = Vec::with_capacity(n.min(1024));
        for _ in 0..n {
            out.push(f(self)?);
        }
        Ok(out)
    }

    fn opt<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<Option<T>, String> {
        match self.byte()? {
            0 => Ok(None),
            1 => Ok(Some(f(self)?)),
            other => Err(format!("invalid bundle option tag {other}")),
        }
    }

    fn kernel_id(&mut self) -> Result<KernelId, String> {
        let name = self.str()?;
        let version = self.u32()?;
        KernelId::new(&name, version).map_err(|e| e.to_string())
    }

    fn scalar(&mut self) -> Result<Scalar, String> {
        match self.byte()? {
            0 => Ok(Scalar::Float),
            1 => Ok(Scalar::Int),
            2 => Ok(Scalar::Bool),
            3 => Ok(Scalar::Double),
            4 => Ok(Scalar::Long),
            other => Err(format!("invalid scalar tag {other}")),
        }
    }

    fn ty(&mut self) -> Result<Type, String> {
        match self.byte()? {
            0 => Ok(Type::Scalar(self.scalar()?)),
            1 => Ok(Type::Vector(self.scalar()?, self.byte()?)),
            2 => Ok(Type::Matrix(self.scalar()?, self.byte()?)),
            3 => Ok(Type::Struct(self.u32()?)),
            other => Err(format!("invalid type tag {other}")),
        }
    }

    fn shape(&mut self) -> Result<Shape, String> {
        match self.byte()? {
            0 => Ok(Shape::Rows),
            1 => Ok(Shape::Grid),
            2 => Ok(Shape::Volume),
            other => Err(format!("invalid shape tag {other}")),
        }
    }

    fn domain(&mut self) -> Result<Domain, String> {
        match self.byte()? {
            0 => Ok(Domain::Rows(self.u32()?)),
            1 => Ok(Domain::Grid(self.u32()?, self.u32()?)),
            2 => Ok(Domain::Volume(self.u32()?, self.u32()?, self.u32()?)),
            other => Err(format!("invalid domain tag {other}")),
        }
    }

    fn port(&mut self) -> Result<Port, String> {
        let name = self.str()?;
        let element = self.ty()?;
        let shape = self.shape()?;
        let schema: Schema = self.str()?.parse()?;
        Ok(Port {
            name,
            ty: PortType {
                element,
                shape,
                schema,
            },
        })
    }

    fn value(&mut self, depth: usize) -> Result<Value, String> {
        if depth > MAX_NESTING {
            return Err("bundle value nesting exceeds limit".to_owned());
        }
        macro_rules! arr {
            ($n:expr, $zero:expr, $read:ident) => {{
                let mut out = [$zero; $n];
                for slot in &mut out {
                    *slot = self.$read()?;
                }
                out
            }};
        }
        Ok(match self.byte()? {
            0 => Value::Float(self.f32()?),
            1 => Value::Int(self.i32()?),
            2 => Value::Bool(self.bool()?),
            3 => Value::Vec2(arr!(2, 0.0, f32)),
            4 => Value::Vec3(arr!(3, 0.0, f32)),
            5 => Value::Vec4(arr!(4, 0.0, f32)),
            6 => Value::IVec2(arr!(2, 0, i32)),
            7 => Value::IVec3(arr!(3, 0, i32)),
            8 => Value::IVec4(arr!(4, 0, i32)),
            9 => Value::Mat2(arr!(4, 0.0, f32)),
            10 => Value::Mat3(Box::new(arr!(9, 0.0, f32))),
            11 => Value::Mat4(Box::new(arr!(16, 0.0, f32))),
            12 => Value::Double(self.f64()?),
            13 => Value::Long(self.i64()?),
            14 => Value::DVec2(arr!(2, 0.0, f64)),
            15 => Value::DVec3(Box::new(arr!(3, 0.0, f64))),
            16 => Value::DVec4(Box::new(arr!(4, 0.0, f64))),
            17 => Value::LVec2(arr!(2, 0, i64)),
            18 => Value::LVec3(Box::new(arr!(3, 0, i64))),
            19 => Value::DMat2(Box::new(arr!(4, 0.0, f64))),
            20 => Value::DMat3(Box::new(arr!(9, 0.0, f64))),
            21 => Value::DMat4(Box::new(arr!(16, 0.0, f64))),
            22 => Value::Record(self.list(MAX_ITEMS, |r| r.value(depth + 1))?),
            other => return Err(format!("invalid value tag {other}")),
        })
    }

    fn combine(&mut self) -> Result<Combine, String> {
        match self.byte()? {
            0 => Ok(Combine::Sum),
            1 => Ok(Combine::Min),
            2 => Ok(Combine::Max),
            3 => Ok(Combine::Last),
            other => Err(format!("invalid combine tag {other}")),
        }
    }

    fn module(&mut self) -> Result<Module, String> {
        let id = self.kernel_id()?;
        let mode = match self.byte()? {
            0 => Mode::Map,
            1 => Mode::Reduce,
            2 => Mode::Scan,
            3 => Mode::Scatter {
                base: self.u32()?,
                combine: self.combine()?,
            },
            4 => Mode::Prefix {
                combine: self.combine()?,
            },
            5 => Mode::Group { size: self.u32()? },
            other => return Err(format!("invalid mode tag {other}")),
        };
        let mask = self.byte()?;
        let mut requires = Capabilities::none();
        for (bit, cap) in Capability::ALL.into_iter().enumerate() {
            if mask & (1 << bit) != 0 {
                requires.insert(cap);
            }
        }
        let inputs = self.list(MAX_ITEMS, Self::port)?;
        let params = self.list(MAX_ITEMS, |r| {
            Ok(Param {
                name: r.str()?,
                ty: r.ty()?,
                default: r.value(0)?,
            })
        })?;
        let outputs = self.list(MAX_ITEMS, Self::port)?;
        let memories = self.list(MAX_ITEMS, Self::str)?;
        let budget = self.u32()?;
        let structs = self.list(MAX_ITEMS, |r| match r.byte()? {
            0 => Ok(StructDef {
                name: r.str()?,
                fields: r.list(MAX_ITEMS, |r| {
                    Ok(Field {
                        name: r.str()?,
                        ty: r.ty()?,
                    })
                })?,
                element: None,
            }),
            1 => {
                let name = r.str()?;
                let elem = r.ty()?;
                let count = r.u32()?;
                if count == 0 || count > MAX_ARRAY_COUNT {
                    return Err(format!("invalid array length {count}"));
                }
                Ok(StructDef {
                    name,
                    fields: (0..count)
                        .map(|k| Field {
                            name: k.to_string(),
                            ty: elem,
                        })
                        .collect(),
                    element: Some(elem),
                })
            }
            other => Err(format!("invalid struct tag {other}")),
        })?;
        let functions = self.list(MAX_ITEMS, Self::function)?;
        let entry = self.function()?;
        let merge = self.opt(Self::function)?;
        let init = self.opt(Self::function)?;
        let module = Module {
            manifest: Manifest {
                id,
                mode,
                requires,
                inputs,
                params,
                outputs,
                memories,
                budget,
            },
            structs,
            functions,
            entry,
            merge,
            init,
        };
        entelechy_ir::validate(module)
            .map(entelechy_ir::Validated::into_module)
            .map_err(|e| format!("invalid Kernel in bundle: {e}"))
    }

    fn function(&mut self) -> Result<Function, String> {
        let name = self.str()?;
        let args = self.list(MAX_ITEMS, Self::ty)?;
        let mut next = 0u32;
        let body = self.list(MAX_ITEMS, |r| r.instruction(&mut next, 0))?;
        let results = self.list(MAX_ITEMS, |r| Ok(ValueId(r.u32()?)))?;
        Ok(Function {
            name,
            args,
            body,
            results,
            bindings: Vec::new(),
        })
    }

    fn block(&mut self, next: &mut u32, depth: usize) -> Result<Block, String> {
        if depth > MAX_NESTING {
            return Err("bundle block nesting exceeds limit".to_owned());
        }
        let body = self.list(MAX_ITEMS, |r| r.instruction(next, depth + 1))?;
        let result = ValueId(self.u32()?);
        Ok(Block { body, result })
    }

    fn instruction(&mut self, next: &mut u32, depth: usize) -> Result<Instruction, String> {
        let id = ValueId(*next);
        *next = next
            .checked_add(1)
            .ok_or_else(|| "bundle instruction count overflow".to_owned())?;
        let ty = self.ty()?;
        let vid = |r: &mut Self| Ok::<ValueId, String>(ValueId(r.u32()?));
        let op = match self.byte()? {
            0 => Op::Const(self.value(0)?),
            1 => Op::Param(self.u32()?),
            2 => Op::Arg(self.u32()?),
            3 => Op::Unary(
                match self.byte()? {
                    0 => UnaryOp::Neg,
                    1 => UnaryOp::Not,
                    t => return Err(format!("invalid unary tag {t}")),
                },
                vid(self)?,
            ),
            4 => {
                let tag = self.byte()?;
                let bin = *BINARIES
                    .get(tag as usize)
                    .ok_or_else(|| format!("invalid binary op tag {tag}"))?;
                Op::Binary(bin, vid(self)?, vid(self)?)
            }
            5 => Op::Construct(self.list(MAX_ITEMS, vid)?),
            6 => Op::Extract(vid(self)?, self.u32()?),
            7 => Op::Subscript(vid(self)?, vid(self)?),
            8 => Op::Call(self.u32()?, self.list(MAX_ITEMS, vid)?),
            9 => {
                let idx = self.byte()? as usize;
                let (intr, _) = *Intrinsic::ALL
                    .get(idx)
                    .ok_or_else(|| format!("invalid intrinsic tag {idx}"))?;
                Op::Intrinsic(intr, self.list(MAX_ITEMS, vid)?)
            }
            10 => Op::If {
                condition: vid(self)?,
                then: self.block(next, depth)?,
                otherwise: self.block(next, depth)?,
            },
            11 => Op::Match {
                scrutinee: vid(self)?,
                arms: self.list(MAX_ITEMS, |r| {
                    Ok((r.list(MAX_ITEMS, Self::i32)?, r.block(next, depth)?))
                })?,
                otherwise: self.block(next, depth)?,
            },
            12 => Op::Loop {
                count: self.u32()?,
                bound: self.opt(vid)?,
                init: vid(self)?,
                until: self.opt(|r| r.block(next, depth))?,
                body: self.block(next, depth)?,
            },
            13 => Op::Each {
                count: self.u32()?,
                body: self.block(next, depth)?,
            },
            14 => Op::Carried,
            15 => Op::Index,
            16 => Op::Coord,
            17 => Op::Domain,
            18 => Op::Time,
            19 => Op::Clock,
            20 => Op::Rate,
            21 => Op::Queued,
            22 => Op::Pointer,
            23 => Op::PointerDown,
            24 => Op::Button(vid(self)?),
            25 => Op::Wheel,
            26 => Op::Pads,
            27 => Op::Stick(vid(self)?),
            28 => Op::Window,
            29 => Op::Scale,
            30 => Op::Key(vid(self)?),
            31 => Op::Typed(vid(self)?),
            32 => Op::Element,
            33 => Op::Lane,
            34 => Op::Exchange(vid(self)?, vid(self)?),
            35 => Op::Sample(self.u32()?),
            36 => Op::At(self.u32()?, vid(self)?),
            37 => Op::Part {
                input: self.u32()?,
                at: self.opt(vid)?,
                path: self.list(MAX_ITEMS, |r| match r.byte()? {
                    0 => Ok(Step::Field(r.u32()?)),
                    1 => Ok(Step::Index(ValueId(r.u32()?))),
                    t => Err(format!("invalid step tag {t}")),
                })?,
            },
            38 => Op::AtLinear(self.u32()?, vid(self)?),
            39 => Op::Count(self.u32()?),
            40 => Op::Size(self.u32()?),
            41 => Op::Put(self.u32()?, vid(self)?, vid(self)?),
            42 => Op::Emit(vid(self)?),
            43 => Op::Spend(vid(self)?),
            44 => Op::Look,
            45 => Op::Lanes {
                body: self.block(next, depth)?,
            },
            46 => Op::Meet(self.opt(vid)?),
            other => return Err(format!("invalid instruction tag {other}")),
        };
        Ok(Instruction { id, ty, op })
    }

    fn source(&mut self) -> Result<Source, String> {
        match self.byte()? {
            0 => Ok(Source::External(self.str()?)),
            1 => Ok(Source::Node {
                node: NodeId::from(self.str()?),
                port: self.str()?,
            }),
            other => Err(format!("invalid source tag {other}")),
        }
    }

    fn domain_expr(&mut self, depth: usize) -> Result<DomainExpr, String> {
        if depth > MAX_NESTING {
            return Err("bundle domain expression nesting exceeds limit".to_owned());
        }
        Ok(match self.byte()? {
            0 => DomainExpr::Int(self.i64()?),
            1 => DomainExpr::Float(self.f64()?),
            2 => DomainExpr::Name(self.str()?),
            3 => DomainExpr::Count(self.str()?),
            4 => DomainExpr::Size(self.str()?),
            5 => DomainExpr::Value(self.str()?, self.u32()?),
            6 => DomainExpr::Context(self.str()?),
            7 => DomainExpr::Vector(self.list(8, |r| r.domain_expr(depth + 1))?),
            8 => DomainExpr::Call(self.str()?, self.list(16, |r| r.domain_expr(depth + 1))?),
            9 => DomainExpr::Neg(Box::new(self.domain_expr(depth + 1)?)),
            10 => DomainExpr::Binary(
                match self.byte()? {
                    0 => DomainOp::Add,
                    1 => DomainOp::Sub,
                    2 => DomainOp::Mul,
                    3 => DomainOp::Div,
                    4 => DomainOp::Shl,
                    t => return Err(format!("invalid domain op tag {t}")),
                },
                Box::new(self.domain_expr(depth + 1)?),
                Box::new(self.domain_expr(depth + 1)?),
            ),
            other => return Err(format!("invalid domain expr tag {other}")),
        })
    }

    fn graph(&mut self) -> Result<Graph, String> {
        let name = self.str()?;
        let externals = self.list(MAX_ITEMS, Self::str)?;
        let texts = self.list(MAX_ITEMS, |r| Ok((r.str()?, r.str()?)))?;
        let params = self.list(MAX_ITEMS, |r| {
            Ok(GraphParam {
                name: r.str()?,
                ty: r.str()?,
                default: r.value(0)?,
            })
        })?;
        let requirements = self.list(MAX_ITEMS, |r| {
            Ok(Requirement {
                name: r.str()?,
                left: r.domain_expr(0)?,
                compare: match r.byte()? {
                    0 => Compare::Less,
                    1 => Compare::AtMost,
                    2 => Compare::Greater,
                    3 => Compare::AtLeast,
                    4 => Compare::Equal,
                    5 => Compare::Unequal,
                    t => return Err(format!("invalid compare tag {t}")),
                },
                right: r.domain_expr(0)?,
                at: None,
            })
        })?;
        let times_params = self.list(MAX_ITEMS, Self::str)?;
        let nodes = self.list(MAX_ITEMS, |r| {
            let id = NodeId::from(r.str()?);
            let kernel = r.kernel_id()?;
            let in_count = r.len(MAX_ITEMS)?;
            let mut inputs = BTreeMap::new();
            for _ in 0..in_count {
                let port = r.str()?;
                let binding = match r.byte()? {
                    0 => Binding::Now(r.source()?),
                    1 => Binding::Previous {
                        source: r.source()?,
                        initial: r.source()?,
                        migration: r.opt(Self::kernel_id)?,
                    },
                    t => return Err(format!("invalid binding tag {t}")),
                };
                inputs.insert(port, binding);
            }
            let param_count = r.len(MAX_ITEMS)?;
            let mut nparams = BTreeMap::new();
            for _ in 0..param_count {
                let pname = r.str()?;
                let pv = match r.byte()? {
                    0 => ParamValue::Literal(r.value(0)?),
                    1 => ParamValue::Graph(r.str()?),
                    t => return Err(format!("invalid param value tag {t}")),
                };
                nparams.insert(pname, pv);
            }
            let domain = r.opt(|r| match r.byte()? {
                0 => Ok(DomainSpec::Literal(r.domain()?)),
                1 => Ok(DomainSpec::Expr(r.domain_expr(0)?)),
                t => Err(format!("invalid domain spec tag {t}")),
            })?;
            let is_async = r.bool()?;
            let awake = r.list(MAX_ITEMS, |r| {
                Ok(Awake {
                    source: r.source()?,
                    row: r.u32()?,
                })
            })?;
            Ok(Node {
                id,
                kernel,
                inputs,
                params: nparams,
                domain,
                is_async,
                awake,
                at: None,
            })
        })?;
        let pairs = |r: &mut Self| r.list(MAX_ITEMS, |r| Ok((r.str()?, r.source()?)));
        let outputs = pairs(self)?;
        let fallbacks = pairs(self)?;
        let placed = self.list(MAX_ITEMS, |r| Ok((r.str()?, pairs(r)?)))?;
        let repeated = self.list(MAX_ITEMS, |r| Ok((r.str()?, NodeId::from(r.str()?))))?;
        Ok(Graph {
            name,
            externals,
            texts,
            params,
            requirements,
            times_params,
            times_rows: Vec::new(),
            nodes,
            outputs,
            fallbacks,
            placed,
            repeated,
        })
    }

    fn field_slice(&mut self) -> Result<(Schema, Domain, &'a [u8]), String> {
        let schema: Schema = self.str()?.parse()?;
        let domain = self.domain()?;
        let expected = domain
            .count()
            .checked_mul(schema.element_bytes())
            .ok_or_else(|| "bundle Field size overflow".to_owned())?;
        let byte_len = self.u32()? as usize;
        if byte_len != expected {
            return Err(format!(
                "bundle Field has {byte_len} byte(s), expected {expected}"
            ));
        }
        let raw = self.bytes(byte_len)?;
        Ok((schema, domain, raw))
    }

    fn field_data(&mut self) -> Result<FieldData, String> {
        let (schema, domain, raw) = self.field_slice()?;
        decode_field_data(schema, domain, raw)
    }
}

fn decode_field_data(schema: Schema, domain: Domain, raw: &[u8]) -> Result<FieldData, String> {
    let mut words = Vec::with_capacity(raw.len().div_ceil(4));
    let (chunks, rem) = raw.as_chunks::<4>();
    words.extend(chunks.iter().map(|&b| u32::from_le_bytes(b)));
    if !rem.is_empty() {
        let mut tail = [0u8; 4];
        tail[..rem.len()].copy_from_slice(rem);
        words.push(u32::from_le_bytes(tail));
    }
    FieldData::from_storage(schema, domain, words).map_err(|e| e.to_string())
}
