//! A group's entry rewritten for a backend that runs one lane at a time.
//! Every lane reaches each exchange alike ([`crate::group_fault`]), so the
//! group's control round its exchanges is the group's own: the branches,
//! matches, loops and eaches holding an exchange run once for the group,
//! and everything between them runs once for each lane in turn
//! ([`Op::Lanes`]). A value that crosses from one stretch of lanes to
//! another is the array of the lanes' values; an exchange ends a stretch
//! ([`Op::Meet`]) and takes, for each lane, the element its lane names;
//! the group's own branch, count or test is lane 0's, which every lane
//! holds alike. Each lane's work is counted as many times as before, and
//! the group's every output is the array of its lanes' elements.

use std::collections::{HashMap, HashSet};

use crate::validate::runs_lanes;
use crate::{
    BinaryOp, Block, Field, Function, Instruction, Mode, Module, Op, Step, StructDef, Type,
    Validated, ValidationError, Value, ValueId, type_name, validate,
};

/// The group's entry with its lanes run one stretch at a time; the module
/// unchanged when it is not a group or already runs its lanes. The result
/// is validated again: a group whose arrays of lanes pass what a value may
/// hold is refused.
pub fn lanes(module: &Validated) -> Result<Validated, ValidationError> {
    let Mode::Group { size } = module.manifest.mode else {
        return Ok(module.clone());
    };
    if runs_lanes(&module.entry.body) {
        return Ok(module.clone());
    }
    let mut lower = Lower {
        structs: module.structs.clone(),
        arrays: HashMap::new(),
        size,
        next: 0,
        lanes: HashMap::new(),
        loops: HashMap::new(),
        carried: Vec::new(),
        carried_lane: None,
    };
    let entry = &module.entry;
    let mut body = lower.group(&entry.body, &entry.results);
    // The last stretch ends with the entry.
    let meet = lower.fresh();
    body.push(Instruction {
        id: meet,
        ty: Type::Bool,
        op: Op::Meet(None),
    });
    let results = entry
        .results
        .iter()
        .map(|result| lower.lanes[&result.0].0)
        .collect();
    let lowered = Module {
        manifest: module.manifest.clone(),
        structs: lower.structs,
        functions: module.functions.clone(),
        entry: Function {
            name: entry.name.clone(),
            args: entry.args.clone(),
            body,
            results,
            bindings: Vec::new(),
        },
        merge: None,
        init: None,
    };
    validate(lowered)
}

/// Whether an instruction is the group's: an exchange, or a block
/// holding one.
fn meets(instruction: &Instruction) -> bool {
    matches!(instruction.op, Op::Exchange(..))
        || instruction
            .op
            .blocks()
            .iter()
            .any(|block| block.body.iter().any(meets))
}

/// The values an operation reads, its branch's, match's and loop's own
/// included; not the loop a spend counts, which is no value it reads.
fn reads(op: &Op) -> Vec<ValueId> {
    match op {
        Op::If { condition, .. } => vec![*condition],
        Op::Match { scrutinee, .. } => vec![*scrutinee],
        Op::Loop { bound, init, .. } => bound.iter().copied().chain([*init]).collect(),
        Op::Spend(_) => Vec::new(),
        other => crate::optimize::operands(other),
    }
}

/// Every value `instructions` read, at any depth, the results of their
/// blocks too.
fn all_reads(instructions: &[Instruction], out: &mut HashSet<u32>) {
    for instruction in instructions {
        out.extend(reads(&instruction.op).iter().map(|id| id.0));
        for block in instruction.op.blocks() {
            out.insert(block.result.0);
            all_reads(&block.body, out);
        }
    }
}

/// Every value `instructions` define, at any depth.
fn all_defined(instructions: &[Instruction], out: &mut HashSet<u32>) {
    for instruction in instructions {
        out.insert(instruction.id.0);
        for block in instruction.op.blocks() {
            all_defined(&block.body, out);
        }
    }
}

struct Lower {
    structs: Vec<StructDef>,
    /// The array types made, by element and count.
    arrays: HashMap<(Type, u32), Type>,
    size: u32,
    /// The next value of the rewritten entry, in the order it is written.
    next: u32,
    /// Each value the group holds, by its id in the given entry: the array
    /// of its lanes' values, and the element.
    lanes: HashMap<u32, (ValueId, Type)>,
    /// Each loop of the group's own, by its id in the given entry.
    loops: HashMap<u32, ValueId>,
    /// The carried value of each loop of the group's own around, read
    /// once where its block begins: the array of its lanes'.
    carried: Vec<(ValueId, Type)>,
    /// In a stretch, the running lane's element of the innermost of those.
    carried_lane: Option<ValueId>,
}

/// Whether instructions read the carried value of the loop round them,
/// not of one of their own.
fn reads_carried(instructions: &[Instruction]) -> bool {
    instructions
        .iter()
        .any(|instruction| match &instruction.op {
            Op::Carried => true,
            Op::Loop { .. } => false,
            other => other
                .blocks()
                .iter()
                .any(|block| reads_carried(&block.body)),
        })
}

impl Lower {
    fn fresh(&mut self) -> ValueId {
        let id = ValueId(self.next);
        self.next += 1;
        id
    }

    /// The array of `count` elements of `element`, made once.
    fn array(&mut self, element: Type, count: u32) -> Type {
        if let Some(ty) = self.arrays.get(&(element, count)) {
            return *ty;
        }
        let name = type_name(element, &self.structs);
        self.structs.push(StructDef::array(element, &name, count));
        let ty = Type::Struct(self.structs.len() as u32 - 1);
        self.arrays.insert((element, count), ty);
        ty
    }

    /// The array of the group's lanes of `element`.
    fn lanes_of(&mut self, element: Type) -> Type {
        self.array(element, self.size)
    }

    /// The element of the array of lanes `ty`.
    fn element_of(&self, ty: Type) -> Type {
        match ty {
            Type::Struct(index) => self.structs[index as usize]
                .element
                .expect("lanes are an array"),
            _ => unreachable!("lanes are an array"),
        }
    }

    /// The group's loop's carried value, `ty` the array of its lanes, read
    /// where a block of the loop begins, for its stretches to take.
    fn read_carried(&mut self, ty: Type) -> Instruction {
        let id = self.fresh();
        self.carried.push((id, ty));
        Instruction {
            id,
            ty,
            op: Op::Carried,
        }
    }

    /// Lane 0's value of what the group holds as `given`, which every lane
    /// holds alike where the group turns on it.
    fn first(&mut self, given: ValueId, out: &mut Vec<Instruction>) -> ValueId {
        let (array, element) = self.lanes[&given.0];
        let id = self.fresh();
        out.push(Instruction {
            id,
            ty: element,
            op: Op::Extract(array, 0),
        });
        id
    }

    /// A block of the group's: its stretches of lanes between the
    /// instructions holding exchanges, after which every value of `keep`
    /// is held as the array of its lanes.
    fn group(&mut self, instructions: &[Instruction], keep: &[ValueId]) -> Vec<Instruction> {
        let mut out = Vec::new();
        let mut start = 0;
        for (at, instruction) in instructions.iter().enumerate() {
            if meets(instruction) {
                if start < at {
                    self.stretch(
                        &instructions[start..at],
                        &instructions[at..],
                        keep,
                        &mut out,
                    );
                }
                self.group_instruction(instruction, &mut out);
                start = at + 1;
            }
        }
        if start < instructions.len() {
            self.stretch(&instructions[start..], &[], keep, &mut out);
        }
        out
    }

    /// Instructions with no exchange, run once for each lane: what they
    /// define that `rest` or `keep` reads comes out as the lanes' arrays.
    fn stretch(
        &mut self,
        run: &[Instruction],
        rest: &[Instruction],
        keep: &[ValueId],
        out: &mut Vec<Instruction>,
    ) {
        let mut later: HashSet<u32> = keep.iter().map(|id| id.0).collect();
        all_reads(rest, &mut later);
        let given: Vec<(ValueId, Type)> = run
            .iter()
            .filter(|instruction| later.contains(&instruction.id.0))
            .map(|instruction| (instruction.id, instruction.ty))
            .collect();
        // What the run reads from the group, each lane taking its own.
        let mut defined = HashSet::new();
        all_defined(run, &mut defined);
        let mut outer = HashSet::new();
        all_reads(run, &mut outer);
        let mut outer: Vec<u32> = outer
            .into_iter()
            .filter(|id| !defined.contains(id))
            .collect();
        outer.sort_unstable();

        let lanes_id = self.fresh();
        let mut body = Vec::new();
        let mut local: HashMap<u32, ValueId> = HashMap::new();
        // A spend of the group's loop right before is first, as it follows
        // that loop.
        let mut run = run;
        if let [first, rest @ ..] = run
            && let Op::Spend(counted) = first.op
            && self.loops.contains_key(&counted.0)
        {
            let id = self.fresh();
            body.push(Instruction {
                id,
                ty: first.ty,
                op: Op::Spend(self.loops[&counted.0]),
            });
            local.insert(first.id.0, id);
            run = rest;
        }
        let lane = self.fresh();
        body.push(Instruction {
            id: lane,
            ty: Type::Int,
            op: Op::Lane,
        });
        for id in outer {
            let (array, element) = self.lanes[&id];
            let taken = self.fresh();
            body.push(Instruction {
                id: taken,
                ty: element,
                op: Op::Subscript(array, lane),
            });
            local.insert(id, taken);
        }
        // The group's loop's carried value, read for the group where the
        // loop's block begins, the lane taking its own.
        let around = self.carried_lane.take();
        if reads_carried(run) {
            let (carried, ty) = *self
                .carried
                .last()
                .expect("a carried value is read in a loop");
            let element = self.element_of(ty);
            let taken = self.fresh();
            body.push(Instruction {
                id: taken,
                ty: element,
                op: Op::Subscript(carried, lane),
            });
            self.carried_lane = Some(taken);
        }
        for instruction in run {
            self.copy(instruction, &mut local, &mut body, 0);
        }
        self.carried_lane = around;
        let (result, element) = match given.as_slice() {
            [] => {
                let id = self.fresh();
                body.push(Instruction {
                    id,
                    ty: Type::Bool,
                    op: Op::Const(Value::Bool(true)),
                });
                (id, Type::Bool)
            }
            [(id, ty)] => (local[&id.0], *ty),
            many => {
                let fields = many
                    .iter()
                    .enumerate()
                    .map(|(k, (_, ty))| Field {
                        name: k.to_string(),
                        ty: *ty,
                    })
                    .collect();
                self.structs
                    .push(StructDef::record(format!("lanes{}", lanes_id.0), fields));
                let record = Type::Struct(self.structs.len() as u32 - 1);
                let id = self.fresh();
                body.push(Instruction {
                    id,
                    ty: record,
                    op: Op::Construct(many.iter().map(|(id, _)| local[&id.0]).collect()),
                });
                (id, record)
            }
        };
        let ty = self.lanes_of(element);
        out.push(Instruction {
            id: lanes_id,
            ty,
            op: Op::Lanes {
                body: Block { body, result },
            },
        });
        match given.as_slice() {
            [] => {}
            [(id, element)] => {
                self.lanes.insert(id.0, (lanes_id, *element));
            }
            many => {
                // Each value its own array of lanes, from the records.
                for (field, (id, value_ty)) in many.iter().enumerate() {
                    let each = self.fresh();
                    let lane = self.fresh();
                    let record = self.fresh();
                    let value = self.fresh();
                    let ty = self.lanes_of(*value_ty);
                    out.push(Instruction {
                        id: each,
                        ty,
                        op: Op::Lanes {
                            body: Block {
                                body: vec![
                                    Instruction {
                                        id: lane,
                                        ty: Type::Int,
                                        op: Op::Lane,
                                    },
                                    Instruction {
                                        id: record,
                                        ty: element,
                                        op: Op::Subscript(lanes_id, lane),
                                    },
                                    Instruction {
                                        id: value,
                                        ty: *value_ty,
                                        op: Op::Extract(record, field as u32),
                                    },
                                ],
                                result: value,
                            },
                        },
                    });
                    self.lanes.insert(id.0, (each, *value_ty));
                }
            }
        }
    }

    /// One lane's copy of `instruction`, its values renamed through
    /// `local`; `depth` counts the lane's own loops around it, outside of
    /// which a carried value is the group's loop's, the lane's element of
    /// it.
    fn copy(
        &mut self,
        instruction: &Instruction,
        local: &mut HashMap<u32, ValueId>,
        out: &mut Vec<Instruction>,
        depth: usize,
    ) {
        let map = |local: &HashMap<u32, ValueId>, id: ValueId| local[&id.0];
        if let (Op::Carried, 0) = (&instruction.op, depth) {
            let taken = self
                .carried_lane
                .expect("the stretch takes the carried value");
            local.insert(instruction.id.0, taken);
            return;
        }
        let id = self.fresh();
        local.insert(instruction.id.0, id);
        let op = match &instruction.op {
            Op::If {
                condition,
                then,
                otherwise,
            } => Op::If {
                condition: map(local, *condition),
                then: self.copy_block(then, local, depth),
                otherwise: self.copy_block(otherwise, local, depth),
            },
            Op::Match {
                scrutinee,
                arms,
                otherwise,
            } => Op::Match {
                scrutinee: map(local, *scrutinee),
                arms: arms
                    .iter()
                    .map(|(values, arm)| (values.clone(), self.copy_block(arm, local, depth)))
                    .collect(),
                otherwise: self.copy_block(otherwise, local, depth),
            },
            Op::Loop {
                count,
                bound,
                init,
                until,
                body,
            } => Op::Loop {
                count: *count,
                bound: bound.map(|bound| map(local, bound)),
                init: map(local, *init),
                until: until
                    .as_ref()
                    .map(|until| self.copy_block(until, local, depth + 1)),
                body: self.copy_block(body, local, depth + 1),
            },
            Op::Each { count, body } => Op::Each {
                count: *count,
                body: self.copy_block(body, local, depth),
            },
            Op::Spend(counted) => Op::Spend(
                local
                    .get(&counted.0)
                    .copied()
                    .unwrap_or_else(|| self.loops[&counted.0]),
            ),
            other => renamed(other, |id| map(local, id)),
        };
        out.push(Instruction {
            id,
            ty: instruction.ty,
            op,
        });
    }

    fn copy_block(
        &mut self,
        block: &Block,
        local: &mut HashMap<u32, ValueId>,
        depth: usize,
    ) -> Block {
        let mut body = Vec::new();
        for instruction in &block.body {
            self.copy(instruction, local, &mut body, depth);
        }
        Block {
            body,
            result: local[&block.result.0],
        }
    }

    /// A block of the group's own control: its lanes' result as an array.
    fn group_block(&mut self, block: &Block) -> Block {
        let body = self.group(&block.body, &[block.result]);
        Block {
            body,
            result: self.lanes[&block.result.0].0,
        }
    }

    /// An instruction holding an exchange, run once for the group.
    fn group_instruction(&mut self, instruction: &Instruction, out: &mut Vec<Instruction>) {
        let element = instruction.ty;
        let ty = self.lanes_of(element);
        match &instruction.op {
            Op::Exchange(value, lane) => {
                // The stretch ends, the lanes giving `values`; each lane
                // takes the element its lane names, wrapped round the group.
                let (values, _) = self.lanes[&value.0];
                let meet = self.fresh();
                out.push(Instruction {
                    id: meet,
                    ty: Type::Bool,
                    op: Op::Meet(Some(values)),
                });
                let (named, _) = self.lanes[&lane.0];
                let id = self.fresh();
                let ids: Vec<ValueId> = (0..7).map(|_| self.fresh()).collect();
                let [own, asked, size, low, up, wrapped, taken] = ids[..] else {
                    unreachable!()
                };
                let body = vec![
                    Instruction {
                        id: own,
                        ty: Type::Int,
                        op: Op::Lane,
                    },
                    Instruction {
                        id: asked,
                        ty: Type::Int,
                        op: Op::Subscript(named, own),
                    },
                    Instruction {
                        id: size,
                        ty: Type::Int,
                        op: Op::Const(Value::Int(self.size as i32)),
                    },
                    Instruction {
                        id: low,
                        ty: Type::Int,
                        op: Op::Binary(BinaryOp::Rem, asked, size),
                    },
                    Instruction {
                        id: up,
                        ty: Type::Int,
                        op: Op::Binary(BinaryOp::Add, low, size),
                    },
                    Instruction {
                        id: wrapped,
                        ty: Type::Int,
                        op: Op::Binary(BinaryOp::Rem, up, size),
                    },
                    Instruction {
                        id: taken,
                        ty: element,
                        op: Op::Subscript(values, wrapped),
                    },
                ];
                out.push(Instruction {
                    id,
                    ty,
                    op: Op::Lanes {
                        body: Block {
                            body,
                            result: taken,
                        },
                    },
                });
                self.lanes.insert(instruction.id.0, (id, element));
            }
            Op::If {
                condition,
                then,
                otherwise,
            } => {
                let condition = self.first(*condition, out);
                let id = self.fresh();
                let then = self.group_block(then);
                let otherwise = self.group_block(otherwise);
                out.push(Instruction {
                    id,
                    ty,
                    op: Op::If {
                        condition,
                        then,
                        otherwise,
                    },
                });
                self.lanes.insert(instruction.id.0, (id, element));
            }
            Op::Match {
                scrutinee,
                arms,
                otherwise,
            } => {
                let scrutinee = self.first(*scrutinee, out);
                let id = self.fresh();
                let arms = arms
                    .iter()
                    .map(|(values, arm)| (values.clone(), self.group_block(arm)))
                    .collect();
                let otherwise = self.group_block(otherwise);
                out.push(Instruction {
                    id,
                    ty,
                    op: Op::Match {
                        scrutinee,
                        arms,
                        otherwise,
                    },
                });
                self.lanes.insert(instruction.id.0, (id, element));
            }
            Op::Loop {
                count,
                bound,
                init,
                until,
                body,
            } => {
                let bound = bound.map(|bound| self.first(bound, out));
                let init = self.lanes[&init.0].0;
                let id = self.fresh();
                self.loops.insert(instruction.id.0, id);
                let until = until.as_ref().map(|until| {
                    let mut body = vec![self.read_carried(ty)];
                    body.extend(self.group(&until.body, &[until.result]));
                    self.carried.pop();
                    let result = self.first(until.result, &mut body);
                    Block { body, result }
                });
                let mut inner = vec![self.read_carried(ty)];
                inner.extend(self.group(&body.body, &[body.result]));
                self.carried.pop();
                let body = Block {
                    body: inner,
                    result: self.lanes[&body.result.0].0,
                };
                out.push(Instruction {
                    id,
                    ty,
                    op: Op::Loop {
                        count: *count,
                        bound,
                        init,
                        until,
                        body,
                    },
                });
                self.lanes.insert(instruction.id.0, (id, element));
            }
            Op::Each { count, body } => {
                // Built for the group, element by element, each element
                // the lanes'; then each lane takes its elements.
                let lane_element = match element {
                    Type::Struct(index) => self.structs[index as usize]
                        .element
                        .expect("an each builds an array"),
                    _ => unreachable!("an each builds an array"),
                };
                let lanes_element = self.lanes_of(lane_element);
                let built_ty = self.array(lanes_element, *count);
                let built = self.fresh();
                let body = self.group_block(body);
                out.push(Instruction {
                    id: built,
                    ty: built_ty,
                    op: Op::Each {
                        count: *count,
                        body,
                    },
                });
                let id = self.fresh();
                let ids: Vec<ValueId> = (0..5).map(|_| self.fresh()).collect();
                let [own, each, at, row, taken] = ids[..] else {
                    unreachable!()
                };
                out.push(Instruction {
                    id,
                    ty,
                    op: Op::Lanes {
                        body: Block {
                            body: vec![
                                Instruction {
                                    id: own,
                                    ty: Type::Int,
                                    op: Op::Lane,
                                },
                                Instruction {
                                    id: each,
                                    ty: element,
                                    op: Op::Each {
                                        count: *count,
                                        body: Block {
                                            body: vec![
                                                Instruction {
                                                    id: at,
                                                    ty: Type::Int,
                                                    op: Op::Index,
                                                },
                                                Instruction {
                                                    id: row,
                                                    ty: lanes_element,
                                                    op: Op::Subscript(built, at),
                                                },
                                                Instruction {
                                                    id: taken,
                                                    ty: lane_element,
                                                    op: Op::Subscript(row, own),
                                                },
                                            ],
                                            result: taken,
                                        },
                                    },
                                },
                            ],
                            result: each,
                        },
                    },
                });
                self.lanes.insert(instruction.id.0, (id, element));
            }
            other => unreachable!("only an exchange or a block holds one: {other:?}"),
        }
    }
}

/// An operation with no block, its values renamed.
fn renamed(op: &Op, map: impl Fn(ValueId) -> ValueId) -> Op {
    match op {
        Op::Unary(op, a) => Op::Unary(*op, map(*a)),
        Op::Binary(op, a, b) => Op::Binary(*op, map(*a), map(*b)),
        Op::Construct(parts) => Op::Construct(parts.iter().map(|p| map(*p)).collect()),
        Op::Extract(source, index) => Op::Extract(map(*source), *index),
        Op::Subscript(array, index) => Op::Subscript(map(*array), map(*index)),
        Op::Call(function, args) => Op::Call(*function, args.iter().map(|a| map(*a)).collect()),
        Op::Intrinsic(intrinsic, args) => {
            Op::Intrinsic(*intrinsic, args.iter().map(|a| map(*a)).collect())
        }
        Op::At(input, at) => Op::At(*input, map(*at)),
        Op::AtLinear(input, at) => Op::AtLinear(*input, map(*at)),
        Op::Key(at) => Op::Key(map(*at)),
        Op::Typed(at) => Op::Typed(map(*at)),
        Op::Button(at) => Op::Button(map(*at)),
        Op::Stick(at) => Op::Stick(map(*at)),
        Op::Part { input, at, path } => Op::Part {
            input: *input,
            at: at.map(&map),
            path: path
                .iter()
                .map(|step| match step {
                    Step::Index(index) => Step::Index(map(*index)),
                    Step::Field(field) => Step::Field(*field),
                })
                .collect(),
        },
        Op::Put(input, address, value) => Op::Put(*input, map(*address), map(*value)),
        Op::Exchange(value, lane) => Op::Exchange(map(*value), map(*lane)),
        Op::Emit(pair) => Op::Emit(map(*pair)),
        Op::Spend(counted) => Op::Spend(map(*counted)),
        Op::Meet(given) => Op::Meet(given.map(&map)),
        Op::If { .. } | Op::Match { .. } | Op::Loop { .. } | Op::Each { .. } | Op::Lanes { .. } => {
            unreachable!("a block is copied with its blocks")
        }
        Op::Const(_)
        | Op::Param(_)
        | Op::Arg(_)
        | Op::Carried
        | Op::Index
        | Op::Coord
        | Op::Domain
        | Op::Time
        | Op::Clock
        | Op::Rate
        | Op::Queued
        | Op::Pointer
        | Op::PointerDown
        | Op::Wheel
        | Op::Pads
        | Op::Window
        | Op::Scale
        | Op::Element
        | Op::Lane
        | Op::Sample(_)
        | Op::Count(_)
        | Op::Size(_)
        | Op::Look => op.clone(),
    }
}
