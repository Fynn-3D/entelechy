//! Where a run goes by its size, as the runtime holds it, on IR written
//! here and two backends that run only it: a device and the processor
//! beside it. A run a little too large for the processor goes there when
//! the small runs that read it would otherwise stay on the device beside
//! it, and stays there from then on; with too few such readers it stays
//! on the device, its readers with it.

use std::any::Any;

use entelechy_ir::{Function, Instruction, Mode, Module, Op, Port, PortType, Type, ValueId};
use entelechy_kernel::{
    Backend, Capability, Context, Domain, ExecutionError, Handle, Invocation, Manifest, Prepared,
    Runtime, Scope, Validated, Value,
};

/// Kernel `id`, reading `inputs` images and giving each element of its
/// one image `vec4(1, 0, 0, 1)`; its parameter `which` says nothing.
fn constant(id: &str, inputs: usize) -> Validated {
    let module = Module {
        manifest: Manifest {
            id: id.parse().unwrap(),
            mode: Mode::Map,
            requires: [Capability::FieldRead, Capability::FieldWrite]
                .into_iter()
                .collect(),
            inputs: (0..inputs)
                .map(|k| Port {
                    name: format!("in{k}"),
                    ty: PortType::image(),
                })
                .collect(),
            params: vec![entelechy_ir::Param {
                name: "which".into(),
                ty: Type::Int,
                default: Value::Int(0),
            }],
            outputs: vec![Port {
                name: "out".into(),
                ty: PortType::image(),
            }],
            memories: vec![],
            budget: entelechy_ir::DEFAULT_BUDGET,
        },
        structs: vec![],
        functions: vec![],
        entry: Function {
            name: "compute".into(),
            args: vec![],
            body: vec![Instruction {
                id: ValueId(0),
                ty: Type::Vec4,
                op: Op::Const(Value::Vec4([1.0, 0.0, 0.0, 1.0])),
            }],
            results: vec![ValueId(0)],
            bindings: vec![],
        },
        merge: None,
        init: None,
    };
    entelechy_ir::validate(module).unwrap()
}

/// What the stub prepares a module to: the value its entry returns.
struct Given([f32; 4]);

impl Prepared for Given {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A backend for the Kernels written here, named `name`, a device when
/// `device`: each element of its output is the value the entry returns.
struct Stub {
    name: &'static str,
    device: bool,
}

impl Backend for Stub {
    fn name(&self) -> &'static str {
        self.name
    }

    fn has_device(&self) -> bool {
        self.device
    }

    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        match &module.entry.body[..] {
            [
                Instruction {
                    op: Op::Const(Value::Vec4(value)),
                    ..
                },
            ] => Ok(Box::new(Given(*value))),
            _ => Err("the stub runs a constant alone".into()),
        }
    }

    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        let Given(value) = prepared.as_any().downcast_ref::<Given>().unwrap();
        let words: Vec<u32> = value.iter().map(|v| v.to_bits()).collect();
        let port = &module.manifest.outputs[0];
        Ok(vec![scope.adopt_storage(
            port.ty.schema.clone(),
            invocation.domain,
            words.repeat(invocation.domain.count()),
        )?])
    }
}

fn invocation(inputs: Vec<Handle>, elements: u32, which: i32) -> Invocation {
    Invocation {
        inputs,
        params: vec![Value::Int(which)],
        domain: Domain::Grid(elements, 1),
        context: Context::default(),
    }
}

/// A device and the processor beside it, the processor taking runs of
/// at most 100 elements of the Kernels written here.
fn runtime() -> Runtime {
    let mut runtime = Runtime::with_backends(
        1 << 16,
        Box::new(Stub {
            name: "device",
            device: true,
        }),
        Some(Box::new(Stub {
            name: "processor",
            device: false,
        })),
    );
    let typical = constant("test.reader@1", 1).typical_work().max(1);
    runtime.set_small(100 * typical);
    runtime
}

/// Evaluations of a producer of 200 elements, twice what the processor
/// takes, and `readers` runs of 80 elements reading it, each small enough
/// alone: where the producer and the readers went, by evaluation.
fn placed(readers: usize, evaluations: usize) -> Vec<(&'static str, Vec<&'static str>)> {
    let mut runtime = runtime();
    let producer = runtime.load(constant("test.producer@1", 0), "").unwrap();
    let reader = runtime.load(constant("test.reader@1", 1), "").unwrap();
    let mut placed = Vec::new();
    for _ in 0..evaluations {
        runtime.begin_evaluation();
        let made = runtime
            .execute(&producer, &invocation(vec![], 200, 0))
            .unwrap()[0];
        let made_on = runtime.last_placement().unwrap().backend;
        let mut read_on = Vec::new();
        for _ in 0..readers {
            let read = runtime
                .execute(&reader, &invocation(vec![made], 80, 0))
                .unwrap()[0];
            read_on.push(runtime.last_placement().unwrap().backend);
            runtime.host_mut().release(read).unwrap();
        }
        runtime.host_mut().release(made).unwrap();
        placed.push((made_on, read_on));
    }
    placed
}

#[test]
fn a_run_its_small_readers_spare_goes_to_the_processor_with_them() {
    // Six readers spare it 20 elements' work each, more than the 100 it
    // is over by: the first evaluation finds them, the rest hold.
    let placed = placed(6, 4);
    assert_eq!(placed[0], ("device", vec!["device"; 6]));
    for later in &placed[1..] {
        assert_eq!(*later, ("processor", vec!["processor"; 6]));
    }
}

#[test]
fn a_run_too_few_readers_spare_stays_on_the_device_with_them() {
    let placed = placed(2, 4);
    for each in &placed {
        assert_eq!(*each, ("device", vec!["device"; 2]));
    }
}

#[test]
fn two_runs_of_one_kernel_apart_by_a_parameter_are_spared_apart() {
    // Of two producers alike but for a parameter, the one six readers
    // spare goes to the processor; the one nothing reads stays.
    let mut runtime = runtime();
    let producer = runtime.load(constant("test.producer@1", 0), "").unwrap();
    let reader = runtime.load(constant("test.reader@1", 1), "").unwrap();
    let mut placed = Vec::new();
    for _ in 0..3 {
        runtime.begin_evaluation();
        let read = runtime
            .execute(&producer, &invocation(vec![], 200, 0))
            .unwrap()[0];
        let read_on = runtime.last_placement().unwrap().backend;
        let alone = runtime
            .execute(&producer, &invocation(vec![], 200, 1))
            .unwrap()[0];
        let alone_on = runtime.last_placement().unwrap().backend;
        for _ in 0..6 {
            let out = runtime
                .execute(&reader, &invocation(vec![read], 80, 0))
                .unwrap()[0];
            runtime.host_mut().release(out).unwrap();
        }
        runtime.host_mut().release(read).unwrap();
        runtime.host_mut().release(alone).unwrap();
        placed.push((read_on, alone_on));
    }
    assert_eq!(placed[0], ("device", "device"));
    assert_eq!(placed[1..], [("processor", "device"); 2]);
}
