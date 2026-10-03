//! The contract every backend implements. The interpreter, the JIT and
//! the GPU each implement [`Backend`] in their own crate; the Kernel
//! Runtime holds whichever it is given. This crate is independent of
//! the runtime.

use std::any::Any;

use orla_abi::{Handle, Invocation, Scope, Shape};
use orla_ir::{Capabilities, Capability, KernelId, Manifest, Port, PortType, Type, Validated};

use crate::ExecutionError;

/// An operation a backend performs natively: device access, not a Kernel.
/// The runtime publishes these under generation zero; they are fixed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NativeOp {
    /// Rasterizes `vertices` and `triangles` into a `surface` grid; see
    /// [`crate::raster`] for the contract.
    Rasterize,
    /// The same, dropping each fragment whose `mask` alpha is below
    /// `cutout`.
    RasterizeCutout,
}

impl NativeOp {
    /// Every native operation.
    pub const ALL: [NativeOp; 2] = [NativeOp::Rasterize, NativeOp::RasterizeCutout];

    /// The id a graph binds to.
    pub fn id(self) -> KernelId {
        match self {
            Self::Rasterize => KernelId::new("native.rasterize", 1).expect("valid id"),
            Self::RasterizeCutout => KernelId::new("native.rasterize", 2).expect("valid id"),
        }
    }

    /// The structs the manifest's ports name, by index: a vertex, a
    /// triangle, a surface texel and, for the cutout, a mask pixel.
    pub fn structs(self) -> Vec<orla_ir::StructDef> {
        let def = |name: &str, fields: &[(&str, Type)]| orla_ir::StructDef {
            name: name.to_owned(),
            fields: fields
                .iter()
                .map(|(name, ty)| orla_ir::Field {
                    name: (*name).to_owned(),
                    ty: *ty,
                })
                .collect(),
            element: None,
        };
        vec![
            def(
                "Vertex",
                &[
                    ("clip", Type::Vec4),
                    ("normal", Type::Vec3),
                    ("uv", Type::Vec2),
                ],
            ),
            def(
                "Triangle",
                &[("a", Type::Int), ("b", Type::Int), ("c", Type::Int)],
            ),
            def(
                "Surface",
                &[
                    ("depth", Type::Float),
                    ("normal", Type::Vec3),
                    ("uv", Type::Vec2),
                    ("id", Type::Int),
                ],
            ),
            def(
                "Pixel",
                &[
                    ("r", Type::Float),
                    ("g", Type::Float),
                    ("b", Type::Float),
                    ("a", Type::Float),
                ],
            ),
        ]
    }

    /// The manifest: ports by layout and shape, capabilities as a Kernel
    /// would declare them. Element types are anonymous structs.
    pub fn manifest(self) -> Manifest {
        if self == Self::RasterizeCutout {
            let mut manifest = Self::Rasterize.manifest();
            manifest.id = self.id();
            manifest.inputs.push(Port {
                name: "mask".into(),
                ty: PortType {
                    element: Type::Struct(3),
                    shape: Shape::Grid,
                    schema: crate::raster::mask_schema(),
                },
            });
            manifest.params.push(orla_ir::Param {
                name: "cutout".into(),
                ty: Type::Float,
                default: orla_abi::Value::Float(0.5),
            });
            return manifest;
        }
        match self {
            Self::RasterizeCutout => unreachable!("answered above"),
            Self::Rasterize => Manifest {
                id: self.id(),
                mode: orla_ir::Mode::Map,
                memories: Vec::new(),
                budget: orla_ir::DEFAULT_BUDGET,
                requires: [Capability::FieldRead, Capability::FieldWrite]
                    .into_iter()
                    .collect::<Capabilities>(),
                inputs: vec![
                    Port {
                        name: "vertices".into(),
                        ty: PortType {
                            element: Type::Struct(0),
                            shape: Shape::Rows,
                            schema: crate::raster::vertex_schema(),
                        },
                    },
                    Port {
                        name: "triangles".into(),
                        ty: PortType {
                            element: Type::Struct(1),
                            shape: Shape::Rows,
                            schema: crate::raster::triangle_schema(),
                        },
                    },
                ],
                params: vec![],
                outputs: vec![Port {
                    name: "surface".into(),
                    ty: PortType {
                        element: Type::Struct(2),
                        shape: Shape::Grid,
                        schema: crate::raster::surface_schema(),
                    },
                }],
            },
        }
    }

    /// The contract as text, shown where a Kernel shows its source.
    pub fn source(self) -> &'static str {
        match self {
            Self::RasterizeCutout => {
                "\
// native.rasterize@2: the backend's rasterizer, with a cutout.
//   input vertices: Rows<{ clip: Vec4; normal: Vec3; uv: Vec2; }>
//   input triangles: Rows<{ a: Int; b: Int; c: Int; }>
//   input mask: Image
//   param cutout: Float = 0.5
//   output surface: Grid<{ depth: Float; normal: Vec3; uv: Vec2; id: Int; }>
// As `native.rasterize@1`, except that a fragment whose alpha in `mask`,
// read at its own uv as `at_linear(mask, uv * size(mask))` reads it, is
// below `cutout` is not drawn: it writes no surface and settles no
// depth, so what is behind a leaf's edge is seen through it.
"
            }
            Self::Rasterize => {
                "\
// native.rasterize@1: the backend's rasterizer.
//   input vertices: Rows<{ clip: Vec4; normal: Vec3; uv: Vec2; }>
//   input triangles: Rows<{ a: Int; b: Int; c: Int; }>
//   output surface: Grid<{ depth: Float; normal: Vec3; uv: Vec2; id: Int; }>
// `clip` is a clip-space position (x, y in [-1, 1] after the divide by w,
// z in [0, 1]); the nearest fragment at each pixel centre is written with
// perspective-correct attributes and its triangle's row as `id`; empty
// pixels hold depth 1, zeros, and id -1. No culling, no clipping: a
// triangle with a vertex at w <= 0, or a clip component that is not
// finite, is skipped whole rather than clipped, and one whose three
// corners are the same vertex is skipped too — which is how a row of a
// fixed length draws a varying number of triangles.
"
            }
        }
    }
}

/// Backend-owned state built from a module at preparation time.
pub trait Prepared: Any + Send + Sync {
    /// For downcasting by the backend that built it.
    fn as_any(&self) -> &dyn Any;
}

/// A deferred verdict on a run already handed back. Waiting on it waits
/// for the run.
pub trait Check: Send {
    /// The run's first failure, as the reference names it, if any.
    fn wait(self: Box<Self>) -> Result<(), ExecutionError>;
}

/// A run's outputs and its deferred check, if any.
pub type Checked = (Vec<Handle>, Option<Box<dyn Check>>);

/// An execution backend. `prepare` does all expensive work so that
/// publishing is a pointer swap.
pub trait Backend: Send + Sync {
    /// Short backend name for diagnostics, e.g. `cpu`.
    fn name(&self) -> &'static str;

    /// Why this backend leaves `module` to another (one it cannot run,
    /// or runs badly), or `None` when it takes it. A runtime with a
    /// second backend prepares the module there and reports the reason
    /// beside the node; otherwise it runs here and [`Backend::prepare`]
    /// reports what it cannot do.
    fn declines(&self, module: &Validated) -> Option<String> {
        let _ = module;
        None
    }

    /// Bytes this backend has uploaded from Host memory to its device
    /// since it was made. Zero for a backend that computes in Host memory.
    fn uploaded(&self) -> u64 {
        0
    }

    /// Builds backend state for a validated module.
    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String>;

    /// Executes a prepared module over the invocation's domain, returning one
    /// image handle per manifest output, owned by the scope's Host. Outputs
    /// may be left resident on the backend's device.
    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError>;

    /// Executes as [`Backend::execute`] does, adding the check for a
    /// failure the backend learns of later (a device). A caller that
    /// waits for the check before committing matches the reference's
    /// failures exactly. The default has no check.
    fn execute_checked(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Checked, ExecutionError> {
        self.execute(prepared, module, invocation, scope)
            .map(|outputs| (outputs, None))
    }

    /// Executes a native operation. The default fails.
    fn execute_native(
        &self,
        op: NativeOp,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        let _ = (invocation, scope);
        Err(ExecutionError::Device(format!(
            "backend `{}` does not implement {}",
            self.name(),
            op.id()
        )))
    }

    /// The last execution's time by the device's own clock, when the
    /// backend measures it.
    fn last_device_time(&self) -> Option<std::time::Duration> {
        None
    }

    /// The overhead a measured run adds to every node (submission and
    /// wait). Zero for a backend whose work is done when `execute`
    /// returns.
    fn measurement_floor(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }

    /// Makes every execution wait for the device before it returns, so
    /// a profile measures the work. It stalls per node; it is for
    /// measuring.
    fn set_measured(&self, on: bool) {
        let _ = on;
    }

    /// Whether a measured run's times come from the device's own clock.
    fn measures_the_device(&self) -> bool {
        false
    }

    /// Runs a map `repeat` times, feeding input `chain.0` from output
    /// `chain.1` each time, and answers with the last outputs. `None`
    /// (the default) leaves the caller to run the times one by one.
    fn execute_repeated(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
        repeat: u32,
        chain: (usize, usize),
    ) -> Option<Result<Vec<Handle>, ExecutionError>> {
        let _ = (prepared, module, invocation, scope, repeat, chain);
        None
    }

    /// Waits for everything this backend has submitted to finish.
    fn settle_device(&self) {}

    /// Whether a device of its own finishes this backend's work, so
    /// [`Backend::settle_device`] is what waits for it.
    fn has_device(&self) -> bool {
        false
    }

    /// Downloads `byte_len` bytes of little-endian words of a resource
    /// this backend left resident.
    fn download(
        &self,
        device: &(dyn Any + Send + Sync),
        byte_len: usize,
    ) -> Result<Vec<u8>, String> {
        let _ = (device, byte_len);
        Err("this backend keeps no resident resources".into())
    }

    /// Whether two resources this backend left resident hold the same
    /// first `byte_len` bytes. The default downloads both.
    fn same_resident(
        &self,
        a: &(dyn Any + Send + Sync),
        b: &(dyn Any + Send + Sync),
        byte_len: usize,
    ) -> Result<bool, String> {
        Ok(self.download(a, byte_len)? == self.download(b, byte_len)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every struct a native port names is defined by `structs`, laid
    /// out as the port is.
    #[test]
    fn every_struct_a_native_port_names_is_defined_as_it_is_laid_out() {
        for op in NativeOp::ALL {
            let structs = op.structs();
            let manifest = op.manifest();
            for port in manifest.inputs.iter().chain(&manifest.outputs) {
                let schema = orla_ir::schema_of(port.ty.element, &structs)
                    .unwrap_or_else(|| panic!("{} names an undefined struct", port.name));
                assert_eq!(schema, port.ty.schema, "{:?} port {}", op, port.name);
            }
        }
    }
}
