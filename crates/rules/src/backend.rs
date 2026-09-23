//! The contract every backend implements: the interpreter, the JIT and
//! the GPU each implement [`Backend`] in their own crate, and the
//! Kernel Runtime holds whichever it is given. Nothing here depends on
//! the runtime, so no backend does either.

use std::any::Any;

use orla_abi::{Handle, Invocation, Scope, Shape};
use orla_ir::{Capabilities, Capability, KernelId, Manifest, Port, PortType, Type, Validated};

use crate::ExecutionError;

/// An operation a backend performs natively: device access, not a Kernel.
/// Native operations are published by the runtime itself, under
/// generation zero, and cannot be replaced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NativeOp {
    /// Rasterizes `vertices` and `triangles` into a `surface` grid; see
    /// [`crate::raster`] for the contract.
    Rasterize,
    /// The same, testing each fragment against the alpha of a `mask`: a
    /// fragment below `cutout` is not drawn at all, which is what a
    /// leaf's edge or a fence's holes need.
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

    /// The contract as text, for inspection where a Kernel would show its
    /// source.
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

/// An execution backend. `prepare` does all expensive work so that
/// publishing is a pointer swap.
pub trait Backend: Send + Sync {
    /// Short backend name for diagnostics, e.g. `cpu`.
    fn name(&self) -> &'static str;

    /// For interfaces that know a specific backend, such as a window that
    /// presents a resident image directly.
    fn as_any(&self) -> &dyn Any;

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

    /// Executes a native operation. The default is for backends that
    /// have none.
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

    /// What the device took over the last execution, by its own clock,
    /// when the backend measures it; the default is for backends whose
    /// work is done when `execute` returns, which the caller times.
    fn last_device_time(&self) -> Option<std::time::Duration> {
        None
    }

    /// What measuring one execution costs when the execution is
    /// nothing: the submission and the wait a measured run puts around
    /// every node, which is in every number it prints and belongs to
    /// none of them. Zero for a backend whose work is done when
    /// `execute` returns, since then the caller's clock is the whole of
    /// it.
    fn measurement_floor(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }

    /// Runs a map `repeat` times, feeding input `chain.0` from output
    /// `chain.1` each time: what a node placed several times over its
    /// own output is. A backend that can do the whole run without
    /// making a Field between the times answers with what the last time
    /// gave; the default answers `None` and the caller runs the times
    /// one by one.
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

    /// Waits for everything this backend has submitted to finish. The
    /// default is for backends whose work is done when `execute`
    /// returns; a device queue's is not, so anything that measures a
    /// frame, or hands the machine to something else, has to ask.
    fn settle_device(&self) {}

    /// Whether this backend's work is finished by a device of its own,
    /// so that `execute` returning is not the work being done and
    /// [`Backend::settle_device`] is what waits for it.
    fn has_device(&self) -> bool {
        false
    }

    /// Downloads the bytes of a resource this backend left resident:
    /// `byte_len` bytes of little-endian words. The default is for backends
    /// that never do.
    fn download(
        &self,
        device: &(dyn Any + Send + Sync),
        byte_len: usize,
    ) -> Result<Vec<u8>, String> {
        let _ = (device, byte_len);
        Err("this backend keeps no resident resources".into())
    }
}
