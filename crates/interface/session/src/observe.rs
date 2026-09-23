//! What a caller observes of a session without changing what it
//! computes: an output saved as a file, a pixel, an element's field, an
//! element traced through its Kernel, a picture exported.

use std::path::Path;

use orla_formats::ppm;
use orla_runtime::{Context, Handle, KernelId};

use crate::{Session, SessionError};

/// A vector's component by name, as a Kernel spells it (`x` or `r` and
/// the rest): which word it is, and what type it holds.
fn component_of(ty: orla_ir::Type, name: &str) -> Option<(usize, orla_ir::Type)> {
    use orla_ir::Type;
    let (lanes, element) = match ty {
        Type::Vec2 => (2, Type::Float),
        Type::Vec3 => (3, Type::Float),
        Type::Vec4 => (4, Type::Float),
        Type::IVec2 => (2, Type::Int),
        Type::IVec3 => (3, Type::Int),
        _ => return None,
    };
    let at = ["xyzw", "rgba"]
        .iter()
        .find_map(|alphabet| alphabet.find(name))?;
    (name.len() == 1 && at < lanes).then_some((at, element))
}

/// `node.port[index].field.field` as a replay writes it and the
/// protocol takes it: the node, the port, the element, and the path
/// through the struct to a value.
pub fn parse_target(text: &str) -> Result<(String, String, u32, Vec<String>), String> {
    let (head, rest) = text
        .split_once('[')
        .ok_or_else(|| "name an element as `node.port[index]`".to_string())?;
    let (index, tail) = rest
        .split_once(']')
        .ok_or_else(|| "name an element as `node.port[index]`".to_string())?;
    let (node, port) = head
        .rsplit_once('.')
        .ok_or_else(|| "name `node.port`".to_string())?;
    let index = index
        .parse::<u32>()
        .map_err(|_| format!("element index `{index}` is not a whole number"))?;
    let path: Vec<String> = tail
        .split('.')
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect();
    Ok((node.to_owned(), port.to_owned(), index, path))
}

impl Session {
    /// What a scene output of this name is wired to, when the name is
    /// an output rather than a node.
    fn output_source(
        &self,
        node: &orla_graph::NodeId,
        port: &str,
    ) -> Option<(orla_graph::NodeId, String)> {
        let graph = self.scene.graph()?;
        if graph.nodes.iter().any(|n| n.id == *node) {
            return None;
        }
        let (name, _) = graph
            .outputs
            .iter()
            .find(|(name, _)| name == &node.0 && port == "result")?;
        match &graph.outputs.iter().find(|(o, _)| o == name)?.1 {
            orla_graph::Source::Node { node, port } => Some((node.clone(), port.clone())),
            orla_graph::Source::External(_) => None,
        }
    }

    /// Why a node's output is not there to be saved: the name may be a
    /// placed graph, which is its nodes by the time the graph runs, or
    /// a node of one written without its repetition, or no node at all.
    fn why_not_saved(&self, node: &orla_graph::NodeId, port: &str) -> String {
        let Some(graph) = self.scene.graph() else {
            return format!("`{node}.{port}` has no value; no scene is loaded");
        };
        let named: Vec<&str> = graph
            .nodes
            .iter()
            .map(|n| n.id.0.as_str())
            .filter(|id| {
                id.starts_with(&format!("{node}."))
                    || id.starts_with(&format!("{node}/"))
                    || id.split('/').next() == Some(node.0.as_str())
            })
            .collect();
        if !graph.nodes.iter().any(|n| n.id == *node) {
            if let Some(last) = named.last() {
                return format!(
                    "`{node}` is not a node: it is placed as {} of them, the last `{last}`, which is what a `--save` names (or give the scene an `output`)",
                    named.len()
                );
            }
            return format!("`{node}` is not a node of this scene");
        }
        format!("`{node}.{port}` has no value; pin it and evaluate")
    }

    /// Writes a node's output as a file: `.field` for any Field, `.png` or
    /// `.ppm` for a grid of pixels (a `Grid<Float>` as gray), `.pgm` for a
    /// `Grid<Float>`, `.wav` at 48 kHz for rows of one or two floats. The
    /// node must have been evaluated and kept materialized (pinned).
    pub fn save(
        &mut self,
        node: &orla_graph::NodeId,
        port: &str,
        path: &Path,
    ) -> Result<(), String> {
        // A scene's own output name stands for what it is wired to, so
        // `--save settled=...` works with the names the scene reads by.
        let (node, port) = match self.output_source(node, port) {
            Some((node, port)) => (node, port),
            None => (node.clone(), port.to_owned()),
        };
        let (node, port) = (&node, port.as_str());
        let handle = match self.cached(node, port) {
            Some(handle) => handle,
            None => return Err(self.why_not_saved(node, port)),
        };
        self.runtime
            .materialize(handle)
            .map_err(|e| e.to_string())?;
        let host = self.runtime.host();
        let data = host.data(handle).map_err(|e| e.to_string())?;
        let bytes =
            orla_formats::encode(data, &orla_formats::extension(path), None).map_err(|why| {
                match why {
                    orla_formats::EncodeError::Unknown => why.to_string(),
                    orla_formats::EncodeError::Unfit(what) => {
                        format!("`{node}.{port}` is {what}; save it as .field")
                    }
                }
            })?;
        std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// How many elements a node's output holds: what a node whose
    /// `domain` a Kernel decided came out as.
    pub fn element_count(&self, node: &orla_graph::NodeId, port: &str) -> Result<usize, String> {
        let handle = self
            .cached(node, port)
            .ok_or_else(|| "the node has no materialized output; is it wired?".to_string())?;
        self.runtime()
            .host()
            .domain(handle)
            .map(|domain| domain.count())
            .map_err(|e| e.to_string())
    }

    /// Runs one element of `node`'s Kernel on the interpreter and returns
    /// every binding of it: element `at` of a map, step `at[0]` of a scan.
    /// The scene is evaluated first under `context`, with the node's
    /// sources pinned so their Fields exist, and the node's invocation is
    /// what an evaluation would give it now; nothing is committed by the
    /// trace itself. Pins the trace added are removed again.
    pub fn trace(
        &mut self,
        node: &orla_graph::NodeId,
        at: [u32; 3],
        context: &Context,
    ) -> Result<(orla_cpu::Trace, KernelId), String> {
        let graph = self.scene.graph().ok_or("no scene is loaded")?;
        let target = graph
            .nodes
            .iter()
            .find(|n| &n.id == node)
            .ok_or_else(|| format!("no node named `{node}`"))?
            .clone();
        let version = self
            .runtime
            .kernel(&target.kernel)
            .ok_or_else(|| format!("`{}` is not published", target.kernel))?;
        if version.module().is_none() {
            return Err(format!(
                "`{}` is a native operation; there is nothing to trace",
                target.kernel
            ));
        }
        // The sources the node reads must have Fields: pin them for this
        // evaluation, then let fusion have them back.
        let mut pinned = Vec::new();
        for binding in target.inputs.values() {
            let sources = match binding {
                orla_graph::Binding::Now(source) => vec![source],
                orla_graph::Binding::Previous {
                    source, initial, ..
                } => vec![source, initial],
            };
            for source in sources {
                if let orla_graph::Source::Node { node, .. } = source
                    && !self.evaluator.pins().contains(node)
                {
                    self.evaluator.pin(node.clone());
                    pinned.push(node.clone());
                }
            }
        }
        let traced = (|| {
            self.evaluate(context).map_err(|error| error.to_string())?;
            // The graph is read while the runtime is written to (a
            // `value(port)` domain brings a Field to the Host), so the
            // scene is taken out for the call and put back.
            let graph = self.scene.take_graph().ok_or("no scene is loaded")?;
            let invocation = self
                .evaluator
                .invocation(
                    &graph,
                    &target,
                    &version,
                    &mut self.runtime,
                    self.inputs.handles(),
                    context,
                    self.scene.params(),
                )
                .map_err(|error| error.to_string());
            self.scene.restore(graph);
            let invocation = invocation?;
            for handle in &invocation.inputs {
                self.runtime
                    .materialize(*handle)
                    .map_err(|error| error.to_string())?;
            }
            // A trace walks the Kernel as written: the same source
            // without the optimizer, so its functions are still called
            // and its `let`s are all there. The running module answers
            // for a Kernel no document owns.
            let written = self.kernels.unoptimized(&target.kernel);
            let module = written
                .as_ref()
                .unwrap_or_else(|| version.module().expect("checked above"));
            // It runs with its work accounted, as a backend runs it.
            let module = orla_ir::account(module);
            let granted = module.manifest.requires.clone();
            let mut scope = orla_runtime::Scope::new(self.runtime.host_mut(), granted);
            orla_cpu::trace(&module, &invocation, &mut scope, at).map_err(|error| error.to_string())
        })();
        for node in pinned {
            self.evaluator.unpin(&node);
        }
        traced.map(|trace| (trace, target.kernel.clone()))
    }

    /// One element field of a node's output: the value's type and its
    /// words. `node.port[index].field.field` is how a replay and the
    /// protocol both name it; [`parse_target`] reads that spelling.
    pub fn element_field(
        &mut self,
        node: &orla_graph::NodeId,
        port: &str,
        index: u32,
        path: &[String],
    ) -> Result<(orla_ir::Type, Vec<u32>), String> {
        use orla_ir::Type;
        let kernel = self
            .graph()
            .and_then(|graph| graph.nodes.iter().find(|n| &n.id == node))
            .map(|n| n.kernel.clone())
            .ok_or_else(|| format!("no node `{node}` in the scene"))?;
        let version = self
            .runtime()
            .kernel(&kernel)
            .ok_or_else(|| format!("`{kernel}` is not published"))?;
        let output = version
            .manifest()
            .outputs
            .iter()
            .find(|o| o.name == port)
            .ok_or_else(|| format!("`{kernel}` has no output `{port}`"))?;
        let structs: Vec<orla_ir::StructDef> = version
            .module()
            .map(|m| m.structs.clone())
            .unwrap_or_default();
        // Follow the field path through the struct table to a word
        // offset and a type.
        let mut ty = output.element();
        let mut offset = 0usize;
        let words_of = |ty: Type| orla_ir::schema_of(ty, &structs).map_or(1, |s| s.value_words());
        for field in path {
            // A struct's field by name, or a vector's component by the
            // letter the language spells it with.
            if let Some((component, element)) = component_of(ty, field) {
                offset += component;
                ty = element;
                continue;
            }
            let Type::Struct(id) = ty else {
                return Err(format!(
                    "`{field}` names a field or a component, but the value is a {ty}"
                ));
            };
            let def = &structs[id as usize];
            let mut found = None;
            for f in &def.fields {
                if &f.name == field {
                    found = Some(f.ty);
                    break;
                }
                offset += words_of(f.ty);
            }
            ty = found.ok_or_else(|| format!("struct `{}` has no field `{field}`", def.name))?;
        }
        let handle = self
            .cached(node, port)
            .ok_or_else(|| "the node has no materialized output; is it wired?".to_string())?;
        self.runtime_mut()
            .materialize(handle)
            .map_err(|e| e.to_string())?;
        let element = self
            .runtime()
            .host()
            .element(handle, index)
            .map_err(|e| e.to_string())?;
        Ok((ty, element[offset..offset + words_of(ty)].to_vec()))
    }

    /// Writes a cached image to a PPM file, materializing it if it was
    /// resident on the device.
    pub fn export(&mut self, image: Handle, path: &Path) -> Result<(), SessionError> {
        let image = self
            .runtime
            .image(image)
            .map_err(|e| SessionError::Observe(e.to_string()))?;
        ppm::write(path, &image).map_err(|e| SessionError::Image(path.to_path_buf(), e))
    }

    /// Reads one pixel of a cached image, materializing it if needed.
    pub fn read_pixel(
        &mut self,
        image: Handle,
        x: u32,
        y: u32,
    ) -> Result<orla_runtime::Pixel, SessionError> {
        // Four words out of the Field, not a copy of the picture: a
        // replay checks a handful of pixels a frame, and the picture it
        // checks them in is thirty-three megabytes at 1920x1080. The
        // Field comes to the Host once; every pixel after that is an
        // index.
        self.runtime
            .materialize(image)
            .map_err(|e| SessionError::Observe(e.to_string()))?;
        let host = self.runtime.host();
        let domain = host.domain(image).map_err(SessionError::Host)?;
        if !host.schema(image).map_err(SessionError::Host)?.is_pixel() {
            return Err(SessionError::Observe(
                "the output is not a picture".to_owned(),
            ));
        }
        let (width, height) = (domain.width(), domain.height() * domain.depth());
        if x >= width || y >= height {
            return Err(SessionError::Host(orla_runtime::HostError::OutOfBounds {
                x,
                y,
                width,
                height,
            }));
        }
        let words = host
            .element(image, y * width + x)
            .map_err(SessionError::Host)?;
        Ok([
            f32::from_bits(words[0]),
            f32::from_bits(words[1]),
            f32::from_bits(words[2]),
            f32::from_bits(words[3]),
        ])
    }
}
