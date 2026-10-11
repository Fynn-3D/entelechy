//! Semantic analysis and lowering from AST to IR.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr;

use entelechy_ir::{
    Block as IrBlock, Capability, Field, Function, Instruction, Intrinsic, KernelId,
    MAX_ARRAY_COUNT, Manifest, Mode, Module, Op, Port, PortType, Shape, StructDef, Type, ValueId,
};
use entelechy_language::{Block, Expr, ExprKind, FnDecl, Kernel, Span, Spanned};

use crate::{CompileError, Variants};

mod builtins;

/// Every builtin name a call resolves to, by the tables that resolve it:
/// the Field, memory, context and construction builtins, the ones composed
/// from other operations, the formulas, and the IR's intrinsics; what
/// help's list is held to.
#[cfg(test)]
pub(crate) fn callable() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = declare::RESERVED_BUILTINS
        .iter()
        .chain(builtins::COMPOSED.iter())
        .chain(builtins::YIELDING.iter())
        .chain(builtins::FORMULAS.iter())
        .copied()
        .chain(entelechy_ir::Intrinsic::ALL.iter().map(|(_, name)| *name))
        .collect();
    names.sort_unstable();
    names.dedup();
    names
}
mod composite;
mod declare;
mod expr;
mod matrix;
mod written;

use declare::{RESERVED_BUILTINS, aliases, manifest, port_type, structs};
pub(crate) use declare::{literal, literal_exact, literal_in, literal_of_type, with_variants};

/// A name with its article: "an Int", "a Float", "a u8".
fn a(name: &str) -> String {
    let article = if name.starts_with(['A', 'E', 'I', 'O', 'a', 'e', 'i', 'o']) {
        "an"
    } else {
        "a"
    };
    format!("{article} {name}")
}

fn semantic(span: Span, message: impl Into<String>) -> CompileError {
    CompileError::Semantic {
        span,
        message: message.into(),
        declaration: None,
    }
}

fn parse_type(name: &Spanned<String>) -> Result<Type, CompileError> {
    Type::from_str(&name.value).map_err(|error| {
        let known: Vec<&str> = Type::NAMED.iter().map(|(_, name)| *name).collect();
        semantic(
            name.span,
            format!(
                "{error}{}",
                entelechy_language::did_you_mean(&name.value, known)
            ),
        )
    })
}

/// The struct table while a Kernel lowers: declared structs, then every
/// array type met, `[T; N]` interned once by element and count.
/// The struct table while a Kernel lowers, with its names indexed: a
/// struct is looked up by name in constant time. The table only grows,
/// so the index catches up with what was appended when it is asked.
#[derive(Default)]
struct Structs {
    table: RefCell<Vec<StructDef>>,
    index: RefCell<std::collections::HashMap<String, u32>>,
    indexed: std::cell::Cell<usize>,
    /// The declared enums, each with its variants in order and the `Int`
    /// each is.
    enums: RefCell<Vec<(String, Variants)>>,
    /// Each enum's variants by name, for a variant found without a walk.
    numbers: RefCell<std::collections::HashMap<String, std::collections::HashMap<String, i32>>>,
}

impl std::ops::Deref for Structs {
    type Target = RefCell<Vec<StructDef>>;

    fn deref(&self) -> &Self::Target {
        &self.table
    }
}

impl Structs {
    /// The table, the index let go.
    fn into_inner(self) -> Vec<StructDef> {
        self.table.into_inner()
    }

    /// Declares the enum `name` with its variants in order.
    fn declare_enum(&self, name: String, variants: Variants) {
        self.numbers.borrow_mut().insert(
            name.clone(),
            variants
                .iter()
                .map(|(variant, number)| (variant.clone(), *number))
                .collect(),
        );
        self.enums.borrow_mut().push((name, variants));
    }

    /// Whether an enum named `name` is declared.
    fn is_enum(&self, name: &str) -> bool {
        self.numbers.borrow().contains_key(name)
    }

    /// The variants of the enum named `name`, when one is declared.
    fn variants(&self, name: &str) -> Option<Variants> {
        self.enums
            .borrow()
            .iter()
            .find(|(declared, _)| declared == name)
            .map(|(_, variants)| variants.clone())
    }

    /// The struct named `name`, the first of the name.
    fn find(&self, name: &str) -> Option<u32> {
        let table = self.table.borrow();
        let mut index = self.index.borrow_mut();
        for at in self.indexed.get()..table.len() {
            index.entry(table[at].name.clone()).or_insert(at as u32);
        }
        self.indexed.set(table.len());
        index.get(name).copied()
    }
}

/// The `Int` an enum's variant is.
fn variant_of(
    structs: &Structs,
    name: &str,
    variant: &str,
    span: Span,
) -> Result<i32, CompileError> {
    let found = structs
        .numbers
        .borrow()
        .get(name)
        .map(|numbers| numbers.get(variant).copied());
    match found {
        None => Err(semantic(span, format!("`{name}` is not an enum"))),
        Some(Some(number)) => Ok(number),
        // A variant it lacks: the error names the nearest.
        Some(None) => variant_in(
            &structs.variants(name).unwrap_or_default(),
            name,
            variant,
            span,
        ),
    }
}

/// The `Int` the variant `variant` of the enum `name`, whose variants are
/// `variants`, is.
fn variant_in(
    variants: &[(String, i32)],
    name: &str,
    variant: &str,
    span: Span,
) -> Result<i32, CompileError> {
    variants
        .iter()
        .find(|(declared, _)| declared == variant)
        .map(|(_, number)| *number)
        .ok_or_else(|| {
            semantic(
                span,
                format!(
                    "`{name}` has no variant `{variant}`{}",
                    entelechy_language::did_you_mean(
                        variant,
                        variants.iter().map(|(declared, _)| declared.as_str())
                    )
                ),
            )
        })
}

/// Resolves a type name: a plain type, a struct, or `[Type; N]`
/// (interned).
fn resolve_type(name: &Spanned<String>, structs: &Structs) -> Result<Type, CompileError> {
    if let Some(inner) = name.value.strip_prefix('[') {
        let Some((element, count)) = inner
            .strip_suffix(']')
            .and_then(|inner| inner.rsplit_once(';'))
        else {
            return Err(semantic(
                name.span,
                format!(
                    "`{}` is not a type; an array is `[Type; count]`",
                    name.value
                ),
            ));
        };
        let element = resolve_type(
            &Spanned {
                value: element.trim().to_owned(),
                span: name.span,
            },
            structs,
        )?;
        let count: u32 = count.trim().parse().map_err(|_| {
            semantic(
                name.span,
                format!(
                    "`{}` is not a type; an array is `[Type; count]`",
                    name.value
                ),
            )
        })?;
        return array_type(structs, element, count, name.span);
    }
    // An enum's value is the `Int` of its variant's position.
    if structs.is_enum(&name.value) {
        return Ok(Type::Int);
    }
    if let Some(index) = structs.find(&name.value) {
        return Ok(Type::Struct(index));
    }
    parse_type(name)
}

/// The most values a function's `[each ...]` builders may unroll: one
/// body copy per element, multiplied through nested builders.
pub(crate) const MAX_UNROLLED_VALUES: usize = 1 << 18;

/// Whether an array may have `count` elements.
pub(crate) fn check_array_count(count: u32, span: Span) -> Result<(), CompileError> {
    if count == 0 || count > MAX_ARRAY_COUNT {
        return Err(semantic(
            span,
            format!("an array has 1 to {MAX_ARRAY_COUNT} elements, not {count}"),
        ));
    }
    Ok(())
}

/// The array type `[element; count]`, interned in the struct table.
fn array_type(
    structs: &Structs,
    element: Type,
    count: u32,
    span: Span,
) -> Result<Type, CompileError> {
    check_array_count(count, span)?;
    let existing = structs
        .borrow()
        .iter()
        .position(|def| def.element == Some(element) && def.fields.len() == count as usize);
    if let Some(index) = existing {
        return Ok(Type::Struct(index as u32));
    }
    let name = describe(element, &structs.borrow());
    let def = StructDef::array(element, &name, count);
    fits(&structs.borrow(), &def, span)?;
    let mut table = structs.borrow_mut();
    table.push(def);
    Ok(Type::Struct((table.len() - 1) as u32))
}

/// Whether a value of the struct `def`, over the types in `table`, is
/// within the bounds every value is.
pub(crate) fn fits(table: &[StructDef], def: &StructDef, span: Span) -> Result<(), CompileError> {
    let measured = entelechy_ir::measure_struct(def, &|ty| entelechy_ir::measure(table, ty));
    match entelechy_ir::too_large(measured) {
        Some(why) => Err(semantic(
            span,
            format!("`{}` is too large: {why}", def.name),
        )),
        None => Ok(()),
    }
}

/// What `[index]` takes from a type, and how many: an array's element, a
/// matrix's column, a vector's component.
fn subscript_element(ty: Type, structs: &[StructDef]) -> Option<(Type, usize)> {
    match ty {
        Type::Struct(index) => {
            let def = &structs[index as usize];
            def.element.map(|element| (element, def.fields.len()))
        }
        _ => ty
            .column()
            .map(|column| (column, ty.order().unwrap_or(0) as usize))
            .or_else(|| ty.lanes().map(|(n, lane)| (lane, n as usize))),
    }
}

/// Human-readable type for diagnostics, naming structs.
fn describe(ty: Type, structs: &[StructDef]) -> String {
    entelechy_ir::type_name(ty, structs)
}

/// Registers the record of a map's several outputs: a struct named
/// `{ color, depth }` whose fields are their elements in declaration
/// order, laid out as every backend writes an element.
fn outputs_record(manifest: &Manifest, structs: &Structs) -> Option<u32> {
    if manifest.outputs.len() < 2 || manifest.mode != Mode::Map {
        return None;
    }
    let names: Vec<&str> = manifest
        .outputs
        .iter()
        .map(|port| port.name.as_str())
        .collect();
    let fields = manifest
        .outputs
        .iter()
        .map(|port| Field {
            name: port.name.clone(),
            ty: port.element(),
        })
        .collect();
    let mut table = structs.borrow_mut();
    table.push(StructDef::record(
        format!("{{ {} }}", names.join(", ")),
        fields,
    ));
    Some((table.len() - 1) as u32)
}

/// Lowers `kernel`. Each function lowers where it is first called, so a
/// Kernel lowers only the functions it reaches; `checked` names functions
/// lowered whether called or not, which is how a document's own functions
/// are checked once.
pub(crate) fn lower(
    kernel: &Kernel,
    id: Option<&KernelId>,
    checked: &[String],
) -> Result<Module, CompileError> {
    lower_placed(kernel, id, checked).map(|(module, _)| module)
}

/// Each function `checked` names lowered on its own, as [`lower`] checks
/// them: every fault found, each once, rather than the first.
pub(crate) fn check_each(kernel: &Kernel, checked: &[String]) -> Vec<CompileError> {
    let setup = (|| {
        let structs = structs(kernel)?;
        let aliases = aliases(kernel, &structs)?;
        let manifest = manifest(kernel, None, &structs, &aliases)?;
        Ok::<_, CompileError>((structs, aliases, manifest))
    })();
    let (structs, aliases, manifest) = match setup {
        Ok(setup) => setup,
        Err(error) => return vec![error],
    };
    let record = outputs_record(&manifest, &structs);
    let shared = match shared_of(kernel, &manifest, &structs, &aliases, record) {
        Ok(shared) => shared,
        Err(error) => return vec![error],
    };
    let checking = Builder::new(&shared, false, "", Vec::new());
    let mut faults: Vec<CompileError> = Vec::new();
    for name in checked {
        if let Err(error) = checking.function_index(name, Span::default())
            && !faults.iter().any(|seen| {
                seen.to_string() == error.to_string() && seen.declaration() == error.declaration()
            })
        {
            faults.push(error);
        }
    }
    faults
}

/// [`lower`], and where each value of it is written.
pub(crate) fn lower_placed(
    kernel: &Kernel,
    id: Option<&KernelId>,
    checked: &[String],
) -> Result<(Module, crate::SourcePlaces), CompileError> {
    let structs = structs(kernel)?;
    let aliases = aliases(kernel, &structs)?;
    let manifest = manifest(kernel, id, &structs, &aliases)?;
    let record = outputs_record(&manifest, &structs);
    let shared = shared_of(kernel, &manifest, &structs, &aliases, record)?;
    {
        let checking = Builder::new(&shared, false, "", Vec::new());
        for name in checked {
            checking.function_index(name, Span::default())?;
        }
    }
    let expected = match manifest.outputs.first() {
        Some(port) => port.element(),
        None => {
            return Err(semantic(
                kernel.id.as_ref().map(|id| id.span).unwrap_or_default(),
                "kernel declares no output; add `output result: Image;`",
            ));
        }
    };
    let mut places = crate::SourcePlaces::default();
    let entry = {
        let mut builder = Builder::new(&shared, true, "compute", Vec::new());
        if let Some(carry) = &kernel.carry {
            builder.locals.push((
                carry.name.value.clone(),
                Local::Carried {
                    depth: 0,
                    ty: expected,
                    pair: None,
                },
            ));
        }
        let mut result = builder.block(&kernel.compute)?;
        if shared.record.is_none() && !matches!(manifest.mode, Mode::Scatter { .. }) {
            result = builder.adapt(result, expected);
        }
        let mut split: Option<Vec<ValueId>> = None;
        let actual = builder.ty(result);
        if let Mode::Scatter { .. } = manifest.mode {
            result = builder.scatter_pair(result, kernel.compute.result.span)?;
        } else if let Some(record) = shared.record {
            // Several outputs: the body gives their record, and the entry
            // gives each output its field, in declaration order.
            if actual != Type::Struct(record) {
                let named: Vec<String> = manifest
                    .outputs
                    .iter()
                    .map(|port| format!("{}: ..", port.name))
                    .collect();
                return Err(semantic(
                    kernel.compute.result.span,
                    format!(
                        "a Kernel with several outputs gives them by name, `{{ {} }}`, but the body gives {}",
                        named.join(", "),
                        describe(actual, &structs.borrow())
                    ),
                ));
            }
            let results = manifest
                .outputs
                .iter()
                .enumerate()
                .map(|(position, port)| {
                    builder.emit(port.element(), Op::Extract(result, position as u32))
                })
                .collect();
            split = Some(results);
        } else if actual != expected {
            return Err(semantic(
                kernel.compute.result.span,
                format!(
                    "output `{}` needs {} per element, but the body gives {}",
                    manifest.outputs[0].name,
                    a(&describe(expected, &structs.borrow())),
                    describe(actual, &structs.borrow())
                ),
            ));
        }
        places.entry = std::mem::take(&mut builder.places);
        builder.finish(split.unwrap_or_else(|| vec![result]))?
    };
    let merge = match &kernel.merge {
        None => None,
        Some(decl) => {
            let mut builder = Builder::new(&shared, false, "merge", Vec::new());
            if decl.a.value == decl.b.value {
                return Err(semantic(
                    decl.b.span,
                    format!("`merge` names both values `{}`", decl.a.value),
                ));
            }
            builder.args.push((decl.a.value.clone(), expected));
            builder.args.push((decl.b.value.clone(), expected));
            let result = builder.block(&decl.body)?;
            let result = builder.adapt(result, expected);
            let actual = builder.ty(result);
            if actual != expected {
                return Err(semantic(
                    decl.body.result.span,
                    format!(
                        "`merge` must give {} like `compute`, but the body gives {}",
                        a(&describe(expected, &structs.borrow())),
                        describe(actual, &structs.borrow())
                    ),
                ));
            }
            places.others.push(std::mem::take(&mut builder.places));
            Some(builder.finish(vec![result])?)
        }
    };
    let init = match &kernel.carry {
        None => None,
        Some(carry) => {
            let mut builder = Builder::new(&shared, false, "init", Vec::new());
            let result = builder.value(&carry.init)?;
            let result = builder.adapt(result, expected);
            let actual = builder.ty(result);
            if actual != expected {
                return Err(semantic(
                    carry.init.span,
                    format!(
                        "`scan` carries {} like its output, but the initial value is {}",
                        a(&describe(expected, &structs.borrow())),
                        describe(actual, &structs.borrow())
                    ),
                ));
            }
            places.others.push(std::mem::take(&mut builder.places));
            Some(builder.finish(vec![result])?)
        }
    };
    let Shared {
        functions,
        places: function_places,
        ..
    } = shared;
    places.functions = function_places.into_inner();
    let functions: Vec<Function> = functions.into_inner().into_iter().map(|(_, f)| f).collect();
    Ok((
        Module {
            manifest,
            structs: structs.into_inner(),
            functions,
            entry,
            merge,
            init,
        },
        places,
    ))
}

/// What every block of `kernel` resolves names against: its functions
/// registered, a function with a Field or type parameter as a template.
fn shared_of<'a>(
    kernel: &'a Kernel,
    manifest: &'a Manifest,
    structs: &'a Structs,
    aliases: &'a [(String, PortType)],
    record: Option<u32>,
) -> Result<Shared<'a>, CompileError> {
    // A function with a Field parameter is a template, instantiated per
    // port binding; the others lower where first called. A function may
    // call any other, whatever their order, so long as no chain of calls
    // reaches back to it.
    let mut shared = Shared {
        manifest,
        structs,
        aliases,
        functions: RefCell::new(Vec::new()),
        templates: std::collections::HashMap::new(),
        plain: std::collections::HashMap::new(),
        calls: std::cell::Cell::new(0),
        places: RefCell::new(Vec::new()),
        last_places: RefCell::new(crate::Places::new()),
        lowered: RefCell::new(BTreeMap::new()),
        instances: RefCell::new(BTreeMap::new()),
        lowering: RefCell::new(Vec::new()),
        needs: RefCell::new(BTreeMap::new()),
        record,
    };
    let mut seen = BTreeSet::new();
    for (position, declaration) in kernel.functions.iter().enumerate() {
        if RESERVED_BUILTINS.contains(&declaration.name.value.as_str()) {
            return Err(semantic(
                declaration.name.span,
                format!(
                    "`{}` is a builtin of the language; a function cannot take its name (a math builtin's, like `step`, it can, and stands for it here)",
                    declaration.name.value
                ),
            ));
        }
        if !seen.insert(declaration.name.value.as_str()) {
            return Err(semantic(
                declaration.name.span,
                format!("function `{}` is defined twice", declaration.name.value),
            ));
        }
        let mut has_field = false;
        for (_, ty) in &declaration.args {
            if declaration.generics.iter().any(|g| g.value == ty.value) {
                continue;
            }
            if resolve_type(ty, structs).is_err() {
                port_type(ty, structs, aliases).map_err(|_| {
                    semantic(ty.span, format!("`{}` is not a type", ty.value))
                        .in_declaration(&declaration.name.value)
                })?;
                has_field = true;
            }
        }
        if has_field || !declaration.generics.is_empty() {
            shared
                .templates
                .insert(declaration.name.value.as_str(), (position, declaration));
        } else {
            shared
                .plain
                .insert(declaration.name.value.as_str(), (position, declaration));
        }
    }
    Ok(shared)
}

/// What every block of a Kernel resolves names against.
struct Shared<'a> {
    manifest: &'a Manifest,
    structs: &'a Structs,
    aliases: &'a [(String, PortType)],
    functions: RefCell<Vec<(String, Function)>>,
    /// Functions with a Field or type parameter, by their position among
    /// the Kernel's functions.
    templates: std::collections::HashMap<&'a str, (usize, &'a FnDecl)>,
    /// The other functions by name, with their position; each lowers
    /// where first called.
    plain: std::collections::HashMap<&'a str, (usize, &'a FnDecl)>,
    /// How deep the function being lowered is called, at most
    /// [`entelechy_language::MAX_NESTING`], so a chain of calls cannot exhaust
    /// the compiling thread's stack.
    calls: std::cell::Cell<usize>,
    /// Each function's places, by its index in `functions`.
    places: RefCell<Vec<crate::Places>>,
    /// The places of the function lowered last, until it is added.
    last_places: RefCell<crate::Places>,
    /// Each plain function lowered so far, by position, to its index in
    /// `functions`.
    lowered: RefCell<BTreeMap<usize, u32>>,
    /// Instantiated templates by binding, to function index.
    instances: RefCell<BTreeMap<Instance, u32>>,
    /// The functions being lowered, the outermost first: a call to one of
    /// them is a chain of calls back to it, which a Kernel, having no
    /// stack, cannot make.
    lowering: RefCell<Vec<String>>,
    /// What each function needs, by its index in `functions`: each
    /// capability with the builtin that needs it, checked where a
    /// Kernel's own code calls the function.
    needs: RefCell<BTreeMap<u32, Vec<(Capability, String)>>>,
    /// The record of a map's several outputs, which `{ name: .., ... }`
    /// builds and `compute` gives.
    record: Option<u32>,
}

/// Lowers a function body. `fields` binds a template's Field parameters
/// to input ports; `name` names the result.
fn lower_function(
    shared: &Shared<'_>,
    declaration: &FnDecl,
    fields: Vec<(String, u32)>,
    generics: Vec<(String, Type)>,
    name: &str,
) -> Result<(Function, Vec<(Capability, String)>), CompileError> {
    // A function lowers where first called, inside its caller's lowering.
    let depth = shared.calls.get() + 1;
    if depth > entelechy_language::MAX_NESTING {
        return Err(semantic(
            declaration.name.span,
            format!(
                "calls nest more than {} deep to reach `{}`; a chain of calls that long is a loop",
                entelechy_language::MAX_NESTING,
                declaration.name.value
            ),
        ));
    }
    shared.calls.set(depth);
    shared
        .lowering
        .borrow_mut()
        .push(declaration.name.value.clone());
    let lowered = lower_function_body(shared, declaration, fields, generics, name)
        .map_err(|error| error.in_declaration(&declaration.name.value));
    shared.lowering.borrow_mut().pop();
    shared.calls.set(depth - 1);
    lowered
}

fn lower_function_body(
    shared: &Shared<'_>,
    declaration: &FnDecl,
    fields: Vec<(String, u32)>,
    generics: Vec<(String, Type)>,
    name: &str,
) -> Result<(Function, Vec<(Capability, String)>), CompileError> {
    let structs = shared.structs;
    let bound = |ty: &Spanned<String>| -> Option<Type> {
        generics
            .iter()
            .find(|(parameter, _)| parameter == &ty.value)
            .map(|(_, bound)| *bound)
    };
    let mut builder = Builder::new(shared, false, name, fields);
    builder.needs = Some(RefCell::new(Vec::new()));
    for (name, ty) in &declaration.args {
        if builder.fields.iter().any(|(field, _)| field == &name.value) {
            continue;
        }
        let ty = match bound(ty) {
            Some(ty) => ty,
            None => resolve_type(ty, structs)?,
        };
        if builder
            .args
            .iter()
            .any(|(existing, _)| existing == &name.value)
        {
            return Err(semantic(
                name.span,
                format!("argument `{}` is declared twice", name.value),
            ));
        }
        builder.args.push((name.value.clone(), ty));
    }
    let ret = match bound(&declaration.ret) {
        Some(ty) => ty,
        None => {
            if declaration
                .generics
                .iter()
                .any(|parameter| parameter.value == declaration.ret.value)
            {
                return Err(semantic(
                    declaration.ret.span,
                    format!(
                        "`{}` returns `{}`, which no argument binds; a type parameter is bound by an argument's type",
                        declaration.name.value, declaration.ret.value
                    ),
                ));
            }
            resolve_type(&declaration.ret, structs)?
        }
    };
    let result = builder.block(&declaration.body)?;
    let result = builder.adapt(result, ret);
    let actual = builder.ty(result);
    if actual != ret {
        return Err(semantic(
            declaration.body.result.span,
            format!(
                "function `{}` returns {} but the body gives {}",
                declaration.name.value,
                describe(ret, &structs.borrow()),
                describe(actual, &structs.borrow())
            ),
        ));
    }
    let needs = builder
        .needs
        .take()
        .map(RefCell::into_inner)
        .unwrap_or_default();
    // The function's places, taken by whoever adds it to `functions`.
    shared
        .last_places
        .replace(std::mem::take(&mut builder.places));
    Ok((builder.finish(vec![result])?, needs))
}

/// A template instance: its name, the input ports bound to its Field
/// parameters, and its type parameters as the IR spells them.
type Instance = (String, Vec<u32>, Vec<String>);

/// What a name resolves to inside a block.
#[derive(Clone)]
enum Local {
    Value(ValueId),
    /// The carried value of the loop that introduced the name.
    Carried {
        depth: usize,
        ty: Type,
        /// For a loop that may `break`: the struct of the value and
        /// whether it stopped; the name is its first field.
        pair: Option<Type>,
    },
}

struct Builder<'a> {
    manifest: &'a Manifest,
    shared: &'a Shared<'a>,
    is_entry: bool,
    /// In a function's body, what it needs, each capability with the
    /// builtin that needs it; a Kernel's own code needs its `requires`.
    needs: Option<RefCell<Vec<(Capability, String)>>>,
    name: String,
    args: Vec<(String, Type)>,
    /// Field parameters bound to input ports, for a template's instance.
    fields: Vec<(String, u32)>,
    /// Visible bindings, innermost last.
    locals: Vec<(String, Local)>,
    /// Index into `locals` where each open block scope starts.
    scopes: Vec<usize>,
    /// The names bound in each open block scope, so a repeated name in
    /// one block is found in logarithmic time.
    bound: Vec<BTreeSet<String>>,
    /// Instruction buffers, one per open block; the last is current.
    buffers: Vec<Vec<Instruction>>,
    /// Type of every emitted value by id.
    types: Vec<Type>,
    /// Carried types of enclosing loops, innermost last.
    loops: Vec<Type>,
    /// Open `[each ...]` builders: the element's position and the loop
    /// depth at the builder, so `index()` is that position unless a loop
    /// opened since.
    each: Vec<(u32, usize)>,
    /// Every `let`, in source order, with its value: what a trace shows.
    bindings: Vec<(ValueId, String)>,
    /// Unnamed numbers written without a width, exact (a float as a
    /// `Double`, an integer as a `Long`) until they meet a width.
    literals: std::collections::HashMap<u32, entelechy_ir::Number>,
    /// Unnamed vectors, matrices and arrays of written numbers alone;
    /// each takes the kind it meets as a whole.
    aggregates: BTreeSet<u32>,
    /// Written integers a division truncated (`1 / 2`); a float or
    /// double refuses them.
    truncated: BTreeSet<u32>,
    /// Written numbers past the largest `Float`, where each is written;
    /// one still a `Float` when the function is finished is refused.
    too_large: std::collections::HashMap<u32, Span>,
    /// Where each value is written in the source.
    places: crate::Places,
}

impl<'a> Builder<'a> {
    fn new(shared: &'a Shared<'a>, is_entry: bool, name: &str, fields: Vec<(String, u32)>) -> Self {
        Self {
            manifest: shared.manifest,
            shared,
            is_entry,
            needs: None,
            name: name.to_owned(),
            args: Vec::new(),
            fields,
            locals: Vec::new(),
            scopes: vec![0],
            bound: vec![BTreeSet::new()],
            buffers: vec![Vec::new()],
            types: Vec::new(),
            loops: Vec::new(),
            each: Vec::new(),
            bindings: Vec::new(),
            literals: std::collections::HashMap::new(),
            aggregates: BTreeSet::new(),
            truncated: BTreeSet::new(),
            too_large: std::collections::HashMap::new(),
            places: crate::Places::new(),
        }
    }

    /// A scalar converted to `kind` when it differs, as a constructor
    /// converts a component.
    fn converted(&mut self, id: ValueId, kind: Type) -> ValueId {
        let from = self.ty(id);
        if from == kind {
            return id;
        }
        let intrinsic = match kind {
            Type::Float => Intrinsic::ToFloat,
            Type::Int => Intrinsic::ToInt,
            Type::Double => Intrinsic::ToDouble,
            _ => Intrinsic::ToLong,
        };
        self.emit(kind, Op::Intrinsic(intrinsic, vec![id]))
    }

    fn finish(mut self, results: Vec<ValueId>) -> Result<Function, CompileError> {
        // A written number past the largest `Float` that stayed one.
        let mut large: Vec<(u32, Span)> = self
            .too_large
            .iter()
            .map(|(id, span)| (*id, *span))
            .collect();
        large.sort_by_key(|(_, span)| (span.line, span.column));
        for (id, _) in large {
            self.refuse_large(ValueId(id))?;
        }
        let body = self.buffers.pop().unwrap_or_default();
        Ok(Function {
            name: self.name,
            args: self.args.into_iter().map(|(_, ty)| ty).collect(),
            body,
            results,
            bindings: self.bindings,
        })
    }

    /// Fixes a written value's width as it stands: it is no longer
    /// retyped by what it meets.
    fn fix_width(&mut self, id: ValueId) -> Result<(), CompileError> {
        self.refuse_large(id)?;
        self.literals.remove(&id.0);
        self.aggregates.remove(&id.0);
        self.truncated.remove(&id.0);
        Ok(())
    }

    /// Refuses a written number past the largest `Float` that is a
    /// `Float`.
    fn refuse_large(&self, id: ValueId) -> Result<(), CompileError> {
        let (Some(span), Some(entelechy_ir::Number::Double(written))) =
            (self.too_large.get(&id.0), self.literals.get(&id.0))
        else {
            return Ok(());
        };
        if self.types[id.0 as usize] != Type::Float {
            return Ok(());
        }
        Err(semantic(
            *span,
            format!(
                "`{written:e}` is past the largest `Float`; write it as a `Double`, `{written:e}d`"
            ),
        ))
    }

    /// Reserves the next value id; `place` emits into it.
    fn reserve(&mut self, ty: Type) -> ValueId {
        self.types.push(ty);
        ValueId((self.types.len() - 1) as u32)
    }

    fn place(&mut self, id: ValueId, ty: Type, op: Op) -> ValueId {
        self.types[id.0 as usize] = ty;
        if let Op::Construct(parts) = &op
            && !matches!(ty, Type::Struct(index) if !self.structs()[index as usize].is_array())
            && parts.iter().all(|part| self.is_written(*part))
        {
            self.aggregates.insert(id.0);
        }
        self.buffers
            .last_mut()
            .expect("a block is open")
            .push(Instruction { id, ty, op });
        id
    }

    fn emit(&mut self, ty: Type, op: Op) -> ValueId {
        let id = self.reserve(ty);
        self.place(id, ty, op)
    }

    fn ty(&self, id: ValueId) -> Type {
        self.types[id.0 as usize]
    }

    /// A scatter's pair, from the body or `emit`: a struct of an address
    /// into the target and a value of its element. An integer address is
    /// converted to float, exact for every Field coordinate.
    fn scatter_pair(&mut self, pair: ValueId, span: Span) -> Result<ValueId, CompileError> {
        let Mode::Scatter { base, .. } = self.manifest.mode else {
            return Ok(pair);
        };
        let expected = self.manifest.outputs[0].element();
        let actual = self.ty(pair);
        let mut pair = pair;
        let target = &self.manifest.inputs[base as usize];
        let address = target.ty.index_type();
        let table = self.shared.structs.borrow();
        let fields = match actual {
            Type::Struct(index) if !table[index as usize].is_array() => {
                Some(&table[index as usize].fields)
            }
            _ => None,
        };
        let well_formed = fields.is_some_and(|fields| {
            fields.len() == 2 && fields[0].ty == address && fields[1].ty == expected
        });
        // An integer address, such as `IVec2` into a grid.
        let integer_address = fields.is_some_and(|fields| {
            fields.len() == 2
                && fields[0].ty.other_kind() == Some(address)
                && fields[1].ty == expected
        });
        if integer_address {
            let integer = fields.map(|fields| fields[0].ty).unwrap_or(address);
            let name = format!(
                "({}, {})",
                describe(address, &table),
                describe(expected, &table)
            );
            let existing = table.iter().position(|def| def.name == name);
            drop(table);
            let normal = match existing {
                Some(index) => Type::Struct(index as u32),
                None => {
                    let mut table = self.shared.structs.borrow_mut();
                    table.push(StructDef::record(
                        name,
                        vec![
                            Field {
                                name: "address".into(),
                                ty: address,
                            },
                            Field {
                                name: "value".into(),
                                ty: expected,
                            },
                        ],
                    ));
                    Type::Struct((table.len() - 1) as u32)
                }
            };
            let at = self.emit(integer, Op::Extract(pair, 0));
            let at = self.emit(address, Op::Intrinsic(Intrinsic::ToFloat, vec![at]));
            let value = self.emit(expected, Op::Extract(pair, 1));
            pair = self.emit(normal, Op::Construct(vec![at, value]));
        } else if !well_formed {
            // A struct given is shown with its fields' types.
            let given = match fields {
                Some(fields) => format!(
                    "{} {{ {} }}",
                    describe(actual, &table),
                    fields
                        .iter()
                        .map(|f| format!("{}: {}", f.name, describe(f.ty, &table)))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                None => describe(actual, &table),
            };
            return Err(semantic(
                span,
                format!(
                    "a scatter into `{}` gives a struct of two fields per element, an address ({}) and a value ({}), but the body gives {given}",
                    target.name,
                    describe(address, &table),
                    describe(expected, &table),
                ),
            ));
        }
        Ok(pair)
    }

    fn describe(&self, ty: Type) -> String {
        describe(ty, &self.structs())
    }

    /// Reads the current loop's carried value, or its first field when
    /// the loop may `break`.
    pub(super) fn read_carried(&mut self, ty: Type, pair: Option<Type>) -> ValueId {
        match pair {
            None => self.emit(ty, Op::Carried),
            Some(pair) => {
                let carried = self.emit(pair, Op::Carried);
                self.emit(ty, Op::Extract(carried, 0))
            }
        }
    }

    /// The struct a loop with a `break` carries, `(value, stopped)`,
    /// made once per value type.
    pub(super) fn pair_type(&self, value: Type) -> Type {
        let name = format!("({}, stopped)", self.describe(value));
        if let Some(index) = self.shared.structs.find(&name) {
            return Type::Struct(index);
        }
        let mut table = self.shared.structs.borrow_mut();
        table.push(StructDef::record(
            name,
            vec![
                Field {
                    name: "value".into(),
                    ty: value,
                },
                Field {
                    name: "stopped".into(),
                    ty: Type::Bool,
                },
            ],
        ));
        Type::Struct((table.len() - 1) as u32)
    }

    /// The struct table, borrowed for the Kernel's lifetime so a value
    /// read through it may be emitted at once.
    fn structs(&self) -> std::cell::Ref<'a, Vec<StructDef>> {
        self.shared.structs.borrow()
    }

    /// The array type `[element; count]`.
    fn array_type(&self, element: Type, count: u32, span: Span) -> Result<Type, CompileError> {
        array_type(self.shared.structs, element, count, span)
    }

    /// Lowers a block into a fresh buffer and returns it with its result.
    fn nested(&mut self, block: &Block) -> Result<IrBlock, CompileError> {
        self.buffers.push(Vec::new());
        let result = self.block(block);
        let body = self.buffers.pop().expect("the nested buffer is open");
        Ok(IrBlock {
            body,
            result: result?,
        })
    }

    /// Lowers a single expression as a block (for `else if` and `until`).
    fn nested_expr(&mut self, expr: &Expr) -> Result<IrBlock, CompileError> {
        self.buffers.push(Vec::new());
        let result = self.value(expr);
        let body = self.buffers.pop().expect("the nested buffer is open");
        Ok(IrBlock {
            body,
            result: result?,
        })
    }

    fn block(&mut self, block: &Block) -> Result<ValueId, CompileError> {
        self.scoped(block, |builder, result| builder.value(result))
    }

    /// Lowers a block's bindings in their own scope, then its result
    /// through `result`, since a loop body's result may `break`.
    pub(super) fn scoped<F>(&mut self, block: &Block, result: F) -> Result<ValueId, CompileError>
    where
        F: FnOnce(&mut Self, &Expr) -> Result<ValueId, CompileError>,
    {
        self.scopes.push(self.locals.len());
        self.bound.push(BTreeSet::new());
        let lowered = (|| {
            for (name, value) in &block.lets {
                // A statement for its effect: `put(...)`, or `let _ = ...`.
                if name.value == "_" {
                    self.value(value)?;
                    continue;
                }
                if self
                    .bound
                    .last()
                    .expect("a scope is open")
                    .contains(&name.value)
                {
                    return Err(semantic(
                        name.span,
                        format!(
                            "`{}` is already bound in this block; bindings are immutable",
                            name.value
                        ),
                    ));
                }
                let id = self.value(value)?;
                // A name fixes a number's width: every use sees one value.
                self.fix_width(id)?;
                if name.value != "_" {
                    self.bindings.push((id, name.value.clone()));
                }
                self.bound
                    .last_mut()
                    .expect("a scope is open")
                    .insert(name.value.clone());
                self.locals.push((name.value.clone(), Local::Value(id)));
            }
            result(self, &block.result)
        })();
        let start = self.scopes.pop().expect("a scope is open");
        self.bound.pop();
        self.locals.truncate(start);
        lowered
    }

    /// Whether `name` is bound here: a local, an argument or a parameter.
    fn binds(&self, name: &str) -> bool {
        self.locals.iter().any(|(local, _)| local == name)
            || self.args.iter().any(|(arg, _)| arg == name)
            || self.manifest.params.iter().any(|param| param.name == name)
    }

    fn name(&mut self, name: &str, span: Span) -> Result<ValueId, CompileError> {
        if let Some((_, local)) = self.locals.iter().rev().find(|(local, _)| local == name) {
            return match local.clone() {
                Local::Value(id) => Ok(id),
                Local::Carried { depth, ty, pair } => {
                    if depth != self.loops.len() {
                        return Err(semantic(
                            span,
                            format!(
                                "`{name}` is carried by an outer loop; pass it through this loop's carried value"
                            ),
                        ));
                    }
                    Ok(self.read_carried(ty, pair))
                }
            };
        }
        if let Some(index) = self.args.iter().position(|(arg, _)| arg == name) {
            let ty = self.args[index].1;
            return Ok(self.emit(ty, Op::Arg(index as u32)));
        }
        if let Some(index) = self.manifest.params.iter().position(|p| p.name == name) {
            let ty = self.manifest.params[index].ty;
            return Ok(self.emit(ty, Op::Param(index as u32)));
        }
        if self.manifest.inputs.iter().any(|input| input.name == name) {
            return Err(semantic(
                span,
                format!(
                    "`{name}` is an input port; read it with sample({name}), fetch({name}, coord), size({name}), at({name}, i) or count({name})"
                ),
            ));
        }
        if self
            .manifest
            .outputs
            .iter()
            .any(|output| output.name == name)
        {
            return Err(semantic(
                span,
                format!("`{name}` is the output; produce it with `return`"),
            ));
        }
        // Every name in reach, for a near-miss suggestion.
        let known: Vec<&str> = self
            .locals
            .iter()
            .map(|(local, _)| local.as_str())
            .chain(self.args.iter().map(|(arg, _)| arg.as_str()))
            .chain(self.manifest.params.iter().map(|p| p.name.as_str()))
            .chain(self.manifest.inputs.iter().map(|port| port.name.as_str()))
            .collect();
        Err(semantic(
            span,
            format!(
                "`{name}` is not defined{}",
                entelechy_language::did_you_mean(name, known)
            ),
        ))
    }

    /// Calls a template: its Field parameters bind to the input ports the
    /// arguments name, checked by shape and layout; each binding's
    /// instance lowers once and takes the other arguments by value.
    fn instantiate(
        &mut self,
        template: &FnDecl,
        args: &[Expr],
        span: Span,
    ) -> Result<ValueId, CompileError> {
        let name = template.name.value.as_str();
        self.arity(name, args, template.args.len(), span)?;
        let mut fields = Vec::new();
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut generics: Vec<(String, Type)> = Vec::new();
        for ((param, ty), arg) in template.args.iter().zip(args) {
            // A type parameter binds to its first argument's type; later
            // arguments of it must agree.
            if template.generics.iter().any(|g| g.value == ty.value) {
                let id = self.value(arg)?;
                let actual = self.ty(id);
                match generics
                    .iter()
                    .find(|(parameter, _)| parameter == &ty.value)
                {
                    Some((_, bound)) if *bound != actual => {
                        return Err(semantic(
                            arg.span,
                            format!(
                                "`{name}` takes `{}` as {} here, but as {} for an earlier argument",
                                ty.value,
                                self.describe(actual),
                                self.describe(*bound)
                            ),
                        ));
                    }
                    Some(_) => {}
                    None => generics.push((ty.value.clone(), actual)),
                }
                values.push(id);
                continue;
            }
            match resolve_type(ty, self.shared.structs) {
                Ok(expected) => {
                    let id = self.value(arg)?;
                    let id = self.adapt(id, expected);
                    if self.ty(id) != expected {
                        return Err(semantic(
                            arg.span,
                            format!(
                                "`{name}` expects {} here, found {}",
                                self.describe(expected),
                                self.describe(self.ty(id))
                            ),
                        ));
                    }
                    values.push(id);
                }
                Err(_) => {
                    let declared = port_type(ty, self.shared.structs, self.shared.aliases)?;
                    let (index, port) = self.input_index(arg, name)?;
                    if port.ty.shape != declared.shape || port.ty.schema != declared.schema {
                        return Err(semantic(
                            arg.span,
                            format!(
                                "`{name}` expects {} for `{}`, but `{}` is {}",
                                ty.value,
                                param.value,
                                port.name,
                                entelechy_ir::port_type_name(&port.ty, &self.structs())
                            ),
                        ));
                    }
                    fields.push((param.value.clone(), index));
                    indices.push(index);
                }
            }
        }
        let key = (
            name.to_owned(),
            indices.clone(),
            generics
                .iter()
                .map(|(_, ty)| format!("{ty:?}"))
                .collect::<Vec<_>>(),
        );
        let existing = self.shared.instances.borrow().get(&key).copied();
        let index = match existing {
            Some(index) => index,
            None => {
                self.refuse_cycle(name, span)?;
                let bound = indices
                    .iter()
                    .map(|i| self.manifest.inputs[*i as usize].name.clone())
                    .chain(generics.iter().map(|(_, ty)| self.describe(*ty)))
                    .collect::<Vec<_>>()
                    .join(", ");
                let instance = lower_function(
                    self.shared,
                    template,
                    fields,
                    generics.clone(),
                    &format!("{name}<{bound}>"),
                )
                .map_err(|error| match (error, generics.is_empty()) {
                    // An error in a generic body names the instance.
                    (
                        CompileError::Semantic {
                            span,
                            message,
                            declaration,
                        },
                        false,
                    ) => {
                        let bound = generics
                            .iter()
                            .map(|(parameter, ty)| {
                                format!("`{parameter}` as {}", self.describe(*ty))
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        CompileError::Semantic {
                            span,
                            message: format!("{message} (in `{name}` with {bound})"),
                            declaration,
                        }
                    }
                    (error, _) => error,
                });
                let (instance, needs) = instance?;
                let index = {
                    let mut functions = self.shared.functions.borrow_mut();
                    functions.push((format!("{name}<{bound}>"), instance));
                    self.shared
                        .places
                        .borrow_mut()
                        .push(self.shared.last_places.take());
                    (functions.len() - 1) as u32
                };
                self.shared.needs.borrow_mut().insert(index, needs);
                self.shared.instances.borrow_mut().insert(key, index);
                index
            }
        };
        let ret = {
            let functions = self.shared.functions.borrow();
            functions[index as usize].1.result_type()
        };
        self.called(index, span)?;
        Ok(self.emit(ret, Op::Call(index, values)))
    }

    fn input_index(&self, expr: &Expr, what: &str) -> Result<(u32, &Port), CompileError> {
        let ExprKind::Name(name) = &expr.kind else {
            return Err(semantic(
                expr.span,
                format!("`{what}` expects an input port name"),
            ));
        };
        // A Field parameter stands for the port it was bound to.
        if let Some((_, index)) = self.fields.iter().find(|(field, _)| field == name) {
            return Ok((*index, &self.manifest.inputs[*index as usize]));
        }
        let index = self
            .manifest
            .inputs
            .iter()
            .position(|input| &input.name == name)
            .ok_or_else(|| {
                let known: Vec<&str> = self
                    .manifest
                    .inputs
                    .iter()
                    .map(|port| port.name.as_str())
                    .collect();
                semantic(
                    expr.span,
                    format!(
                        "`{name}` is not an input{}",
                        entelechy_language::did_you_mean(name, known)
                    ),
                )
            })?;
        Ok((index as u32, &self.manifest.inputs[index]))
    }

    /// The shape the entry point maps over: the output's, or the first
    /// input's for a reduction.
    fn output_shape(&self) -> Shape {
        match self.manifest.mode {
            Mode::Reduce | Mode::Scan | Mode::Scatter { .. } | Mode::Prefix { .. } => self
                .manifest
                .inputs
                .first()
                .map_or(Shape::Rows, Port::shape),
            Mode::Map => self
                .manifest
                .outputs
                .first()
                .map_or(Shape::Grid, Port::shape),
        }
    }

    fn entry_only(&self, name: &str, span: Span) -> Result<(), CompileError> {
        if self.is_entry {
            Ok(())
        } else {
            Err(semantic(
                span,
                format!(
                    "`{name}` is only available inside `compute`; pass its value as an argument"
                ),
            ))
        }
    }

    fn require(&self, name: &str, capability: Capability, span: Span) -> Result<(), CompileError> {
        if let Some(needs) = &self.needs {
            let mut needs = needs.borrow_mut();
            if !needs.iter().any(|(had, _)| *had == capability) {
                needs.push((capability, name.to_owned()));
            }
            return Ok(());
        }
        if self.manifest.requires.contains(capability) {
            Ok(())
        } else {
            Err(semantic(
                span,
                format!("`{name}` needs `{capability}`; add it to `requires`"),
            ))
        }
    }

    /// Refuses a call to `name` while it is being lowered: the chain of
    /// calls would reach back to it, and a Kernel has no stack.
    fn refuse_cycle(&self, name: &str, span: Span) -> Result<(), CompileError> {
        let lowering = self.shared.lowering.borrow();
        let Some(start) = lowering.iter().position(|n| n == name) else {
            return Ok(());
        };
        let message = if start + 1 == lowering.len() {
            format!(
                "`{name}` calls itself, and a Kernel has no stack to call into; a `loop` with the work as its carried value is the form"
            )
        } else {
            let chain = lowering[start..]
                .iter()
                .map(String::as_str)
                .chain(std::iter::once(name))
                .collect::<Vec<_>>()
                .join(" -> ");
            format!(
                "`{name}` is reached again through the calls it makes ({chain}), and a Kernel has no stack to call into; a `loop` with the work as its carried value is the form"
            )
        };
        Err(semantic(span, message))
    }

    /// The index in `functions` of the plain function `name`, lowered on
    /// its first call, at `span`.
    fn function_index(&self, name: &str, span: Span) -> Result<Option<u32>, CompileError> {
        let Some(&(position, declaration)) = self.shared.plain.get(name) else {
            return Ok(None);
        };
        if let Some(index) = self.shared.lowered.borrow().get(&position) {
            return Ok(Some(*index));
        }
        self.refuse_cycle(name, span)?;
        let (function, needs) =
            lower_function(self.shared, declaration, Vec::new(), Vec::new(), name)?;
        let index = {
            let mut functions = self.shared.functions.borrow_mut();
            functions.push((name.to_owned(), function));
            self.shared
                .places
                .borrow_mut()
                .push(self.shared.last_places.take());
            (functions.len() - 1) as u32
        };
        self.shared.needs.borrow_mut().insert(index, needs);
        self.shared.lowered.borrow_mut().insert(position, index);
        Ok(Some(index))
    }

    /// A call to the function at `index`: what it needs is the caller's
    /// to hold, a function's passed on to its own callers.
    fn called(&self, index: u32, span: Span) -> Result<(), CompileError> {
        let needs = self
            .shared
            .needs
            .borrow()
            .get(&index)
            .cloned()
            .unwrap_or_default();
        for (capability, via) in needs {
            if let Some(own) = &self.needs {
                let mut own = own.borrow_mut();
                if !own.iter().any(|(had, _)| *had == capability) {
                    own.push((capability, via));
                }
            } else if !self.manifest.requires.contains(capability) {
                let function = self.shared.functions.borrow()[index as usize].0.clone();
                return Err(semantic(
                    span,
                    format!(
                        "`{function}` calls `{via}`, which needs `{capability}`; add it to `requires`"
                    ),
                ));
            }
        }
        Ok(())
    }

    /// An integer vector address converted to the float one a Field
    /// takes, exact for every Field coordinate; other values pass.
    fn float_address(&mut self, address: ValueId, expected: Type) -> ValueId {
        if self.ty(address).other_kind() == Some(expected) {
            self.emit(expected, Op::Intrinsic(Intrinsic::ToFloat, vec![address]))
        } else {
            address
        }
    }

    fn arity(
        &self,
        name: &str,
        args: &[Expr],
        count: usize,
        span: Span,
    ) -> Result<(), CompileError> {
        if args.len() == count {
            Ok(())
        } else {
            Err(semantic(
                span,
                format!("`{name}` takes {count} argument(s), found {}", args.len()),
            ))
        }
    }
}
