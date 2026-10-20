//! What a Kernel reaches of the declarations it holds: the functions it
//! calls, and those they call in turn; the structs and types it names,
//! and those their fields and targets name. What it does not reach
//! cannot change what it compiles to, so a session that knows what a
//! Kernel reaches compiles it again only when that changes.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::rename::{visit, visit_block};
use crate::{Document, FnDecl, Kernel, NodeKind, NodeMember, StructDecl, TypeAlias};

/// The declarations a Kernel reaches, each once, in the order it holds
/// them.
pub struct Reached<'a> {
    /// The functions it calls, directly or through another.
    pub functions: Vec<&'a FnDecl>,
    /// The structs it names, directly or through another declaration.
    pub structs: Vec<&'a StructDecl>,
    /// The types it names, directly or through another declaration.
    pub types: Vec<&'a TypeAlias>,
}

/// Each word of a type as written, `Rows<Ray>` or `[Ray; 4]`.
fn words(ty: &str, out: &mut Vec<String>) {
    out.extend(
        ty.split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .filter(|word| !word.is_empty())
            .map(str::to_owned),
    );
}

impl Kernel {
    /// The names this Kernel's own parts mention, called or named: its
    /// ports, params, memories and domain, and the blocks it computes
    /// with; not yet through the functions they call.
    fn roots(&self, called: &mut Vec<String>, named: &mut Vec<String>) {
        {
            let mut seen = |name: &str, call: bool| {
                if call {
                    called.push(name.to_owned());
                } else {
                    named.push(name.to_owned());
                }
            };
            visit_block(&self.compute, &mut seen);
            if let Some(domain) = &self.domain {
                visit(domain, &mut seen);
            }
            for param in &self.params {
                visit(&param.default, &mut seen);
            }
            if let Some(merge) = &self.merge {
                visit_block(&merge.body, &mut seen);
            }
            if let Some(carry) = &self.carry {
                visit(&carry.init, &mut seen);
            }
        }
        for port in self
            .inputs
            .iter()
            .chain(&self.outputs)
            .chain(&self.memories)
        {
            words(&port.ty.value, named);
        }
        for param in &self.params {
            words(&param.ty.value, named);
        }
        if let Some(scatter) = &self.scatter {
            named.push(scatter.into.value.clone());
            called.push(scatter.combine.value.clone());
        }
        if let Some(prefix) = &self.prefix {
            called.push(prefix.combine.value.clone());
        }
    }

    /// The declarations this Kernel reaches from its own parts
    /// ([`Kernel::roots`]); then through each function reached, its
    /// signature and body, and each struct or type reached, its fields
    /// and target.
    pub fn reached(&self) -> Reached<'_> {
        let mut called: Vec<String> = Vec::new();
        let mut named: Vec<String> = Vec::new();
        self.roots(&mut called, &mut named);
        let mut functions: BTreeSet<usize> = BTreeSet::new();
        let mut structs: BTreeSet<usize> = BTreeSet::new();
        let mut types: BTreeSet<usize> = BTreeSet::new();
        while !called.is_empty() || !named.is_empty() {
            while let Some(name) = called.pop() {
                for (i, function) in self.functions.iter().enumerate() {
                    if function.name.value == name && functions.insert(i) {
                        visit_block(&function.body, &mut |name, call| {
                            if call {
                                called.push(name.to_owned());
                            } else {
                                named.push(name.to_owned());
                            }
                        });
                        for (_, ty) in &function.args {
                            words(&ty.value, &mut named);
                        }
                        words(&function.ret.value, &mut named);
                        for generic in &function.generics {
                            named.push(generic.value.clone());
                        }
                    }
                }
            }
            while let Some(name) = named.pop() {
                for (i, decl) in self.structs.iter().enumerate() {
                    if decl.name.value == name && structs.insert(i) {
                        for (_, ty) in &decl.fields {
                            words(&ty.value, &mut named);
                        }
                    }
                }
                for (i, decl) in self.types.iter().enumerate() {
                    if decl.name.value == name && types.insert(i) {
                        words(&decl.target.value, &mut named);
                    }
                }
                // A function passed by name, as a scan's or a fold's rule.
                if self.functions.iter().any(|f| f.name.value == name) {
                    called.push(name);
                }
            }
        }
        Reached {
            functions: functions.into_iter().map(|i| &self.functions[i]).collect(),
            structs: structs.into_iter().map(|i| &self.structs[i]).collect(),
            types: types.into_iter().map(|i| &self.types[i]).collect(),
        }
    }
}

impl Kernel {
    /// This Kernel holding only the declarations it reaches
    /// ([`Kernel::reached`]): it compiles to what it compiled to, and keeps
    /// none of the rest of its document.
    pub fn trimmed(&self) -> Kernel {
        let reached = self.reached();
        let functions = reached.functions.into_iter().cloned().collect();
        let structs = reached.structs.into_iter().cloned().collect();
        let types = reached.types.into_iter().cloned().collect();
        Kernel {
            functions,
            structs,
            types,
            ..self.clone()
        }
    }
}

/// Each name a type as written may hold, scoped ones (`Ray@a.ent`) whole
/// and in their words: more than [`words`] finds, never less.
fn type_names(ty: &str, out: &mut Vec<String>) {
    out.extend(
        ty.split(|c: char| c.is_whitespace() || "<>[];,()".contains(c))
            .filter(|word| !word.is_empty())
            .map(str::to_owned),
    );
    words(ty, out);
}

/// The declarations of `outer` reached by name from `names`, in the order
/// `outer` holds them: each declaration a name names, and what its
/// signature, body, fields or target name in turn. A name more than one
/// declaration holds is followed into each, so what is reached holds all
/// a Kernel naming only `names` reaches ([`Kernel::reached`]), whichever
/// of them its names come to stand for.
pub(crate) fn reached_by_name<'a>(
    names: Vec<String>,
    outer: (&'a [StructDecl], &'a [TypeAlias], &'a [FnDecl]),
) -> (Vec<&'a StructDecl>, Vec<&'a TypeAlias>, Vec<&'a FnDecl>) {
    let (structs, types, functions) = outer;
    fn index<'a, T>(
        decls: &'a [T],
        name: impl Fn(&'a T) -> &'a str,
    ) -> HashMap<&'a str, Vec<usize>> {
        let mut by: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, decl) in decls.iter().enumerate() {
            by.entry(name(decl)).or_default().push(i);
        }
        by
    }
    let by_struct = index(structs, |d| d.name.value.as_str());
    let by_type = index(types, |d| d.name.value.as_str());
    let by_function = index(functions, |d| d.name.value.as_str());
    let (mut found_structs, mut found_types, mut found_functions) =
        (BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
    let mut followed: HashSet<String> = HashSet::new();
    let mut pending = names;
    while let Some(name) = pending.pop() {
        if followed.contains(&name) {
            continue;
        }
        for &i in by_function.get(name.as_str()).into_iter().flatten() {
            if found_functions.insert(i) {
                let function = &functions[i];
                visit_block(&function.body, &mut |name, _| pending.push(name.to_owned()));
                for (_, ty) in &function.args {
                    type_names(&ty.value, &mut pending);
                }
                type_names(&function.ret.value, &mut pending);
                pending.extend(function.generics.iter().map(|g| g.value.clone()));
            }
        }
        for &i in by_struct.get(name.as_str()).into_iter().flatten() {
            if found_structs.insert(i) {
                for (_, ty) in &structs[i].fields {
                    type_names(&ty.value, &mut pending);
                }
            }
        }
        for &i in by_type.get(name.as_str()).into_iter().flatten() {
            if found_types.insert(i) {
                type_names(&types[i].target.value, &mut pending);
            }
        }
        followed.insert(name);
    }
    (
        found_structs.into_iter().map(|i| &structs[i]).collect(),
        found_types.into_iter().map(|i| &types[i]).collect(),
        found_functions.into_iter().map(|i| &functions[i]).collect(),
    )
}

impl Kernel {
    /// Every name this Kernel's own code mentions, called or named: its
    /// parts ([`Kernel::roots`]) and every declaration it holds, whether
    /// its parts reach it or not.
    pub(crate) fn names(&self, out: &mut Vec<String>) {
        let mut named = Vec::new();
        self.roots(out, &mut named);
        out.append(&mut named);
        for function in &self.functions {
            function_names(function, out);
        }
        for decl in &self.structs {
            for (_, ty) in &decl.fields {
                words(&ty.value, out);
            }
        }
        for decl in &self.types {
            words(&decl.target.value, out);
        }
    }
}

fn function_names(function: &FnDecl, out: &mut Vec<String>) {
    visit_block(&function.body, &mut |name, _| out.push(name.to_owned()));
    for (_, ty) in &function.args {
        words(&ty.value, out);
    }
    words(&function.ret.value, out);
}

impl Document {
    /// Every name this document's own code mentions, called or named: its
    /// functions, structs and types, its Kernels (inline in its nodes
    /// too), its inputs and params, its requirements, its nodes' values
    /// and the types they fill. A name it does not mention, it cannot
    /// reach for, whether or not it sees it.
    pub fn mentions(&self) -> BTreeSet<String> {
        let mut out: Vec<String> = Vec::new();
        for function in &self.functions {
            function_names(function, &mut out);
        }
        for decl in &self.structs {
            for (_, ty) in &decl.fields {
                words(&ty.value, &mut out);
            }
        }
        for decl in &self.types {
            words(&decl.target.value, &mut out);
        }
        let kernel_names = |kernel: &Kernel, out: &mut Vec<String>| kernel.names(out);
        for kernel in &self.kernels {
            kernel_names(kernel, &mut out);
        }
        for port in &self.inputs {
            words(&port.ty.value, &mut out);
        }
        for param in &self.params {
            words(&param.ty.value, &mut out);
            visit(&param.default, &mut |name, _| out.push(name.to_owned()));
        }
        for require in &self.requires {
            visit(&require.condition, &mut |name, _| out.push(name.to_owned()));
        }
        for node in &self.nodes {
            match &node.kind {
                NodeKind::Inline(kernel) => kernel_names(kernel, &mut out),
                NodeKind::Fill(ty) | NodeKind::Fields { ty, .. } => words(&ty.value, &mut out),
                NodeKind::Reference(_)
                | NodeKind::SubGraph(_)
                | NodeKind::Font(_)
                | NodeKind::Files(_) => {}
            }
            for member in &node.members {
                if let NodeMember::Set(_, value) = member {
                    visit(value, &mut |name, _| out.push(name.to_owned()));
                }
            }
        }
        out.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::parse_document;

    #[test]
    fn a_kernel_reaches_what_it_calls_and_names_through_them_alone() {
        let document = parse_document(
            "struct Hit { at: Inner; }
struct Inner { v: Float; }
struct Apart { v: Float; }
type Hits = Rows<Hit>;
fn deep() -> Hit { return Hit { at: Inner { v: 1.0 } }; }
fn mid() -> Float { return deep().at.v; }
fn unused() -> Apart { return Apart { v: 2.0 }; }
kernel t.k@1 { requires { field.read; field.write; } input hits: Hits; output result: Rows<Float>; compute { return mid(); } }
",
        )
        .unwrap();
        // The document's declarations, as a Kernel holds them once resolved.
        let mut kernel = document.kernels[0].clone();
        kernel.functions = document.functions.clone();
        kernel.structs = document.structs.clone();
        kernel.types = document.types.clone();
        let reached = kernel.reached();
        let names = |list: Vec<&String>| list.into_iter().cloned().collect::<Vec<_>>();
        assert_eq!(
            names(reached.functions.iter().map(|f| &f.name.value).collect()),
            ["deep", "mid"]
        );
        assert_eq!(
            names(reached.structs.iter().map(|d| &d.name.value).collect()),
            ["Hit", "Inner"]
        );
        assert_eq!(
            names(reached.types.iter().map(|d| &d.name.value).collect()),
            ["Hits"]
        );
        // Trimmed, it holds those alone, its own parts as they were.
        let trimmed = kernel.trimmed();
        assert_eq!(
            names(trimmed.functions.iter().map(|f| &f.name.value).collect()),
            ["deep", "mid"]
        );
        assert_eq!(
            names(trimmed.structs.iter().map(|d| &d.name.value).collect()),
            ["Hit", "Inner"]
        );
        assert_eq!(trimmed.compute, kernel.compute);
    }

    #[test]
    fn a_document_mentions_what_its_own_code_names() {
        let document = parse_document(
            "use \"lib.ent\";
fn own() -> Float { return helper(); }
kernel t.k@1 { requires { field.write; } output result: Rows<Shape>; compute { return Shape { v: own() }; } }
",
        )
        .unwrap();
        let mentions = document.mentions();
        for name in ["helper", "own", "Shape", "Rows", "Float"] {
            assert!(mentions.contains(name), "{name}: {mentions:?}");
        }
        assert!(!mentions.contains("deep"));
    }
}
