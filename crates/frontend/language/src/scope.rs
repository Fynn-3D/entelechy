//! The names a document sees through `use`. The loader finds the
//! documents and which uses which; the names are the language's.
//!
//! Functions, structs and types are scoped by document: a name reaches
//! what its own document declares, or else the one declaration of that
//! name in the documents it uses (an identical struct or type declared by
//! several is one). The using document's own declarations are its own; a
//! declaration it sees by a name keeps the name, and any other of that
//! name is renamed `name@source`, each gathered declaration naming what
//! its own document sees. A function of a builtin's name stands for it in
//! its own document only.

use std::collections::{BTreeSet, HashMap};

use crate::{
    Document, FnDecl, Kernel, NodeKind, Origin, Own, Spanned, StructDecl, TypeAlias, renamed_type,
};

/// What a document's `use` items bring in: structs, type aliases,
/// functions.
pub type Declarations = (Vec<StructDecl>, Vec<TypeAlias>, Vec<FnDecl>);

/// A document reached through `use`, as the loader found it: the name a
/// message and a scoped name write it by, what it declares, and the
/// documents it uses directly, by their place among the sources.
pub struct Source<'a> {
    /// How the document is written in messages and scoped names.
    pub name: String,
    /// Its functions.
    pub functions: &'a [FnDecl],
    /// Its structs.
    pub structs: &'a [StructDecl],
    /// Its types.
    pub types: &'a [TypeAlias],
    /// The sources it uses directly.
    pub uses: Vec<usize>,
}

/// The name a declaration of the source written `source` takes where
/// another holds its own name.
fn scoped_name(name: &str, source: &str) -> String {
    format!("{name}@{source}")
}

/// The declarations the using document reaches. `sources` are every
/// document reached, each after the ones it uses (as a walk of the `use`
/// items finds them); `direct` the ones the using document's own `use`
/// items name; `own` what it declares itself; `is_builtin` the language's
/// builtins by name.
pub fn scope(
    sources: &[Source<'_>],
    direct: &[usize],
    own: Own<'_>,
    is_builtin: &dyn Fn(&str) -> bool,
) -> Result<Declarations, String> {
    let mut reach = Reach::of(sources);
    let (structs, types) = scoped_types(&mut reach, sources, direct, own)?;
    Ok((
        structs,
        types,
        scoped(&reach, sources, direct, own.functions, is_builtin)?,
    ))
}

/// A struct or a type a document declares.
#[derive(Clone)]
enum Declared {
    Struct(StructDecl),
    Type(TypeAlias),
}

impl Declared {
    fn name(&self) -> &str {
        match self {
            Declared::Struct(d) => &d.name.value,
            Declared::Type(d) => &d.name.value,
        }
    }

    /// Whether `other` declares the same: a struct with the same fields in
    /// the same order, or a type of the same target.
    fn same(&self, other: &Declared) -> bool {
        match (self, other) {
            (Declared::Struct(a), Declared::Struct(b)) => a
                .fields
                .iter()
                .map(|(name, ty)| (&name.value, &ty.value))
                .eq(b.fields.iter().map(|(name, ty)| (&name.value, &ty.value))),
            (Declared::Type(a), Declared::Type(b)) => a.target.value == b.target.value,
            _ => false,
        }
    }

    /// This declaration named `name`, each struct or type named in it that
    /// `rename` maps written by the new name.
    fn renamed(&self, name: String, rename: &dyn Fn(&str) -> Option<String>) -> Declared {
        let retype = |ty: &Spanned<String>| Spanned {
            value: renamed_type(&ty.value, rename),
            span: ty.span,
        };
        match self {
            Declared::Struct(d) => Declared::Struct(StructDecl {
                origin: d.origin.clone(),
                name: Spanned {
                    value: name,
                    span: d.name.span,
                },
                fields: d
                    .fields
                    .iter()
                    .map(|(field, ty)| (field.clone(), retype(ty)))
                    .collect(),
            }),
            // An enum's variants name no type.
            Declared::Type(d) => Declared::Type(TypeAlias {
                origin: d.origin.clone(),
                name: Spanned {
                    value: name,
                    span: d.name.span,
                },
                target: if d.variants().is_some() {
                    d.target.clone()
                } else {
                    retype(&d.target)
                },
            }),
        }
    }
}

/// Every source's functions, dependencies first, and each distinct struct
/// or type with every source that declares it.
struct Reach {
    functions: Vec<(FnDecl, usize)>,
    declared: Vec<(Declared, Vec<usize>)>,
}

impl Reach {
    fn of(sources: &[Source<'_>]) -> Self {
        let mut functions = Vec::new();
        let mut declared: Vec<(Declared, Vec<usize>)> = Vec::new();
        for (file, source) in sources.iter().enumerate() {
            // Each declaration knows where it was written, and as what.
            let document: std::sync::Arc<str> = source.name.as_str().into();
            let origin = |name: &Spanned<String>| {
                Some(Origin {
                    document: document.clone(),
                    name: name.value.clone(),
                })
            };
            for decl in source.functions {
                functions.push((
                    FnDecl {
                        origin: origin(&decl.name),
                        ..decl.clone()
                    },
                    file,
                ));
            }
            let mine = source
                .structs
                .iter()
                .map(|d| {
                    Declared::Struct(StructDecl {
                        origin: origin(&d.name),
                        ..d.clone()
                    })
                })
                .chain(source.types.iter().map(|d| {
                    Declared::Type(TypeAlias {
                        origin: origin(&d.name),
                        ..d.clone()
                    })
                }));
            for decl in mine {
                match declared
                    .iter_mut()
                    .find(|(d, _)| d.name() == decl.name() && d.same(&decl))
                {
                    Some((_, files)) => {
                        if !files.contains(&file) {
                            files.push(file);
                        }
                    }
                    None => declared.push((decl, vec![file])),
                }
            }
        }
        Self {
            functions,
            declared,
        }
    }
}

/// The sources `from` reaches through `use`, recursively.
fn closure(from: &[usize], sources: &[Source<'_>], out: &mut BTreeSet<usize>) {
    for &file in from {
        if out.insert(file) {
            closure(&sources[file].uses, sources, out);
        }
    }
}

/// The gathered structs and types, each under one name in the using
/// document's reach, as [`scoped`] names functions: a name two distinct
/// declarations share, or one the using document declares itself, is
/// the using document's own, or else the one it reaches; every other
/// declaration of it is renamed `name@source`; the using document's own
/// declaration alike to the one it reaches is that one, and no clash.
/// Each gathered declaration, and each gathered function, then names the
/// structs and types its own document sees. A name a document reaches
/// two declarations of, and declares neither, is refused where that
/// document names it.
fn scoped_types(
    reach: &mut Reach,
    sources: &[Source<'_>],
    direct: &[usize],
    own: Own<'_>,
) -> Result<(Vec<StructDecl>, Vec<TypeAlias>), String> {
    let named = |name: &str| {
        reach
            .declared
            .iter()
            .filter(|(d, _)| d.name() == name)
            .count()
    };
    // What the using document declares itself of a name, if anything: a
    // declaration alike to the one it reaches is that one.
    let own_decl = |name: &str| -> Option<Declared> {
        own.structs
            .iter()
            .find(|d| d.name.value == name)
            .map(|d| Declared::Struct(d.clone()))
            .or_else(|| {
                own.types
                    .iter()
                    .find(|d| d.name.value == name)
                    .map(|d| Declared::Type(d.clone()))
            })
    };
    let own_differs = |name: &str| {
        own_decl(name).is_some_and(|mine| {
            reach
                .declared
                .iter()
                .any(|(d, _)| d.name() == name && !d.same(&mine))
        })
    };
    let clashing = |name: &str| named(name) > 1 || own_differs(name);
    // A reached declaration the using document makes alike itself is its
    // own, and is not gathered again.
    let own_alike = |d: &Declared| own_decl(d.name()).is_some_and(|mine| d.same(&mine));
    let split = |declared: Vec<Declared>| {
        let mut structs = Vec::new();
        let mut types = Vec::new();
        for decl in declared {
            match decl {
                Declared::Struct(d) => structs.push(d),
                Declared::Type(d) => types.push(d),
            }
        }
        (structs, types)
    };
    // Nothing declared twice and nothing the using document declares:
    // every name is plain, as written.
    if !reach.declared.iter().any(|(d, _)| clashing(d.name())) {
        return Ok(split(
            reach
                .declared
                .iter()
                .filter(|(d, _)| !own_alike(d))
                .map(|(d, _)| d.clone())
                .collect(),
        ));
    }
    let mut top = BTreeSet::new();
    closure(direct, sources, &mut top);
    let reached_from = |file: usize| {
        let mut reached = BTreeSet::new();
        closure(&sources[file].uses, sources, &mut reached);
        reached
    };
    let declared = &reach.declared;
    // The declarations of `name` a set of documents reaches.
    let reaching = |name: &str, reached: &BTreeSet<usize>| -> Vec<usize> {
        (0..declared.len())
            .filter(|&i| {
                declared[i].0.name() == name
                    && declared[i].1.iter().any(|file| reached.contains(file))
            })
            .collect()
    };
    let plain = |i: usize| {
        let name = declared[i].0.name();
        !own_differs(name) && reaching(name, &top) == [i]
    };
    let out_name = |i: usize| {
        let name = declared[i].0.name();
        if plain(i) {
            name.to_owned()
        } else {
            scoped_name(name, &sources[declared[i].1[0]].name)
        }
    };
    // How a document names what it sees: the declaration it makes itself,
    // or the one it reaches; two it reaches and neither its own is refused.
    let mut refused: Option<String> = None;
    let mut reaches: HashMap<usize, BTreeSet<usize>> = HashMap::new();
    let mut view = |file: usize, name: &str, refused: &mut Option<String>| -> Option<String> {
        if !clashing(name) {
            return None;
        }
        let target = match (0..declared.len())
            .find(|&i| declared[i].0.name() == name && declared[i].1.contains(&file))
        {
            Some(i) => i,
            None => {
                let reached = reaches.entry(file).or_insert_with(|| reached_from(file));
                let found = reaching(name, reached);
                match found.as_slice() {
                    [] => return None,
                    [one] => *one,
                    many => {
                        let files: Vec<&str> = many
                            .iter()
                            .map(|&i| sources[declared[i].1[0]].name.as_str())
                            .collect();
                        refused.get_or_insert_with(|| {
                            format!(
                                "`{name}` is declared differently by {}, which {} uses: declare `{name}` in it, or rename one of them",
                                files.join(" and "),
                                sources[file].name
                            )
                        });
                        return None;
                    }
                }
            }
        };
        let renamed = out_name(target);
        (renamed != name).then_some(renamed)
    };
    let mut out = Vec::with_capacity(declared.len());
    for (i, (decl, files)) in declared.iter().enumerate() {
        if own_alike(decl) {
            continue;
        }
        let names = std::cell::RefCell::new(|name: &str| view(files[0], name, &mut refused));
        let rename = |name: &str| (names.borrow_mut())(name);
        out.push(decl.renamed(out_name(i), &rename));
    }
    let functions: Vec<(FnDecl, usize)> = reach
        .functions
        .iter()
        .map(|(decl, file)| {
            let names = std::cell::RefCell::new(|name: &str| view(*file, name, &mut refused));
            let rename = |name: &str| (names.borrow_mut())(name);
            (decl.retyped(&rename), *file)
        })
        .collect();
    if let Some(refused) = refused {
        return Err(refused);
    }
    reach.functions = functions;
    Ok(split(out))
}

/// The gathered functions, each under one name in the using document's
/// reach. Each document sees its own functions and those of every
/// document it reaches through `use`; a name two of those declare is
/// the document's own, or else ambiguous. The using document's view
/// keeps plain names: each function it sees by name keeps that name,
/// every other is renamed `name@source`, and each body's calls follow
/// its own document's view. A function of a builtin's name stands for it
/// in its own document only: everywhere else the name is the builtin's.
fn scoped(
    reach: &Reach,
    sources: &[Source<'_>],
    direct: &[usize],
    own: &[FnDecl],
    is_builtin: &dyn Fn(&str) -> bool,
) -> Result<Vec<FnDecl>, String> {
    // Which documents declare each name.
    let mut declaring: HashMap<&str, Vec<usize>> = HashMap::new();
    for (decl, file) in &reach.functions {
        let files = declaring.entry(decl.name.value.as_str()).or_default();
        if !files.contains(file) {
            files.push(*file);
        }
    }
    let hides = |name: &str| is_builtin(name);
    // Nothing declared twice, nothing the user shadows and no builtin
    // hidden: every name is plain, as written.
    let clashing = |name: &str| {
        declaring.get(name).is_some_and(|files| files.len() > 1)
            || own.iter().any(|f| f.name.value == name)
            || hides(name)
    };
    if !reach
        .functions
        .iter()
        .any(|(decl, _)| clashing(&decl.name.value))
    {
        return Ok(reach
            .functions
            .iter()
            .map(|(decl, _)| decl.clone())
            .collect());
    }
    // Each document's reach: what it uses, recursively.
    let mut top = BTreeSet::new();
    closure(direct, sources, &mut top);
    // The document a name reaches from `reached`: one declaring it.
    let resolve = |name: &str, reached: &BTreeSet<usize>| -> Option<usize> {
        let files: Vec<usize> = declaring
            .get(name)?
            .iter()
            .copied()
            .filter(|file| reached.contains(file))
            .collect();
        (files.len() == 1).then(|| files[0])
    };
    let seen_by_user = |name: &str, file: usize| {
        !hides(name)
            && !own.iter().any(|f| f.name.value == name)
            && resolve(name, &top) == Some(file)
    };
    let internal = |name: &str, file: usize| {
        if seen_by_user(name, file) {
            name.to_owned()
        } else {
            scoped_name(name, &sources[file].name)
        }
    };
    let mut reaches: HashMap<usize, BTreeSet<usize>> = HashMap::new();
    let mut out = Vec::with_capacity(reach.functions.len());
    for (decl, file) in &reach.functions {
        let file = *file;
        let reached = reaches.entry(file).or_insert_with(|| {
            let mut reached = BTreeSet::new();
            closure(&sources[file].uses, sources, &mut reached);
            reached
        });
        let reached = &*reached;
        let declares = |name: &str| {
            declaring
                .get(name)
                .is_some_and(|files| files.contains(&file))
        };
        let rename = |name: &str| -> Option<String> {
            let target = if declares(name) {
                file
            } else if hides(name) {
                return None;
            } else {
                resolve(name, reached)?
            };
            let renamed = internal(name, target);
            (renamed != name).then_some(renamed)
        };
        out.push(decl.renamed(internal(&decl.name.value, file), &rename));
    }
    Ok(out)
}
/// A document's own declarations, put in front of each of its Kernels'
/// own: every Kernel of a document sees what its document declares, with
/// one layout. The parser gives each Kernel so.
pub(crate) fn document_scope(document: &mut Document) {
    if document.structs.is_empty() && document.types.is_empty() && document.functions.is_empty() {
        return;
    }
    let shared = document.structs.clone();
    let shared_types = document.types.clone();
    let shared_functions = document.functions.clone();
    let prepend = |kernel: &mut Kernel| {
        let mut structs = shared.clone();
        structs.append(&mut kernel.structs);
        kernel.structs = structs;
        let mut types = shared_types.clone();
        types.append(&mut kernel.types);
        kernel.types = types;
        let mut functions = shared_functions.clone();
        functions.append(&mut kernel.functions);
        kernel.functions = functions;
    };
    for kernel in &mut document.kernels {
        prepend(kernel);
    }
    for node in &mut document.nodes {
        if let NodeKind::Inline(kernel) = &mut node.kind {
            prepend(kernel);
        }
    }
}

/// The declarations `used` (what a document's `use` items bring in, as
/// [`scope`] names them) put in front of every Kernel's own and its
/// document's, so a shared struct or type has one layout everywhere. A
/// Kernel's own function, struct or type is the innermost scope: the
/// shared one of its name is renamed out of its way, and the shared
/// declarations that name it follow it, so a library keeps calling and
/// naming its own.
pub(crate) fn resolve(document: &mut Document, used: (&[StructDecl], &[TypeAlias], &[FnDecl])) {
    let (structs, types, functions) = used;
    if structs.is_empty() && types.is_empty() && functions.is_empty() {
        return;
    }
    let prepend = |kernel: &mut Kernel| {
        let own_fns: Vec<String> = kernel
            .functions
            .iter()
            .map(|f| f.name.value.clone())
            .collect();
        // Only a name the shared declarations hold too is renamed:
        // one the using document declares alike is not passed, and the
        // shared functions name the Kernel's own.
        let own_types: Vec<String> = kernel
            .structs
            .iter()
            .map(|s| s.name.value.clone())
            .chain(kernel.types.iter().map(|t| t.name.value.clone()))
            .filter(|name| {
                structs.iter().any(|d| &d.name.value == name)
                    || types.iter().any(|d| &d.name.value == name)
            })
            .collect();
        let call = |name: &str| {
            own_fns
                .iter()
                .any(|own| own == name)
                .then(|| format!("{name}@shared"))
        };
        let ty = |name: &str| {
            own_types
                .iter()
                .any(|own| own == name)
                .then(|| format!("{name}@shared"))
        };
        let spanned = |old: &Spanned<String>, value: String| Spanned {
            value,
            span: old.span,
        };
        let mut all: Vec<StructDecl> = structs
            .iter()
            .map(|decl| StructDecl {
                origin: decl.origin.clone(),
                name: spanned(
                    &decl.name,
                    ty(&decl.name.value).unwrap_or_else(|| decl.name.value.clone()),
                ),
                fields: decl
                    .fields
                    .iter()
                    .map(|(name, t)| (name.clone(), spanned(t, renamed_type(&t.value, &ty))))
                    .collect(),
            })
            .collect();
        all.append(&mut kernel.structs);
        kernel.structs = all;
        let mut aliases: Vec<TypeAlias> = types
            .iter()
            .map(|alias| TypeAlias {
                origin: alias.origin.clone(),
                name: spanned(
                    &alias.name,
                    ty(&alias.name.value).unwrap_or_else(|| alias.name.value.clone()),
                ),
                target: spanned(&alias.target, renamed_type(&alias.target.value, &ty)),
            })
            .collect();
        aliases.append(&mut kernel.types);
        kernel.types = aliases;
        let mut shared: Vec<FnDecl> = functions
            .iter()
            .map(|f| {
                f.renamed(
                    call(&f.name.value).unwrap_or_else(|| f.name.value.clone()),
                    &call,
                )
                .retyped(&ty)
            })
            .collect();
        shared.append(&mut kernel.functions);
        kernel.functions = shared;
    };
    for kernel in &mut document.kernels {
        prepend(kernel);
    }
    for node in &mut document.nodes {
        if let NodeKind::Inline(kernel) = &mut node.kind {
            prepend(kernel);
        }
    }
}
