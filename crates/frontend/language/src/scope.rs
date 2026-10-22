//! The names a document sees through `use`. The loader finds the
//! documents and which uses which; the names are the language's.
//!
//! Functions, structs and types are scoped by document: a name reaches
//! what its own document declares, or else the one declaration of that
//! name in the documents its own `use` items name, not what those use in
//! turn (a struct or type several declare alike, each name in it reaching
//! the same declarations, is one). The using document's own declarations are its own; a
//! declaration it sees by a name keeps the name, and any other of that
//! name is renamed `name@source`, each gathered declaration naming what
//! its own document sees. A function of a builtin's name stands for it in
//! its own document only.
//!
//! A name qualified by a package, `random::fbm2`, reaches the one
//! declaration of that name in the documents of that package the writing
//! document uses: two packages may declare one name, and a document using
//! both names each by its package. The qualified name is written by the
//! name the declaration it reaches holds, so it never reaches the IR.

use std::collections::{BTreeSet, HashMap};

use crate::{
    Document, FnDecl, Kernel, NodeKind, Origin, Own, Spanned, StructDecl, TypeAlias, renamed_type,
};

/// What a document's `use` items bring in: structs, type aliases and
/// functions, each under the name the using document's view gives it; and
/// the names the using document writes qualified by a package, each with
/// the name the declaration it reaches holds.
#[derive(Clone, Debug, Default)]
pub struct Declarations {
    /// The structs.
    pub structs: Vec<StructDecl>,
    /// The type aliases.
    pub types: Vec<TypeAlias>,
    /// The functions.
    pub functions: Vec<FnDecl>,
    /// The qualified names the using document writes.
    pub qualified: Qualified,
}

impl Declarations {
    /// The structs, types and functions, as a Kernel holds them.
    pub fn held(&self) -> (&[StructDecl], &[TypeAlias], &[FnDecl]) {
        (&self.structs, &self.types, &self.functions)
    }
}

/// Names a document writes qualified by a package (`random::fbm2`),
/// each with the name of the declaration it reaches: a function called,
/// or a struct or type named.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Qualified {
    /// Functions called.
    pub calls: Vec<(String, String)>,
    /// Structs and types named.
    pub types: Vec<(String, String)>,
}

impl Qualified {
    /// The name the declaration `name` reaches holds, called (`call`) or
    /// named as a type; none for a name not qualified.
    pub fn of(&self, name: &str, call: bool) -> Option<String> {
        let names = if call { &self.calls } else { &self.types };
        names
            .iter()
            .find(|(written, _)| written == name)
            .map(|(_, held)| held.clone())
    }

    /// Whether it holds no name.
    pub fn is_empty(&self) -> bool {
        self.calls.is_empty() && self.types.is_empty()
    }
}

/// Qualified names as written, each with the name the declaration it
/// reaches holds.
type QualifiedNames = Vec<(String, String)>;

/// A name qualified by a package, `random::fbm2`, as the package's name
/// and the name it declares.
pub fn qualified(name: &str) -> Option<(&str, &str)> {
    name.split_once("::")
}

/// Why the qualified `name`, written by `writer` (a document as a message
/// writes it) that uses `uses`, reaches no declaration of its `kind`:
/// the document uses no package of that name, or the package declares
/// none, the nearest of what it declares (`declared`) said.
fn unreached(
    sources: &[Source<'_>],
    uses: &BTreeSet<usize>,
    writer: &str,
    name: &str,
    declared: &dyn Fn(&Source<'_>) -> Vec<String>,
) -> String {
    let (package, plain) = qualified(name).unwrap_or(("", name));
    let theirs: Vec<&Source<'_>> = uses
        .iter()
        .map(|&f| &sources[f])
        .filter(|source| source.package.as_deref() == Some(package))
        .collect();
    if theirs.is_empty() {
        return format!(
            "`{name}` names the package `{package}`, which {writer} does not use: `use \"{package}.entpkg\";` brings it in"
        );
    }
    let known: Vec<String> = theirs.iter().flat_map(|source| declared(source)).collect();
    let near = crate::did_you_mean(plain, known.iter().map(String::as_str));
    format!("`{name}`: the package `{package}` declares no `{plain}`{near}")
}

/// Every name a source declares, functions, structs and types.
fn every_name(source: &Source<'_>) -> Vec<String> {
    source
        .functions
        .iter()
        .map(|d| d.name.value.clone())
        .chain(source.structs.iter().map(|d| d.name.value.clone()))
        .chain(source.types.iter().map(|d| d.name.value.clone()))
        .collect()
}

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
    /// The package it belongs to, by name; none outside a package. A name
    /// qualified by a package (`random::fbm2`) reaches what that
    /// package's documents declare.
    pub package: Option<String>,
}

/// The name a declaration of the source written `source` takes where
/// another holds its own name.
fn scoped_name(name: &str, source: &str) -> String {
    format!("{name}@{source}")
}

/// The name a call to `name` takes where its document sees no
/// declaration of it: one no declaration holds, written `name` again in
/// a message ([`written_name`]).
fn unseen_name(name: &str) -> String {
    format!("{name}@")
}

/// A name as its source writes it: a scoped or unseen name without what
/// follows its `@`.
pub fn written_name(name: &str) -> &str {
    name.split('@').next().unwrap_or(name)
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
    // What the document's own code mentions: a declaration it does not
    // see stays out of its way by a name of its own only when the
    // document reaches for that name; one it never mentions is left as
    // it is, and the names stay plain.
    let mentioned = std::cell::OnceCell::new();
    let mentions = |name: &str| match own.document {
        Some(document) => mentioned.get_or_init(|| document.mentions()).contains(name),
        None => true,
    };
    // The names the using document writes qualified by a package.
    let written: Vec<String> = match own.document {
        Some(document) => mentioned
            .get_or_init(|| document.mentions())
            .iter()
            .filter(|name| qualified(name).is_some())
            .cloned()
            .collect(),
        None => Vec::new(),
    };
    let (structs, types, qualified_types) =
        scoped_types(&mut reach, sources, direct, own, &mentions, &written)?;
    let (functions, qualified_calls) = scoped(
        &reach,
        sources,
        direct,
        own.functions,
        is_builtin,
        &mentions,
        &written,
    )?;
    // Each qualified name reaches a function, or a struct or type.
    if let Some(name) = written.iter().find(|name| {
        !qualified_calls.iter().any(|(q, _)| q == *name)
            && !qualified_types.iter().any(|(q, _)| q == *name)
    }) {
        let mut top = BTreeSet::new();
        closure(direct, &mut top);
        return Err(unreached(sources, &top, "this document", name, &every_name));
    }
    Ok(Declarations {
        structs,
        types,
        functions,
        qualified: Qualified {
            calls: qualified_calls,
            types: qualified_types,
        },
    })
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

    /// The names it writes for types: its fields' types, or its target's;
    /// an enum's variants name none.
    fn names(&self) -> Vec<String> {
        let names = std::cell::RefCell::new(Vec::new());
        let note = |name: &str| {
            names.borrow_mut().push(name.to_string());
            None
        };
        match self {
            Declared::Struct(d) => {
                for (_, ty) in &d.fields {
                    renamed_type(&ty.value, &note);
                }
            }
            Declared::Type(d) if d.variants().is_none() => {
                renamed_type(&d.target.value, &note);
            }
            Declared::Type(_) => {}
        }
        names.into_inner()
    }

    /// This declaration as written in `document`.
    fn written_in(&self, document: std::sync::Arc<str>) -> Declared {
        let origin = |name: &Spanned<String>| {
            Some(Origin {
                document,
                name: name.value.clone(),
            })
        };
        match self {
            Declared::Struct(d) => Declared::Struct(StructDecl {
                origin: origin(&d.name),
                ..d.clone()
            }),
            Declared::Type(d) => Declared::Type(TypeAlias {
                origin: origin(&d.name),
                ..d.clone()
            }),
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

/// The declarations of `name` the document `file` reaches, by their place
/// in `declared`: its own, or else those of the documents it uses.
fn reaching_from(
    declared: &[(Declared, Vec<usize>)],
    sources: &[Source<'_>],
    file: usize,
    name: &str,
) -> Vec<usize> {
    let of = |within: &dyn Fn(usize) -> bool| -> Vec<usize> {
        (0..declared.len())
            .filter(|&i| declared[i].0.name() == name && declared[i].1.iter().any(|&f| within(f)))
            .collect()
    };
    let own = of(&|f| f == file);
    if own.is_empty() {
        of(&|f| sources[file].uses.contains(&f))
    } else {
        own
    }
}

/// Whether the declaration at `at` is alike to the using document's own
/// `mine`: written alike, and each name it writes reaching, from every
/// document that writes it, what that name reaches from the using
/// document, which sees its own declarations (`own_decl`) or else those of
/// the documents it uses (`direct`).
fn alike_to_own(
    declared: &[(Declared, Vec<usize>)],
    sources: &[Source<'_>],
    direct: &[usize],
    own_decl: &dyn Fn(&str) -> Option<Declared>,
    at: usize,
    mine: &Declared,
    depth: usize,
) -> bool {
    let (decl, files) = &declared[at];
    if depth > 16 || !decl.same(mine) {
        return false;
    }
    decl.names().iter().all(|name| {
        let used: Vec<usize> = (0..declared.len())
            .filter(|&i| {
                declared[i].0.name() == name && declared[i].1.iter().any(|f| direct.contains(f))
            })
            .collect();
        let theirs = own_decl(name);
        files.iter().all(|&file| {
            let reached = reaching_from(declared, sources, file, name);
            match &theirs {
                Some(own) => matches!(reached.as_slice(), [one]
                    if alike_to_own(declared, sources, direct, own_decl, *one, own, depth + 1)),
                None => reached == used,
            }
        })
    })
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
        // Declarations written alike are one only where each name they
        // write reaches the same declarations from every document that
        // writes them: those that reach others are another declaration.
        while let Some((at, apart)) = (0..declared.len()).find_map(|at| {
            let (decl, files) = &declared[at];
            let reached = |file: usize| -> Vec<Vec<usize>> {
                decl.names()
                    .iter()
                    .map(|name| reaching_from(&declared, sources, file, name))
                    .collect()
            };
            let first = reached(files[0]);
            let apart: Vec<usize> = files
                .iter()
                .copied()
                .filter(|&file| reached(file) != first)
                .collect();
            (!apart.is_empty()).then_some((at, apart))
        }) {
            declared[at].1.retain(|file| !apart.contains(file));
            let document: std::sync::Arc<str> = sources[apart[0]].name.as_str().into();
            let decl = declared[at].0.written_in(document);
            declared.push((decl, apart));
        }
        Self {
            functions,
            declared,
        }
    }
}

/// The sources `from` names: what a document sees is what it uses, not
/// what those use in turn.
fn closure(from: &[usize], out: &mut BTreeSet<usize>) {
    out.extend(from.iter().copied());
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
    mentions: &dyn Fn(&str) -> bool,
    written: &[String],
) -> Result<(Vec<StructDecl>, Vec<TypeAlias>, QualifiedNames), String> {
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
    let alike = |at: usize, mine: &Declared| {
        alike_to_own(&reach.declared, sources, direct, &own_decl, at, mine, 0)
    };
    let own_differs = |name: &str| {
        own_decl(name).is_some_and(|mine| {
            (0..reach.declared.len())
                .any(|at| reach.declared[at].0.name() == name && !alike(at, &mine))
        })
    };
    let clashing = |name: &str| named(name) > 1 || own_differs(name);
    // A reached declaration the using document makes alike itself is its
    // own, and is not gathered again.
    let own_alike =
        |at: usize| own_decl(reach.declared[at].0.name()).is_some_and(|mine| alike(at, &mine));
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
    let mut top = BTreeSet::new();
    closure(direct, &mut top);
    // Nothing declared twice, nothing the using document declares, and
    // nothing it mentions but does not see: every name is plain, as
    // written.
    let reached_from = |file: usize| {
        let mut reached = BTreeSet::new();
        closure(&sources[file].uses, &mut reached);
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
    // The declarations a name qualified by a package reaches from a set of
    // documents: those of its name in that package's documents among them.
    let reaching_qualified = |name: &str, reached: &BTreeSet<usize>| -> Vec<usize> {
        let Some((package, plain)) = qualified(name) else {
            return Vec::new();
        };
        (0..declared.len())
            .filter(|&i| {
                declared[i].0.name() == plain
                    && declared[i].1.iter().any(|&file| {
                        reached.contains(&file) && sources[file].package.as_deref() == Some(package)
                    })
            })
            .collect()
    };
    // A gathered declaration or function naming a type qualified by a package.
    let names_qualified = {
        let found = std::cell::Cell::new(false);
        let look = |name: &str| {
            if qualified(name).is_some() {
                found.set(true);
            }
            None
        };
        for (decl, _) in declared {
            let _ = decl.renamed(decl.name().to_owned(), &look);
        }
        for (decl, _) in &reach.functions {
            let _ = decl.retyped(&look);
        }
        found.get()
    };
    // A name a gathered document declares, which a gathered declaration
    // or function names from a document that neither declares nor sees it.
    let unseen = |file: usize, name: &str| {
        declared.iter().any(|(d, _)| d.name() == name)
            && !declared
                .iter()
                .any(|(d, files)| d.name() == name && files.contains(&file))
            && reaching(name, &reached_from(file)).is_empty()
    };
    let names_unseen = {
        let found = std::cell::Cell::new(false);
        let look = |file: usize| {
            let found = &found;
            move |name: &str| {
                if !found.get() && unseen(file, name) {
                    found.set(true);
                }
                None
            }
        };
        for (decl, files) in declared {
            let _ = decl.renamed(decl.name().to_owned(), &look(files[0]));
        }
        for (decl, file) in &reach.functions {
            let _ = decl.retyped(&look(*file));
        }
        found.get()
    };
    let seen = |files: &Vec<usize>| files.iter().any(|file| top.contains(file));
    if !names_unseen
        && !names_qualified
        && written.is_empty()
        && !reach
            .declared
            .iter()
            .any(|(d, files)| clashing(d.name()) || (!seen(files) && mentions(d.name())))
    {
        let (structs, types) = split(
            reach
                .declared
                .iter()
                .enumerate()
                .filter(|&(at, _)| !own_alike(at))
                .map(|(_, (d, _))| d.clone())
                .collect(),
        );
        return Ok((structs, types, Vec::new()));
    }
    // Plain: the one the using document sees by its name, or the only
    // one of its name, which the document neither declares nor mentions.
    let plain = |i: usize| {
        let name = declared[i].0.name();
        (!own_differs(name) && reaching(name, &top) == [i])
            || (named(name) == 1 && own_decl(name).is_none() && !mentions(name))
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
        // A qualified name: the one declaration of its name in that
        // package's documents this one uses.
        if qualified(name).is_some() {
            let reached = reaches.entry(file).or_insert_with(|| reached_from(file));
            return match reaching_qualified(name, reached).as_slice() {
                [one] => Some(out_name(*one)),
                _ => {
                    refused.get_or_insert_with(|| {
                        unreached(sources, reached, &sources[file].name, name, &|source| {
                            source
                                .structs
                                .iter()
                                .map(|d| d.name.value.clone())
                                .chain(source.types.iter().map(|d| d.name.value.clone()))
                                .collect()
                        })
                    });
                    None
                }
            };
        }
        let target = match (0..declared.len())
            .find(|&i| declared[i].0.name() == name && declared[i].1.contains(&file))
        {
            Some(i) => i,
            None => {
                let reached = reaches.entry(file).or_insert_with(|| reached_from(file));
                let found = reaching(name, reached);
                match found.as_slice() {
                    // Declared by a gathered document this one does not
                    // see: a name no declaration holds.
                    [] if declared.iter().any(|(d, _)| d.name() == name) => {
                        return Some(unseen_name(name));
                    }
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
        if own_alike(i) {
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
    // The using document's own qualified names: the declaration each
    // reaches from what it uses, by the name it holds there.
    let mut qualified_names = Vec::new();
    for name in written {
        match reaching_qualified(name, &top).as_slice() {
            [one] => qualified_names.push((name.clone(), out_name(*one))),
            [] => {}
            many => {
                let files: Vec<&str> = many
                    .iter()
                    .map(|&i| sources[declared[i].1[0]].name.as_str())
                    .collect();
                refused.get_or_insert_with(|| {
                    format!(
                        "`{name}` is declared differently by {}, of one package",
                        files.join(" and ")
                    )
                });
            }
        }
    }
    if let Some(refused) = refused {
        return Err(refused);
    }
    reach.functions = functions;
    let (structs, types) = split(out);
    Ok((structs, types, qualified_names))
}

/// The gathered functions, each under one name in the using document's
/// reach. Each document sees its own functions and those of every
/// document it uses itself; a name two of those declare is
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
    mentions: &dyn Fn(&str) -> bool,
    written: &[String],
) -> Result<(Vec<FnDecl>, QualifiedNames), String> {
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
    // What the using document sees: what it uses.
    let mut top = BTreeSet::new();
    closure(direct, &mut top);
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
    let mut reaches: HashMap<usize, BTreeSet<usize>> = HashMap::new();
    for (_, file) in &reach.functions {
        reaches.entry(*file).or_insert_with(|| {
            let mut reached = BTreeSet::new();
            closure(&sources[*file].uses, &mut reached);
            reached
        });
    }
    // A call a gathered function makes to a name a gathered document
    // declares, which its own document neither declares nor sees: it
    // reaches no declaration, wherever the function is called from.
    let unseen = |name: &str, file: usize| {
        declaring
            .get(name)
            .is_some_and(|files| !files.contains(&file))
            && !hides(name)
            && resolve(name, &reaches[&file]).is_none()
    };
    let calls_unseen = reach.functions.iter().any(|(decl, file)| {
        let mut found = false;
        crate::rename::visit_block(&decl.body, &mut |name, call| {
            found = found || (call && unseen(name, *file));
        });
        found
    });
    let calls_qualified = reach.functions.iter().any(|(decl, _)| {
        let mut found = false;
        crate::rename::visit_block(&decl.body, &mut |name, call| {
            found = found || (call && qualified(name).is_some());
        });
        found
    });
    if !calls_unseen
        && !calls_qualified
        && written.is_empty()
        && !reach.functions.iter().any(|(decl, file)| {
            clashing(&decl.name.value) || (!top.contains(file) && mentions(&decl.name.value))
        })
    {
        return Ok((
            reach
                .functions
                .iter()
                .map(|(decl, _)| decl.clone())
                .collect(),
            Vec::new(),
        ));
    }
    // The document a qualified call reaches from `reached`: the one of
    // that package's documents among them declaring its name; `Err` when
    // none or several do.
    let resolve_qualified = |name: &str, reached: &BTreeSet<usize>| -> Result<usize, Vec<usize>> {
        let (package, plain) = qualified(name).unwrap_or(("", name));
        let files: Vec<usize> = declaring
            .get(plain)
            .map(|files| {
                files
                    .iter()
                    .copied()
                    .filter(|&file| {
                        reached.contains(&file) && sources[file].package.as_deref() == Some(package)
                    })
                    .collect()
            })
            .unwrap_or_default();
        match files.as_slice() {
            [one] => Ok(*one),
            _ => Err(files),
        }
    };
    let functions_of = |source: &Source<'_>| -> Vec<String> {
        source
            .functions
            .iter()
            .map(|d| d.name.value.clone())
            .collect()
    };
    let refused: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);
    let seen_by_user = |name: &str, file: usize| {
        !hides(name)
            && !own.iter().any(|f| f.name.value == name)
            && resolve(name, &top) == Some(file)
    };
    // A function the using document neither sees, nor mentions, nor
    // shares a name with another of keeps its name: nothing reaches for
    // it by that name but the documents that see it.
    let unique_unmentioned = |name: &str| {
        declaring.get(name).is_some_and(|files| files.len() == 1)
            && !own.iter().any(|f| f.name.value == name)
            && !hides(name)
            && !mentions(name)
    };
    let internal = |name: &str, file: usize| {
        if seen_by_user(name, file) || unique_unmentioned(name) {
            name.to_owned()
        } else {
            scoped_name(name, &sources[file].name)
        }
    };
    let mut out = Vec::with_capacity(reach.functions.len());
    for (decl, file) in &reach.functions {
        let file = *file;
        let reached = &reaches[&file];
        let declares = |name: &str| {
            declaring
                .get(name)
                .is_some_and(|files| files.contains(&file))
        };
        let rename = |name: &str| -> Option<String> {
            if let Some((_, plain)) = qualified(name) {
                return match resolve_qualified(name, reached) {
                    Ok(target) => Some(internal(plain, target)),
                    Err(_) => {
                        refused.borrow_mut().get_or_insert_with(|| {
                            unreached(sources, reached, &sources[file].name, name, &functions_of)
                        });
                        None
                    }
                };
            }
            let target = if declares(name) {
                file
            } else if hides(name) {
                return None;
            } else if unseen(name, file) {
                return Some(unseen_name(name));
            } else {
                resolve(name, reached)?
            };
            let renamed = internal(name, target);
            (renamed != name).then_some(renamed)
        };
        out.push(decl.renamed(internal(&decl.name.value, file), &rename));
    }
    if let Some(refused) = refused.into_inner() {
        return Err(refused);
    }
    // The using document's own qualified calls, by the name each function
    // it reaches holds.
    let mut calls = Vec::new();
    for name in written {
        match resolve_qualified(name, &top) {
            Ok(target) => {
                let plain = qualified(name).map_or(name.as_str(), |(_, plain)| plain);
                calls.push((name.clone(), internal(plain, target)));
            }
            Err(files) if files.len() > 1 => {
                let names: Vec<&str> = files.iter().map(|&f| sources[f].name.as_str()).collect();
                return Err(format!(
                    "`{name}` is declared by {}, of one package",
                    names.join(" and ")
                ));
            }
            Err(_) => {}
        }
    }
    Ok((out, calls))
}
/// A document's own declarations, put in front of each of its Kernels'
/// own: every Kernel of a document sees what its document declares, with
/// one layout. A Kernel's own function, struct or type is the innermost
/// scope, as over what a document uses ([`resolve`]). The parser gives
/// each Kernel so.
pub(crate) fn document_scope(document: &mut Document) {
    if document.structs.is_empty() && document.types.is_empty() && document.functions.is_empty() {
        return;
    }
    let (structs, types, functions) = (
        document.structs.clone(),
        document.types.clone(),
        document.functions.clone(),
    );
    for_each_kernel(document, &|kernel| {
        put_before(kernel, (&structs, &types, &functions), "document")
    });
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
    for_each_kernel(document, &|kernel| {
        put_before(kernel, (structs, types, functions), SHARED)
    });
}

/// Each name `document` writes qualified by a package written by the name
/// the declaration it reaches holds (see [`Qualified`]).
pub(crate) fn unqualify(document: &mut Document, qualified: &Qualified) {
    if qualified.is_empty() {
        return;
    }
    // A Kernel declaring a name itself holds the shared declaration of
    // that name as `name@shared` ([`resolve`]): what it writes qualified
    // reaches that one.
    let kernel = |kernel: &mut Kernel| {
        let own = |held: String, declared: bool| {
            if declared {
                format!("{held}@{SHARED}")
            } else {
                held
            }
        };
        let functions: Vec<String> = kernel
            .functions
            .iter()
            .map(|f| f.name.value.clone())
            .collect();
        let types: Vec<String> = kernel
            .structs
            .iter()
            .map(|d| d.name.value.clone())
            .chain(kernel.types.iter().map(|d| d.name.value.clone()))
            .collect();
        kernel.rename_all(
            &|name| {
                qualified.of(name, true).map(|held| {
                    let declared = functions.contains(&held);
                    own(held, declared)
                })
            },
            &|name| {
                qualified.of(name, false).map(|held| {
                    let declared = types.contains(&held);
                    own(held, declared)
                })
            },
        );
    };
    for_each_kernel(document, &kernel);
    document.rename_all(&|name| qualified.of(name, true), &|name| {
        qualified.of(name, false)
    });
}

/// The suffix a shared declaration takes in a Kernel declaring its name.
const SHARED: &str = "shared";

/// `change` made to every Kernel of `document`, its inline ones too.
fn for_each_kernel(document: &mut Document, change: &dyn Fn(&mut Kernel)) {
    for kernel in &mut document.kernels {
        change(kernel);
    }
    for node in &mut document.nodes {
        if let NodeKind::Inline(kernel) = &mut node.kind {
            change(kernel);
        }
    }
}

/// `outer`'s declarations put in front of `kernel`'s own; one of a name
/// the Kernel declares itself is renamed `name@suffix` out of its way,
/// and the declarations that name it follow it. Every declaration is
/// put, so a fault in one is found wherever it is written; a function
/// the Kernel does not reach by name ([`crate::reach::reached_by_name`])
/// is put with its name and signature renamed and its body as it was: nothing
/// the Kernel holds calls it, so its body is never lowered, and renaming
/// a body is what putting costs.
fn put_before(kernel: &mut Kernel, outer: (&[StructDecl], &[TypeAlias], &[FnDecl]), suffix: &str) {
    let mut names = Vec::new();
    kernel.names(&mut names);
    let reached: std::collections::HashSet<*const FnDecl> =
        crate::reach::reached_by_name(names, outer)
            .2
            .into_iter()
            .map(std::ptr::from_ref)
            .collect();
    let (structs, types, functions) = outer;
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
            .then(|| format!("{name}@{suffix}"))
    };
    let ty = |name: &str| {
        own_types
            .iter()
            .any(|own| own == name)
            .then(|| format!("{name}@{suffix}"))
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
            if !reached.contains(&std::ptr::from_ref(f)) {
                let mut unreached = f.signature_retyped(&ty);
                if let Some(name) = call(&f.name.value) {
                    unreached.name.value = name;
                }
                return unreached;
            }
            f.renamed(
                call(&f.name.value).unwrap_or_else(|| f.name.value.clone()),
                &call,
            )
            .retyped(&ty)
        })
        .collect();
    shared.append(&mut kernel.functions);
    kernel.functions = shared;
}
