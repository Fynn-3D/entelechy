//! `include/orla.h` is the C ABI's contract, and what a C caller
//! compiles against; the Rust is what it links to. They are held to
//! each other here: the same functions, the same version, and every
//! struct the header lays out laid out the same, field by field. A change
//! to either side that the other does not make fails, so the ABI changes
//! only on purpose.

const HEADER: &str = include_str!("../include/orla.h");
const SOURCE: &str = include_str!("../src/lib.rs");

/// The header without its comments, which name functions in prose.
fn header_code() -> String {
    let mut code = String::new();
    let mut rest = HEADER;
    while let Some(start) = rest.find("/*") {
        code.push_str(&rest[..start]);
        rest = rest[start..]
            .find("*/")
            .map_or("", |end| &rest[start + end + 2..]);
    }
    code.push_str(rest);
    code
}

/// Every `orla_` name followed by `(` in `text` and preceded by
/// `before`, in order, once each.
fn functions(text: &str, before: &str) -> Vec<String> {
    let mut names = Vec::new();
    for (at, _) in text.match_indices("orla_") {
        let head = &text[..at];
        let bounded = !head
            .chars()
            .next_back()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
        if !bounded || !head.ends_with(before) {
            continue;
        }
        let tail = &text[at..];
        let name: String = tail
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if name.starts_with("orla_")
            && tail[name.len()..].starts_with('(')
            && !names.contains(&name)
        {
            names.push(name);
        }
    }
    names
}

/// A C field type as the Rust spells it.
fn rust_of(c: &str) -> String {
    match c {
        "float" => "f32",
        "uint32_t" => "u32",
        "const uint32_t *" => "*const u32",
        "const char *" => "*const c_char",
        other => panic!("the header uses `{other}`, which this test does not map"),
    }
    .to_owned()
}

/// The fields of the header's `typedef struct <name> { ... }`, as
/// `(name, Rust type)` in order.
fn header_fields(name: &str) -> Vec<(String, String)> {
    let code = header_code();
    let open = format!("typedef struct {name} {{");
    let start = code
        .find(&open)
        .unwrap_or_else(|| panic!("no struct {name} in the header"))
        + open.len();
    let body = &code[start..start + code[start..].find('}').unwrap()];
    body.split(';')
        .map(str::trim)
        .filter(|field| !field.is_empty())
        .map(|field| {
            let (ty, declarator) = field.rsplit_once([' ', '*']).unwrap();
            let pointer = field[ty.len()..].starts_with('*');
            let ty = if pointer {
                format!("{} *", ty.trim())
            } else {
                ty.trim().to_owned()
            };
            match declarator.split_once('[') {
                Some((field, count)) => (
                    field.to_owned(),
                    format!("[{}; {}]", rust_of(&ty), count.trim_end_matches(']')),
                ),
                None => (declarator.to_owned(), rust_of(&ty)),
            }
        })
        .collect()
}

/// The fields of the Rust `#[repr(C)] pub struct <name>`, in order.
fn rust_fields(name: &str) -> Vec<(String, String)> {
    let open = format!("#[repr(C)]\n#[derive(Clone, Copy, Debug)]\npub struct {name} {{");
    let start = SOURCE
        .find(&open)
        .unwrap_or_else(|| panic!("no #[repr(C)] struct {name} in the source"))
        + open.len();
    let body = &SOURCE[start..start + SOURCE[start..].find('}').unwrap()];
    body.lines()
        .map(str::trim)
        .filter_map(|line| line.strip_prefix("pub "))
        .map(|line| {
            let (field, ty) = line.trim_end_matches(',').split_once(": ").unwrap();
            (field.to_owned(), ty.to_owned())
        })
        .collect()
}

#[test]
fn the_header_declares_what_the_library_exports() {
    let declared = functions(&header_code(), "");
    let mut exported = functions(SOURCE, "extern \"C\" fn ");
    exported.sort();
    let mut sorted = declared.clone();
    sorted.sort();
    assert_eq!(sorted, exported, "the header and the exports differ");
    assert!(declared.len() >= 9, "found only {declared:?}");
}

#[test]
fn the_header_and_the_library_are_the_same_version() {
    let line = HEADER
        .lines()
        .find_map(|line| line.strip_prefix("#define ORLA_ABI_VERSION "))
        .expect("the header defines ORLA_ABI_VERSION");
    assert_eq!(line.trim().parse::<u32>().unwrap(), orla_ffi::ABI_VERSION);
}

#[test]
fn every_struct_the_header_lays_out_is_laid_out_alike() {
    for name in ["OrlaInput", "OrlaField"] {
        assert_eq!(
            header_fields(name),
            rust_fields(name),
            "{name} is laid out differently in C and in Rust"
        );
    }
}
