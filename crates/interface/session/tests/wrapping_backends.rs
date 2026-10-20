//! A backend wrapping another passes every method of `Backend` through or
//! says what it does instead: a method left out falls to the trait's
//! default, which answers as a backend with no device would, and nothing
//! fails to compile.

const TRAIT: &str = include_str!("../../../core/rules/src/backend.rs");
const WRAPPERS: [(&str, &str); 2] = [
    ("Remembered", include_str!("../src/remember.rs")),
    ("Apart", include_str!("../src/apart.rs")),
];

/// The names of the methods a block of Rust from `opening` to its closing
/// `}` at the start of a line declares at its first indent.
fn methods(source: &str, opening: &str) -> Vec<String> {
    let start = source
        .find(opening)
        .unwrap_or_else(|| panic!("`{opening}` is not in the source"));
    let block = &source[start..];
    let block = &block[..block.find("\n}\n").expect("the block closes")];
    let mut names: Vec<String> = block
        .lines()
        .filter_map(|line| line.strip_prefix("    fn "))
        .map(|rest| {
            rest.split(|c: char| !(c.is_alphanumeric() || c == '_'))
                .next()
                .unwrap_or_default()
                .to_owned()
        })
        .collect();
    names.sort();
    names
}

#[test]
fn every_wrapping_backend_passes_every_method_through() {
    let declared = methods(TRAIT, "pub trait Backend");
    assert!(declared.len() > 20, "{declared:?}");
    for (wrapper, source) in WRAPPERS {
        let written = methods(source, &format!("impl Backend for {wrapper} {{"));
        let missing: Vec<&String> = declared
            .iter()
            .filter(|name| !written.contains(name))
            .collect();
        assert!(
            missing.is_empty(),
            "`{wrapper}` leaves {missing:?} to the trait's defaults"
        );
    }
}
