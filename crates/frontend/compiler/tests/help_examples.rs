//! What help shows is written as the language takes it: every piece of a
//! form's shape that declares something whole (it begins with a keyword
//! and elides nothing with `...`) parses where such a piece is written: a
//! document, a Kernel's members, a Kernel's body, or a node's members.

use orla_compiler::help;
use orla_language::parse_document;

/// The words a whole declaration begins with.
const DECLARES: [&str; 16] = [
    "struct", "type", "param", "input", "output", "use", "let", "node", "kernel", "package",
    "requires", "budget", "memory", "async", "times", "fn",
];

/// Whether `piece` parses in one of the places a piece is written.
fn parses(piece: &str) -> bool {
    let places = [
        piece.to_owned(),
        format!("kernel t.k@1 {{ {piece} output probe: Rows<Float>; compute {{ return 0.0; }} }}"),
        format!(
            "kernel t.k@1 {{ requires {{ field.write; }} {piece} output probe: Rows<Float>; compute {{ return 0.0; }} }}"
        ),
        format!(
            "kernel t.k@1 {{ requires {{ field.write; }} output probe: Rows<Float>; compute {{ {piece} return 0.0; }} }}"
        ),
        format!("node probe: t.k@1 {{ {piece} }}"),
    ];
    places.iter().any(|text| parse_document(text).is_ok())
}

#[test]
fn every_whole_declaration_help_shows_parses() {
    let mut wrong = Vec::new();
    for form in help::forms() {
        for piece in form.shape.split("   ").map(str::trim) {
            let first = piece.split(|c: char| !c.is_alphanumeric()).next();
            let whole =
                first.is_some_and(|word| DECLARES.contains(&word)) && !piece.contains("...");
            if whole && !parses(piece) {
                wrong.push(format!("form `{}`: `{piece}`", form.name));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "help shows what the language does not take:\n{}",
        wrong.join("\n")
    );
}
