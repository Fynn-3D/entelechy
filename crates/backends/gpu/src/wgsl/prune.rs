//! A shader without the functions its entry point never reaches.
//!
//! Every shader carries the prelude's helpers, used or not; a device's
//! compiler reads, checks and keeps all of them, and a device keeps its
//! reading for as long as the pipeline lives. What the entry point does
//! not reach, directly or through what it calls, goes before the source
//! is given to the device.

use std::collections::{BTreeMap, BTreeSet};

/// A top-level item of a source: its lines, and the function it defines.
struct Item<'a> {
    lines: Vec<&'a str>,
    function: Option<&'a str>,
    entry: bool,
}

/// `source` with only the functions its entry points reach. Every item
/// that is not a function (bindings, structs, aliases) stays, and so
/// does every comment line not right above a function that goes.
pub fn pruned(source: &str) -> String {
    let items = items(source);
    let defined: BTreeMap<&str, usize> = items
        .iter()
        .enumerate()
        .filter_map(|(at, item)| item.function.map(|name| (name, at)))
        .collect();
    let mut reached = BTreeSet::new();
    let mut waiting: Vec<usize> = items
        .iter()
        .enumerate()
        .filter(|(_, item)| item.function.is_none() || item.entry)
        .map(|(at, _)| at)
        .collect();
    while let Some(at) = waiting.pop() {
        if !reached.insert(at) {
            continue;
        }
        for line in &items[at].lines {
            for word in words(code(line)) {
                if let Some(&callee) = defined.get(word)
                    && !reached.contains(&callee)
                {
                    waiting.push(callee);
                }
            }
        }
    }
    let mut out = String::with_capacity(source.len());
    for (at, item) in items.iter().enumerate() {
        if reached.contains(&at) {
            for line in &item.lines {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    out
}

/// The source's top-level items, in order: a function with the comment
/// and attribute lines right above it, and every other line on its own.
fn items(source: &str) -> Vec<Item<'_>> {
    let mut items = Vec::new();
    let mut lead: Vec<&str> = Vec::new();
    let mut lines = source.lines();
    while let Some(line) = lines.next() {
        let trimmed = line.trim_start();
        let at_top = trimmed.len() == line.len();
        if at_top
            && (trimmed.starts_with("//") || trimmed.starts_with('@'))
            && !trimmed.starts_with("@group")
        {
            lead.push(line);
            continue;
        }
        if at_top && trimmed.starts_with("fn ") {
            let name = trimmed[3..]
                .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                .next()
                .unwrap_or_default();
            let entry = lead.iter().any(|lead| lead.trim_start().starts_with('@'));
            let mut item_lines = std::mem::take(&mut lead);
            item_lines.push(line);
            let mut depth = braces(line);
            let mut opened = code(line).contains('{');
            while !(opened && depth <= 0) {
                let Some(next) = lines.next() else { break };
                item_lines.push(next);
                depth += braces(next);
                opened |= code(next).contains('{');
            }
            items.push(Item {
                lines: item_lines,
                function: Some(name),
                entry,
            });
            continue;
        }
        for kept in lead.drain(..).chain(std::iter::once(line)) {
            items.push(Item {
                lines: vec![kept],
                function: None,
                entry: false,
            });
        }
    }
    for kept in lead {
        items.push(Item {
            lines: vec![kept],
            function: None,
            entry: false,
        });
    }
    items
}

/// A line without its comment.
fn code(line: &str) -> &str {
    line.find("//").map_or(line, |at| &line[..at])
}

/// How many more braces a line opens than it closes.
fn braces(line: &str) -> i64 {
    code(line).chars().fold(0, |depth, c| match c {
        '{' => depth + 1,
        '}' => depth - 1,
        _ => depth,
    })
}

/// The identifiers of a line of code.
fn words(code: &str) -> impl Iterator<Item = &str> {
    code.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|word| word.starts_with(|c: char| c.is_alphabetic() || c == '_'))
}

#[cfg(test)]
mod tests {
    use super::pruned;

    const SOURCE: &str = "@group(0) @binding(0) var<uniform> ctx: array<vec4<f32>, 4>;
struct Pair { a: f32, b: f32 }
// Used through `twice`.
fn half(x: f32) -> f32 {
    return x * 0.5;
}
fn twice(x: f32) -> f32 { return half(x) * 4.0; }
// Never called.
fn unused(x: f32) -> f32 {
    if (x > 0.0) { return x; }
    return twice(x); // calls back, but nothing reaches it
}
fn fail(code: u32) {}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = Pair(twice(ctx[0].x), 1.0);
}
";

    #[test]
    fn what_the_entry_point_reaches_stays_and_the_rest_goes() {
        let out = pruned(SOURCE);
        assert!(
            out.contains("fn half(") && out.contains("fn twice("),
            "{out}"
        );
        assert!(out.contains("// Used through `twice`."), "{out}");
        assert!(
            !out.contains("fn unused(") && !out.contains("// Never called."),
            "{out}"
        );
        assert!(
            !out.contains("fn fail("),
            "a one-line function nothing calls goes: {out}"
        );
        assert!(
            out.contains("@compute @workgroup_size(8, 8, 1)\nfn main("),
            "{out}"
        );
        assert!(
            out.contains("struct Pair") && out.contains("@group(0) @binding(0)"),
            "{out}"
        );
        assert!(
            out.trim_end().ends_with('}'),
            "the entry point is whole: {out}"
        );
    }

    #[test]
    fn a_source_whose_functions_are_all_reached_is_unchanged() {
        let source = "fn a() -> f32 { return 1.0; }\n@compute @workgroup_size(1)\nfn main() {\n    let x = a();\n}\n";
        assert_eq!(pruned(source), source);
    }
}
