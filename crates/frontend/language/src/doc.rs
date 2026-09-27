//! What a document says about what it declares: the `///` lines directly
//! above a declaration are its doc. They are comments to the grammar: the
//! text, not the parse, holds them.

/// The `///` lines directly above line `line` (from 1), joined; `None`
/// when there are none. Blank lines and `//` lines end the doc.
fn doc_above(text: &str, line: u32) -> Option<String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut at = (line as usize).checked_sub(1)?;
    let mut doc = Vec::new();
    while at > 0 {
        at -= 1;
        let Some(rest) = lines
            .get(at)
            .and_then(|l| l.trim_start().strip_prefix("///"))
        else {
            break;
        };
        doc.push(rest.strip_prefix(' ').unwrap_or(rest).trim_end());
    }
    joined(doc.into_iter().rev())
}

/// `range` widened back over the `///` lines directly above it, so a
/// declaration cut out of its document carries its doc.
pub fn with_doc(text: &str, range: std::ops::Range<usize>) -> std::ops::Range<usize> {
    let line_start = |at: usize| text[..at].rfind('\n').map_or(0, |i| i + 1);
    let mut start = line_start(range.start);
    // A declaration after other text on its line (an inline Kernel) has
    // no line of its own to be documented above.
    if !text[start..range.start].trim().is_empty() {
        return range;
    }
    while start > 0 {
        let above = line_start(start - 1);
        if !text[above..start].trim_start().starts_with("///") {
            break;
        }
        start = above;
    }
    start..range.end
}

/// What a Kernel's source says of it and of each of its ports and params.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelDocs {
    /// The Kernel's own.
    pub doc: Option<String>,
    /// Each documented input, output, memory or param, by name.
    pub ports: Vec<(String, String)>,
}

/// The docs of the first Kernel in `source`; empty when it does not
/// parse.
pub fn kernel_docs(source: &str) -> KernelDocs {
    let Ok(document) = crate::parse_document(source) else {
        return KernelDocs::default();
    };
    let Some(kernel) = document.kernels.first() else {
        return KernelDocs::default();
    };
    let line = source[..kernel.range.start].matches('\n').count() as u32 + 1;
    let mut ports = Vec::new();
    let named = kernel
        .inputs
        .iter()
        .chain(&kernel.outputs)
        .chain(&kernel.memories)
        .map(|port| &port.name)
        .chain(kernel.params.iter().map(|param| &param.name));
    for name in named {
        if let Some(doc) = doc_above(source, name.span.line) {
            ports.push((name.value.clone(), doc));
        }
    }
    KernelDocs {
        doc: doc_above(source, line),
        ports,
    }
}

/// Doc lines as one text: a paragraph reads as one line, an empty line
/// parts paragraphs, and an indented line (code) keeps its own.
fn joined<'a>(lines: impl Iterator<Item = &'a str>) -> Option<String> {
    let mut out = String::new();
    let mut code = false;
    for line in lines {
        if line.is_empty() {
            out.push_str("\n\n");
            code = false;
        } else if line.starts_with(' ') {
            // An indented line is code: it keeps its line.
            out.push('\n');
            out.push_str(line);
            code = true;
        } else {
            if code {
                out.push('\n');
            } else if !out.is_empty() && !out.ends_with('\n') {
                out.push(' ');
            }
            out.push_str(line);
            code = false;
        }
    }
    let out = out.trim().to_owned();
    (!out.is_empty()).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_declaration_owns_the_doc_lines_directly_above_it() {
        let text = "//! A preset\n//! that blurs.\n\n// not a doc\n/// Rows kept.\n///\n/// Second paragraph.\nkernel a.b@1 {\n    /// The flags.\n    input flags: Rows<Int>;\n\n    input bare: Rows<Int>;\n}\n";
        assert_eq!(
            doc_above(text, 8).as_deref(),
            Some("Rows kept.\n\nSecond paragraph.")
        );
        assert_eq!(doc_above(text, 10).as_deref(), Some("The flags."));
        assert_eq!(doc_above(text, 12), None);
        assert_eq!(doc_above(text, 1), None);
    }

    #[test]
    fn a_kernel_cut_from_its_document_carries_its_docs() {
        let text = "// file\n\n/// Keeps rows.\nkernel a.b@1 {\n    requires { field.read; field.write; }\n    /// The flags.\n    input flags: Rows<Int>;\n    /// How far.\n    param by: Int = 1;\n    output result: Rows<Int>;\n    compute { return sample(flags) + by; }\n}\n";
        let kernel = crate::parse_document(text).unwrap().kernels[0]
            .range
            .clone();
        let cut = &text[with_doc(text, kernel)];
        assert!(cut.starts_with("/// Keeps rows.\nkernel"), "{cut}");
        let inline = "/// Not the node's.\nnode n: kernel { }";
        let at = inline.find("kernel").unwrap();
        assert_eq!(with_doc(inline, at..inline.len()), at..inline.len());
        let docs = kernel_docs(cut);
        assert_eq!(docs.doc.as_deref(), Some("Keeps rows."));
        assert_eq!(
            docs.ports,
            [
                ("flags".to_owned(), "The flags.".to_owned()),
                ("by".to_owned(), "How far.".to_owned())
            ]
        );
    }
}
