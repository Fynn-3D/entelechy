//! Edits of a document's text by what it declares: a statement named by
//! its kind and name, or a member of a node's block by the node and the
//! member, is replaced, added, or removed, and only its bytes change.
//! Everything else stays as it was written, comments and layout
//! included. What the result means is not checked here; the caller
//! checks the whole document, as it checks any text.

use std::fmt;
use std::ops::Range;

use orla_language::{Outline, Statement, outline};

pub use orla_language::StatementKind;

/// What an edit names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    /// A top-level statement: its kind and the name it declares (a
    /// Kernel's id as written, a `use`'s path).
    Statement {
        /// Its kind.
        kind: StatementKind,
        /// Its name.
        name: String,
    },
    /// A member of a node's block, by the parameter or port it sets.
    Member {
        /// The node.
        node: String,
        /// The member.
        member: String,
    },
}

/// One edit: the text that stands for its target from now on, or `None`
/// to remove it. The text is the whole statement or member, declaring
/// exactly what the target names; a target not yet written is added
/// after the last of its kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edit {
    /// What it changes.
    pub target: Target,
    /// Its new text, or `None` to remove it.
    pub text: Option<String>,
}

/// Why an edit could not be made, and which one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditError {
    /// The edit's place in the list, from 0.
    pub index: usize,
    /// What is wrong.
    pub message: String,
}

impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "edit {}: {}", self.index, self.message)
    }
}

impl std::error::Error for EditError {}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Statement { kind, name } => write!(f, "{} `{name}`", kind.keyword()),
            Self::Member { node, member } => write!(f, "`{member}` of node `{node}`"),
        }
    }
}

/// `document` with `edits` made in order, each on the text the one
/// before left: all of them, or an error naming the first that cannot
/// be made.
pub fn edit(document: &str, edits: &[Edit]) -> Result<String, EditError> {
    let mut text = document.to_owned();
    for (index, one) in edits.iter().enumerate() {
        text = apply(&text, one).map_err(|message| EditError { index, message })?;
    }
    Ok(text)
}

fn apply(text: &str, edit: &Edit) -> Result<String, String> {
    let outline = outline(text).map_err(|e| format!("the document does not parse: {e}"))?;
    let fragment = match &edit.text {
        Some(fragment) => Some(declares(fragment, &edit.target)?),
        None => None,
    };
    match &edit.target {
        Target::Statement { kind, name } => {
            let at = find(&outline, *kind, name)?;
            Ok(match (at, fragment) {
                (Some(statement), Some(fragment)) => {
                    splice(text, statement.range.clone(), fragment)
                }
                (Some(statement), None) => remove(text, statement.range.clone(), true),
                (None, Some(fragment)) => add_statement(text, &outline, *kind, fragment),
                (None, None) => {
                    return Err(format!("there is no {} to remove", edit.target));
                }
            })
        }
        Target::Member { node, member } => {
            let Some(statement) = find(&outline, StatementKind::Node, node)? else {
                return Err(format!("there is no node `{node}`"));
            };
            let Some(block) = &statement.block else {
                return Err(format!(
                    "node `{node}` is a Kernel written in place, with no members: replace the node whole"
                ));
            };
            let mut named = block.members.iter().filter(|m| &m.name == member);
            let at = named.next();
            if named.next().is_some() {
                return Err(format!("node `{node}` has more than one `{member}`"));
            }
            Ok(match (at, fragment) {
                (Some(at), Some(fragment)) => splice(text, at.range.clone(), fragment),
                (Some(at), None) => remove(text, at.range.clone(), false),
                (None, Some(fragment)) => add_member(text, block, fragment),
                (None, None) => {
                    return Err(format!("there is no {} to remove", edit.target));
                }
            })
        }
    }
}

/// The fragment, trimmed, when it is exactly one statement or member
/// declaring what `target` names.
fn declares<'a>(fragment: &'a str, target: &Target) -> Result<&'a str, String> {
    let fragment = fragment.trim();
    match target {
        Target::Statement { kind, name } => {
            let outline = outline(fragment)
                .map_err(|e| format!("the text for {target} does not parse: {e}"))?;
            match outline.statements.as_slice() {
                [one] if one.kind == *kind && &one.name == name => Ok(fragment),
                [one] => Err(format!(
                    "the text for {target} declares {} `{}`",
                    one.kind.keyword(),
                    one.name
                )),
                statements => Err(format!(
                    "the text for {target} is {} statements, not one",
                    statements.len()
                )),
            }
        }
        Target::Member { member, .. } => {
            // A member parses inside a block: one is made for it, and it
            // must stay one member of that one block.
            let probe = format!("node probe: probe.probe@1 {{\n{fragment}\n}}");
            let outline = outline(&probe)
                .map_err(|e| format!("the text for {target} does not parse: {e}"))?;
            let members = match outline.statements.as_slice() {
                [one] => one.block.as_ref().map(|b| b.members.as_slice()),
                _ => None,
            };
            match members {
                Some([one]) if &one.name == member => Ok(fragment),
                Some([one]) => Err(format!("the text for {target} sets `{}`", one.name)),
                _ => Err(format!("the text for {target} is not one member")),
            }
        }
    }
}

/// The one statement of `kind` named `name`, if it is written.
fn find<'a>(
    outline: &'a Outline,
    kind: StatementKind,
    name: &str,
) -> Result<Option<&'a Statement>, String> {
    let mut named = outline
        .statements
        .iter()
        .filter(|s| s.kind == kind && s.name == name);
    let at = named.next();
    if named.next().is_some() {
        return Err(format!(
            "the document has more than one {} `{name}`",
            kind.keyword()
        ));
    }
    Ok(at)
}

fn splice(text: &str, range: Range<usize>, with: &str) -> String {
    let mut out = String::with_capacity(text.len() + with.len());
    out.push_str(&text[..range.start]);
    out.push_str(with);
    out.push_str(&text[range.end..]);
    out
}

/// Takes `range` out. Alone on its lines, the lines go with it, a
/// comment after it on its last line too, and for a statement the
/// comment lines right above it, which were about it; sharing a line,
/// the spaces before it go.
fn remove(text: &str, range: Range<usize>, comments: bool) -> String {
    let line_start = text[..range.start].rfind('\n').map_or(0, |i| i + 1);
    let line_end = text[range.end..]
        .find('\n')
        .map_or(text.len(), |i| range.end + i + 1);
    let after = text[range.end..line_end].trim();
    let alone = text[line_start..range.start].trim().is_empty()
        && (after.is_empty() || after.starts_with("//"));
    if !alone {
        let start = text[..range.start].trim_end_matches([' ', '\t']).len();
        return splice(text, start..range.end, "");
    }
    let mut start = line_start;
    if comments {
        while start > 0 {
            let above = text[..start - 1].rfind('\n').map_or(0, |i| i + 1);
            if !text[above..start].trim_start().starts_with("//") {
                break;
            }
            start = above;
        }
    }
    splice(text, start..line_end, "")
}

/// Where statements of each kind go when the document has none of that
/// kind yet: after the last of the kinds that come before it.
fn order(kind: StatementKind) -> u8 {
    match kind {
        StatementKind::Package => 0,
        StatementKind::Use => 1,
        StatementKind::Struct => 2,
        StatementKind::Type => 3,
        StatementKind::Fn => 4,
        StatementKind::Kernel => 5,
        StatementKind::Input => 6,
        StatementKind::Param => 7,
        StatementKind::Node => 8,
        StatementKind::Output => 9,
    }
}

/// Adds a statement after the last of its kind; the first of its kind
/// after the last of the kinds before it, a blank line between; the
/// first of all at the end.
fn add_statement(text: &str, outline: &Outline, kind: StatementKind, fragment: &str) -> String {
    let same = outline.statements.iter().rev().find(|s| s.kind == kind);
    let before = outline
        .statements
        .iter()
        .rev()
        .find(|s| order(s.kind) < order(kind));
    let (after, gap) = match (same, before) {
        (Some(same), _) => (Some(same), ""),
        (None, Some(before)) => (Some(before), "\n"),
        (None, None) => (None, ""),
    };
    match after {
        Some(after) => {
            let at = line_end(text, after.range.end);
            splice(text, at..at, &format!("\n{gap}{fragment}"))
        }
        None => match outline.statements.first() {
            // Nothing comes before it: it goes first, a blank line
            // between it and what was first.
            Some(first) => {
                let at = text[..first.range.start].rfind('\n').map_or(0, |i| i + 1);
                splice(text, at..at, &format!("{fragment}\n\n"))
            }
            None => {
                let sep = if text.is_empty() || text.ends_with('\n') {
                    ""
                } else {
                    "\n"
                };
                format!("{text}{sep}{fragment}\n")
            }
        },
    }
}

/// Adds a member at the end of a node's block, laid out as the block is:
/// on the line of the last member when they share the block's line, on
/// a line of its own, indented as the last member, when not.
fn add_member(text: &str, block: &orla_language::NodeBlock, fragment: &str) -> String {
    let same_line = |a: usize, b: usize| !text[a..b].contains('\n');
    match block.members.last() {
        Some(last) if same_line(block.open, last.range.start) => splice(
            text,
            last.range.end..last.range.end,
            &format!(" {fragment}"),
        ),
        Some(last) => {
            let line = text[..last.range.start].rfind('\n').map_or(0, |i| i + 1);
            let indent = &text[line..last.range.start];
            let at = line_end(text, last.range.end);
            splice(text, at..at, &format!("\n{indent}{fragment}"))
        }
        None if same_line(block.open, block.close) => {
            splice(text, block.open + 1..block.close, &format!(" {fragment} "))
        }
        None => {
            let line = text[..block.close].rfind('\n').map_or(0, |i| i + 1);
            let indent = &text[line..block.close];
            let at = line_end(text, block.open);
            splice(text, at..at, &format!("\n{indent}    {fragment}"))
        }
    }
}

/// The end of the line `at` is on, before its newline: where something
/// added after a statement begins, past whatever comment follows it.
fn line_end(text: &str, at: usize) -> usize {
    text[at..].find('\n').map_or(text.len(), |i| at + i)
}
