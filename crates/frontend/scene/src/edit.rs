//! Edits of a document's text by what it declares: a statement by kind
//! and name, or a node's member by node and member, is replaced, added
//! or removed, changing only its bytes. The caller checks the result.

use std::fmt;
use std::ops::Range;

use orla_language::{Outline, Statement, outline};

pub use orla_language::StatementKind;

/// What an edit names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    /// A top-level statement by kind and declared name (a Kernel's id as
    /// written, a `use`'s path).
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

/// One edit: the target's new text, the whole statement or member
/// declaring exactly what the target names, or `None` to remove it. A
/// new target is added after the last of its kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edit {
    /// What it changes.
    pub target: Target,
    /// Its new text, or `None` to remove it.
    pub text: Option<String>,
}

/// Why an edit failed, and which one.
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

/// `document` with `edits` made in order, all or none: the error names
/// the first that fails.
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
                    replace(text, statement.range.clone(), fragment)
                }
                (Some(statement), None) => remove(text, statement.range.clone(), true),
                (None, Some((fragment, _))) => add_statement(text, &outline, *kind, fragment),
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
                (Some(at), Some(fragment)) => replace(text, at.range.clone(), fragment),
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
/// declaring what `target` names, and whether it ends in a comment.
fn declares<'a>(fragment: &'a str, target: &Target) -> Result<(&'a str, bool), String> {
    let fragment = fragment.trim();
    match target {
        Target::Statement { kind, name } => {
            let outline = outline(fragment)
                .map_err(|e| format!("the text for {target} does not parse: {e}"))?;
            match outline.statements.as_slice() {
                [one] if one.kind == *kind && &one.name == name => {
                    Ok((fragment, !fragment[one.range.end..].trim().is_empty()))
                }
                [one] => Err(format!(
                    "the text for {target} declares {} `{}`; an edit changes the statement it names: to rename it, remove it (`text`: null) and add the new one in another edit",
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
            // A member parses inside a block made for it, as its one member.
            let probe = format!("node probe: probe.probe@1 {{\n{fragment}\n}}");
            let outline = outline(&probe)
                .map_err(|e| format!("the text for {target} does not parse: {e}"))?;
            let members = match outline.statements.as_slice() {
                [one] => one.block.as_ref().map(|b| b.members.as_slice()),
                _ => None,
            };
            match members {
                Some([one]) if &one.name == member => {
                    let rest = &probe[one.range.end..probe.len() - 1];
                    Ok((fragment, !rest.trim().is_empty()))
                }
                Some([one]) => Err(format!(
                    "the text for {target} sets `{}`; an edit changes the member it names: to rename `{member}`, remove it (`text`: null) and add `{}` in another edit",
                    one.name, one.name
                )),
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

/// `range` replaced by a fragment, which, ending in a comment, ends its
/// line before what followed `range` there.
fn replace(text: &str, range: Range<usize>, (fragment, comment): (&str, bool)) -> String {
    let (with, end) = ended(text, range.start, range.end, fragment, comment);
    splice(text, range.start..end, &with)
}

/// `fragment` put in at `start` before what is at `end`, and where what
/// follows it then begins: ending in a comment with more on the line
/// after `end`, a new line indented as the line at `start` follows it,
/// and the spaces after `end` go.
fn ended(text: &str, start: usize, end: usize, fragment: &str, comment: bool) -> (String, usize) {
    if !comment || text[end..line_end(text, end)].trim().is_empty() {
        return (fragment.to_owned(), end);
    }
    let line = text[..start].rfind('\n').map_or(0, |i| i + 1);
    let indent: String = text[line..]
        .chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .collect();
    let spaces = text[end..].len() - text[end..].trim_start_matches([' ', '\t']).len();
    (format!("{fragment}\n{indent}"), end + spaces)
}

fn splice(text: &str, range: Range<usize>, with: &str) -> String {
    let mut out = String::with_capacity(text.len() + with.len());
    out.push_str(&text[..range.start]);
    out.push_str(with);
    out.push_str(&text[range.end..]);
    out
}

/// Takes `range` out. Alone on its lines, the lines go too, with a
/// trailing comment and, for a statement, the comment lines right above
/// it, and one blank line of two it stood between; sharing a line, the
/// spaces before it go.
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
    // Between two blank lines, one of them goes with it.
    let mut end = line_end;
    let blank_before = start == 0 || text[..start].ends_with("\n\n");
    if blank_before && text[end..].starts_with('\n') {
        end += 1;
    }
    splice(text, start..end, "")
}

/// The order of statement kinds, for placing the first of a kind.
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
        StatementKind::Require => 8,
        StatementKind::Node => 9,
        StatementKind::Output => 10,
    }
}

/// Adds a statement after the last of its kind (with a blank line
/// between when either spans several lines), or else after the last of
/// the kinds before it with a blank line between, or else first.
fn add_statement(text: &str, outline: &Outline, kind: StatementKind, fragment: &str) -> String {
    let same = outline.statements.iter().rev().find(|s| s.kind == kind);
    let before = outline
        .statements
        .iter()
        .rev()
        .find(|s| order(s.kind) < order(kind));
    // One-line statements of a kind stay together; a statement of
    // several lines, or one after it, stands apart by a blank line.
    let lines = |text: &str| text.trim_end().contains('\n');
    let (after, gap) = match (same, before) {
        (Some(same), _) if lines(&text[same.range.clone()]) || lines(fragment) => {
            (Some(same), "\n")
        }
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
            // First of all: a blank line follows it.
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
/// on the block's line when the members share it, else on its own line
/// indented as the last member.
fn add_member(
    text: &str,
    block: &orla_language::NodeBlock,
    (fragment, comment): (&str, bool),
) -> String {
    let same_line = |a: usize, b: usize| !text[a..b].contains('\n');
    match block.members.last() {
        Some(last) if same_line(block.open, last.range.start) => {
            let at = last.range.end;
            let (with, end) = ended(text, at, at, fragment, comment);
            splice(text, at..end, &format!(" {with}"))
        }
        Some(last) => {
            let line = text[..last.range.start].rfind('\n').map_or(0, |i| i + 1);
            let indent = &text[line..last.range.start];
            let at = line_end(text, last.range.end);
            splice(text, at..at, &format!("\n{indent}{fragment}"))
        }
        None if same_line(block.open, block.close) => {
            let (with, _) = ended(text, block.open, block.close, fragment, comment);
            let gap = if with.ends_with(char::is_whitespace) {
                ""
            } else {
                " "
            };
            splice(text, block.open + 1..block.close, &format!(" {with}{gap}"))
        }
        None => {
            let line = text[..block.close].rfind('\n').map_or(0, |i| i + 1);
            let indent = &text[line..block.close];
            let at = line_end(text, block.open);
            splice(text, at..at, &format!("\n{indent}    {fragment}"))
        }
    }
}

/// The end of the line `at` is on, before its newline, past any
/// trailing comment.
fn line_end(text: &str, at: usize) -> usize {
    text[at..].find('\n').map_or(text.len(), |i| at + i)
}
