//! One rule, one owner: each rule's act (the cut, the write) is done in
//! its owner's files and in no other crate source. Each rule names the
//! commit that gave it one owner, and is kept only while it finds that
//! commit's second owner in the source before it. See `docs/DECISIONS.md`,
//! *A rule that had two owners is held to one*.

use std::path::{Path, PathBuf};

struct Rule {
    /// The rule, as a sentence.
    rule: &'static str,
    /// Where the rule is followed, for the failure to point to.
    owner: &'static str,
    /// The commit that gave the rule its one owner.
    since: &'static str,
    /// The directories, from the repository root, whose sources are read.
    within: &'static [&'static str],
    /// The files, from the repository root, that may do the act.
    owners: &'static [&'static str],
    /// The act, as written in code or in a string.
    acts: &'static [&'static str],
}

const RULES: &[Rule] = &[
    Rule {
        rule: "`node.port` is cut in one place, at the last dot",
        owner: "orla_graph::split_port",
        since: "d000b28",
        within: &["crates"],
        owners: &["crates/core/graph/src/lib.rs"],
        acts: &[
            "split_once('.')",
            "rsplit_once('.')",
            "split_once(\".\")",
            "rsplit_once(\".\")",
        ],
    },
    Rule {
        rule: "a scene output is read by its name, or the first declared, in one place",
        owner: "`Session::output`, over `Graph::primary_output`",
        since: "3e4d345",
        within: &["crates/interface"],
        owners: &["crates/interface/session/src/session.rs"],
        acts: &["outputs.first()"],
    },
    Rule {
        // `Session::store` is private to the crate; this rule holds a write
        // that goes around it.
        rule: "a session writes only through its store, where its allowance is asked, \
               and the codecs it writes with make bytes, not files",
        owner: "`Session::write_file`, `Session::write_scene`, `Session::save`, \
                `Session::export`, or a scene's `to` writes",
        since: "8248001",
        within: &[
            "crates/interface/session/src",
            "crates/interface/protocol/src",
            "crates/interface/ffi/src",
            "crates/interface/player/src",
            "crates/interface/formats/src",
        ],
        owners: &["crates/interface/session/src/store.rs"],
        acts: &[
            "fs::write",
            "fs::rename",
            "fs::copy",
            "fs::create_dir",
            "fs::remove",
            "File::create",
            "OpenOptions",
        ],
    },
];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// Every Rust source under `dir`, except a crate's `tests` directories.
fn sources(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries {
        let path = entry.unwrap().path();
        if path.is_dir() {
            if path.file_name().is_none_or(|name| name != "tests") {
                sources(&path, found);
            }
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            found.push(path);
        }
    }
}

/// A source's non-test code: comments and `#[cfg(test)]` items removed.
/// String and character literals are kept and read as such.
fn code(text: &str) -> String {
    // Each character, and whether it is code (outside a literal).
    let mut kept: Vec<(char, bool)> = Vec::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        if c == '/' && next == Some('/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '/' && next == Some('*') {
            let mut depth = 0;
            while i < chars.len() {
                if chars[i] == '/' && chars.get(i + 1) == Some(&'*') {
                    depth += 1;
                    i += 2;
                } else if chars[i] == '*' && chars.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    i += 1;
                }
            }
            continue;
        }
        // A raw string: `r"…"`, `r#"…"#`, `br#"…"#`.
        if c == 'r'
            && !chars[..i]
                .last()
                .is_some_and(|p| p.is_alphanumeric() || *p == '_')
        {
            let mut j = i + 1;
            while chars.get(j) == Some(&'#') {
                j += 1;
            }
            if chars.get(j) == Some(&'"') {
                let hashes = j - i - 1;
                kept.extend(chars[i..=j].iter().map(|c| (*c, false)));
                i = j + 1;
                while i < chars.len() {
                    let closes =
                        chars[i] == '"' && (1..=hashes).all(|k| chars.get(i + k) == Some(&'#'));
                    kept.push((chars[i], false));
                    i += 1;
                    if closes {
                        kept.extend(chars[i..i + hashes].iter().map(|c| (*c, false)));
                        i += hashes;
                        break;
                    }
                }
                continue;
            }
        }
        if c == '"' {
            kept.push((c, false));
            i += 1;
            while i < chars.len() {
                kept.push((chars[i], false));
                if chars[i] == '\\' {
                    if let Some(escaped) = chars.get(i + 1) {
                        kept.push((*escaped, false));
                    }
                    i += 2;
                    continue;
                }
                i += 1;
                if chars[i - 1] == '"' {
                    break;
                }
            }
            continue;
        }
        // A character literal (`'.'`, `'\n'`, `'{'`); otherwise a lifetime.
        if c == '\'' {
            let end = if next == Some('\\') {
                chars[i + 3..]
                    .iter()
                    .position(|c| *c == '\'')
                    .map(|at| i + 3 + at)
            } else if chars.get(i + 2) == Some(&'\'') {
                Some(i + 2)
            } else {
                None
            };
            if let Some(end) = end {
                kept.extend(chars[i..=end].iter().map(|c| (*c, false)));
                i = end + 1;
                continue;
            }
        }
        kept.push((c, true));
        i += 1;
    }
    // Drop each `#[cfg(test)]` item: to the end of its block, or of its
    // statement when it has none.
    let text: String = kept.iter().map(|(c, _)| *c).collect();
    let marker = "#[cfg(test)]";
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    while let Some(found) = text[at..].find(marker) {
        let start = at + found;
        out.push_str(&text[at..start]);
        let mut j = text[..start].chars().count() + marker.chars().count();
        let mut depth = 0;
        while j < kept.len() {
            let (c, is_code) = kept[j];
            j += 1;
            if !is_code {
                continue;
            }
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                ';' if depth == 0 => break,
                _ => {}
            }
        }
        at = kept[..j].iter().map(|(c, _)| c.len_utf8()).sum();
    }
    out.push_str(&text[at..]);
    out
}

#[test]
fn every_rule_has_its_one_owner() {
    let root = root();
    let mut wrong = Vec::new();
    for rule in RULES {
        for owner in rule.owners {
            assert!(
                root.join(owner).is_file(),
                "{owner}, the owner of \"{}\", is gone; move the rule with it",
                rule.rule
            );
        }
        let mut files = Vec::new();
        for dir in rule.within {
            sources(&root.join(dir), &mut files);
        }
        assert!(!files.is_empty(), "\"{}\" reads no source", rule.rule);
        for file in files {
            let name = file
                .strip_prefix(&root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            if rule.owners.contains(&name.as_str()) {
                continue;
            }
            let text = code(&std::fs::read_to_string(&file).unwrap());
            for act in rule.acts {
                if text.contains(act) {
                    wrong.push(format!(
                        "{name} does `{act}`: {} ({}); use {}",
                        rule.rule, rule.since, rule.owner
                    ));
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "a rule has a second owner:\n{}",
        wrong.join("\n")
    );
}

#[test]
fn a_rule_reads_code_and_strings_not_comments_or_tests() {
    let text = r####"
        // text.rsplit_once('.') in a comment
        /* and /* nested */ rsplit_once('.') */
        let url = "https://example"; let (a, b) = text.rsplit_once('.')?;
        let brace = '{'; let quote = '"'; let tick = '\''; let life: &'static str = "ORLAPKG";
        let raw = r#"a "quoted" // not a comment"#; std::fs::write(p, b)?;
        #[cfg(test)]
        mod tests { fn f() { let s = "}"; std::fs::rename(a, b); } }
        #[cfg(test)]
        use std::fs::File::create;
        fn after() { OpenOptions::new(); }
    "####;
    let code = code(text);
    assert!(!code.contains("in a comment") && !code.contains("nested"));
    assert_eq!(code.matches("rsplit_once('.')").count(), 1, "{code}");
    assert!(code.contains("\"ORLAPKG\"") && code.contains("'static"));
    assert!(
        code.contains("// not a comment\"#; std::fs::write"),
        "{code}"
    );
    assert!(
        !code.contains("fs::rename") && !code.contains("File::create"),
        "{code}"
    );
    assert!(
        code.contains("OpenOptions::new()"),
        "code after a test is read: {code}"
    );
}
