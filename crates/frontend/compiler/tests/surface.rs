//! What a Kernel may call and write is a contract with every source
//! already written: `surface.txt` beside this file holds it at
//! `LANGUAGE_VERSION` — every builtin with its signatures, the capability
//! it needs and whether only the entry may call it, and every form with
//! its shape. Docs are left out; they may say anything.
//!
//! The surface may grow at a version: a new builtin, a new signature, a
//! capability no longer needed. What a source could write before must
//! still compile, so dropping or changing anything else is a new version.
//! The test fails on any difference. `ORLA_BLESS=1` rewrites the file,
//! and refuses to when what it would drop has not had its version bumped.

use std::collections::BTreeMap;
use std::path::Path;

use orla_compiler::help;
use orla_language::LANGUAGE_VERSION;

/// One entry: its header line (`builtin name`, with `requires` and
/// `entry` as they apply, or `form name`) and the lines under it.
#[derive(Clone, Debug, PartialEq)]
struct Entry {
    header: String,
    lines: Vec<String>,
}

/// The surface as this compiler has it, by the name that keys each entry.
fn current() -> BTreeMap<String, Entry> {
    let mut entries = BTreeMap::new();
    for builtin in help::builtins() {
        let mut header = format!("builtin {}", builtin.name);
        if let Some(capability) = builtin.requires {
            header.push_str(&format!(" requires {capability}"));
        }
        if builtin.entry_only {
            header.push_str(" entry");
        }
        let mut lines = builtin.signatures.clone();
        lines.sort();
        entries.insert(format!("builtin {}", builtin.name), Entry { header, lines });
    }
    for form in help::forms() {
        entries.insert(
            format!("form {}", form.name),
            Entry {
                header: format!("form {}", form.name),
                lines: form.shape.lines().map(str::to_owned).collect(),
            },
        );
    }
    entries
}

fn render(version: u32, entries: &BTreeMap<String, Entry>) -> String {
    let mut text = format!("language {version}\n");
    for entry in entries.values() {
        text.push_str(&entry.header);
        text.push('\n');
        for line in &entry.lines {
            text.push_str("  ");
            text.push_str(line);
            text.push('\n');
        }
    }
    text
}

fn parse(text: &str) -> (u32, BTreeMap<String, Entry>) {
    let mut lines = text.lines();
    let version = lines
        .next()
        .and_then(|line| line.strip_prefix("language "))
        .and_then(|number| number.parse().ok())
        .expect("surface.txt begins `language N`");
    let mut entries = BTreeMap::new();
    let mut open: Option<Entry> = None;
    let key = |header: &str| header.split(' ').take(2).collect::<Vec<_>>().join(" ");
    for line in lines {
        match line.strip_prefix("  ") {
            Some(body) => open
                .as_mut()
                .expect("a line belongs to an entry")
                .lines
                .push(body.to_owned()),
            None => {
                if let Some(entry) = open.take() {
                    entries.insert(key(&entry.header), entry);
                }
                open = Some(Entry {
                    header: line.to_owned(),
                    lines: Vec::new(),
                });
            }
        }
    }
    if let Some(entry) = open {
        entries.insert(key(&entry.header), entry);
    }
    (version, entries)
}

/// What a source written against `before` could write and `now` no
/// longer accepts, one line each.
fn broken(before: &BTreeMap<String, Entry>, now: &BTreeMap<String, Entry>) -> Vec<String> {
    let flag = |header: &str, word: &str| header.split(' ').any(|part| part == word);
    let needs = |header: &str| {
        header
            .split(" requires ")
            .nth(1)
            .map(|rest| rest.split(' ').next().unwrap_or_default().to_owned())
    };
    let mut broken = Vec::new();
    for (key, was) in before {
        let Some(is) = now.get(key) else {
            broken.push(format!("{key} is gone"));
            continue;
        };
        if key.starts_with("form ") {
            if is.lines != was.lines {
                broken.push(format!("{key} is written differently"));
            }
            continue;
        }
        for signature in &was.lines {
            if !is.lines.contains(signature) {
                broken.push(format!("{key} no longer takes `{signature}`"));
            }
        }
        if let Some(capability) = needs(&is.header)
            && needs(&was.header).as_ref() != Some(&capability)
        {
            broken.push(format!("{key} now needs `{capability}`"));
        }
        if flag(&is.header, "entry") && !flag(&was.header, "entry") {
            broken.push(format!("{key} may now be called only from the entry"));
        }
    }
    broken
}

#[test]
fn the_surface_a_source_may_use_is_the_one_recorded() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/surface.txt");
    let now = current();
    let recorded = std::fs::read_to_string(&path).unwrap_or_default();
    let (version, before) = if recorded.is_empty() {
        (0, BTreeMap::new())
    } else {
        parse(&recorded)
    };
    let broken = broken(&before, &now);
    if std::env::var_os("ORLA_BLESS").is_some() {
        assert!(
            broken.is_empty() || LANGUAGE_VERSION > version,
            "refusing to record a surface that breaks language {version} sources without a \
             new LANGUAGE_VERSION:\n{}",
            broken.join("\n")
        );
        std::fs::write(&path, render(LANGUAGE_VERSION, &now)).unwrap();
        return;
    }
    let expected = render(LANGUAGE_VERSION, &now);
    if recorded == expected {
        return;
    }
    let added: Vec<&String> = now
        .keys()
        .filter(|key| !before.contains_key(*key))
        .collect();
    if broken.is_empty() {
        panic!(
            "the surface grew ({added:?} and signatures within); record it with \
             `ORLA_BLESS=1 cargo test -p orla-compiler --test surface`"
        );
    }
    panic!(
        "sources written for language {version} would stop compiling:\n{}\nraise \
         LANGUAGE_VERSION, then record the surface with \
         `ORLA_BLESS=1 cargo test -p orla-compiler --test surface`",
        broken.join("\n")
    );
}
