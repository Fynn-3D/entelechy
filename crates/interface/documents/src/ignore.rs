//! What a directory's listing leaves out, as git does: the rules of every
//! `.gitignore` on the way to a file, each relative to its own directory.
//! A line is a pattern, `#` a comment, `!` taking a pattern back; a pattern
//! ending `/` matches directories alone; one with a `/` elsewhere is held
//! to the `.gitignore`'s directory, one without matches a name at any
//! depth; `*` and `?` match within a part, `[...]` one of a set, `**` any
//! parts. The last rule that matches decides, and a directory left out is
//! not walked, so nothing under it comes back.

/// The rules met so far, in order, each with the directory it is under.
#[derive(Clone, Debug, Default)]
pub(crate) struct Ignore {
    rules: Vec<Rule>,
}

#[derive(Clone, Debug)]
struct Rule {
    /// The directory the `.gitignore` is in, relative to the listing's
    /// root, ending `/` (empty at the root).
    under: String,
    pattern: Vec<char>,
    negated: bool,
    directories: bool,
    anchored: bool,
}

impl Ignore {
    /// These rules with those of the `.gitignore` whose text is `text`, in
    /// the directory `under` (relative to the root, ending `/`, empty at
    /// the root).
    pub(crate) fn with(&self, under: &str, text: &str) -> Ignore {
        let mut rules = self.rules.clone();
        for line in text.lines() {
            let line = line.trim_end_matches([' ', '\t', '\r']);
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (negated, line) = match line.strip_prefix('!') {
                Some(rest) => (true, rest),
                None => (false, line.strip_prefix('\\').unwrap_or(line)),
            };
            let (directories, line) = match line.strip_suffix('/') {
                Some(rest) => (true, rest),
                None => (false, line),
            };
            if line.is_empty() {
                continue;
            }
            let anchored = line.contains('/');
            let line = line.strip_prefix('/').unwrap_or(line);
            rules.push(Rule {
                under: under.to_owned(),
                pattern: line.chars().collect(),
                negated,
                directories,
                anchored,
            });
        }
        Ignore { rules }
    }

    /// Whether the file or directory at `path` (relative to the root) is
    /// left out.
    pub(crate) fn ignores(&self, path: &str, directory: bool) -> bool {
        let mut out = false;
        for rule in &self.rules {
            let Some(inside) = path.strip_prefix(rule.under.as_str()) else {
                continue;
            };
            if rule.directories && !directory {
                continue;
            }
            let text: Vec<char> = if rule.anchored {
                inside.chars().collect()
            } else {
                inside
                    .rsplit('/')
                    .next()
                    .unwrap_or(inside)
                    .chars()
                    .collect()
            };
            if matches(&rule.pattern, &text) {
                out = !rule.negated;
            }
        }
        out
    }
}

/// Whether `text` is matched by the glob `pattern`.
fn matches(pattern: &[char], text: &[char]) -> bool {
    match pattern.first() {
        None => text.is_empty(),
        Some('*') if pattern.get(1) == Some(&'*') => {
            let rest = &pattern[2..];
            if rest.is_empty() {
                return true;
            }
            // `**/` is no part or any parts each ended by `/`.
            if rest.first() == Some(&'/') {
                let after = &rest[1..];
                return matches(after, text)
                    || (0..text.len())
                        .any(|at| text[at] == '/' && matches(after, &text[at + 1..]));
            }
            (0..=text.len()).any(|at| matches(rest, &text[at..]))
        }
        Some('*') => {
            let rest = &pattern[1..];
            (0..=text.len())
                .take_while(|&at| at == 0 || text[at - 1] != '/')
                .any(|at| matches(rest, &text[at..]))
        }
        Some('?') => text.first().is_some_and(|c| *c != '/') && matches(&pattern[1..], &text[1..]),
        Some('[') => {
            let Some(end) = pattern
                .iter()
                .skip(2)
                .position(|c| *c == ']')
                .map(|at| at + 2)
            else {
                return text.first() == Some(&'[') && matches(&pattern[1..], &text[1..]);
            };
            let Some(&c) = text.first() else {
                return false;
            };
            let set = &pattern[1..end];
            let (negated, set) = match set.first() {
                Some('!' | '^') => (true, &set[1..]),
                _ => (false, set),
            };
            let mut found = false;
            let mut at = 0;
            while at < set.len() {
                if at + 2 < set.len() && set[at + 1] == '-' {
                    found |= set[at] <= c && c <= set[at + 2];
                    at += 3;
                } else {
                    found |= set[at] == c;
                    at += 1;
                }
            }
            c != '/' && found != negated && matches(&pattern[end + 1..], &text[1..])
        }
        Some('\\') if pattern.len() > 1 => {
            text.first() == Some(&pattern[1]) && matches(&pattern[2..], &text[1..])
        }
        Some(c) => text.first() == Some(c) && matches(&pattern[1..], &text[1..]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules(text: &str) -> Ignore {
        Ignore::default().with("", text)
    }

    #[test]
    fn a_name_matches_at_any_depth_and_a_slash_holds_it_to_its_directory() {
        let ignore = rules("target\n/build\nassets/*.png\n");
        assert!(ignore.ignores("target", true));
        assert!(ignore.ignores("crates/x/target", true));
        assert!(ignore.ignores("build", true));
        assert!(!ignore.ignores("src/build", true));
        assert!(ignore.ignores("assets/a.png", false));
        assert!(!ignore.ignores("assets/deep/a.png", false));
        assert!(!ignore.ignores("other/a.png", false));
    }

    #[test]
    fn stars_classes_and_double_stars() {
        let ignore = rules("*.field\nnote?.md\n[ab]*.tmp\n**/cache/**\ndocs/**/draft.md\n");
        assert!(ignore.ignores("beasts/wolf.field", false));
        assert!(ignore.ignores("note1.md", false));
        assert!(!ignore.ignores("note12.md", false));
        assert!(ignore.ignores("a1.tmp", false));
        assert!(!ignore.ignores("c1.tmp", false));
        assert!(ignore.ignores("x/cache/y/z", false));
        assert!(ignore.ignores("docs/draft.md", false));
        assert!(ignore.ignores("docs/a/b/draft.md", false));
        assert!(!ignore.ignores("docs/a/b/final.md", false));
    }

    #[test]
    fn a_directory_pattern_skips_files_and_the_last_rule_decides() {
        let ignore = rules("# saves\nsaves/\n*.log\n!keep.log\n");
        assert!(ignore.ignores("saves", true));
        assert!(!ignore.ignores("saves", false));
        assert!(ignore.ignores("run.log", false));
        assert!(!ignore.ignores("keep.log", false));
    }

    #[test]
    fn a_nested_gitignore_holds_below_its_own_directory() {
        let ignore = rules("*.tmp\n").with("tools/", "out\n/local.py\n");
        assert!(ignore.ignores("tools/out", true));
        assert!(ignore.ignores("tools/deep/out", true));
        assert!(!ignore.ignores("out", true));
        assert!(ignore.ignores("tools/local.py", false));
        assert!(!ignore.ignores("tools/deep/local.py", false));
        assert!(ignore.ignores("tools/x.tmp", false));
    }
}
