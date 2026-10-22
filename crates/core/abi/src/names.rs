//! Near-miss suggestions for unknown names: the one rule every layer that
//! reports an unknown name uses.

/// The name among `known` that `wanted` was most likely meant to be.
///
/// Near means an edit distance of at most a third of the length (at least
/// one), ignoring case; a name under three characters is near nothing.
pub fn nearest<'a>(wanted: &str, known: impl IntoIterator<Item = &'a str>) -> Option<&'a str> {
    let length = wanted.chars().count();
    if length < 3 {
        return None;
    }
    let allowed = (length / 3).max(1);
    known
        .into_iter()
        .filter_map(|name| {
            let distance = edit_distance(&wanted.to_lowercase(), &name.to_lowercase());
            (distance <= allowed).then_some((distance, name))
        })
        .min_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)))
        .map(|(_, name)| name)
}

/// `; did you mean `x`?` for the nearest name, or empty.
pub fn did_you_mean<'a>(wanted: &str, known: impl IntoIterator<Item = &'a str>) -> String {
    suggestion(nearest(wanted, known))
}

/// `; did you mean `name`?` for a near miss already found, or empty.
pub fn suggestion(near: Option<&str>) -> String {
    match near {
        Some(name) => format!("; did you mean `{name}`?"),
        None => String::new(),
    }
}

/// How many single-character insertions, deletions, replacements and
/// swaps of two neighbors turn one string into the other; a swap
/// (`resutl` for `result`) counts as one.
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut grid = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for (i, row) in grid.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in grid[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let same = a[i - 1] == b[j - 1];
            let cost = usize::from(!same);
            let mut best = (grid[i - 1][j] + 1)
                .min(grid[i][j - 1] + 1)
                .min(grid[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                best = best.min(grid[i - 2][j - 2] + 1);
            }
            grid[i][j] = best;
        }
    }
    grid[a.len()][b.len()]
}

#[cfg(test)]
mod nearest_tests {
    use super::*;

    #[test]
    fn a_near_miss_is_named_and_a_different_name_is_not() {
        let known = ["source", "target", "scale", "element"];
        assert_eq!(nearest("sorce", known), Some("source"));
        assert_eq!(nearest("Source", known), Some("source"));
        // Two letters the wrong way round counts as one.
        assert_eq!(nearest("scael", known), Some("scale"));
        assert_eq!(nearest("resutl", ["result", "reset"]), Some("result"));
        assert_eq!(nearest("Flaot", ["Float", "Int"]), Some("Float"));
        assert_eq!(nearest("elemnt", known), Some("element"));
        assert_eq!(nearest("nope", known), None);
        assert_eq!(nearest("x", ["y"]), None);
        // The nearest of several, and the first by name where two tie.
        assert_eq!(nearest("targe", ["target", "targets"]), Some("target"));
        assert_eq!(did_you_mean("sorce", known), "; did you mean `source`?");
        assert_eq!(did_you_mean("nope", known), "");
    }
}
