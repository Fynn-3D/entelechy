//! The tree every backend folds a reduction in, so a float sum is the
//! same wherever it runs.
//!
//! A level of `n` elements becomes a level of `n.div_ceil(2)`: element
//! `i` is `merge(in[2i], in[2i + 1])`, and an odd last element is
//! carried up untouched, until one is left.
//!
//! A backend may fold [`LEVELS_AT_ONCE`] levels in one pass, turning a
//! run of [`FOLD`] elements into one by the same pairwise tree.

/// How many levels of the tree a backend may fold in one pass.
pub const LEVELS_AT_ONCE: u32 = 4;

/// The run of elements one pass of [`LEVELS_AT_ONCE`] levels folds
/// into one.
pub const FOLD: usize = 1 << LEVELS_AT_ONCE;

/// How many elements are left after folding `count` of them by
/// [`FOLD`].
pub fn after_fold(count: usize) -> usize {
    let mut left = count;
    for _ in 0..LEVELS_AT_ONCE {
        left = left.div_ceil(2);
    }
    left
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Folding a run by the pairwise tree equals folding the whole
    /// level pairwise that many times.
    #[test]
    fn a_run_folded_at_once_is_the_levels_one_at_a_time() {
        let pairwise = |level: &[String]| -> Vec<String> {
            let mut next = Vec::new();
            let mut at = 0;
            while at + 1 < level.len() {
                next.push(format!("({} {})", level[at], level[at + 1]));
                at += 2;
            }
            if at < level.len() {
                next.push(level[at].clone());
            }
            next
        };
        for count in [1usize, 2, 3, 15, 16, 17, 100, 4096] {
            let start: Vec<String> = (0..count).map(|i| i.to_string()).collect();
            let mut one_at_a_time = start.clone();
            for _ in 0..LEVELS_AT_ONCE {
                one_at_a_time = pairwise(&one_at_a_time);
            }
            assert_eq!(
                one_at_a_time.len(),
                after_fold(count),
                "{count} elements leave this many"
            );
            // A run at a time reaches the same expressions.
            let at_once: Vec<String> = (0..after_fold(count))
                .map(|element| {
                    let start_at = element * FOLD;
                    let run = FOLD.min(count - start_at);
                    let mut held: Vec<String> = start[start_at..start_at + run].to_vec();
                    while held.len() > 1 {
                        held = pairwise(&held);
                    }
                    held.remove(0)
                })
                .collect();
            assert_eq!(at_once, one_at_a_time, "{count} elements fold the same");
        }
    }
}
