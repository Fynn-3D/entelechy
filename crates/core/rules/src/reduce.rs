//! The tree a reduction folds in, which every backend folds in, so that
//! a float sum does not depend on where it ran.
//!
//! A level of `n` elements becomes a level of `n.div_ceil(2)`: element
//! `i` is `merge(in[2i], in[2i + 1])`, and an odd last element is
//! carried up untouched. Repeat until one is left. `merge` is
//! associative and commutative by the language's rule, so the shape of
//! the tree is free; what is not free is that the three backends agree
//! on it, since a float sum rounds in the order it is added.
//!
//! A backend may do several of those levels in one pass, since the tree
//! is the same however it is walked: [`LEVELS_AT_ONCE`] of them fold a
//! run of [`FOLD`] into one element, by the same pairwise tree. It
//! matters on a device, where a level is a dispatch with a bind group
//! and a uniform of its own — for a few thousand elements those were
//! most of what a reduction cost. It does not matter where the levels
//! are loops, and the interpreter and the JIT walk one level at a time:
//! the sixteen merges of a run are a chain, and a level of pairs is
//! sixteen chains the machine can run at once.

/// How many levels of the tree a backend may fold in one pass.
pub const LEVELS_AT_ONCE: u32 = 4;

/// How many elements of a level one pass of [`LEVELS_AT_ONCE`] levels
/// folds into one: the run a backend takes at a time.
pub const FOLD: usize = 1 << LEVELS_AT_ONCE;

/// How many elements are left after folding `count` of them by
/// [`FOLD`], which is [`LEVELS_AT_ONCE`] levels of the tree at once.
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

    /// Folding a run by the pairwise tree is the same as folding the
    /// whole level pairwise that many times, which is what lets a
    /// backend do several levels in one pass and change no value.
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
