//! How big a module is as a backend compiles it: its instructions, nested
//! blocks counted, a function at a time, each function once however often
//! it is called. A device's compiler takes the whole of it, and may take
//! long over a very large one, or fail on it; what makes one large is
//! most often a function reached that was not meant to be.

use crate::{Function, Instruction, Module, Op};

/// A module's size: each function's own instructions, the roots' (the
/// entry, and a reduction's merge or a scan's first value), and whom each
/// calls.
#[derive(Clone, Debug)]
pub struct Sizes {
    /// Each function's own instructions, by its index.
    pub functions: Vec<usize>,
    /// Whom each function calls, by index, each once.
    pub calls: Vec<Vec<u32>>,
    /// The roots' own instructions, all told.
    pub roots: usize,
    /// Whom the roots call.
    pub root_calls: Vec<u32>,
}

impl Sizes {
    /// The module's instructions, all told.
    pub fn total(&self) -> usize {
        self.roots + self.functions.iter().sum::<usize>()
    }

    /// The calls from the roots down to function `f` by the shortest way,
    /// each function by index, `f` last; empty when nothing reaches it.
    pub fn way_to(&self, f: u32) -> Vec<u32> {
        let n = self.functions.len();
        let mut from: Vec<Option<u32>> = vec![None; n];
        let mut seen = vec![false; n];
        let mut queue: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
        for &c in &self.root_calls {
            if (c as usize) < n && !seen[c as usize] {
                seen[c as usize] = true;
                queue.push_back(c);
            }
        }
        while let Some(at) = queue.pop_front() {
            if at == f {
                let mut way = vec![f];
                let mut k = f;
                while let Some(back) = from[k as usize] {
                    way.push(back);
                    k = back;
                }
                way.reverse();
                return way;
            }
            for &c in &self.calls[at as usize] {
                if (c as usize) < n && !seen[c as usize] {
                    seen[c as usize] = true;
                    from[c as usize] = Some(at);
                    queue.push_back(c);
                }
            }
        }
        Vec::new()
    }
}

/// A module's [`Sizes`].
pub fn sizes(module: &Module) -> Sizes {
    let measured: Vec<(usize, Vec<u32>)> = module.functions.iter().map(measure).collect();
    let roots: Vec<(usize, Vec<u32>)> = std::iter::once(&module.entry)
        .chain(&module.merge)
        .chain(&module.init)
        .map(measure)
        .collect();
    let mut root_calls: Vec<u32> = roots
        .iter()
        .flat_map(|(_, calls)| calls.iter().copied())
        .collect();
    root_calls.sort_unstable();
    root_calls.dedup();
    Sizes {
        functions: measured.iter().map(|(size, _)| *size).collect(),
        calls: measured.into_iter().map(|(_, calls)| calls).collect(),
        roots: roots.iter().map(|(size, _)| *size).sum(),
        root_calls,
    }
}

/// A function's own instructions and whom it calls.
fn measure(function: &Function) -> (usize, Vec<u32>) {
    let mut calls = Vec::new();
    let size = count(&function.body, &mut calls);
    calls.sort_unstable();
    calls.dedup();
    (size, calls)
}

fn count(body: &[Instruction], calls: &mut Vec<u32>) -> usize {
    body.iter()
        .map(|instruction| {
            if let Op::Call(index, _) = &instruction.op {
                calls.push(*index);
            }
            1 + instruction
                .op
                .blocks()
                .into_iter()
                .map(|block| count(&block.body, calls))
                .sum::<usize>()
        })
        .sum()
}
