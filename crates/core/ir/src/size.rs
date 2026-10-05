//! How big a module is as a backend compiles it: its instructions, nested
//! blocks counted, a function at a time, each function once however often
//! it is called; and what it carries through its branches, each function
//! as often as a device's compiler writes it out where it is called.

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

/// What a module carries through its branches ([`carried`]): the words of
/// every struct value an `if`, a `match` or a `loop` yields, each function
/// counted as often as it is written out where it is called, as a
/// device's compiler writes out every call. Such a compiler keeps each
/// value's words live together where the branches meet, and takes longer,
/// past in proportion, the more it carries: what makes it long is most
/// often a wide struct passed through many branches.
#[derive(Clone, Debug)]
pub struct Carried {
    /// The words carried, all told.
    pub total: u64,
    /// By struct index, how many values of it are carried, all told.
    pub structs: Vec<u64>,
    /// By function index, the words its own branches carry, all its
    /// writings out told.
    pub functions: Vec<u64>,
    /// The words the roots' own branches carry.
    pub roots: u64,
}

/// A module's [`Carried`].
pub fn carried(module: &Module) -> Carried {
    let n = module.functions.len();
    let roots: Vec<&Function> = std::iter::once(&module.entry)
        .chain(&module.merge)
        .chain(&module.init)
        .collect();
    // How often each function is written out: a function calls only those
    // below it, so its callers are above it and counted first.
    let mut calls_from = vec![vec![0u64; n]; n];
    let mut from_roots = vec![0u64; n];
    for (f, function) in module.functions.iter().enumerate() {
        call_sites(&function.body, &mut calls_from[f]);
    }
    for root in &roots {
        call_sites(&root.body, &mut from_roots);
    }
    let mut written = vec![0u64; n];
    for f in (0..n).rev() {
        written[f] = (f + 1..n).fold(from_roots[f], |sum, caller| {
            sum.saturating_add(written[caller].saturating_mul(calls_from[caller][f]))
        });
    }
    let mut structs = vec![0u64; module.structs.len()];
    let mut functions = vec![0u64; n];
    let mut total = 0u64;
    let mut roots_words = 0u64;
    for (f, function) in module.functions.iter().enumerate() {
        let words = merged(module, &function.body, written[f], &mut structs);
        functions[f] = words;
        total = total.saturating_add(words);
    }
    for root in &roots {
        let words = merged(module, &root.body, 1, &mut structs);
        roots_words = roots_words.saturating_add(words);
        total = total.saturating_add(words);
    }
    Carried {
        total,
        structs,
        functions,
        roots: roots_words,
    }
}

/// Each function's call sites in `body`, added to `into` by index.
fn call_sites(body: &[Instruction], into: &mut [u64]) {
    for instruction in body {
        if let Op::Call(f, _) = &instruction.op
            && let Some(sites) = into.get_mut(*f as usize)
        {
            *sites += 1;
        }
        for block in instruction.op.blocks() {
            call_sites(&block.body, into);
        }
    }
}

/// The words of the struct values `body`'s branches yield, `times` over,
/// each value's struct counted in `structs`.
fn merged(module: &Module, body: &[Instruction], times: u64, structs: &mut [u64]) -> u64 {
    body.iter()
        .map(|instruction| {
            let own = match (&instruction.op, instruction.ty) {
                (Op::If { .. } | Op::Match { .. } | Op::Loop { .. }, orla_abi::Type::Struct(s)) => {
                    if let Some(count) = structs.get_mut(s as usize) {
                        *count = count.saturating_add(times);
                    }
                    (crate::words(&module.structs, instruction.ty) as u64).saturating_mul(times)
                }
                _ => 0,
            };
            instruction.op.blocks().into_iter().fold(own, |sum, block| {
                sum.saturating_add(merged(module, &block.body, times, structs))
            })
        })
        .fold(0, u64::saturating_add)
}
