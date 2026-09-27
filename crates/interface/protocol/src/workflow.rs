//! How an agent works through the operations: the loop the architecture
//! gives AI (understand, search, generate, compose, run, observe,
//! improve), each step with the operations that serve it. The operations
//! are [`crate::ops`]'s; this table only orders them for use. `help`'s
//! index and the MCP instructions are written from it.

/// One step of the loop.
pub struct Step {
    /// Its name.
    pub name: &'static str,
    /// What it is for, one sentence.
    pub about: &'static str,
    /// The operations that serve it, by name.
    pub ops: &'static [&'static str],
}

/// The loop, in order; every operation but `quit` serves a step.
pub const WORKFLOW: [Step; 7] = [
    Step {
        name: "understand",
        about: "what Orla is, and what this session holds and runs",
        ops: &["help", "status", "kernels", "kernel", "graph", "document"],
    },
    Step {
        name: "search",
        about: "what the world already knows, before writing anew",
        ops: &["search"],
    },
    Step {
        name: "generate",
        about: "a Kernel written, compiled and checked, publishing nothing",
        ops: &["prepare"],
    },
    Step {
        name: "compose",
        about: "the world changed: a law published or retired, the scene rewritten on disk, a param or a state set; a refused change leaves it as it was",
        ops: &["publish", "retire", "compose", "sync", "param", "write"],
    },
    Step {
        name: "run",
        about: "frames evaluated, one or many, under a clock and inputs",
        ops: &["evaluate", "advance", "window", "reset"],
    },
    Step {
        name: "observe",
        about: "what was computed and why: a picture seen, a Field summed up, an element traced, the history of changes",
        ops: &["read", "trace", "bench", "history", "pin", "unpin"],
    },
    Step {
        name: "improve",
        about: "a world kept and returned to, to try a change and undo it",
        ops: &["snapshot", "restore"],
    },
];

/// The loop as one paragraph: each step, what it is for, its operations.
pub fn text() -> String {
    WORKFLOW
        .iter()
        .map(|step| {
            let ops: Vec<String> = step.ops.iter().map(|op| format!("`{op}`")).collect();
            format!("{} ({}): {}.", step.name, ops.join(", "), step.about)
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The loop as a Markdown table, as the documents render it.
pub fn table() -> String {
    let mut out = String::from("| Step | Operations | For |\n| --- | --- | --- |\n");
    for step in &WORKFLOW {
        let ops: Vec<String> = step.ops.iter().map(|op| format!("`{op}`")).collect();
        out.push_str(&format!(
            "| {} | {} | {} |\n",
            step.name,
            ops.join(", "),
            step.about
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::WORKFLOW;

    /// Every operation a step names exists, and every operation but `quit`
    /// serves a step.
    #[test]
    fn the_loop_names_the_operations_and_every_one_serves_it() {
        let names = crate::ops::names();
        for step in &WORKFLOW {
            for op in step.ops {
                assert!(
                    names.contains(op),
                    "{} names no operation `{op}`",
                    step.name
                );
            }
        }
        for op in names.iter().filter(|op| **op != "quit") {
            assert!(
                WORKFLOW.iter().any(|step| step.ops.contains(op)),
                "`{op}` serves no step of the loop"
            );
        }
    }
}
