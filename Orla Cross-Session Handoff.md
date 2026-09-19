# Cross-session handoff

At session start, follow the reading order in AGENTS.md, inspect Git state, and
verify the code relevant to HANDOFF.md → Next. Repository state is authoritative.

Keep HANDOFF.md short:

- **Context:** non-obvious user decisions or constraints needed to continue.
- **Next:** one executable next action and its acceptance criteria.
- **Open:** unresolved blockers or unverified behavior, only when relevant.

Replace stale context instead of appending a diary. Do not copy feature lists,
commit history, version numbers, test counts, or successful build inventories.
Git records changes; code and tests establish behavior; the handoff preserves
intent that cannot be recovered from them.

Before ending, inspect the diff, run relevant checks, and update the next action.
Report actual verification in the commit/PR or session result. Preserve failures
or environment constraints in the handoff only when the next session needs them.
