# Orla Code Rules

## Principle

Code should reveal the architecture.

Not hide it.

---

## Keep Boundaries

Respect:

```text
Host
↓
ABI
↓
IR
↓
Kernel Runtime
↓
Graph Runtime
```

Do not bypass layers for convenience.

---

## Keep Modules Small

One module.

One responsibility.

Avoid:

```text
God objects
hidden globals
cross-layer helpers
implicit ownership
```

---

## Make State Explicit

Every state should have an owner.

Prefer:

```text
immutable values
explicit mutation
explicit lifetime
explicit dependencies
```

---

## Make Errors Visible

Do not silently recover from invalid runtime state.

Return meaningful errors.

Keep the previous valid Kernel alive when hot reload fails.

---

## Protect Public Contracts

Treat these as stable boundaries:

```text
Capability ABI
Orla IR
KernelID
Serialized Graph Format
```

Change them intentionally.

Test compatibility.

---

## Prefer Simple Code

Prefer:

```text
clear
small
boring
testable
```

over:

```text
clever
abstract
magical
prematurely generic
```

---

## Automation

Formatting and basic style belong to tools.

Use:

```text
formatter
linter
compiler warnings
tests
CI
```

Do not spend human attention on what machines can enforce.

---

## Final Rule

> **Formatting is automatic.  
> Architecture is deliberate.**