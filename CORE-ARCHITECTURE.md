# Orla

**An AI-native dynamic kernel runtime.**

Orla is built on one idea:

> **The host defines the world.  
> Kernels define what can happen inside it.  
> Graphs compose capabilities.  
> AI creates new ones.**

---

## Core Model

``` text
Human / AI
    ↓
 Intent
    ↓
 Graph
    ↓
 Kernel
    ↓
 Orla Runtime
    ↓
 Hardware
```

---

## The Stack

``` text
Interface

AI Runtime

Graph Runtime

Kernel Runtime

Orla IR

Capability ABI

Host Core

Hardware
```

---

## Host

The Host should remain small and stable.

It provides only fundamental capabilities:

``` text
Memory
Compute
GPU
IO
Resources
Tasks
Security
```

The Host does not define product features.

It defines the world in which features can exist.

---

## Kernel

A Kernel is a dynamic capability.

``` text
Input
+
Parameters
+
Context
↓
Kernel
↓
Output
```

Formally:

``` text
Y = K(X, P, C)
```

Kernels should be:

``` text
Small
Typed
Safe
Composable
Hot-reloadable
AI-generatable
```

---

## Orla IR

Kernel source does not belong to any specific machine.

``` text
.orla
↓
AST
↓
Orla IR
↓
CPU / GPU / WASM / NPU
```

The language describes computation.

The backend decides how computation happens.

---

## Capability ABI

Kernels never own the machine.

They receive capabilities.

``` text
BufferHandle
ImageHandle
TextureHandle
Compute
Time
File
Network
```

Access is explicit.

``` text
requires gpu.compute
requires image.read
```

No capability.

No access.

---

## Hot Reload

Change a Kernel.

Save it.

Orla:

``` text
Compile
↓
Validate
↓
Prepare
↓
Atomic Swap
↓
Continue
```

No restart.

No broken runtime.

The old Kernel survives until the new one is valid.

---

## Graph

A Kernel defines one capability.

A Graph defines how capabilities become a system.

``` text
Noise
 ↓
Warp
 ↓
Pixelate
 ↓
Palette
 ↓
Output
```

Therefore:

``` text
Kernel = Capability

Graph = Program
```

---

## AI

AI should not control Orla by injecting arbitrary native code.

AI operates through:

``` text
Kernel Language
Graph API
Capability API
```

The AI loop becomes:

``` text
Understand
↓
Search
↓
Generate
↓
Compile
↓
Validate
↓
Compose
↓
Run
↓
Observe
↓
Improve
```

AI is not merely a copilot.

It becomes a **capability compiler**.

---

## Orla's Architecture

``` text
                 Human
                   │
                   ▼
                  AI
                   │
            Graph / Kernel
                   │
                   ▼
             Orla Runtime
                   │
          ┌────────┴────────┐
          │                 │
      Graph Runtime    Kernel Runtime
          │                 │
          └────────┬────────┘
                   │
                Orla IR
                   │
            Capability ABI
                   │
               Host Core
                   │
          CPU / GPU / NPU
```

---

## The Boundary

Orla separates two worlds:

``` text
Stable World
────────────
Host
ABI
Runtime

Dynamic World
─────────────
Kernel
Graph
AI-generated capabilities
```

This boundary is the architecture.

---

## The Principle

Traditional software asks:

> What features were built into the application?

Orla asks:

> What capabilities can exist inside the runtime?

Traditional software ships functionality.

Orla ships a world where functionality can be created.

---

## The First Proof

The first version of Orla only needs to prove four things:

``` text
.orla can compile

Kernel can call Host through ABI

Kernel can hot-swap safely

Graph can reference KernelID
```

If these four things work,

the architecture is alive.

Everything else can grow later.

---

## Final Definition

> **Orla is a runtime for software that can evolve while it is running.**

Or even shorter:

> **Host is the world.  
> Kernel is the law.  
> Graph is the structure.  
> AI is the creator.**