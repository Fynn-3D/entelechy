# Entelechy Architecture Plan

## 0. Positioning

**Entelechy** is a dynamic Kernel Runtime and software framework designed for the AI-native era.

Its purpose is not to provide a fixed set of application features.

Its purpose is to provide a stable execution world in which computational capabilities can be:

```text
defined
generated
compiled
validated
loaded
hot-swapped
composed
reused
```

The core architectural equation of Entelechy is:

```text
Stable Host
+
Kernel Language
+
Kernel Runtime
+
Graph Runtime
+
AI Compiler
=
Dynamic Software System
```

In Entelechy, application capabilities are not assumed to be permanently embedded into the host binary.

Instead, a capability is treated as a:

```text
Runtime Object
```

The central idea is simple:

> Software functionality itself should be able to emerge at runtime.

---

# 1. Vision

Traditional software development typically follows this model:

```text
Developer
    ↓
Source Code
    ↓
Build
    ↓
Application
    ↓
User
```

Entelechy introduces a different model:

```text
Human / AI
     ↓
Intent
     ↓
Kernel / Graph
     ↓
Entelechy Runtime
     ↓
Live Capability
```

For example, a user may request:

```text
"Create a real-time pixel fluid simulator."
```

An Entelechy-based system could potentially perform:

```text
Intent Analysis
↓
Capability Search
↓
Kernel Generation
↓
Compilation
↓
Validation
↓
Hot Load
↓
Graph Composition
↓
Execution
```

The host application itself does not need to be rebuilt.

---

# 2. Core Philosophy

Entelechy is based on several architectural principles.

## 2.1 Stable Host, Dynamic Kernel

The first principle is:

```text
Host = Stable
Kernel = Dynamic
```

The host should change slowly.

Higher-level capabilities should evolve independently from the host whenever possible.

The host defines the world.

Kernels define what can happen inside that world.

---

## 2.2 Capability-First Architecture

Entelechy should not fundamentally organize itself around concepts such as:

```text
Class
Module
Plugin
Application
```

Its primary abstraction is:

```text
Capability
```

A capability may later appear as:

```text
Kernel
Node
Tool
Filter
Shader
Brush
Simulation
Agent Skill
Operator
```

These are different interfaces around the same underlying computational capability.

---

# 3. High-Level Architecture

```text
┌─────────────────────────────────────────┐
│              Interface Layer            │
│                                         │
│ UI / Node / CLI / Agent / API / Canvas │
└───────────────────┬─────────────────────┘
                    │
             Intent / Commands
                    │
┌───────────────────▼─────────────────────┐
│              AI Control Layer           │
│                                         │
│ Planner / Generator / Graph Builder     │
└───────────────────┬─────────────────────┘
                    │
┌───────────────────▼─────────────────────┐
│               Graph Runtime             │
│                                         │
│ Composition / Dependency / Cache        │
└───────────────────┬─────────────────────┘
                    │
┌───────────────────▼─────────────────────┐
│           Entelechy Kernel Runtime      │
│                                         │
│ Compile / Validate / Load / Hot Swap    │
└───────────────────┬─────────────────────┘
                    │
┌───────────────────▼─────────────────────┐
│               Entelechy IR              │
│                                         │
│ Typed Intermediate Representation       │
└──────────────┬──────────────┬───────────┘
               │              │
              CPU            GPU
               │              │
┌──────────────▼──────────────▼───────────┐
│                 Host Core               │
│                                         │
│ Memory / IO / GPU / Thread / Resource   │
└───────────────────┬─────────────────────┘
                    │
                 Hardware
```

---

# 4. Layer Model

The long-term Entelechy stack can be described as:

```text
L0  Hardware

L1  Host Core

L2  Capability ABI

L3  Entelechy IR

L4  Kernel Runtime

L5  Kernel Language

L6  Graph Runtime

L7  AI Runtime

L8  Interface Layer
```

Each layer should depend primarily on the layer beneath it.

Higher layers should avoid leaking implementation details into lower layers.

---

# 5. L0 — Hardware

The hardware layer includes:

```text
CPU
GPU
NPU
Memory
Storage
Network
```

Upper Entelechy layers should avoid depending directly on specific hardware architectures.

Hardware-specific behavior should remain behind backend abstractions.

---

# 6. L1 — Host Core

The Host Core should be one of the most stable parts of Entelechy.

Its responsibilities include:

```text
Memory

Resources

Threads

Jobs

GPU Device Access

Filesystem

Networking

Time

Logging

Serialization

Security

Lifecycle Management
```

The central rule is:

> The Host Core provides primitive capabilities, not product-level functionality.

For example, the Host may provide:

```text
create_buffer()
dispatch_compute()
read_file()
schedule_task()
```

It should generally not provide:

```text
blur_image()
generate_pixel_art()
create_fluid_effect()
```

Those belong at the Kernel level.

---

# 7. Host API

Kernel code should not directly access arbitrary host objects.

The Host API should expose controlled handles such as:

```text
BufferHandle

TextureHandle

ImageHandle

FileHandle

ModelHandle

TaskHandle
```

The Kernel should not receive unrestricted access to:

```text
Raw Pointers

Native Objects

OS Handles

Host Internal Memory
```

This separation is fundamental to safety, portability, and hot reload.

---

# 8. L2 — Capability ABI

The Capability ABI defines the world visible to Kernels.

Potential capability namespaces include:

```text
MemoryAPI

ComputeAPI

ResourceAPI

ImageAPI

TimeAPI

FileAPI

NetworkAPI
```

However, access should always be explicit.

A Kernel should declare what it requires.

---

# 9. Capability Security

A Kernel may declare capabilities such as:

```text
requires:
    gpu.compute
    image.read
    image.write
```

For example:

```text
kernel Blur

requires {
    gpu.compute
    image.sample
}
```

AI-generated Kernels should not automatically receive access to:

```text
Arbitrary Files

Arbitrary Networking

Native Code Execution

Raw System Memory
```

Entelechy should use capability-based security rather than implicit global access.

---

# 10. Capability Manifest

Each Kernel should have a manifest.

A manifest may contain:

```text
name
version
author
api_version

inputs
outputs

capabilities

backend

permissions
```

Example:

```text
kernel: pixel.dither

version: 1.2

requires:
    gpu.compute

inputs:
    image: Image
    strength: Float

outputs:
    image: Image
```

This manifest becomes part of the stable contract between the Kernel and the Runtime.

---

# 11. L3 — Entelechy IR

The Entelechy Kernel Language should not compile directly into a single backend format.

There should be an intermediate representation:

**Entelechy IR**

Compilation pipeline:

```text
Kernel Source
↓
Parser
↓
AST
↓
Semantic Analysis
↓
Entelechy IR
↓
Backend Compiler
```

---

# 12. Why Entelechy Needs an IR

Without an IR, the architecture may become tightly coupled:

```text
Kernel Language
↓
WGSL
```

This creates problems when new backends are introduced:

```text
CPU
Metal
CUDA
WASM
NPU
```

A better design is:

```text
             Entelechy IR

        ┌──────┼──────┐
        │      │      │
       CPU    GPU    WASM
```

This allows the language frontend and hardware backends to evolve independently.

---

# 13. Desired Properties of Entelechy IR

The IR should ideally be:

```text
Typed

SSA-like

Deterministic

Serializable

Inspectable

Optimizable

Backend-neutral
```

The IR should become Entelechy's true internal computational representation.

---

# 14. L4 — Kernel Runtime

The Kernel Runtime is the execution heart of Entelechy.

Its responsibilities include:

```text
Load

Compile

Validate

Instantiate

Execute

Cache

Unload

Hot Reload

Hot Swap
```

Possible architecture:

```text
KernelRuntime

├── KernelRegistry
├── Compiler
├── Validator
├── Loader
├── InstanceManager
├── HotReloadManager
├── BackendManager
└── KernelCache
```

---

# 15. Kernel Lifecycle

A Kernel may move through the following lifecycle:

```text
Source
↓
Parse
↓
Compile
↓
Validate
↓
Register
↓
Instantiate
↓
Execute
↓
Update
↓
Unload
```

Every stage should be observable and diagnosable.

---

# 16. Kernel Identity

Every Kernel should have a stable identity.

For example:

```text
KernelID
```

Possible identifier:

```text
image.blur@2
```

Graph Nodes should not depend directly on source file paths.

They should bind against:

```text
KernelID
```

This means source files can be reorganized without breaking Graphs.

---

# 17. Kernel Registry

The Kernel Registry stores metadata such as:

```text
Kernel ID

Version

Signature

Inputs

Outputs

Capabilities

Compiled IR

Backend Variants
```

The Registry acts as the capability index of the system.

---

# 18. L5 — Entelechy Kernel Language

The Kernel Language should be the primary capability definition language of Entelechy.

Its initial design should prioritize:

```text
Small

Typed

Safe

Deterministic by Default

AI-Friendly

Hot-Reload Friendly
```

The first version should not attempt to become a general-purpose programming language.

---

# 19. Example Kernel Syntax

Conceptual syntax:

```text
kernel Blur {

    input image: Image

    param radius: Float = 4.0

    output result: Image

    compute {
        ...
    }
}
```

The syntax should make capability boundaries obvious.

---

# 20. Node Definitions from Kernel Files

Node definitions can potentially live in Kernel source files.

Example:

```text
node Pixelate {

    input image: Image

    param size: Int = 8

    output image: Image

    kernel {
        ...
    }
}
```

When:

```text
pixelate.ent
```

is added or modified, the Runtime can:

```text
Compile
↓
Register
↓
Expose Node
```

without rebuilding the Host.

---

# 21. What the First Language Version Should Avoid

The first version should avoid features such as:

```text
Classes

Inheritance

Deep Reflection

Complex Generics

Unrestricted Macros

Raw Pointers
```

A minimal useful language could initially support:

```text
Functions

Structs

Scalar Types

Vectors

Buffers

Textures

Control Flow

Kernel Entry Points
```

The objective is computational clarity, not language feature count.

---

# 22. Hot Reload

Hot Reload should be treated as a first-class architectural capability.

It should not be added as an afterthought.

Basic flow:

```text
File Changed
↓
Compile New Version
↓
Validate
↓
Create New Instance
↓
Check State Compatibility
↓
Atomic Swap
↓
Invalidate Dependents
```

---

# 23. Atomic Swap

The Runtime should never destroy the active Kernel before the replacement is known to be valid.

Bad approach:

```text
Unload Old Kernel
↓
Compile New Kernel
```

If compilation fails, the capability disappears.

Preferred approach:

```text
Old Kernel
      │
      │ still active
      │
Compile New Kernel
      │
Validate
      │
Success
      │
Atomic Swap
      ↓
New Kernel
```

The old version remains active until the replacement is ready.

---

# 24. State Migration

Some Kernels will eventually contain persistent runtime state.

Examples:

```text
Simulation State

Particle State

Model State

Procedural History
```

Therefore Hot Reload may require:

```text
State Schema
```

For example:

```text
State V1
```

becoming:

```text
State V2
```

The Kernel may optionally define:

```text
migrate(v1) -> v2
```

State migration should be explicit.

---

# 25. Stateless Kernels First

Entelechy should strongly prefer stateless Kernels when possible.

Ideal form:

```text
Output =
Kernel(
    Input,
    Parameters,
    Context
)
```

Stateless execution simplifies:

```text
Hot Reload

Caching

Parallelism

Replay

Distributed Execution
```

This should be the default computational model.

---

# 26. L6 — Graph Runtime

A Kernel defines:

```text
one capability
```

A Graph defines:

```text
how capabilities are composed
```

Therefore:

```text
Kernel ≠ Graph
```

The two layers solve different problems.

---

# 27. Graph Runtime Components

The Graph Runtime may include:

```text
Graph

Node

Port

Edge

Evaluator

DependencyGraph

DirtyTracker

Cache

Scheduler
```

The Graph layer is responsible for composition, not low-level implementation.

---

# 28. Relationship Between Node and Kernel

Recommended model:

```text
Node
    ↓
Kernel Binding
```

Example:

```text
Blur Node

kernel:
    image.blur@2
```

The Node stores:

```text
parameters
connections
UI metadata
```

The Kernel stores:

```text
execution logic
```

This keeps presentation and computation separate.

---

# 29. Graph as Program

A Graph such as:

```text
Noise
↓
Warp
↓
Dither
↓
Palette
```

should be understood internally as:

```text
Program
```

The Graph Compiler can convert it into:

```text
Execution Plan
```

and potentially later into:

```text
Fused Kernel
```

---

# 30. Kernel Fusion

A future optimizer may detect:

```text
Kernel A
↓
Kernel B
↓
Kernel C
```

and compile them into:

```text
ABC Fused Kernel
```

This can reduce:

```text
Memory Transfers

GPU Dispatches

Intermediate Textures

Temporary Buffers
```

Kernel fusion could become a major performance feature of Entelechy.

---

# 31. L7 — AI Runtime

AI should not be treated as a chat feature inside Entelechy.

It should be treated as a:

```text
Meta Runtime
```

Its responsibilities may include:

```text
Understand Intent

Search Capabilities

Generate Kernels

Generate Graphs

Repair Compile Errors

Optimize Graphs

Observe Runtime Behavior

Iterate
```

---

# 32. AI Capability Flow

User request:

```text
"Create an NES-style flame."
```

The AI Runtime may:

```text
Parse Intent
↓
Search Kernel Registry
↓
Find:
noise
warp
palette
dither
↓
Construct Graph
```

If an important capability is missing, such as:

```text
NES Flame Quantizer
```

the system may:

```text
Generate Kernel
↓
Compile
↓
Validate
↓
Register
↓
Insert into Graph
```

This turns AI into a capability compiler rather than merely an assistant.

---

# 33. AI Should Not Control the Host Directly

AI interaction should remain inside safe structured surfaces:

```text
Kernel Language

Graph API

Capability API
```

AI-generated code should not be allowed to directly inject arbitrary native code into the Host.

That would bypass Entelechy's safety model.

---

# 34. L8 — Interface Layer

Entelechy should not depend on one user interface.

Possible interfaces include:

```text
Node Editor

Canvas

Timeline

CLI

IDE

Natural Language

Agent Interface

API

Headless Server
```

All of them operate on the same underlying Runtime.

---

# 35. Entelechy and AnimaComposer

If AnimaComposer remains the creative product layer, the relationship can be:

```text
AnimaComposer
=
Entelechy Runtime
+
Creative Interface
```

Entelechy is the platform.

AnimaComposer is one product powered by that platform.

---

# 36. Suggested Repository Structure

```text
entelechy/

host/

    memory/
    resource/
    compute/
    io/
    scheduler/
    security/

abi/

    capability/
    handles/
    permissions/

language/

    lexer/
    parser/
    ast/
    semantics/

ir/

    types/
    instructions/
    optimizer/
    serializer/

compiler/

    frontend/
    validator/
    lowering/
    codegen/

runtime/

    kernel/
    registry/
    instance/
    hotreload/
    cache/

graph/

    model/
    evaluator/
    compiler/
    scheduler/
    cache/

backend/

    cpu/
    webgpu/
    metal/
    wasm/

ai/

    planner/
    generator/
    capability_search/
    graph_builder/

editor/

    node/
    inspector/
    preview/

sdk/

tests/
```

---

# 37. Phase 1 — MVP

The first phase should not attempt to build the entire language ecosystem.

The goal is simple:

> Prove that an Entelechy Kernel can be dynamically compiled, loaded, executed, and hot-swapped.

Initial components:

```text
Host Core

Kernel ABI

Minimal Kernel Language

Compiler

Kernel Registry

Hot Reload

CPU Backend
```

Initial types:

```text
Float
Int
Vec2
Vec4
Image
```

Initial language constructs:

```text
input
output
parameter
function
kernel
```

---

# 38. MVP Demonstration

The first demonstration should be extremely small:

```text
Image
↓
Entelechy Kernel: Grayscale
↓
Output
```

Then modify:

```text
grayscale.ent
```

Save the file.

The application remains running.

The output updates immediately.

If this works reliably, the architectural core of Entelechy is proven.

---

# 39. Phase 2 — Graph + GPU

Add:

```text
WebGPU Backend

Texture / Buffer Support

Graph Runtime

Node Binding

Dirty Propagation

Caching
```

Demonstration:

```text
Noise
↓
Pixelate
↓
Palette
↓
Output
```

Every node is implemented by:

```text
.ent
```

Kernel files.

---

# 40. Phase 3 — AI-Generated Capabilities

Add:

```text
AI Kernel Generation

Graph Generation

Compiler Error Feedback

Capability Search

Dynamic Node Creation
```

Example user request:

```text
"Create a scanline effect."
```

The system generates:

```text
scanline.ent
```

Then:

```text
Compile
↓
Validate
↓
Register
↓
Expose Node
↓
Connect to Graph
```

The software has effectively created a new capability while running.

---

# 41. Phase 4 — Optimization and Portability

Add:

```text
Entelechy IR Optimizer

Kernel Fusion

Parallel Scheduler

WASM Backend

Metal Backend

Async Kernels

State Migration

Plugin Packages
```

At this stage, Entelechy becomes a serious runtime architecture rather than a simple scripting layer.

---

# 42. Phase 5 — General Dynamic Compute Runtime

Entelechy can later expand beyond visual computing.

Potential domains:

```text
Image

Video

Audio

Geometry

Simulation

AI

Automation

Agents
```

At this point Entelechy becomes:

```text
General Dynamic Compute Runtime
```

rather than only a graphics engine.

---

# 43. Entelechy Package Format

A future package format could be:

```text
.entpkg
```

Containing:

```text
manifest

kernels

assets

graphs

metadata
```

Example:

```text
nes_fx.entpkg
```

Installing the package may automatically provide:

```text
8 Kernels

4 Nodes

3 Graph Presets
```

This creates the foundation for an ecosystem.

---

# 44. Versioning

Version boundaries should be explicit from the beginning.

Recommended independent versions:

```text
Host API Version

Capability ABI Version

Kernel Language Version

IR Version

Package Version
```

For example:

```text
entelechy.api = 1

entelechy.language = 1

entelechy.ir = 1
```

These should not be collapsed into one global version number.

---

# 45. Determinism

Kernels should be deterministic by default whenever possible.

Given the same:

```text
Input

Parameters

Context
```

the Kernel should ideally produce the same:

```text
Output
```

This greatly improves:

```text
Caching

Replay

Debugging

Distributed Compute

Collaboration
```

Non-deterministic behavior should be explicit.

---

# 46. Side Effects

Side effects should also be explicit.

Default Kernel model:

```text
pure kernel
```

Meaning:

```text
No IO

No Network

No Persistent Mutation
```

If side effects are required:

```text
effect kernel
```

the Kernel must declare permissions such as:

```text
requires file.write
```

This makes effects visible to both the Runtime and the user.

---

# 47. Pure Kernel Model

The ideal primitive computational unit of Entelechy is:

```text
Kernel:
    Inputs
    Parameters
    Context

        ↓

    Compute

        ↓

    Outputs
```

Mathematically:

```text
Y = K(X, P, C)
```

This should be the conceptual foundation of the entire Kernel model.

---

# 48. Runtime State Model

System state should be separated into four categories:

```text
Persistent State

Runtime State

Cached State

UI State
```

Persistent State:

```text
Graphs

Parameters

Resources
```

Runtime State:

```text
Kernel Instances

Tasks

GPU State
```

Cached State:

```text
Computed Results
```

UI State:

```text
Node Position

Selection

Zoom

Panel Layout
```

These categories should not be mixed.

---

# 49. Long-Term Technical Moat

The long-term technical moat of Entelechy is not primarily:

```text
Node Editor
```

and not merely:

```text
Kernel Syntax
```

The deeper assets are:

```text
Entelechy Capability ABI

Entelechy IR

Kernel Runtime

Graph Runtime

AI Capability Compiler
```

In particular:

```text
IR + Runtime
```

should be treated as strategic core infrastructure.

---

# 50. Entelechy vs Traditional Plugins

Traditional plugin systems often look like:

```text
Native Code
↓
Host API
```

Entelechy instead follows:

```text
Capability
↓
Kernel Language
↓
Entelechy IR
↓
Sandbox Runtime
↓
Host Capability ABI
```

This architecture is more naturally compatible with:

```text
AI Generation

Hot Reload

Cross-Platform Execution

Sandboxing

Live Composition
```

---

# 51. End-State Architecture

The final vision of Entelechy is not:

```text
Software that supports custom Kernels.
```

It is:

> A computational world in which AI can safely create, modify, and compose software capabilities at runtime.

Conceptually:

```text
                    Human
                      │
                      │ Intent
                      ▼
                  AI Runtime
                      │
                Graph / Kernel
                      │
                      ▼
                 Entelechy Runtime
              ┌───────┴───────┐
              │               │
           Graph VM        Kernel VM
              │               │
              └───────┬───────┘
                      │
                   Entelechy IR
                      │
                Capability ABI
                      │
                  Host Core
                      │
        ┌─────────────┼─────────────┐
        │             │             │
       CPU           GPU           NPU
```

---

# 52. Product Structure

A possible product hierarchy:

```text
Entelechy
    Runtime / Kernel Platform

AnimaComposer
    Visual Creative Environment
    powered by Entelechy
```

The Entelechy ecosystem may later include:

```text
Entelechy Runtime

Entelechy Kernel

Entelechy Graph

Entelechy Studio

Entelechy SDK
```

---

# 53. One-Sentence Definition

> **Entelechy is an AI-native dynamic Kernel Runtime where the Host defines a stable world, Kernels define capabilities, Graphs compose them, and AI can generate new capabilities at runtime.**

---

# 54. The Four Proofs That Matter First

The first implementation should prove only four things:

```text
1. A .ent file can be compiled.

2. Compiled code can call the Host through a stable ABI.

3. A modified .ent file can be atomically hot-swapped.

4. A Graph can reference a KernelID instead of host-language implementation code.
```

If these four properties work cleanly, the architectural foundation of Entelechy is valid.

Everything else:

```text
GPU

AI

Node Editor

Graph Compiler

Packages

Cloud

Agents
```

can grow on top of that foundation.