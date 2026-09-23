# Rigs

**Answers:** what rows a rig is, so any tool can write one and any scene
can read it.

A rig is five kinds of row. `orla import` writes them out of a glTF, the
Blender bridge writes them out of an armature, and the
[`skin` package](../packages/skin.orlapkg) reads them. Nothing in the
runtime knows what a joint is: these are Fields like any other, and the
posing is Kernels.

## The rows

| File | Layout | One row is |
| --- | --- | --- |
| `<name>.vertices.field` | `f,f,f,f,f,f,f,f` | a vertex of the mesh **at rest**: position, normal, uv |
| `<name>.triangles.field` | `i,i,i` | three vertex indices |
| `<name>.weights.field` | `i,i,i,i,f,f,f,f` | a vertex's four joints, then how much of each |
| `<name>.skeleton.field` | `i,f×16` | a joint's parent, then its inverse bind matrix |
| `<name>.<clip>.clip.field` | `f×10` | a joint's local translation (3), rotation (4) and scale (3) |

`i` is an `Int` and `f` a `Float`, in the order the ABI lays a struct
out, which is the order the fields are written in. The `skin` package
declares them as `Point`, `Tri`, `Bound`, `Joint` and `Pose`.

## What the numbers mean

- **Space.** Positions, normals and matrices are y-up, as glTF is. The
  mesh is written in the skin's own space — the space the joints are in
  — because the joints place it; a rig's mesh carries no object
  transform of its own.
- **Order.** `weights.field` has one row per row of `vertices.field`, in
  the same order. A joint index is a row of `skeleton.field`.
- **Parents.** A joint's `parent` is a row of the same file, or `-1` at a
  root. Any order does: the palette walks up from each joint rather than
  down from the roots.
- **The bind.** `bind` is the *inverse* bind matrix — where the joint
  stood when the mesh was bound to it, undone — by columns, as the ABI
  lays a `Mat4` out.
- **Clips.** A clip holds `frames * joints` rows: every joint of a frame
  together, frame after frame. A pose is local, in its parent's space,
  and a root's is in the skin's. The rotation is a quaternion, `x y z w`.
  Nothing in the rows says what rate they were sampled at: `orla import`
  prints it, the bridge uses the scene's, and a scene passes it to
  `skin.sample@1` as `rate`.
- **Weights.** Four at most, and they should sum to one. All four zero
  means the vertex is bound to nothing, and the package leaves it where
  it is — which is how a model that mixes skinned and unskinned meshes
  comes in whole.

## What holds it

[`crates/cli/tests/rigs.rs`](../crates/cli/tests/rigs.rs) is the
contract as a test: one table of layouts, checked against what `orla
import` writes, against the rows this repository ships, and against the
package by binding fresh rows to it and evaluating. The Blender bridge's
push is held to the same table in
[`crates/cli/tests/bridges.rs`](../crates/cli/tests/bridges.rs), which
also poses a pushed rig and its own glTF and checks they land in the
same place.

A tool of your own writes these five files and is done; there is nothing
else to register, and no format to teach the runtime.

## What does not come in yet

`orla import` takes the first skin of a model and leaves the rest, and
says so. Morph targets, a second material's texture and anything else
these five rows do not hold are named in its notes rather than dropped
in silence. The way back out is a mesh, not a rig: `orla-export` writes
vertices, triangles and a texture as a GLB, and a rig exported from Orla
would need a skin and animations written there too.

## More than one of them

Nothing above changes for a crowd. A scene that poses many instances of
one rig holds `instances * joints` poses and `instances * vertices`
mesh rows, each instance a block of its own in the same order, and the
package's Kernels read them that way when their node is sized for it.
The rig's own rows — the mesh, the weights, the skeleton, the clips —
are still one of each.
