# Rigs

**Answers:** what rows a rig is, so any tool can write one and any scene
can read it.

A rig is five kinds of row. `entelechy import` writes them from a glTF, the
Blender bridge from an armature, and the
[`animation` package](../packages/animation.entpkg) reads them. They are Fields
like any other; the runtime has no notion of a joint, and posing is
Kernels.

## The rows

| File | Layout | One row is |
| --- | --- | --- |
| `<name>.vertices.field` | `f,f,f,f,f,f,f,f` | a vertex of the mesh **at rest**: position, normal, uv |
| `<name>.triangles.field` | `i,i,i` | three vertex indices |
| `<name>.weights.field` | `i,i,i,i,f,f,f,f` | a vertex's four joints, then how much of each |
| `<name>.skeleton.field` | `i,f×16` | a joint's parent, then its inverse bind matrix |
| `<name>.<clip>.clip.field` | `f×10` | a joint's local translation (3), rotation (4) and scale (3) |

`i` is an `Int` and `f` a `Float`, in the order the ABI lays a struct
out. A vertex may carry a tangent after its uv (`f×12`), the way a normal
map is read along: its `w` is the handedness, the bitangent
`cross(normal, tangent) * w`; `entelechy export` writes it as the mesh's
tangents, and `entelechy import` reads the position, normal and uv. The `animation` package declares them as `Point`, `Tri`, `Bound`,
`Joint` and `Pose`.

## What the numbers mean

- **Space.** Positions, normals and matrices are y-up, as in glTF. The
  mesh is in the skin's own space, the joints' space; a rig's mesh
  carries no object transform.
- **Order.** `weights.field` has one row per row of `vertices.field`, in
  the same order. A joint index is a row of `skeleton.field`.
- **Parents.** A joint's `parent` is a row of the same file, or `-1` at a
  root, in any order.
- **The bind.** `bind` is the *inverse* bind matrix, by columns, as the
  ABI lays a `Mat4` out.
- **Clips.** A clip holds `frames * joints` rows: every joint of a frame,
  frame after frame. A pose is local, in its parent's space; a root's is
  in the skin's. The rotation is a quaternion, `x y z w`. The rows carry
  no sample rate: `entelechy import` prints it, the bridge uses the scene's,
  and a scene passes it to `animation.sample@1` as `rate`.
- **Travel.** A clip whose root travels closes on itself: its last frame
  is its first with the root moved on by one stride, so it loops every
  `frames - 1` frames. `animation.stride@1` plays it in place and
  `animation.travel@1` gives the travel for a scene to carry; a clip that
  does not travel is played by `animation.sample@1`, looping every `frames`.
- **Weights.** At most four, summing to one. All four zero binds the
  vertex to nothing, and the package leaves it in place, so skinned and
  unskinned meshes mix.
- **Instances.** A scene posing many instances of one rig holds
  `instances * joints` poses and `instances * vertices` mesh rows, each
  instance a block in the same order, read that way by the package's
  Kernels when their node is sized for it. The rig's own rows stay one
  of each.

`entelechy import` takes a model's first skin and names in its notes what
these rows do not hold (other skins, morph targets, a second material's
texture).

## What holds it

[`crates/interface/entelechy/tests/rigs.rs`](../crates/interface/entelechy/tests/rigs.rs) is the
contract as a test: one table of layouts, checked against what `entelechy
import` writes of a rig `entelechy export` wrote, and by a scene posing the
fresh rows and evaluating. The `animation` package reads the same rows in
[`examples/animation/parts/blend`](../examples/animation/parts/blend), whose replays hold it. The Blender bridge's push
is held to the same table in
[`crates/interface/entelechy/tests/bridges.rs`](../crates/interface/entelechy/tests/bridges.rs),
which also poses a pushed rig and its own glTF and checks they land in
the same place.

A tool that writes these five files is done: nothing is registered.

## The way back out

`entelechy-export` writes the five rows as a GLB: the mesh and its triangles,
a skin whose joints carry the rest pose the inverse binds imply, and one
animation per clip, sampled at `--rate`. The mesh, weights and skeleton
round-trip bit for bit, the clips to a quaternion's rounding; Blender
opens it with its bones, vertex groups and actions. The rows are told
apart by layout, so no flag says which file is which.

The rows hold no names. A joint is `joint0`, `joint1` and on in the GLB
unless `--joints` names them, one name a skeleton row in its order; a
count that is not the skeleton's is refused. Names are the GLB's, and the
rows it came from are the same whatever they are called.

```sh
entelechy-export examples/animation/parts/blend/assets/arm.*.field --out arm.glb --rate 30
entelechy-export examples/animation/parts/blend/assets/arm.*.field --out arm.glb --rate 30 --joints shoulder,elbow
```
