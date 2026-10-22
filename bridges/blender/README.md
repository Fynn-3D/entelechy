# Entelechy bridge for Blender

A Blender add-on that is a client of Entelechy's public contracts and nothing
more: the Host's file formats and the creation protocol. Nothing in the
Rust workspace depends on it.

Install `entelechy_bridge.py` through Edit > Preferences > Add-ons > Install;
the panel is in the 3D view's sidebar (N), tab "Entelechy". Preferences hold
the host and port of a running session:

```sh
entelechy player examples/model/scene.esn --serve --listen 127.0.0.1:7788
```

**Axes**: Blender is z-up and Entelechy's scenes, like glTF, are y-up; the
bridge turns axes on the way out and back (Blender x, y, z is Entelechy x, z,
-y), so a model stands the same on both sides. "As is" passes
coordinates unchanged.

**Files** push the active mesh into a scene's assets directory as the
rows the runtime uses, `<name>.vertices.field` (position, normal, uv per
triangle corner, in world space) and `<name>.triangles.field`, and the
material's base color image as `<name>.albedo.png`. A scene reads them as
inputs (`input mesh: Model = "assets/Cube.vertices.field";`) and the Host
watches them, so with **Live** on, editing the mesh in Blender changes the
window. This path needs no connection.

**Rigs**: *Push rig* writes what a posed mesh needs — [the same five
rows](../../docs/RIGS.md) `entelechy import` writes out of a rigged glTF and
the [`animation` package](../../packages/animation.entpkg) reads: `<name>.vertices.field`
(the mesh at rest, in its armature's space, since the joints place it),
`<name>.weights.field` (the four heaviest vertex groups that name a bone,
normalized), `<name>.skeleton.field` (each bone's parent and inverse bind
matrix) and one `<name>.<action>.clip.field` per action, sampled at the
scene's own frame rate. With **Live** on, a rigged object pushes only the
pose it stands in — one clip of one frame, `<name>.pose.clip.field` —
so posing a bone in Blender moves the mesh in the window without
rewriting the mesh or the clips.

With **Live** on, posing in Pose Mode pushes too: Pose Mode makes the
armature the active object, so an armature stands for the meshes it
deforms.

`test_rig.py` builds a rig, pushes it, poses a bone with Live on, and
exports the same rig as glTF;
`crates/interface/entelechy/tests/bridges.rs` poses both and checks they put the arm in
the same place, which is what "the two ways in agree" means when their
vertices are in different orders:

```sh
/Applications/Blender.app/Contents/MacOS/Blender -b --python bridges/blender/test_rig.py -- --out /tmp/entelechy-rig
```

**Session** talks to the running window: pull two nodes' rows out as a
mesh (rows of 8 or 9 floats and rows of three ints, as the mesh and model
examples produce; the triangles may also be a `.field` file when the scene
reads them as an input), the nodes pinned first so fusion keeps them and
the corners welded back to shared vertices unless **Weld** is off; **Watch**
keeps an object current by reading the rows every interval and rebuilding
when they change, so a mesh a Kernel computes grows in Blender as its
code or parameters change; pull the picture into an Image named
EntelechyFrame; set a graph parameter (a param that sizes the graph, like the mesh
example's subdivision `level`, resizes the running scene, and a watch
follows); publish the text editor's active text as a Kernel and watch it
land.

`test_bridge.py` runs headless against a serving session and checks all
of it:

```sh
entelechy serve examples/mesh/scene.esn --listen 127.0.0.1:7788 &
/Applications/Blender.app/Contents/MacOS/Blender -b --python bridges/blender/test_bridge.py -- --port 7788 --out /tmp/entelechy-bridge
```
