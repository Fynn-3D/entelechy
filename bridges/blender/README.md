# Orla bridge for Blender

A Blender add-on that is a client of Orla's public contracts and nothing
more: the Host's file formats and the creation protocol. Nothing in the
Rust workspace depends on it.

Install `orla_bridge.py` through Edit > Preferences > Add-ons > Install;
the panel is in the 3D view's sidebar (N), tab "Orla". Preferences hold
the host and port of a running session:

```sh
cargo run --release -p orla-player -- examples/model/scene.osn --serve --listen 127.0.0.1:7788
```

**Files** push the active mesh into a scene's assets directory as the
rows the runtime uses, `<name>.vertices.field` (position, normal, uv per
triangle corner, in world space) and `<name>.triangles.field`, and the
material's base colour image as `<name>.albedo.png`. A scene reads them as
inputs (`input mesh: Model = "assets/Cube.vertices.field";`) and the Host
watches them, so with **Live** on, editing the mesh in Blender changes the
window. This path needs no connection.

**Session** talks to the running window: pull two nodes' rows out as a
mesh (rows of 8 or 9 floats and rows of three ints, as the mesh and
model examples produce), pull the picture into an Image named OrlaFrame,
set a graph parameter, publish the text editor's active text as a Kernel
and watch it land.

`test_bridge.py` runs headless against a serving session and checks all
of it:

```sh
cargo run -- serve examples/mesh/scene.osn --listen 127.0.0.1:7788 &
/Applications/Blender.app/Contents/MacOS/Blender -b --python bridges/blender/test_bridge.py -- --port 7788 --out /tmp/orla-bridge
```
