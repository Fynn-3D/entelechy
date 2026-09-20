# Orla bridge for Godot

A Godot 4 project that is a client of Orla's creation protocol and
nothing more, in GDScript: a socket that speaks the protocol, a Control
that is a session's window, and a demo scene around them. Nothing in the
Rust workspace depends on it, and it knows nothing about how Orla is
built.

```sh
cargo run --release -- serve examples/fluid/scene.osn --backend jit --listen 127.0.0.1:7788
godot --path bridges/godot          # or open the folder in the Godot editor and press play
```

**Connect** opens the socket and asks for `status` and the graph's
params; **Run** evaluates frame after frame: every `evaluate` carries the
pointer in output pixels, the buttons held, what the wheel scrolled, the
keys down (letters and digits by uppercase ASCII, arrows 37 to 40), the
view's size in device pixels as the window and the screen's scale factor;
every frame is then `read` as a PPM and drawn, letterboxed like the
Player. Click the picture to give it the keys; drag in the fluid, paint
in the voxel editor. The side panel has one control per graph param (a
SpinBox, a CheckBox, or a field taking `[x, y]`), set with `param`; a
param that sizes the graph resizes the running scene. **Reset state** is
the Player's F5. The status line shows the round trip and what ran.

- `orla_client.gd` (`OrlaClient`) is the client on its own: `open`,
  `request(op, fields)` (blocking; the session answers in order, so a
  frame's evaluate-then-read stays two calls), `close`; the decoders
  `decode_ppm` (an RGB8 `Image`), `decode_words` with `float_at`,
  `int_at`, `bool_at`; `placement`; and `mesh_of(vertex_node,
  triangle_node)`, which pins two nodes, evaluates them, and builds an
  `ArrayMesh` from rows of eight or nine floats and rows of three ints
  (the mesh and model examples' layout), winding turned for Godot's
  front face. A Kernel's mesh is then a `MeshInstance3D` away.
- `orla_view.gd` (`OrlaView`) is the window: a Control holding a client,
  `step()` for one frame (evaluate, read, show), `start`/`stop` for
  frame after frame in `_process`, the input from `_gui_input`, and the
  signals `framed(report, millis)` and `stopped(message)`.
- `main.tscn`/`main.gd` is the demo above, built in code.

Both scripts declare a `class_name` for the editor and reach each other
by `preload`, so they also load headless, before the editor has
catalogued the classes.

`test.gd` checks all of it headless against three sessions it starts
itself (Godot spawns `orla serve` through a pipe and reads each port from
its stderr): it reads the text example's picture and republishes its
background, drives the voxel editor through an `OrlaView` with a window
and a pointer, places a voxel and resizes the window, pulls the mesh
example's torus as an `ArrayMesh` and sets its `level`, and connects the
demo scene itself. Godot 4.6 (installed at `/usr/local/bin/godot` here),
nothing else:

```sh
cargo build --release
godot --headless --path bridges/godot -s test.gd        # ends with RESULT: ok
```

The picture crosses the socket as a PPM every frame; a 384 by 216 fluid
runs at the session's speed, a Retina-sized window is a few milliseconds
of transfer a frame. A GDExtension linking the crates in-process would
remove that copy; this bridge stays a protocol client on purpose.
