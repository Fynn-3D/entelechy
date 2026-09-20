# Orla bridge for Godot

A Godot 4 project that is a client of Orla's creation protocol, in
GDScript: a socket that speaks the protocol, a Control that is a
session's window, and a demo scene around them; and, when the
`orla-godot` extension is built, the same project runs a session inside
Godot's own process through the C ABI, the picture an `Image` built on
the Host's words. Nothing in the Rust workspace depends on the project,
and it knows nothing about how Orla is built.

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
- `main.tscn`/`main.gd` is the demo above, built in code. A path to a
  `.osn` in the address field instead of `host:port` opens the scene in
  this process (see below), on the backend the button beside it names.
- `orla_local.gd` (`OrlaLocal`) is the in-process session, wrapped to
  answer like `OrlaClient` (`request(op, fields)`), plus `evaluate(input)`
  and `image()`; `OrlaView` drives either.

## In this process

`crates/godot` is a GDExtension over `orla-ffi` (godot-rust 0.4, built
against the 4.5 API, loaded by Godot 4.6) with one class, `OrlaSession`:
`open(scene, backend)`, `evaluate(input: Dictionary)` (time, pointer,
buttons, wheel, keys, window, scale), `output_image(name)` (an RGBAF
`Image`, one copy from the Host's words), `output(name)` and
`node(node, port)` (words as a `PackedByteArray` with the layout),
`request(line)`, `report()`, `error()`, `close()`.

```sh
cargo build --release -p orla-godot
godot --path bridges/godot      # type examples/fluid/scene.osn in the address field, Connect, Run
```

`orla.gdextension` points at the workspace's `target/release`, so
nothing is copied; `OrlaLocal.available()` loads it at runtime when the
editor has not (a headless `-s` run has no cache of extensions). What
the process saves is the frame's trip through a PPM and the socket;
what it does not save is the evaluation itself:

| scene, window | over the socket | in this process |
|---|---|---|
| text, 192 x 88 | 4.4 ms | 3.2 ms |
| voxel (JIT), 480 x 270 | 31.5 ms | 25.1 ms |
| voxel (JIT), 960 x 540 | 116.7 ms | 92.0 ms |
| voxel (JIT), 1920 x 1080 | 463 ms | 364 ms |

The voxel editor's frame is the JIT's work; the transport was a
quarter of it at 1080p and is gone. The GPU backend's frames still come
back to host memory before Godot sees them: a shared texture between
wgpu and Godot's renderer is a further step.

Both scripts declare a `class_name` for the editor and reach each other
by `preload`, so they also load headless, before the editor has
catalogued the classes.

`test.gd` checks all of it headless against three sessions it starts
itself (Godot spawns `orla serve` through a pipe and reads each port from
its stderr): it reads the text example's picture and republishes its
background, drives the voxel editor through an `OrlaView` with a window
and a pointer, places a voxel and resizes the window, pulls the mesh
example's torus as an `ArrayMesh` and sets its `level`, connects the
demo scene itself, and, when the extension is built, does the text and
voxel checks again in this process (`local:` lines; skipped otherwise).
Godot 4.6 (installed at `/usr/local/bin/godot` here), nothing else:

```sh
cargo build --release
godot --headless --path bridges/godot -s test.gd        # ends with RESULT: ok
```

Over the socket the picture crosses as a PPM every frame, a few
milliseconds at small sizes and a quarter of a 1080p frame; the
extension removes that, the protocol client stays for a session running
elsewhere.
