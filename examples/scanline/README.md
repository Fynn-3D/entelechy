# Scanline

The architecture plan's third-phase demonstration, recorded: an agent
creates a capability while the scene runs. `scene.osn` shows a picture
and nothing else. `agent.jsonl` is what an agent sent over the creation
protocol, one request a line, and this is what happened:

1. `status`, `kernels`, `help coord`: it looked at the scene, the
   Kernels there, and the contract of a builtin it meant to use.
2. `evaluate`, `read`: it took the picture as it was.
3. `prepare` a scanline Kernel written with `coord().y % 2`: refused at
   line 9, column 20, "cannot apply `%` to Float and Int"; nothing was
   published.
4. `help icoord`, `prepare` again with `icoord().y % 2`: accepted, still
   unpublished; `publish` with `expect: "absent"`: `scanline.effect@1`
   exists now, in the running session.
5. `document`, `compose`: it read the scene and wrote it back with a
   `scan` node between the picture and the output; the scene reloaded.
6. `evaluate`, `read`: every odd row is half as bright, the even rows
   untouched. `compose` again with `strength = 0.8`, `evaluate`, `read`:
   the odd rows at a fifth, and the evaluator fused the picture into the
   effect on its own.
7. `kernel scanline.effect@1`: its own Kernel back, source and all, the
   repaired line in it.

The application never restarted. The test in
`crates/cli/tests/protocol.rs` runs this record against a copy of the
example and holds every step to the above, so it stays true.

To watch it yourself, on a copy, since `compose` rewrites the scene:

```sh
cargo build --release
cp -r examples/scanline /tmp/scanline
./target/release/orla serve /tmp/scanline/scene.osn < examples/scanline/agent.jsonl
```

The first take of this record named the node `lines`, because `scan`,
the keyword of a scan Kernel, could not name a node; a node may take any
word now, so the natural name stands.
