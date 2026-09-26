# Scanline

An agent creates a capability while the scene runs. `scene.osn` shows a
picture; `agent.jsonl` is a creation-protocol session, one request a
line:

1. `status`, `kernels`, `help coord`: read the scene, its Kernels, and a
   builtin's contract.
2. `evaluate`, `read`: take the picture as it is.
3. `prepare` a scanline Kernel using `coord().y & 1`: refused with a
   type error (a coordinate is a Float); nothing is published.
4. `help icoord`, `prepare` with `icoord().y % 2`, then `publish` with
   `expect: "absent"`: `scanline.effect@1` exists in the running session.
5. `document`, `compose`: the scene gains a `scan` node between the
   picture and the output, and reloads.
6. `evaluate`, `read`: odd rows at half brightness. `compose` with
   `strength = 0.8`, then `evaluate`, `read` again.
7. `kernel scanline.effect@1`: the published Kernel's source.

`crates/interface/orla/tests/protocol.rs` runs this record against a
copy of the example and checks every step.

`compose` rewrites the scene, so run it on a copy:

```sh
cargo build --release
cp -r examples/scanline /tmp/scanline
./target/release/orla serve /tmp/scanline/scene.osn < examples/scanline/agent.jsonl
```
